"""教学型精读引擎：先建立论文专属解释合同，再逐层偿还解释债务。

调用结构：
1. 长文先逐块建立可追溯证据索引；Planner 只输出 JSON 教学计划，不产出发布正文。
2. Writer 按 source_refs 回到原文逐节写作；每节由 Reviewer 做事实与连续性审核。
3. heavy × N：按流程、比较、推导、分类或证据模式生成各层；案例按论文需要使用。
4. heavy+视觉：**全部图表逐张讲解**——每张图独立绑定编号、图注和正文上下文；
   视觉不可用时只使用图注与正文证据，不猜图内读数。
5. heavy：逐项核对解释债务，并做只依赖精读正文的闭卷复述终审。

目标不是覆盖全文，而是让只懂 LLM 基础的读者能复述核心技术如何工作。
"""

from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import json
import logging
import math
import mimetypes
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..config import PROJECT_ROOT
from ..kimi import KimiClient, KimiError, KimiParameterError
from .figures import FigureAsset, figure_inventory_text, plain_figure_caption
from .formulas import FormulaAsset, formula_inventory_text, normalize_formula_label
from .reading_evidence import (
    EvidenceBundle, EvidenceError, EvidenceIndexer, split_markdown_sections,
)
from .reading_workflow import PIPELINE_VERSION, ReadingWorkspace

logger = logging.getLogger("feishu_assistant.reading")

PROMPTS_DIR = PROJECT_ROOT / "prompts"
STYLE_PATH = PROMPTS_DIR / "paper_reading.md"        # 风格圣经
SKELETON_PATH = PROMPTS_DIR / "reading_skeleton.md"

HARD_INPUT_CAP = 200_000  # 仅保留给显式调用者；正式精读禁止使用字符硬截尾
FIGURE_MAX_WORKERS = 3    # 独立逐图、受控并行；不把多张图塞进同一模型上下文

# 输出预算 = 正文 + reasoning 思考。思考型模型（k3 系）在难任务上 reasoning 本身要烧
# 8k~18k token，预算按「正文上限」设会把额度全烧在思考上、正文一个字都出不来。
# 注意：这些档位只对 Anthropic 协议（max_tokens 必填）下发；OpenAI 兼容网关对显式
# max_tokens 会按 ~16384 封顶（实测，kimi.py 已剥离），不传时网关默认预算更大。
PLANNER_OUTPUT_TOKENS = 32_768
WRITER_OUTPUT_TOKENS = 32_768
LOCAL_REVIEW_OUTPUT_TOKENS = 16_384
FIGURE_OUTPUT_TOKENS = 16_384
FINAL_REVIEW_OUTPUT_TOKENS = 24_576
# 局审一次常抛出多条混合意见，一轮改写修不干净；允许两轮改写（验收标准不变），仍不过才判死
LOCAL_REVIEW_MAX_REWRITES = 2
# 风险分层：叙事/边界类章节只做结构检查（事实核查交终审），其余章节与 core 层逐节模型审
LIGHT_REVIEW_SECTION_PURPOSES = frozenset({"orientation", "boundary"})


def _degradation_banner(issues: str) -> str:
    """节级批注横幅：全文保留，顶部列明未通过核验的存疑点（透明状态）。
    处置粒度对齐错误粒度：审核失败是句级问题，不再整节陪葬。"""
    items = [item.strip() for item in issues.split("；") if item.strip()][:5]
    lines = ["", "> ⚠️ **本节以下内容未全部通过原文核验**（全文保留并标注，请对照原论文核实）："]
    lines += [f"> {i}. {item}" for i, item in enumerate(items, 1)]
    return "\n".join(lines) + "\n"


def _annotate_degraded(content: str, issues: str) -> str:
    """在标题行后插入批注横幅；原文一字不删。"""
    lines = content.split("\n")
    for idx, line in enumerate(lines):
        if line.startswith("#"):
            lines.insert(idx + 1, _degradation_banner(issues))
            break
    return "\n".join(lines)


def _preparations_text(section: dict[str, Any]) -> str:
    """本节建立义务（前置概念，唯一建立者）的展示文本；空为（无）。"""
    return "；".join(
        f"为「{p.get('for_layer')}」建立概念「{p.get('concept')}」"
        for p in (section.get("preparations") or []) if isinstance(p, dict)
    ) or "（无）"


def _skeleton_review_check_gaps(
    checks: Any,
    skeleton_data: dict[str, Any],
    chunk_ids: set[str] | None,
    source: str,
) -> list[str]:
    """骨架语义审核的举证门禁（程序级，fail-closed）：

    覆盖：多层必须逐层各有一条断链检查（汇总检查不算）；单层必须含核心主张检查；
    所有骨架必须覆盖 preparations 分配与 purpose/mode 举证。
    追溯：长文 evidence 必须引用真实 chunk id；短文必须含可定位锚点
    （原文摘录/章节标题/图表标签），引用了不存在的章/节/图/chunk 即拒绝。
    """
    def norm(v: Any) -> str:
        return re.sub(r"[^\w一-鿿]+", "", str(v or "")).casefold()

    if not isinstance(checks, list) or not checks:
        return ["骨架审核缺少 checks 举证明细"]
    valid = [c for c in checks if isinstance(c, dict)]
    if len(valid) != len(checks):
        return ["checks 含非对象项"]
    gaps: list[str] = []
    layers = [l for l in (skeleton_data.get("layers") or []) if isinstance(l, dict)]
    layer_names = [str(l.get("name", "")).strip() for l in layers if str(l.get("name", "")).strip()]

    def covering(check: dict, token_norm: str) -> bool:
        item = norm(check.get("item"))
        if token_norm not in item:
            return False
        hits = sum(1 for n in layer_names if norm(n) and norm(n) in item)
        return hits == 1  # 汇总检查冒充逐层：命中多层 → 对每层都不算数

    if len(layer_names) >= 2:
        for name in layer_names:
            if not any(covering(c, norm(name)) and
                       ("断链" in str(c.get("item")) or "合并" in str(c.get("item")))
                       for c in valid):
                gaps.append(f"checks 未逐层覆盖层「{name}」的断链声明（汇总检查不算）")
    elif len(layer_names) == 1:
        if not any("核心主张" in str(c.get("item")) for c in valid):
            gaps.append("单层骨架 checks 缺少「单层核心主张」检查")
    if not any(t in str(c.get("item")) for c in valid
               for t in ("preparations", "前置概念", "建立义务")):
        gaps.append("checks 未覆盖 preparations 分配")
    if not any(t in str(c.get("item")) for c in valid
               for t in ("purpose", "mode", "功能举证", "模式举证")):
        gaps.append("checks 未覆盖 purpose/mode 举证")

    for c in valid:
        evidence_text = str(c.get("evidence") or "").strip()
        label = str(c.get("item") or "").strip()[:30]
        if not evidence_text:
            gaps.append(f"check「{label}」缺 evidence")
            continue
        if chunk_ids is not None:
            found = re.findall(r"chunk-\d{3,4}", evidence_text)
            if not found:
                gaps.append(f"check「{label}」evidence 未引用真实 chunk id")
            elif any(f not in chunk_ids for f in found):
                gaps.append(f"check「{label}」evidence 引用不存在的 chunk")
            continue
        if re.search(r"原文.{0,6}(章|中|里)?[。，,]?$|见论文|见原文|论文提到", evidence_text):
            gaps.append(f"check「{label}」evidence 无法定位（空泛指引）")
            continue
        bad_ref = False
        for num in re.findall(r"第\s*([\d.]+)\s*[章节]", evidence_text):
            if not re.search(rf"^#{{1,3}}\s*(?:第\s*)?{re.escape(num)}[\s.、．]", source, re.M):
                gaps.append(f"check「{label}」evidence 引用不存在的第 {num} 章/节")
                bad_ref = True
        for kind, num in re.findall(r"(Figure|Table|Equation)\s*(\d+)", evidence_text):
            if f"{kind} {num}" not in source:
                gaps.append(f"check「{label}」evidence 引用不存在的 {kind} {num}")
                bad_ref = True
        if bad_ref:
            continue
        quotes = [q.strip() for q in
                  re.findall(r"[「\"“‘]([^」\"”’]{12,120})[」\"”’]", evidence_text)]
        headings = [h.strip("# \n") for h in re.findall(r"^#{1,3} .+$", source, re.M)]
        located = (
            any(q in source for q in quotes)
            or any(h and len(h) >= 4 and h in evidence_text for h in headings)
            or bool(re.findall(r"第\s*[\d.]+\s*[章节]", evidence_text))
            or any(f"{k} {n}" in source for k, n in
                   re.findall(r"(Figure|Table|Equation)\s*(\d+)", evidence_text))
        )
        if not located:
            gaps.append(f"check「{label}」evidence 无可定位锚点（摘录/标题/标签均缺）")
    return gaps


def _unit_binding_deps(unit: Any) -> dict[str, int]:
    """单元依赖边（供 DependencyManifest 精准失效传播）：
    source span 引用 + concept/assertion 绑定，静态 rev 1。"""
    deps = {f"node:{nid}": 1 for nid in (*unit.concept_ids, *unit.assertion_ids)}
    deps.update({f"span:{ref}": 1 for ref in (unit.data.get("source_refs") or [])
                 if str(ref).strip()})
    return deps


def _plan_review_check_gaps(
    checks: Any,
    plan: "ReadingPlan",
    source: str,
) -> list[str]:
    """Plan Reviewer 的语义准入举证门禁（程序级）：复用骨架审核的反汇总覆盖逻辑
    （多层逐层断链/单层核心主张/preparations/purpose·mode），另要求每条 check 带
    可追溯 evidence 与指向真实单元的 target_ids。不合格不判死：由调用方 correction
    重试一次，仍不合格标注「语义准入审核不完整」继续。"""
    layers = [plan.units[uid].data for uid in (plan.order.get("layers") or [])
              if uid in plan.units]
    skeleton_data = {"layers": layers, "preparations": plan.preparations}
    chunk_ids = set(re.findall(r"chunk-\d{3,4}", source)) or None
    gaps = _skeleton_review_check_gaps(checks, skeleton_data, chunk_ids, source)
    valid_targets: set[str] = set(plan.units)
    for uid, unit in plan.units.items():
        for key in ("name", "title"):
            text = str(unit.data.get(key) or "").strip()
            if text:
                valid_targets.add(text)
    for check in [c for c in checks if isinstance(c, dict)] if isinstance(checks, list) else []:
        label = str(check.get("item") or "").strip()[:30]
        targets = [str(t).strip() for t in (check.get("target_ids") or [])
                   if str(t).strip()]
        if not targets:
            gaps.append(f"check「{label}」缺 target_ids")
        elif any(t not in valid_targets for t in targets):
            gaps.append(f"check「{label}」target_ids 指向不存在单元")
    return gaps


def _full_map_text(skeleton: "Skeleton") -> str:
    """全局归属图：审核器专用的节/层/建立义务全表（写手不可见）。
    用于区分"成体系复述"与"合法回指"。"""
    lines: list[str] = []
    for idx, section in enumerate(skeleton.sections, 1):
        preps = "；".join(
            f"为「{p.get('for_layer')}」建立「{p.get('concept')}」"
            for p in (section.get("preparations") or []) if isinstance(p, dict)
        ) or "无"
        lines.append(
            f"节{idx}《{section.get('title')}》（功能：{section.get('purpose')}）建立义务：{preps}"
        )
    for i, layer in enumerate(skeleton.layers, 1):
        lines.append(
            f"层{i}《{layer.name}》（{layer.depth}）问题：{layer.question}；"
            f"债务：{'、'.join(layer.must_explain)}"
        )
    return "\n".join(lines)


def _patched_context(content: str, edits: Any, *, max_chars: int = 2_000) -> str:
    """补丁复核的上下文窗口：含替换片段的段落及其前后各一段（不整节传输）。"""
    if not isinstance(edits, list) or not edits:
        return content[:max_chars]
    needles = [
        str(e.get("replace") or "")[:40]
        for e in edits if isinstance(e, dict) and str(e.get("replace") or "").strip()
    ]
    paras = content.split("\n\n")
    if not needles:
        return "\n\n".join(paras[:2])[:max_chars]
    keep: set[int] = set()
    for idx, para in enumerate(paras):
        if any(needle in para for needle in needles):
            keep.update({idx - 1, idx, idx + 1})
    selected = [para for idx, para in enumerate(paras) if idx in keep]
    return "\n\n".join(selected)[:max_chars]


def _apply_edits(content: str, edits: Any) -> str | None:
    """机械应用 find/replace 补丁：find 必须在正文中恰好出现一次，否则放弃补丁
    （返回 None，调用方回退全量重写）。拼接过程零模型参与，禁止歧义替换。"""
    if not isinstance(edits, list):
        return None
    patched = content
    for edit in edits:
        if not isinstance(edit, dict):
            return None
        find = str(edit.get("find") or "")
        replace = str(edit.get("replace") or "")
        if not find:
            continue
        if patched.count(find) != 1:
            return None
        patched = patched.replace(find, replace, 1)
    return patched


_JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
_DROP_REFS_RE = re.compile(
    r"^#{1,3}\s*(references|bibliography|参考文献|引用文献)\s*$", re.IGNORECASE | re.MULTILINE
)
_H1_RE = re.compile(r"^#\s+\S")
_INLINE_NOTE_RE = re.compile(r"【INLINE】\s*(.+?)\s*【NOTE】\s*(.+)", re.DOTALL)
_LAYER_TEMPLATE_PHRASES = (
    "这一层只回答一个问题", "先看它为什么需要存在", "跟着一个输入走到输出",
    "现在再给它技术名字", "必要时再看公式或代码", "读完这层，你应该能说清",
)

DIRECTIONS = ("Agent Harness", "大模型开发", "Agent 系统", "其他")


class ReadingError(Exception):
    """论文精读生成失败，信息为中文。"""


# ---------------------------------------------------------------------------
# 全文保留：只删参考文献列表节
# ---------------------------------------------------------------------------

def strip_references(md: str, hard_cap: int | None = None) -> str:
    """只删参考文献列表节（REFERENCES/Bibliography），正文+附录全文保留。"""
    m = _DROP_REFS_RE.search(md)
    if m:
        rest = md[m.end():]
        next_heading = re.search(r"^#{1,3}\s+", rest, re.MULTILINE)
        md = md[: m.start()] + ("\n" + rest[next_heading.start():] if next_heading else "")
        logger.info("已删除参考文献列表节")
    if hard_cap and len(md) > hard_cap:
        logger.warning("论文超长（%d 字符），硬截断至 %d（极罕见）", len(md), hard_cap)
        md = md[:hard_cap] + "\n\n…（超出硬上限，尾部截断）"
    return md


def strip_h1(md: str) -> str:
    """剥离正文里的 H1 行（Stage 产物的内部标题泄漏），代码围栏内的 # 注释不动。"""
    out: list[str] = []
    in_fence = False
    for line in md.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if not in_fence and _H1_RE.match(line):
            continue
        out.append(line)
    return "\n".join(out)


def _locate_failed_units(
    review: dict[str, Any],
    skeleton: "Skeleton",
    contract_gaps: list[str],
    figure_labels: set[str],
    plan: Any | None = None,
) -> tuple[dict[str, str], bool]:
    """终审失败定位：把失败原因映射到具体 层/节/图 的检查点键，返回
    （{检查点键: 返工意见}, 是否涉及 core 层核心链路）。无法定位时返回空映射
    （调用方按 core 判死——修不了的问题不许交付）。"""
    units: dict[str, list[str]] = {}
    core_involved = False

    def mark(key: str, note: str, *, core: bool = False) -> None:
        nonlocal core_involved
        units.setdefault(key, []).append(note)
        core_involved = core_involved or core

    layer_by_norm = {_normalize_heading(layer.name): (idx, layer)
                     for idx, layer in enumerate(skeleton.layers, 1)}
    section_by_norm = {}
    for idx, section in enumerate(skeleton.sections, 1):
        title = str(section.get("title") or "").strip()
        if title:
            section_by_norm[_normalize_heading(title)] = idx

    def _unit_key(kind: str, idx: int) -> str:
        """稳定 ID 键：有 plan 用 unit_id，无 plan 退回序号键（兼容路径）。"""
        if plan is not None:
            bucket = plan.order.get("layers" if kind == "layer" else "sections") or []
            if 0 < idx <= len(bucket):
                return f"body-{bucket[idx - 1]}"
        return f"{kind}-{idx:03d}"

    def mark_layer_by_name(name: str, note: str) -> bool:
        norm = _normalize_heading(name)
        if norm in layer_by_norm:
            idx, layer = layer_by_norm[norm]
            mark(_unit_key("layer", idx), note, core=layer.depth == "core")
            return True
        return False

    # ① layer_coverage / teach_back：债务或检查未过 → 该层返工
    for entry in review.get("layer_coverage") or []:
        if not isinstance(entry, dict):
            continue
        problems = [
            str(d.get("item") or "") for d in entry.get("debts") or []
            if isinstance(d, dict) and _review_bool(d.get("covered")) is not True
        ] + [
            str(t.get("item") or "") for t in entry.get("success_tests") or []
            if isinstance(t, dict) and _review_bool(t.get("passed")) is not True
        ]
        if problems:
            mark_layer_by_name(str(entry.get("layer") or ""),
                               "解释债务/闭卷检查未完成：" + "；".join(problems))
    for entry in review.get("teach_back") or []:
        if isinstance(entry, dict) and _review_bool(entry.get("answerable")) is not True:
            mark_layer_by_name(str(entry.get("layer") or ""),
                               "闭卷复述失败：" + str(entry.get("missing") or "无法复述"))

    # ② issues[].location：按层名/节标题/图标签模糊归位
    for item in review.get("issues") or []:
        if not isinstance(item, dict):
            continue
        location = _normalize_heading(item.get("location"))
        problem = str(item.get("problem") or "").strip()
        if not problem:
            continue
        severity = str(item.get("severity") or "高").strip()
        placed = False
        for norm, (idx, layer) in layer_by_norm.items():
            if norm and norm in location:
                mark(_unit_key("layer", idx), f"[{severity}] {problem}",
                     core=layer.depth == "core")
                placed = True
                break
        if placed:
            continue
        for norm, idx in section_by_norm.items():
            if norm and norm in location:
                mark(_unit_key("section", idx), f"[{severity}] {problem}")
                placed = True
                break
        if not placed:
            for label in figure_labels:
                if _normalize_heading(label) in location:
                    mark(f"figure-{label}", f"[{severity}] {problem}")
                    break

    # ③ 合同核对缺口：消息以层名开头 → 归该层（core 缺口即 core 判死）
    layer_names = list(layer_by_norm.values())
    for gap in contract_gaps:
        placed = False
        for idx, layer in layer_names:
            if gap.startswith(layer.name):
                mark(_unit_key("layer", idx), f"合同核对：{gap}", core=layer.depth == "core")
                placed = True
                break
        if not placed:
            core_involved = True  # 无法归位的合同缺口按核心问题处理

    return (
        {key: "\n".join(dict.fromkeys(notes)) for key, notes in units.items()},
        core_involved,
    )


def _degraded_skeleton(meta_hint: dict[str, str], arxiv_id: str) -> tuple["Skeleton", dict[str, Any]]:
    """确定性降级 Plan：规划完全不可用时的保底结构。
    orientation/evidence/boundary 三节 + 单层综合精讲——不依赖任何模型输出。"""
    skeleton = Skeleton(
        "",
        [KeyLayer(
            "核心机制综合精讲",
            "规划服务不可用时的合并精讲",
            question="论文的核心技术怎样把输入变成输出？",
            teaching_mode="process", depth="core",
            must_explain=["论文的核心机制是什么"],
            success_test=["能复述论文的核心方法"],
        )],
        "",
        sections=[
            {"purpose": "orientation", "title": "问题与关键转向",
             "purpose_reason": "建立任务背景", "preparations": []},
            {"purpose": "evidence", "title": "证据与结果",
             "purpose_reason": "说明证据支持的判断", "preparations": []},
            {"purpose": "boundary", "title": "边界与代价",
             "purpose_reason": "防止外推过度", "preparations": []},
        ],
        deep_dive_title="核心内容精讲",
    )
    metadata = dict(meta_hint or {})
    metadata.setdefault("title_zh", meta_hint.get("title", arxiv_id or "未命名论文"))
    return skeleton, metadata


def _final_banner(detail: str, *, title: str = "本篇精读未全部通过发布前终审") -> str:
    """终审降级横幅：终审不过也强制交付（用户裁定：token 与时间必须有产出），
    core 问题用强标题；对话框结果卡同步透出失败明细。"""
    items = [item.strip() for item in detail.split("；") if item.strip()][:5]
    lines = ["", f"> ⚠️ **{title}**（全文保留并标注，请对照原论文核实）："]
    lines += [f"> {i}. {item}" for i, item in enumerate(items, 1)]
    return "\n".join(lines) + "\n"


def _fit_heading(content: str, level: int, title: str) -> str:
    """标题机械归一：首行若是任意标题则替换为合同标题，否则前置合同标题。

    模型常把标题写成自己的措辞（内容其实没问题）——过去重写两次仍不对就判死，
    纯格式问题不该耗模型更不该判死。幂等：已正确则原样返回。
    """
    if _starts_with_heading(content, level, title):
        return content
    prefix = "#" * level
    lines = content.split("\n")
    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            lines[idx] = f"{prefix} {title}"
        else:
            lines[idx:idx] = [f"{prefix} {title}", ""]
        break
    else:
        lines.insert(0, f"{prefix} {title}")
    return "\n".join(lines)


def _clean_layer_surface(content: str, layer_name: str) -> str:
    """技术层表面机械清理：标题归一、层内多余 ### 降级为加粗、删除教案口令行。

    teaching_layer_gaps 的三类可修形态（标题/层级/模板泄漏）都是格式问题；
    清理后仍不过（如正文为空）才判死。
    """
    content = _fit_heading(content, 3, layer_name)
    lines = content.split("\n")
    cleaned: list[str] = []
    first_heading_seen = False
    for line in lines:
        if re.match(r"^###\s*", line):
            if not first_heading_seen:
                first_heading_seen = True
                cleaned.append(line)
                continue
            cleaned.append(f"**{line.lstrip('#').strip()}**")  # 层内第二个 ### 降级
            continue
        if any(phrase in line for phrase in _LAYER_TEMPLATE_PHRASES):
            continue  # 教案口令行：整行删除（模板产物，非正文）
        cleaned.append(line)
    return "\n".join(cleaned)


def _strip_inline_latex(text: str) -> str:
    """行内 LaTeX 机械转纯文本：$...$ 去定界符，常见命令转 Unicode，
    \\command{内容} 保留内容去命令名。模型偶把简单公式写成 LaTeX，
    成品禁止残留未渲染标记——格式问题机械修，不耗模型重写。"""
    commands = {
        "\\pi": "π", "\\alpha": "α", "\\beta": "β", "\\gamma": "γ", "\\theta": "θ",
        "\\lambda": "λ", "\\mu": "μ", "\\sigma": "σ", "\\tau": "τ", "\\epsilon": "ε",
        "\\times": "×", "\\cdot": "·", "\\leq": "≤", "\\geq": "≥", "\\neq": "≠",
        "\\pm": "±", "\\approx": "≈", "\\infty": "∞", "\\log": "log", "\\exp": "exp",
        "\\max": "max", "\\min": "min", "\\arg": "arg", "\\in": "∈", "\\to": "→", "\\%": "%",
    }

    def _plain(match: re.Match) -> str:
        inner = match.group(1)
        for cmd, uni in commands.items():
            inner = inner.replace(cmd, uni)
        inner = re.sub(r"\\[a-zA-Z]+", "", inner)  # 未知命令名去反斜杠
        return inner.replace("{", "").replace("}", "")

    text = re.sub(r"\$([^$\n]+)\$", _plain, text)
    text = re.sub(r"\\[a-zA-Z]{2,}\{([^{}]*)\}", lambda m: m.group(1), text)
    return text


def _hoist_media_markers(text: str) -> str:
    """把嵌在段落中的 [[FIG:...]]/[[MATH:...]] 标记提升为独立成行（确定性修复）。

    docx 渲染只认独立成行的媒体标记；模型偶把标记写进句子，过去要靠一次
    模型重写碰运气，重写仍嵌行则整轮判死。格式问题机械修，不耗模型调用。
    标记内位置后缀（模型抄图清单格式：[[FIG:Figure 1（第 3 页…）]]）一并归一。
    """
    text = re.sub(r"\[\[(FIG|MATH):([^（\]]+)（[^）]*）\]\]", r"[[\1:\2]]", text)
    pattern = re.compile(r"\[\[(?:FIG|MATH):[^\]\n]+\]\]")
    out_lines: list[str] = []
    for line in text.split("\n"):
        markers = pattern.findall(line)
        if not markers or pattern.fullmatch(line.strip()):
            out_lines.append(line)
            continue
        segments = pattern.split(line)
        for i, segment in enumerate(segments):
            if segment.strip():
                out_lines.append(segment.strip())
            if i < len(markers):
                out_lines.append(markers[i])
    return "\n".join(out_lines)


def _media_marker_gaps(text: str) -> list[str]:
    """Docx 只识别独立成行的媒体标记；生成阶段就阻止字面量泄漏。"""
    gaps: list[str] = []
    for match in re.finditer(r"\[\[(?:FIG|MATH):[^\]\n]+\]\]", text):
        start = text.rfind("\n", 0, match.start()) + 1
        end = text.find("\n", match.end())
        line = text[start:(len(text) if end < 0 else end)].strip()
        if line != match.group(0):
            gaps.append("图表或公式标记必须独立成行")
            break
    return gaps


def teaching_layer_gaps(text: str, layer: Any | None = None) -> list[str]:
    """只做确定性表面检查；解释债务由终审语义核验。

    层正文结构合同与组装对齐：组装只接受两种形态——以 `###` 层标题开头
    （剥掉换成规范层名），或全文无标题（直接套用规范层名）。先写过渡句
    再给 `###` 会在成品里嵌套出第二个层标题，截断整篇文档的层区间检测。
    """
    headings = re.findall(r"^#{1,3}\s*.+$", text, re.MULTILINE)
    first_is_layer_heading = bool(headings and re.match(r"^###\s*", headings[0]))
    illegal_headings = headings[1:] if first_is_layer_heading else headings
    checks = {
        "正文为空": bool(text.strip()),
        "层内标题层级": not illegal_headings,
        "以层标题开头": (
            not first_is_layer_heading
            or bool(re.match(r"^###\s*[^\n]+\n", text.strip()))
        ),
        "无教案模板泄漏": not any(phrase in text for phrase in _LAYER_TEMPLATE_PHRASES),
    }
    return [name for name, ok in checks.items() if not ok] + _media_marker_gaps(text)


# ---------------------------------------------------------------------------
# 结果结构与计时
# ---------------------------------------------------------------------------

@dataclass
class KeyLayer:
    name: str
    why: str
    question: str = ""
    prerequisite: str = ""
    input: str = ""
    output: str = ""
    source_hints: list[str] = field(default_factory=list)
    formula_hints: list[str] = field(default_factory=list)
    teaching_mode: str = "process"
    depth: str = "core"
    must_explain: list[str] = field(default_factory=list)
    success_test: list[str] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    mode_reason: str = ""
    break_if_merged: dict[str, str] = field(default_factory=dict)


@dataclass
class Skeleton:
    markdown: str
    layers: list[KeyLayer]
    top3: str
    running_example: str = ""
    paper_type: str = "method"
    needs_running_example: bool = False
    sections: list[dict[str, Any]] = field(default_factory=list)
    deep_dive_title: str = "核心内容精讲"


@dataclass
class StageTimer:
    """阶段计时埋点：mark 打点时间，durations 给出相邻阶段耗时。"""

    t0: float = field(default_factory=time.time)
    marks: list[tuple[str, float]] = field(default_factory=list)

    def mark(self, name: str) -> None:
        self.marks.append((name, time.time()))

    def elapsed(self, name: str) -> float:
        """某 mark 距上一个 mark（或 t0）的秒数。"""
        prev = self.t0
        for n, ts in self.marks:
            if n == name:
                return ts - prev
            prev = ts
        return 0.0

    def total(self) -> float:
        return time.time() - self.t0

    def duration(self, start: str, end: str) -> float:
        """两个命名打点之间的耗时；找不到任一端点时返回 0。"""
        timestamps = {name: ts for name, ts in self.marks}
        if start not in timestamps or end not in timestamps:
            return 0.0
        return max(0.0, timestamps[end] - timestamps[start])


@dataclass
class ReadingResult:
    body_md: str
    metadata: dict[str, Any]
    layer_names: list[str]
    figure_notes: dict[str, str]
    figure_inlines: dict[str, str]
    vision_used: bool
    formula_images: dict[str, Path] = field(default_factory=dict)
    model_calls: int = 0
    timer: StageTimer | None = None


def extract_json_block(text: str) -> tuple[str, dict[str, Any]]:
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        stripped = text.strip()
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            # JSON-only 协议下仍容忍模型顺手添加前后缀；只提取首个可解码对象，
            # 前后自然语言永不进入发布正文，也不参与章节标题验收。
            decoder = json.JSONDecoder()
            for match in re.finditer(r"\{", text):
                try:
                    value, _ = decoder.raw_decode(text[match.start():])
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    return text[:match.start()].strip(), value
            return stripped, {}
        return ("", data) if isinstance(data, dict) else (stripped, {})
    body = text[: m.start()].strip()
    if text[m.end():].strip():
        logger.warning("Planner JSON 围栏后存在赘言，已明确丢弃")
    span = m.group(1)
    try:
        data = json.loads(span)
        if not isinstance(data, dict):
            raise ValueError("not a dict")
    except (json.JSONDecodeError, ValueError):
        # 容错：模型尾部赘言时截到 span 内最后一个 } 再试
        last = span.rfind("}")
        if last <= 0:
            return body, {}
        try:
            data = json.loads(span[: last + 1])
            if not isinstance(data, dict):
                raise ValueError("not a dict")
        except (json.JSONDecodeError, ValueError):
            return body, {}
    return body, data


_METADATA_KEYS = ("title_zh", "title_en", "authors", "affiliation", "venue",
                  "year", "keywords", "tldr", "contributions")

_CN_NUM = "零一二三四五六七八九"


def _cn(n: int) -> str:
    """1→一 … 9→九，其余原样返回。"""
    return _CN_NUM[n] if 0 < n <= 9 else str(n)


_HEADING_ENUM_RE = re.compile(
    r"^\s*(?:第?[一二三四五六七八九十百]+[、.．]|\d{1,2}\s*[.、．)]|"
    r"（[一二三四五六七八九十]+）|\([一二三四五六七八九十]+\))\s*")


def _normalize_heading(text: object) -> str:
    value = _HEADING_ENUM_RE.sub("", str(text or "").strip())
    return re.sub(r"[^\w一-鿿]+", "", value).casefold()


def _heading_present(text: str, level: int, expected: str) -> bool:
    target = _normalize_heading(expected)
    return any(
        _normalize_heading(match.group(1)) == target
        for match in re.finditer(
            rf"^#{{{level}}}(?!#)\s+(.+?)\s*$", text, re.MULTILINE,
        )
    )


def _starts_with_heading(text: str, level: int, expected: str) -> bool:
    match = re.match(
        rf"^#{{{level}}}(?!#)\s+(.+?)\s*$", text.strip(), re.MULTILINE,
    )
    return bool(match and _normalize_heading(match.group(1)) == _normalize_heading(expected))


def _section_spans(body: str) -> list[tuple[int, int, int, str]]:
    """围栏感知地切出各章节区间：(起, 止, 标题级别, 规范化标题)。

    复用 split_markdown_sections（忽略 ``` / ~~~ 围栏中的伪标题），
    偏移按切片在原文中的真实位置锚定，避免纯子串定位被同名前缀劫持。
    """
    spans: list[tuple[int, int, int, str]] = []
    cursor = 0
    for title, text in split_markdown_sections(body):
        start = body.find(text, cursor)
        if start < 0:  # 切片必为原文连续子串；防御性跳过
            continue
        heading = re.match(r"^(#{1,3})\s", text)
        level = len(heading.group(1)) if heading else 0
        spans.append((start, start + len(text), level, _normalize_heading(title)))
        cursor = start + len(text)
    return spans


def _review_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "pass", "passed", "是", "通过", "1"}:
            return True
        if normalized in {"false", "no", "fail", "failed", "否", "未通过", "0"}:
            return False
    return None


def parse_skeleton_full(output: str) -> tuple[Skeleton, dict[str, Any]]:
    """骨架输出 → (Skeleton, metadata)；调用方应先通过 skeleton_output_gaps 验收。"""
    _, data = extract_json_block(output)
    layers = [
        KeyLayer(
            name=str(l.get("name", "")).strip(),
            why=str(l.get("why", "")).strip(),
            question=str(l.get("question", "")).strip(),
            prerequisite=str(l.get("prerequisite", "")).strip(),
            input=str(l.get("input", "")).strip(),
            output=str(l.get("output", "")).strip(),
            source_hints=[str(v).strip() for v in (l.get("source_hints") or []) if str(v).strip()][:5],
            source_refs=[str(v).strip() for v in (l.get("source_refs") or []) if str(v).strip()][:12],
            formula_hints=[str(v).strip() for v in (l.get("formula_hints") or []) if str(v).strip()][:3],
            teaching_mode=str(l.get("teaching_mode") or "process").strip(),
            depth=str(l.get("depth") or "core").strip(),
            mode_reason=str(l.get("mode_reason") or "").strip(),
            break_if_merged=(dict(l.get("break_if_merged"))
                             if isinstance(l.get("break_if_merged"), dict) else {}),
            must_explain=[
                str(v).strip() for v in (l.get("must_explain") or []) if str(v).strip()
            ],
            success_test=[
                str(v).strip() for v in (l.get("success_test") or []) if str(v).strip()
            ],
        )
        for l in (data.get("layers") or [])
        if isinstance(l, dict) and l.get("name")
    ]
    if not layers:
        logger.warning("骨架未给出有效技术层清单（仅解析工具使用兜底层）")
        layers = [KeyLayer(
            "核心机制综合精讲",
            "骨架未产出分层清单，合并精讲全文核心机制",
            question="论文的核心技术怎样把输入变成输出？",
        )]
    metadata = {k: data[k] for k in _METADATA_KEYS if k in data}
    sections = [
        {"purpose": str(item.get("purpose") or "").strip(),
         "title": str(item.get("title") or "").strip(),
         "purpose_reason": str(item.get("purpose_reason") or "").strip(),
         "preparations": [
             {"for_layer": str(p.get("for_layer") or "").strip(),
              "concept": str(p.get("concept") or "").strip()}
             for p in (item.get("preparations") or [])
             if isinstance(p, dict) and str(p.get("concept") or "").strip()
         ],
         "source_refs": [
             str(v).strip() for v in (item.get("source_refs") or []) if str(v).strip()
         ][:12]}
        for item in (data.get("sections") or [])
        if isinstance(item, dict) and str(item.get("title") or "").strip()
    ]
    return Skeleton(
        # Planner 的非 JSON 输出一律丢弃；Reader 正文只能来自受审核的 Writer。
        markdown="",
        layers=layers,
        top3=("\n".join(str(v).strip() for v in data.get("takeaways", []) if str(v).strip())
              if isinstance(data.get("takeaways"), list) else str(data.get("top3", ""))),
        running_example=str(data.get("running_example", "")).strip(),
        paper_type=str(data.get("paper_type") or "method").strip(),
        needs_running_example=bool(data.get("needs_running_example")),
        sections=sections,
        deep_dive_title=str(data.get("deep_dive_title") or "核心内容精讲").strip(),
    ), metadata


def skeleton_output_gaps(
    output: str,
    evidence: EvidenceBundle | None = None,
) -> list[str]:
    """只验 JSON 教学计划；模型额外输出的正文既不发布，也不参与验收。"""
    _, data = extract_json_block(output)
    gaps: list[str] = []
    if not data:        return ["元数据 JSON 无法解析"]
    paper_type = str(data.get("paper_type") or "").strip()
    if paper_type not in {"method", "theory", "survey", "experiment", "system"}:
        gaps.append("paper_type 缺失或非法")
    raw_sections = data.get("sections")
    prep_refs: list[tuple[str, str]] = []  # (for_layer, 章节标题)，层名集合后验
    if not isinstance(raw_sections, list) or not raw_sections:
        gaps.append("缺少动态章节规划 sections")
    else:
        purposes: list[str] = []
        for idx, section in enumerate(raw_sections, 1):
            if not isinstance(section, dict):
                gaps.append(f"第 {idx} 个章节规划不是对象")
                continue
            purpose = str(section.get("purpose") or "").strip()
            title = str(section.get("title") or "").strip()
            if not purpose or not title:
                gaps.append(f"第 {idx} 个章节规划缺少 purpose 或 title")
                continue
            purposes.append(purpose)
            reason = str(section.get("purpose_reason") or "").strip()
            if not reason or re.fullmatch(
                r"(?:因为|由于)?(?:需要|应该|有必要)?(?:介绍|说明|讲清|解释|阐述)(?:本文|该文|论文)?(?:的)?(?:内容|原理|方法|背景|意义|价值)?[。.]?",
                reason, re.IGNORECASE,
            ):
                gaps.append(f"第 {idx} 个章节规划 purpose_reason 缺失或套话")
        # 前置概念唯一建立者：形状/字段/唯一性门禁（静默丢弃非法项是不允许的）
        owners: dict[str, str] = {}
        for idx, section in enumerate(raw_sections, 1):
            if not isinstance(section, dict):
                continue
            preps = section.get("preparations")
            if preps is not None and not isinstance(preps, list):
                gaps.append(f"第 {idx} 个章节 preparations 必须是数组")
                continue
            for prep in preps or []:
                if not isinstance(prep, dict):
                    gaps.append(f"第 {idx} 个章节 preparations 含非对象项")
                    continue
                concept = str(prep.get("concept") or "").strip()
                for_layer = str(prep.get("for_layer") or "").strip()
                if not concept or not for_layer:
                    gaps.append(f"第 {idx} 个章节 preparations 项缺少 concept 或 for_layer")
                    continue
                prep_refs.append((for_layer, str(section.get("title") or "")))
                concept_key = re.sub(r"[^\w一-鿿]+", "", concept).casefold()
                owner = str(section.get("title") or "")
                if concept_key in owners and owners[concept_key] != owner:
                    gaps.append(
                        f"前置概念「{concept}」被重复分配给多个建立章节"
                        f"（{owners[concept_key]}、{owner}）"
                    )
                else:
                    owners[concept_key] = owner
        normalized_titles = [
            re.sub(r"\s+", "", str(item.get("title") or "")).casefold()
            for item in raw_sections
            if isinstance(item, dict) and str(item.get("title") or "").strip()
        ]
        if len(set(normalized_titles)) != len(normalized_titles):
            gaps.append("动态章节标题重复")
        for required_purpose in ("orientation", "evidence", "boundary"):
            if required_purpose not in purposes:
                gaps.append(f"章节规划缺少功能：{required_purpose}")
    deep_dive_title = str(data.get("deep_dive_title") or "").strip()
    if not deep_dive_title:
        gaps.append("缺少 deep_dive_title")
    elif isinstance(raw_sections, list) and any(
        isinstance(item, dict) and str(item.get("title") or "").strip() == deep_dive_title
        for item in raw_sections
    ):
        gaps.append("deep_dive_title 不能与骨架正文章节重复")
    raw_layers = data.get("layers")
    if not isinstance(raw_layers, list):
        gaps.append("教学层必须是数组")
    elif not raw_layers:
        gaps.append("教学层至少包含一个核心机制")
    else:
        required_fields = (
            "name", "question", "why", "prerequisite", "teaching_mode", "depth",
        )
        layer_names: list[str] = []
        layer_questions: list[str] = []
        for idx, layer in enumerate(raw_layers, 1):
            if not isinstance(layer, dict):
                gaps.append(f"第 {idx} 层不是对象")
                continue
            missing = [field for field in required_fields if not str(layer.get(field, "")).strip()]
            if missing:
                gaps.append(f"第 {idx} 层缺少字段：{'、'.join(missing)}")
            wrong_types = [field for field in required_fields
                           if field in layer and not isinstance(layer.get(field), str)]
            if wrong_types:
                gaps.append(f"第 {idx} 层字段类型错误：{'、'.join(wrong_types)}")
            layer_names.append(str(layer.get("name", "")).strip())
            layer_questions.append(str(layer.get("question", "")).strip())
            hints = layer.get("source_hints")
            if (not isinstance(hints, list) or not hints
                    or not all(isinstance(v, str) and v.strip() for v in hints)):
                gaps.append(f"第 {idx} 层缺少 source_hints")
            formula_hints = layer.get("formula_hints")
            if (not isinstance(formula_hints, list)
                    or not all(isinstance(v, str) and v.strip() for v in formula_hints)):
                gaps.append(f"第 {idx} 层 formula_hints 类型错误")
            mode = str(layer.get("teaching_mode") or "").strip()
            if mode not in {"process", "comparison", "derivation", "taxonomy", "evidence"}:
                gaps.append(f"第 {idx} 层 teaching_mode 非法")
            if str(layer.get("depth") or "").strip() not in {"core", "supporting"}:
                gaps.append(f"第 {idx} 层 depth 非法")
            if mode == "process" and not (
                str(layer.get("input") or "").strip()
                and str(layer.get("output") or "").strip()
            ):
                gaps.append(f"第 {idx} 层 process 模式缺少 input 或 output")
            for debt_field in ("must_explain", "success_test"):
                debts = layer.get(debt_field)
                if (not isinstance(debts, list) or not debts
                        or not all(isinstance(v, str) and v.strip() for v in debts)):
                    gaps.append(f"第 {idx} 层缺少 {debt_field}")
                elif any(re.fullmatch(
                    r"(?:讲清|解释|说明|介绍|理解|掌握|知道)(?:本层|该层|该机制|核心)?(?:原理|作用|内容|方法)?[。.]?",
                    value.strip(), re.IGNORECASE,
                ) for value in debts):
                    gaps.append(f"第 {idx} 层 {debt_field} 过于空泛")
            mode_reason = str(layer.get("mode_reason") or "").strip()
            if not mode_reason:
                gaps.append(f"第 {idx} 层缺少 mode_reason（mode 举证）")
            # 断链声明：层数 ≥ 2 必填，into 必须是相邻层，breaks 不得为空或套话
            if len(raw_layers) >= 2:
                bim = layer.get("break_if_merged")
                if not isinstance(bim, dict):
                    gaps.append(f"第 {idx} 层缺少 break_if_merged 断链声明")
                else:
                    into = str(bim.get("into") or "").strip()
                    breaks = str(bim.get("breaks") or "").strip()
                    neighbors = {
                        str(raw_layers[j].get("name", "")).strip()
                        for j in (idx - 2, idx)
                        if 0 <= j < len(raw_layers) and isinstance(raw_layers[j], dict)
                    }
                    if into not in neighbors:
                        gaps.append(f"第 {idx} 层 break_if_merged.into 非相邻层：{into or '空'}")
                    if not breaks or re.fullmatch(
                        r"(?:讲清|解释|说明|理解|掌握|失去|缺少).{0,20}(?:原理|作用|内容|方法|知识)?[。.]?",
                        breaks, re.IGNORECASE,
                    ):
                        gaps.append(f"第 {idx} 层 break_if_merged.breaks 为空或套话")
        normalized_names = [re.sub(r"[^\w一-鿿]+", "", v).casefold()
                            for v in layer_names]
        normalized_questions = [re.sub(r"[^\w一-鿿]+", "", v).casefold()
                                for v in layer_questions]
        if len(set(normalized_names)) != len(normalized_names):
            gaps.append("教学层名称重复")
        if len(set(normalized_questions)) != len(normalized_questions):
            gaps.append("教学层核心问题重复")
        # preparations 的 for_layer 必须指向真实技术层（形状/字段校验已在章节段完成）
        if isinstance(raw_layers, list) and isinstance(raw_sections, list):
            real_layers = {str(l.get("name", "")).strip()
                           for l in raw_layers if isinstance(l, dict)}
            for for_layer, owner in prep_refs:
                if for_layer not in real_layers:
                    gaps.append(
                        f"章节「{owner}」的 preparations 指向不存在的技术层：{for_layer}"
                    )
    needs_example = data.get("needs_running_example")
    if not isinstance(needs_example, bool):
        gaps.append("needs_running_example 必须是布尔值")
    elif needs_example and (
        not isinstance(data.get("running_example"), str)
        or not str(data.get("running_example") or "").strip()
    ):
        gaps.append("规划要求贯穿案例但 running_example 为空")
    if not (str(data.get("title_zh", "")).strip() or str(data.get("title_en", "")).strip()):
        gaps.append("缺少论文标题")
    for field_name in ("title_zh", "title_en", "authors", "affiliation", "venue", "year"):
        value = data.get(field_name)
        if value is not None and not isinstance(value, (str, int, float)):
            gaps.append(f"元数据字段类型错误：{field_name}")
    if not str(data.get("tldr", "")).strip():
        gaps.append("缺少元数据字段：tldr")
    elif not isinstance(data.get("tldr"), str):
        gaps.append("元数据字段类型错误：tldr")
    keywords = data.get("keywords")
    if (keywords is not None and (
            not isinstance(keywords, list)
            or not all(isinstance(v, str) and v.strip() for v in keywords))):
        gaps.append("元数据字段类型错误：keywords")
    contributions = data.get("contributions")
    if (not isinstance(contributions, list) or not contributions
            or not all(isinstance(v, str) and v.strip() for v in contributions)):
        gaps.append("缺少元数据字段：contributions")
    takeaways = data.get("takeaways")
    if (not isinstance(takeaways, list) or not takeaways
            or not all(isinstance(value, str) and value.strip() for value in takeaways)):
        gaps.append("缺少有效 takeaways")
    if evidence is not None and evidence.long_mode:
        known = set(evidence.chunk_map)

        def validate_refs(owner: str, item: dict[str, Any]) -> None:
            refs = item.get("source_refs")
            if (not isinstance(refs, list) or not refs
                    or not all(isinstance(ref, str) and ref.strip() for ref in refs)):
                gaps.append(f"{owner}缺少 source_refs")
                return
            cleaned = [ref.strip() for ref in refs]
            unknown = sorted(set(cleaned) - known)
            if unknown:
                gaps.append(f"{owner}引用未知原文块：{'、'.join(unknown)}")
                return
            try:
                evidence.source_packet(cleaned)
            except EvidenceError as exc:
                gaps.append(f"{owner}范围过大：{exc}")

        for idx, section in enumerate(raw_sections or [], 1):
            if isinstance(section, dict):
                validate_refs(f"第 {idx} 个正文规划", section)
        for idx, layer in enumerate(raw_layers or [], 1):
            if isinstance(layer, dict):
                validate_refs(f"第 {idx} 个教学层", layer)
    return gaps


# ---- [V7.1 清理] 已停用：V3 长文按 source_hints 挑本层章节（_markdown_sections / select_layer_source），已被证据索引取代。经全项目核对生产链路零引用，保留备查 ----
# def _markdown_sections(md: str) -> list[tuple[str, str]]:
#     """复用证据层的围栏感知 Markdown 切分器。"""
#     return split_markdown_sections(md)
#
#
# def select_layer_source(paper_md: str, layer: KeyLayer, max_chars: int = 70_000) -> str:
#     """长论文按骨架给出的来源提示挑本层章节；短论文保留全文。"""
#     if len(paper_md) <= max_chars:
#         return paper_md
#     sections = _markdown_sections(paper_md)
#     hints = [*layer.source_hints, layer.name, layer.question, layer.input, layer.output]
#     terms: set[str] = set()
#     for hint in hints:
#         terms.update(t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,8}", hint))
#     priority_re = re.compile(r"abstract|introduction|conclusion|摘要|引言|结论", re.IGNORECASE)
#     ranked: list[tuple[int, int, str]] = []
#     for order, (title, section) in enumerate(sections):
#         haystack = f"{title}\n{section}".lower()
#         score = sum(3 if term in title.lower() else 1 for term in terms if term in haystack)
#         if priority_re.search(title):
#             score += 2
#         ranked.append((score, -order, section))
#     selected: list[str] = []
#     total = 0
#     for score, _, section in sorted(ranked, reverse=True):
#         if score <= 0 and selected:
#             continue
#         remaining = max_chars - total
#         if remaining <= 0:
#             break
#         selected.append(section[:remaining])
#         total += min(len(section), remaining)
#     return "\n\n".join(selected) or paper_md[:max_chars]


def continuity_context(previous_layers: list[str], max_chars: int = 8_000) -> str:
    """保留首层术语定义和最近层状态，避免长链路截掉基础概念。"""
    if not previous_layers:
        return "（这是第一层，尚无前文技术层。）"
    chunks: list[str] = []
    for text in previous_layers:
        if len(text) <= 1800:
            chunks.append(text)
        else:
            chunks.append(text[:1100] + "\n……\n" + text[-600:])
    joined = "\n\n".join(chunks)
    if len(joined) <= max_chars:
        return joined
    first = chunks[0]
    if len(first) >= max_chars:
        return first[:max_chars]
    separator = "\n\n……（中间层上下文已压缩）……\n\n"
    tail_budget = max(0, max_chars - len(first) - len(separator))
    tail = "\n\n".join(chunks[1:])[-tail_budget:] if tail_budget else ""
    return (first + separator + tail)[:max_chars]


# ---------------------------------------------------------------------------
# 图表讲解解析
# ---------------------------------------------------------------------------

def parse_figure_note(text: str) -> tuple[str, str]:
    """拆【INLINE】/【NOTE】标记 → (inline, note)；无标记时 inline 取首句。"""
    m = _INLINE_NOTE_RE.search(text)
    if m:
        return m.group(1).strip()[:80], m.group(2).strip()
    note = text.strip()
    first = re.split(r"[。！？\n]", note, maxsplit=1)[0]
    return (first + "。")[:80], note


def parse_multi_figure_notes(text: str, labels: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """按传入标签解析合并调用的多图输出，不依赖固定图号。"""
    notes: dict[str, str] = {}
    inlines: dict[str, str] = {}
    if not labels:
        return notes, inlines
    label_re = re.compile(r"【(" + "|".join(re.escape(label) for label in labels) + r")】")
    parts = label_re.split(text)
    # parts: [前言, label1, body1, label2, body2, ...]
    for i in range(1, len(parts) - 1, 2):
        label, body = parts[i], parts[i + 1]
        inline, note = parse_figure_note(body)
        inlines[label] = inline
        notes[label] = note
    missing = [l for l in labels if l not in notes]
    if missing:
        logger.warning("合并图表讲解缺标签：%s", missing)
    return notes, inlines


def _json_object(text: str) -> dict[str, Any]:
    """解析模型 JSON：fenced → 裸 JSON → 首尾大括号跨度（编码向模型常把 JSON 裹在解释文字里）。"""
    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("payload is not an object")
    return value


def _as_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _reading_path_text(path: list[str]) -> str:
    if not path:
        return ""
    if len(path) == 1:
        return f"先看{path[0]}"
    return "；".join(
        (f"先看{item}" if index == 0 else f"再看{item}")
        for index, item in enumerate(path)
    )


def _evidence_parts(value: Any) -> tuple[list[str], list[str]]:
    observations: list[str] = []
    meanings: list[str] = []
    if not isinstance(value, list):
        return observations, meanings
    for item in value:
        if not isinstance(item, dict):
            continue
        observation = str(item.get("observation") or "").strip()
        meaning = str(item.get("meaning") or "").strip()
        if observation and meaning:
            observations.append(observation)
            meanings.append(meaning)
    return observations, meanings


def parse_structured_figure_notes(
    text: str,
    assets: list[FigureAsset],
    purposes: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """结构化图表证据 → 稳定中文讲解；模型不再自由拼接最终段落。"""
    try:
        payload = _json_object(text)
    except (json.JSONDecodeError, ValueError):
        logger.warning("图表讲解未返回结构化 JSON，尝试兼容旧格式")
        return parse_multi_figure_notes(text, [asset.label for asset in assets])
    raw_items = payload.get("figures")
    if not isinstance(raw_items, list):
        return {}, {}
    by_label = {
        str(item.get("label") or "").strip(): item
        for item in raw_items if isinstance(item, dict)
    }
    notes: dict[str, str] = {}
    inlines: dict[str, str] = {}
    for asset in assets:
        item = by_label.get(asset.label)
        if not item:
            continue
        purpose = purposes[asset.label]
        inline = str(item.get("inline") or item.get("takeaway") or "").strip()[:80]
        question = str(item.get("question") or "").strip()
        takeaway = str(item.get("takeaway") or "").strip()
        path = _as_text_list(item.get("reading_path"))
        observations, meanings = _evidence_parts(item.get("evidence"))
        supports = str(item.get("supports") or "").strip()
        boundary = str(item.get("does_not_support") or "").strip()
        path_text = _reading_path_text(path)
        evidence_text = "；".join(
            f"{observation}，这说明{meaning}"
            for observation, meaning in zip(observations, meanings)
        )
        common = (
            f"【图表类型】{FIGURE_PURPOSE_NAMES[purpose]}\n"
            f"【要回答的问题】{question}\n"
            f"【先说结论】{takeaway}\n"
        )
        if purpose == "training_configuration":
            selected = "；".join(
                f"{index}. {observation}：{meaning}"
                for index, (observation, meaning) in enumerate(
                    zip(observations, meanings), 1,
                )
            )
            note = (
                common + f"【配置分组】{path_text}\n"
                f"【值得记住的设置】{selected}\n"
                f"【能说明什么】{supports}\n【不能说明什么】{boundary}"
            )
        elif purpose == "architecture_process":
            note = (
                common + f"【阅读路径】{path_text}\n"
                f"【关键节点与流向】{evidence_text}\n"
                f"【能说明什么】{supports}\n【不能说明什么】{boundary}"
            )
        elif purpose == "dataset_statistics":
            note = (
                common + f"【口径与关键数据】{path_text}；{evidence_text}\n"
                f"【数据解释】{'；'.join(meanings)}\n【使用边界】{boundary}"
            )
        elif purpose == "task_definition":
            note = (
                common + f"【对象与口径】{path_text}\n【关键区别】{evidence_text}\n"
                f"【能说明什么】{supports}\n【不能说明什么】{boundary}"
            )
        elif purpose == "method_taxonomy":
            note = (
                common + f"【比较维度】{path_text}\n【关键区别】{evidence_text}\n"
                f"【能说明什么】{supports}\n【不能说明什么】{boundary}"
            )
        elif purpose in ("qualitative_example", "error_analysis"):
            note = (
                common + f"【案例对照】{path_text}；{evidence_text}\n"
                f"【现象解释】{'；'.join(meanings)}\n【使用边界】{boundary}"
            )
        else:
            # 结果类图表不展示逐项数据走读（用户判断不需要）；关键数字由【先说结论】总括
            note = (
                common
                + f"【论证作用】这支持以下判断：{supports}。同时，{boundary}"
            )
        notes[asset.label] = note
        inlines[asset.label] = inline
    return notes, inlines


FIGURE_PURPOSE_NAMES = {
    "training_configuration": "训练配置",
    "architecture_process": "架构或流程",
    "dataset_statistics": "数据集或样本统计",
    "task_definition": "任务或数据定义",
    "method_taxonomy": "方法定义或设计空间",
    "qualitative_example": "定性案例",
    "error_analysis": "错误分析",
    "sensitivity": "敏感性分析",
    "ablation": "消融实验",
    "result_comparison": "结果比较",
    "other": "其他图表",
}


def _purpose_from_text(text: str) -> str | None:
    """按强特征识别图表用途；顺序用于避免把配置表误判成结果表。"""
    patterns = (
        ("training_configuration",
         r"training (?:configuration|details?|settings?)|hyperparameters?|implementation details?|"
         r"optimization settings?|all methods (?:load|start)|learning rate|epochs?.{0,30}(?:batch|group)|"
         r"训练配置|训练设置|超参数|实现细节|优化设置|所有方法.{0,12}(?:加载|起点)"),
        ("method_taxonomy",
         r"comparison of (?:representative )?(?:methods|approaches)|design space|training paradigm|"
         r"weighting strategy|方法(?:分类|概览|对照)|设计空间|训练范式|加权策略"),
        ("task_definition",
         r"task overview|task description|answer types?|benchmark overview|任务概览|任务定义|"
         r"任务描述|答案类型|基准概览"),
        ("dataset_statistics",
         r"dataset (?:statistics|distribution|composition)|data statistics|sample distribution|"
         r"(?:answer|reward|label|length) distribution|"
         r"数据集(?:统计|分布|构成)|样本(?:统计|分布)|数据分布|(?:答案|奖励|标签|长度)分布"),
        ("architecture_process",
         r"architecture|framework|pipeline|workflow|method overview|system overview|"
         r"架构|框架|流程图|工作流|方法概览|系统概览"),
        ("ablation", r"ablation|w/o\b|without (?:the )?(?:module|component)|消融|去掉.{0,12}(?:模块|组件)"),
        ("sensitivity", r"sensitivity|parameter sweep|varying .{0,20}(?:coefficient|temperature)|敏感性|参数扫描"),
        ("error_analysis", r"error analysis|failure analysis|failure cases?|错误分析|失败分析|错误类型"),
        ("qualitative_example",
         r"qualitative|case stud(?:y|ies)|examples? of|generated examples?|output examples?|"
         r"定性|案例分析|生成示例|输出示例|失败案例"),
        ("result_comparison",
         r"results?|performance|comparison|benchmark|accuracy|win rate|success rate|"
         r"(?:detailed |evaluation )?metrics?|precision|recall|confusion matrix|"
         r"实验结果|性能|对比|准确率|胜率|成功率|详细指标|精确率|召回率|混淆矩阵"),
    )
    for purpose, pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return purpose
    return None


def classify_figure_purpose(asset: FigureAsset, context: str = "") -> str:
    """判断一张图表应该怎样讲；图注/章节优先，正文上下文只作回退。"""
    return (
        _purpose_from_text(asset.caption)
        or _purpose_from_text(asset.section)
        or _purpose_from_text(context)
        or "other"
    )


def _figure_note_field(note: str, marker: str) -> str:
    match = re.search(rf"【{marker}】\s*(.*?)(?=【[^】]+】|$)", note, re.DOTALL)
    return match.group(1).strip() if match else ""


def estimate_table_item_count(context: str) -> int | None:
    """从来源上下文估计表格数据行数；只使用结构，不识别任何领域参数名。"""
    html_rows = len(re.findall(r"<tr\b", context, re.IGNORECASE))
    if html_rows >= 2:
        return html_rows - 1
    markdown_rows = [
        line for line in context.splitlines()
        if line.strip().startswith("|") and line.strip().endswith("|")
        and not re.match(r"^\s*\|?\s*:?-{3,}", line)
    ]
    if len(markdown_rows) >= 2:
        return len(markdown_rows) - 1
    return None


def figure_explanation_gaps(
    note: str,
    purpose: str = "result_comparison",
    source_item_count: int | None = None,
) -> list[str]:
    """只验结构化字段与危险断言，不用中文关键词猜测一段解释是否合格。"""
    if "【讲解状态】自动讲解未通过局部验收" in note:
        return []
    question = _figure_note_field(note, "要回答的问题")
    conclusion = _figure_note_field(note, "先说结论")
    gaps = []
    if len(question) < 4:
        gaps.append("图表问题")
    if len(conclusion) < 4:
        gaps.append("人话结论")

    def require(marker: str, minimum: int = 4) -> str:
        value = _figure_note_field(note, marker)
        if len(value) < minimum:
            gaps.append(marker)
        return value

    if purpose == "training_configuration":
        require("配置分组")
        settings = require("值得记住的设置")
        require("能说明什么")
        require("不能说明什么")
        configuration_items = re.findall(
            r"(?:^|[\n；;])\s*(?:[-*]|\d{1,2}[.、)])\s*([^\n；;]+)",
            settings,
        )
        if not configuration_items:
            gaps.append("关键设置采用可计数条目")
        elif not all(re.search(r"[:：]", item) for item in configuration_items):
            # observation：meaning 是 JSON schema 的结构约束，不依赖具体措辞。
            gaps.append("每条设置缺少结构化理由")
        if (source_item_count is not None and source_item_count > 4
                and len(configuration_items) / max(1, source_item_count) >= 0.7):
            gaps.append("避免逐项复述")
        unsupported = re.search(
            r"核心旋钮|关键旋钮|直接对应|(?:导致|决定|带来).{0,18}(?:提升|性能|效果)|"
            r"证明.{0,18}(?:有效|优越)|确保.{0,24}(?:只|完全).{0,12}(?:归因|来自)",
            note,
        )
        if unsupported is not None:
            gaps.append("无配置因果臆测")
        return gaps

    if purpose == "architecture_process":
        for marker in ("阅读路径", "关键节点与流向", "能说明什么", "不能说明什么"):
            require(marker)
        return gaps

    if purpose == "dataset_statistics":
        for marker in ("口径与关键数据", "数据解释", "使用边界"):
            require(marker)
        return gaps

    if purpose == "task_definition":
        for marker in ("对象与口径", "关键区别", "能说明什么", "不能说明什么"):
            require(marker)
        return gaps

    if purpose == "method_taxonomy":
        for marker in ("比较维度", "关键区别", "能说明什么", "不能说明什么"):
            require(marker)
        return gaps

    if purpose in ("qualitative_example", "error_analysis"):
        for marker in ("案例对照", "现象解释", "使用边界"):
            require(marker)
        return gaps

    role = require("论证作用")
    if "判断：。" in role:
        gaps.append("论证依据为空")
    return gaps


def missing_figure_references(paper_md: str, figures: list[FigureAsset]) -> list[str]:
    """找出正文明确引用、但解析产物中不存在的 Figure/Table 编号。"""
    referenced: set[str] = set()
    for kind, number in re.findall(r"\b(Figure|Fig\.?|Table)\s*(\d+)\b", paper_md,
                                   re.IGNORECASE):
        canonical = "Table" if kind.lower().startswith("table") else "Figure"
        referenced.add(f"{canonical} {number}")
    for kind, number in re.findall(r"(图|表)\s*(\d+)", paper_md):
        referenced.add(f"{'Table' if kind == '表' else 'Figure'} {number}")
    # 补足 "Figures 2, 3 and 4" / "Tables 1–3" 这类复数与范围写法。
    for kind, numbers in re.findall(
        r"\b(Figures?|Tables?)\s+((?:\d+[a-z]?(?:\s*(?:,|and|&|–|-|to)\s*)?)+)",
        paper_md,
        re.IGNORECASE,
    ):
        canonical = "Table" if kind.lower().startswith("table") else "Figure"
        values = [int(v) for v in re.findall(r"\d+", numbers)]
        if re.search(r"(?:–|-|\bto\b)", numbers, re.IGNORECASE) and len(values) == 2:
            values = list(range(min(values), max(values) + 1))
        referenced.update(f"{canonical} {value}" for value in values)
    available = {
        match.group(0)
        for asset in figures
        if (match := re.match(r"(?:Figure|Table)\s*\d+", asset.label))
    }
    return sorted(referenced - available)


# ---------------------------------------------------------------------------
# 方向分类（light 模型）
# ---------------------------------------------------------------------------

def classify_direction(kimi: KimiClient, title: str, tldr: str) -> str:
    """论文方向归类（light 模型），非法输出/失败回退「其他」。"""
    prompt = (
        "把下面这篇论文归入且仅归入一个方向标签，只输出标签本身：\n"
        f"可选标签：{' / '.join(DIRECTIONS)}\n\n"
        f"标题：{title}\n摘要：{tldr[:300]}"
    )
    try:
        label = kimi.chat([{"role": "user", "content": prompt}], heavy=False).strip()
    except KimiError as exc:
        logger.warning("方向分类失败（归为其他）：%s", exc)
        return "其他"
    label = label.strip("。 .\"'")
    for d in DIRECTIONS:
        if d in label:
            return d
    logger.warning("方向分类输出非法（%r），归为其他", label)
    return "其他"


# ---------------------------------------------------------------------------
# 教学型精读引擎
# ---------------------------------------------------------------------------

class ReadingEngine:
    def __init__(
        self,
        kimi: KimiClient,
        prompts_dir: Path = PROMPTS_DIR,
    ) -> None:
        self.kimi = kimi
        self.prompts_dir = prompts_dir
        self._vision_ok: bool | None = None
        self._fingerprint_cache: str | None = None

    def workflow_fingerprint(self) -> str:
        """缓存语义指纹：协议代码或任一精读 prompt 变化都会自动失效旧检查点。"""
        if self._fingerprint_cache is not None:
            return self._fingerprint_cache
        digest = hashlib.sha256(PIPELINE_VERSION.encode())
        for path in sorted(self.prompts_dir.glob("reading_*.md")):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        for path in (Path(__file__), Path(__file__).with_name("reading_evidence.py"),
                     Path(__file__).with_name("reading_graph.py"),
                     Path(__file__).with_name("reading_plan.py")):
            digest.update(path.read_bytes())
        self._fingerprint_cache = digest.hexdigest()[:20]
        return self._fingerprint_cache

    def _artifact_sig(self, artifact_id: str, revision: int) -> dict[str, Any]:
        """产物签名（随产物落盘）：workflow 指纹 + unit semantic_revision + schema。
        修复/重规划改动合同后，旧正文/旧审核不得静默复用。"""
        from dataclasses import asdict
        from .reading_graph import ArtifactSignature
        from .reading_plan import PLAN_SCHEMA_VERSION
        return asdict(ArtifactSignature(
            artifact_id=artifact_id,
            revision=revision,
            prompt_digest=self.workflow_fingerprint(),
            schema_digest=PLAN_SCHEMA_VERSION,
            producer_version=PIPELINE_VERSION,
        ))

    def _artifact_sig_matches(self, stored: Any, artifact_id: str, revision: int) -> bool:
        """命中校验：无签名（历史产物）宽容命中；签名存在但不匹配 → 失效重写。"""
        if not isinstance(stored, dict):
            return True
        from .reading_graph import ArtifactSignature
        try:
            current = ArtifactSignature(
                artifact_id=str(stored.get("artifact_id") or ""),
                revision=int(stored.get("revision") or 0),
                prompt_digest=str(stored.get("prompt_digest") or ""),
                schema_digest=str(stored.get("schema_digest") or ""),
                producer_version=str(stored.get("producer_version") or ""),
                input_revisions={str(k): int(v) for k, v in
                                 (stored.get("input_revisions") or {}).items()},
            )
        except (TypeError, ValueError):
            return False
        return current.matches(
            ArtifactSignature(**self._artifact_sig(artifact_id, revision)))

    # ---- 图像输入实测（每进程一次）----

    def probe_vision(self, client: Any | None = None) -> tuple[bool, int]:
        """实测图像输入，返回 (是否支持, 本次产生的模型调用数)。"""
        if self._vision_ok is not None:
            return self._vision_ok, 0
        try:
            from PIL import Image
            import io

            buf = io.BytesIO()
            Image.new("RGB", (8, 8), (255, 255, 255)).save(buf, format="PNG")
            tiny_png = base64.b64encode(buf.getvalue()).decode()
        except Exception as exc:
            self._vision_ok = False
            logger.warning("无法生成探测图片（改用文本证据）：%s", exc)
            return self._vision_ok, 0
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "这张图里有什么？一句话。"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{tiny_png}"}},
            ],
        }]
        try:
            result = (client or self.kimi).chat_full(messages, heavy=True)
            self._vision_ok = bool(result.content.strip())
            logger.info("图像输入实测：%s", "支持" if self._vision_ok else "返回为空，改用文本证据")
        except KimiError as exc:
            self._vision_ok = False
            logger.warning("图像输入实测失败（改用文本证据）：%s", exc)
        return self._vision_ok, 1

    # ---- prompt 构建 ----

    def _style(self) -> str:
        return _load(self.prompts_dir / "paper_reading.md")

    def build_skeleton_prompt(self, paper_md: str, figures: list[FigureAsset],
                              formulas: list[FormulaAsset] | None = None) -> str:
        return _render(
            _load(self.prompts_dir / "reading_skeleton.md"),
            STYLE=self._style(),
            PAPER_CONTENT=paper_md,
        ) + (
            "\n\n## 本文图表清单（配对成功，写作时按编号引用）\n\n"
            + figure_inventory_text(figures)
            + "\n\n## 本文复杂公式清单\n\n"
            + formula_inventory_text(formulas or [])
        )

    def build_layer_prompt(self, skeleton: Skeleton, layer: KeyLayer,
                           idx: int, total: int, paper_md: str,
                           figures: list[FigureAsset],
                           previous_layers: list[str] | None = None,
                           formulas: list[FormulaAsset] | None = None,
                           evidence: EvidenceBundle | None = None) -> str:
        """单层教学 prompt：带相关来源章节和前文连续状态。"""
        formulas = formulas or []
        formula_by_key = {normalize_formula_label(asset.label): asset for asset in formulas}
        hinted: list[FormulaAsset] = []
        unmatched: list[str] = []
        for hint in layer.formula_hints:
            asset = formula_by_key.get(normalize_formula_label(hint))
            if asset is None:
                unmatched.append(hint)
            elif asset not in hinted:
                hinted.append(asset)
        if unmatched:
            logger.warning("技术层「%s」公式提示未匹配原文清单：%s",
                           layer.name, "、".join(unmatched))
        return _render(
            _load(self.prompts_dir / "reading_layer.md"),
            STYLE=self._style(),
            LAYER_CN=_cn(idx),
            LAYER_NAME=layer.name,
            LAYER_IDX=str(idx),
            LAYER_TOTAL=str(total),
            LAYER_WHY=layer.why,
            LAYER_QUESTION=layer.question or f"{layer.name}到底怎样工作？",
            LAYER_PREREQUISITE=layer.prerequisite or "只假设读者理解前一层内容",
            LAYER_MODE=layer.teaching_mode,
            LAYER_DEPTH=layer.depth,
            LAYER_INPUT=layer.input or "从论文原文识别本层输入",
            LAYER_OUTPUT=layer.output or "从论文原文识别本层输出",
            MUST_EXPLAIN="\n".join(f"- {item}" for item in layer.must_explain),
            SUCCESS_TEST="\n".join(f"- {item}" for item in layer.success_test),
            RUNNING_EXAMPLE_POLICY=(
                "使用骨架中的同一贯穿对象，保持变量和状态连续"
                if skeleton.needs_running_example
                else "本文不依赖贯穿案例；使用原文对象、定义或证据，不得临时编造故事"
            ),
            RUNNING_EXAMPLE=skeleton.running_example or "无；本层不得临时编造案例",
            SKELETON=self._plan_text(skeleton, focus_layer=layer.name),
            CONTINUITY_CONTEXT=continuity_context(previous_layers or []),
            FIGURE_HINTS=figure_inventory_text(figures),
            FORMULA_HINTS=formula_inventory_text(
                hinted, empty_scope="layer" if formulas else "paper",
            ),
            PAPER_CONTENT=(
                evidence.source_packet(layer.source_refs, [*layer.source_hints, layer.name, layer.question])
                if evidence is not None and evidence.long_mode
                # 短文模式传全文：各层共享同一前缀，网关前缀缓存跨层命中（缓存价）
                else paper_md
            ),
        )

    @staticmethod
    def _plan_text(skeleton: Skeleton, focus_layer: str = "") -> str:
        """写手侧骨架视图（局部披露）：focus_layer 非空时（技术层写手），
        本层给完整合同，其他层只列名——知道"别讲别人的"，但看不到别人的债务细节。
        章节仅列标题与功能（不披露建立义务分配细节之外的内部信息）。"""
        sections_view = [
            {"title": str(s.get("title") or ""), "purpose": str(s.get("purpose") or "")}
            for s in skeleton.sections
        ]
        layers_view: list[dict[str, Any]] = []
        for layer in skeleton.layers:
            if focus_layer and layer.name != focus_layer:
                layers_view.append({"name": layer.name})
                continue
            layers_view.append({
                "name": layer.name,
                "question": layer.question,
                "why": layer.why,
                "prerequisite": layer.prerequisite,
                "teaching_mode": layer.teaching_mode,
                "depth": layer.depth,
                "must_explain": layer.must_explain,
                "success_test": layer.success_test,
                "source_refs": layer.source_refs,
            })
        return json.dumps({
            "paper_type": skeleton.paper_type,
            "sections": sections_view,
            "deep_dive_title": skeleton.deep_dive_title,
            "needs_running_example": skeleton.needs_running_example,
            "running_example": skeleton.running_example,
            "layers": layers_view,
        }, ensure_ascii=False, indent=2)

    def build_section_prompt(
        self,
        skeleton: Skeleton,
        section: dict[str, Any],
        paper_md: str,
        figures: list[FigureAsset],
        evidence: EvidenceBundle,
        continuity_state: dict[str, Any],
    ) -> tuple[str, str]:
        refs = [str(v) for v in (section.get("source_refs") or [])]
        title = str(section.get("title") or "").strip()
        purpose = str(section.get("purpose") or "overview").strip()
        source = (
            evidence.source_packet(refs, [title, purpose])
            if evidence.long_mode
            else paper_md
        )
        prep_text = _preparations_text(section)
        prompt = _render(
            _load(self.prompts_dir / "reading_section.md"),
            SECTION_TITLE=title,
            SECTION_PURPOSE=purpose,
            PAPER_TYPE=skeleton.paper_type,
            PREPARATIONS=prep_text,
            CONTINUITY_CONTEXT=json.dumps(continuity_state, ensure_ascii=False, indent=2),
            FIGURE_HINTS=figure_inventory_text(figures),
            SOURCE_CONTENT=source,
        )
        return prompt, source

    def _review_section(
        self,
        client: Any,
        content: str,
        contract: dict[str, Any],
        continuity_state: dict[str, Any],
        source: str,
        *,
        full_map: str = "",
        scope_only: bool = False,
        preparations: str = "（无）",
    ) -> dict[str, Any]:
        """局审：全审=事实+范围（reading_section_review.md），
        轻审=只审范围（reading_scope_review.md，不查事实）。
        审核器带全局归属图判断复述与合法回指；写手不可见图。"""
        if scope_only:
            prompt = _render(
                _load(self.prompts_dir / "reading_scope_review.md"),
                SOURCE_CONTENT=source,
                PREPARATIONS=preparations,
                FULL_MAP=full_map,
                SECTION_CONTENT=content,
            )
        else:
            prompt = _render(
                _load(self.prompts_dir / "reading_section_review.md"),
                SECTION_CONTRACT=json.dumps(contract, ensure_ascii=False, indent=2),
                FULL_MAP=full_map,
                CONTINUITY_STATE=json.dumps(continuity_state, ensure_ascii=False, indent=2),
                SECTION_CONTENT=content,
                SOURCE_CONTENT=source,
            )
        def _unavailable(reason: str) -> dict[str, Any]:
            # 产出优先：审核器故障不杀正文——该单元按 review_unavailable 放行并标注
            logger.warning("章节审核不可用（%s），按 review_unavailable 放行", reason)
            return {"pass": True, "tier": "review_unavailable",
                    "issues": [{"severity": "中",
                                "problem": "审核服务不可用，本单元未经完整审核"}]}

        try:
            raw = client.chat(
                [{"role": "user", "content": prompt}],
                heavy=True, json_mode=True, low_reasoning=True,
                max_tokens=LOCAL_REVIEW_OUTPUT_TOKENS,
            )
        except KimiError as exc:
            return _unavailable(str(exc))
        for _attempt in range(2):
            try:
                return _json_object(raw)
            except (json.JSONDecodeError, ValueError):
                # 非法输出不立即判死：带修正指令重试（最多两次），仍非法才降级
                try:
                    raw = client.chat(
                        [{"role": "user", "content": (
                            prompt + "\n\n## 本次必须修正\n"
                            "上一次输出无法解析为 JSON。只输出一个 JSON 对象，不要任何解释、"
                            "前后缀或 markdown 围栏；字段与要求保持一致。"
                        )}],
                        heavy=True, json_mode=True, low_reasoning=True,
                        max_tokens=LOCAL_REVIEW_OUTPUT_TOKENS,
                    )
                except KimiError as exc:
                    return _unavailable(str(exc))
        try:
            return _json_object(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("章节审核连续非法 JSON，按 review_unavailable 放行：%s",
                           str(raw).strip()[:120])
            return {"pass": True, "tier": "review_unavailable",
                    "issues": [{"severity": "中",
                                "problem": "审核服务输出异常，本单元未经完整审核"}]}

    def _try_patch(
        self,
        client: Any,
        content: str,
        correction: str,
        source: str,
        *,
        structural: bool = False,
    ) -> str | None:
        """修补式改写：只让模型输出 find/replace 最小补丁，代码机械拼回。

        再生≠修补：全量重写让每条句子重新掷骰，常修 A 坏 B。补丁把动作空间
        压到审核指出的几处，引入新错的概率大幅下降。结构类问题（标题/嵌行标记）
        不走补丁——标题由代码修正、标记由 _hoist_media_markers 机械提升。
        任何一步失败（解析/锚点缺失/歧义）都返回 None，调用方回退全量重写。
        """
        if structural:
            return None
        prompt = _render(
            _load(self.prompts_dir / "reading_section_patch.md"),
            SOURCE_CONTENT=source,
            ISSUES=correction,
            SECTION_CONTENT=content,
        )
        try:
            raw = client.chat(
                [{"role": "user", "content": prompt}],
                heavy=True, json_mode=True, low_reasoning=True,
                max_tokens=LOCAL_REVIEW_OUTPUT_TOKENS,
            )
            payload = _json_object(raw)
        except (KimiError, json.JSONDecodeError, ValueError) as exc:
            logger.info("修补式改写失败（%s），回退全量重写", type(exc).__name__)
            return None
        patched = _apply_edits(content, payload.get("edits"))
        if patched is None:
            logger.info("补丁锚点缺失或歧义，回退全量重写")
            return None
        if patched == content and payload.get("edits"):
            return None  # 有补丁但内容没变（异常防御）
        return patched, list(payload.get("edits") or [])

    def _patch_review(
        self,
        client: Any,
        edits: list[dict],
        patched_content: str,
        correction: str,
        source: str,
    ) -> dict[str, Any]:
        """补丁复核：只审改动区域（合同覆盖与连续性首轮已验、补丁外内容未变，
        不重审整节）。输出与局审同构（pass/issues），continuity 不更新。
        复核不可用（KimiError/连续非法 JSON）口径：revert_patch=True——放弃
        未验证的补丁、保留补丁前正文、标注，重写 0 次（与 _try_patch 对齐）。"""
        edits_summary = json.dumps([
            {"find": str(e.get("find") or "")[:80],
             "replace": str(e.get("replace") or "")[:80]}
            for e in edits if isinstance(e, dict)
        ], ensure_ascii=False)
        context = _patched_context(patched_content, edits)
        prompt = _render(
            _load(self.prompts_dir / "reading_patch_review.md"),
            SOURCE_CONTENT=source,
            EDITS_SUMMARY=edits_summary,
            ISSUES=correction,
            PATCHED_CONTEXT=context,
        )

        def _unavailable(reason: str) -> dict[str, Any]:
            logger.warning("补丁复核不可用（%s），放弃补丁保留补丁前正文", reason)
            return {"pass": False, "revert_patch": True,
                    "issues": [{"severity": "中",
                                "problem": "补丁复核不可用，已保留补丁前正文"}]}

        try:
            raw = client.chat(
                [{"role": "user", "content": prompt}],
                heavy=True, json_mode=True, low_reasoning=True,
                max_tokens=LOCAL_REVIEW_OUTPUT_TOKENS,
            )
        except KimiError as exc:
            return _unavailable(f"调用失败：{exc}")
        for _attempt in range(2):
            try:
                return _json_object(raw)
            except (json.JSONDecodeError, ValueError):
                try:
                    raw = client.chat(
                        [{"role": "user", "content": (
                            prompt + "\n\n## 本次必须修正\n"
                            "上一次输出无法解析为 JSON。只输出一个 JSON 对象，不要任何解释、"
                            "前后缀或 markdown 围栏；字段与要求保持一致。"
                        )}],
                        heavy=True, json_mode=True, low_reasoning=True,
                        max_tokens=LOCAL_REVIEW_OUTPUT_TOKENS,
                    )
                except KimiError as exc:
                    return _unavailable(f"调用失败：{exc}")
        try:
            return _json_object(raw)
        except (json.JSONDecodeError, ValueError):
            return _unavailable("连续非法 JSON")

    def build_key_figures_messages(self, assets: list[FigureAsset],
                                   paper_md: str, skeleton_md: str = "",
                                   correction: str = "",
                                   include_images: bool = True) -> list:
        """图表教学调用：先识别用途，再采用对应的阅读合同。"""
        labels = "、".join(a.label for a in assets)
        figure_inputs = [
            (asset, self._figure_context(paper_md, asset, 1200)) for asset in assets
        ]
        purposes = {
            asset.label: classify_figure_purpose(asset, context)
            for asset, context in figure_inputs
        }
        per_figure = "\n\n".join(
            f"{asset.label} 图注：{asset.caption}\n所属章节：{asset.section or '未知'}\n"
            f"系统判定用途：{FIGURE_PURPOSE_NAMES[purposes[asset.label]]}\n"
            f"正文上下文：{context}"
            for asset, context in figure_inputs
        )
        source_notice = (
            "本次已附原图，请以图中实际内容为准。"
            if include_images else
            "本次无法读取原图，只能依据图注与正文上下文解释。不得臆测图内读数、颜色、"
            "坐标或布局；把无法核实的内容放进 does_not_support，明确说明文本证据边界，"
            "再解释正文能够确认的用途。"
        )
        role_rules = {
            "training_configuration": (
                "把设置按功能分组；evidence 只选择复现、预算或结果解读所需的最小充分集合，"
                "每项说明选择理由。小表可以全部保留，大表不得逐行抄写。配置值本身不是性能证据。"
            ),
            "architecture_process": (
                "reading_path 跟随一个输入从起点到输出；evidence 只保留决定机制的节点和流向。"
                "架构存在不等于性能有效。"
            ),
            "dataset_statistics": (
                "先交代统计单位和口径，再选决定分布判断的读数；解释数据偏斜、覆盖和实验解读边界。"
            ),
            "task_definition": (
                "说明对象、输入输出或类别如何划分，以及这些区别为何影响后续方法或评测；"
                "不得逐行翻译名称，也不得把任务定义当性能证据。"
            ),
            "method_taxonomy": (
                "先给比较维度，再指出方法在设计空间中的关键区别；没有性能指标时不得宣布谁更强。"
            ),
            "qualitative_example": (
                "挑最有区分度的少量输入输出案例，解释现象；个例不能代替总体统计。"
            ),
            "error_analysis": (
                "按错误模式而非案例顺序组织，说明触发条件和影响；不能从少量错误样本估计总体发生率。"
            ),
            "ablation": (
                "明确改变或删掉了什么、关键指标变化及代价；只能支持被实际操纵的组件结论。"
            ),
            "sensitivity": (
                "说明扫描变量、稳定区间和拐点；未扫描区域不能外推，单次最优点不是普遍最优。"
            ),
            "result_comparison": (
                "明确比较对象、指标方向和控制条件；选择足以支撑结论的关键差值，并报告代价或例外。"
            ),
            "other": (
                "只解释图表直接展示的问题、阅读路径、必要证据和边界；不确定用途时保守陈述。"
            ),
        }
        contracts = "\n".join(
            f"- {asset.label}：{FIGURE_PURPOSE_NAMES[purposes[asset.label]]}。"
            f"{role_rules[purposes[asset.label]]}"
            for asset in assets
        )
        text = (
            f"以下是论文中的 {len(assets)} 张图表（{labels}）。"
            "先按系统判定的用途讲，不得把配置表、架构图、数据统计和性能结果套进同一个模板。"
            "你的任务不是翻译英文图注，也不是把坐标轴和数据逐项念一遍。\n\n"
            "原始图注会在文档中单独展示，所以【NOTE】不必重复翻译整段图注。"
            "关键数据必须讲，不能因为防止复读就全删；但只保留回答该类图表问题所需的"
            "数据，并立即解释为什么要看它。"
            "只把论文来源直接支持的内容写成事实；推测必须标明，来源没有的因果关系不得补写。"
            f"\n\n{source_notice}\n\n"
            "## 每张图表的用途与专用要求\n"
            + contracts
            + "\n\n## 输出协议\n只输出一个合法 JSON 对象，不要 Markdown 围栏或解释性前后缀。"
            "figures 必须按输入顺序逐项覆盖，label 原样复制。evidence 中每项都必须同时包含"
            " observation（原图或正文可核对的读数/结构）和 meaning（为什么需要看它）。"
            "reading_path 是阅读顺序或功能分组，不是整段图注翻译。supports 只写直接支持的判断；"
            "does_not_support 必须写清证据边界。\n"
            '{"figures":[{"label":"原始标签","inline":"不超过40字的人话结论",'
            '"primary_role":"系统给定用途","secondary_roles":[],"contains_performance_evidence":false,'
            '"question":"图表回答的问题","takeaway":"不依赖逐项报数的结论",'
            '"reading_path":["先看或第一功能组","再看或第二功能组"],'
            '"evidence":[{"observation":"必要证据","meaning":"证据含义"}],'
            '"supports":"能够支持的判断","does_not_support":"不能支持的判断"}]}'
            + f"\n\n论文教学主线（只用于定位图表作用）：\n{skeleton_md[:1800] or '未提供'}"
            + f"\n\n各图图注与上下文：\n{per_figure}"
        )
        if correction:
            text += f"\n\n## 上一次讲解的问题\n{correction}\n请重新输出上述格式，不要沿用数据复读式写法。"
        content: str | list[dict[str, Any]] = text
        if include_images:
            content = [{"type": "text", "text": text}]
            for a in assets:
                try:
                    b64 = base64.b64encode(a.img_path.read_bytes()).decode()
                    mime = mimetypes.guess_type(a.img_path.name)[0] or "image/jpeg"
                    content.append({"type": "image_url",
                                    "image_url": {"url": f"data:{mime};base64,{b64}"}})
                except OSError as exc:
                    raise ReadingError(f"读取原图失败（{a.label}）：{exc}") from exc
        return [
            {"role": "system", "content": (
                "你是论文课的助教。先判断图表承担的是复现、流程说明、数据描述、案例展示"
                "还是结果证据，再按对应合同讲解。保留必要数据，但不逐项报数，不从配置值"
                "臆测性能因果，也不把结构图当实验证据。严格按用户给定 JSON schema 输出。"
            )},
            {"role": "user", "content": content},
        ]

    def _figure_context(self, paper_md: str, asset: FigureAsset, span: int = 500) -> str:
        """优先按完整图注定位，其次在所属章节内兼容 Figure/Fig./图 等写法。"""
        caption_probe = re.sub(r"\s+", " ", asset.caption).strip()[:80]
        words = re.findall(r"[A-Za-z0-9_-]+|[\u4e00-\u9fff]{2,}", caption_probe)
        if len(words) >= 3:
            # 图注原文常在编号后插入冒号、破折号或换行，按稳定词序宽松定位。
            caption_pattern = r"[\s:：.。,_\-–—/()（）]{0,12}".join(
                re.escape(word) for word in words[:8]
            )
            caption_match = re.search(caption_pattern, paper_md, re.IGNORECASE)
            if caption_match:
                idx = caption_match.start()
                return paper_md[max(0, idx - span):idx + span]
        number_match = re.search(r"(\d+)", asset.label)
        probes = [asset.label]
        if number_match:
            n = number_match.group(1)
            if asset.is_table:
                probes += [f"Table {n}", f"表 {n}", f"表{n}"]
            else:
                probes += [f"Figure {n}", f"Fig. {n}", f"Fig {n}", f"图 {n}", f"图{n}"]
        search_start = 0
        if asset.section:
            section_idx = paper_md.lower().find(asset.section.lower())
            if section_idx >= 0:
                search_start = section_idx
        candidates = [paper_md.lower().find(probe.lower(), search_start) for probe in probes]
        candidates = [idx for idx in candidates if idx >= 0]
        if not candidates:
            return ""
        idx = min(candidates)
        return paper_md[max(0, idx - span): idx + span]

    def _call_skeleton(
        self,
        paper_md: str,
        figures: list[FigureAsset],
        formulas: list[FormulaAsset],
        *,
        client: Any | None = None,
        workspace: ReadingWorkspace | None = None,
        evidence: EvidenceBundle | None = None,
    ) -> tuple[str, int]:
        """骨架调用：流式低压思考 → 流式常规。

        kimi-k3-highspeed 在复杂长 prompt 上会把 max_tokens 全烧在 reasoning
        （正文为空）；reasoning_effort=low 实测 ~20s 出稿。参数不被某 provider
        支持（400）或仍失败时调整同一模型的调用参数，绝不切换模型或 API。
        网关默认 max_tokens 偏小，骨架（正文+```json 清单）经常被截断在正文
        中段、JSON 围栏整个丢失——显式抬预算；思考型模型（k3 系）reasoning 本身
        就要烧 ~8k token，8192 会挤得正文一个字都出不来，故抬到 16384。
        长输入从第一次调用就使用流式，避免客户端把完整生成时间当成
        read timeout。每次都执行完整结构验收，两次均不合格则停止发布。
        """
        client = client or self.kimi
        prompt = self.build_skeleton_prompt(paper_md, figures, formulas)
        if workspace:
            cached = workspace.get_text("plan")
            if cached and not skeleton_output_gaps(cached, evidence):
                logger.info("教学计划命中检查点")
                return cached, 0
        attempts = [
            ("当前模型：流式低压思考", dict(heavy=True, low_reasoning=True)),
            ("当前模型：流式常规", dict(heavy=True)),
        ]
        last_err: Exception | None = None
        correction = ""
        for i, (label, kwargs) in enumerate(attempts, 1):
            try:
                logger.info("骨架调用：%s", label)
                current_prompt = prompt
                if correction:
                    current_prompt += (
                        "\n\n## 上一次骨架未通过验收\n"
                        f"{correction}\n请重新完整输出，不要省略任何必填章节和 JSON 字段。"
                    )
                out = client.chat(
                    [{"role": "user", "content": current_prompt}],
                    max_tokens=PLANNER_OUTPUT_TOKENS, **kwargs,
                )
            except KimiParameterError as exc:
                logger.warning("骨架调用失败（%s）：%s", label, exc)
                last_err = exc
                continue
            except KimiError:
                # 认证、额度、限流重试耗尽等不是骨架参数问题：立即上报，
                # 避免同一坏 API 再空跑后续调用方式。
                raise
            gaps = skeleton_output_gaps(out, evidence)
            if not gaps:
                _, skeleton_data = extract_json_block(out)
                known_formulas = {normalize_formula_label(asset.label) for asset in formulas}
                unknown_hints = sorted({
                    hint
                    for layer in (skeleton_data.get("layers") or [])
                    if isinstance(layer, dict)
                    for hint in (layer.get("formula_hints") or [])
                    if normalize_formula_label(hint) not in known_formulas
                })
                if unknown_hints:
                    gaps.append("公式标签不在原文清单：" + "、".join(unknown_hints))
            if not gaps:
                if workspace:
                    workspace.put_text("plan", out)
                return out, i
            correction = "；".join(gaps)
            logger.warning("骨架验收失败（%s）：%s", label, correction)
        detail = correction or str(last_err or "未知错误")
        raise ReadingError(f"骨架连续两次未通过教学验收：{detail}")


    def _review_plan_and_repair(
        self,
        plan: "ReadingPlan",
        graph: "SemanticGraph",
        reviewer_client: Any,
        planner_client: Any,
        source: str,
        paper_md: str,
        meta_hint: dict[str, str],
        arxiv_id: str,
        workspace: Any,
        progress: Callable[[str], None],
        evidence_chunks: set[str] | None = None,
    ) -> tuple["ReadingPlan", Any, dict[str, Any]]:
        """Plan Reviewer 四状态循环：clean 放行；repairable 走 RepairEngine；
        unsupported_repair 转有界局部重规划（一次）；blocked 标记后止步。
        Reviewer 输出格式/版本错误只重试 Reviewer，绝不重跑 Planner。
        审核侧任何故障（KimiError/连续非法输出/修复空转）口径统一：
        标注 + 返回当前 Plan 继续，绝不判死。"""
        from .reading_repair import RepairAction, RepairEngine, RepairRejected

        def _call_reviewer(correction: str = "") -> dict[str, Any]:
            plan_view = plan.to_dict()
            prompt = _render(
                _load(self.prompts_dir / "reading_plan_review.md"),
                SOURCE_CONTENT=source,
                PLAN_JSON=json.dumps(plan_view, ensure_ascii=False)[:60_000],
                FULL_MAP=_full_map_text(skeleton_from_plan(plan)),
            )
            if correction:
                prompt += ("\n\n## 上次输出未通过程序校验\n" + correction
                           + "\n请按协议重新输出。")
            try:
                raw = reviewer_client.chat(
                    [{"role": "user", "content": prompt}],
                    heavy=True, json_mode=True, low_reasoning=True,
                    max_tokens=LOCAL_REVIEW_OUTPUT_TOKENS,
                )
            except KimiError as exc:
                # 产出优先：审核服务不可用不判死——特殊状态由调用方标注继续
                logger.warning("Plan Reviewer 调用失败（review_unavailable）：%s", exc)
                return {"status": "review_unavailable",
                        "reason": str(exc)[:200], "issues": [], "repairs": []}
            return _json_object(raw)

        def skeleton_from_plan(p: "ReadingPlan"):
            sk, _ = p.to_skeleton()
            return sk

        def _annotate_and_return(marker: str) -> tuple["ReadingPlan", Any, dict[str, Any]]:
            plan.meta.setdefault("issues", []).append(marker)
            return plan, *plan.to_skeleton()

        retries = 0
        replan_used = 0  # 每 Plan 最多一次局部重规划
        repair_rounds = 0  # repairable 总轮次上限（合法 repair 也不得无限空转）
        while True:
            try:
                review = _call_reviewer()
                status = str(review.get("status") or "").strip()
                if status == "review_unavailable":
                    # 产出优先：与 blocked 同语义——标注后继续用当前 Plan
                    logger.warning("Plan 审核不可用（review_unavailable，标注继续）")
                    return _annotate_and_return(
                        "Plan 审核不可用 review_unavailable："
                        + str(review.get("reason") or ""))
                if status == "clean":
                    if review.get("repairs"):
                        raise RepairRejected("clean 状态不得携带 repairs")
                    check_gaps = _plan_review_check_gaps(review.get("checks"), plan, source)
                    if check_gaps:
                        # 语义准入举证不完整 = reviewer 输出不完整：correction 重试一次
                        logger.warning("Plan 审核 checks 不合格（带修正重试一次）：%s",
                                       "；".join(check_gaps[:3]))
                        retry_review = _call_reviewer(
                            "checks 不合格：" + "；".join(check_gaps[:6]))
                        retry_status = str(retry_review.get("status") or "").strip()
                        if retry_status == "clean" and not retry_review.get("repairs"):
                            retry_gaps = _plan_review_check_gaps(
                                retry_review.get("checks"), plan, source)
                        else:
                            retry_gaps = ["重试后仍未给出合格的 clean 审核"]
                        if retry_gaps:
                            # 仍不合格 → 标注 degraded 继续，绝不判死
                            logger.warning("语义准入审核不完整（标注继续）：%s",
                                           "；".join(retry_gaps[:3]))
                            plan.meta.setdefault("issues", []).append(
                                "语义准入审核不完整：" + "；".join(retry_gaps[:4]))
                    return plan, *plan.to_skeleton()
                if status == "blocked":
                    # 产出优先：标记后继续（证据不足不阻断已生成的 Plan）
                    logger.warning("Plan 审核 blocked（标记继续）：%s",
                                   json.dumps(review.get("issues"),
                                              ensure_ascii=False)[:200])
                    return _annotate_and_return(
                        "Plan 审核 blocked：" + json.dumps(
                            review.get("issues"), ensure_ascii=False)[:200])
                if status == "unsupported_repair":
                    if replan_used >= 1:
                        # 产出优先：标注后继续，不循环也不杀
                        logger.warning("局部重规划后仍 unsupported（标注继续）")
                        return _annotate_and_return(
                            "unsupported_repair（已局部重规划一次）："
                            + json.dumps(review.get("issues"),
                                         ensure_ascii=False)[:200])
                    replan_used += 1
                    logger.warning("Plan 需局部重规划：%s",
                                   json.dumps(review.get("issues"),
                                              ensure_ascii=False)[:200])
                    progress("🔧 局部重规划（一次）……")
                    try:
                        plan = self._local_replan(
                            plan, graph, review, planner_client, paper_md, workspace,
                            evidence_chunks=evidence_chunks)
                    except ReadingError as replan_exc:
                        # 产出优先：重规划失败不杀，标注后继续用当前 Plan
                        logger.warning("局部重规划失败（标注继续）：%s", replan_exc)
                        return _annotate_and_return(f"局部重规划失败：{replan_exc}")
                    continue
                if status == "repairable":
                    repair_rounds += 1
                    if repair_rounds > 6:
                        # 产出优先：修复轮次有界——超限标注返回，不无限烧 token
                        logger.warning("Plan 修复超过 6 轮仍未 clean（标注继续）")
                        return _annotate_and_return(
                            "Plan 修复超过 6 轮仍未 clean（degraded 继续）")
                    actions = [RepairAction.from_dict(a)
                               for a in (review.get("repairs") or [])]
                    engine = RepairEngine(graph, evidence_chunks=evidence_chunks)
                    new_plan, stats = engine.apply(plan, actions)
                    if workspace:
                        workspace.put_json("plan-json", new_plan.to_dict())
                    logger.info("Plan 修复：%d 操作，影响 %d 单元，新 plan_revision=%d",
                                stats.applied, len(stats.changed_units),
                                new_plan.plan_revision)
                    plan = new_plan
                    if not stats.changed_units:
                        # no-op 熔断：修复无实质变化，不再 continue 空转烧 token
                        logger.warning("Plan 修复无实质变化（no-op 熔断，标注继续）")
                        return _annotate_and_return(
                            "Plan 修复无实质变化（no-op 熔断，标注继续）")
                    continue  # 修复后再审一次
                raise RepairRejected(f"非法状态：{status or '缺失'}")
            except (RepairRejected, json.JSONDecodeError, ValueError) as exc:
                retries += 1
                if retries > 2:
                    # 产出优先：审核输出连续非法不判死——标注 degraded 返回当前 Plan
                    logger.warning("Plan 审核重试耗尽（degraded 继续）：%s", exc)
                    return _annotate_and_return(f"Plan 审核重试耗尽（degraded 继续）：{exc}")
                logger.warning("Plan Reviewer 输出被拒（第 %d 次）：%s", retries, exc)
                try:
                    review = _call_reviewer(str(exc))
                    status = str(review.get("status") or "").strip()
                    if status == "review_unavailable":
                        logger.warning("Plan 审核不可用（review_unavailable，标注继续）")
                        return _annotate_and_return(
                            "Plan 审核不可用 review_unavailable："
                            + str(review.get("reason") or ""))
                    if status == "clean":
                        if review.get("repairs"):
                            raise RepairRejected("clean 状态不得携带 repairs")
                        check_gaps = _plan_review_check_gaps(
                            review.get("checks"), plan, source)
                        if check_gaps:
                            # 重试后 checks 仍不合格：标注继续（不再消耗重试）
                            logger.warning("语义准入审核不完整（标注继续）：%s",
                                           "；".join(check_gaps[:3]))
                            plan.meta.setdefault("issues", []).append(
                                "语义准入审核不完整：" + "；".join(check_gaps[:4]))
                        return plan, *plan.to_skeleton()
                    if status == "repairable":
                        actions = [RepairAction.from_dict(a)
                                   for a in (review.get("repairs") or [])]
                        engine = RepairEngine(graph, evidence_chunks=evidence_chunks)
                        plan, stats = engine.apply(plan, actions)
                        if workspace:
                            workspace.put_json("plan-json", plan.to_dict())
                        return plan, *plan.to_skeleton()
                    raise RepairRejected(f"重试后状态仍非法：{status}")
                except (RepairRejected, json.JSONDecodeError, ValueError) as exc2:
                    # 产出优先：同上——标注 degraded 返回当前 Plan，不动 Plan
                    logger.warning("Plan 审核重试耗尽（degraded 继续）：%s", exc2)
                    return _annotate_and_return(f"Plan 审核重试耗尽（degraded 继续）：{exc2}")

    def _local_replan(
        self,
        plan: "ReadingPlan",
        graph: "SemanticGraph",
        review: dict[str, Any],
        planner_client: Any,
        source: str,
        workspace: Any,
        evidence_chunks: set[str] | None = None,
    ) -> "ReadingPlan":
        """有界局部重规划：只重出受影响单元的合同，合并后完整校验。
        v1 简化：用骨架规划器对受影响区域重出字段（unit_id 保留，身份不动）。
        source 为论文原文（paper_md）：锚点偏移基于它，scope 内的 source spans
        从这里截取。"""
        from .reading_plan import ReadingPlan, validate_plan

        # 显式 target_ids：Reviewer 必须给稳定 unit_id，不做层名/标题模糊匹配
        explicit = [str(t) for t in (review.get("target_ids") or [])]
        if not explicit:
            explicit = [
                str(item.get("target_id") or "")
                for item in (review.get("issues") or []) if isinstance(item, dict)
            ]
        explicit = [t for t in explicit if t]
        if not explicit:
            raise ReadingError("局部重规划缺少显式 target_ids（禁止模糊定位）")
        unknown = [t for t in explicit if t not in plan.units]
        if unknown:
            raise ReadingError("局部重规划指向不存在的单元：" + "、".join(unknown))
        affected = set(explicit)
        correction = json.dumps(review.get("issues"), ensure_ascii=False)
        # 范围输入：目标单元 + 顺序直接邻居 + 语义依赖邻居 + preparations 归属
        # + source spans（有界截断）+ 只读全局图
        order_all = (plan.order.get("sections") or []) + (plan.order.get("layers") or [])
        neighbors: set[str] = set()
        for uid in affected:
            if uid in order_all:
                i = order_all.index(uid)
                # 邻居是单元 ID 不是下标（此前把整数 j 加进集合，邻居从未生效）
                neighbors.update(order_all[j] for j in (i - 1, i + 1)
                                 if 0 <= j < len(order_all) and order_all[j] not in affected)
        # 语义依赖邻居：与受影响单元共享 concept/assertion 绑定，或经 preparations 相连
        affected_bindings: set[str] = set()
        for uid in affected:
            unit = plan.units.get(uid)
            if unit is not None:
                affected_bindings |= set(unit.concept_ids) | set(unit.assertion_ids)
        for uid, unit in plan.units.items():
            if uid in affected or uid in neighbors:
                continue
            if (set(unit.concept_ids) | set(unit.assertion_ids)) & affected_bindings:
                neighbors.add(uid)
        for prep in plan.preparations:
            owner = str(prep.get("owner_section_id") or "")
            targets = [str(t) for t in (prep.get("target_layer_ids") or [])]
            if owner in affected:
                neighbors.update(t for t in targets
                                 if t in plan.units and t not in affected)
            elif any(t in affected for t in targets):
                if owner in plan.units and owner not in affected:
                    neighbors.add(owner)
        scoped_units = {uid: plan.units[uid].data
                        for uid in (*affected, *neighbors) if uid in plan.units}
        scoped_preps = [p for p in plan.preparations
                        if p.get("owner_section_id") in affected
                        or any(t in affected for t in (p.get("target_layer_ids") or []))]
        # source spans：受影响单元绑定节点的锚点摘录，从原文有界截取（每单元 ≤1500 字符）
        span_excerpts: dict[str, str] = {}
        for uid in sorted(affected):
            unit = plan.units.get(uid)
            if unit is None:
                continue
            excerpts: list[str] = []
            for nid in (*unit.concept_ids, *unit.assertion_ids):
                node = graph.nodes.get(nid)
                anchor = node.anchor if node is not None else {}
                start, end = anchor.get("start", -1), anchor.get("end", -1)
                if 0 <= start < end <= len(source):
                    excerpts.append(source[start:end][:600])
            for ref in unit.data.get("source_refs") or []:
                if str(ref).strip():
                    excerpts.append(f"[source_ref: {ref}]")
            if excerpts:
                span_excerpts[uid] = "\n---\n".join(excerpts)[:1_500]
        scope_context = json.dumps({
            "target_ids": sorted(affected),
            "units": scoped_units,
            "preparations": scoped_preps,
            "source_spans": span_excerpts,
            "full_map": _full_map_text(plan.to_skeleton()[0]),
        }, ensure_ascii=False, indent=1)[:20_000]
        work = ReadingPlan.from_dict(plan.to_dict())
        for uid in affected:
            unit = work.units[uid]
            prompt = (
                f"你是论文精读规划器。以下是{('章节' if unit.kind == 'section' else '技术层')}"
                f"「{unit.data.get('title') or unit.data.get('name')}」的当前合同与审核意见，"
                "请只输出修正后的合同 JSON 对象（字段同输入），不要输出其他内容。\n\n"
                f"当前合同：{json.dumps(unit.data, ensure_ascii=False)}\n"
                f"审核意见：{correction}\n"
                f"受限范围（只读上下文，不得输出范围外单元）：{scope_context}\n"
                "规则：禁止修改 unit_id/kind/绑定；只修复审核指出的问题。"
            )
            raw = planner_client.chat(
                [{"role": "user", "content": prompt}],
                heavy=True, json_mode=True, low_reasoning=True,
                max_tokens=LOCAL_REVIEW_OUTPUT_TOKENS,
            )
            try:
                fixed = _json_object(raw)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ReadingError(f"局部重规划输出非法 JSON：{uid}") from exc
            forbidden = {"unit_id", "kind", "concept_ids", "assertion_ids"}
            for key, value in fixed.items():
                if key not in forbidden and key in unit.data:
                    unit.data[key] = value
        gaps = validate_plan(work, graph, evidence_chunks=evidence_chunks)
        if gaps:
            raise ReadingError("局部重规划后 Plan 不变量未通过：" + "；".join(gaps))
        # digest 重算；仅当任一受影响单元 digest 实际变化才递增 plan_revision
        any_changed = False
        for uid in affected:
            unit = work.units[uid]
            view = ({"preparations": [p for p in work.preparations
                                      if p.get("owner_section_id") == uid]}
                    if unit.kind == "section" else None)
            sem_changed, disp_changed = unit.refresh_digests(view)
            any_changed = any_changed or sem_changed or disp_changed
        if any_changed:
            work.plan_revision += 1
        if workspace:
            workspace.put_json("plan-json", work.to_dict())
        return work
    # ---- [V7.1 清理] 已停用：V3 骨架语义审核，生产已改用 _review_plan_and_repair。经全项目核对生产链路零引用，保留备查 ----
    # def _review_skeleton_semantic(
    #     self,
    #     client: Any,
    #     skeleton_out: str,
    #     source: str,
    #     chunk_ids: set[str] | None = None,
    # ) -> list[str]:
    #     """骨架语义审核（正文动工前）：审核器带原文证据裁决断链声明真实性、
    #     preparations 重复建立/欠分配/下放错误、purpose·mode 举证分流。
    #     fail-closed：非法 JSON 修正重试后仍异常按未通过处理；pass=true 必须
    #     携带非空、非套话、带来源的 checks 明细，否则不予通过。"""
    #     _, skeleton_data = extract_json_block(skeleton_out)
    #     prompt = _render(
    #         _load(self.prompts_dir / "reading_skeleton_review.md"),
    #         SOURCE_CONTENT=source,
    #         SKELETON_JSON=json.dumps(skeleton_data, ensure_ascii=False, indent=2),
    #     )
    #     raw = client.chat(
    #         [{"role": "user", "content": prompt}],
    #         heavy=True, json_mode=True, low_reasoning=True,
    #         max_tokens=LOCAL_REVIEW_OUTPUT_TOKENS,
    #     )
    #     review: dict[str, Any] | None = None
    #     for _attempt in range(2):
    #         try:
    #             review = _json_object(raw)
    #             break
    #         except (json.JSONDecodeError, ValueError):
    #             raw = client.chat(
    #                 [{"role": "user", "content": (
    #                     prompt + "\n\n## 本次必须修正\n"
    #                     "上一次输出无法解析为 JSON。只输出一个 JSON 对象，不要任何解释、"
    #                     "前后缀或 markdown 围栏。"
    #                 )}],
    #                 heavy=True, json_mode=True, low_reasoning=True,
    #                 max_tokens=LOCAL_REVIEW_OUTPUT_TOKENS,
    #             )
    #     if review is None:
    #         # 末次重试的结果同样要解析：直接丢弃会白烧一次调用还误判未通过
    #         try:
    #             review = _json_object(raw)
    #         except (json.JSONDecodeError, ValueError):
    #             review = None
    #     if review is None:
    #         # fail-closed：审核输出两轮仍非法 → 视为本轮规划未通过（进入有界重试）
    #         logger.warning("骨架语义审核连续非法 JSON，按未通过处理")
    #         return ["骨架语义审核输出异常（连续非法 JSON），请保持字段完整重出骨架"]
    #     issues_raw = review.get("issues") or []
    #     checks = review.get("checks")
    #     # pass=true 但携带问题：矛盾输出按问题处理，不得直接放行
    #     if _review_bool(review.get("pass")) is True and not issues_raw:
    #         # 程序级举证门禁：逐层覆盖/必需检查项/evidence 可追溯，全部 fail-closed
    #         return _skeleton_review_check_gaps(
    #             checks, skeleton_data, chunk_ids, source,
    #         )
    #     gaps: list[str] = []
    #     for item in issues_raw:
    #         if not isinstance(item, dict):
    #             continue
    #         problem = str(item.get("problem") or "").strip()
    #         if not problem:
    #             continue
    #         kind = str(item.get("kind") or "")
    #         location = str(item.get("location") or "")
    #         action = str(item.get("action") or "").strip()
    #         gaps.append(
    #             f"[{kind}]{location}：{problem}" + (f"（动作：{action}）" if action else "")
    #         )
    #     return gaps or ["骨架语义审核未通过但未说明具体问题"]

    # ---- 主流程 ----

    @staticmethod
    def _merge_continuity(state: dict[str, Any], review: dict[str, Any]) -> None:
        update = review.get("continuity_update")
        if not isinstance(update, dict):
            return
        for key in ("defined_terms", "established_claims", "open_questions"):
            values = update.get(key)
            if not isinstance(values, list):
                continue
            target = state.setdefault(key, [])
            for value in values:
                text = str(value).strip()
                if text and text not in target:
                    target.append(text)

    @staticmethod
    def _local_review_passed(review: Any) -> bool:
        """高严重度（无来源事实/逻辑矛盾）就地拦截；中/低严重度（措辞、标记方式）
        放行并记入 carried_issues 随审阅存档，终审通读全篇时再独立复核。
        缺省 severity 按高处理（保守，防止审核漏标绕过闸门）。"""
        if not isinstance(review, dict):
            return False
        high: list[str] = []
        minor: list[str] = []
        issues = review.get("issues")
        if isinstance(issues, list):
            for item in issues:
                if not isinstance(item, dict):
                    continue
                problem = str(item.get("problem") or "").strip()
                if not problem:
                    continue
                if "结构重复：" in problem:
                    high.append(problem)  # 结构重复硬判高严重度，模型降级无效
                    continue
                if str(item.get("severity") or "高").strip() == "高":
                    high.append(problem)
                else:
                    minor.append(problem)
        if high:
            return False
        if minor:
            review["carried_issues"] = minor
            logger.info("局审放行 %d 条中低严重度意见（记入存档待终审复核）", len(minor))
            return True
        return _review_bool(review.get("pass")) is True

    @staticmethod
    def _review_issues(review: dict[str, Any]) -> str:
        issues = review.get("issues")
        if not isinstance(issues, list):
            return str(review.get("summary") or "审核未说明具体问题")
        details = [
            str(item.get("problem") or "").strip()
            for item in issues if isinstance(item, dict) and str(item.get("problem") or "").strip()
        ]
        return "；".join(details[:8]) or "审核未说明具体问题"

    def run(
        self,
        paper_md: str,
        figures: list[FigureAsset] | None = None,
        formulas: list[FormulaAsset] | None = None,
        arxiv_id: str = "",
        meta_hint: dict[str, str] | None = None,
        progress: Callable[[str], None] = lambda s: None,
        workspace: ReadingWorkspace | None = None,
        rework_round: int = 0,
        calls_carry: int = 0,
    ) -> ReadingResult:
        """生成、验收并终审一篇完整精读；只返回可发布的 canonical body。

        rework_round：终审未过时第几轮定点返工（0=首跑）；失败单元被定位后
        作废其检查点并递归重进本方法，其余产物全部走检查点续跑。
        calls_carry：跨返工轮累计的模型调用数（日志/结果如实汇总）。
        """
        figures = figures or []
        formulas = formulas or []
        meta_hint = meta_hint or {}
        timer = StageTimer()
        model_calls = 0
        if workspace:
            workspace.ensure_signature(self.workflow_fingerprint())
        # 任务启动后让 Planner/Writer/Reviewer/Vision 冻结同一个全局选择。
        # 中途手动切换只影响下一项任务，不污染正在生成的文档；每个角色均为单候选。
        if isinstance(self.kimi, KimiClient):
            routes = self.kimi.frozen_reading()
            planner_client: Any = routes.planner
            writer_client: Any = routes.writer
            reviewer_client: Any = routes.reviewer
            vision_client: Any = routes.vision
            if workspace:
                workspace.put_json("route-manifest", routes.route_manifest())
        else:
            planner_client = writer_client = reviewer_client = vision_client = self.kimi
        # 测试 fake 保持旧注入协议；真实客户端对所有正式正文显式提供 16K 输出余量。
        writer_kwargs = (
            {"max_tokens": WRITER_OUTPUT_TOKENS}
            if isinstance(self.kimi, KimiClient) else {}
        )

        # 禁止尾部硬截断。长论文通过证据索引缩小每次调用的上下文，但 Writer 仍回到原文块。
        paper_md = strip_references(paper_md, hard_cap=None)
        missing_figures = missing_figure_references(paper_md, figures)
        if missing_figures:
            # 产出优先：缺图不判死，图号+图注占位降级（该图的讲解走透明降级状态）
            logger.warning("缺少对应图片，按占位降级继续：%s", "、".join(missing_figures))
            progress("⚠️ 部分图表缺图，将以图注占位降级：" + "、".join(missing_figures[:3]))

        progress("🧭 正在建立全文证据索引（长论文会逐块处理，原文不会截尾）……")
        evidence: EvidenceBundle | None = None
        evidence_calls = 0
        pre_plan_notes: list[str] = []  # 规划前的降级标注，metadata 就位后并入
        try:
            evidence, evidence_calls = EvidenceIndexer(planner_client, self.prompts_dir).build(
                paper_md, workspace=workspace, progress=progress,
            )
            planner_material = evidence.planner_material(paper_md)
        except EvidenceError as exc:
            # 产出优先：证据索引失败 → 确定性重建分块（分块不依赖模型，失败发生在
            # 建卡阶段），全部块降级为原文导航卡继续
            logger.warning("证据索引失败，全部导航卡降级：%s", exc)
            from .reading_evidence import (
                LONG_PAPER_CHARS, _source_fallback_card, split_source,
            )
            chunks = split_source(paper_md)
            evidence = EvidenceBundle(
                chunks, [_source_fallback_card(chunk) for chunk in chunks],
                len(paper_md) > LONG_PAPER_CHARS,
            )
            evidence_calls = 0  # 失败前已发生的计数不可知，按 0 如实从简
            note = f"证据索引不可用，全部导航卡降级：{str(exc)[:160]}"
            pre_plan_notes.append(note)
            progress("⚠️ 证据索引不可用，按原文导航卡降级继续")
            try:
                planner_material = evidence.planner_material(paper_md)
            except EvidenceError:
                # 导航卡汇总仍超索引上限 → 退回极简规划材料（标题清单+文首摘录），不硬停
                logger.warning("降级证据索引仍过长，退回极简规划材料")
                titles = "、".join(chunk.title for chunk in chunks[:40])
                planner_material = (
                    f"论文「{meta_hint.get('title', arxiv_id or '未命名论文')}」"
                    f"章节标题：{titles}\n\n文首摘录：\n{paper_md[:6_000]}"
                )
        model_calls += evidence_calls
        timer.mark("证据索引")

        # ---- 教学计划：只产 JSON；兼容旧测试/历史检查点中的正文+JSON 结构。----
        progress("🧠 教学计划生成中（只规划，不在一次调用里硬写整篇）……")
        # 相似论文只有标题和相似度，没有来源正文，不注入生成提示词，避免编造差异。
        from .reading_graph import (
            GraphError, SemanticGraph, build_semantic_graph, validate_graph_for_plan,
        )
        from .reading_plan import ReadingPlan, validate_plan
        try:
            graph, graph_stats = build_semantic_graph(
                paper_md, evidence, planner_client, self.prompts_dir, workspace=workspace,
            )
        except (GraphError, KimiError) as exc:
            # 产出优先：图构建失败 → 空图降级继续（绑定自然减少，缺口进 graph_issues）
            logger.warning("语义图构建失败，空图降级继续：%s", exc)
            graph = SemanticGraph()
            graph.issues.append({"kind": "build_failed", "reason": str(exc)[:200]})
            graph_stats = {"extract_calls": 0, "link_calls": 0, "anchor_dropped": 0}
            pre_plan_notes.append(f"语义图构建失败，空图降级：{str(exc)[:160]}")
        model_calls += graph_stats.get("extract_calls", 0) + graph_stats.get("link_calls", 0)
        graph_gaps = validate_graph_for_plan(graph)
        if graph_gaps:
            # 产出优先：图缺口记入 graph_issues 继续（绑定受影响的节点自然减少）
            logger.warning("语义图验收缺口（降级继续）：%s", "；".join(graph_gaps[:4]))
            graph.issues.append({"kind": "validation", "gaps": graph_gaps[:6]})
        plan: ReadingPlan | None = None
        if workspace:
            raw_plan = workspace.get_json("plan-json")
            if isinstance(raw_plan, dict):
                try:
                    plan = ReadingPlan.from_dict(raw_plan)
                    logger.info("canonical Plan 命中检查点：plan_revision=%d", plan.plan_revision)
                except (ValueError, TypeError, KeyError):
                    plan = None
        plan_needs_review = plan is None
        if plan is None:
            try:
                skeleton_out, skeleton_calls = self._call_skeleton(
                    planner_material, figures, formulas,
                    client=planner_client, workspace=workspace, evidence=evidence,
                )
                model_calls += skeleton_calls
                timer.mark("骨架")
                skeleton, metadata = parse_skeleton_full(skeleton_out)
                metadata = {**meta_hint, **metadata}
                if not metadata.get("title_zh") and not metadata.get("title_en"):
                    metadata["title_zh"] = meta_hint.get("title", arxiv_id or "未命名论文")
                from .reading_plan import PlanCollisionError
                try:
                    plan = ReadingPlan.from_skeleton(
                        skeleton, graph, self.workflow_fingerprint()[:16],
                        meta={"metadata": metadata},
                    )
                except PlanCollisionError as collision:
                    # 有界重规划一次：碰撞信息注入；再碰撞/调用失败/骨架不合格统一
                    # 收敛到下方 except ReadingError 的确定性降级 Plan，绝不判死。
                    # （降级骨架三节 purpose/reason 各异 + 单层，ID seed 唯一不自碰撞）
                    logger.warning("Plan ID 碰撞，有界重规划一次：%s", collision)
                    replan_prompt = self.build_skeleton_prompt(
                        planner_material, figures, formulas,
                    ) + ("\n\n## 上一次规划存在 ID 碰撞\n" + str(collision)
                         + "\n请合并重复单元或明确各自语义角色后重新输出。")
                    try:
                        out = planner_client.chat(
                            [{"role": "user", "content": replan_prompt}],
                            heavy=True, max_tokens=PLANNER_OUTPUT_TOKENS,
                        )
                        model_calls += 1
                        gaps2 = skeleton_output_gaps(out, evidence)
                        if gaps2:
                            raise ReadingError("重规划骨架不合格：" + "；".join(gaps2))
                        skeleton, metadata = parse_skeleton_full(out)
                        metadata = {**meta_hint, **metadata}
                        plan = ReadingPlan.from_skeleton(
                            skeleton, graph, self.workflow_fingerprint()[:16],
                            meta={"metadata": metadata},
                        )
                    except (PlanCollisionError, KimiError) as replan_exc:
                        raise ReadingError(
                            f"碰撞重规划仍不可用：{replan_exc}") from replan_exc
            except ReadingError as plan_exc:
                # 产出优先：骨架/规划失败 → 确定性降级 Plan（三功能节 + 单层综合精讲）
                logger.warning("骨架/规划失败，启用确定性降级 Plan：%s", plan_exc)
                skeleton, metadata = _degraded_skeleton(meta_hint, arxiv_id)
                plan = ReadingPlan.from_skeleton(
                    skeleton, graph, self.workflow_fingerprint()[:16],
                    meta={"metadata": metadata},
                )
                plan_needs_review = False
                progress("⚠️ 教学规划不可用，按确定性降级 Plan 继续生成")
            if workspace:
                workspace.put_json("plan-json", plan.to_dict())
        else:
            timer.mark("骨架")
            skeleton, metadata = plan.to_skeleton()
        evidence_chunk_ids = {c.chunk_id for c in evidence.chunks}
        plan_gaps = validate_plan(plan, graph, evidence_chunks=evidence_chunk_ids)
        if plan_gaps:
            # 产出优先：净化（移除指向不存在单元的引用）后继续；仍坏则降级 Plan
            logger.warning("Plan 不变量缺口（净化后继续）：%s", "；".join(plan_gaps[:4]))
            for bucket in ("sections", "layers"):
                plan.order[bucket] = [uid for uid in plan.order.get(bucket, [])
                                      if uid in plan.units]
            metadata.setdefault("plan_issues", plan_gaps)
            plan_gaps = validate_plan(plan, None)
            if plan_gaps and not plan.order.get("layers"):
                skeleton, metadata = _degraded_skeleton(meta_hint, arxiv_id)
                plan = ReadingPlan.from_skeleton(
                    skeleton, graph, self.workflow_fingerprint()[:16],
                    meta={"metadata": metadata},
                )
                plan_needs_review = False
        # 规划前阶段的降级标注并入结果卡（透明状态，不判死）
        for note in pre_plan_notes:
            metadata.setdefault("evidence_issues", []).append(note)
        if graph.issues:
            metadata.setdefault("graph_issues", []).extend(
                json.dumps(item, ensure_ascii=False)[:160] for item in graph.issues[:6])
        unbound_core = [plan.units[uid].data.get("name")
                        for uid in plan.order.get("layers") or []
                        if plan.units[uid].data.get("depth") == "core"
                        and not plan.units[uid].assertion_ids]
        if unbound_core:
            logger.warning("core 层未绑定 assertion（降级继续）：%s", "、".join(unbound_core))
            metadata.setdefault("graph_issues", []).append(
                "core 层未绑定 assertion：" + "、".join(str(n) for n in unbound_core))

        from .reading_graph import DependencyManifest
        manifest = DependencyManifest()
        if workspace:
            raw_manifest = workspace.get_json("dependency-manifest")
            if isinstance(raw_manifest, dict):
                for artifact, inputs in (raw_manifest.get("inputs") or {}).items():
                    manifest.register(artifact, inputs)

        # ---- Plan Reviewer 四状态（ID 化之后，带原文证据）----
        if plan_needs_review:  # 命中 plan-json 的新建路径才审；缓存命中不再审
            plan, skeleton, metadata = self._review_plan_and_repair(
                plan, graph, reviewer_client, planner_client,
                evidence.review_material() if evidence.long_mode else paper_md,
                paper_md, meta_hint, arxiv_id, workspace, progress,
                evidence_chunks=evidence_chunk_ids,
            )
            if workspace:
                # 审核标注（blocked/review_unavailable/degraded 等）随 Plan 落盘可审计
                workspace.put_json("plan-json", plan.to_dict())
        continuity: dict[str, Any] = {
            "defined_terms": [], "established_claims": [], "open_questions": [],
        }

        # Planner 的任何文字都不发布；Writer 按计划逐节读取原文包并接受局部审核。
        overview_sections: list[str] = []
        degraded: list[str] = []  # 透明降级撤下的节/层，写入 metadata 供结果卡与报告统计
        full_map = _full_map_text(skeleton)  # 审核器全局归属图（写手不可见）
        for idx, section_contract in enumerate(skeleton.sections, 1):
            title = str(section_contract.get("title") or "").strip()
            light_review = (
                str(section_contract.get("purpose") or "").strip()
                in LIGHT_REVIEW_SECTION_PURPOSES
            )
            progress(f"正文生成 {idx}/{len(skeleton.sections)}：{title}")
            unit_id = plan.order["sections"][idx - 1]
            key = f"body-{unit_id}"
            review_key = f"review-{unit_id}"
            content = workspace.get_text(key) if workspace else None
            if content and str(section_contract.get("title") or "").strip():
                content = _fit_heading(content, 2, title)  # 改名机械同步，不耗模型
            review = workspace.get_json(review_key) if workspace else None
            if isinstance(review, dict) and not self._artifact_sig_matches(
                    review.get("_artifact_sig"), key,
                    plan.units[unit_id].semantic_revision):
                # 产物签名不匹配（合同 revision/工作流指纹已变）：失效重写本单元
                logger.info("产物签名不匹配，失效重写：%s", key)
                content, review = None, None
            if content and (
                not _starts_with_heading(content, 2, title)
                or _media_marker_gaps(content)
            ):
                review = None
            if not content or not self._local_review_passed(review):
                prompt, source = self.build_section_prompt(
                    skeleton, section_contract, paper_md, figures, evidence, continuity,
                )
                if workspace:
                    rework_note = workspace.get_text(f"rework-{key}")
                    if rework_note:
                        prompt += ("\n\n## 前次终审意见（本单元返工原因，必须逐条解决）\n"
                                   + rework_note)
                try:
                    content = writer_client.chat_full(
                        [{"role": "user", "content": prompt}], heavy=True,
                        **writer_kwargs,
                    ).content
                except KimiError as exc:
                    # 产出优先：Writer 失败 → 占位段+degraded 标注，继续后续单元；
                    # 占位不写检查点，下次重跑可重新生成
                    logger.warning("章节「%s」生成失败，占位降级继续：%s", title, exc)
                    degraded.append(f"章节「{title}」")
                    overview_sections.append(
                        f"## {title}\n\n> ⚠️ 本节生成失败（{exc}），"
                        "以占位段降级交付，请对照原论文对应章节核实。")
                    continue
                content = _hoist_media_markers(content)  # 格式机械修，省一次模型重写
                content = _strip_inline_latex(content)  # LaTeX 机械转纯文本
                content = _fit_heading(content, 2, title)  # 标题机械归一
                model_calls += 1
                section_gaps = _media_marker_gaps(content)
                if not _starts_with_heading(content, 2, title):
                    section_gaps.append(f"必须以二级标题「{title}」开头")
                if section_gaps:
                    review = {"pass": False, "_structural": True, "issues": [{
                        "problem": "；".join(section_gaps),
                    }]}
                elif light_review:
                    # 轻审=只审范围（越界/跨单元重复，收窄判定），事实归全审与终审
                    review = self._review_section(
                        reviewer_client, content, section_contract, continuity, source,
                        full_map=full_map, scope_only=True,
                        preparations=_preparations_text(section_contract),
                    )
                    review["tier"] = "light"
                    model_calls += 1
                else:
                    review = self._review_section(
                        reviewer_client, content, section_contract, continuity, source,
                        full_map=full_map,
                    )
                    model_calls += 1
                for _rewrite in range(LOCAL_REVIEW_MAX_REWRITES):
                    if self._local_review_passed(review):
                        break
                    correction = self._review_issues(review if isinstance(review, dict) else {})
                    # 结构重复禁入补丁：整章需按全局归属图重审，局部补丁无权放行
                    struct_dup = isinstance(review, dict) and any(
                        "结构重复：" in str(it.get("problem") or "")
                        for it in (review.get("issues") or []) if isinstance(it, dict)
                    )
                    patch_edits: list[dict] | None = None
                    pre_patch: tuple[str, dict[str, Any]] | None = None
                    if (_rewrite == 0 and not struct_dup
                            and isinstance(review, dict) and not review.get("_structural")):
                        patched_result = self._try_patch(writer_client, content, correction, source)
                        if patched_result is not None:
                            pre_patch = (content, review)
                            content, patch_edits = patched_result
                            model_calls += 1
                    if patch_edits is None:
                        retry_prompt = (
                            prompt + "\n\n## 审核要求重写\n" + correction
                            + "\n重新完整输出本章，不要输出解释或其他章节。"
                        )
                        try:
                            content = writer_client.chat_full(
                                [{"role": "user", "content": retry_prompt}], heavy=True,
                                **writer_kwargs,
                            ).content
                        except KimiError as exc:
                            # Writer 重写失败：保留上一版正文，出循环走批注交付
                            logger.warning("章节「%s」重写失败，保留上一版并批注：%s",
                                           title, exc)
                            break
                        content = _hoist_media_markers(content)  # 格式机械修，省一次模型重写
                        content = _strip_inline_latex(content)  # LaTeX 机械转纯文本
                        content = _fit_heading(content, 2, title)  # 标题机械归一
                        model_calls += 1
                    retry_gaps = _media_marker_gaps(content)
                    if not _starts_with_heading(content, 2, title):
                        retry_gaps.append(f"必须以二级标题「{title}」开头")
                    if retry_gaps:
                        # 产出优先：结构机械修复已穷尽 → 批注该节继续，不判死
                        logger.warning("章节「%s」结构缺口无法机械修复，批注继续：%s",
                                       title, "；".join(retry_gaps))
                        degraded.append(f"章节「{title}」")
                        content = _annotate_degraded(
                            content, "结构缺口：" + "；".join(retry_gaps))
                        review = {"pass": True, "tier": "degraded-structure",
                                  "issues": [], "degraded": True}
                    if light_review:
                        review = self._review_section(
                            reviewer_client, content, section_contract, continuity,
                            source, full_map=full_map, scope_only=True,
                            preparations=_preparations_text(section_contract),
                        )
                        review["tier"] = "light"
                        model_calls += 1
                    elif patch_edits is not None:
                        # 补丁复核：只审改动区域，不重审整节（合同覆盖首轮已验）
                        review = self._patch_review(
                            reviewer_client, patch_edits, content, correction, source,
                        )
                        model_calls += 1
                        if (isinstance(review, dict) and review.get("revert_patch")
                                and pre_patch is not None):
                            # 补丁复核不可用：放弃未验证补丁，保留补丁前正文并标注
                            # （重写 0 次）；原审核意见随下方批注横幅交付
                            content, review = pre_patch
                            review = {**review, "issues": [
                                *(review.get("issues") or []),
                                {"severity": "中",
                                 "problem": "补丁复核不可用，已保留补丁前正文"},
                            ]}
                            break
                    else:
                        review = self._review_section(
                            reviewer_client, content, section_contract, continuity, source,
                            full_map=full_map,
                        )
                        model_calls += 1
                if not self._local_review_passed(review):
                    # 保留+批注：三轮仍带高严重度问题 → 全文保留，顶部插入存疑横幅；
                    # 不再整节撤下（句级错误不连坐），终审独立复核核心链路
                    brief = self._review_issues(review if isinstance(review, dict) else {})
                    logger.warning("章节「%s」三轮审核未过，保留全文加批注：%s", title, brief[:120])
                    degraded.append(f"章节「{title}」")
                    content = _annotate_degraded(content, brief)
                    review = {**(review if isinstance(review, dict) else {}),
                              "degraded": True, "degrade_mode": "annotated", "pass": False}
                if workspace:
                    workspace.put_text(key, content)
                    if isinstance(review, dict):
                        # 产物签名随审核落盘：workflow 指纹 + unit semantic_revision
                        review = {**review, "_artifact_sig": self._artifact_sig(
                            key, plan.units[unit_id].semantic_revision)}
                    workspace.put_json(review_key, review)
                    workspace.invalidate_from((f"rework-{key}",))  # 返工意见一次有效
                    prev_dep = {}
                    if idx > 1:
                        prev_uid = plan.order["sections"][idx - 2]
                        prev_dep[f"body-{prev_uid}"] = 1
                    manifest.register(
                        f"body-{unit_id}",
                        {f"plan:{unit_id}": plan.units[unit_id].semantic_revision,
                         **prev_dep,
                         **_unit_binding_deps(plan.units[unit_id])},
                    )
                    manifest.register(
                        f"review-{unit_id}",
                        {f"body-{unit_id}": 1},
                    )
                    workspace.put_json("dependency-manifest",
                                       {"inputs": manifest._inputs})
            if isinstance(review, dict) and not review.get("degraded"):
                self._merge_continuity(continuity, review)
            overview_sections.append(content.strip())
        overview_md = "\n\n".join(overview_sections)

        progress(f"✅ 教学计划完成（识别出 {len(skeleton.layers)} 个技术层），开始分层精讲")

        # ---- 调用 2..N+1：每层一次 heavy 教学，缺理解闭环时修正重试 ----
        total_layers = len(skeleton.layers)
        layer_sections: list[str] = []
        timer.mark("精讲开始")
        for i, layer in enumerate(skeleton.layers, 1):
            progress(f"精讲 层{i}/{total_layers}：{layer.name}")
            layer_uid = plan.order["layers"][i - 1]
            layer_key = f"body-{layer_uid}"
            layer_review_key = f"review-{layer_uid}"
            content = workspace.get_text(layer_key) if workspace else None
            local_review = workspace.get_json(layer_review_key) if workspace else None
            if isinstance(local_review, dict) and not self._artifact_sig_matches(
                    local_review.get("_artifact_sig"), layer_key,
                    plan.units[layer_uid].semantic_revision):
                # 产物签名不匹配（合同 revision/工作流指纹已变）：失效重写本层
                logger.info("产物签名不匹配，失效重写：%s", layer_key)
                content, local_review = None, None
            # 层 prompt 要打包原文包，缓存命中时白构建一次太贵；只在真要调模型前懒构建。
            prompt: str | None = None

            def layer_prompt() -> str:
                nonlocal prompt
                if prompt is None:
                    prompt = self.build_layer_prompt(
                        skeleton, layer, i, total_layers, paper_md, figures,
                        previous_layers=layer_sections,
                        formulas=formulas,
                        evidence=evidence,
                    )
                    if workspace:
                        rework_note = workspace.get_text(f"rework-{layer_key}")
                        if rework_note:
                            prompt += ("\n\n## 前次终审意见（本单元返工原因，必须逐条解决）\n"
                                       + rework_note)
                return prompt

            if not content or not self._local_review_passed(local_review):
                try:
                    content = writer_client.chat_full(
                        [{"role": "user", "content": layer_prompt()}], heavy=True,
                        **writer_kwargs,
                    ).content
                except KimiError as exc:
                    # 产出优先：本层生成失败 → 占位层+degraded 标注，继续后续层；
                    # 占位不写检查点，下次重跑可重新生成
                    logger.warning("技术层「%s」生成失败，占位降级继续：%s", layer.name, exc)
                    degraded.append(f"技术层「{layer.name}」")
                    layer_sections.append(
                        f"### {layer.name}\n\n> ⚠️ 本层生成失败（{exc}），"
                        "以占位段降级交付，请对照原论文核实。")
                    timer.mark(f"层{i}")
                    continue
                content = _hoist_media_markers(content)  # 格式机械修，省一次模型重写
                content = _strip_inline_latex(content)  # LaTeX 机械转纯文本
                content = _clean_layer_surface(content, layer.name)  # 表面机械清理
                model_calls += 1
            gaps = teaching_layer_gaps(content, layer)
            if gaps:
                reason = f"表面检查未通过：{'、'.join(gaps)}"
                logger.warning("技术层%d未通过表面检查（%s），修正重试一次", i, reason)
                retry_prompt = (
                    layer_prompt()
                    + "\n\n## 本次必须修正\n"
                    + f"上一次生成未通过检查（{reason}）。重新完整输出本层。不要改写或"
                    "缩并合同；逐项偿还以下解释债务，并让闭卷检查真正可完成：\n"
                    + "\n".join(f"- {item}" for item in layer.must_explain)
                    + "\n闭卷检查：\n"
                    + "\n".join(f"- {item}" for item in layer.success_test)
                    + f"\n本层采用 {layer.teaching_mode} 讲法，禁止强行套输入输出、反事实或固定结尾。"
                )
                try:
                    content = writer_client.chat_full(
                        [{"role": "user", "content": retry_prompt}], heavy=True,
                        **writer_kwargs,
                    ).content
                except KimiError as exc:
                    # 修正重试失败：保留原版内容，走下方批注交付
                    logger.warning("技术层「%s」修正重试失败，保留原版并批注：%s",
                                   layer.name, exc)
                else:
                    content = _hoist_media_markers(content)  # 格式机械修，省一次模型重写
                    content = _strip_inline_latex(content)  # LaTeX 机械转纯文本
                    content = _clean_layer_surface(content, layer.name)  # 表面机械清理
                    model_calls += 1
                    gaps = teaching_layer_gaps(content, layer)
            if gaps:
                # 产出优先：批注该层继续，不判死
                logger.warning("技术层「%s」表面验收未过，批注继续：%s",
                               layer.name, "、".join(gaps))
                degraded.append(f"技术层「{layer.name}」")
                content = _annotate_degraded(
                    content, "结构缺口：" + "、".join(gaps))
                local_review = {"pass": True, "tier": "degraded-structure",
                                "issues": [], "degraded": True}
            if not self._local_review_passed(local_review):
                layer_contract = {
                    "kind": "technical_layer",
                    "name": layer.name,
                    "question": layer.question,
                    "why": layer.why,
                    "teaching_mode": layer.teaching_mode,
                    "depth": layer.depth,
                    "must_explain": layer.must_explain,
                    "success_test": layer.success_test,
                    "source_refs": layer.source_refs,
                }
                source = (
                    evidence.source_packet(layer.source_refs, [*layer.source_hints, layer.name])
                    if evidence.long_mode else paper_md  # 与写作同一份全文，审核共享缓存前缀
                )
                if layer.depth != "core":
                    # 轻审=只审范围；事实与债务核验收归终审；core 层仍逐层模型审
                    local_review = self._review_section(
                        reviewer_client, content, layer_contract, continuity, source,
                        full_map=full_map, scope_only=True,
                    )
                    local_review["tier"] = "light"
                    model_calls += 1
                else:
                    local_review = self._review_section(
                        reviewer_client, content, layer_contract, continuity, source,
                        full_map=full_map,
                    )
                    model_calls += 1
                for _rewrite in range(LOCAL_REVIEW_MAX_REWRITES):
                    if self._local_review_passed(local_review):
                        break
                    correction = self._review_issues(local_review)
                    # 结构重复禁入补丁：整章需按全局归属图重审
                    struct_dup = isinstance(local_review, dict) and any(
                        "结构重复：" in str(it.get("problem") or "")
                        for it in (local_review.get("issues") or []) if isinstance(it, dict)
                    )
                    patch_edits: list[dict] | None = None
                    pre_patch: tuple[str, dict[str, Any]] | None = None
                    if _rewrite == 0 and not struct_dup:
                        patched_result = self._try_patch(writer_client, content, correction, source)
                        if patched_result is not None:
                            pre_patch = (content, local_review)
                            content, patch_edits = patched_result
                            model_calls += 1
                    if patch_edits is None:
                        try:
                            content = writer_client.chat_full(
                                [{"role": "user", "content": (
                                    layer_prompt() + "\n\n## 局部事实与教学审核要求重写\n" + correction
                                    + "\n重新完整输出本层，所有事实必须由原文包直接支持。"
                                )}],
                                heavy=True, **writer_kwargs,
                            ).content
                        except KimiError as exc:
                            # Writer 重写失败：保留上一版正文，出循环走批注交付
                            logger.warning("技术层「%s」重写失败，保留上一版并批注：%s",
                                           layer.name, exc)
                            break
                        content = _hoist_media_markers(content)  # 格式机械修，省一次模型重写
                        content = _strip_inline_latex(content)  # LaTeX 机械转纯文本
                        content = _clean_layer_surface(content, layer.name)  # 表面机械清理
                        model_calls += 1
                    gaps = teaching_layer_gaps(content, layer)
                    if gaps:
                        logger.warning("技术层「%s」重写后结构缺口，批注继续：%s",
                                       layer.name, "、".join(gaps))
                        degraded.append(f"技术层「{layer.name}」")
                        content = _annotate_degraded(
                            content, "结构缺口：" + "、".join(gaps))
                    if patch_edits is not None:
                        local_review = self._patch_review(
                            reviewer_client, patch_edits, content, correction, source,
                        )
                        if (isinstance(local_review, dict)
                                and local_review.get("revert_patch")
                                and pre_patch is not None):
                            # 补丁复核不可用：放弃未验证补丁，保留补丁前正文并标注
                            # （重写 0 次）；原审核意见随下方批注横幅交付
                            content, local_review = pre_patch
                            local_review = {**local_review, "issues": [
                                *(local_review.get("issues") or []),
                                {"severity": "中",
                                 "problem": "补丁复核不可用，已保留补丁前正文"},
                            ]}
                            model_calls += 1
                            break
                    else:
                        local_review = self._review_section(
                            reviewer_client, content, layer_contract, continuity, source,
                            full_map=full_map,
                        )
                    model_calls += 1
                if not self._local_review_passed(local_review):
                    brief = self._review_issues(local_review if isinstance(local_review, dict) else {})
                    logger.warning("技术层「%s」三轮审核未过，保留全文加批注：%s", layer.name, brief[:120])
                    degraded.append(f"技术层「{layer.name}」")
                    content = _annotate_degraded(content, brief)
                    local_review = {**(local_review if isinstance(local_review, dict) else {}),
                                    "degraded": True, "degrade_mode": "annotated", "pass": False}
                if workspace:
                    workspace.put_text(layer_key, content)
                    if isinstance(local_review, dict):
                        # 产物签名随审核落盘：workflow 指纹 + unit semantic_revision
                        local_review = {**local_review, "_artifact_sig": self._artifact_sig(
                            layer_key, plan.units[layer_uid].semantic_revision)}
                    workspace.put_json(layer_review_key, local_review)
                    workspace.invalidate_from((f"rework-{layer_key}",))  # 返工意见一次有效
                    layer_deps = {f"plan:{layer_uid}":
                                  plan.units[layer_uid].semantic_revision,
                                  **_unit_binding_deps(plan.units[layer_uid])}
                    if i > 1:
                        prev_layer_uid = plan.order["layers"][i - 2]
                        layer_deps[f"body-{prev_layer_uid}"] = 1
                    manifest.register(layer_key, layer_deps)
                    workspace.put_json("dependency-manifest",
                                       {"inputs": manifest._inputs})
            if isinstance(local_review, dict) and not local_review.get("degraded"):
                self._merge_continuity(continuity, local_review)
            section_body = re.sub(r"^###\s*[^\n]+\n+", "", content.strip(), count=1)
            section = f"### {layer.name}\n\n{section_body}"
            layer_sections.append(section)
            timer.mark(f"层{i}")
        layers_md = "\n\n".join(layer_sections)
        timer.mark("精讲")
        progress(
            f"✅ 精讲完成（{timer.duration('精讲开始', '精讲'):.0f}s，"
            f"{total_layers} 层已通过验收），开始图表讲解"
        )

        # ---- 调用 N+2..：全部图表按原文顺序逐张讲解；视觉不可用时使用文本证据 ----
        figure_notes: dict[str, str] = {}
        figure_inlines: dict[str, str] = {}
        figure_contexts = {
            asset.label: self._figure_context(paper_md, asset, 1200)
            for asset in figures
        }
        figure_purposes = {
            asset.label: classify_figure_purpose(asset, figure_contexts[asset.label])
            for asset in figures
        }
        figure_source_items = {
            asset.label: estimate_table_item_count(figure_contexts[asset.label])
            for asset in figures if asset.is_table
        }

        def note_gaps(asset: FigureAsset, notes: dict[str, str]) -> list[str]:
            note = notes.get(asset.label)
            if note is None:
                return ["完整输出标签"]
            return figure_explanation_gaps(
                note, figure_purposes[asset.label], figure_source_items.get(asset.label),
            )

        if figures:
            vision_ok, probe_calls = self.probe_vision(vision_client)
            model_calls += probe_calls
        else:
            vision_ok = False
        if figures:
            include_images = vision_ok

            def generate_one(asset: FigureAsset) -> tuple[str, str, str, int, str]:
                calls = 0
                local_notes: dict[str, str] = {}
                local_inlines: dict[str, str] = {}
                try:
                    cache_key = f"figure-{asset.label}"
                    cached = workspace.get_json(cache_key) if workspace else None
                    if isinstance(cached, dict):
                        note = str(cached.get("note") or "")
                        inline = str(cached.get("inline") or "")
                        if not note_gaps(asset, {asset.label: note}):
                            return asset.label, note, inline, 0, ""
                    calls += 1
                    rework_note = (
                        workspace.get_text(f"rework-{cache_key}") if workspace else None
                    )
                    raw = vision_client.chat(
                        self.build_key_figures_messages(
                            [asset], paper_md,
                            overview_md or self._plan_text(skeleton, focus_layer="__figures__"),
                            correction=rework_note or "",
                            include_images=include_images,
                        ),
                        heavy=True, json_mode=True, max_tokens=FIGURE_OUTPUT_TOKENS,
                    )
                    local_notes, local_inlines = parse_structured_figure_notes(
                        raw, [asset], figure_purposes,
                    )
                    gaps = note_gaps(asset, local_notes)
                    if gaps:
                        details = (
                            f"- {asset.label}（{FIGURE_PURPOSE_NAMES[figure_purposes[asset.label]]}）"
                            f"缺少：{'、'.join(gaps)}"
                        )
                        logger.warning("图表讲解未通过分类验收，局部重试：%s", details)
                        calls += 1
                        retry_raw = vision_client.chat(
                            self.build_key_figures_messages(
                                [asset], paper_md,
                                overview_md or self._plan_text(skeleton, focus_layer="__figures__"),
                                correction=details, include_images=include_images,
                            ),
                            heavy=True, json_mode=True, max_tokens=FIGURE_OUTPUT_TOKENS,
                        )
                        local_notes, local_inlines = parse_structured_figure_notes(
                            retry_raw, [asset], figure_purposes,
                        )
                    if workspace and not note_gaps(asset, local_notes):
                        workspace.invalidate_from((f"rework-{cache_key}",))
                        workspace.put_json(cache_key, {
                            "label": asset.label,
                            "note": local_notes.get(asset.label, ""),
                            "inline": local_inlines.get(asset.label, ""),
                        })
                    return (
                        asset.label,
                        local_notes.get(asset.label, ""),
                        local_inlines.get(asset.label, ""),
                        calls,
                        "",
                    )
                except (KimiError, ReadingError) as exc:
                    return asset.label, "", "", calls, str(exc)

            # 真正的 KimiClient 才开启并行；测试 fake 和其他注入实现保持确定性串行。
            workers = min(FIGURE_MAX_WORKERS, len(figures)) if isinstance(self.kimi, KimiClient) else 1
            completed = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(generate_one, asset): asset for asset in figures}
                for future in concurrent.futures.as_completed(futures):
                    label, note, inline, calls, error = future.result()
                    model_calls += calls
                    completed += 1
                    mode = "视觉读图" if include_images else "文本证据解释"
                    progress(f"图表讲解 {completed}/{len(figures)}（{mode}）")
                    if error:
                        logger.warning("%s 讲解失败，将透明降级并继续正文发布：%s", label, error)
                        continue
                    if note:
                        figure_notes[label] = note
                    if inline:
                        figure_inlines[label] = inline
        invalid_figures = {
            asset.label: gaps
            for asset in figures
            if (gaps := note_gaps(asset, figure_notes))
        }
        if invalid_figures:
            # 图表是可独立降级的附属产物，不能在正文已完成后把整篇精读枪毙。
            # 降级稿不猜读数、不伪装成已完成讲解，只保留原图、图注和明确状态。
            by_label = {asset.label: asset for asset in figures}
            for label, gaps in invalid_figures.items():
                asset = by_label[label]
                figure_notes[label] = (
                    "【讲解状态】自动讲解未通过局部验收；本次仅保留原图与原始图注。\n"
                    "【阅读建议】请直接查看原图，并结合图注定位它在论文中的用途；"
                    "系统未据此补写任何读数、比较或因果结论。"
                )
                figure_inlines[label] = "本图自动讲解未通过局部验收，请以原图和图注为准。"
                logger.warning("图表讲解透明降级：%s（%s，%s）", label, asset.caption[:80], "、".join(gaps))
            progress(
                f"⚠️ {len(invalid_figures)} 张图表讲解未通过局部验收；"
                "保留原图和图注，正文继续发布"
            )
        timer.mark("图表")

        # ---- 组装最终文档 ----
        body = self._assemble_final(
            metadata, skeleton, layers_md, figures, figure_notes, figure_inlines,
            overview_md=overview_md,
        )
        final_gaps = self._final_document_gaps(
            body, skeleton, figures, figure_notes, formulas, figure_purposes,
            vision_available=vision_ok or not figures,
        )
        if final_gaps:
            # 产出优先：逐项机械修复（占位符/标记清理+标题归一），剩余横幅交付
            repaired = _hoist_media_markers(_strip_inline_latex(body))
            repaired = re.sub(r"\{\{[A-Z_]+\}\}", "", repaired)  # 未替换占位符清除
            remaining = self._final_document_gaps(
                repaired, skeleton, figures, figure_notes, formulas, figure_purposes,
                vision_available=vision_ok or not figures,
            )
            if remaining:
                logger.warning("文档缺口修复后仍存（横幅交付）：%s", "；".join(remaining[:4]))
                repaired = _final_banner("；".join(remaining[:5]),
                                         title="本篇精读存在未修复的结构缺口") + "\n\n" + repaired
                metadata.setdefault("document_gaps", remaining)
            body = repaired
        review_source = evidence.review_material() if evidence.long_mode else paper_md
        cached_body = workspace.get_text("final-body") if workspace else None
        review = workspace.get_json("final-review") if workspace and cached_body == body else None
        if isinstance(review, dict) and not self._artifact_sig_matches(
                review.get("_artifact_sig"), "final-body", plan.plan_revision):
            # 产物签名不匹配（plan revision/工作流指纹已变）：终审缓存失效，重新终审
            logger.info("终审产物签名不匹配，重新终审")
            review = None
        if not isinstance(review, dict):
            review = self._review_final(
                body, review_source, formulas, skeleton, client=reviewer_client,
            )
            model_calls += 1
        passed = _review_bool(review.get("pass")) is True
        score = review.get("score")
        score_valid = (not isinstance(score, bool) and isinstance(score, (int, float))
                       and math.isfinite(float(score)) and 0 <= score <= 10)
        review_unavailable = bool(review.get("review_unavailable"))
        # 终审不可用时不做合同核对定位：空 coverage 会误报全层缺口、触发全体重写；
        # 直接横幅批注交付（正文重写 0 次）
        contract_gaps = ([] if review_unavailable
                         else self._review_contract_gaps(review, skeleton))
        struct_dup = any(
            isinstance(item, dict) and "结构重复：" in str(item.get("problem") or "")
            for item in (review.get("issues") or [])
        )
        final_ok = (passed and score_valid and float(score) >= 8
                    and not contract_gaps and not struct_dup
                    and not review_unavailable)
        rework_interrupted = False
        if not final_ok:
            issues = review.get("issues") or []
            detail = "；".join(
                f"{item.get('location', '全文')}：{item.get('problem', '')}"
                for item in issues[:5] if isinstance(item, dict)
            ) or str(review.get("summary") or "终审未通过")
            if not score_valid:
                detail = ("终审返回的评分缺失或异常（审核模型输出不完整）；"
                          "正文按程序检查与局审结果交付")
            failed_units, core_involved = _locate_failed_units(
                review if isinstance(review, dict) else {},
                skeleton, contract_gaps, {asset.label for asset in figures},
                plan=plan,
            )
            # 定点返工：能定位且还有返工轮次 → 作废失败单元检查点+留返工意见
            if failed_units and workspace is not None and rework_round < 2 and score_valid:
                # 返工前留存最后完整稿：返工中途任何失败都能回滚交付本稿，
                # 不再存在「有稿变无稿」路径
                workspace.put_text("last-good-body", body)
                for key, note in failed_units.items():
                    workspace.put_text(f"rework-{key}", note)
                    sister = key.replace("body-", "review-", 1) if key.startswith("body-") else key
                    targets = {key, sister}
                    dirty: set[str] = set()
                    if key.startswith("body-"):
                        dirty = manifest.dirty_closure({key: 0})  # rev 0 强制失配
                        targets |= dirty
                        for artifact in dirty:
                            targets.add(artifact.replace("body-", "review-", 1))
                    workspace.invalidate_from(tuple(targets) + ("final-review", "final-body"))
                    # 如实输出闭包大小：非 body- 键（如 figure-xxx）无依赖闭包，记 0
                    logger.info("精准失效：%s（含依赖闭包 %d 项）",
                                key, len(dirty))
                logger.warning(
                    "终审未过，定点返工 %d 个单元（第 %d 轮）：%s",
                    len(failed_units), rework_round + 1, "、".join(failed_units),
                )
                progress(f"🔁 终审意见返工 {len(failed_units)} 处（第 {rework_round + 1}/2 轮）……")
                try:
                    return self.run(
                        paper_md, figures, formulas, arxiv_id, meta_hint, progress,
                        workspace, rework_round=rework_round + 1,
                        calls_carry=calls_carry + model_calls,
                    )
                except Exception as exc:
                    # 产出优先：返工中断 → 交付返工前最后完整稿（强横幅标注），不上抛
                    last_good = workspace.get_text("last-good-body")
                    if not last_good:
                        raise  # 无稿可回滚才上抛（保持既有例外口径）
                    logger.warning("返工中断，交付返工前完整稿：%s", exc)
                    body = _final_banner(
                        f"终审返工第 {rework_round + 1} 轮中断（{str(exc)[:120]}）；"
                        "以下为返工前最后完整稿，请对照原论文核实",
                        title="返工中断，交付返工前完整稿") + "\n\n" + last_good
                    metadata["final_review_degraded"] = f"终审返工中断：{str(exc)[:160]}"
                    rework_interrupted = True
            if not rework_interrupted:
                # 终局一律批注交付（用户裁定：token 与时间必须有产出）；
                # core 问题用强横幅标注，结果卡同步透出失败单元与原因
                if contract_gaps and not failed_units:
                    gap_detail = "；".join(contract_gaps[:5])
                    detail = gap_detail if score_valid else f"{detail}；{gap_detail}"
                banner_title = ("本篇精读未通过发布前终审（含核心链路问题）"
                                if core_involved else "本篇精读未全部通过发布前终审")
                logger.warning("终审降级交付（%s）：%s", banner_title, detail[:200])
                body = _final_banner(detail, title=banner_title) + "\n\n" + body
                metadata["final_review_degraded"] = detail
                if core_involved:
                    metadata["final_review_core_failed"] = True
        if workspace:
            workspace.put_text("final-body", body)
            if isinstance(review, dict):
                review = {**review, "_artifact_sig": self._artifact_sig(
                    "final-body", plan.plan_revision)}
            workspace.put_json("final-review", review)
        logger.info("教学型精读完成：调用 %d 次，总耗时 %.0fs",
                    calls_carry + model_calls, timer.total())
        metadata["graph_stats"] = graph_stats
        if degraded:
            metadata["degraded_sections"] = degraded
        return ReadingResult(
            body_md=body, metadata=metadata,
            layer_names=[l.name for l in skeleton.layers],
            figure_notes=figure_notes, figure_inlines=figure_inlines,
            vision_used=vision_ok, model_calls=calls_carry + model_calls, timer=timer,
            formula_images={asset.label: asset.img_path for asset in formulas
                            if f"[[MATH:{asset.label}]]" in body},
        )

    def _review_final(self, body: str, paper_md: str,
                      formulas: list[FormulaAsset] | None = None,
                      skeleton: Skeleton | None = None,
                      *, client: Any | None = None) -> dict[str, Any]:
        contract = {
            "paper_type": skeleton.paper_type,
            "layers": [
                {
                    "name": layer.name,
                    "question": layer.question,
                    "depth": layer.depth,
                    "teaching_mode": layer.teaching_mode,
                    "must_explain": layer.must_explain,
                    "success_test": layer.success_test,
                }
                for layer in skeleton.layers
            ],
        } if skeleton else {"layers": []}
        prompt = _render(
            _load(self.prompts_dir / "reading_final_review.md"),
            READING_CONTENT=body,
            PAPER_CONTENT=paper_md,
            FORMULA_EVIDENCE=formula_inventory_text(formulas or []),
            EXPLANATION_CONTRACT=json.dumps(contract, ensure_ascii=False, indent=2),
        )
        try:
            raw = (client or self.kimi).chat(
                [{"role": "user", "content": prompt}], heavy=True,
                json_mode=True, low_reasoning=True,
                max_tokens=FINAL_REVIEW_OUTPUT_TOKENS,
            )
        except KimiError as exc:
            # 产出优先：终审服务不可用不杀正文——review_unavailable 横幅交付
            logger.warning("终审调用失败，标记 review_unavailable 保留正文：%s", exc)
            return {"pass": True, "score": 8, "review_unavailable": True,
                    "summary": "终审服务不可用，未经终审复核",
                    "issues": [], "layer_coverage": [], "teach_back": []}
        try:
            review = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("终审返回非法 JSON，标记 review_unavailable 保留正文")
            return {"pass": True, "score": 8, "review_unavailable": True,
                    "summary": "终审服务输出异常，未经终审复核",
                    "issues": [], "layer_coverage": [], "teach_back": []}
        if not isinstance(review, dict):
            return {"pass": True, "score": 8, "review_unavailable": True,
                    "summary": "终审服务输出异常，未经终审复核",
                    "issues": [], "layer_coverage": [], "teach_back": []}
        return review

    @staticmethod
    def _review_contract_gaps(review: dict[str, Any], skeleton: Skeleton) -> list[str]:
        """用程序核对终审是否逐项审过合同，避免模型一句“已覆盖”放行。"""
        def key(value: Any) -> str:
            return re.sub(r"[^\w一-鿿]+", "", str(value or "")).casefold()

        def substantive(value: Any) -> bool:
            text = str(value or "").strip()
            return bool(text) and not re.fullmatch(
                r"(?:已|已经)?(?:覆盖|解释|说明|讲清|通过|满足|完成)(?:了|该项|要求|债务|检查)?[。.]?",
                text, re.IGNORECASE,
            )

        coverage_items = review.get("layer_coverage")
        teach_back_items = review.get("teach_back")
        if not isinstance(coverage_items, list):
            return ["终审缺少 layer_coverage"]
        if not isinstance(teach_back_items, list):
            return ["终审缺少 teach_back"]
        coverage_by_layer = {
            key(item.get("layer")): item for item in coverage_items if isinstance(item, dict)
        }
        teach_back_by_layer = {
            key(item.get("layer")): item for item in teach_back_items if isinstance(item, dict)
        }
        gaps: list[str] = []
        for layer in skeleton.layers:
            coverage = coverage_by_layer.get(key(layer.name))
            if not isinstance(coverage, dict):
                gaps.append(f"{layer.name} 缺少逐项债务审查")
                continue
            debt_by_item = {
                key(item.get("item")): item for item in coverage.get("debts", [])
                if isinstance(item, dict)
            }
            for debt in layer.must_explain:
                item = debt_by_item.get(key(debt))
                if not item:
                    gaps.append(f"{layer.name} 未审查解释债务：{debt}")
                elif _review_bool(item.get("covered")) is True:
                    if not substantive(item.get("evidence")):
                        gaps.append(f"{layer.name} 的解释债务缺少正文证据：{debt}")
                elif layer.depth == "core":
                    gaps.append(f"{layer.name} 的核心解释债务未偿还：{debt}")
                elif not str(item.get("skip_reason") or "").strip():
                    gaps.append(f"{layer.name} 跳过辅助债务但未说明理由：{debt}")
            test_by_item = {
                key(item.get("item")): item for item in coverage.get("success_tests", [])
                if isinstance(item, dict)
            }
            for test in layer.success_test:
                item = test_by_item.get(key(test))
                if not item:
                    gaps.append(f"{layer.name} 未执行闭卷检查：{test}")
                elif _review_bool(item.get("passed")) is True:
                    if not substantive(item.get("reason")):
                        gaps.append(f"{layer.name} 的闭卷检查缺少依据：{test}")
                elif layer.depth == "core":
                    gaps.append(f"{layer.name} 的核心闭卷检查未通过：{test}")
                elif not str(item.get("skip_reason") or "").strip():
                    gaps.append(f"{layer.name} 跳过辅助检查但未说明理由：{test}")
            teach_back = teach_back_by_layer.get(key(layer.name))
            if not isinstance(teach_back, dict):
                gaps.append(f"{layer.name} 缺少闭卷复述")
            elif layer.depth == "core" and not (
                _review_bool(teach_back.get("answerable")) is True
                and substantive(teach_back.get("answer"))
            ):
                gaps.append(f"{layer.name} 无法从精读闭卷复述")
            elif layer.depth == "supporting" and not (
                (_review_bool(teach_back.get("answerable")) is True
                 and str(teach_back.get("answer") or "").strip())
                or str(teach_back.get("missing") or "").strip()
            ):
                gaps.append(f"{layer.name} 的辅助层复述结论不完整")
        return gaps

    @staticmethod
    def _final_document_gaps(body: str, skeleton: Skeleton,
                             figures: list[FigureAsset],
                             figure_notes: dict[str, str],
                             formulas: list[FormulaAsset] | None = None,
                             figure_purposes: dict[str, str] | None = None,
                             *, vision_available: bool = True) -> list[str]:
        gaps: list[str] = []
        for section in skeleton.sections:
            title = str(section.get("title") or "").strip()
            if title and not _heading_present(body, 2, title):
                gaps.append(f"缺少规划章节：## {title}")
        if not _heading_present(body, 2, skeleton.deep_dive_title):
            gaps.append(f"缺少 ## {skeleton.deep_dive_title}")
        for layer in skeleton.layers:
            if not _heading_present(body, 3, layer.name):
                gaps.append(f"缺少技术层：{layer.name}")
        if skeleton.top3 and not _heading_present(body, 2, "读完后应该记住什么"):
            gaps.append("缺少 takeaways 发布章节")
        if re.search(r"\{\{[A-Z_]+\}\}", body):
            gaps.append("存在未替换模板占位符")
        unknown_internal_markers = sorted(set(re.findall(
            r"\[\[(?!FIG:|MATH:)([A-Z][A-Z0-9_-]*)(?::|\]\])", body,
        )))
        if unknown_internal_markers:
            gaps.append("存在未渲染内部标记：" + "、".join(unknown_internal_markers))
        gaps.extend(_media_marker_gaps(body))
        prose_body = re.sub(r"```.*?```", "", body, flags=re.DOTALL)
        if re.search(r"\$[^$\n]+\$|\\[a-zA-Z]{2,}\{", prose_body):
            gaps.append("存在未渲染 LaTeX")
        if re.search(r"(?:^|\s)\\(?:\s|[A-Za-z]+\b)", prose_body):
            gaps.append("存在破损公式反斜杠")
        if vision_available and re.search(
            r"基于图注解读|未提供图像|详细读数.{0,12}(?:原文|论文)|仅凭文本无法确认",
            body,
        ):
            gaps.append("存在假装完成读图的降级话术")
        leaked_layer_phrases = [phrase for phrase in _LAYER_TEMPLATE_PHRASES if phrase in body]
        if leaked_layer_phrases:
            gaps.append("存在教案模板泄漏：" + "、".join(leaked_layer_phrases))
        if re.search(
            r"确保.{0,30}(?:差异|提升).{0,20}(?:只|完全).{0,12}(?:归因|来自)|"
            r"(?:差异|提升).{0,20}(?:只|完全)归因于|唯一变量是",
            body,
        ):
            gaps.append("存在未经限定的单因果归因")
        known_labels = {asset.label for asset in figures}
        used_labels = {
            label.strip() for label in re.findall(r"\[\[FIG:([^\]]+)\]\]", body)
        }
        unknown_labels = sorted(used_labels - known_labels)
        if unknown_labels:
            gaps.append("引用了不存在的图表：" + "、".join(unknown_labels))
        formulas = formulas or []
        known_formulas = {asset.label for asset in formulas}
        used_formulas = {
            label.strip() for label in re.findall(r"\[\[MATH:([^\]]+)\]\]", body)
        }
        unknown_formulas = sorted(used_formulas - known_formulas)
        if unknown_formulas:
            gaps.append("引用了不存在的公式：" + "、".join(unknown_formulas))
        layer_keys = {_normalize_heading(layer.name) for layer in skeleton.layers}
        spans = _section_spans(body)
        layer_ranges = [
            (start, end) for start, end, level, title in spans
            if level == 3 and title in layer_keys
        ]
        for match in re.finditer(r"\[\[MATH:[^\]]+\]\]", body):
            if not any(start <= match.start() < end for start, end in layer_ranges):
                gaps.append("复杂公式只能出现在规划的核心精讲章节中")
                break
        for label in sorted(used_formulas & known_formulas):
            marker_pos = body.find(f"[[MATH:{label}]]")
            explanation = body[marker_pos + len(label) + 9:marker_pos + len(label) + 1200]
            checks = (
                bool(re.search(r"表示|含义|其中|符号", explanation)),
                bool(re.search(r"案例|代入|例如|假设", explanation)),
                bool(re.search(r"说明|意味着|作用|用于", explanation)),
            )
            if not all(checks):
                gaps.append(f"公式 {label} 缺少符号、案例或作用解释")
        h3_titles = {title for _, _, level, title in spans if level == 3}
        for asset in figures:
            if _normalize_heading(asset.label) not in h3_titles:
                gaps.append(f"图表总览缺少 {asset.label}")
            note = figure_notes.get(asset.label, "")
            purpose = (figure_purposes or {}).get(
                asset.label, classify_figure_purpose(asset),
            )
            note_gaps = figure_explanation_gaps(note, purpose) if note else ["讲解缺失"]
            if note_gaps:
                gaps.append(f"{asset.label} 缺少：{'、'.join(note_gaps)}")
        return gaps

    # ---- 最终组装 ----

    def _assemble_final(self, metadata: dict, skeleton: Skeleton, layers_md: str,
                        figures: list[FigureAsset], figure_notes: dict[str, str],
                        figure_inlines: dict[str, str], *,
                        overview_md: str | None = None) -> str:
        title = metadata.get("title_zh") or metadata.get("title_en") or "未命名论文"
        parts = [f"# 《{title}》精读笔记\n"]
        parts.append(self._info_card(metadata))
        parts.append(strip_h1(skeleton.markdown if overview_md is None else overview_md))
        parts.append(strip_h1(f"## {skeleton.deep_dive_title}\n\n" + layers_md.strip()))
        takeaways = [item.strip(" -*•\t") for item in skeleton.top3.splitlines() if item.strip()]
        if takeaways:
            parts.append(
                "## 读完后应该记住什么\n\n"
                + "\n".join(f"- {item}" for item in takeaways)
            )
        body = "\n\n".join(p.strip() for p in parts if p and p.strip())
        body = self._insert_inline_notes(body, figure_inlines)
        if figures:
            body += "\n\n" + self._figure_gallery(figures, figure_notes)
        return body + "\n"

    @staticmethod
    def _insert_inline_notes(body: str, figure_inlines: dict[str, str]) -> str:
        def repl(m: re.Match) -> str:
            label = m.group(1).strip()
            inline = figure_inlines.get(label)
            if not inline:
                return m.group(0)
            return f"[[FIG:{label}]]\n\n**这张图怎么读**：{inline}"
        return re.sub(r"\[\[FIG:([^\]]+)\]\]", repl, body)

    @staticmethod
    def _info_card(metadata: dict) -> str:
        def cell(value: Any) -> str:
            return str(value).replace("\n", " ").replace("|", "\\|").strip()

        rows = [
            ("题目（中文）", metadata.get("title_zh", "")),
            ("Title", metadata.get("title_en", "")),
            ("作者", metadata.get("authors", "")),
            ("机构", metadata.get("affiliation", "")),
            ("Venue", metadata.get("venue", "")),
            ("年份", str(metadata.get("year", ""))),
        ]
        lines = ["## 一、论文信息卡", "", "| 字段 | 内容 |", "|---|---|"]
        lines += [f"| {cell(k)} | {cell(v)} |" for k, v in rows if v]
        return "\n".join(lines)

    @staticmethod
    def _figure_gallery(figures: list[FigureAsset], figure_notes: dict[str, str]) -> str:
        parts = ["## 图表总览", "", "本文全部图表逐一展示与讲解（正文相应位置亦有内嵌辅助阅读）："]
        for asset in figures:
            note = figure_notes[asset.label]
            parts.append(
                f"### {asset.label}\n\n"
                f"[[FIG:{asset.label}]]\n\n"
                f"> 原始图注：{plain_figure_caption(asset.caption)}\n\n"
                f"**怎么读**：{note}"
            )
        return "\n\n".join(parts)


def _load(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReadingError(f"prompt 模板缺失：{path}（{exc}）")


def _render(template: str, **values: str) -> str:
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    return template
