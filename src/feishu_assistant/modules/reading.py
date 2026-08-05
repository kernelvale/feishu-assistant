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
from typing import Any, Callable, Iterable

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
PLAN_COMPILE_PATH = PROMPTS_DIR / "reading_plan_compile.md"  # PlanCompiler 规划协议（V8 第 3 步）

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
# 正文修复预算（V8 第 6 步收敛）：block 补丁一次 + 整单元重写一次；仍失败 →
# last_valid 恢复快照 + 批注交付（标准不变，机会收敛，产出优先）
LOCAL_REVIEW_MAX_REWRITES = 1
# plan/graph 类回流全局上限：每篇 ≤2 次（每单元每轮至多一次，同动作签名熔断）
PLAN_REFLUX_MAX = 2
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

    gaps.extend(_review_checks_evidence_gaps(valid, chunk_ids, source))
    return gaps


def _review_checks_evidence_gaps(
    valid: list[dict[str, Any]],
    chunk_ids: set[str] | None,
    source: str,
) -> list[str]:
    """checks 举证追溯门禁（骨架/Plan 审核共用）：evidence 必须可定位——
    长文引用真实 chunk id；短文含可定位锚点（原文摘录/章节标题/图表标签），
    引用不存在的章/节/图/chunk 或空泛指引即拒收。"""
    gaps: list[str] = []
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


def _plan_boundary_pair(check: dict[str, Any]) -> tuple[str, str] | None:
    """boundary check 的结构化层对：left/right 都非空才有效。"""
    left = str(check.get("left_layer_id") or "").strip()
    right = str(check.get("right_layer_id") or "").strip()
    return (left, right) if left and right else None


def _plan_boundary_ref_gaps(
    check: dict[str, Any],
    graph: Any | None,
    registry: Any | None,
) -> list[str]:
    """boundary 判定的 lost_* 引用真实性：reviewer 可确认/修正程序预填，
    但给出的债务/断言/span ID 必须真实存在（注册表降级时债务项只校验形状）。"""
    gaps: list[str] = []
    label = f"{check.get('left_layer_id')}→{check.get('right_layer_id')}"
    debts = getattr(registry, "debts", None) or {}
    registry_usable = bool(debts) and getattr(registry, "admission_status", "") != "degraded"
    raw_debts = check.get("lost_debt_ids")
    if raw_debts is not None:
        if not isinstance(raw_debts, list):
            gaps.append(f"boundary {label} 的 lost_debt_ids 必须是数组")
        elif registry_usable:
            unknown = [str(d) for d in raw_debts if str(d) not in debts]
            if unknown:
                gaps.append(f"boundary {label} 的 lost_debt_ids 引用不存在债务："
                            + "、".join(unknown[:3]))
    if graph is not None:
        raw_assertions = check.get("lost_assertion_ids")
        if raw_assertions is not None:
            if not isinstance(raw_assertions, list):
                gaps.append(f"boundary {label} 的 lost_assertion_ids 必须是数组")
            else:
                unknown = [str(a) for a in raw_assertions if str(a) not in graph.nodes]
                if unknown:
                    gaps.append(f"boundary {label} 的 lost_assertion_ids 引用"
                                "不存在节点：" + "、".join(unknown[:3]))
        raw_spans = check.get("evidence_span_ids")
        if raw_spans is not None:
            if not isinstance(raw_spans, list):
                gaps.append(f"boundary {label} 的 evidence_span_ids 必须是数组")
            else:
                unknown = [str(s) for s in raw_spans if str(s) not in graph.spans]
                if unknown:
                    gaps.append(f"boundary {label} 的 evidence_span_ids 引用"
                                "不存在 span：" + "、".join(unknown[:3]))
    return gaps


def _plan_review_check_gaps(
    checks: Any,
    plan: "ReadingPlan",
    source: str,
    graph: Any | None = None,
    registry: Any | None = None,
) -> list[str]:
    """PlanAuditor 准入 checks 门禁（程序级，fail-closed 但出口不判死）：

    覆盖（V8 第 5 步 LayerContract 时代协议）：
    - 逐 boundary 一条 outcome=pass 判定（结构化 left/right unit_id；
      汇总冒充逐条拒收；fail/unknown 不得 clean）；lost_* 引用必须真实；
    - 单层 Plan（无 boundary）必须有一条「核心主张」准入检查且 pass；
    - 债务归属冲突 / preparations 分配 / purpose·mode 举证各一条且 pass；
    - 每条 check：outcome 合法、evidence 可定位、target_ids 指向真实单元。
    不合格不判死：由调用方 correction 重试一次，仍不合格标注
    「语义准入审核不完整」继续。"""
    from .reading_plan import BOUNDARY_OUTCOMES

    if not isinstance(checks, list) or not checks:
        return ["Plan 审核缺少 checks 举证明细"]
    valid = [c for c in checks if isinstance(c, dict)]
    if len(valid) != len(checks):
        return ["checks 含非对象项"]
    gaps: list[str] = []
    layer_uids = [uid for uid in (plan.order.get("layers") or []) if uid in plan.units]
    valid_targets: set[str] = set(plan.units)
    for uid, unit in plan.units.items():
        for key in ("name", "title"):
            text = str(unit.data.get(key) or "").strip()
            if text:
                valid_targets.add(text)

    # ---- 逐 boundary：结构化层对匹配（左右互换宽容），汇总冒充拒收 ----
    boundary_checks: list[tuple[dict[str, Any], tuple[str, str]]] = []
    for check in valid:
        pair = _plan_boundary_pair(check)
        if pair is not None:
            boundary_checks.append((check, pair))
        elif "boundary" in str(check.get("item") or "").lower() \
                or "边界" in str(check.get("item") or ""):
            gaps.append("boundary 检查必须逐条给出 left_layer_id/right_layer_id"
                        "（汇总判定拒收）")
    real_pairs = {(str(b.get("left_layer_id")), str(b.get("right_layer_id")))
                  for b in plan.boundaries}
    unmatched = [f"{left}→{right}" for _c, (left, right) in boundary_checks
                 if (left, right) not in real_pairs
                 and (right, left) not in real_pairs]
    if unmatched:
        gaps.append("boundary check 指向不存在的相邻层对：" + "、".join(unmatched[:3]))
    for b in plan.boundaries:
        left, right = str(b.get("left_layer_id")), str(b.get("right_layer_id"))
        hit = next((c for c, pair in boundary_checks
                    if pair == (left, right) or pair == (right, left)), None)
        if hit is None:
            gaps.append(f"checks 缺少 boundary 逐条判定：{left}→{right}")
            continue
        outcome = str(hit.get("outcome") or "").strip()
        if outcome != "pass":
            gaps.append(f"boundary {left}→{right} outcome={outcome or '缺失'}，"
                        "clean 要求逐条 pass（fail/unknown 不得 clean）")
        gaps.extend(_plan_boundary_ref_gaps(hit, graph, registry))

    # ---- 单层 Plan：核心主张准入检查 ----
    if not plan.boundaries and len(layer_uids) == 1:
        hit = next((c for c in valid if "核心主张" in str(c.get("item") or "")), None)
        if hit is None:
            gaps.append("单层 Plan checks 缺少「单层核心主张」准入检查")
        elif str(hit.get("outcome") or "").strip() != "pass":
            gaps.append("单层核心主张准入检查 outcome 不是 pass，不得 clean")

    # ---- 债务归属 / preparations / purpose·mode：各一条且 pass ----
    for tokens, label in (
        (("债务",), "债务归属冲突"),
        (("preparations", "前置概念", "建立义务"), "preparations 分配"),
        (("purpose", "mode", "功能举证", "模式举证"), "purpose/mode 举证"),
    ):
        hit = next((c for c in valid
                    if any(t in str(c.get("item") or "") for t in tokens)), None)
        if hit is None:
            gaps.append(f"checks 未覆盖{label}")
        elif str(hit.get("outcome") or "").strip() != "pass":
            gaps.append(f"{label}检查 outcome 不是 pass，不得 clean")

    # ---- 每条 check：outcome 合法 + target_ids 真实 + evidence 可定位 ----
    for check in valid:
        label = str(check.get("item") or "").strip()[:30]
        outcome = str(check.get("outcome") or "").strip()
        if outcome not in BOUNDARY_OUTCOMES:
            gaps.append(f"check「{label}」outcome 非法或缺失：{outcome or '空'}")
        targets = [str(t).strip() for t in (check.get("target_ids") or [])
                   if str(t).strip()]
        if not targets:
            gaps.append(f"check「{label}」缺 target_ids")
        elif any(t not in valid_targets for t in targets):
            gaps.append(f"check「{label}」target_ids 指向不存在单元")
    chunk_ids = set(re.findall(r"chunk-\d{3,4}", source)) or None
    gaps.extend(_review_checks_evidence_gaps(valid, chunk_ids, source))
    return gaps


def _plan_merge_boundary_outcomes(plan: "ReadingPlan", checks: Any) -> int:
    """把 reviewer 的逐 boundary 判定写回 plan.boundaries（outcome/reason/lost_*）。

    只在准入门禁通过后调用（引用真实性已校验）；reviewer 未给的 lost_* 字段
    保留程序预填值。返回写回条数。"""
    from .reading_plan import BOUNDARY_OUTCOMES

    if not isinstance(checks, list):
        return 0
    by_pair: dict[frozenset[str], dict[str, Any]] = {}
    for check in checks:
        if not isinstance(check, dict):
            continue
        pair = _plan_boundary_pair(check)
        if pair is not None:
            by_pair[frozenset(pair)] = check
    merged = 0
    for boundary in plan.boundaries:
        check = by_pair.get(frozenset((str(boundary.get("left_layer_id")),
                                       str(boundary.get("right_layer_id")))))
        if check is None:
            continue
        outcome = str(check.get("outcome") or "").strip()
        if outcome not in BOUNDARY_OUTCOMES:
            continue  # 防御：门禁外的非法判定不写回
        boundary["outcome"] = outcome
        boundary["reason"] = str(check.get("reason") or "").strip()
        for field_name in ("lost_debt_ids", "lost_assertion_ids", "evidence_span_ids"):
            raw = check.get(field_name)
            if isinstance(raw, list):
                boundary[field_name] = [str(v) for v in raw]
        merged += 1
    return merged


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
        # 债务文本后附〔debt_id〕：审核器三分流判定与 debt_ids 引用的事实源
        debt_ids = list(getattr(layer, "debt_ids", None) or [])
        debt_items = [
            f"{action}〔{debt_ids[d]}〕" if d < len(debt_ids) else action
            for d, action in enumerate(layer.must_explain)
        ]
        lines.append(
            f"层{i}《{layer.name}》（{layer.depth}）问题：{layer.question}；"
            f"债务：{'、'.join(debt_items)}"
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


class _PlanChecksIncomplete(Exception):
    """Plan 审核 checks 未过准入门禁：走一次格式纠正，仍不合格标注继续。"""


# PlanAuditor 预算（V8 第 5 步裁定）：一次格式纠正 + 最多两轮语义修复
PLAN_MAX_REPAIR_ROUNDS = 2


def _plan_action_sig(action: Any) -> str:
    """修复动作签名（op+目标+参数）：重复动作熔断的对账凭据。"""
    return "|".join([
        str(action.op),
        ",".join(sorted(str(t) for t in (action.target_ids or []))),
        json.dumps(action.params or {}, ensure_ascii=False, sort_keys=True,
                   default=str),
    ])


def _plan_review_issue_sig(review: dict[str, Any]) -> tuple[frozenset[str], int]:
    """本轮审核 issue 身份集合与最高严重度（修复进展三选一判定的对账凭据）。"""
    from .reading_debts import SEVERITY_RANK
    from .reading_issues import ReviewIssue
    ids: list[str] = []
    max_sev = 0
    for item in (review.get("issues") or []) if isinstance(review, dict) else []:
        if not isinstance(item, dict):
            continue
        try:
            issue = ReviewIssue.from_legacy(item, "plan")
        except ValueError:
            continue
        ids.append(issue.issue_id)
        max_sev = max(max_sev, SEVERITY_RANK.get(issue.severity, 0))
    return frozenset(ids), max_sev


# merge/split 语义的 plan 类 issue kind（第 7 步：优先 replace_plan_fragment 事务）
FRAGMENT_REFLUX_KINDS = frozenset(
    {"contract_gap", "debt_uncovered", "boundary_fail"})


def _parse_fragment_repairs(raw_repairs: Any, plan_revision: int) -> list[Any]:
    """审核器显式给出的 replace_plan_fragment 动作解析（v1/v2 字段兼容）。

    版本号按当前 Plan 对齐（局审/终审材料不含 revision，乐观并发由引擎的
    digest 校验把关）；非 fragment 动作与解析失败项跳过，不判死。"""
    from .reading_repair import RepairAction, RepairRejected
    actions: list[Any] = []
    for item in (raw_repairs or []) if isinstance(raw_repairs, list) else []:
        if not isinstance(item, dict):
            continue
        if str(item.get("op") or "").strip() != "replace_plan_fragment":
            continue
        data = dict(item)
        if not str(data.get("issue_id") or "").strip():
            resolves = data.get("resolves_issue_ids") or []
            data["issue_id"] = str(resolves[0]) if resolves else "reflux-fragment"
        data["expected_plan_revision"] = plan_revision
        data["expected_revisions"] = {}
        try:
            actions.append(RepairAction.from_dict(data))
        except RepairRejected:
            continue
    return actions


def _mechanical_fragment_params(replan_issues: list[Any],
                                replan_targets: list[str],
                                plan: Any, debts: Any) -> dict[str, Any]:
    """merge/split 语义的机械翻译（无模型）：吸收到首个 target，其余退休；
    issue 点名且无幸存 owner 的债务经 debt_changes 显式指派给继承层。
    合法性由 RepairEngine 事务内的 validate_plan 把关，失败退 _local_replan。"""
    keep = replan_targets[0]
    retiring = set(replan_targets[1:])
    params: dict[str, Any] = {
        "target_ids": list(replan_targets),
        "keeps": [keep],
        "retires": [{"unit_id": t, "replaced_by": [keep]}
                    for t in replan_targets[1:]],
    }
    keep_debts = set(plan.units[keep].data.get("debt_ids") or [])
    changes: list[dict[str, str]] = []
    for issue in replan_issues:
        for did in issue.debt_ids:
            if did not in debts or did in keep_debts:
                continue
            surviving_owners = [
                uid for uid in (plan.order.get("layers") or [])
                if uid in plan.units and uid not in retiring
                and did in (plan.units[uid].data.get("debt_ids") or [])]
            if surviving_owners:
                continue
            changes.append({"debt_id": did, "owner_layer_id": keep})
            keep_debts.add(did)
    if changes:
        params["debt_changes"] = changes
    return params


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
    *,
    block_index: dict[str, str] | None = None,
    debt_owner: dict[str, str] | None = None,
) -> tuple[dict[str, str], bool]:
    """终审失败定位：把失败原因映射到具体 层/节/图 的检查点键，返回
    （{检查点键: 返工意见}, 是否涉及 core 层核心链路）。无法定位时返回空映射
    （调用方按 core 判死——修不了的问题不许交付）。
    第 6 步：issues 优先用 body_block_ids → unit（block manifest 反查）与
    debt_ids → owner（Plan 归属反查）结构化定位；展示名模糊匹配降为兼容兜底。"""
    units: dict[str, list[str]] = {}
    core_involved = False

    def mark(key: str, note: str, *, core: bool = False) -> None:
        nonlocal core_involved
        units.setdefault(key, []).append(note)
        core_involved = core_involved or core

    def _uid_core(uid: str) -> bool:
        if plan is None:
            return False
        unit = plan.units.get(uid)
        return bool(unit is not None and unit.data.get("depth") == "core")

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

    # ② issues[]：优先 block_id/debt_id 结构化定位；展示名模糊匹配兜底
    for item in review.get("issues") or []:
        if not isinstance(item, dict):
            continue
        location = _normalize_heading(item.get("location"))
        problem = str(item.get("problem") or "").strip()
        if not problem:
            continue
        severity = str(item.get("severity") or "高").strip()
        placed = False
        # block_id → unit（block manifest 反查，精确）
        for bid in item.get("body_block_ids") or []:
            uid = (block_index or {}).get(str(bid).strip())
            if uid:
                mark(f"body-{uid}", f"[{severity}] {problem}", core=_uid_core(uid))
                placed = True
        # debt_id → owner 层（Plan 归属反查，精确）
        if not placed:
            for did in item.get("debt_ids") or []:
                uid = (debt_owner or {}).get(str(did).strip())
                if uid:
                    mark(f"body-{uid}", f"[{severity}] {problem}",
                         core=_uid_core(uid))
                    placed = True
        if placed:
            continue
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
    # V8 第 3 步：PlanCompiler 直出的 graph-ID / debt-ID（程序校验真实性后绑定）
    primary_assertion_id: str = ""
    debt_ids: list[str] = field(default_factory=list)
    supporting_node_ids: list[str] = field(default_factory=list)
    prerequisite_concept_ids: list[str] = field(default_factory=list)
    source_span_ids: list[str] = field(default_factory=list)


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
    # 全文覆盖义务声明：orientation/evidence/boundary 由哪节以何方式承担
    coverage: dict[str, str] = field(default_factory=dict)


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
    """PlanCompiler 输出 → (Skeleton, metadata)；调用方应先通过 skeleton_output_gaps 验收。

    第 3 步协议：层直出 graph-ID/debt-ID（primary_assertion_id、debt_ids、
    supporting_node_ids、prerequisite_concept_ids、source_span_ids）；
    must_explain/success_test/break_if_merged 不再由 Planner 输出（出现则忽略），
    由程序从债务注册表派生/预计算。"""
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
            primary_assertion_id=str(l.get("primary_assertion_id") or "").strip(),
            debt_ids=[str(v).strip() for v in (l.get("debt_ids") or []) if str(v).strip()],
            supporting_node_ids=[str(v).strip() for v in (l.get("supporting_node_ids") or [])
                                 if str(v).strip()],
            prerequisite_concept_ids=[str(v).strip()
                                      for v in (l.get("prerequisite_concept_ids") or [])
                                      if str(v).strip()],
            source_span_ids=[str(v).strip() for v in (l.get("source_span_ids") or [])
                             if str(v).strip()],
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

    sections: list[dict[str, Any]] = []
    for item in (data.get("sections") or []):
        if not isinstance(item, dict) or not str(item.get("title") or "").strip():
            continue
        raw_purposes = item.get("purposes")
        if isinstance(raw_purposes, list):
            purposes = [str(v).strip() for v in raw_purposes if str(v).strip()]
        else:
            purposes = []
        if not purposes:  # 兼容旧单值 purpose
            single = str(item.get("purpose") or "").strip()
            purposes = [single] if single else []
        sections.append({
            "purposes": purposes,
            "purpose": purposes[0] if purposes else "",
            "title": str(item.get("title") or "").strip(),
            "purpose_reason": str(item.get("purpose_reason") or "").strip(),
            "preparations": [
                {"for_layer": str(p.get("for_layer") or "").strip(),
                 "concept": str(p.get("concept") or "").strip(),
                 "concept_id": str(p.get("concept_id") or "").strip()}
                for p in (item.get("preparations") or [])
                if isinstance(p, dict) and (str(p.get("concept") or "").strip()
                                            or str(p.get("concept_id") or "").strip())
            ],
            "source_refs": [
                str(v).strip() for v in (item.get("source_refs") or []) if str(v).strip()
            ][:12],
        })
    coverage = {
        str(k).strip(): str(v).strip()
        for k, v in (data.get("coverage") or {}).items()
        if str(k).strip() and str(v).strip()
    } if isinstance(data.get("coverage"), dict) else {}
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
        coverage=coverage,
    ), metadata


def _debt_registry_summary(registry: Any | None, graph: "SemanticGraph | None",
                           max_chars: int = 6_000) -> str:
    """Canonical DebtRegistry 摘要（PlanCompiler prompt 注入用）：
    debt_id | kind | subject statement | relation | learner_action（截断）。

    注册表缺失/为空/degraded 时给出明确「不可用」说明——Planner 退化为
    不用 debt_ids 的模式（各层 debt_ids 给空数组），程序侧按 degraded 标注继续。
    """
    debts = getattr(registry, "debts", None) or {}
    order = getattr(registry, "order", None) or []
    usable = bool(debts) and getattr(registry, "admission_status", "") != "degraded"
    if not usable:
        reason = ""
        issues = getattr(registry, "issues", None) or []
        if issues:
            reason = f"（{issues[0]}）"
        return (
            f"债务注册表不可用{reason}：本次规划不使用 debt_ids，"
            "所有层的 debt_ids 必须给空数组；每层的 must_explain/success_test "
            "由程序兜底派生，你只需给准 primary_assertion_id 与 question。"
        )
    lines: list[str] = []
    used = 0
    for did in order:
        debt = debts.get(did)
        if debt is None:
            continue
        subject = debt.subject_node_id
        if graph is not None and subject in graph.nodes:
            subject = graph.nodes[subject].statement
        action = re.sub(r"\s+", " ", debt.learner_action).strip()
        if len(action) > 60:
            action = action[:60] + "…"
        subject_text = re.sub(r"\s+", " ", subject).strip()
        if len(subject_text) > 60:
            subject_text = subject_text[:60] + "…"
        line = f"{did} | {debt.kind} | {subject_text} | {debt.relation} | {action}"
        if used + len(line) + 1 > max_chars:
            remaining = len(order) - len(lines)
            lines.append(f"…（{remaining} 条债务超出摘要预算已省略，不得引用）")
            break
        lines.append(line)
        used += len(line) + 1
    header = (f"共 {len(lines)} 条在册债务；debt_ids 只能引用下列真实 debt_id，"
              "每条债务只能归一层：")
    return header + "\n" + "\n".join(lines)


def skeleton_output_gaps(
    output: str,
    evidence: EvidenceBundle | None = None,
    graph: "SemanticGraph | None" = None,
    registry: Any | None = None,
) -> list[str]:
    """PlanCompiler 输出验收（第 3 步协议）：只验 JSON 教学计划；
    模型额外输出的正文既不发布，也不参与验收。

    与旧骨架协议的差异：sections 用 purposes 数组 + coverage 全文覆盖义务声明
    （不再强制 orientation/evidence/boundary 各自单独成节）；层直出 graph-ID
    （primary_assertion_id/supporting_node_ids/prerequisite_concept_ids/
    source_span_ids，程序校验真实性）与 debt_ids（只能引注册表真实债务）；
    must_explain/success_test/break_if_merged 不再由 Planner 输出（出现则忽略）。
    """
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
        for idx, section in enumerate(raw_sections, 1):
            if not isinstance(section, dict):
                gaps.append(f"第 {idx} 个章节规划不是对象")
                continue
            raw_purposes = section.get("purposes")
            purposes = ([str(v).strip() for v in raw_purposes if str(v).strip()]
                        if isinstance(raw_purposes, list) else [])
            if not purposes:  # 兼容旧单值 purpose
                single = str(section.get("purpose") or "").strip()
                purposes = [single] if single else []
            title = str(section.get("title") or "").strip()
            if not purposes or not title:
                gaps.append(f"第 {idx} 个章节规划缺少 purposes 或 title")
                continue
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
                concept_id = str(prep.get("concept_id") or "").strip()
                for_layer = str(prep.get("for_layer") or "").strip()
                if not (concept or concept_id) or not for_layer:
                    gaps.append(f"第 {idx} 个章节 preparations 项缺少 concept 或 for_layer")
                    continue
                prep_refs.append((for_layer, str(section.get("title") or "")))
                # 唯一性按 concept_id（直出身份）或归一文本（兼容）判定
                concept_key = concept_id or re.sub(
                    r"[^\w一-鿿]+", "", concept).casefold()
                owner = str(section.get("title") or "")
                if concept_key in owners and owners[concept_key] != owner:
                    gaps.append(
                        f"前置概念「{concept or concept_id}」被重复分配给多个建立章节"
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
    # 全文覆盖义务声明：orientation/evidence/boundary 必须逐项声明承担方式
    coverage = data.get("coverage")
    if not isinstance(coverage, dict):
        gaps.append("缺少 coverage 全文覆盖义务声明（orientation/evidence/boundary）")
    else:
        for required in ("orientation", "evidence", "boundary"):
            text = str(coverage.get(required) or "").strip()
            if len(text) < 8:
                gaps.append(f"coverage 缺少 {required} 的有效承担方式声明")
    deep_dive_title = str(data.get("deep_dive_title") or "").strip()
    if not deep_dive_title:
        gaps.append("缺少 deep_dive_title")
    elif isinstance(raw_sections, list) and any(
        isinstance(item, dict) and str(item.get("title") or "").strip() == deep_dive_title
        for item in raw_sections
    ):
        gaps.append("deep_dive_title 不能与骨架正文章节重复")
    # 图/注册表可用性决定 ID 检查口径（弱图/无注册表 → 对应检查跳过或收紧）
    graph_has_assertions = bool(graph is not None and any(
        n.kind == "assertion" for n in graph.nodes.values()))
    debts = getattr(registry, "debts", None) or {}
    registry_usable = bool(debts) and getattr(
        registry, "admission_status", "") != "degraded"
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
        primary_seen: dict[str, int] = {}
        debt_seen: dict[str, int] = {}
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
            mode_reason = str(layer.get("mode_reason") or "").strip()
            if not mode_reason:
                gaps.append(f"第 {idx} 层缺少 mode_reason（mode 举证）")
            # ---- 直出 ID 真实性（编造 ID 一律拒收，由 correction 修正一次）----
            if graph_has_assertions:
                primary = str(layer.get("primary_assertion_id") or "").strip()
                if not primary:
                    gaps.append(f"第 {idx} 层缺少 primary_assertion_id")
                else:
                    node = graph.nodes.get(primary)
                    if node is None:
                        gaps.append(f"第 {idx} 层 primary_assertion_id 不存在：{primary}")
                    elif node.kind != "assertion":
                        gaps.append(f"第 {idx} 层 primary 指向非 assertion 节点：{primary}")
                    elif primary in primary_seen:
                        gaps.append(f"第 {idx} 层与第 {primary_seen[primary]} 层共用同一 "
                                    f"primary assertion：{primary}")
                    else:
                        primary_seen[primary] = idx
                for id_field in ("supporting_node_ids", "prerequisite_concept_ids"):
                    value = layer.get(id_field) or []
                    if not isinstance(value, list):
                        gaps.append(f"第 {idx} 层 {id_field} 必须是数组")
                        continue
                    for nid in value:
                        if str(nid).strip() and str(nid).strip() not in graph.nodes:
                            gaps.append(f"第 {idx} 层 {id_field} 引用不存在的图节点：{nid}")
                spans = layer.get("source_span_ids") or []
                if not isinstance(spans, list):
                    gaps.append(f"第 {idx} 层 source_span_ids 必须是数组")
                else:
                    for sid in spans:
                        if str(sid).strip() and str(sid).strip() not in graph.spans:
                            gaps.append(f"第 {idx} 层 source_span_ids 引用不存在的 span：{sid}")
            raw_debt_ids = layer.get("debt_ids") or []
            debt_ids = ([str(v).strip() for v in raw_debt_ids if str(v).strip()]
                        if isinstance(raw_debt_ids, list) else [])
            if registry_usable:
                if not debt_ids:
                    gaps.append(f"第 {idx} 层缺少 debt_ids（每层至少 1 条独占债务）")
                for did in debt_ids:
                    if did not in debts:
                        gaps.append(f"第 {idx} 层引用注册表外的债务：{did}")
                    elif did in debt_seen:
                        gaps.append(f"债务 {did} 被第 {debt_seen[did]} 层与第 {idx} 层"
                                    "重复引用（一条债务只能归一层）")
                    else:
                        debt_seen[did] = idx
            elif debt_ids:
                gaps.append(f"第 {idx} 层引用了 debt_ids，但债务注册表不可用"
                            "（注册表缺失/为空/degraded 时 debt_ids 必须给空数组）")
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
    # 历史 V8 引擎继续使用 reading-v4 工作区；正式生产入口由 V9ReadingEngine 接管。
    workspace_version = PIPELINE_VERSION

    def __init__(
        self,
        kimi: KimiClient,
        prompts_dir: Path = PROMPTS_DIR,
    ) -> None:
        self.kimi = kimi
        self.prompts_dir = prompts_dir
        self._vision_ok: bool | None = None
        self._fingerprint_cache: str | None = None
        # V8 第 4 步：统一问题台账（run() 起点加载/重开；审核器出口归一入册）
        self._issue_ledger: Any | None = None
        # V8 第 6 步：plan/graph 类回流预算（run() 首跑重置，递归重跑继承）
        self._reflux_count = 0
        self._reflux_sigs: set[str] = set()
        self._graph_rebound: set[str] = set()

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
                     Path(__file__).with_name("reading_plan.py"),
                     Path(__file__).with_name("reading_blocks.py"),
                     Path(__file__).with_name("reading_continuity.py")):
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

    def build_plan_compile_prompt(
        self, paper_md: str, figures: list[FigureAsset],
        formulas: list[FormulaAsset] | None = None, *,
        graph: "SemanticGraph | None" = None,
        registry: Any | None = None,
    ) -> str:
        """PlanCompiler 规划 prompt（第 3 步）：原文包 + 图目录 + 债务注册表摘要
        （CACHE_BREAK 前共享前缀），任务协议在哨兵后。"""
        from .reading_graph import render_graph_catalog  # 延迟导入防环

        catalog = (render_graph_catalog(graph)
                   if graph is not None and (graph.nodes or graph.spans)
                   else "（语义图不可用：无 node/span 目录可引，"
                        "各 ID 字段按空数组输出）")
        return _render(
            _load(self.prompts_dir / "reading_plan_compile.md"),
            STYLE=self._style(),
            SOURCE_CONTENT=paper_md,
            GRAPH_CATALOG=catalog,
            DEBT_REGISTRY=_debt_registry_summary(registry, graph),
        ) + (
            "\n\n## 本文图表清单（配对成功，写作时按编号引用）\n\n"
            + figure_inventory_text(figures)
            + "\n\n## 本文复杂公式清单\n\n"
            + formula_inventory_text(formulas or [])
        )

    def build_layer_prompt(self, skeleton: Skeleton, layer: KeyLayer,
                           idx: int, total: int, paper_md: str,
                           figures: list[FigureAsset],
                           continuity_state: dict[str, Any] | None = None,
                           formulas: list[FormulaAsset] | None = None,
                           evidence: EvidenceBundle | None = None) -> str:
        """单层教学 prompt：带相关来源章节和前文连续状态（第 8 步起为
        ContinuityLedger 按真实依赖装包的条目，不再是前层全文）。"""
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
            CONTINUITY_CONTEXT=json.dumps(
                continuity_state or {"defined_terms": [], "established_claims": [],
                                     "open_questions": []},
                ensure_ascii=False, indent=2),
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
        *,
        context_override: str | None = None,
    ) -> dict[str, Any]:
        """补丁复核：只审改动区域（合同覆盖与连续性首轮已验、补丁外内容未变，
        不重审整节）。输出与局审同构（pass/issues），continuity 不更新。
        context_override：块级机械操作（delete/move blocks）时由调用方给出
        改动块+相邻句上下文，替代 _patched_context 的补丁锚点定位。
        复核不可用（KimiError/连续非法 JSON）口径：revert_patch=True——放弃
        未验证的补丁、保留补丁前正文、标注，重写 0 次（与 _try_patch 对齐）。"""
        edits_summary = json.dumps([
            {"find": str(e.get("find") or "")[:80],
             "replace": str(e.get("replace") or "")[:80]}
            for e in edits if isinstance(e, dict)
        ], ensure_ascii=False)
        context = context_override if context_override is not None \
            else _patched_context(patched_content, edits)
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

    def _compile_debts(
        self,
        graph: "SemanticGraph",
        source_material: str,
        client: Any,
        workspace: ReadingWorkspace | None,
    ) -> tuple["DebtRegistry", dict[str, Any]]:
        """DebtCompiler（第 2 步）：图目录 + 原文包 → 候选债务 → 程序算 ID/去重/校验。

        口径（产出优先，绝不判死）：
        - 命中 workspace 的 debt-registry.json（schema 匹配）直接复用并返回；
        - 弱图（零 assertion）→ 空 Registry + degraded 标注，不空烧模型调用；
        - KimiError / 连续非法 JSON → 空 Registry + degraded + notes；
        - validate 有 gaps（编造 ID/空字段）→ 带 correction 重试一次 →
          仍有 gaps 则剔除非法项继续（记入 registry.issues 与 stats.notes）。
        返回 (registry, stats{compile_calls, dedup_removed, notes})。"""
        from .reading_debts import DEBT_REGISTRY_SCHEMA, DebtRegistry
        from .reading_graph import render_graph_catalog

        notes: list[str] = []
        stats: dict[str, Any] = {"compile_calls": 0, "dedup_removed": 0, "notes": notes}
        registry = DebtRegistry()
        if workspace is not None:
            cached = workspace.get_json("debt-registry")
            if isinstance(cached, dict) and cached.get("schema") == DEBT_REGISTRY_SCHEMA:
                try:
                    registry = DebtRegistry.from_dict(cached)
                    logger.info("债务注册表命中检查点：%d 条债务（%s）",
                                len(registry.debts), registry.admission_status)
                    notes.append("债务注册表命中检查点，直接复用")
                    return registry, stats
                except (ValueError, TypeError, KeyError):
                    logger.warning("债务注册表检查点损坏，重新编译")

        def _persist() -> None:
            if workspace is not None:
                workspace.put_json("debt-registry", registry.to_dict())

        def _degraded_empty(reason: str) -> tuple["DebtRegistry", dict[str, Any]]:
            # 产出优先：编译不可用 → 空 Registry + degraded 批注继续，绝不判死
            registry.admission_status = "degraded"
            registry.issues.append(reason)
            notes.append(reason)
            logger.warning("债务编译降级：%s", reason)
            _persist()
            return registry, stats

        # 弱图（零 assertion）：债务池必薄，标注降级继续（现实张力裁定 2）
        if not any(n.kind == "assertion" for n in graph.nodes.values()):
            return _degraded_empty("弱图（零 assertion）：债务编译跳过，空 Registry 降级继续")

        prompt = _render(
            _load(self.prompts_dir / "reading_debt_compile.md"),
            SOURCE_CONTENT=source_material,
            GRAPH_CATALOG=render_graph_catalog(graph),
        )
        try:
            raw = client.chat(
                [{"role": "user", "content": prompt}],
                heavy=True, json_mode=True, low_reasoning=True,
                max_tokens=LOCAL_REVIEW_OUTPUT_TOKENS,
            )
            stats["compile_calls"] += 1
        except KimiError as exc:
            return _degraded_empty(f"DebtCompiler 调用失败：{str(exc)[:160]}")
        payload: dict[str, Any] | None = None
        for _attempt in range(2):
            try:
                payload = _json_object(raw)
                break
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
                    stats["compile_calls"] += 1
                except KimiError as exc:
                    return _degraded_empty(
                        f"DebtCompiler 重试调用失败：{str(exc)[:160]}")
        if payload is None:
            # 末次重试的结果同样要解析：直接丢弃会白烧一次调用还误判失败
            try:
                payload = _json_object(raw)
            except (json.JSONDecodeError, ValueError):
                return _degraded_empty("DebtCompiler 连续返回非法 JSON")

        registry, gaps = self._debts_from_payload(payload, graph)
        if gaps:
            # 编造 ID/空字段：带 correction 重试一次（唯一一次），回灌程序检出的缺口
            correction = "；".join(gaps[:8])
            logger.warning("债务编译检出 %d 项缺口（带 correction 重试一次）：%s",
                           len(gaps), correction[:200])
            try:
                raw2 = client.chat(
                    [{"role": "user", "content": (
                        prompt + "\n\n## 上一次输出存在非法债务项\n" + correction
                        + "\n请修正后重新完整输出（仍是一个 JSON 对象，"
                          "core_assertion_ids + debts）。"
                    )}],
                    heavy=True, json_mode=True, low_reasoning=True,
                    max_tokens=LOCAL_REVIEW_OUTPUT_TOKENS,
                )
                stats["compile_calls"] += 1
            except KimiError as exc:
                notes.append(
                    f"债务编译 correction 调用失败，保留首版并剔除非法项：{str(exc)[:120]}")
                self._prune_illegal_debts(registry, gaps, graph, notes)
                stats["dedup_removed"] = registry.duplicate_count
                _persist()
                return registry, stats
            try:
                payload2 = _json_object(raw2)
            except (json.JSONDecodeError, ValueError):
                payload2 = None
            if payload2 is not None:
                registry2, gaps2 = self._debts_from_payload(payload2, graph)
                if len(gaps2) < len(gaps):
                    # 重试版更优才采用；同等不折腾，保留首版
                    registry, gaps = registry2, gaps2
        if gaps:
            # 仍有缺口：剔除 validate 点名的非法项，其余保留（批注可审计）
            self._prune_illegal_debts(registry, gaps, graph, notes)
        stats["dedup_removed"] = registry.duplicate_count
        _persist()
        logger.info("债务编译完成：%d 条债务入册（去重 %d 条，调用 %d 次）",
                    len(registry.debts), registry.duplicate_count,
                    stats["compile_calls"])
        return registry, stats

    @staticmethod
    def _debts_from_payload(
        payload: dict[str, Any], graph: "SemanticGraph",
    ) -> tuple["DebtRegistry", list[str]]:
        """模型输出 → DebtRegistry：程序算 debt_id、同 ID 幂等去重、validate 收集 gaps。"""
        from .reading_debts import DebtRegistry, DebtSpec

        registry = DebtRegistry()
        items = payload.get("debts")
        if not isinstance(items, list):
            items = []
        for item in items:
            if isinstance(item, dict):
                registry.add(DebtSpec.from_payload(item))
        return registry, registry.validate(graph)

    @staticmethod
    def _prune_illegal_debts(
        registry: "DebtRegistry", gaps: list[str], graph: "SemanticGraph",
        notes: list[str],
    ) -> list[str]:
        """剔除 validate 点名的非法债务项（编造 ID/空字段），批注后继续。

        返回剔除后仍剩余的 gaps（未点名具体债务的表级缺口）。"""
        bad_ids: set[str] = set()
        for gap in gaps:
            match = re.match(r"债务 (debt-[0-9a-f]+)", gap)
            if match:
                bad_ids.add(match.group(1))
        for debt_id in sorted(bad_ids):
            if registry.remove(debt_id):
                notes.append(f"剔除非法债务 {debt_id}")
        remaining = registry.validate(graph)
        for gap in remaining:
            registry.issues.append(f"编译侧未决缺口：{gap}")
        if bad_ids:
            notes.append(f"债务编译剔除 {len(bad_ids)} 条非法项（编造 ID/空字段），"
                         "其余保留待审")
        return remaining

    def _call_skeleton(
        self,
        paper_md: str,
        figures: list[FigureAsset],
        formulas: list[FormulaAsset],
        *,
        client: Any | None = None,
        workspace: ReadingWorkspace | None = None,
        evidence: EvidenceBundle | None = None,
        graph: "SemanticGraph | None" = None,
        registry: Any | None = None,
    ) -> tuple[str, int]:
        """PlanCompiler 调用（第 3 步）：graph-ID + debt-ID 直出规划，流式低压思考 → 流式常规。

        kimi-k3-highspeed 在复杂长 prompt 上会把 max_tokens 全烧在 reasoning
        （正文为空）；reasoning_effort=low 实测 ~20s 出稿。参数不被某 provider
        支持（400）或仍失败时调整同一模型的调用参数，绝不切换模型或 API。
        网关默认 max_tokens 偏小，骨架（正文+```json 清单）经常被截断在正文
        中段、JSON 围栏整个丢失——显式抬预算；思考型模型（k3 系）reasoning 本身
        就要烧 ~8k token，8192 会挤得正文一个字都出不来，故抬到 16384。
        长输入从第一次调用就使用流式，避免客户端把完整生成时间当成
        read timeout。每次都执行完整结构验收（含直出 ID 真实性）；编造 ID →
        一次 correction；两次均不合格则抛 ReadingError，由调用方降级继续（不判死）。
        """
        client = client or self.kimi
        prompt = self.build_plan_compile_prompt(
            paper_md, figures, formulas, graph=graph, registry=registry)
        if workspace:
            cached = workspace.get_text("plan")
            if cached and not skeleton_output_gaps(
                    cached, evidence, graph=graph, registry=registry):
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
                logger.info("规划调用：%s", label)
                current_prompt = prompt
                if correction:
                    current_prompt += (
                        "\n\n## 上一次规划未通过验收\n"
                        f"{correction}\n请重新完整输出，不要省略任何必填章节和 JSON 字段；"
                        "node/span/debt ID 只能引用目录与注册表摘要中真实存在的条目。"
                    )
                out = client.chat(
                    [{"role": "user", "content": current_prompt}],
                    max_tokens=PLANNER_OUTPUT_TOKENS, **kwargs,
                )
            except KimiParameterError as exc:
                logger.warning("规划调用失败（%s）：%s", label, exc)
                last_err = exc
                continue
            except KimiError:
                # 认证、额度、限流重试耗尽等不是骨架参数问题：立即上报，
                # 避免同一坏 API 再空跑后续调用方式。
                raise
            gaps = skeleton_output_gaps(out, evidence, graph=graph, registry=registry)
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
            logger.warning("规划验收失败（%s）：%s", label, correction)
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
        debt_registry: Any | None = None,
    ) -> tuple["ReadingPlan", Any, dict[str, Any]]:
        """PlanAuditor 四状态循环（V8 第 5 步：确定性准入 + boundary 门禁）：

        - clean 程序门禁：逐 boundary outcome=pass（fail/unknown 不得 clean）、
          单层核心主张准入、债务归属/preparations/purpose·mode 各一条且 pass；
          缺项/不合格 → 一次格式纠正 → 仍不合格标注「语义准入审核不完整」继续；
          准入通过时把逐 boundary 判定写回 plan.boundaries 落盘可审计。
        - boundary fail：reviewer 按协议走 repairable（带 replace 建议）或
          unsupported_repair（需 merge/split）通道，程序不为 fail 开 clean 口子。
        - repairable：RepairEngine 有界修复，修后必重审（含格式纠正通道修复，
          统一回环）；继续条件三选一（issue 集合或最高严重度下降 / plan digest
          实际变化 / 无重复相同 action），全不满足熔断标注。
        - unsupported_repair：有界局部重规划（一次），仍不行标注继续。
        预算：一次格式纠正（JSON 非法/协议违反/checks 不完整共用）+ 最多两轮
        语义修复。Reviewer 输出错误只重试 Reviewer，绝不重跑 Planner；
        审核侧任何故障口径统一：标注 + 返回当前 Plan 继续，绝不判死。"""
        from .reading_issues import validate_review_outcome
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

        ledger = self._ledger()

        def _register_plan_issues(review: dict[str, Any]) -> None:
            # 第 4 步：reviewer 输出归一进 IssueLedger（幂等去重，落盘可审计）
            self._ledger_register_review(review, stage="plan", workspace=workspace)

        def _open_plan_ids() -> list[str]:
            return [issue.issue_id for issue in ledger.open_issues(stage="plan")]

        def _annotate_and_return(marker: str) -> tuple["ReadingPlan", Any, dict[str, Any]]:
            # 批注交付路径：遗留 open 的 Plan issue 带原因降级（禁静默丢失）
            self._ledger_degrade(_open_plan_ids(), marker, workspace)
            plan.meta.setdefault("issues", []).append(marker)
            return plan, *plan.to_skeleton()

        def _accept_clean() -> tuple["ReadingPlan", Any, dict[str, Any]]:
            # clean 放行：此前 repairable/unsupported 轮次的 open issue 视为已偿还
            self._ledger_resolve(_open_plan_ids(), "plan-review:clean", workspace)
            return plan, *plan.to_skeleton()

        def _review_unavailable_return(reason: str) -> tuple["ReadingPlan", Any, dict[str, Any]]:
            # review_unavailable：无修复路由（route 为 None），入册后走标注交付
            self._ledger_register_review(
                {"tier": "review_unavailable",
                 "issues": [{"severity": "中",
                             "problem": "Plan 审核服务不可用，未经完整审核"}]},
                stage="plan", workspace=workspace)
            return _annotate_and_return("Plan 审核不可用 review_unavailable：" + reason)

        format_budget = 1      # 全流程共用的一次格式纠正（JSON 非法/协议违反/checks 不完整）
        replan_used = 0        # 每 Plan 最多一次局部重规划
        repair_rounds = 0      # 语义修复轮次（合法 repair 也不得无限空转）
        prev_issue_sig: frozenset[str] | None = None  # 上一轮 repairable 的 issue 身份集合
        prev_max_sev = 0                              # 上一轮 repairable 的最高严重度
        seen_action_sigs: set[str] = set()            # 已执行修复动作签名（重复动作熔断）
        correction = ""
        while True:
            try:
                review = _call_reviewer(correction)
                correction = ""
                _register_plan_issues(review)
                status = str(review.get("status") or "").strip()
                if status == "review_unavailable":
                    # 产出优先：与 blocked 同语义——标注后继续用当前 Plan
                    logger.warning("Plan 审核不可用（review_unavailable，标注继续）")
                    return _review_unavailable_return(str(review.get("reason") or ""))
                if status == "clean":
                    outcome_gaps = validate_review_outcome(
                        status, review.get("checks"), review.get("issues"),
                        review.get("repairs"))
                    if review.get("repairs"):
                        raise RepairRejected(
                            "；".join(outcome_gaps) or "clean 状态不得携带 repairs")
                    # 第 5 步准入：统一一致性校验 + boundary/单层/债务门禁合并
                    check_gaps = outcome_gaps + _plan_review_check_gaps(
                        review.get("checks"), plan, source,
                        graph=graph, registry=debt_registry)
                    if check_gaps:
                        # 语义准入举证不完整 = reviewer 输出不完整：走格式纠正
                        raise _PlanChecksIncomplete("；".join(check_gaps[:6]))
                    # 准入通过：逐 boundary 判定写回 plan.boundaries（落盘可审计）
                    _plan_merge_boundary_outcomes(plan, review.get("checks"))
                    return _accept_clean()
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
                            evidence_chunks=evidence_chunks, registry=debt_registry)
                    except (ReadingError, KimiError) as replan_exc:
                        # 产出优先：重规划失败（含 Planner KimiError，第 8 步收口）
                        # 不杀——旧 Plan 未被改动（_local_replan 克隆内工作），标注后继续
                        logger.warning("局部重规划失败（回滚旧 Plan，标注继续）：%s",
                                       replan_exc)
                        return _annotate_and_return(f"局部重规划失败：{replan_exc}")
                    continue
                if status == "repairable":
                    actions = [RepairAction.from_dict(a)
                               for a in (review.get("repairs") or [])]
                    sigs = [_plan_action_sig(a) for a in actions]
                    issue_sig, max_sev = _plan_review_issue_sig(review)
                    if prev_issue_sig is not None and issue_sig == prev_issue_sig \
                            and max_sev >= prev_max_sev \
                            and all(sig in seen_action_sigs for sig in sigs):
                        # 继续条件三选一全不满足：issue 未降 / 动作全重复
                        # （digest 未变由下方 no-op 熔断覆盖）→ 熔断不空转烧 token
                        logger.warning("Plan 修复无实质进展（issue 未降/动作重复，标注继续）")
                        return _annotate_and_return(
                            "Plan 修复无实质进展（issue 未降/动作重复，标注继续）")
                    repair_rounds += 1
                    if repair_rounds > PLAN_MAX_REPAIR_ROUNDS:
                        # 产出优先：语义修复预算有界——超限标注返回，不无限烧 token
                        logger.warning("Plan 修复超过 %d 轮仍未 clean（标注继续）",
                                       PLAN_MAX_REPAIR_ROUNDS)
                        return _annotate_and_return(
                            f"Plan 修复超过 {PLAN_MAX_REPAIR_ROUNDS} 轮仍未 clean"
                            "（语义修复预算耗尽，degraded 继续）")
                    engine = RepairEngine(graph, evidence_chunks=evidence_chunks,
                                          registry=debt_registry)
                    new_plan, stats = engine.apply(plan, actions)
                    if workspace:
                        workspace.put_json("plan-json", new_plan.to_dict())
                        if stats.registry is not None:
                            # 修复改变债务归属时（如 replace_plan_fragment）：
                            # owner 重绑后的注册表随 Plan 原子落盘
                            workspace.put_json("debt-registry",
                                               stats.registry.to_dict())
                    logger.info("Plan 修复：%d 操作，影响 %d 单元，新 plan_revision=%d",
                                stats.applied, len(stats.changed_units),
                                new_plan.plan_revision)
                    plan = new_plan
                    if not stats.changed_units and not stats.created \
                            and not stats.retired:
                        # no-op 熔断：修复无实质变化（plan digest 未变且无拓扑变化），不再空转
                        logger.warning("Plan 修复无实质变化（no-op 熔断，标注继续）")
                        return _annotate_and_return(
                            "Plan 修复无实质变化（no-op 熔断，标注继续）")
                    seen_action_sigs.update(sigs)
                    prev_issue_sig, prev_max_sev = issue_sig, max_sev
                    continue  # 修后必重审（含格式纠正通道的修复，统一回环）
                raise RepairRejected(f"非法状态：{status or '缺失'}")
            except _PlanChecksIncomplete as exc:
                if format_budget:
                    format_budget -= 1
                    correction = "checks 不合格：" + str(exc)
                    logger.warning("Plan 审核 checks 不合格（带修正重试一次）：%s", exc)
                    continue
                # 纠正后仍不合格 → 标注 degraded 继续，绝不判死
                logger.warning("语义准入审核不完整（标注继续）：%s", exc)
                marker = "语义准入审核不完整：" + str(exc)[:200]
                plan.meta.setdefault("issues", []).append(marker)
                self._ledger_degrade(_open_plan_ids(), marker, workspace)
                return plan, *plan.to_skeleton()
            except (RepairRejected, json.JSONDecodeError, ValueError) as exc:
                if format_budget:
                    format_budget -= 1
                    correction = str(exc)
                    logger.warning("Plan Reviewer 输出被拒（格式纠正一次）：%s", exc)
                    continue
                # 产出优先：审核输出连续非法不判死——标注 degraded 返回当前 Plan
                logger.warning("Plan 审核重试耗尽（degraded 继续）：%s", exc)
                return _annotate_and_return(f"Plan 审核重试耗尽（degraded 继续）：{exc}")

    def _local_replan(
        self,
        plan: "ReadingPlan",
        graph: "SemanticGraph",
        review: dict[str, Any],
        planner_client: Any,
        source: str,
        workspace: Any,
        evidence_chunks: set[str] | None = None,
        registry: Any | None = None,
    ) -> "ReadingPlan":
        """有界局部重规划：只重出受影响单元的合同，合并后完整校验。
        v1 简化：用骨架规划器对受影响区域重出字段（unit_id 保留，身份不动）。
        source 为论文原文（paper_md）：锚点偏移基于它，scope 内的 source spans
        从这里截取。registry 为 Canonical DebtRegistry（第 5 步接线）：修复后
        校验与 boundary 重算都要感知债务归属。"""
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
        gaps = validate_plan(work, graph, evidence_chunks=evidence_chunks,
                             registry=registry)
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
        # 重规划可能改写 debt_ids/source_span_ids：boundary 交集按新合同确定性重算
        # （outcome 复位 unknown，由回环后的重审重新判定）
        work.precompute_boundaries()
        if workspace:
            workspace.put_json("plan-json", work.to_dict())
        return work

    # ---- 主流程 ----


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

    # ---- V8 第 4 步：IssueLedger 接线（统一 ReviewIssue 归一 + 终态收口）----
    # 只记录与收口，不改任何审核判定逻辑；判定改造在第 5/6 步。

    def _ledger(self) -> Any:
        from .reading_issues import IssueLedger
        if self._issue_ledger is None:
            self._issue_ledger = IssueLedger()
        return self._issue_ledger

    def _ledger_load(self, workspace: Any) -> None:
        """run() 起点加载台账：schema 兼容则续用（恢复/返工不丢历史），否则重开。"""
        from .reading_issues import IssueLedger
        ledger = IssueLedger()
        if workspace is not None:
            raw = workspace.get_json("issue-ledger")
            if isinstance(raw, dict):
                try:
                    ledger = IssueLedger.from_dict(raw)
                except ValueError:
                    logger.warning("issue-ledger schema 不兼容，重开新台账")
        self._issue_ledger = ledger

    def _ledger_persist(self, workspace: Any) -> None:
        if workspace is not None:
            workspace.put_json("issue-ledger", self._ledger().to_dict())

    def _ledger_register_review(self, review: Any, *, stage: str,
                                target_ids: Iterable[str] = (),
                                workspace: Any = None) -> list[str]:
        """审核器出口归一：自由 issues 经 from_legacy 进 IssueLedger（幂等去重）并落盘。
        review_unavailable 口径以对应 kind 入册（无修复路由，走标注交付）。"""
        from .reading_issues import ReviewIssue
        if not isinstance(review, dict):
            return []
        unavailable = (str(review.get("tier") or "") == "review_unavailable"
                       or bool(review.get("review_unavailable")))
        issues = [
            ReviewIssue.from_legacy(
                item, stage,
                kind_override="review_unavailable" if unavailable else None,
                default_target_ids=target_ids)
            for item in (review.get("issues") or [])
            if isinstance(item, dict)
        ]
        if unavailable and not issues:
            # 终审等 unavailable 协议 issues 为空：合成一条入册（标注交付，重写 0 次）
            issues = [ReviewIssue.from_legacy(
                {"severity": "中", "problem": "审核服务不可用，未经完整审核"},
                stage, kind_override="review_unavailable",
                default_target_ids=target_ids)]
        ids = self._ledger().register(issues)
        if issues:
            self._ledger_persist(workspace)
        return ids

    def _ledger_resolve(self, issue_ids: Iterable[str], by: str,
                        workspace: Any = None) -> None:
        """幂等收口：只对仍 open 的入册 issue 置 resolved（重复调用安全）。"""
        ledger = self._ledger()
        open_ids = {issue.issue_id for issue in ledger.open_issues()}
        ids = [iid for iid in dict.fromkeys(issue_ids) if iid in open_ids]
        if ids:
            ledger.mark_resolved(ids, by)
            self._ledger_persist(workspace)

    def _ledger_degrade(self, issue_ids: Iterable[str], reason: str,
                        workspace: Any = None) -> None:
        """幂等收口：只对仍 open 的入册 issue 置 degraded_with_reason（带原因批注）。"""
        ledger = self._ledger()
        open_ids = {issue.issue_id for issue in ledger.open_issues()}
        ids = [iid for iid in dict.fromkeys(issue_ids) if iid in open_ids]
        if ids:
            ledger.mark_degraded(ids, reason)
            self._ledger_persist(workspace)

    def _gate_review(self, review: Any, *, unit_id: str,
                     workspace: Any = None) -> bool:
        """局审闸门（第 4 步接线）：先把审核 issues 归一进台账（幂等），再按原口径判定。"""
        self._ledger_register_review(review, stage="unit", target_ids=[unit_id],
                                     workspace=workspace)
        return self._local_review_passed(review)

    def _close_unit_issues(self, unit_id: str, *, passed: bool, review: Any,
                           label: str, workspace: Any = None) -> None:
        """单元审核循环出口的终态收口（禁静默丢失）：
        - 多轮未过 → 该单元全部 open issue 置 degraded（批注交付）；
        - 通过 → 早前轮次已修复的置 resolved；本轮携带的中低严重度意见按局审口径
          放行（随终审独立复核）置 degraded；review_unavailable 标注交付置 degraded。"""
        ledger = self._ledger()
        open_unit = [issue for issue in ledger.open_issues(stage="unit")
                     if unit_id in issue.target_ids]
        if not open_unit:
            return
        if not passed:
            self._ledger_degrade(
                [issue.issue_id for issue in open_unit],
                f"「{label}」多轮审核未过，全文保留加批注交付", workspace)
            return
        current_ids = set(self._ledger_register_review(
            review, stage="unit", target_ids=[unit_id], workspace=workspace))
        fixed = [issue.issue_id for issue in open_unit
                 if issue.issue_id not in current_ids]
        carried = [issue for issue in open_unit if issue.issue_id in current_ids]
        self._ledger_resolve(fixed, f"unit-review:pass:{unit_id}", workspace)
        self._ledger_degrade(
            [issue.issue_id for issue in carried if issue.kind == "review_unavailable"],
            "审核不可用，按 review_unavailable 口径标注交付（重写 0 次）", workspace)
        self._ledger_degrade(
            [issue.issue_id for issue in carried if issue.kind != "review_unavailable"],
            "中低严重度意见按局审口径放行，随终审独立复核", workspace)

    # ---- V8 第 6 步：block manifest + root_scope 分流 + plan/graph 回流 ----

    @staticmethod
    def _parse_review_issues(review: Any, *, stage: str,
                             default_target_ids: Iterable[str] = ()) -> list[Any]:
        """审核器自由 issues → ReviewIssue 列表（from_legacy 归一；非法项跳过不判死）。"""
        from .reading_issues import ReviewIssue
        out: list[Any] = []
        for item in (review.get("issues") or []) if isinstance(review, dict) else []:
            if not isinstance(item, dict):
                continue
            try:
                out.append(ReviewIssue.from_legacy(
                    item, stage, default_target_ids=default_target_ids))
            except ValueError:
                continue
        return out

    def _unit_blocks(self, unit_id: str, content: str, workspace: Any,
                     registry: Any, graph: Any) -> tuple[Any, str]:
        """计算并落盘单元 block manifest（workspace key `blocks-{unit_id}`），
        返回 (manifest, 注入 block 标记的审核稿)。审核稿只给审核器看，
        发布稿始终取未标注正文。"""
        from .reading_blocks import render_for_review, split_blocks
        manifest = split_blocks(unit_id, content, registry=registry, graph=graph)
        if workspace is not None:
            workspace.put_json(f"blocks-{unit_id}", manifest.to_dict())
        return manifest, render_for_review(content, manifest)

    def _delete_issue_blocks(self, review: Any, *, unit_id: str, content: str,
                             registry: Any, graph: Any
                             ) -> tuple[str, list[dict], str] | None:
        """root_scope=body 且带 body_block_ids 的 issue → 程序机械删块（零模型调用）。

        只执行删除点真实存在的块（审核器编造的块 ID 一律忽略）；返回
        (新正文, 合成 edits, 删除点相邻块上下文)；无可执行定位返回 None。"""
        from .reading_blocks import delete_blocks, deletion_neighbor_context, split_blocks
        block_ids: list[str] = []
        for issue in self._parse_review_issues(review, stage="unit",
                                               default_target_ids=[unit_id]):
            if issue.root_scope != "body":
                continue
            for bid in issue.body_block_ids:
                if bid not in block_ids:
                    block_ids.append(bid)
        if not block_ids:
            return None
        manifest = split_blocks(unit_id, content, registry=registry, graph=graph)
        known = [bid for bid in block_ids if manifest.get(bid) is not None]
        if not known:
            return None
        new_content = delete_blocks(content, manifest, known)
        if new_content.strip() == content.strip():
            return None  # 删除无实质变化（异常防御）
        edits = [{"find": (manifest.get(bid).text[:80] if manifest.get(bid) else bid),
                  "replace": ""} for bid in known]
        context = deletion_neighbor_context(manifest, known)
        return new_content, edits, context

    # ---- V8 第 8 步：ContinuityLedger（真实 term/claim/state 依赖）----

    @staticmethod
    def _unit_continuity_pack(unit_id: str, *, plan: Any, ledger: Any,
                              registry: Any, graph: Any
                              ) -> tuple[dict[str, Any], list[tuple[str, int]]]:
        """Writer 装包：只注入与本单元有真实依赖的 continuity 条目。

        返回（注入 dict, continuity_input_ids=[(entry_id, revision)]）——实际注入
        什么登记什么；manifest 依赖边与审核侧连续状态都用同一份。"""
        from .reading_continuity import unit_probe_text
        unit = plan.units[unit_id]
        probe = unit_probe_text(unit, plan=plan, registry=registry, graph=graph)
        return ledger.pack(probe)

    @staticmethod
    def _invalidate_continuity_consumers(changed: dict[str, int], manifest: Any,
                                         workspace: Any) -> None:
        """continuity 条目变化 → 按 manifest 依赖边精准失效消费者检查点。
        消费者总在建立者之后处理（装包时只能注入已建立条目），循环到达前失效即可。"""
        if not changed or workspace is None or manifest is None:
            return
        dirty = manifest.dirty_closure(
            {f"continuity:{eid}": rev for eid, rev in changed.items()})
        targets: set[str] = set(dirty)
        for artifact in dirty:
            if artifact.startswith("body-"):
                uid = artifact[len("body-"):]
                targets.update({f"review-{uid}", f"blocks-{uid}"})
        if targets:
            logger.info("continuity 条目变化（%d 项），精准失效消费者：%s",
                        len(changed), "、".join(sorted(targets)[:6]))
            workspace.invalidate_from(tuple(sorted(targets)))

    @staticmethod
    def _register_unit_continuity(ledger: Any, unit_id: str, review: Any,
                                  manifest: Any, workspace: Any) -> None:
        """局审出口的 continuity 登记（未降级才建立；降级/缺失撤下旧条目）并落盘。"""
        update = review.get("continuity_update") \
            if isinstance(review, dict) and not review.get("degraded") else None
        changed = ledger.register_unit(unit_id, update)
        if workspace is not None:
            workspace.put_json("continuity-ledger", ledger.to_dict())
        ReadingEngine._invalidate_continuity_consumers(changed, manifest, workspace)

    def _route_unit_issues(self, review: Any, *, unit_id: str, plan: Any,
                           registry: Any, graph: Any, planner_client: Any,
                           paper_md: str, workspace: Any, evidence_chunks: Any,
                           manifest: Any, progress: Callable[[str], None],
                           ) -> tuple[bool, Any, list[str]]:
        """局审失败先按 root_scope 分流（第 6 步）：plan/graph → 回流（committed 时
        调用方递归重跑）；asset → 只重做对应 figure 讲解；body 留给调用方的正文修复。

        返回 (committed, review, notes)：review 已剔除成功路由/降级的 issue；
        notes 为需要随正文批注的降级项（plan/graph 回流失败）。"""
        issues = self._parse_review_issues(review, stage="unit",
                                           default_target_ids=[unit_id])
        notes: list[str] = []
        if not issues:
            return False, review, notes
        plan_issues = [i for i in issues
                       if i.root_scope == "plan" and i.kind != "review_unavailable"]
        graph_issues = [i for i in issues if i.root_scope == "graph"]
        asset_issues = [i for i in issues if i.root_scope == "asset"]
        routed_ids: set[str] = set()
        for issue in asset_issues:
            # 只重做对应 figure 讲解：失效检查点，图阶段在单元循环之后重生成
            labels = [t for t in issue.target_ids if t and t != unit_id]
            if workspace is not None:
                for label in labels:
                    workspace.invalidate_from((f"figure-{label}",))
            self._ledger_resolve([issue.issue_id], f"asset-routing:{unit_id}", workspace)
            routed_ids.add(issue.issue_id)
            logger.info("asset 类问题路由：重做图表讲解 %s（%s）",
                        "、".join(labels) or "未指明", unit_id)
        if plan_issues:
            try:
                committed, plan_notes = self._attempt_plan_reflux(
                    plan_issues, plan=plan, registry=registry, graph=graph,
                    planner_client=planner_client, paper_md=paper_md,
                    workspace=workspace, evidence_chunks=evidence_chunks,
                    manifest=manifest, progress=progress,
                    raw_repairs=(review.get("repairs")
                                 if isinstance(review, dict) else None))
            except Exception as exc:  # 产出优先：回流任何异常 → 回滚+批注，绝不判死
                committed = False
                plan_notes = [f"plan 类回流异常（已回滚）：{str(exc)[:160]}"]
                self._ledger_degrade([i.issue_id for i in plan_issues],
                                     plan_notes[0], workspace)
            notes.extend(plan_notes)
            if committed:
                return True, review, notes
            routed_ids.update(i.issue_id for i in plan_issues)  # 失败已降级批注
        if graph_issues:
            try:
                g_committed, g_notes = self._attempt_graph_rebind(
                    graph_issues, unit_id=unit_id, plan=plan, graph=graph,
                    paper_md=paper_md, workspace=workspace, manifest=manifest)
            except Exception as exc:  # 产出优先：任何异常 → 批注继续，绝不判死
                g_committed = False
                g_notes = [f"graph 类回流异常（已回滚）：{str(exc)[:160]}"]
                self._ledger_degrade([i.issue_id for i in graph_issues],
                                     g_notes[0], workspace)
            notes.extend(g_notes)
            if g_committed:
                return True, review, notes
            routed_ids.update(i.issue_id for i in graph_issues)
        if routed_ids and isinstance(review, dict):
            # 已路由/降级的 issue 从本轮 review 剔除：剩余 issue 走正文修复路径
            remaining: list[dict] = []
            for item in (review.get("issues") or []):
                if not isinstance(item, dict):
                    continue
                parsed = self._parse_review_issues(
                    {"issues": [item]}, stage="unit", default_target_ids=[unit_id])
                if parsed and parsed[0].issue_id in routed_ids:
                    continue
                remaining.append(item)
            review = {**review, "issues": remaining}
            if not remaining:
                # 阻塞项全部完成路由（降级项已随 notes 批注）：不再进正文修复
                review["pass"] = True
        return False, review, notes

    def _attempt_plan_reflux(self, issues: list[Any], *, plan: Any, registry: Any,
                             graph: Any, planner_client: Any, paper_md: str,
                             workspace: Any, evidence_chunks: Any, manifest: Any,
                             progress: Callable[[str], None],
                             raw_repairs: Any = None,
                             ) -> tuple[bool, list[str]]:
        """plan 类 issue 回流：债务归属修复（DebtRepairEngine）、replace_plan_fragment
        拓扑事务（第 7 步）或 _local_replan 一次。

        通道（产出优先，绝不判死）：克隆 → 应用 → validate_plan 轻量复核 →
        原子提交落盘；任何一步失败回滚（克隆丢弃，旧 Plan/Registry 字节不变）
        + ledger degraded + 批注 notes。成功时：bind_debt_owners 重绑 →
        plan_revision 按 digest 变化递增 → boundary 重算 → 受影响单元
        （新旧 owner + manifest dirty_closure）精准失效 → issue resolved →
        调用方递归重跑（只重新生成受影响单元，其余正文不碰）。
        replan 分支第 7 步起：issue 带 merge/split 语义（kind 为
        contract_gap/debt_uncovered/boundary_fail 且 target_ids ≥2，或审核器
        经 raw_repairs 显式给出 fragment payload）时优先 replace_plan_fragment
        事务，被拒再退 _local_replan 一次，再失败批注。
        预算：全局每篇 ≤ PLAN_REFLUX_MAX 次；同动作签名熔断。"""
        from .reading_debts import DebtRepairAction, DebtRepairEngine
        from .reading_debts import RepairRejected as DebtRepairRejected
        from .reading_issues import IssueRouteError, route
        from .reading_plan import ReadingPlan, depends_on_closure, validate_plan
        from .reading_repair import RepairAction as PlanRepairAction
        from .reading_repair import RepairEngine as PlanRepairEngine
        from .reading_repair import RepairRejected as PlanRepairRejected

        notes: list[str] = []
        plan_issues = [i for i in issues
                       if i.root_scope == "plan" and i.kind != "review_unavailable"]
        if not plan_issues or workspace is None:
            return False, notes

        def _bail(reason: str) -> tuple[bool, list[str]]:
            self._ledger_degrade([i.issue_id for i in plan_issues], reason, workspace)
            logger.warning("plan 类回流未执行（回滚+批注继续）：%s", reason)
            return False, [reason]

        if self._reflux_count >= PLAN_REFLUX_MAX:
            return _bail(f"plan 类回流全局预算耗尽（{PLAN_REFLUX_MAX} 次/篇），标注继续")

        debts = getattr(registry, "debts", None) or {}
        layer_uids = [uid for uid in plan.order.get("layers") or []
                      if uid in plan.units]

        def _owners(did: str) -> list[str]:
            return [uid for uid in layer_uids
                    if did in (plan.units[uid].data.get("debt_ids") or [])]

        # ---- ① 债务归属动作（canonicalize/reassign）；② 其余 plan 类 → 局部重规划 ----
        debt_actions: list[Any] = []
        debt_edits: dict[str, list[str]] = {}   # uid → 修复后的 debt_ids
        replan_targets: list[str] = []
        actioned_issues: list[Any] = []
        replan_issues: list[Any] = []

        def _edit(uid: str, remove: Iterable[str] = (), add: Iterable[str] = ()) -> None:
            # 注意空列表也是有效编辑结果（不得用 or 回退到 plan 原值）
            current = list(debt_edits[uid] if uid in debt_edits
                           else plan.units[uid].data.get("debt_ids") or [])
            current = [d for d in current if d not in set(remove)]
            for did in add:
                if did not in current:
                    current.append(did)
            debt_edits[uid] = current

        for issue in plan_issues:
            try:
                ops = route(issue) or ()
            except IssueRouteError:
                ops = ()
            dids = [d for d in issue.debt_ids if d in debts]
            acted = False
            if dids and any(op in ("canonicalize_debt", "reassign_debt_owner")
                            for op in ops):
                if issue.kind == "debt_duplicated" and len(dids) >= 2:
                    # 两 debt 实同：机械合并（保留第一条，退休其余，引用重定向）；
                    # 重定向后归属收束到顺序首 owner（其余层移除，保持唯一归属）
                    keep, retire = dids[0], dids[1:]
                    debt_actions.append(DebtRepairAction(
                        op="canonicalize_debt",
                        expected_registry_revision=registry.registry_revision,
                        params={"keep_debt_id": keep, "retire_debt_ids": retire}))
                    for uid in layer_uids:
                        current = plan.units[uid].data.get("debt_ids") or []
                        if any(d in current for d in retire):
                            _edit(uid, remove=retire, add=[keep])
                    keep_owners = [
                        uid for uid in layer_uids
                        if keep in (debt_edits[uid] if uid in debt_edits
                                    else plan.units[uid].data.get("debt_ids") or [])]
                    for uid in keep_owners[1:]:
                        _edit(uid, remove=[keep])
                    acted = True
                else:
                    for did in dids:
                        owners = _owners(did)
                        if len(owners) > 1:
                            # 同 debt 双 owner：保留 subject 落在 primary 闭包内的
                            # 归属层（都不满足时保留顺序首层），其余移除
                            keep_owner = owners[0]
                            subject = debts[did].subject_node_id
                            for uid in owners:
                                primary = str(plan.units[uid].data.get(
                                    "primary_assertion_id") or "")
                                if subject and primary and (
                                        subject == primary or subject in
                                        depends_on_closure(graph, primary)):
                                    keep_owner = uid
                                    break
                            debt_actions.append(DebtRepairAction(
                                op="reassign_debt_owner",
                                expected_registry_revision=registry.registry_revision,
                                params={"debt_id": did,
                                        "owner_layer_id": keep_owner}))
                            for uid in owners:
                                if uid != keep_owner:
                                    _edit(uid, remove=[did])
                            acted = True
            if acted:
                actioned_issues.append(issue)
            else:
                replan_issues.append(issue)
                targets = [t for t in issue.target_ids if t in plan.units]
                if not targets:
                    targets = [uid for did in dids for uid in _owners(did)]
                replan_targets.extend(t for t in targets
                                      if t not in replan_targets)

        # ---- 第 7 步：merge/split 语义优先 replace_plan_fragment 事务 ----
        # 条件：全部 replan issue 为 contract_gap/debt_uncovered/boundary_fail 且
        # 定位目标 ≥2（机械翻译：吸收到首 target），或审核器显式给出 fragment
        # payload；事务被拒再退 _local_replan 一次，再失败批注
        fragment_action: Any | None = None
        if not debt_actions:
            explicit = _parse_fragment_repairs(raw_repairs, plan.plan_revision)
            merge_semantics = bool(replan_issues) and all(
                i.kind in FRAGMENT_REFLUX_KINDS for i in replan_issues)
            if explicit:
                fragment_action = explicit[0]
            elif merge_semantics and len(replan_targets) >= 2:
                fragment_action = PlanRepairAction(
                    issue_id="reflux-fragment:" + ",".join(
                        sorted(i.issue_id for i in replan_issues))[:80],
                    op="replace_plan_fragment",
                    target_ids=list(replan_targets),
                    expected_plan_revision=plan.plan_revision,
                    params=_mechanical_fragment_params(
                        replan_issues, replan_targets, plan, debts))

        # 同动作签名熔断（重复回流不空转）
        if debt_actions:
            sig_material = [
                f"{a.op}:{json.dumps(a.params, ensure_ascii=False, sort_keys=True)}"
                for a in debt_actions]
        elif fragment_action is not None:
            sig_material = ["replace_plan_fragment:" + json.dumps(
                fragment_action.params, ensure_ascii=False, sort_keys=True,
                default=str)]
        else:
            sig_material = [f"_local_replan:{','.join(sorted(replan_targets))}"]
        sig = "|".join(sig_material)
        if sig in self._reflux_sigs:
            return _bail("plan 类回流动作与历史重复（熔断，标注继续）")

        try:
            if debt_actions:
                work = ReadingPlan.from_dict(plan.to_dict())  # 克隆
                new_registry, stats = DebtRepairEngine().apply(
                    registry, graph, debt_actions)
                changed_units: list[str] = []
                for uid, new_dids in debt_edits.items():
                    unit = work.units[uid]
                    unit.data["debt_ids"] = new_dids
                    # must_explain/success_test 程序重派生（Registry 是唯一事实源）
                    unit.data["must_explain"] = [
                        new_registry.debts[d].learner_action
                        for d in new_dids if d in new_registry.debts]
                    unit.data["success_test"] = [
                        new_registry.debts[d].success_test
                        for d in new_dids if d in new_registry.debts]
                    sem_changed, _ = unit.refresh_digests()
                    if sem_changed:
                        changed_units.append(uid)
                if changed_units:
                    work.plan_revision += 1
                final_registry = work.bind_debt_owners(new_registry)  # owner 重绑
                gaps = validate_plan(work, graph, evidence_chunks=evidence_chunks,
                                     registry=final_registry)
                if gaps:
                    return _bail("回流后 Plan 不变量未通过：" + "；".join(gaps[:4]))
                # 轻量复核 = 程序 validate 通过（boundary 交集重算，判定待下轮全审）
                work.precompute_boundaries()
                work.reviewed_revision = work.plan_revision
                work.debt_registry_revision = final_registry.registry_revision
                work.meta.setdefault("issues", []).append(
                    f"第 6 步正文回流：{stats.applied} 个债务操作，"
                    f"影响 {len(changed_units)} 单元（boundary 待复核）")
                final_registry.issues.append(
                    f"第 6 步正文回流：{stats.applied} 个债务操作"
                    f"（{'、'.join(a.op for a in debt_actions)}）")
                workspace.put_json("plan-json", work.to_dict())
                workspace.put_json("debt-registry", final_registry.to_dict())
                affected = set(changed_units)
            else:
                # 其他 plan 类：第 7 步起 merge/split 语义优先 fragment 事务，
                # 被拒再退 _local_replan 通道一次（再失败由外层 _bail 批注）
                if not replan_targets and fragment_action is None:
                    return _bail("plan 类问题无可定位的回流目标，标注继续")
                work = None
                affected: set[str] = set()
                if fragment_action is not None:
                    progress("🔧 正文审核回流 Plan（replace_plan_fragment 事务）……")
                    try:
                        work, fstats = PlanRepairEngine(
                            graph, evidence_chunks=evidence_chunks,
                            registry=registry).apply(plan, [fragment_action])
                    except PlanRepairRejected as frag_exc:
                        logger.warning(
                            "replace_plan_fragment 被拒（退局部重规划一次）：%s",
                            frag_exc)
                        work = None
                    if work is not None:
                        # 轻量复核 = 程序 validate 通过（boundary 已重算，待下轮全审）
                        affected = (set(fstats.changed_units) | set(fstats.created)
                                    | set(fstats.retired))
                        final_registry = fstats.registry
                        work.reviewed_revision = work.plan_revision
                        if final_registry is not None:
                            work.debt_registry_revision = (
                                final_registry.registry_revision)
                            final_registry.issues.append(
                                f"第 7 步 fragment 回流：{len(fstats.created)} 建/"
                                f"{len(fstats.retired)} 退（boundary 待复核）")
                            workspace.put_json("debt-registry",
                                               final_registry.to_dict())
                        work.meta.setdefault("issues", []).append(
                            f"第 7 步 fragment 回流：{len(fstats.created)} 建/"
                            f"{len(fstats.retired)} 退，影响 {len(affected)} 单元"
                            "（boundary 待复核）")
                        workspace.put_json("plan-json", work.to_dict())
                if work is None:
                    if not replan_targets:
                        return _bail(
                            "plan 类问题无可定位的回流目标（fragment 已被拒），标注继续")
                    progress("🔧 正文审核回流 Plan（局部重规划一次）……")
                    work = self._local_replan(
                        plan, graph,
                        {"target_ids": replan_targets,
                         "issues": [i.to_dict() for i in replan_issues]},
                        planner_client, paper_md, workspace,
                        evidence_chunks=evidence_chunks, registry=registry)
                    affected = set(replan_targets)
            # ---- 精准失效：受影响单元 + manifest dirty_closure ----
            changed_inputs = {f"plan:{uid}": 0 for uid in affected}  # rev 0 强制失配
            dirty = manifest.dirty_closure(changed_inputs) if manifest else set()
            targets: set[str] = set(dirty)
            for artifact in dirty:
                if artifact.startswith("body-"):
                    uid = artifact[len("body-"):]
                    targets.update({f"review-{uid}", f"blocks-{uid}"})
            for uid in affected:
                targets.update({f"body-{uid}", f"review-{uid}", f"blocks-{uid}"})
            targets.update({"final-review", "final-body"})
            workspace.invalidate_from(tuple(targets))
            self._reflux_count += 1
            self._reflux_sigs.add(sig)
            resolved = actioned_issues if debt_actions else replan_issues
            self._ledger_resolve([i.issue_id for i in resolved],
                                 f"plan-reflux:{sig[:60]}", workspace)
            notes.append(f"plan 类回流已提交：失效 {len(affected)} 单元"
                         f"（含依赖闭包 {len(dirty)} 项），递归重跑")
            logger.warning("plan 类回流提交：影响 %s；精准失效 %d 项",
                           "、".join(sorted(affected)), len(dirty))
            progress("⤴ 审核发现 Plan 问题，已回写并局部重生成")
            return True, notes
        except (DebtRepairRejected, PlanRepairRejected, ReadingError, KimiError,
                ValueError) as exc:
            return _bail(f"回流动作被拒/失败（已回滚）：{str(exc)[:200]}")

    def _attempt_graph_rebind(self, issues: list[Any], *, unit_id: str, plan: Any,
                              graph: Any, paper_md: str, workspace: Any,
                              manifest: Any) -> tuple[bool, list[str]]:
        """graph 类 issue：锚点重定位（rebind_source_spans，每单元有界一次）。

        成功：重绑 supported_by + 落盘图 + plan.graph_revision 同步 +
        只失效真实消费者（manifest dirty_closure）→ committed（调用方递归）；
        失败：degraded 批注继续，绝不判死。"""
        from .reading_graph import GraphError, _span_for_offset, resolve_anchor
        from .reading_plan import graph_content_revision

        notes: list[str] = []
        graph_issues = [i for i in issues if i.root_scope == "graph"]
        if not graph_issues or workspace is None:
            return False, notes
        if unit_id in self._graph_rebound or self._reflux_count >= PLAN_REFLUX_MAX:
            reason = "graph 类回流有界一次已用完或全局预算耗尽，标注继续"
            self._ledger_degrade([i.issue_id for i in graph_issues], reason, workspace)
            return False, [reason]
        self._graph_rebound.add(unit_id)
        rebound: list[str] = []
        for issue in graph_issues:
            node_ids = [n for n in issue.node_ids if n in graph.nodes]
            if not node_ids:  # 兼容只给 span 的：反查 supported_by 起点节点
                for sid in issue.source_span_ids:
                    node_ids.extend(
                        e.from_id for e in graph.edges
                        if e.kind == "supported_by" and e.to_id == sid)
            for nid in dict.fromkeys(node_ids):
                node = graph.nodes.get(nid)
                quote = str((node.anchor or {}).get("quote") or "") if node else ""
                if node is None or not quote:
                    continue
                result = resolve_anchor(paper_md, quote)
                if not result.resolved:
                    continue
                node.anchor = {"quote": result.quote, "start": result.start,
                               "end": result.end, "digest": result.digest}
                node.unresolved_anchor = False
                span_id = _span_for_offset(graph.spans, result.start, result.end)
                if span_id:
                    graph.edges = [e for e in graph.edges
                                   if not (e.kind == "supported_by"
                                           and e.from_id == nid)]
                    try:
                        graph.add_edge("supported_by", nid, span_id)
                    except GraphError:
                        pass
                rebound.append(nid)
        if not rebound:
            reason = "锚点重定位未决（无可用摘录或低于阈值），标注继续"
            self._ledger_degrade([i.issue_id for i in graph_issues], reason, workspace)
            return False, [reason]
        workspace.put_json("semantic-graph", graph.to_dict())
        if plan is not None:
            # plan 的图摘要同步刷新为本次面对的图版本（避免恢复侧误判强制重审）
            plan.graph_revision = graph_content_revision(graph)
            workspace.put_json("plan-json", plan.to_dict())
        dirty = manifest.dirty_closure({f"node:{nid}": 0 for nid in rebound}) \
            if manifest else set()
        targets: set[str] = set(dirty)
        for artifact in dirty:
            if artifact.startswith("body-"):
                uid = artifact[len("body-"):]
                targets.update({f"review-{uid}", f"blocks-{uid}"})
        targets.update({"final-review", "final-body"})
        workspace.invalidate_from(tuple(targets))
        self._reflux_count += 1
        self._ledger_resolve([i.issue_id for i in graph_issues],
                             f"graph-rebind:{unit_id}", workspace)
        notes.append(f"锚点重定位 {len(rebound)} 节点，失效消费者 {len(dirty)} 项")
        logger.info("graph 类回流提交：重绑 %s；精准失效 %d 项",
                    "、".join(rebound), len(dirty))
        return True, notes

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
        # 第 4 步：统一问题台账加载（恢复/返工续用历史，schema 不兼容重开）
        self._ledger_load(workspace)
        # 第 6 步：plan/graph 类回流预算——首跑重置，回流/返工递归继承计数
        if rework_round == 0 and calls_carry == 0:
            self._reflux_count = 0
            self._reflux_sigs = set()
            self._graph_rebound = set()
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
        from .reading_plan import ReadingPlan, graph_content_revision, validate_plan
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

        # ---- 理解债务：DebtCompiler → DebtAuditor → Canonical DebtRegistry ----
        # 第 2 步：债务先于分层独立生成/审核；第 3 步：Plan 直接消费 Registry
        # （debt_ids 直出、must_explain/success_test 程序派生）。
        # 任何异常路径都不判死：degraded 批注继续。
        from .reading_debts import DebtAuditor
        progress("🧾 理解债务编译与准入审核中……")
        debt_registry, debt_stats = self._compile_debts(
            graph, planner_material, planner_client, workspace)
        model_calls += debt_stats.get("compile_calls", 0)
        audit_report: dict[str, Any] = {}
        if debt_registry.admission_status == "pending" and debt_registry.debts:
            try:
                debt_registry, audit_report = DebtAuditor().audit(
                    debt_registry, graph, reviewer_client, self.prompts_dir)
            except Exception as exc:  # 防御兜底：审核侧任何异常不判死
                logger.warning("债务审核异常（degraded 继续）：%s", exc)
                debt_registry.admission_status = "degraded"
                debt_registry.issues.append(f"债务审核异常：{str(exc)[:160]}")
                audit_report = {"notes": [f"债务审核异常：{str(exc)[:160]}"]}
            model_calls += int(audit_report.get("audit_calls", 0))
            debt_stats["notes"].extend(audit_report.get("notes") or [])
            if workspace:
                workspace.put_json("debt-registry", debt_registry.to_dict())
        debt_stats["audit_calls"] = int(audit_report.get("audit_calls", 0))
        debt_stats["repair_rounds"] = int(audit_report.get("repair_rounds", 0))
        timer.mark("债务编译")

        plan: ReadingPlan | None = None
        if workspace:
            raw_plan = workspace.get_json("plan-json")
            if isinstance(raw_plan, dict):
                try:
                    plan = ReadingPlan.from_dict(raw_plan)
                    logger.info("canonical Plan 命中检查点：plan_revision=%d", plan.plan_revision)
                except (ValueError, TypeError, KeyError):
                    plan = None
        # 恢复重审三项条件（第 5 步 PlanAuditor 准入语义的恢复侧）：
        # ① admission_status=pending（从未完成审核）；② 审核后 Plan 又被改动
        # （reviewed_revision≠plan_revision）；③ 图内容已变（graph_revision 与
        # 当前图摘要不符）。任一命中必须重审；三项都吻合的已审 Plan 才接缓存直通。
        plan_needs_review = plan is None
        if plan is not None and not plan_needs_review:
            current_graph_revision = graph_content_revision(graph)
            if plan.admission_status == "pending":
                plan_needs_review = True
                logger.info("plan-json 命中但 admission_status=pending：强制重审")
            elif plan.reviewed_revision != plan.plan_revision:
                plan_needs_review = True
                logger.info("plan-json reviewed_revision(%d)≠plan_revision(%d)：强制重审",
                            plan.reviewed_revision, plan.plan_revision)
            elif plan.graph_revision != current_graph_revision:
                plan_needs_review = True
                logger.info("plan-json graph_revision 与当前图摘要不符：强制重审")
        if plan is None:
            try:
                skeleton_out, skeleton_calls = self._call_skeleton(
                    planner_material, figures, formulas,
                    client=planner_client, workspace=workspace, evidence=evidence,
                    graph=graph, registry=debt_registry,
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
                        meta={"metadata": metadata}, registry=debt_registry,
                    )
                except PlanCollisionError as collision:
                    # 有界重规划一次：碰撞信息注入；再碰撞/调用失败/骨架不合格统一
                    # 收敛到下方 except ReadingError 的确定性降级 Plan，绝不判死。
                    # （降级骨架三节 purpose/reason 各异 + 单层，ID seed 唯一不自碰撞）
                    logger.warning("Plan ID 碰撞，有界重规划一次：%s", collision)
                    replan_prompt = self.build_plan_compile_prompt(
                        planner_material, figures, formulas,
                        graph=graph, registry=debt_registry,
                    ) + ("\n\n## 上一次规划存在 ID 碰撞\n" + str(collision)
                         + "\n请合并重复单元或明确各自语义角色后重新输出。")
                    try:
                        out = planner_client.chat(
                            [{"role": "user", "content": replan_prompt}],
                            heavy=True, max_tokens=PLANNER_OUTPUT_TOKENS,
                        )
                        model_calls += 1
                        gaps2 = skeleton_output_gaps(
                            out, evidence, graph=graph, registry=debt_registry)
                        if gaps2:
                            raise ReadingError("重规划骨架不合格：" + "；".join(gaps2))
                        skeleton, metadata = parse_skeleton_full(out)
                        metadata = {**meta_hint, **metadata}
                        plan = ReadingPlan.from_skeleton(
                            skeleton, graph, self.workflow_fingerprint()[:16],
                            meta={"metadata": metadata}, registry=debt_registry,
                        )
                    except (PlanCollisionError, KimiError) as replan_exc:
                        raise ReadingError(
                            f"碰撞重规划仍不可用：{replan_exc}") from replan_exc
            except (ReadingError, KimiError) as plan_exc:
                # 产出优先：骨架/规划失败（含 PlanCompiler KimiError，第 8 步收口）
                # → 确定性降级 Plan（三功能节 + 单层综合精讲），绝不判死
                logger.warning("骨架/规划失败，启用确定性降级 Plan：%s", plan_exc)
                skeleton, metadata = _degraded_skeleton(meta_hint, arxiv_id)
                plan = ReadingPlan.from_skeleton(
                    skeleton, graph, self.workflow_fingerprint()[:16],
                    meta={"metadata": metadata}, registry=debt_registry,
                )
                plan_needs_review = False
                progress("⚠️ 教学规划不可用，按确定性降级 Plan 继续生成")
            if workspace:
                workspace.put_json("plan-json", plan.to_dict())
        else:
            timer.mark("骨架")
            skeleton, metadata = plan.to_skeleton()
        # 债务 owner 写回（registry 克隆上改，不落盘——审核后统一提交落盘）
        debt_owned = plan.bind_debt_owners(debt_registry)
        evidence_chunk_ids = {c.chunk_id for c in evidence.chunks}
        plan_gaps = validate_plan(plan, graph, evidence_chunks=evidence_chunk_ids,
                                  registry=debt_registry)
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
        # 债务注册表状态透出（debt_count/去重/准入状态/降级原因），供结果卡与报告统计
        metadata["debt_registry"] = {
            "debt_count": len(debt_registry.debts),
            "dedup_removed": debt_stats.get("dedup_removed", 0),
            "admission_status": debt_registry.admission_status,
            "registry_revision": debt_registry.registry_revision,
            "notes": (debt_stats.get("notes") or [])[:8],
        }
        if graph.issues:
            metadata.setdefault("graph_issues", []).extend(
                json.dumps(item, ensure_ascii=False)[:160] for item in graph.issues[:6])
        # 直出 ID 绑定下，assertion_ids 来自 planner 直出 ∩ 图节点；
        # 图里有 assertion 可引而 core 层一个都没绑上才告警（弱图不重复告警）
        graph_has_assertions = any(n.kind == "assertion" for n in graph.nodes.values())
        unbound_core = [plan.units[uid].data.get("name")
                        for uid in plan.order.get("layers") or []
                        if plan.units[uid].data.get("depth") == "core"
                        and not plan.units[uid].assertion_ids
                        and graph_has_assertions]
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
        # 第 8 步：ContinuityLedger 加载（恢复/返工续用历史，schema 不兼容重开）
        from .reading_continuity import ContinuityLedger
        ledger = ContinuityLedger()
        if workspace:
            raw_ledger = workspace.get_json("continuity-ledger")
            if isinstance(raw_ledger, dict):
                try:
                    ledger = ContinuityLedger.from_dict(raw_ledger)
                except ValueError:
                    logger.warning("continuity-ledger schema 不兼容，重开新台账")

        # ---- PlanAuditor 四状态（ID 化之后，带原文证据）----
        if plan_needs_review:  # 新建/恢复三项重审条件命中必审；已审缓存命中不再审
            plan, skeleton, metadata = self._review_plan_and_repair(
                plan, graph, reviewer_client, planner_client,
                evidence.review_material() if evidence.long_mode else paper_md,
                paper_md, meta_hint, arxiv_id, workspace, progress,
                evidence_chunks=evidence_chunk_ids,
                debt_registry=debt_registry,
            )
            # 审核收口：记录已审 revision 与准入状态（pending → clean/degraded），
            # 图摘要同步刷新为本次审核面对的图版本——供恢复侧三项比对
            # （pending / reviewed_revision≠plan_revision / graph_revision 不符 → 重审）
            plan.reviewed_revision = plan.plan_revision
            plan.graph_revision = graph_content_revision(graph)
            if plan.admission_status == "pending":
                plan.admission_status = (
                    "degraded" if plan.meta.get("issues") else "clean")
            # 债务 owner 在审核后统一重绑：审核循环可能经 replace_plan_fragment/
            # _local_replan 改动各层 debt_ids（重绑幂等，未动时同值），下游消费者
            # （块清单/回流/落盘）一律使用审核后的归属
            debt_owned = plan.bind_debt_owners(debt_registry)
            if workspace:
                # 审核标注（blocked/review_unavailable/degraded 等）随 Plan 落盘可审计
                workspace.put_json("plan-json", plan.to_dict())
                workspace.put_json("debt-registry", debt_owned.to_dict())
        # 第 8 步：旧 continuity dict 已由 ContinuityLedger 取代——每单元处理时登记
        # （_register_unit_continuity），Writer/审核装包按真实依赖注入（_unit_continuity_pack）。

        # Planner 的任何文字都不发布；Writer 按计划逐节读取原文包并接受局部审核。
        overview_sections: list[str] = []
        overview_marked: list[str] = []   # 注入 block 标记的终审审核材料（不发布）
        block_index: dict[str, str] = {}  # block_id → unit_id（终审定位反查）
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
            if not content or not self._gate_review(review, unit_id=unit_id,
                                                    workspace=workspace):
                # 第 8 步：Writer 装包按真实依赖注入 continuity（实际注入条目
                # 登记为 manifest 依赖边 continuity:{entry_id}@revision）
                cont_state, cont_inputs = self._unit_continuity_pack(
                    unit_id, plan=plan, ledger=ledger, registry=debt_owned, graph=graph)
                prompt, source = self.build_section_prompt(
                    skeleton, section_contract, paper_md, figures, evidence, cont_state,
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
                    progress(f"⚠ {title}：生成失败，以占位段降级交付")
                    degraded.append(f"章节「{title}」")
                    # 占位单元不建立任何连续状态：撤下旧条目，失效旧消费者
                    self._register_unit_continuity(ledger, unit_id, None,
                                                   manifest, workspace)
                    placeholder = (f"## {title}\n\n> ⚠️ 本节生成失败（{exc}），"
                                   "以占位段降级交付，请对照原论文对应章节核实。")
                    overview_sections.append(placeholder)
                    overview_marked.append(placeholder)  # 占位无块标记，原样进审核材料
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
                else:
                    # 审核材料注入 block 标记（只给审核器看，发布稿不带走）
                    _, marked_content = self._unit_blocks(
                        unit_id, content, workspace, debt_owned, graph)
                    if light_review:
                        # 轻审=只审范围（越界/跨单元重复，收窄判定），事实归全审与终审
                        review = self._review_section(
                            reviewer_client, marked_content, section_contract, cont_state,
                            source, full_map=full_map, scope_only=True,
                            preparations=_preparations_text(section_contract),
                        )
                        review["tier"] = "light"
                        model_calls += 1
                    else:
                        review = self._review_section(
                            reviewer_client, marked_content, section_contract,
                            cont_state, source, full_map=full_map,
                        )
                        model_calls += 1
                # ---- 第 6 步：审核失败先按 root_scope 分流（plan/graph/asset 先行）----
                unit_notes: list[str] = []
                if not self._gate_review(review, unit_id=unit_id, workspace=workspace):
                    committed, review, unit_notes = self._route_unit_issues(
                        review, unit_id=unit_id, plan=plan, registry=debt_owned,
                        graph=graph, planner_client=planner_client, paper_md=paper_md,
                        workspace=workspace, evidence_chunks=evidence_chunk_ids,
                        manifest=manifest, progress=progress)
                    if committed:
                        # 回流已提交：受影响单元精准失效，递归重跑——检查点保证
                        # 未受影响单元正文不碰、Writer 零调用
                        return self.run(
                            paper_md, figures, formulas, arxiv_id, meta_hint, progress,
                            workspace, rework_round=rework_round,
                            calls_carry=calls_carry + model_calls)
                # ---- body 路径：block 机械删块 → 补丁一次 → 重写一次 → 保留+批注 ----
                last_valid = content      # 重写前快照（失败恢复+批注）
                patch_used = False        # block 机械操作与模型补丁共用一次预算
                rewrites = 0
                while not self._gate_review(review, unit_id=unit_id, workspace=workspace):
                    correction = self._review_issues(review if isinstance(review, dict) else {})
                    # 结构重复禁入补丁：按新分流走 plan/块级路径，局部补丁无权放行
                    struct_dup = isinstance(review, dict) and any(
                        "结构重复：" in str(it.get("problem") or "")
                        for it in (review.get("issues") or []) if isinstance(it, dict)
                    )
                    patch_edits: list[dict] | None = None
                    pre_patch: tuple[str, dict[str, Any]] | None = None
                    block_context: str | None = None
                    if not patch_used and isinstance(review, dict) \
                            and not review.get("_structural"):
                        # ① 块级机械修复（零模型）：issue 带 body_block_ids 直接删块
                        block_result = self._delete_issue_blocks(
                            review, unit_id=unit_id, content=content,
                            registry=debt_owned, graph=graph)
                        if block_result is not None:
                            pre_patch = (content, review)
                            content, patch_edits, block_context = block_result
                            patch_used = True
                            progress(f"✎ 补丁修复：{title}")
                        elif not struct_dup:
                            # ② 无 block 定位 → 现有补丁一次
                            patched_result = self._try_patch(
                                writer_client, content, correction, source)
                            if patched_result is not None:
                                pre_patch = (content, review)
                                content, patch_edits = patched_result
                                model_calls += 1
                                progress(f"✎ 补丁修复：{title}")
                            patch_used = True  # 补丁不可用（锚点缺失等）：预算消耗转入重写
                        else:
                            patch_used = True  # 结构重复禁入补丁（现行规则保留）
                    if patch_edits is None:
                        # ③ 仍失败 → 整单元重写一次（LOCAL_REVIEW_MAX_REWRITES=1）
                        if rewrites >= LOCAL_REVIEW_MAX_REWRITES:
                            break
                        last_valid = content
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
                        rewrites += 1
                        progress(f"✎ 重写：{title}"
                                 f"（{correction.split('；', 1)[0].strip()[:30]}）")
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
                    if patch_edits is not None:
                        # 补丁/删块复核：只审改动块+相邻句，不重审整节
                        review = self._patch_review(
                            reviewer_client, patch_edits, content, correction, source,
                            context_override=block_context,
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
                        _, marked_content = self._unit_blocks(
                            unit_id, content, workspace, debt_owned, graph)
                        if light_review:
                            review = self._review_section(
                                reviewer_client, marked_content, section_contract,
                                cont_state, source, full_map=full_map, scope_only=True,
                                preparations=_preparations_text(section_contract),
                            )
                            review["tier"] = "light"
                            model_calls += 1
                        else:
                            review = self._review_section(
                                reviewer_client, marked_content, section_contract,
                                cont_state, source, full_map=full_map,
                            )
                            model_calls += 1
                unit_passed = self._gate_review(review, unit_id=unit_id,
                                                workspace=workspace)
                if unit_passed and isinstance(review, dict):
                    if str(review.get("tier") or "") == "review_unavailable":
                        # 审核服务不可用：放行但透明标注（重写 0 次）
                        progress(f"⚠ {title}：审核服务不可用，本单元未经完整审核")
                    elif review.get("carried_issues"):
                        # 高严重度为零、中低意见放行：随终审独立复核
                        progress(f"✓ 局审通过：{title}"
                                 f"（带 {len(review['carried_issues'])} 条待复核意见）")
                if not unit_passed:
                    # 保留+批注：预算内仍带高严重度问题 → 恢复 last_valid 快照，
                    # 全文保留，顶部插入存疑横幅；不整节撤下（句级错误不连坐）
                    if rewrites:
                        content = last_valid
                    brief = self._review_issues(review if isinstance(review, dict) else {})
                    if unit_notes:
                        brief = "；".join([*unit_notes, brief])
                    logger.warning("章节「%s」审核未过，保留全文加批注：%s", title, brief[:120])
                    progress(f"⚠ {title}：审核未过，保留全文加批注")
                    degraded.append(f"章节「{title}」")
                    content = _annotate_degraded(content, brief)
                    review = {**(review if isinstance(review, dict) else {}),
                              "degraded": True, "degrade_mode": "annotated", "pass": False}
                elif unit_notes:
                    # 路由阶段遗留（plan/graph 回流失败降级）：批注透明交付，不重写
                    logger.warning("章节「%s」携带回流降级批注交付：%s",
                                   title, "；".join(unit_notes)[:120])
                    degraded.append(f"章节「{title}」")
                    content = _annotate_degraded(content, "；".join(unit_notes))
                    review = {**(review if isinstance(review, dict) else {}),
                              "degraded": True, "degrade_mode": "annotated"}
                # 第 4 步：本单元 issue 终态收口（resolved / degraded_with_reason）
                self._close_unit_issues(unit_id, passed=unit_passed, review=review,
                                        label=title, workspace=workspace)
                if workspace:
                    workspace.put_text(key, content)
                    if isinstance(review, dict):
                        # 产物签名随审核落盘：workflow 指纹 + unit semantic_revision；
                        # continuity_input_ids 随审核留痕（实际注入什么登记什么）
                        review = {**review, "_artifact_sig": self._artifact_sig(
                            key, plan.units[unit_id].semantic_revision),
                            "_continuity_input_ids": [eid for eid, _ in cont_inputs]}
                    workspace.put_json(review_key, review)
                    workspace.invalidate_from((f"rework-{key}",))  # 返工意见一次有效
                    # 第 8 步：删除 body-{prev_uid} 默认前序边——依赖只登记
                    # plan 合同 revision + 实际注入的 continuity 条目 + 绑定 deps
                    manifest.register(
                        f"body-{unit_id}",
                        {f"plan:{unit_id}": plan.units[unit_id].semantic_revision,
                         **{f"continuity:{eid}": rev for eid, rev in cont_inputs},
                         **_unit_binding_deps(plan.units[unit_id])},
                    )
                    manifest.register(
                        f"review-{unit_id}",
                        {f"body-{unit_id}": 1},
                    )
                    workspace.put_json("dependency-manifest",
                                       {"inputs": manifest._inputs})
            # 第 8 步：局审出口登记 continuity（未降级才建立；降级撤下旧条目），
            # 条目变化按 manifest 依赖边精准失效消费者
            self._register_unit_continuity(ledger, unit_id, review, manifest, workspace)
            # 每单元 block manifest 落盘 + 终审审核材料（注入标记版）
            bm_final, marked_final = self._unit_blocks(
                unit_id, content.strip(), workspace, debt_owned, graph)
            block_index.update({b.block_id: unit_id for b in bm_final.blocks})
            overview_sections.append(content.strip())
            overview_marked.append(marked_final)
        overview_md = "\n\n".join(overview_sections)

        progress(f"✅ 教学计划完成（识别出 {len(skeleton.layers)} 个技术层），开始分层精讲")

        # ---- 调用 2..N+1：每层一次 heavy 教学，缺理解闭环时修正重试 ----
        total_layers = len(skeleton.layers)
        layer_sections: list[str] = []
        layers_marked: list[str] = []   # 注入 block 标记的终审审核材料（不发布）
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
            # continuity 装包很轻（台账条目 × 探针 bigram），逐轮确定性重算：
            # Writer 注入与审核侧连续状态、manifest 依赖边共用同一份。
            cont_state, cont_inputs = self._unit_continuity_pack(
                layer_uid, plan=plan, ledger=ledger, registry=debt_owned, graph=graph)
            prompt: str | None = None

            def layer_prompt() -> str:
                nonlocal prompt
                if prompt is None:
                    prompt = self.build_layer_prompt(
                        skeleton, layer, i, total_layers, paper_md, figures,
                        continuity_state=cont_state,
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
                    progress(f"⚠ {layer.name}：生成失败，以占位段降级交付")
                    degraded.append(f"技术层「{layer.name}」")
                    # 占位单元不建立任何连续状态：撤下旧条目，失效旧消费者
                    self._register_unit_continuity(ledger, layer_uid, None,
                                                   manifest, workspace)
                    placeholder = (f"### {layer.name}\n\n> ⚠️ 本层生成失败（{exc}），"
                                   "以占位段降级交付，请对照原论文核实。")
                    layer_sections.append(placeholder)
                    layers_marked.append(placeholder)  # 占位无块标记，原样进审核材料
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
            if not self._gate_review(local_review, unit_id=layer_uid,
                                     workspace=workspace):
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
                # 审核材料注入 block 标记（只给审核器看，发布稿不带走）
                _, marked_content = self._unit_blocks(
                    layer_uid, content, workspace, debt_owned, graph)
                if layer.depth != "core":
                    # 轻审=只审范围；事实与债务核验收归终审；core 层仍逐层模型审
                    local_review = self._review_section(
                        reviewer_client, marked_content, layer_contract, cont_state, source,
                        full_map=full_map, scope_only=True,
                    )
                    local_review["tier"] = "light"
                    model_calls += 1
                else:
                    local_review = self._review_section(
                        reviewer_client, marked_content, layer_contract, cont_state, source,
                        full_map=full_map,
                    )
                    model_calls += 1
                # ---- 第 6 步：审核失败先按 root_scope 分流（plan/graph/asset 先行）----
                unit_notes: list[str] = []
                if not self._gate_review(local_review, unit_id=layer_uid,
                                         workspace=workspace):
                    committed, local_review, unit_notes = self._route_unit_issues(
                        local_review, unit_id=layer_uid, plan=plan, registry=debt_owned,
                        graph=graph, planner_client=planner_client, paper_md=paper_md,
                        workspace=workspace, evidence_chunks=evidence_chunk_ids,
                        manifest=manifest, progress=progress)
                    if committed:
                        # 回流已提交：受影响单元精准失效，递归重跑——检查点保证
                        # 未受影响单元正文不碰、Writer 零调用
                        return self.run(
                            paper_md, figures, formulas, arxiv_id, meta_hint, progress,
                            workspace, rework_round=rework_round,
                            calls_carry=calls_carry + model_calls)
                # ---- body 路径：block 机械删块 → 补丁一次 → 重写一次 → 保留+批注 ----
                last_valid = content      # 重写前快照（失败恢复+批注）
                patch_used = False        # block 机械操作与模型补丁共用一次预算
                rewrites = 0
                while not self._gate_review(local_review, unit_id=layer_uid,
                                            workspace=workspace):
                    correction = self._review_issues(local_review)
                    # 结构重复禁入补丁：按新分流走 plan/块级路径，局部补丁无权放行
                    struct_dup = isinstance(local_review, dict) and any(
                        "结构重复：" in str(it.get("problem") or "")
                        for it in (local_review.get("issues") or []) if isinstance(it, dict)
                    )
                    patch_edits: list[dict] | None = None
                    pre_patch: tuple[str, dict[str, Any]] | None = None
                    block_context: str | None = None
                    if not patch_used and isinstance(local_review, dict):
                        # ① 块级机械修复（零模型）：issue 带 body_block_ids 直接删块
                        block_result = self._delete_issue_blocks(
                            local_review, unit_id=layer_uid, content=content,
                            registry=debt_owned, graph=graph)
                        if block_result is not None:
                            pre_patch = (content, local_review)
                            content, patch_edits, block_context = block_result
                            patch_used = True
                            progress(f"✎ 补丁修复：{layer.name}")
                        elif not struct_dup:
                            # ② 无 block 定位 → 现有补丁一次
                            patched_result = self._try_patch(
                                writer_client, content, correction, source)
                            if patched_result is not None:
                                pre_patch = (content, local_review)
                                content, patch_edits = patched_result
                                model_calls += 1
                                progress(f"✎ 补丁修复：{layer.name}")
                            patch_used = True  # 补丁不可用（锚点缺失等）：预算消耗转入重写
                        else:
                            patch_used = True  # 结构重复禁入补丁（现行规则保留）
                    if patch_edits is None:
                        # ③ 仍失败 → 整单元重写一次（LOCAL_REVIEW_MAX_REWRITES=1）
                        if rewrites >= LOCAL_REVIEW_MAX_REWRITES:
                            break
                        last_valid = content
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
                        rewrites += 1
                        progress(f"✎ 重写：{layer.name}"
                                 f"（{correction.split('；', 1)[0].strip()[:30]}）")
                    gaps = teaching_layer_gaps(content, layer)
                    if gaps:
                        logger.warning("技术层「%s」重写后结构缺口，批注继续：%s",
                                       layer.name, "、".join(gaps))
                        degraded.append(f"技术层「{layer.name}」")
                        content = _annotate_degraded(
                            content, "结构缺口：" + "、".join(gaps))
                    if patch_edits is not None:
                        # 补丁/删块复核：只审改动块+相邻句，不重审整层
                        local_review = self._patch_review(
                            reviewer_client, patch_edits, content, correction, source,
                            context_override=block_context,
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
                        _, marked_content = self._unit_blocks(
                            layer_uid, content, workspace, debt_owned, graph)
                        local_review = self._review_section(
                            reviewer_client, marked_content, layer_contract, cont_state, source,
                            full_map=full_map,
                        )
                    model_calls += 1
                layer_passed = self._gate_review(local_review, unit_id=layer_uid,
                                                 workspace=workspace)
                if layer_passed and isinstance(local_review, dict):
                    if str(local_review.get("tier") or "") == "review_unavailable":
                        # 审核服务不可用：放行但透明标注（重写 0 次）
                        progress(f"⚠ {layer.name}：审核服务不可用，本单元未经完整审核")
                    elif local_review.get("carried_issues"):
                        # 高严重度为零、中低意见放行：随终审独立复核
                        progress(f"✓ 局审通过：{layer.name}"
                                 f"（带 {len(local_review['carried_issues'])} 条待复核意见）")
                if not layer_passed:
                    # 保留+批注：预算内仍带高严重度问题 → 恢复 last_valid 快照，
                    # 全文保留，顶部插入存疑横幅
                    if rewrites:
                        content = last_valid
                    brief = self._review_issues(local_review if isinstance(local_review, dict) else {})
                    if unit_notes:
                        brief = "；".join([*unit_notes, brief])
                    logger.warning("技术层「%s」审核未过，保留全文加批注：%s", layer.name, brief[:120])
                    progress(f"⚠ {layer.name}：审核未过，保留全文加批注")
                    degraded.append(f"技术层「{layer.name}」")
                    content = _annotate_degraded(content, brief)
                    local_review = {**(local_review if isinstance(local_review, dict) else {}),
                                    "degraded": True, "degrade_mode": "annotated", "pass": False}
                elif unit_notes:
                    # 路由阶段遗留（plan/graph 回流失败降级）：批注透明交付，不重写
                    logger.warning("技术层「%s」携带回流降级批注交付：%s",
                                   layer.name, "；".join(unit_notes)[:120])
                    degraded.append(f"技术层「{layer.name}」")
                    content = _annotate_degraded(content, "；".join(unit_notes))
                    local_review = {**(local_review if isinstance(local_review, dict) else {}),
                                    "degraded": True, "degrade_mode": "annotated"}
                # 第 4 步：本层 issue 终态收口（resolved / degraded_with_reason）
                self._close_unit_issues(layer_uid, passed=layer_passed,
                                        review=local_review, label=layer.name,
                                        workspace=workspace)
                if workspace:
                    workspace.put_text(layer_key, content)
                    if isinstance(local_review, dict):
                        # 产物签名随审核落盘：workflow 指纹 + unit semantic_revision；
                        # continuity_input_ids 随审核留痕（实际注入什么登记什么）
                        local_review = {**local_review, "_artifact_sig": self._artifact_sig(
                            layer_key, plan.units[layer_uid].semantic_revision),
                            "_continuity_input_ids": [eid for eid, _ in cont_inputs]}
                    workspace.put_json(layer_review_key, local_review)
                    workspace.invalidate_from((f"rework-{layer_key}",))  # 返工意见一次有效
                    # 第 8 步：删除 body-{prev_layer_uid} 默认前序边——依赖只登记
                    # plan 合同 revision + 实际注入的 continuity 条目 + 绑定 deps
                    layer_deps = {f"plan:{layer_uid}":
                                  plan.units[layer_uid].semantic_revision,
                                  **{f"continuity:{eid}": rev for eid, rev in cont_inputs},
                                  **_unit_binding_deps(plan.units[layer_uid])}
                    manifest.register(layer_key, layer_deps)
                    workspace.put_json("dependency-manifest",
                                       {"inputs": manifest._inputs})
            # 第 8 步：局审出口登记 continuity（未降级才建立；降级撤下旧条目），
            # 条目变化按 manifest 依赖边精准失效消费者
            self._register_unit_continuity(ledger, layer_uid, local_review,
                                           manifest, workspace)
            section_body = re.sub(r"^###\s*[^\n]+\n+", "", content.strip(), count=1)
            section = f"### {layer.name}\n\n{section_body}"
            # 每单元 block manifest 落盘 + 终审审核材料（注入标记版）
            bm_final, marked_final = self._unit_blocks(
                layer_uid, section, workspace, debt_owned, graph)
            block_index.update({b.block_id: layer_uid for b in bm_final.blocks})
            layer_sections.append(section)
            layers_marked.append(marked_final)
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
                # 第 4 步：图表验收缺口入台账并带原因降级（禁静默丢失）
                self._ledger_register_review(
                    {"issues": [{"severity": "中",
                                 "problem": f"图表讲解未通过局部验收：{'、'.join(gaps)}"}]},
                    stage="figure", target_ids=[label], workspace=workspace)
                self._ledger_degrade(
                    [issue.issue_id for issue in self._ledger().open_issues(stage="figure")
                     if label in issue.target_ids],
                    "图表讲解透明降级：保留原图与图注，未补写读数/比较/因果结论",
                    workspace)
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
        # 终审审核材料：注入 block 标记的版本（只给审核器看，发布稿 body 不带走）
        review_body = self._assemble_final(
            metadata, skeleton, "\n\n".join(layers_marked), figures, figure_notes,
            figure_inlines, overview_md="\n\n".join(overview_marked),
        ) if (overview_marked or layers_marked) else body
        cached_body = workspace.get_text("final-body") if workspace else None
        review = workspace.get_json("final-review") if workspace and cached_body == body else None
        if isinstance(review, dict) and not self._artifact_sig_matches(
                review.get("_artifact_sig"), "final-body", plan.plan_revision):
            # 产物签名不匹配（plan revision/工作流指纹已变）：终审缓存失效，重新终审
            logger.info("终审产物签名不匹配，重新终审")
            review = None
        if not isinstance(review, dict):
            review = self._review_final(
                review_body, review_source, formulas, skeleton, client=reviewer_client,
            )
            model_calls += 1
            if review.get("review_unavailable"):
                # 产出优先：终审不可用不杀正文——横幅标注交付（正文重写 0 次）
                progress("⚠ 终审服务不可用，按未经终审复核标注交付")
            # 第 8 步机制 A 程序对账：同一图讲解段跨单元重复 → issue 并入终审
            # （body 类块定位，走有界返工；仅新鲜终审补一次，缓存终审已含；
            # 终审不可用时不加注——review_unavailable 口径正文重写 0 次）
            if not review.get("review_unavailable"):
                repeat_issues = self._figure_repeat_issues(
                    body, figures, figure_notes,
                    [*zip(plan.order.get("sections") or [], overview_sections),
                     *zip(plan.order.get("layers") or [], layer_sections)])
                if repeat_issues:
                    logger.info("图讲解跨单元重复对账：%d 条（body 类块定位）",
                                len(repeat_issues))
                    review = {**review, "issues": [
                        *(review.get("issues") or []), *repeat_issues]}
        # 第 4 步：终审 issues 归一进台账（幂等；review_unavailable 以对应 kind 入册）
        self._ledger_register_review(review, stage="final", workspace=workspace)
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
            if review_unavailable:
                # 审核不可用不是“评分异常通过”，正文必须明确写出未经终审复核。
                detail = str(review.get("summary") or "终审服务不可用，未经终审复核")
            elif not score_valid:
                detail = ("终审返回的评分缺失或异常（审核模型输出不完整）；"
                          "正文按程序检查与局审结果交付")
            # 第 6 步：终审 plan 类问题也走回流（预算内），不只重写正文
            if score_valid and workspace is not None:
                final_plan_issues = [
                    i for i in self._parse_review_issues(review, stage="final")
                    if i.root_scope == "plan" and i.kind != "review_unavailable"
                ]
                if final_plan_issues:
                    committed, reflux_notes = self._attempt_plan_reflux(
                        final_plan_issues, plan=plan, registry=debt_owned, graph=graph,
                        planner_client=planner_client, paper_md=paper_md,
                        workspace=workspace, evidence_chunks=evidence_chunk_ids,
                        manifest=manifest, progress=progress,
                        raw_repairs=review.get("repairs"))
                    if committed:
                        logger.warning("终审 plan 类问题回流成功，递归重跑：%s",
                                       "；".join(reflux_notes))
                        return self.run(
                            paper_md, figures, formulas, arxiv_id, meta_hint, progress,
                            workspace, rework_round=rework_round,
                            calls_carry=calls_carry + model_calls,
                        )
                    if reflux_notes:
                        metadata.setdefault("final_plan_reflux_notes",
                                            []).extend(reflux_notes)
            # 终审定位：block_id/debt_id 结构化反查优先，展示名模糊匹配兜底
            debt_owner = {
                did: uid
                for uid in (plan.order.get("layers") or []) if uid in plan.units
                for did in (plan.units[uid].data.get("debt_ids") or [])
            }
            failed_units, core_involved = _locate_failed_units(
                review if isinstance(review, dict) else {},
                skeleton, contract_gaps, {asset.label for asset in figures},
                plan=plan, block_index=block_index, debt_owner=debt_owner,
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
                            if artifact.startswith("body-"):
                                uid = artifact[len("body-"):]
                                targets.add(f"review-{uid}")
                                targets.add(f"blocks-{uid}")  # 块清单随正文一并失效
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
                # 第 4 步：终审未过横幅交付——遗留 open issue 全部带原因降级
                self._ledger_degrade(
                    [issue.issue_id for issue in self._ledger().open_issues()],
                    f"终审未全部通过，横幅交付：{detail[:120]}", workspace)
        if final_ok:
            # 第 4 步：终审通过——全部遗留 open issue 视为已被终审复核接受
            self._ledger_resolve(
                [issue.issue_id for issue in self._ledger().open_issues()],
                "final-review:pass", workspace)
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
        # 第 4 步收口：入册 issue 不得静默丢失——仍 open 的一律带原因降级并透出汇总
        stray = [issue.issue_id for issue in self._ledger().open_issues()]
        if stray:
            logger.warning("流程收口仍有 %d 条未终结 issue（带原因降级）", len(stray))
            self._ledger_degrade(
                stray, "流程收口时仍未解决，随正文批注/横幅透明交付", workspace)
        metadata["issue_ledger"] = self._ledger().summary()
        self._ledger_persist(workspace)
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
            return {"pass": False, "score": None, "review_unavailable": True,
                    "summary": "终审服务不可用，未经终审复核",
                    "issues": [], "layer_coverage": [], "teach_back": []}
        try:
            review = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("终审返回非法 JSON，标记 review_unavailable 保留正文")
            return {"pass": False, "score": None, "review_unavailable": True,
                    "summary": "终审服务输出异常，未经终审复核",
                    "issues": [], "layer_coverage": [], "teach_back": []}
        if not isinstance(review, dict):
            return {"pass": False, "score": None, "review_unavailable": True,
                    "summary": "终审服务输出异常，未经终审复核",
                    "issues": [], "layer_coverage": [], "teach_back": []}
        return review

    # 机制 A 程序对账：块文本近乎逐字重复的 bigram 包含度阈值
    _FIGURE_REPEAT_MIN_CONTAINMENT = 0.75
    _FIGURE_REPEAT_MIN_CHARS = 60

    def _figure_repeat_issues(self, body: str, figures: list[FigureAsset],
                              figure_notes: dict[str, str],
                              unit_bodies: list[tuple[str, str]]) -> list[dict[str, Any]]:
        """机制 A 程序对账（第 8 步）：同一图讲解段跨单元重复出现 → 终审 issue
        （body 类块定位，走 _locate_failed_units 有界返工；只报不修）。

        两种确定性形态：
        ① figure note 完整文本（签名段）在图表总览之外的正文出现；
        ② 两个单元引用同一图的块彼此近乎逐字（bigram 包含度 ≥ 阈值）——
           按单元顺序后出现的块判重复。
        unit_bodies：[(unit_id, 单元正文)] 按文档顺序（节→层）；与 blocks-{uid}
        同一算法同一输入切块 → block_id 一致，可供 block_index 定位反查。
        返回可直接并入终审 review["issues"] 的 dict 列表。"""
        from .reading_blocks import _containment, split_blocks

        def _norm(text: str) -> str:
            return re.sub(r"[^\w一-鿿]+", "", str(text or "")).casefold()

        main_body = body.split("\n## 图表总览", 1)[0]  # 总览合法持有完整讲解
        main_norm = _norm(main_body)
        # 逐单元切块（block_id 与落盘的 blocks-{uid} 一致）；per_unit 保留单元边界
        blocks: list[tuple[str, str]] = []          # (block_id, text) 按单元顺序
        per_unit: list[list[tuple[str, str]]] = []  # 同序的按单元分组
        for uid, unit_body in unit_bodies:
            manifest = split_blocks(uid, unit_body or "")
            unit_blocks = [(b.block_id, b.text) for b in manifest.blocks]
            blocks.extend(unit_blocks)
            per_unit.append(unit_blocks)
        issues: list[dict[str, Any]] = []
        for asset in figures:
            label = asset.label
            dup_ids: list[str] = []
            # ① figure note 完整文本（签名段）出现在图表总览之外
            note = re.sub(r"\s+", "", str(figure_notes.get(label) or ""))
            signature = _norm(note)[:120]
            if len(signature) >= 60 and signature in main_norm:
                for bid, text in blocks:
                    if signature[:60] in _norm(text):
                        dup_ids.append(bid)
            # ② 跨单元近乎逐字的同图讲解段（保留首次，后出现的判重复）。
            # 讲解段 = 标记块本身（标记与讲解同块时）或标记块+紧随的讲解块
            # （split_blocks 会把空行隔开的标记与讲解切成两块）。
            regions: list[tuple[str, str]] = []  # (讲解块 block_id, 区域文本) 按单元顺序
            for unit_blocks in per_unit:
                for pos, (bid, text) in enumerate(unit_blocks):
                    if f"[[FIG:{label}]]" not in text and label not in text:
                        continue
                    if len(_norm(text)) >= self._FIGURE_REPEAT_MIN_CHARS:
                        regions.append((bid, text))
                    elif pos + 1 < len(unit_blocks):
                        next_bid, next_text = unit_blocks[pos + 1]
                        regions.append((next_bid, text + "\n\n" + next_text))
            for prev_i, (_, prev_region) in enumerate(regions):
                if len(_norm(prev_region)) < self._FIGURE_REPEAT_MIN_CHARS:
                    continue
                for bid, region in regions[prev_i + 1:]:
                    if len(_norm(region)) < self._FIGURE_REPEAT_MIN_CHARS:
                        continue
                    if max(_containment(region, prev_region),
                           _containment(prev_region, region)
                           ) >= self._FIGURE_REPEAT_MIN_CONTAINMENT:
                        if bid not in dup_ids:
                            dup_ids.append(bid)
            if dup_ids:
                issues.append({
                    "severity": "高", "kind": "structural_duplication",
                    "root_scope": "body", "location": label,
                    "problem": f"结构重复：{label} 的讲解段跨单元重复出现；只保留首次"
                               "内联位置的完整讲解，其余位置改为「讲解见首次内联位置/"
                               "图表总览」的短引用",
                    "debt_ids": [], "body_block_ids": dup_ids,
                })
        return issues

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
        """内联图讲解注入（第 8 步机制 A 去重）：同一张图的完整讲解只保留首次
        内联位置；后续内联位置替换为短引用（指向首次内联节标题）。
        图表总览在调用方之后拼接，不受影响。"""
        headings = [(m.start(), m.group(1).strip())
                    for m in re.finditer(r"^#{2,3}\s+(.+?)\s*$", body, re.M)]
        seen: dict[str, str] = {}  # label → 首次内联位置的节标题

        def repl(m: re.Match) -> str:
            label = m.group(1).strip()
            inline = figure_inlines.get(label)
            if not inline:
                return m.group(0)
            if label in seen:
                return f"[[FIG:{label}]]\n\n（{label} 讲解见「{seen[label]}」）"
            heading = next((title for pos, title in reversed(headings)
                            if pos < m.start()), "图表总览")
            seen[label] = heading
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
