"""V9 叙事优先精读引擎。

V9 不是在 V8 的 Debt/Plan/局审/回流闭环上继续加规则，而是保留解析、证据、
缓存和发布底盘，替换为：同一作者 Brief → 连续初稿 → 两次专项审查 → 一次结构化
补丁修订 → 定点验证。任何修订不可验证时回滚到不可覆盖的 author-draft。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from ..config import PROJECT_ROOT
from ..kimi import KimiClient, KimiError
from .figures import FigureAsset, figure_inventory_text
from .formulas import FormulaAsset, formula_inventory_text
from .reading import (
    ReadingError,
    ReadingResult,
    StageTimer,
    _hoist_media_markers,
    _strip_inline_latex,
    strip_h1,
    strip_references,
)
from .reading_evidence import (
    EvidenceBundle,
    EvidenceError,
    EvidenceIndexer,
    LONG_PAPER_CHARS,
    PaperBudgetProfile,
    _source_fallback_card,
    paper_budget_profile,
    paper_size_tier,
    split_source,
)
from .reading_v9_models import AuditReport, NarrativeBrief, VerificationReport
from .reading_v9_quality import (
    apply_edit_script,
    duplicate_candidates,
    mechanical_quality_report,
    parse_revision_edits,
    select_edit_cards,
)
from .reading_workflow import ReadingWorkspace, V9_PIPELINE_VERSION

logger = logging.getLogger("feishu_assistant.reading.v9")

PROMPTS_DIR = PROJECT_ROOT / "prompts"
STYLE_FILE = "paper_reading.md"
V9_PROMPT_FILES = (
    "reading_v9_brief.md",
    "reading_v9_author.md",
    "reading_v9_fact_audit.md",
    "reading_v9_reader_audit.md",
    "reading_v9_revision.md",
    "reading_v9_verify.md",
)

# 兼容外部导入的缺省值；正式 V9 每次运行都使用 ``paper_budget_profile``
# 返回的尺寸预算，不再用这些常量给所有论文一刀切。
BRIEF_OUTPUT_TOKENS = 16_384
AUTHOR_OUTPUT_TOKENS = 32_768
AUDIT_OUTPUT_TOKENS = 16_384
REVISION_OUTPUT_TOKENS = 16_384
VERIFY_OUTPUT_TOKENS = 12_288

# 短论文尽量一次写完；超长论文最多三个宏观续写段。这里限制的是单次来源包，
# 不是论文截尾：长文按 Narrative Brief 的 source_refs 回到原文块取材。
FULL_SOURCE_CHARS = 150_000
AUTHOR_PACKET_CHARS = 100_000
FACT_PACKET_CHARS = 145_000
# 长文（paper_size_tier == "long"）来源包分层归并：命中块全文 + 未命中块证据卡
# 摘要，预算放宽。估算宁大勿小（用户裁定）：k3 为 1M token 窗口，25 万字符
# ≈ 15 万 token，余量充足。
AUTHOR_PACKET_CHARS_LONG = 320_000
FACT_PACKET_CHARS_LONG = 340_000
MAX_AUTHOR_PARTS = 4
MAX_EDIT_RATIO = 0.20
# 一次修订即使没有超过比例，也不能演变成“几十处小改”的隐性重写。
# 这是总批次预算，不是每篇论文必须用满的额度。
MAX_EDIT_CHARS = 6_000
MAX_FACT_EDIT_CARDS = 3
MAX_READER_EDIT_CARDS = 2
MAX_TOTAL_EDIT_CARDS = 5

_PLANNING_JARGON = (
    "debt_id", "must_explain", "success_test", "root_scope",
    "scope_violation", "合同缺口", "有限判断", "证据支持什么",
)


def _severe_mechanical_findings(report: Any) -> list[str]:
    """把少数明显会破坏交付的机械信号升级为草稿状态。

    普通告警不冒充阅读质量；这里只拦截 V8 已经实证的结构性坏稿：标题数量
    明显失控、内部合同黑话大量进入正文和未完成占位/资产错误。
    """
    findings = list(getattr(report, "errors", ()) or ())
    reasons = [str(getattr(item, "message", "机械错误")) for item in findings]
    heading_counts = dict(getattr(report, "heading_counts", ()) or ())
    h2 = int(heading_counts.get(2, 0))
    total = sum(int(value) for value in heading_counts.values())
    if h2 > 10:
        reasons.append(f"H2 数量 {h2} 明显超过 V9 上限")
    if total > 24:
        reasons.append(f"标题总数 {total} 明显超过 V9 上限")
    if int(getattr(report, "planning_jargon_count", 0) or 0) > 12:
        reasons.append("规划/审核黑话数量明显过高")
    return reasons


def _edit_scope_allows(text: str, scope: str) -> bool:
    blank_breaks = len(re.findall(r"\n[ \t]*\n", text))
    if scope == "sentence":
        return blank_breaks == 0 and "\n" not in text
    if scope == "paragraph":
        return blank_breaks == 0
    return blank_breaks <= 1


def _replacement_scope_allows(
    old_text: str,
    new_text: str,
    scope: str,
    card_operation: str,
) -> bool:
    """限制补丁的新增文字不能偷偷升级成新小节或跨段重写。"""
    if not _edit_scope_allows(new_text, scope):
        return False
    if re.search(r"(?m)^\s{0,3}#{1,6}[ \t]+\S", new_text):
        return False
    if card_operation == "shorten" and len(new_text) >= len(old_text):
        return False
    # 事实纠正和桥接可适度增加，但不能把一个局部目标放大成新的讲解篇章。
    additions = {"sentence": 120, "paragraph": 600, "adjacent_paragraphs": 1_000}
    ratio = {"sentence": 2.0, "paragraph": 1.6, "adjacent_paragraphs": 1.5}
    absolute_caps = {"sentence": 600, "paragraph": 2_500, "adjacent_paragraphs": 4_000}
    if len(new_text) > absolute_caps.get(scope, 2_500):
        return False
    if len(new_text) > max(
        int(len(old_text) * ratio.get(scope, 1.5)),
        len(old_text) + additions.get(scope, 800),
    ):
        return False
    return True


def _allowed_revision_operations(card_operation: str) -> set[str]:
    return {
        "correct": {"replace", "shorten", "bridge", "delete"},
        "delete": {"delete"},
        "shorten": {"replace", "shorten", "delete"},
        "merge": {"replace", "merge", "delete"},
        "bridge": {"replace", "bridge"},
        "move": {"move", "replace"},
    }.get(card_operation, set())


def _load(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReadingError(f"V9 prompt 模板缺失：{path}（{exc}）") from exc


def _render(template: str, **values: str) -> str:
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    return template


def _json_payload(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ReadingError("V9 模型返回不是 JSON 对象") from exc
        try:
            value = json.loads(text[start:end + 1])
        except json.JSONDecodeError as inner:
            raise ReadingError("V9 模型返回不是合法 JSON 对象") from inner
    if not isinstance(value, dict):
        raise ReadingError("V9 模型返回必须是 JSON 对象")
    return value


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _digest_payload(value: Any) -> str:
    """为阶段缓存生成稳定摘要；不把模型自报分数带入缓存协议。"""
    payload = json.dumps(_plain(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _digest_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _status(value: Any, default: str = "unavailable") -> str:
    if isinstance(value, dict):
        return str(value.get("status") or default)
    return str(getattr(value, "status", default) or default)


def _issues(value: Any) -> list[Any]:
    if isinstance(value, dict):
        raw = value.get("issues") or []
    else:
        raw = getattr(value, "issues", []) or []
    return list(raw) if isinstance(raw, (list, tuple)) else []


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _safe_chat(client: Any, messages: list[dict[str, Any]], **kwargs: Any) -> str:
    """真实客户端接收完整参数；测试 fake 接口较窄时退回公共子集。"""
    try:
        return client.chat(messages, **kwargs)
    except TypeError as exc:
        # 只为测试 fake/旧适配器缺少关键字参数时降级；模型请求已经开始后
        # 抛出的业务 TypeError 不得盲目再发一次，避免重复计费。
        message = str(exc).casefold()
        if not any(token in message for token in (
            "unexpected keyword", "got an unexpected keyword",
            "positional argument", "keyword-only argument",
        )):
            raise
        minimal = {"heavy": bool(kwargs.get("heavy", False))}
        if kwargs.get("json_mode") is not None:
            minimal["json_mode"] = bool(kwargs.get("json_mode"))
        try:
            return client.chat(messages, **minimal)
        except TypeError:
            return client.chat(messages, heavy=minimal["heavy"])


class _StrictJSONCallError(ReadingError):
    """严格 JSON 调用在有界纠错后仍失败，并携带真实调用次数。"""

    def __init__(self, message: str, *, calls: int) -> None:
        super().__init__(message)
        self.calls = calls


def _strict_json_correction_prompt(
    task_heading: str,
    raw: str,
    reason: str,
    correction_rules: str = "",
) -> str:
    """为 Brief/事实审查/读者审查生成同一套带原因格式纠正提示。"""
    rules = str(correction_rules or "").strip()
    return (
        f"{task_heading}\n\n"
        "## 仅纠正上一条 JSON\n\n"
        "上一条输出未通过程序的严格结构校验。请只修正 JSON 结构、缺失字段或非法字段值，"
        "不要重新分析来源/正文，不要新增或删除语义条目，不要改变原有判断；然后只返回一份"
        "符合原任务 schema 的 JSON 对象，不要 Markdown 代码围栏。\n\n"
        f"程序校验原因：{reason[:500]}\n\n"
        + (f"本任务额外规则：{rules}\n\n" if rules else "")
        + "上一条原始输出：\n"
        + str(raw or "")[:60000]
    )


def _chat_strict_json_with_correction(
    client: Any,
    prompt: str,
    *,
    task_heading: str,
    parser: Callable[[dict[str, Any]], Any],
    correction_rules: str = "",
    max_tokens: int,
    reasoning_effort: str = "high",
    attempt_recorder: Callable[[int, str, str], None] | None = None,
) -> tuple[Any, int]:
    """执行一次严格 JSON 调用；仅解析失败时带精确原因纠错一次。"""
    calls = 1  # 请求一旦发起即计数；失败不能伪装成 0 次。
    try:
        raw = _safe_chat(
            client, [{"role": "user", "content": prompt}],
            heavy=True, json_mode=True, reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
        )
    except (KimiError, ReadingError, TypeError) as exc:
        if attempt_recorder:
            attempt_recorder(1, "", str(exc))
        raise _StrictJSONCallError(
            f"首次调用失败：{str(exc)[:240]}", calls=calls,
        ) from exc

    if attempt_recorder:
        attempt_recorder(1, raw, "")
    try:
        return parser(_json_payload(raw)), calls
    except (ReadingError, ValueError, TypeError, KeyError) as first_error:
        if attempt_recorder:
            attempt_recorder(1, raw, str(first_error))
        # 绑定到独立变量；Python 会在离开 except 后清理异常名。
        parse_error = first_error
        correction_prompt = _strict_json_correction_prompt(
            task_heading, raw, str(parse_error), correction_rules,
        )

    calls += 1
    try:
        corrected_raw = _safe_chat(
            client, [{"role": "user", "content": correction_prompt}],
            heavy=True, json_mode=True, reasoning_effort="low",
            max_tokens=max_tokens,
        )
        if attempt_recorder:
            attempt_recorder(2, corrected_raw, "")
        parsed = parser(_json_payload(corrected_raw))
    except (KimiError, ReadingError, ValueError, TypeError, KeyError) as second_error:
        if attempt_recorder:
            attempt_recorder(
                2,
                corrected_raw if "corrected_raw" in locals() else "",
                str(second_error),
            )
        raise _StrictJSONCallError(
            "首次解析失败："
            f"{str(parse_error)[:180]}；格式纠正重试失败：{str(second_error)[:180]}",
            calls=calls,
        ) from second_error
    logger.info("V9 %s 首次解析失败，格式纠正重试成功", task_heading.lstrip("# "))
    return parsed, calls


def _audit_unavailable(kind: str, reason: str) -> AuditReport:
    maker = getattr(AuditReport, "unavailable", None)
    if callable(maker):
        try:
            return maker(kind=kind, reason=reason)
        except TypeError:
            return maker(kind, reason)
    payload = {"status": "unavailable", "issues": [], "reason": reason}
    return AuditReport.from_dict(payload, kind=kind)


def _normalize_audit_payload(payload: dict[str, Any], kind: str) -> dict[str, Any]:
    """在严格校验前修复不影响语义的读者审查字段。

    读者审查的 ``keep``/``span_b`` 只对 duplication 有意义。模型偶尔会把
    ``keep=a`` 一并复制到 mechanical_tone、missing_bridge 等 issue；这不是
    一条需要重新判断的读者诊断，程序可以按 issue kind 确定性地清空它，避免
    一个字段错误让整份审查报告丢失。其它字段仍交给严格 schema 校验，不能猜。
    """
    if not isinstance(payload, dict):
        return payload
    normalized = dict(payload)
    raw_issues = payload.get("issues")
    if not isinstance(raw_issues, list):
        return normalized
    issues: list[Any] = []
    for raw_issue in raw_issues:
        if isinstance(raw_issue, dict):
            issue = dict(raw_issue)
            if kind == "reader" and issue.get("kind") != "duplication":
                # 对非 duplication，缺失/错误值都可机械归一；span_b 同理。
                issue["keep"] = "none"
                issue["span_b"] = ""
            issues.append(issue)
        else:
            issues.append(raw_issue)
    normalized["issues"] = issues
    return normalized


def _parse_audit(payload: dict[str, Any], kind: str, body: str) -> AuditReport:
    payload = _normalize_audit_payload(payload, kind)
    try:
        return AuditReport.from_dict(payload, kind=kind, body=body)
    except TypeError:
        try:
            return AuditReport.from_dict(payload, kind=kind)
        except TypeError:
            return AuditReport.from_dict(payload)


def _parse_brief(
    payload: dict[str, Any],
    *,
    meta_hint: dict[str, str],
    source_titles: list[str],
) -> NarrativeBrief:
    # 参数保留用于兼容既有测试/调用；fallback 只能由 _brief 在共享纠错预算耗尽后创建。
    del meta_hint, source_titles
    return NarrativeBrief.from_dict(payload)


def _brief_dict(brief: NarrativeBrief) -> dict[str, Any]:
    value = _plain(brief)
    return value if isinstance(value, dict) else {}


def _apply_brief_budget_floor(
    brief: NarrativeBrief,
    budget: PaperBudgetProfile,
) -> NarrativeBrief:
    """模型可把目标写得更长，但不能把尺寸预算给出的推荐值压低。"""
    payload = _brief_dict(brief)
    target = dict(payload.get("target") or {})
    declared = int(target.get("target_chars") or 0)
    target["target_chars"] = max(declared, budget.recommended_draft_chars)
    payload["target"] = target
    return NarrativeBrief.from_dict(payload)


def _budget_guidance(budget: PaperBudgetProfile) -> str:
    return (
        f"本论文原文约 {budget.paper_chars:,} 字符，预算档位为 {budget.tier}。"
        f"完整精读建议约 {budget.recommended_draft_chars:,} 字符，"
        f"正文硬下限 {budget.minimum_draft_chars:,} 字符。"
        "target_chars 可以高于建议值，但不得低于建议值；"
        "不要因为原文很长、来源包含摘要层，就把精读退化成论文摘要。"
    )


def _brief_degraded_reason(brief: NarrativeBrief) -> str:
    """识别“调用看似成功、内容却是保守兜底简报”的状态。

    仅看 status 文件不够：首跑曾出现 status=complete，但 tldr 明确写着
    “叙事简报不可用”，且所有 homes/source_refs 都为空。这个判断不把
    “没有图表”误判成失败，只拦截明确的兜底标记。
    """
    tldr = str(_field(brief, "tldr", "") or "").strip()
    if "叙事简报不可用" in tldr:
        return "Brief 内容带有不可用兜底标记"
    return ""


def _public_brief_tldr(brief: NarrativeBrief) -> str:
    """只暴露真实内容摘要；兜底状态文本永不进入正文或结果卡。"""
    if _brief_degraded_reason(brief):
        return ""
    return str(_field(brief, "tldr", "") or "").strip()


def _brief_takeaway_checklist(brief: NarrativeBrief) -> list[dict[str, str]]:
    """只向读者审查暴露语义落点，不暴露完整 Brief/来源合同。"""
    checklist: list[dict[str, str]] = []
    for beat in _beat_dicts(brief):
        takeaway = str(beat.get("reader_takeaway") or "").strip()
        if not takeaway:
            continue
        checklist.append({
            "beat_id": str(beat.get("id") or ""),
            "title": str(beat.get("title") or ""),
            "length": str(beat.get("length") or "medium"),
            "reader_takeaway": takeaway,
        })
    return checklist


def _beat_dicts(brief: NarrativeBrief) -> list[dict[str, Any]]:
    raw = _field(brief, "beats", []) or []
    return [dict(_plain(item)) for item in raw if isinstance(_plain(item), dict)]


def _title_from_brief(brief: NarrativeBrief, meta_hint: dict[str, str]) -> str:
    return str(
        _field(brief, "title_zh", "")
        or meta_hint.get("title_zh")
        or _field(brief, "title_en", "")
        or meta_hint.get("title_en")
        or "未命名论文"
    ).strip()


def _split_groups(items: list[dict[str, Any]], count: int) -> list[list[dict[str, Any]]]:
    if not items:
        return [[]]
    count = max(1, min(count, len(items), MAX_AUTHOR_PARTS))
    size = math.ceil(len(items) / count)
    return [items[i:i + size] for i in range(0, len(items), size)]


def _marker_labels(body: str, kind: str) -> set[str]:
    return {m.group(1).strip() for m in re.finditer(rf"\[\[{kind}:([^\]]+)\]\]", body)}


def _asset_manifest(items: Iterable[Any] | None) -> list[dict[str, Any]]:
    """生成轻量资产指纹；不把图片二进制塞进 workflow manifest。"""
    manifest: list[dict[str, Any]] = []
    for item in items or []:
        label = str(getattr(item, "label", "") or (item.get("label") if isinstance(item, dict) else ""))
        caption = str(getattr(item, "caption", "") or (item.get("caption") if isinstance(item, dict) else ""))
        path_value = getattr(item, "img_path", "") or (item.get("img_path") if isinstance(item, dict) else "")
        path = Path(str(path_value)) if path_value else None
        stat: dict[str, Any] = {}
        if path is not None:
            try:
                info = path.stat()
                stat = {"size": info.st_size, "mtime_ns": info.st_mtime_ns}
            except OSError:
                stat = {"missing": True}
        manifest.append({"label": label, "caption": caption, "path": str(path or ""), **stat})
    return manifest


def _input_signature(
    paper_md: str,
    figures: list[FigureAsset],
    formulas: list[FormulaAsset],
    meta_hint: dict[str, str],
    arxiv_id: str,
    route_manifest: dict[str, Any],
) -> str:
    payload = {
        "paper_sha256": hashlib.sha256(paper_md.encode("utf-8")).hexdigest(),
        "figures": _asset_manifest(figures),
        "formulas": _asset_manifest(formulas),
        "meta_hint": meta_hint,
        "arxiv_id": arxiv_id,
        "route": route_manifest,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]


def _norm_ref(value: str) -> str:
    return re.sub(r"[^a-z0-9一-鿿]+", "", str(value or "").casefold())


def _resolve_source_refs(
    bundle: EvidenceBundle,
    raw_refs: Iterable[Any],
) -> tuple[list[str], list[str]]:
    """同时支持 chunk_id 与章节标题；未知引用进入诊断而不是静默丢掉。"""
    chunk_ids = set(bundle.chunk_map)
    title_pairs = [(_norm_ref(chunk.title), chunk.chunk_id) for chunk in bundle.chunks]
    refs: list[str] = []
    unknown: list[str] = []
    for raw in raw_refs:
        ref = str(raw or "").strip()
        if not ref:
            continue
        if ref in chunk_ids:
            refs.append(ref)
            continue
        normalized = _norm_ref(ref)
        matches = [
            chunk_id for title_norm, chunk_id in title_pairs
            if normalized and (normalized == title_norm
                               or normalized in title_norm
                               or title_norm in normalized)
        ]
        if matches:
            refs.extend(matches)
        else:
            unknown.append(ref)
    return list(dict.fromkeys(refs)), list(dict.fromkeys(unknown))


def _sanitize_unknown_markers(
    body: str,
    figures: list[FigureAsset],
    formulas: list[FormulaAsset],
) -> tuple[str, list[str]]:
    """只做确定性的资产标签防护，避免 strict_images 因模型编造标签导致发布失败。"""
    figure_labels = {asset.label for asset in figures}
    formula_labels = {asset.label for asset in formulas}
    notes: list[str] = []

    def replace_figure(match: re.Match[str]) -> str:
        label = match.group(1).strip()
        if label in figure_labels:
            return f"[[FIG:{label}]]"
        notes.append(f"删除不存在的图表标记：{label}")
        return f"**{label}**"

    def replace_formula(match: re.Match[str]) -> str:
        label = match.group(1).strip()
        if label in formula_labels:
            return f"[[MATH:{label}]]"
        notes.append(f"删除不存在的公式标记：{label}")
        return f"**{label}**"

    body = re.sub(r"\[\[FIG:([^\]]+)\]\]", replace_figure, body)
    body = re.sub(r"\[\[MATH:([^\]]+)\]\]", replace_formula, body)
    return body, notes


def _release_banner(status: str, detail: str) -> str:
    titles = {
        "fact_review_unavailable": "事实审查未完成",
        "reader_review_unavailable": "真实读者审查未完成",
        "revision_rolled_back": "修订未通过验证，已回滚作者初稿",
        "draft_rejected": "核心事实问题尚未解决，本稿仅作草稿",
        "author_draft": "本稿为作者初稿",
    }
    title = titles.get(status, "本稿尚未完成全部发布检查")
    clean = re.sub(r"\s+", " ", detail).strip()[:300]
    suffix = f"：{clean}" if clean else "。"
    return f"> ⚠️ **{title}**{suffix}\n"


class V9ReadingEngine:
    """正式 V9 精读引擎。V8 ReadingEngine 只保留作历史回归与人工回退。"""

    workspace_version = V9_PIPELINE_VERSION

    def __init__(self, kimi: KimiClient | Any, prompts_dir: Path = PROMPTS_DIR) -> None:
        self.kimi = kimi
        self.prompts_dir = prompts_dir
        self._fingerprint_cache: str | None = None
        # PaperReadingService 会并发跑不同论文；覆盖问题和任务冻结路由必须按线程隔离，
        # 否则一篇长文的降级状态可能串到另一篇短文。
        self._run_state = threading.local()
        self._last_classifier = kimi
        self._source_coverage_issues = []
        self._active_budget = paper_budget_profile(0)

    @property
    def _last_classifier(self) -> Any:
        return getattr(self._run_state, "last_classifier", self.kimi)

    @_last_classifier.setter
    def _last_classifier(self, value: Any) -> None:
        self._run_state.last_classifier = value

    @property
    def _source_coverage_issues(self) -> list[str]:
        issues = getattr(self._run_state, "source_coverage_issues", None)
        if issues is None:
            issues = []
            self._run_state.source_coverage_issues = issues
        return issues

    @_source_coverage_issues.setter
    def _source_coverage_issues(self, value: list[str]) -> None:
        self._run_state.source_coverage_issues = list(value)

    @property
    def _active_budget(self) -> PaperBudgetProfile:
        return getattr(self._run_state, "active_budget", paper_budget_profile(0))

    @_active_budget.setter
    def _active_budget(self, value: PaperBudgetProfile) -> None:
        self._run_state.active_budget = value

    def workflow_fingerprint(self) -> str:
        if self._fingerprint_cache is not None:
            return self._fingerprint_cache
        digest = hashlib.sha256(self.workspace_version.encode())
        for name in (STYLE_FILE, *V9_PROMPT_FILES, "reading_evidence.md"):
            path = self.prompts_dir / name
            digest.update(name.encode())
            digest.update(path.read_bytes())
        for path in (
            Path(__file__),
            Path(__file__).with_name("reading_v9_models.py"),
            Path(__file__).with_name("reading_v9_quality.py"),
            Path(__file__).with_name("reading_evidence.py"),
        ):
            digest.update(path.read_bytes())
        self._fingerprint_cache = digest.hexdigest()[:20]
        return self._fingerprint_cache

    def _clients(self) -> tuple[Any, Any, Any, dict[str, Any]]:
        if isinstance(self.kimi, KimiClient):
            routes = self.kimi.frozen_reading()
            route_manifest = routes.route_manifest()
            # Brief 和正文必须使用同一 Author；两个专项审查使用 reviewer。
            self._last_classifier = routes.reviewer
            return routes.writer, routes.reviewer, routes.writer, route_manifest
        route_manifest = {
            "heavy": [f"fake:{type(self.kimi).__module__}.{type(self.kimi).__qualname__}"],
            "light": [],
        }
        self._last_classifier = self.kimi
        return self.kimi, self.kimi, self.kimi, route_manifest

    def classification_client(self) -> Any:
        """给 Paper 服务复用同一任务冻结的审查路由，避免中途热切换。"""
        return self._last_classifier

    def _style(self) -> str:
        return _load(self.prompts_dir / STYLE_FILE)

    def _evidence(
        self,
        client: Any,
        paper_md: str,
        workspace: ReadingWorkspace | None,
        progress: Callable[[str], None],
    ) -> tuple[EvidenceBundle, str, int, str]:
        try:
            bundle, calls = EvidenceIndexer(client, self.prompts_dir).build(
                paper_md, workspace=workspace, progress=progress,
            )
            fallback_cards = sum(
                str(getattr(card, "orientation", "")).startswith("原文导航摘录：")
                for card in bundle.cards
            )
            degraded_chunks = dict(getattr(bundle, "degraded_chunks", {}) or {})
            if fallback_cards and not degraded_chunks:
                degraded_chunks = {
                    str(getattr(card, "chunk_id", "")): "确定性原文导航卡"
                    for card in bundle.cards
                    if str(getattr(card, "orientation", "")).startswith("原文导航摘录：")
                }
            if degraded_chunks:
                detail = "；".join(
                    f"{chunk_id}: {reason[:140]}"
                    for chunk_id, reason in list(degraded_chunks.items())[:6]
                )
                evidence_status = f"degraded:{len(degraded_chunks)} 个证据块降级；{detail}"
            else:
                evidence_status = "complete"
            planner_budget = self._active_budget.source_budget("planner")
            return (
                bundle,
                bundle.planner_material(paper_md, max_chars=planner_budget),
                calls,
                evidence_status,
            )
        except (EvidenceError, KimiError) as exc:
            logger.warning("V9 证据索引不可用，退回确定性导航卡：%s", exc)
            chunks = split_source(paper_md)
            fallback_reason = str(exc)[:300]
            bundle = EvidenceBundle(
                chunks=chunks,
                cards=[_source_fallback_card(chunk) for chunk in chunks],
                long_mode=paper_size_tier(len(paper_md)) == "long",
                degraded_chunks={chunk.chunk_id: fallback_reason for chunk in chunks},
                index_metrics={
                    "workers": 0,
                    "chunks": len(chunks),
                    "calls": 1,
                    "elapsed_seconds": 0,
                    "max_inflight_calls": 0,
                    "degraded_chunks": len(chunks),
                    "catastrophic": True,
                },
            )
            if not bundle.long_mode:
                material = paper_md
            else:
                try:
                    material = bundle.planner_material(
                        paper_md,
                        max_chars=self._active_budget.source_budget("planner"),
                    )
                except EvidenceError:
                    material = "\n".join(
                        f"--- SOURCE {chunk.chunk_id} · {chunk.title} ---\n{chunk.text[:1800]}"
                        for chunk in chunks
                    )
            # 失败前至少尝试过一次来源调用；精确消耗未知时按“至少 1 次”记账，
            # 不再把实际请求伪装成 0 次。
            return bundle, material, 1, f"degraded:{fallback_reason}"

    def _brief(
        self,
        author: Any,
        planner_material: str,
        figures: list[FigureAsset],
        formulas: list[FormulaAsset],
        meta_hint: dict[str, str],
        bundle: EvidenceBundle,
        workspace: ReadingWorkspace | None,
    ) -> tuple[NarrativeBrief, int, str]:
        cached = workspace.get_json("narrative-brief") if workspace else None
        cached_state = workspace.get_json("narrative-brief-status") if workspace else None
        previous_digest = None
        if workspace:
            previous_manifest = workspace.get_json("narrative-brief-manifest")
            if isinstance(previous_manifest, dict):
                previous_digest = str(previous_manifest.get("brief_sha256") or "") or None
        if isinstance(cached, dict):
            try:
                # 缓存必须严格校验；不能让 _parse_brief 的兜底把损坏缓存
                # 静默变成另一份 Brief，再错误复用旧 author-draft。
                parsed = _apply_brief_budget_floor(
                    NarrativeBrief.from_dict(cached), self._active_budget,
                )
                cached_quality_reason = _brief_degraded_reason(parsed)
                cached_quality_status = "degraded" if cached_quality_reason else "complete"
                brief_digest = _digest_payload(_brief_dict(parsed))
                stored_status = (
                    str(cached_state.get("status") or "")
                    if isinstance(cached_state, dict) else ""
                )
                manifest_status = ""
                if workspace:
                    manifest = workspace.get_json("narrative-brief-manifest")
                    if isinstance(manifest, dict):
                        manifest_status = str(manifest.get("status") or "")
                        manifest_digest = str(manifest.get("brief_sha256") or "")
                    else:
                        manifest_digest = ""
                else:
                    manifest_digest = ""
                if workspace and (
                    stored_status != cached_quality_status
                    or manifest_status != cached_quality_status
                    or manifest_digest != brief_digest
                ):
                    # 内容是事实源：兜底 tldr + 空 homes 就是 degraded；正常内容
                    # 即使旧状态误写 degraded，也应纠正为 complete。这里只修协议
                    # 状态，不为追求形式完整而重写已经产出的正文。
                    downstream_manifest = workspace.get_json("author-manifest")
                    downstream_brief_digest = (
                        str(downstream_manifest.get("brief_sha256") or "")
                        if isinstance(downstream_manifest, dict) else ""
                    )
                    if (
                        manifest_digest
                        and manifest_digest != brief_digest
                        and downstream_brief_digest
                        and downstream_brief_digest != brief_digest
                    ):
                        # 只有下游明确绑定了另一版 Brief 才失效；没有下游 manifest
                        # 时交给 _author_draft 的既有 hash 检查，避免无谓删除检查点。
                        workspace.invalidate_from((
                            "author-part", "author-draft", "author-manifest",
                            "mechanical-audit", "duplicate-candidates", "fact-audit",
                            "fact-audit-manifest", "reader-audit", "reader-audit-manifest",
                            "edit-cards", "edit-card-rejections", "revision-script",
                            "revision-manifest", "edited-candidate", "verification",
                            "verification-manifest", "verified-final", "final-body",
                            "final-mechanical-audit", "release-metadata",
                        ))
                    workspace.put_json("narrative-brief-status", {
                        "status": cached_quality_status,
                        "detail": cached_quality_reason or "complete",
                    })
                    workspace.put_json("narrative-brief-manifest", {
                        "brief_sha256": brief_digest,
                        "status": cached_quality_status,
                        **(
                            {"quality_reason": cached_quality_reason}
                            if cached_quality_reason else {}
                        ),
                    })
                if cached_quality_status == "degraded":
                    # 兜底 Brief 不是成功检查点：修复代码或模型恢复后，下一次运行
                    # 必须主动补生成，不能永久复用旧的五步导航。若本次仍失败，
                    # 下面的 fallback 会保留同一份降级内容和真实调用记录。
                    logger.info("V9 缓存 Narrative Brief 为 degraded，本轮自动补生成")
                else:
                    return parsed, 0, cached_quality_status
            except Exception:
                logger.warning("V9 缓存 Narrative Brief 非法，重新生成")

        prompt = _render(
            _load(self.prompts_dir / "reading_v9_brief.md"),
            STYLE=self._style(),
            SOURCE_CONTENT=planner_material,
            FIGURE_INVENTORY=figure_inventory_text(figures),
            FORMULA_INVENTORY=formula_inventory_text(formulas),
            BUDGET_GUIDANCE=_budget_guidance(self._active_budget),
            TARGET_CHARS=str(self._active_budget.recommended_draft_chars),
        )
        calls = 0
        status = "complete"
        quality_reason = ""
        source_titles = [chunk.title for chunk in bundle.chunks]
        try:
            brief, calls = _chat_strict_json_with_correction(
                author,
                prompt,
                task_heading="# 任务：为论文精读建立叙事简报（Narrative Brief）",
                parser=lambda payload: _parse_brief(
                    payload, meta_hint=meta_hint, source_titles=source_titles,
                ),
                correction_rules=(
                    "保留原有 5～7 个 beats 与 source_refs；tldr 应是一句真实内容摘要，"
                    "程序会在 200 字符处确定性截断，不要用工程状态话术替代；"
                    f"target_chars 不得低于 {self._active_budget.recommended_draft_chars}。"
                ),
                max_tokens=self._active_budget.brief_output_tokens,
                reasoning_effort="high",
            )
            brief = _apply_brief_budget_floor(brief, self._active_budget)
            quality_reason = _brief_degraded_reason(brief)
            if quality_reason:
                status = "degraded"
        except _StrictJSONCallError as exc:
            logger.warning(
                "V9 Narrative Brief 在一次格式纠正后仍不可用，使用保守简报：%s", exc,
            )
            fallback = getattr(NarrativeBrief, "fallback", None)
            if not callable(fallback):
                raise ReadingError(f"Narrative Brief 不可用：{exc}") from exc
            title_zh = (
                meta_hint.get("title_zh") or meta_hint.get("title_en")
                or (bundle.chunks[0].title if bundle.chunks else "论文精读")
            )
            brief = fallback(
                str(title_zh)[:120], title_en=str(meta_hint.get("title_en") or "")[:240],
            )
            brief = _apply_brief_budget_floor(brief, self._active_budget)
            calls = exc.calls
            status = "degraded"
            quality_reason = f"Brief 调用或校验失败：{str(exc)[:200]}"
        if workspace:
            brief_payload = _brief_dict(brief)
            brief_digest = _digest_payload(brief_payload)
            # Brief 是正文的上游；它发生变化时，不能继续使用旧的 author-part/
            # author-draft，更不能让旧审查结果挂在新 Brief 上。
            if cached is not None and previous_digest != brief_digest:
                workspace.invalidate_from((
                    "author-part", "author-draft", "author-manifest",
                    "mechanical-audit", "duplicate-candidates", "fact-audit",
                    "fact-audit-manifest", "reader-audit", "reader-audit-manifest",
                    "edit-cards", "edit-card-rejections", "revision-script",
                    "revision-manifest", "edited-candidate", "verification",
                    "verification-manifest", "verified-final", "final-body",
                    "final-mechanical-audit", "release-metadata",
                ))
            workspace.put_json("narrative-brief", brief_payload)
            workspace.put_json("narrative-brief-manifest", {
                "brief_sha256": brief_digest,
                "status": status,
                **({"quality_reason": quality_reason} if quality_reason else {}),
            })
            workspace.put_json("narrative-brief-status", {
                "status": status,
                "detail": quality_reason or status,
            })
        return brief, calls, status

    @staticmethod
    def _part_count(brief: NarrativeBrief, bundle: EvidenceBundle, paper_len: int) -> int:
        target = _field(brief, "target", {}) or {}
        if not isinstance(target, dict):
            target = _plain(target) if isinstance(_plain(target), dict) else {}
        try:
            target_chars = int(target.get("target_chars") or 36_000)
        except (TypeError, ValueError):
            target_chars = 36_000
        budget = paper_budget_profile(paper_len)
        count = budget.author_parts
        # Brief 可以主动把文章规划得更长；只允许它增加分段，不能降低尺寸档位。
        if paper_len > 20_000 and target_chars > budget.recommended_draft_chars * 1.35:
            count += 1
        return max(1, min(count, MAX_AUTHOR_PARTS, len(_beat_dicts(brief)) or 1))

    @staticmethod
    def _group_weight(group: list[dict[str, Any]]) -> float:
        weights = {"short": 1.0, "medium": 1.6, "deep": 2.4}
        return sum(weights.get(str(beat.get("length") or "medium"), 1.6) for beat in group)

    @classmethod
    def _group_length_budget(
        cls,
        groups: list[list[dict[str, Any]]],
        index: int,
        budget: PaperBudgetProfile,
    ) -> tuple[int, int]:
        weights = [cls._group_weight(group) for group in groups]
        total = sum(weights) or 1.0
        weight = weights[index]
        target = max(900, round(budget.recommended_draft_chars * weight / total))
        minimum = max(600, round(budget.minimum_draft_chars * weight / total))
        return target, minimum

    @staticmethod
    def _beat_scope(group: list[dict[str, Any]], index: int, total: int) -> str:
        items = []
        for beat in group:
            items.append(
                f"- {beat.get('id')}: {beat.get('title')}｜问题：{beat.get('question')}｜"
                f"目的：{beat.get('purpose')}｜读者落点：{beat.get('reader_takeaway')}｜"
                f"长度：{beat.get('length', 'medium')}"
            )
        prefix = f"第 {index}/{total} 个连续写作范围。"
        if index > 1:
            prefix += " 已有正文是正式前文，直接承接，不重新开场。"
        return prefix + "\n" + "\n".join(items)

    def _source_for_group(
        self,
        paper_md: str,
        bundle: EvidenceBundle,
        group: list[dict[str, Any]],
        *,
        max_chars: int | None = None,
    ) -> str:
        budget = paper_budget_profile(len(paper_md))
        if paper_size_tier(len(paper_md)) == "long":
            # 长文分层归并：source_refs 命中块给全文，未命中块给证据卡摘要，
            # 任何块都不截尾丢弃；未解析引用只作为提示词参与全文候选排序，
            # 不再触发「退回关键词召回」降级（摘要层保证来源覆盖完整）。
            refs: list[str] = []
            hints: list[str] = []
            for beat in group:
                hints.extend([
                    str(beat.get("title") or ""),
                    str(beat.get("question") or ""),
                    str(beat.get("purpose") or ""),
                ])
                resolved, unknown = _resolve_source_refs(
                    bundle, list(beat.get("source_refs") or []),
                )
                refs.extend(resolved)
                hints.extend(unknown)
                if unknown:
                    logger.info("长文来源包以摘要层兜底未解析 source_ref：%s", unknown[:4])
            packet_info = bundle.layered_source_packet_info(
                list(dict.fromkeys(refs)), hints=[h for h in hints if h],
                max_chars=max_chars or budget.author_packet_chars,
            )
            self._note_source_packet(packet_info, "作者")
            return packet_info.text
        if not bundle.long_mode and len(paper_md) <= FULL_SOURCE_CHARS:
            return paper_md
        refs: list[str] = []
        hints: list[str] = []
        for beat in group:
            hints.extend([
                str(beat.get("title") or ""),
                str(beat.get("question") or ""),
                str(beat.get("purpose") or ""),
            ])
            raw_refs = list(beat.get("source_refs") or [])
            resolved, unknown = _resolve_source_refs(bundle, raw_refs)
            refs.extend(resolved)
            hints.extend(unknown)
            if unknown and bundle.long_mode:
                self._source_coverage_issues.extend(
                    f"未解析 source_ref：{item}" for item in unknown
                )
                # 混合引用时不能进入 strict_refs 模式，否则 unknown 的章节标题
                # 会被静默丢掉；退回关键词召回并保留降级标记更诚实。
                refs = []
                hints.extend(str(item) for item in raw_refs if str(item).strip())
        packet_info = bundle.source_packet_info(
            list(dict.fromkeys(refs)), hints=[h for h in hints if h],
            max_chars=max_chars or budget.author_packet_chars,
        )
        self._note_source_packet(packet_info, "作者")
        return packet_info.text

    def _note_source_packet(self, packet: Any, purpose: str) -> None:
        """把原文包覆盖缺口记入发布状态，不让字符串提示承担状态协议。"""
        if getattr(packet, "used_fallback", False):
            self._source_coverage_issues.append(
                f"{purpose}来源包未能按精确 source_refs 装载，退回关键词召回"
            )
        omitted = tuple(getattr(packet, "omitted_ids", ()) or ())
        if omitted:
            self._source_coverage_issues.append(
                f"{purpose}来源包省略原文块：" + "、".join(str(item) for item in omitted[:6])
            )
        truncated = tuple(getattr(packet, "truncated_ids", ()) or ())
        if truncated:
            self._source_coverage_issues.append(
                f"{purpose}来源包截断原文块：" + "、".join(str(item) for item in truncated[:6])
            )

    def _author_draft(
        self,
        author: Any,
        brief: NarrativeBrief,
        paper_md: str,
        bundle: EvidenceBundle,
        figures: list[FigureAsset],
        formulas: list[FormulaAsset],
        workspace: ReadingWorkspace | None,
        progress: Callable[[str], None],
    ) -> tuple[str, int, list[str]]:
        brief_digest = _digest_payload(_brief_dict(brief))
        source_digest = _digest_payload({
            "paper": _digest_text(paper_md),
            "chunks": [
                {
                    "id": chunk.chunk_id,
                    "digest": chunk.digest or _digest_text(chunk.text),
                }
                for chunk in bundle.chunks
            ],
            "figures": _asset_manifest(figures),
            "formulas": _asset_manifest(formulas),
        })
        expected_manifest = {
            "brief_sha256": brief_digest,
            "source_sha256": source_digest,
        }
        cached = workspace.get_text("author-draft") if workspace else None
        cached_manifest = workspace.get_json("author-manifest") if workspace else None
        manifest_matches = isinstance(cached_manifest, dict) and all(
            cached_manifest.get(key) == value for key, value in expected_manifest.items()
        )
        has_cached_parts = bool(
            workspace and any(workspace.root.glob("author-part-*.md"))
        )
        if workspace and (cached or has_cached_parts) and not manifest_matches:
            # 兼容补丁前生成的工作区：没有 manifest 的正文/续写片段一律重建一次，
            # 避免用旧 Brief/旧来源包静默续跑。
            workspace.invalidate_from((
                "author-part", "author-draft", "author-manifest",
                "mechanical-audit", "duplicate-candidates", "fact-audit",
                "fact-audit-manifest", "reader-audit", "reader-audit-manifest",
                "edit-cards", "edit-card-rejections", "revision-script",
                "revision-manifest", "edited-candidate", "verification",
                "verification-manifest", "verified-final", "final-body",
                "final-mechanical-audit", "release-metadata",
            ))
            cached = None
        if manifest_matches and isinstance(cached_manifest, dict):
            self._source_coverage_issues.extend(
                str(item) for item in (cached_manifest.get("source_coverage_issues") or [])
                if str(item).strip()
            )
        if cached and manifest_matches:
            return cached, 0, []

        beats = _beat_dicts(brief)
        if not beats:
            raise ReadingError("Narrative Brief 没有可写的 beats")
        groups = _split_groups(beats, self._part_count(brief, bundle, len(paper_md)))
        budget = paper_budget_profile(len(paper_md))
        prior = ""
        calls = 0
        if workspace:
            workspace.put_json("author-manifest", {
                **expected_manifest,
                "source_coverage_issues": list(dict.fromkeys(self._source_coverage_issues)),
            })
        for index, group in enumerate(groups, 1):
            key = f"author-part-{index:02d}"
            cached_part = workspace.get_text(key) if workspace else None
            target_chars, minimum_chars = self._group_length_budget(
                groups, index - 1, budget,
            )
            if cached_part:
                part = cached_part
            else:
                progress(f"✍️ V9 单一作者正在写第 {index}/{len(groups)} 个连续部分……")
                brief_json = json.dumps(_brief_dict(brief), ensure_ascii=False, indent=2)
                fixed_chars = (
                    len(prior) + len(brief_json)
                    + len(figure_inventory_text(figures))
                    + len(formula_inventory_text(formulas))
                    + 12_000
                )
                source = self._source_for_group(
                    paper_md,
                    bundle,
                    group,
                    max_chars=budget.source_budget("author", fixed_chars=fixed_chars),
                )
                scope = self._beat_scope(group, index, len(groups))
                length_guidance = (
                    f"本部分建议约 {target_chars:,} 字符，硬下限 {minimum_chars:,} 字符；"
                    f"全文建议约 {budget.recommended_draft_chars:,} 字符，"
                    f"硬下限 {budget.minimum_draft_chars:,} 字符。"
                    "不要求机械凑字数，但低于硬下限通常意味着机制、证据或边界尚未讲透。"
                )
                prompt = _render(
                    _load(self.prompts_dir / "reading_v9_author.md"),
                    STYLE=self._style(),
                    NARRATIVE_BRIEF=brief_json,
                    SOURCE_CONTENT=source,
                    FIGURE_INVENTORY=figure_inventory_text(figures),
                    FORMULA_INVENTORY=formula_inventory_text(formulas),
                    PRIOR_BODY=prior or "（无；这是文章开头。）",
                    BEAT_SCOPE=scope,
                    LENGTH_GUIDANCE=length_guidance,
                )
                try:
                    part = _safe_chat(
                        author, [{"role": "user", "content": prompt}],
                        heavy=True, reasoning_effort="high",
                        max_tokens=budget.author_output_tokens,
                    )
                except KimiError as exc:
                    raise ReadingError(
                        f"V9 作者正文第 {index}/{len(groups)} 部分生成失败；"
                        "已完成部分保留在检查点，下次会继续：" + str(exc)
                    ) from exc
                calls += 1
                part = strip_h1(str(part)).strip()
                if len(part) < minimum_chars:
                    progress(
                        f"🧩 第 {index}/{len(groups)} 部分低于深度下限，"
                        "正在做一次同范围补写……"
                    )
                    continuation_scope = (
                        scope
                        + "\n\n这是同一部分的补写，不要复述或重写已有文字。"
                        f"当前部分约 {len(part):,} 字符，至少还需补足到 {minimum_chars:,} 字符；"
                        "只补尚未讲透的机制链、证据解释、反例或边界。"
                    )
                    continuation_prompt = _render(
                        _load(self.prompts_dir / "reading_v9_author.md"),
                        STYLE=self._style(),
                        NARRATIVE_BRIEF=brief_json,
                        SOURCE_CONTENT=source,
                        FIGURE_INVENTORY=figure_inventory_text(figures),
                        FORMULA_INVENTORY=formula_inventory_text(formulas),
                        PRIOR_BODY=(prior + "\n\n" + part).strip(),
                        BEAT_SCOPE=continuation_scope,
                        LENGTH_GUIDANCE=length_guidance,
                    )
                    try:
                        extension = _safe_chat(
                            author, [{"role": "user", "content": continuation_prompt}],
                            heavy=True, reasoning_effort="high",
                            max_tokens=budget.author_output_tokens,
                        )
                    except KimiError as exc:
                        raise ReadingError(
                            f"V9 作者正文第 {index}/{len(groups)} 部分低于深度下限，"
                            "补写调用失败；已完成部分保留在检查点，下次会继续：" + str(exc)
                        ) from exc
                    calls += 1
                    extension = strip_h1(str(extension)).strip()
                    if extension:
                        part = (part + "\n\n" + extension).strip()
                if len(part) < minimum_chars:
                    raise ReadingError(
                        f"V9 作者正文第 {index}/{len(groups)} 部分过短："
                        f"{len(part)} 字符，尺寸预算下限为 {minimum_chars} 字符；"
                        "为避免把长论文发布成摘要，本次停止发布"
                    )
                if workspace:
                    workspace.put_text(key, part)
                    workspace.put_json("author-manifest", {
                        **expected_manifest,
                        "source_coverage_issues": list(dict.fromkeys(self._source_coverage_issues)),
                    })
            if len(part) < minimum_chars:
                raise ReadingError(
                    f"V9 作者正文缓存第 {index}/{len(groups)} 部分只有 {len(part)} 字符，"
                    f"低于当前尺寸预算下限 {minimum_chars}；请清理旧检查点后重跑"
                )
            prior = (prior + "\n\n" + part).strip()

        if len(prior) < budget.minimum_draft_chars:
            raise ReadingError(
                f"V9 作者正文总长 {len(prior)} 字符，低于 {budget.tier} 档论文下限 "
                f"{budget.minimum_draft_chars} 字符；为避免过度简略，本次停止发布"
            )

        body = _hoist_media_markers(prior)
        body = _strip_inline_latex(body)
        body, marker_notes = _sanitize_unknown_markers(body, figures, formulas)
        title = _title_from_brief(brief, {})
        tldr = _public_brief_tldr(brief)
        head = f"# 《{title}》精读笔记"
        if tldr:
            head += f"\n\n> **一句话理解**：{tldr}"
        body = head + "\n\n" + body.strip() + "\n"
        if workspace:
            workspace.put_text("author-draft", body)
            workspace.put_json("author-manifest", {
                **expected_manifest,
                "source_coverage_issues": list(dict.fromkeys(self._source_coverage_issues)),
            })
        return body, calls, marker_notes

    def _fact_source(
        self,
        paper_md: str,
        bundle: EvidenceBundle,
        brief: NarrativeBrief,
        *,
        fixed_chars: int = 0,
    ) -> str:
        budget = paper_budget_profile(len(paper_md))
        max_chars = budget.source_budget("fact", fixed_chars=fixed_chars)
        if paper_size_tier(len(paper_md)) == "long":
            # 长文事实审查同样走分层归并包：命中块全文 + 未命中块证据卡摘要。
            refs: list[str] = []
            hints: list[str] = []
            for beat in _beat_dicts(brief):
                hints.extend([str(beat.get("title") or ""), str(beat.get("question") or "")])
                resolved, unknown = _resolve_source_refs(
                    bundle, list(beat.get("source_refs") or []),
                )
                refs.extend(resolved)
                hints.extend(unknown)
                if unknown:
                    logger.info("长文事实来源包以摘要层兜底未解析 source_ref：%s", unknown[:4])
            packet_info = bundle.layered_source_packet_info(
                list(dict.fromkeys(refs)), hints=[h for h in hints if h],
                max_chars=max_chars,
            )
            self._note_source_packet(packet_info, "事实审查")
            return packet_info.text
        if len(paper_md) <= max_chars:
            return paper_md
        beats = _beat_dicts(brief)
        refs: list[str] = []
        hints: list[str] = []
        for beat in beats:
            hints.extend([str(beat.get("title") or ""), str(beat.get("question") or "")])
            raw_refs = list(beat.get("source_refs") or [])
            resolved, unknown = _resolve_source_refs(bundle, raw_refs)
            refs.extend(resolved)
            hints.extend(unknown)
            if unknown and bundle.long_mode:
                self._source_coverage_issues.extend(
                    f"未解析 source_ref：{item}" for item in unknown
                )
                refs = []
                hints.extend(str(item) for item in raw_refs if str(item).strip())
        packet_info = bundle.source_packet_info(
            list(dict.fromkeys(refs)), hints=[h for h in hints if h],
            max_chars=max_chars,
        )
        self._note_source_packet(packet_info, "事实审查")
        return packet_info.text

    def _audit(
        self,
        reviewer: Any,
        kind: str,
        body: str,
        source: str,
        figures: list[FigureAsset],
        formulas: list[FormulaAsset],
        candidates: Any,
        workspace: ReadingWorkspace | None,
    ) -> tuple[AuditReport, int]:
        key = f"{kind}-audit"
        cached = workspace.get_json(key) if workspace else None
        expected_manifest = {
            "kind": kind,
            "body_sha256": _digest_text(body),
            "source_sha256": _digest_text(source),
        }
        if kind == "reader":
            # reader_takeaway 是读者审查的新增语义输入；它变化时不能复用旧 reader audit。
            expected_manifest["reader_context_sha256"] = _digest_payload(_plain(candidates))
        cached_manifest = workspace.get_json(f"{kind}-audit-manifest") if workspace else None
        if isinstance(cached, dict):
            try:
                manifest_matches = isinstance(cached_manifest, dict) and all(
                    cached_manifest.get(field) == value
                    for field, value in expected_manifest.items()
                )
                parsed = _parse_audit(cached, kind, body) if manifest_matches else None
                if parsed is not None and _status(parsed) == "complete":
                    return parsed, 0
                if not manifest_matches:
                    logger.info("V9 %s audit 缓存与正文/来源不匹配，本轮重新审查", kind)
                logger.info("V9 %s audit 上次不可用，本轮自动补审", kind)
            except Exception:
                logger.warning("V9 %s audit 缓存非法，重新审查", kind)

        if kind == "fact":
            prompt = _render(
                _load(self.prompts_dir / "reading_v9_fact_audit.md"),
                SOURCE_CONTENT=source,
                FIGURE_INVENTORY=figure_inventory_text(figures),
                FORMULA_INVENTORY=formula_inventory_text(formulas),
                BODY=body,
            )
        else:
            reader_context = _plain(candidates)
            if isinstance(reader_context, dict):
                mechanical_candidates = reader_context.get("duplicate_candidates", [])
                takeaway_checklist = reader_context.get("takeaway_checklist", [])
            else:
                mechanical_candidates = reader_context
                takeaway_checklist = []
            prompt = _render(
                _load(self.prompts_dir / "reading_v9_reader_audit.md"),
                BODY=body,
                CANDIDATES=json.dumps(
                    mechanical_candidates, ensure_ascii=False, indent=2,
                ),
                TAKEAWAY_CHECKLIST=json.dumps(
                    takeaway_checklist, ensure_ascii=False, indent=2,
                ),
            )
        task_heading = (
            "# 任务：论文精读 V9 · 事实与证据审查"
            if kind == "fact"
            else "# 任务：论文精读 V9 · 真实读者审查"
        )
        correction_rules = (
            "事实审查必须保留 status/issues/checked_claims/reason；不要改变问题语义。"
            if kind == "fact"
            else (
                "读者审查必须保留 status/issues/reason；span_a/span_b 必须逐字复制正文中"
                "连续且唯一的原文，不能概括或改写；非 duplication issue 的 keep 必须为 "
                "none、span_b 必须为空；duplication 才需要 keep=a/b 和 span_b。"
            )
        )
        audit_attempts: list[dict[str, Any]] = []

        def record_audit_attempt(attempt: int, raw: str, error: str) -> None:
            entry = {
                "attempt": attempt,
                "error": str(error or "")[:500],
                "raw_output_preview": str(raw or "")[:60_000],
            }
            existing = next(
                (index for index, item in enumerate(audit_attempts)
                 if int(item.get("attempt") or 0) == attempt),
                None,
            )
            if existing is None:
                audit_attempts.append(entry)
            else:
                audit_attempts[existing] = entry
            if workspace and raw:
                workspace.put_text(f"{kind}-audit-raw-{attempt:02d}", str(raw)[:60_000])

        try:
            report, calls = _chat_strict_json_with_correction(
                reviewer,
                prompt,
                task_heading=task_heading,
                parser=lambda payload: _parse_audit(payload, kind, body),
                correction_rules=correction_rules,
                max_tokens=self._active_budget.audit_output_tokens,
                reasoning_effort="high",
                attempt_recorder=record_audit_attempt,
            )
        except _StrictJSONCallError as exc:
            calls = exc.calls
            logger.warning("V9 %s audit 不可用：%s", kind, exc)
            report = _audit_unavailable(kind, str(exc)[:300])

        if workspace:
            report_payload = _plain(report)
            workspace.put_json(key, report_payload)
            workspace.put_json(f"{kind}-audit-manifest", expected_manifest)
            workspace.put_json(f"{kind}-audit-attempts", audit_attempts)
            workspace.put_json(f"{kind}-audit-decision", {
                "schema": "reading-v9-audit-decision-v1",
                "kind": kind,
                "status": _status(report),
                "reason": str(_field(report, "reason", "") or ""),
                "manifest": expected_manifest,
                "attempts": audit_attempts,
                "report": report_payload,
            })
        return report, calls

    @staticmethod
    def _fact_evidence(cards: Iterable[Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for card in cards:
            data = _plain(card)
            if not isinstance(data, dict):
                continue
            if str(data.get("audit_kind") or data.get("kind") or "") == "fact" \
                    or data.get("source_evidence") or data.get("evidence"):
                out.append({
                    "issue_ids": data.get("issue_ids") or [data.get("issue_id")],
                    "draft_quote": (
                        data.get("draft_quote") or data.get("span_a")
                        or data.get("target_quote") or data.get("old_text")
                    ),
                    "source_location": data.get("source_location"),
                    "source_evidence": data.get("source_evidence") or data.get("evidence"),
                    "required_outcome": data.get("required_outcome") or data.get("required_result"),
                })
        return out

    @staticmethod
    def _bind_revision_edits(
        edits: list[Any], cards: list[Any], body: str,
    ) -> tuple[list[Any], list[str]]:
        """把 Writer 补丁绑定回唯一 EditCard，拒绝任意游离修改。"""
        by_target: dict[str, list[Any]] = {}
        for card in cards:
            target = str(_field(card, "target_quote", "") or _field(card, "old_text", ""))
            if target:
                by_target.setdefault(target, []).append(card)
        errors: list[str] = []
        bound: list[Any] = []
        seen_cards: set[str] = set()
        for edit in edits:
            target = str(_field(edit, "old_text", ""))
            matches = by_target.get(target, [])
            if len(matches) != 1:
                errors.append(
                    f"{_field(edit, 'edit_id', '')}: old_text 未绑定唯一编辑卡（匹配 {len(matches)} 张）"
                )
                continue
            card = matches[0]
            card_id = str(_field(card, "card_id", ""))
            if card_id in seen_cards:
                errors.append(f"{_field(edit, 'edit_id', '')}: 同一编辑卡重复提交补丁")
                continue
            card_op = str(_field(card, "operation", ""))
            edit_op = str(_field(edit, "operation", ""))
            if edit_op not in _allowed_revision_operations(card_op):
                errors.append(
                    f"{_field(edit, 'edit_id', '')}: operation={edit_op} 不符合编辑卡 {card_op}"
                )
                continue
            scope = str(_field(card, "max_scope", "paragraph"))
            if not _edit_scope_allows(target, scope):
                errors.append(
                    f"{_field(edit, 'edit_id', '')}: old_text 超出编辑卡范围 {scope}"
                )
                continue
            # reason 不是安全绑定依据，但应至少能追溯到 card 或原始 issue。
            reason = str(_field(edit, "reason", ""))
            issue_ids = [str(item) for item in (_field(card, "issue_ids", []) or [])]
            def mentions(identifier: str) -> bool:
                return bool(identifier) and bool(re.search(
                    rf"(?<![A-Za-z0-9_.:-]){re.escape(identifier)}(?![A-Za-z0-9_.:-])",
                    reason,
                ))
            if not mentions(card_id) and not any(mentions(issue_id) for issue_id in issue_ids):
                errors.append(
                    f"{_field(edit, 'edit_id', '')}: reason 未引用 card_id 或 issue_id"
                )
                continue
            if target not in body:
                errors.append(f"{_field(edit, 'edit_id', '')}: old_text 不在原正文")
                continue
            new_text = str(_field(edit, "new_text", ""))
            if not _replacement_scope_allows(target, new_text, scope, card_op):
                errors.append(
                    f"{_field(edit, 'edit_id', '')}: new_text 超出编辑卡范围 {scope}"
                )
                continue
            seen_cards.add(card_id)
            bound.append(edit)

        if errors:
            return [], errors
        return bound, []

    def _revision(
        self,
        author: Any,
        body: str,
        cards: list[Any],
        figures: list[FigureAsset],
        formulas: list[FormulaAsset],
        workspace: ReadingWorkspace | None,
    ) -> tuple[str, list[Any], list[Any], int, str]:
        if not cards:
            return body, [], [], 0, "not_needed"

        expected_manifest = {
            "body_sha256": _digest_text(body),
            "cards_sha256": _digest_payload([_plain(card) for card in cards]),
        }
        cached_script = workspace.get_json("revision-script") if workspace else None
        cached_manifest = workspace.get_json("revision-manifest") if workspace else None
        raw_attempts: list[dict[str, Any]] = []

        def invalidate_revision_downstream() -> None:
            if workspace:
                # revision-script/revision-attempts/revision-decision 是诊断证据，
                # 只能失效候选稿、验证和发布产物，不能销毁本次判定依据。
                workspace.invalidate_from((
                    "edited-candidate", "verification", "verification-manifest",
                    "verified-final", "final-body", "final-mechanical-audit",
                    "release-metadata",
                ))

        def record_attempt(
            attempt: int,
            *,
            raw: str = "",
            payload: dict[str, Any] | None = None,
            error: str = "",
            source: str = "model",
        ) -> None:
            entry: dict[str, Any] = {
                "attempt": attempt,
                "source": source,
                "error": error[:500],
            }
            if payload is not None:
                entry["payload"] = _plain(payload)
            else:
                entry["raw_output_preview"] = str(raw or "")[:60_000]
                if workspace and raw:
                    workspace.put_text(f"revision-raw-{attempt:02d}", str(raw)[:60_000])
            raw_attempts.append(entry)

        def record_decision(
            status: str,
            reason: str,
            *,
            raw_payload: dict[str, Any] | None = None,
            parsed_edits: Iterable[Any] = (),
            accepted: Iterable[Any] = (),
            rejected: Iterable[Any] = (),
            binding_errors: Iterable[str] = (),
            calls: int,
        ) -> None:
            decision: dict[str, Any] = {
                "schema": "reading-v9-revision-decision-v1",
                "status": status,
                "reason": reason,
                "calls": calls,
                "body_sha256": expected_manifest["body_sha256"],
                "cards_sha256": expected_manifest["cards_sha256"],
                "raw_attempt_count": len(raw_attempts),
                "attempts": raw_attempts,
                "parsed_edit_count": len(list(parsed_edits)),
                "accepted": [_plain(item) for item in accepted],
                "rejected": [_plain(item) for item in rejected],
                "binding_errors": [str(item) for item in binding_errors],
            }
            if raw_payload is not None:
                decision["revision_script"] = _plain(raw_payload)
            if workspace:
                workspace.put_json("revision-attempts", raw_attempts)
                workspace.put_json("revision-decision", decision)
            logger.info(
                "V9 修订判定：status=%s reason=%s calls=%d parsed=%d accepted=%d rejected=%d",
                status, reason[:180], calls, decision["parsed_edit_count"],
                len(decision["accepted"]), len(decision["rejected"]),
            )

        # 与当前正文/编辑卡绑定的非空脚本才可复用。空脚本不是成功检查点，
        # 必须重新请求一次，避免 no_safe_edits 永久缓存。
        raw_payload: dict[str, Any] | None = None
        calls = 0
        if isinstance(cached_script, dict):
            manifest_matches = isinstance(cached_manifest, dict) and all(
                cached_manifest.get(key) == value for key, value in expected_manifest.items()
            )
            cached_empty = isinstance(cached_script.get("edits"), list) and not cached_script.get("edits")
            if manifest_matches and not cached_empty:
                raw_payload = cached_script
                record_attempt(0, payload=cached_script, source="cache")
            else:
                if workspace:
                    workspace.put_json("revision-stale-script", cached_script)
                    workspace.put_json("revision-stale-manifest", cached_manifest or {})
                logger.info(
                    "V9 修订脚本缓存不可复用：manifest_matches=%s cached_empty=%s",
                    manifest_matches, cached_empty,
                )

        prompt = ""
        if raw_payload is None:
            prompt = _render(
                _load(self.prompts_dir / "reading_v9_revision.md"),
                STYLE=self._style(),
                BODY=body,
                EDIT_CARDS=json.dumps([_plain(card) for card in cards], ensure_ascii=False, indent=2),
                FACT_EVIDENCE=json.dumps(self._fact_evidence(cards), ensure_ascii=False, indent=2),
            )
            for attempt in (1, 2):
                try:
                    calls += 1
                    raw = _safe_chat(
                        author, [{"role": "user", "content": prompt}],
                        heavy=True, json_mode=True,
                        reasoning_effort="high" if attempt == 1 else "low",
                        max_tokens=self._active_budget.revision_output_tokens,
                    )
                except (KimiError, ReadingError, TypeError) as exc:
                    record_attempt(attempt, error=str(exc), source="model")
                    record_decision("unavailable", str(exc)[:300], calls=calls)
                    logger.warning("V9 修订调用失败，保留 author-draft：%s", exc)
                    return body, [], [], max(1, calls), f"unavailable:{str(exc)[:160]}"
                try:
                    payload = _json_payload(raw)
                except (ReadingError, ValueError, TypeError) as exc:
                    record_attempt(attempt, raw=raw, error=str(exc), source="model")
                    record_decision("unavailable", str(exc)[:300], calls=calls)
                    logger.warning("V9 修订 JSON 解析失败，保留 author-draft：%s", exc)
                    return body, [], [], calls, f"unavailable:{str(exc)[:160]}"
                raw_payload = payload
                record_attempt(attempt, payload=payload, source="model")
                if workspace:
                    workspace.put_json("revision-script", payload)
                    workspace.put_json("revision-manifest", expected_manifest)
                edits_raw = payload.get("edits")
                if attempt == 1 and isinstance(edits_raw, list) and not edits_raw:
                    prompt = (
                        prompt + "\n\n## 本次必须重试修订\n"
                        "上一条返回了空 edits，但本次存在已接受的编辑卡。请重新逐条检查编辑卡，"
                        "只输出能够绑定到 card_id/issue_id 且 old_text 来自正文的最小补丁；"
                        "不要返回整篇文章。若仍无法安全修改，才返回空 edits。"
                    )
                    continue
                break

        if raw_payload is None:
            record_decision("unavailable", "没有获得修订 JSON", calls=calls)
            return body, [], [], max(1, calls), "unavailable:没有获得修订 JSON"

        parsed_edits: list[Any] = []
        try:
            parsed_edits = list(parse_revision_edits(raw_payload, body=body))
            bound_edits, binding_errors = self._bind_revision_edits(parsed_edits, cards, body)
            if binding_errors:
                rejected_records = [
                    {
                        "edit_id": error.split(":", 1)[0].strip(),
                        "code": "binding_error",
                        "reason": error,
                    }
                    for error in binding_errors
                ]
                record_decision(
                    "rejected", "binding_errors", raw_payload=raw_payload,
                    parsed_edits=parsed_edits, rejected=rejected_records,
                    binding_errors=binding_errors, calls=calls,
                )
                invalidate_revision_downstream()
                logger.warning(
                    "V9 修订补丁绑定拒绝：%d 条，原因=%s",
                    len(binding_errors), "；".join(binding_errors[:4]),
                )
                return body, [], rejected_records, calls, f"rejected:{binding_errors[0][:160]}"
            result = apply_edit_script(
                body,
                bound_edits,
                max_change_ratio=MAX_EDIT_RATIO,
                max_change_chars=MAX_EDIT_CHARS,
            )
        except (ValueError, TypeError) as exc:
            record_decision(
                "rejected", str(exc)[:300], raw_payload=raw_payload,
                parsed_edits=parsed_edits, calls=calls,
            )
            invalidate_revision_downstream()
            logger.warning("V9 修订脚本拒绝，保留 author-draft：%s", exc)
            return body, [], [], calls, f"rejected:{str(exc)[:160]}"

        candidate = _field(result, "body", None) or _field(result, "text", None)
        if not isinstance(candidate, str):
            candidate = result[0] if isinstance(result, tuple) and result else body
        accepted = list(_field(result, "accepted", []) or [])
        rejected = list(_field(result, "rejected", []) or [])
        if isinstance(result, tuple):
            if len(result) > 1:
                accepted = list(result[1] or [])
            if len(result) > 2:
                rejected = list(result[2] or [])
        if candidate == body or not accepted:
            reason = "empty_edits" if not parsed_edits else (
                "all_patches_rejected" if rejected else "candidate_unchanged"
            )
            record_decision(
                "no_safe_edits", reason, raw_payload=raw_payload,
                parsed_edits=parsed_edits, accepted=accepted, rejected=rejected,
                calls=calls,
            )
            invalidate_revision_downstream()
            logger.warning(
                "V9 修订无安全补丁：reason=%s parsed=%d accepted=%d rejected=%d",
                reason, len(parsed_edits), len(accepted), len(rejected),
            )
            return body, [], rejected, calls, f"no_safe_edits:{reason}"

        protected_missing = [
            str(item)
            for card in cards
            for item in (_field(card, "protected_content", []) or [])
            if str(item) and str(item) not in candidate
        ]
        if protected_missing:
            reason = "protected_content_lost"
            record_decision(
                "rejected", reason, raw_payload=raw_payload,
                parsed_edits=parsed_edits, accepted=accepted, rejected=rejected,
                calls=calls,
            )
            invalidate_revision_downstream()
            return body, [], accepted, calls, f"rejected:{reason}"

        jargon_growth = [
            term for term in _PLANNING_JARGON
            if candidate.count(term) > body.count(term)
        ]
        if jargon_growth:
            reason = "planning_jargon_growth:" + ",".join(jargon_growth[:4])
            record_decision(
                "rejected", reason, raw_payload=raw_payload,
                parsed_edits=parsed_edits, accepted=accepted, rejected=rejected,
                calls=calls,
            )
            invalidate_revision_downstream()
            return body, [], accepted + rejected, calls, f"rejected:{reason}"

        # 单调门禁只检查已知硬回归，不用机械分数冒充阅读质量。
        before_mechanical = mechanical_quality_report(body, figures=figures, formulas=formulas)
        after_mechanical = mechanical_quality_report(candidate, figures=figures, formulas=formulas)
        regressions: list[str] = []
        if after_mechanical.placeholder_count > before_mechanical.placeholder_count:
            regressions.append("placeholder_growth")
        if len(after_mechanical.errors) > len(before_mechanical.errors):
            regressions.append("mechanical_error_growth")
        before_headings = sum(count for _, count in before_mechanical.heading_counts)
        after_headings = sum(count for _, count in after_mechanical.heading_counts)
        if after_headings > before_headings:
            regressions.append("heading_growth")
        if len(duplicate_candidates(candidate)) > len(duplicate_candidates(body)):
            regressions.append("duplicate_candidate_growth")
        if after_mechanical.untranslated_english_count > before_mechanical.untranslated_english_count:
            regressions.append("untranslated_english_growth")
        before_overuse = sum(
            item.code in {"figure_reference_overuse", "formula_reference_overuse"}
            for item in before_mechanical.findings
        )
        after_overuse = sum(
            item.code in {"figure_reference_overuse", "formula_reference_overuse"}
            for item in after_mechanical.findings
        )
        if after_overuse > before_overuse:
            regressions.append("media_overuse_growth")
        if regressions:
            reason = "monotonic_gate:" + ",".join(regressions)
            record_decision(
                "rejected", reason, raw_payload=raw_payload,
                parsed_edits=parsed_edits, accepted=accepted, rejected=rejected,
                calls=calls,
            )
            invalidate_revision_downstream()
            return body, [], accepted + rejected, calls, f"rejected:{reason}"
        if workspace:
            workspace.put_text("edited-candidate", candidate)
        record_decision(
            "candidate", "补丁通过确定性门禁，等待定点验证",
            raw_payload=raw_payload, parsed_edits=parsed_edits,
            accepted=accepted, rejected=rejected, calls=calls,
        )
        return candidate, accepted, rejected, calls, "candidate"

    @staticmethod
    def _patch_contexts(
        body: str, candidate: str, edits: Iterable[Any], cards: Iterable[Any] = (),
    ) -> list[dict[str, Any]]:
        card_by_target = {
            str(_field(card, "target_quote", "") or _field(card, "old_text", "")): _plain(card)
            for card in cards
        }
        contexts: list[dict[str, Any]] = []
        for edit in edits:
            data = _plain(edit)
            if not isinstance(data, dict):
                continue
            old = str(data.get("old_text") or "")
            new = str(data.get("new_text") or "")
            pos = body.find(old) if old else -1
            before = body[max(0, pos - 280):pos + len(old) + 280] if pos >= 0 else old
            new_pos = candidate.find(new) if new else max(0, pos)
            after = candidate[max(0, new_pos - 280):new_pos + len(new) + 280]
            contexts.append({
                "edit_id": data.get("edit_id"),
                "operation": data.get("operation"),
                "reason": data.get("reason"),
                "preserve": data.get("preserve") or [],
                "edit_card": card_by_target.get(old, {}),
                "before": before,
                "after": after,
            })
        return contexts

    def _verify(
        self,
        reviewer: Any,
        body: str,
        candidate: str,
        edits: list[Any],
        cards: list[Any],
        workspace: ReadingWorkspace | None,
    ) -> tuple[VerificationReport, int]:
        cached = workspace.get_json("verification") if workspace else None
        expected_manifest = {
            "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "candidate_sha256": hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
            "edit_ids": [str(_field(edit, "edit_id", "")) for edit in edits],
            "edits_sha256": _digest_payload([_plain(edit) for edit in edits]),
            "cards_sha256": _digest_payload([_plain(card) for card in cards]),
        }
        cached_manifest = workspace.get_json("verification-manifest") if workspace else None
        if isinstance(cached, dict):
            try:
                parsed = VerificationReport.from_dict(cached)
                manifest_matches = isinstance(cached_manifest, dict) and all(
                    cached_manifest.get(key) == value
                    for key, value in expected_manifest.items()
                )
                if _status(parsed) == "complete" and manifest_matches:
                    return parsed, 0
                logger.info("V9 verification 缓存与当前补丁不匹配，本轮重新验证")
            except Exception:
                logger.warning("V9 verification 缓存非法，重新验证")

        prompt = _render(
            _load(self.prompts_dir / "reading_v9_verify.md"),
            FACT_EVIDENCE=json.dumps(self._fact_evidence(cards), ensure_ascii=False, indent=2),
            PATCH_CONTEXTS=json.dumps(
                self._patch_contexts(body, candidate, edits, cards),
                ensure_ascii=False, indent=2),
            EDIT_CARDS=json.dumps(
                [_plain(card) for card in cards], ensure_ascii=False, indent=2),
        )
        try:
            raw = _safe_chat(
                reviewer, [{"role": "user", "content": prompt}],
                heavy=True, json_mode=True, reasoning_effort="high",
                max_tokens=self._active_budget.verify_output_tokens,
            )
            payload = _json_payload(raw)
            try:
                result = VerificationReport.from_dict(payload)
            except TypeError:
                result = VerificationReport.from_dict(payload)
            calls = 1
        except (KimiError, ReadingError, ValueError, TypeError) as exc:
            maker = getattr(VerificationReport, "unavailable", None)
            if callable(maker):
                result = maker(reason=str(exc)[:180])
            else:
                result = VerificationReport.from_dict({
                    "status": "unavailable", "results": [], "reason": str(exc)[:180],
                })
            calls = 1
        if workspace:
            verification_payload = _plain(result)
            workspace.put_json("verification", verification_payload)
            workspace.put_json("verification-manifest", expected_manifest)
            workspace.put_json("verification-decision", {
                "schema": "reading-v9-verification-decision-v1",
                "status": _status(result),
                "reason": str(_field(result, "reason", "") or ""),
                "manifest": expected_manifest,
                "results": verification_payload.get("results", [])
                if isinstance(verification_payload, dict) else [],
                "raw_output_saved": bool(raw if "raw" in locals() else False),
            })
            if "raw" in locals() and raw:
                workspace.put_text("verification-raw", str(raw)[:60_000])
        return result, calls

    @staticmethod
    def _accepted_edit_ids(verification: VerificationReport) -> set[str]:
        if _status(verification) != "complete":
            return set()
        raw = _field(verification, "results", []) or []
        accepted: set[str] = set()
        for item in raw:
            data = _plain(item)
            if not isinstance(data, dict):
                continue
            if str(data.get("decision") or "") == "accept" \
                    and bool(data.get("target_fixed")) \
                    and not bool(data.get("new_fact_error")) \
                    and not bool(data.get("protected_content_lost")) \
                    and not bool(data.get("broken_reference")):
                accepted.add(str(data.get("edit_id") or ""))
        return {item for item in accepted if item}

    @staticmethod
    def _deterministic_patch_guard(
        body: str,
        candidate: str,
        edits: Iterable[Any],
        cards: Iterable[Any],
    ) -> tuple[bool, str]:
        """在信任验证模型前，拦住“目标根本没变”的确定性假修复。"""
        card_by_target = {
            str(_field(card, "target_quote", "") or _field(card, "old_text", "")): card
            for card in cards
        }
        for edit in edits:
            old_text = str(_field(edit, "old_text", ""))
            card = card_by_target.get(old_text)
            if card is None:
                return False, f"补丁未找到对应编辑卡：{_field(edit, 'edit_id', '')}"
            operation = str(_field(card, "operation", ""))
            if operation == "delete" and old_text and old_text in candidate:
                return False, f"删除目标仍在候选稿中：{_field(edit, 'edit_id', '')}"
            paired = str(_field(card, "paired_quote", "") or "")
            if paired and old_text in candidate and paired in candidate:
                return False, f"重复对的两个原片段仍同时存在：{_field(edit, 'edit_id', '')}"
            for protected in (_field(card, "protected_content", []) or []):
                if str(protected) and str(protected) not in candidate:
                    return False, f"保护内容丢失：{_field(edit, 'edit_id', '')}"
        return True, ""

    @staticmethod
    def _filter_edits(edits: list[Any], ids: set[str]) -> list[Any]:
        return [edit for edit in edits if str(_field(edit, "edit_id", "")) in ids]

    @staticmethod
    def _unresolved_fact_issues(
        report: AuditReport,
        resolved_issue_keys: set[tuple[str, str]],
        *,
        severities: set[str] | frozenset[str] = frozenset({"P0", "P1"}),
    ) -> list[Any]:
        out = []
        for issue in _issues(report):
            issue_id = str(_field(issue, "issue_id", ""))
            severity = str(_field(issue, "severity", ""))
            if severity in severities and ("fact", issue_id) not in resolved_issue_keys:
                out.append(issue)
        return out

    @staticmethod
    def _resolved_issue_ids(
        cards: list[Any], accepted_edits: list[Any],
    ) -> set[tuple[str, str]]:
        """只按已绑定的编辑卡记账，且给 fact/reader issue_id 加命名空间。"""
        cards_by_target = {
            str(_field(card, "target_quote", "") or _field(card, "old_text", "")): card
            for card in cards
        }
        resolved: set[tuple[str, str]] = set()
        for edit in accepted_edits:
            target = str(_field(edit, "old_text", ""))
            card = cards_by_target.get(target)
            if card is None:
                continue
            audit_kind = str(_field(card, "audit_kind", ""))
            for issue_id in (_field(card, "issue_ids", []) or []):
                issue_id = str(issue_id).strip()
                if audit_kind and issue_id:
                    resolved.add((audit_kind, issue_id))
        return resolved

    @staticmethod
    def _metadata(
        brief: NarrativeBrief,
        meta_hint: dict[str, str],
        *, release_status: str,
        fact_report: AuditReport,
        reader_report: AuditReport,
        mechanical: Any,
        evidence_status: str,
        brief_status: str,
        revision_status: str,
        open_fact: list[Any],
        unresolved_p2_fact: list[Any],
        open_reader: list[Any],
        marker_notes: list[str],
        model_calls: int,
        source_coverage_issues: list[str],
        source_coverage_complete: bool,
    ) -> dict[str, Any]:
        data = dict(meta_hint)
        data.update({
            "title_zh": _field(brief, "title_zh", "") or meta_hint.get("title_zh", ""),
            "title_en": _field(brief, "title_en", "") or meta_hint.get("title_en", ""),
            "tldr": _public_brief_tldr(brief),
            "contributions": list(_field(brief, "contributions", []) or [])[:3],
            "paper_type": _field(brief, "paper_type", "other"),
            "reader_question": _field(brief, "reader_question", ""),
            "paper_turn": _field(brief, "paper_turn", ""),
            "pipeline_version": V9_PIPELINE_VERSION,
            "release_status": release_status,
            "fact_audit_status": _status(fact_report),
            "reader_audit_status": _status(reader_report),
            "evidence_status": evidence_status,
            "brief_status": brief_status,
            "revision_status": revision_status,
            # open_fact_issues 保持发布闸门语义（仅 P0/P1）；所有未消账 P2
            # 单列给正文结果区/任务卡，不能让系统已知事实风险静默消失。
            "open_fact_issues": [_plain(item) for item in open_fact],
            "unresolved_p2_fact_issues": [_plain(item) for item in unresolved_p2_fact],
            "open_reader_issues": [_plain(item) for item in open_reader],
            "mechanical_audit": _plain(mechanical),
            "marker_notes": marker_notes,
            "model_calls": model_calls,
            "source_coverage_complete": source_coverage_complete,
            "source_coverage_issues": list(dict.fromkeys(source_coverage_issues)),
        })
        if release_status != "verified_final":
            reasons: list[str] = []
            if _status(fact_report) != "complete":
                reasons.append("事实审查不可用")
            if _status(reader_report) != "complete":
                reasons.append("真实读者审查不可用")
            if open_fact:
                reasons.append(f"仍有 {len(open_fact)} 条高优先级事实问题")
            if revision_status.startswith(("unavailable", "rejected")):
                reasons.append("修订未采用")
            data["final_review_degraded"] = "；".join(reasons) or release_status
            data["final_review_core_failed"] = bool(
                release_status == "draft_rejected"
                or any(str(_field(item, "severity", "")) == "P0" for item in open_fact)
            )
        return data

    def run(
        self,
        paper_md: str,
        figures: list[FigureAsset] | None = None,
        formulas: list[FormulaAsset] | None = None,
        arxiv_id: str = "",
        meta_hint: dict[str, str] | None = None,
        progress: Callable[[str], None] = lambda _: None,
        workspace: ReadingWorkspace | None = None,
        **_: Any,
    ) -> ReadingResult:
        figures = figures or []
        formulas = formulas or []
        meta_hint = dict(meta_hint or {})
        timer = StageTimer()
        model_calls = 0
        self._source_coverage_issues = []

        if workspace:
            if getattr(workspace, "pipeline_version", self.workspace_version) != self.workspace_version:
                raise ReadingError(
                    f"V9 需要 {self.workspace_version} 工作区，收到 "
                    f"{getattr(workspace, 'pipeline_version', 'unknown')}"
                )
        paper_md = strip_references(paper_md, hard_cap=None)
        if not paper_md.strip():
            raise ReadingError("论文正文为空")
        self._active_budget = paper_budget_profile(len(paper_md))
        logger.info(
            "V9 尺寸预算：tier=%s paper=%d planner=%d author_packet=%d "
            "fact_packet=%d draft_target=%d draft_min=%d parts=%d",
            self._active_budget.tier,
            self._active_budget.paper_chars,
            self._active_budget.planner_chars,
            self._active_budget.author_packet_chars,
            self._active_budget.fact_packet_chars,
            self._active_budget.recommended_draft_chars,
            self._active_budget.minimum_draft_chars,
            self._active_budget.author_parts,
        )
        author, reviewer, revision_author, route_manifest = self._clients()
        if workspace:
            run_signature = self.workflow_fingerprint() + ":" + _input_signature(
                paper_md, figures, formulas, meta_hint, arxiv_id, route_manifest,
            )
            workspace.ensure_signature(run_signature)
            workspace.put_json("route-manifest", route_manifest)
            workspace.put_json("input-manifest", {
                "input_signature": run_signature.rsplit(":", 1)[-1],
                "paper_sha256": hashlib.sha256(paper_md.encode("utf-8")).hexdigest(),
                "figures": _asset_manifest(figures),
                "formulas": _asset_manifest(formulas),
                "arxiv_id": arxiv_id,
                "budget_profile": self._active_budget.to_dict(),
            })

        progress("🧭 V9 正在建立来源导航……")
        bundle, planner_material, calls, evidence_status = self._evidence(
            author, paper_md, workspace, progress,
        )
        model_calls += calls
        timer.mark("evidence")

        progress("🗺️ V9 作者正在建立轻量叙事简报……")
        brief, calls, brief_status = self._brief(
            author, planner_material, figures, formulas, meta_hint, bundle, workspace,
        )
        model_calls += calls
        timer.mark("brief")

        author_body, calls, marker_notes = self._author_draft(
            author, brief, paper_md, bundle, figures, formulas, workspace, progress,
        )
        model_calls += calls
        timer.mark("author-draft")

        mechanical = mechanical_quality_report(author_body, figures=figures, formulas=formulas)
        candidates = duplicate_candidates(author_body)
        if workspace:
            workspace.put_json("mechanical-audit", _plain(mechanical))
            workspace.put_json("duplicate-candidates", _plain(candidates))

        progress("🔎 V9 正在并列执行事实审查与真实读者审查……")
        fact_source = self._fact_source(
            paper_md,
            bundle,
            brief,
            fixed_chars=len(author_body) + 12_000,
        )
        fact_report, calls = self._audit(
            reviewer, "fact", author_body, fact_source,
            figures, formulas, candidates, workspace,
        )
        model_calls += calls
        reader_context = {
            "duplicate_candidates": _plain(candidates),
            "takeaway_checklist": _brief_takeaway_checklist(brief),
        }
        reader_report, calls = self._audit(
            reviewer, "reader", author_body, "",
            figures, formulas, reader_context, workspace,
        )
        model_calls += calls
        timer.mark("audits")

        source_degraded_for_revision = (
            bool(self._source_coverage_issues)
            or evidence_status.startswith("degraded")
        )

        selection = select_edit_cards(
            author_body,
            fact_report,
            reader_report,
            max_fact_cards=MAX_FACT_EDIT_CARDS,
            max_reader_cards=MAX_READER_EDIT_CARDS,
            max_total_cards=MAX_TOTAL_EDIT_CARDS,
        )
        cards = list(selection.cards)
        source_blocked_cards: list[Any] = []
        if source_degraded_for_revision:
            # 证据覆盖不完整时，事实卡不能基于半包证据改正文；但读者卡只依赖
            # 成稿和读者审查，仍可执行。这样单块证据失败不会拖垮整篇的阅读级修订，
            # 最终仍因 source_degraded 保持 author_draft，绝不升级 verified_final。
            source_blocked_cards = [
                card for card in cards
                if str(_field(card, "audit_kind", "")) == "fact"
            ]
            cards = [
                card for card in cards
                if str(_field(card, "audit_kind", "")) == "reader"
            ]
        else:
            # 两次审查彼此解耦：哪一侧 complete，哪一侧的 EditCard 才能进入
            # 有限修订；另一侧 unavailable 只阻断对应类别，不浪费已经完成的
            # 事实核查成果。
            if _status(fact_report) != "complete":
                cards = [
                    card for card in cards
                    if str(_field(card, "audit_kind", "")) != "fact"
                ]
            if _status(reader_report) != "complete":
                cards = [
                    card for card in cards
                    if str(_field(card, "audit_kind", "")) != "reader"
                ]
        persisted_rejections = [_plain(item) for item in selection.rejected]
        persisted_rejections.extend({
            "audit_kind": str(_field(card, "audit_kind", "fact")),
            "issue_id": str((_field(card, "issue_ids", []) or [""])[0]),
            "code": "source_coverage_degraded",
            "reason": "事实来源包存在降级块，本轮不基于不完整来源自动改事实",
        } for card in source_blocked_cards)
        if workspace:
            workspace.put_json("edit-cards", [_plain(card) for card in cards])
            workspace.put_json("edit-card-rejections", persisted_rejections)

        if cards:
            progress("🛠️ V9 正在执行一次有限修订，不重写整章……")
        candidate, applied_edits, rejected_edits, calls, revision_status = self._revision(
            revision_author, author_body, cards, figures, formulas, workspace,
        )
        model_calls += calls
        timer.mark("revision")

        accepted_edits: list[Any] = []
        verification: VerificationReport | None = None
        final_body = author_body
        if applied_edits and candidate != author_body:
            progress("✅ V9 正在验证实际改动，不重审全文……")
            verification, calls = self._verify(
                reviewer, author_body, candidate, applied_edits, cards, workspace,
            )
            model_calls += calls
            accepted_ids = self._accepted_edit_ids(verification)
            accepted_edits = self._filter_edits(applied_edits, accepted_ids)
            if accepted_edits:
                try:
                    accepted_result = apply_edit_script(
                        author_body,
                        accepted_edits,
                        max_change_ratio=MAX_EDIT_RATIO,
                        max_change_chars=MAX_EDIT_CHARS,
                    )
                    final_body = (
                        _field(accepted_result, "body", None)
                        or _field(accepted_result, "text", None)
                        or (accepted_result[0] if isinstance(accepted_result, tuple) else author_body)
                    )
                    guard_ok, guard_reason = self._deterministic_patch_guard(
                        author_body, final_body, accepted_edits, cards,
                    )
                    if not guard_ok:
                        final_body = author_body
                        accepted_edits = []
                        revision_status = "verification_rejected:" + guard_reason[:160]
                    else:
                        revision_status = "verified"
                except (ValueError, TypeError):
                    final_body = author_body
                    accepted_edits = []
                    revision_status = "verification_apply_failed"
            else:
                final_body = author_body
                revision_status = (
                    "verification_unavailable"
                    if _status(verification) != "complete" else "verification_rejected"
                )
        timer.mark("verification")

        resolved_issue_keys = self._resolved_issue_ids(cards, accepted_edits)
        open_fact = self._unresolved_fact_issues(fact_report, resolved_issue_keys)
        unresolved_p2_fact = self._unresolved_fact_issues(
            fact_report, resolved_issue_keys, severities=frozenset({"P2"}),
        )
        open_reader = [
            item for item in _issues(reader_report)
            if ("reader", str(_field(item, "issue_id", ""))) not in resolved_issue_keys
            and str(_field(item, "severity", "")) == "P1"
        ]
        final_mechanical = mechanical_quality_report(
            final_body, figures=figures, formulas=formulas,
        )
        severe_mechanical = _severe_mechanical_findings(final_mechanical)
        if severe_mechanical and final_body != author_body:
            # 最终机械硬错误不能只改状态；候选稿与 release_status 必须一致，
            # 否则会出现“标成 author-draft，正文却是未回滚 candidate”。
            final_body = author_body
            accepted_edits = []
            revision_status = "mechanical_rollback"
            final_mechanical = mechanical
            severe_mechanical = _severe_mechanical_findings(final_mechanical)
        source_coverage_issues = list(dict.fromkeys(self._source_coverage_issues))
        source_degraded = source_degraded_for_revision or bool(source_coverage_issues)
        if workspace:
            workspace.put_json("final-mechanical-audit", _plain(final_mechanical))

        if _status(fact_report) != "complete":
            release_status = "fact_review_unavailable"
        elif any(str(_field(item, "severity", "")) == "P0" for item in open_fact):
            release_status = "draft_rejected"
        elif _status(reader_report) != "complete":
            release_status = "reader_review_unavailable"
        elif source_degraded:
            release_status = "author_draft"
        elif open_fact:
            release_status = "author_draft"
        elif severe_mechanical:
            release_status = "author_draft"
        elif cards and revision_status != "verified" and not accepted_edits:
            release_status = "revision_rolled_back"
        elif open_reader:
            release_status = "author_draft"
        else:
            release_status = "verified_final"

        detail = ""
        if release_status == "fact_review_unavailable":
            detail = str(_field(fact_report, "reason", "") or "事实审查服务不可用")
        elif release_status == "reader_review_unavailable":
            detail = str(_field(reader_report, "reason", "") or "真实读者审查服务不可用")
            if open_fact:
                detail += f"；另有 {len(open_fact)} 条高优先级事实问题未自动修改"
        elif release_status == "draft_rejected":
            detail = f"仍有 {len(open_fact)} 条高优先级事实问题"
        elif release_status == "author_draft":
            if open_fact:
                detail = f"仍有 {len(open_fact)} 条高优先级事实问题"
            elif source_degraded:
                detail = "；".join(source_coverage_issues[:3]) or "来源导航或叙事简报处于降级状态"
            elif severe_mechanical:
                detail = "；".join(severe_mechanical[:3])
            elif open_reader:
                detail = f"仍有 {len(open_reader)} 条高优先级读者问题"
        elif release_status == "revision_rolled_back":
            if revision_status.startswith("no_safe_edits"):
                detail = "修订模型未产出可安全应用的补丁；判定依据已保存在修订记录中"
            elif revision_status.startswith("rejected"):
                detail = "修订脚本触发安全护栏并已回滚；逐项原因已保存在修订记录中"
            elif revision_status.startswith("unavailable"):
                detail = "修订调用不可用，已保留作者初稿和失败记录"
            elif revision_status.startswith("verification"):
                detail = "修订补丁未通过定点验证；逐项判定已保存在验证记录中"
            else:
                detail = "自动修订已回滚；详细判定已保存在工作区"
        if release_status != "verified_final":
            final_body = _release_banner(release_status, detail) + "\n" + final_body

        metadata = self._metadata(
            brief, meta_hint,
            release_status=release_status,
            fact_report=fact_report,
            reader_report=reader_report,
            mechanical=final_mechanical,
            evidence_status=evidence_status,
            brief_status=brief_status,
            revision_status=revision_status,
            open_fact=open_fact,
            unresolved_p2_fact=unresolved_p2_fact,
            open_reader=open_reader,
            marker_notes=marker_notes,
            model_calls=model_calls,
            source_coverage_issues=source_coverage_issues,
            source_coverage_complete=not source_degraded,
        )
        metadata["evidence_index_metrics"] = _plain(
            getattr(bundle, "index_metrics", {}) or {}
        )
        metadata["degraded_evidence_chunks"] = _plain(
            getattr(bundle, "degraded_chunks", {}) or {}
        )
        metadata["arxiv_id"] = arxiv_id
        metadata["author_mechanical_audit"] = _plain(mechanical)
        metadata["mechanical_severe_findings"] = severe_mechanical
        metadata["revision_decision"] = (
            workspace.get_json("revision-decision") if workspace else None
        )
        metadata["verification_decision"] = (
            workspace.get_json("verification-decision") if workspace else None
        )
        metadata["rejected_edits"] = [_plain(item) for item in rejected_edits]
        metadata["accepted_edits"] = [_plain(item) for item in accepted_edits]
        metadata["budget_profile"] = self._active_budget.to_dict()
        metadata["author_draft_chars"] = len(author_body)
        metadata["final_body_chars"] = len(final_body)
        if verification is not None:
            metadata["verification_status"] = _status(verification)

        if workspace:
            if release_status == "verified_final":
                workspace.put_text("verified-final", final_body)
            else:
                # 降级重跑不能留下上一次的 verified-final，避免文件存在性被误当成
                # 当前发布状态。
                workspace.invalidate_from(("verified-final",))
            workspace.put_text("final-body", final_body)
            workspace.put_json("release-metadata", metadata)

        timer.mark("done")
        logger.info(
            "V9 精读完成：status=%s calls=%d elapsed=%.0fs",
            release_status, model_calls, timer.total(),
        )
        used_formulas = _marker_labels(final_body, "MATH")
        return ReadingResult(
            body_md=final_body,
            metadata=metadata,
            layer_names=[str(beat.get("title") or beat.get("id") or "") for beat in _beat_dicts(brief)],
            figure_notes={},
            figure_inlines={},
            vision_used=False,
            formula_images={
                asset.label: asset.img_path for asset in formulas if asset.label in used_formulas
            },
            model_calls=model_calls,
            timer=timer,
        )
