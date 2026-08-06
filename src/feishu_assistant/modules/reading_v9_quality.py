"""V9 精读管线的确定性质量检查、重复定位和安全文字补丁。

全部函数都是无 I/O、无模型调用的纯函数。机械信号只用于定位和护栏，不把相似度、
标题数量或术语命中伪装成文章质量分数。
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from .reading_graph import resolve_anchor
from .reading_v9_models import (
    AuditIssue,
    AuditReport,
    EditCard,
    RevisionEdit,
    V9ValidationError,
    decode_json_object,
)

DEFAULT_H2_MIN = 5
DEFAULT_H2_MAX = 7
DEFAULT_TOTAL_HEADING_MAX = 18
DEFAULT_DUPLICATE_MIN_CHARS = 80
DEFAULT_DUPLICATE_THRESHOLD = 0.82
DEFAULT_MAX_CHANGE_RATIO = 0.20
DEFAULT_MAX_FACT_CARDS = 3
DEFAULT_MAX_READER_CARDS = 2
DEFAULT_MAX_TOTAL_CARDS = 5

# V9 不把所有拉丁字母一律判成错误：代码标识、图表标签、模型/环境/游戏名和
# 少量已经约定俗成的技术词可以保留；其余出现在中文正文里的英文单词会被定位。
# 这是“白名单”，不是把整篇论文原文自动放行，避免 hopefully 这类残留漏掉。
DEFAULT_ENGLISH_ALLOWLIST = frozenset({
    "llm", "llms", "figure", "fig", "table", "equation", "eq", "flash",
    "rollout", "agent", "agents", "action", "harness", "autoharness", "textarena",
    "kaggle", "gamearena", "python", "thompson", "sampling", "rushhour", "refiner",
    "refine", "refinement", "othello", "chess", "pegjump", "minesweeper", "lightsout", "bandit",
    "fifteenpuzzle", "reflexion", "uci", "numpy", "blackjack", "secretary",
    "frozenlake", "guessthenumber", "mastermind", "sudoku", "towerofhanoi", "mafia",
    "codenames", "germanwhist", "cryptarithm", "gemini", "gemini-2", "gpt-5", "pro",
    "ruoss", "move", "of",
    "chain-of-thought", "cot", "tot", "harness-as-action-verifier", "harness-as-policy",
    "harness-as-action-filter", "fine-tuning", "hand-coded", "tree", "thoughts",
    "rejection", "sampler", "critic", "heuristic", "value", "exploitation",
    "exploration", "mutation", "operator", "true", "false", "positive", "bug",
    "available", "moves", "illegal", "suboptimal", "prohibited", "token", "code",
    "try-except", "vs", "chess-v0", "minesweeper-v0", "e2e4", "high", "pro",
    "c", "d", "h", "r", "fig", "body", "api", "json", "pdf", "markdown",
})
_ENGLISH_TOKEN_RE = re.compile(
    r"(?<![A-Za-z])([A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*)(?![A-Za-z])"
)

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$")
_FENCE_RE = re.compile(r"^[ \t]*(```+|~~~+)")
_MARKER_RE = re.compile(r"\[\[(FIG|MATH):([^\]\n]+)\]\]", re.IGNORECASE)
_FIGURE_REF_RE = re.compile(
    r"(?<![\w])(?:Figure|Fig\.?|Table)\s*\d+[A-Za-z]?(?![\w])|(?:图|表)\s*\d+[A-Za-z]?",
    re.IGNORECASE,
)
_FORMULA_REF_RE = re.compile(
    r"(?<![\w])(?:Equation|Eq\.?)\s*(?:\d+[A-Za-z]?|p\d+-\d+)(?![\w])|公式\s*\d+[A-Za-z]?",
    re.IGNORECASE,
)

_PLACEHOLDER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("todo", re.compile(r"(?<![A-Za-z])(?:TODO|TBD|FIXME|XXX)(?![A-Za-z])", re.IGNORECASE)),
    ("chinese_placeholder", re.compile(r"(?:待补充|待补|待完善|待核实|待确认|待插入|占位符|此处补充|这里补充)")),
    ("template_braces", re.compile(r"\{\{[^{}\n]{1,100}\}\}")),
    ("angle_placeholder", re.compile(r"<<[^<>\n]{1,100}>>")),
    ("unfinished_media_marker", re.compile(r"\[\[(?:FIG|MATH):[^\]\n]*(?:$|\n)", re.IGNORECASE)),
)

_PLANNING_JARGON: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("debt_id", re.compile(r"\bdebt_ids?\b", re.IGNORECASE)),
    ("must_explain", re.compile(r"\bmust_explain\b", re.IGNORECASE)),
    ("success_test", re.compile(r"\bsuccess_test\b", re.IGNORECASE)),
    ("root_scope", re.compile(r"\broot_scope\b", re.IGNORECASE)),
    ("scope_violation", re.compile(r"\bscope_violation\b", re.IGNORECASE)),
    ("body_block_id", re.compile(r"\bbody_block_ids?\b", re.IGNORECASE)),
    ("source_span_id", re.compile(r"\bsource_span_ids?\b", re.IGNORECASE)),
    ("finite_judgement", re.compile(r"有限判断")),
    ("evidence_supports_what", re.compile(r"证据支持什么")),
    ("contract_gap", re.compile(r"合同缺口|章节合同|写作合同")),
    ("review_voice", re.compile(r"审核意见|审查意见|编辑卡|规划要求")),
    ("chapter_boilerplate", re.compile(r"本章将|本层将|概念已在前文建立")),
)


@dataclass(frozen=True, slots=True)
class QualityFinding:
    code: str
    severity: str
    message: str
    start: int
    end: int
    line: int
    excerpt: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "start": self.start,
            "end": self.end,
            "line": self.line,
            "excerpt": self.excerpt,
        }


@dataclass(frozen=True, slots=True)
class MechanicalQualityReport:
    """机械护栏报告；findings 是告警，不包含模型式 pass/score。"""

    body_sha256: str
    findings: tuple[QualityFinding, ...]
    heading_counts: tuple[tuple[int, int], ...]
    figure_references: tuple[tuple[str, int], ...]
    formula_references: tuple[tuple[str, int], ...]
    placeholder_count: int
    planning_jargon_count: int
    untranslated_english_count: int = 0
    untranslated_english_words: tuple[str, ...] = ()

    @property
    def errors(self) -> tuple[QualityFinding, ...]:
        return tuple(item for item in self.findings if item.severity == "error")

    @property
    def warnings(self) -> tuple[QualityFinding, ...]:
        return tuple(item for item in self.findings if item.severity == "warning")

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "body_sha256": self.body_sha256,
            "findings": [item.to_dict() for item in self.findings],
            "heading_counts": {f"h{level}": count for level, count in self.heading_counts},
            "figure_references": dict(self.figure_references),
            "formula_references": dict(self.formula_references),
            "placeholder_count": self.placeholder_count,
            "planning_jargon_count": self.planning_jargon_count,
            "untranslated_english_count": self.untranslated_english_count,
            "untranslated_english_words": list(self.untranslated_english_words),
        }


@dataclass(frozen=True, slots=True)
class DuplicateCandidate:
    first_start: int
    first_end: int
    first_line: int
    first_text: str
    second_start: int
    second_end: int
    second_line: int
    second_text: str
    similarity: float
    kind: str

    @property
    def span_a(self) -> str:
        return self.first_text

    @property
    def span_b(self) -> str:
        return self.second_text

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "similarity": self.similarity,
            "span_a": {
                "start": self.first_start,
                "end": self.first_end,
                "line": self.first_line,
                "text": self.first_text,
            },
            "span_b": {
                "start": self.second_start,
                "end": self.second_end,
                "line": self.second_line,
                "text": self.second_text,
            },
        }


@dataclass(frozen=True, slots=True)
class EditCardRejection:
    audit_kind: str
    issue_id: str
    code: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "audit_kind": self.audit_kind,
            "issue_id": self.issue_id,
            "code": self.code,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class EditCardSelection:
    cards: tuple[EditCard, ...]
    rejected: tuple[EditCardRejection, ...]

    def __iter__(self):
        return iter(self.cards)

    def __len__(self) -> int:
        return len(self.cards)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cards": [card.to_dict() for card in self.cards],
            "rejected": [item.to_dict() for item in self.rejected],
        }


@dataclass(frozen=True, slots=True)
class PatchRejection:
    edit_id: str
    code: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"edit_id": self.edit_id, "code": self.code, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class EditScriptResult:
    """安全补丁结果；支持 ``body, accepted, rejected = result`` 解包。"""

    body: str
    accepted: tuple[RevisionEdit, ...]
    rejected: tuple[PatchRejection, ...]
    changed_chars: int
    change_ratio: float
    original_sha256: str
    result_sha256: str

    @property
    def changed(self) -> bool:
        return self.original_sha256 != self.result_sha256

    def __iter__(self):
        yield self.body
        yield self.accepted
        yield self.rejected

    def to_dict(self, *, include_body: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "accepted": [edit.to_dict() for edit in self.accepted],
            "rejected": [item.to_dict() for item in self.rejected],
            "changed_chars": self.changed_chars,
            "change_ratio": self.change_ratio,
            "original_sha256": self.original_sha256,
            "result_sha256": self.result_sha256,
        }
        if include_body:
            data["body"] = self.body
        return data


@dataclass(frozen=True, slots=True)
class _Paragraph:
    start: int
    end: int
    line: int
    text: str
    normalized: str


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_body(body: Any) -> str:
    if not isinstance(body, str):
        raise V9ValidationError("body 必须是字符串")
    return body


def _line_number(body: str, offset: int) -> int:
    return body.count("\n", 0, max(0, offset)) + 1


def _excerpt(body: str, start: int, end: int, *, radius: int = 45) -> str:
    left = max(0, start - radius)
    right = min(len(body), end + radius)
    return body[left:right].replace("\n", " ").strip()


def _mask_code_fences(body: str) -> str:
    """用空格覆盖 fenced code，保留长度和换行，使命中 offset 仍对应原文。"""
    chars = list(body)
    in_fence = False
    fence_token = ""
    offset = 0
    for line in body.splitlines(keepends=True):
        match = _FENCE_RE.match(line)
        is_fence = bool(match)
        if is_fence:
            token = match.group(1)[0]
            if not in_fence:
                in_fence = True
                fence_token = token
            elif token == fence_token:
                in_fence = False
                fence_token = ""
        if in_fence or is_fence:
            for index in range(offset, offset + len(line)):
                if chars[index] not in "\r\n":
                    chars[index] = " "
        offset += len(line)
    return "".join(chars)


def _mask_english_exemptions(body: str) -> str:
    """为英文残留检查遮掉标题、代码、函数名、链接和状态横幅。"""
    masked = list(_mask_code_fences(body))

    def blank(start: int, end: int) -> None:
        for index in range(start, end):
            if masked[index] not in "\r\n":
                masked[index] = " "

    offset = 0
    for line in body.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("#") or stripped.startswith("> ⚠️"):
            blank(offset, offset + len(line))
        offset += len(line)

    visible = "".join(masked)
    for match in re.finditer(r"`[^`\n]+`", visible):
        blank(match.start(), match.end())
    # 正文里偶尔会直接写 propose_action() / os.path.join()；这种可执行标识
    # 不是未翻译自然语言。只豁免紧跟左括号的标识，不把普通英文短语放行。
    visible = "".join(masked)
    for match in re.finditer(
        r"(?<![A-Za-z0-9_])(?:[A-Za-z_][A-Za-z0-9_]*\.)*"
        r"[A-Za-z_][A-Za-z0-9_]*(?=[ \t]*\()",
        visible,
    ):
        blank(match.start(), match.end())
    # Markdown link的显示文本仍可被检查，URL/路径本身不属于自然语言正文。
    visible = "".join(masked)
    for match in re.finditer(r"\]\([^\n)]*\)", visible):
        open_paren = visible.find("(", match.start(), match.end())
        if open_paren >= 0:
            blank(open_paren, match.end())
    return "".join(masked)


def _inventory_labels(items: Iterable[Any] | None) -> tuple[str, ...]:
    labels: list[str] = []
    for index, item in enumerate(items or ()):
        if isinstance(item, str):
            label = item
        elif isinstance(item, Mapping):
            label = next(
                (str(item[key]) for key in ("label", "id", "name") if item.get(key)),
                "",
            )
        else:
            label = str(getattr(item, "label", ""))
        label = label.strip()
        if not label:
            raise V9ValidationError(f"inventory[{index}] 缺少可识别 label")
        labels.append(label)
    return tuple(dict.fromkeys(labels))


def _canonical_media_label(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[（(](?:第?\s*\d+\s*页|片段\s*\d+)[^）)]*[）)]$", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    match = re.fullmatch(r"(?:Fig\.?|Figure)\s*(\d+[A-Za-z]?)", text, re.IGNORECASE)
    if match:
        return f"Figure {match.group(1)}".casefold()
    match = re.fullmatch(r"Table\s*(\d+[A-Za-z]?)", text, re.IGNORECASE)
    if match:
        return f"Table {match.group(1)}".casefold()
    match = re.fullmatch(r"图\s*(\d+[A-Za-z]?)", text)
    if match:
        return f"Figure {match.group(1)}".casefold()
    match = re.fullmatch(r"表\s*(\d+[A-Za-z]?)", text)
    if match:
        return f"Table {match.group(1)}".casefold()
    match = re.fullmatch(
        r"(?:Eq\.?|Equation)\s*(\d+[A-Za-z]?|p\d+-\d+)", text, re.IGNORECASE
    )
    if match:
        return f"Equation {match.group(1)}".casefold()
    match = re.fullmatch(r"公式\s*(\d+[A-Za-z]?)", text)
    if match:
        return f"Equation {match.group(1)}".casefold()
    return re.sub(r"\s+", " ", text).casefold()


def _display_label(canonical: str, known: Mapping[str, str]) -> str:
    return known.get(canonical, canonical)


def mechanical_quality_report(
    body: str,
    figures: Iterable[Any] | None = (),
    formulas: Iterable[Any] | None = (),
    english_allowlist: Iterable[str] | None = None,
) -> MechanicalQualityReport:
    """检查标题、占位符、图表/公式引用、规划黑话和英文残留。"""
    body = _require_body(body)
    masked = _mask_code_fences(body)
    findings: list[QualityFinding] = []

    def add(code: str, severity: str, message: str, start: int, end: int) -> None:
        findings.append(QualityFinding(
            code=code,
            severity=severity,
            message=message,
            start=start,
            end=end,
            line=_line_number(body, start),
            excerpt=_excerpt(body, start, end),
        ))

    if not body.strip():
        add("empty_body", "error", "正文为空", 0, 0)

    heading_counts = {level: 0 for level in range(1, 7)}
    headings: list[tuple[int, str, int, int]] = []
    offset = 0
    for line in masked.splitlines(keepends=True):
        raw = line.rstrip("\r\n")
        match = _HEADING_RE.match(raw)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            heading_counts[level] += 1
            headings.append((level, title, offset, offset + len(raw)))
        elif re.match(r"^[ \t]*#{1,6}(?:[ \t]*$|[^# \t])", raw):
            add("malformed_heading", "warning", "Markdown 标题缺少正文或 # 后空格", offset, offset + len(raw))
        offset += len(line)

    previous_level: int | None = None
    seen_titles: dict[str, tuple[int, int]] = {}
    for level, title, start, end in headings:
        if previous_level is not None and level > previous_level + 1:
            add("heading_level_jump", "warning", f"标题层级从 H{previous_level} 跳到 H{level}", start, end)
        previous_level = level
        normalized = re.sub(r"[^\w一-鿿]+", "", title).casefold()
        if normalized in seen_titles:
            add("duplicate_heading", "warning", f"标题重复：{title}", start, end)
        else:
            seen_titles[normalized] = (start, end)

    h2_count = heading_counts[2]
    if h2_count < DEFAULT_H2_MIN:
        add("h2_count_low", "warning", f"H2 数量为 {h2_count}，低于建议下限 {DEFAULT_H2_MIN}", 0, 0)
    if h2_count > DEFAULT_H2_MAX:
        add("h2_count_high", "warning", f"H2 数量为 {h2_count}，高于建议上限 {DEFAULT_H2_MAX}", 0, 0)
    total_headings = sum(heading_counts.values())
    if total_headings > DEFAULT_TOTAL_HEADING_MAX:
        add(
            "heading_count_high",
            "warning",
            f"标题总数为 {total_headings}，高于建议上限 {DEFAULT_TOTAL_HEADING_MAX}",
            0,
            0,
        )

    placeholder_count = 0
    for label, pattern in _PLACEHOLDER_PATTERNS:
        for match in pattern.finditer(masked):
            placeholder_count += 1
            add("placeholder", "error", f"发现未完成占位符：{label}", match.start(), match.end())

    jargon_count = 0
    for label, pattern in _PLANNING_JARGON:
        for match in pattern.finditer(masked):
            jargon_count += 1
            add("planning_jargon", "warning", f"正文出现规划/审核措辞：{label}", match.start(), match.end())

    allowlist = set(DEFAULT_ENGLISH_ALLOWLIST)
    allowlist.update(str(item).strip().casefold() for item in (english_allowlist or ()) if str(item).strip())
    allowlist.update(str(item).strip().casefold() for item in _inventory_labels(figures))
    allowlist.update(str(item).strip().casefold() for item in _inventory_labels(formulas))
    english_words: list[str] = []
    english_masked = _mask_english_exemptions(body)
    for match in _ENGLISH_TOKEN_RE.finditer(english_masked):
        word = match.group(1)
        normalized = word.casefold()
        # 单字符变量、带下划线的代码标识和明确白名单词不算自然语言残留。
        if len(word) == 1 or "_" in word or normalized in allowlist:
            continue
        english_words.append(word)
        add(
            "untranslated_english",
            "warning",
            f"中文正文混入未翻译英文单词：{word}",
            match.start(),
            match.end(),
        )

    figure_inventory = _inventory_labels(figures)
    formula_inventory = _inventory_labels(formulas)
    known_figures = {_canonical_media_label(label): label for label in figure_inventory}
    known_formulas = {_canonical_media_label(label): label for label in formula_inventory}
    figure_counts: dict[str, int] = {}
    formula_counts: dict[str, int] = {}

    marker_spans: list[tuple[int, int]] = []
    for match in _MARKER_RE.finditer(masked):
        marker_spans.append((match.start(), match.end()))
        marker_kind = match.group(1).upper()
        raw_label = match.group(2).strip()
        canonical = _canonical_media_label(raw_label)
        counts = figure_counts if marker_kind == "FIG" else formula_counts
        known = known_figures if marker_kind == "FIG" else known_formulas
        display = _display_label(canonical, known)
        counts[display] = counts.get(display, 0) + 1
        if known and canonical not in known:
            add(
                "unknown_media_marker",
                "error",
                f"{marker_kind} 标记不在资产目录中：{raw_label}",
                match.start(),
                match.end(),
            )
        line_start = body.rfind("\n", 0, match.start()) + 1
        line_end = body.find("\n", match.end())
        if line_end < 0:
            line_end = len(body)
        if body[line_start:line_end].strip() != match.group(0):
            add(
                "embedded_media_marker",
                "warning",
                f"{marker_kind} 标记应独立成行",
                match.start(),
                match.end(),
            )

    marker_free = list(masked)
    for start, end in marker_spans:
        for index in range(start, end):
            if marker_free[index] not in "\r\n":
                marker_free[index] = " "
    visible_without_markers = "".join(marker_free)

    for match in _FIGURE_REF_RE.finditer(visible_without_markers):
        canonical = _canonical_media_label(match.group(0))
        display = _display_label(canonical, known_figures)
        figure_counts[display] = figure_counts.get(display, 0) + 1
        if known_figures and canonical not in known_figures:
            add("unknown_figure_reference", "warning", f"图表引用不在资产目录中：{match.group(0)}", match.start(), match.end())

    for match in _FORMULA_REF_RE.finditer(visible_without_markers):
        canonical = _canonical_media_label(match.group(0))
        display = _display_label(canonical, known_formulas)
        formula_counts[display] = formula_counts.get(display, 0) + 1
        if known_formulas and canonical not in known_formulas:
            add("unknown_formula_reference", "warning", f"公式引用不在资产目录中：{match.group(0)}", match.start(), match.end())

    for label, count in sorted(figure_counts.items()):
        # V9 预算：一次主讲解 + 最多一次承担新功能的回指。
        if count > 2:
            start = masked.find(label)
            add("figure_reference_overuse", "warning", f"{label} 共出现 {count} 次，检查是否重复搬运", max(0, start), max(0, start) + len(label))
    for label, count in sorted(formula_counts.items()):
        if count > 2:
            start = masked.find(label)
            add("formula_reference_overuse", "warning", f"{label} 共出现 {count} 次，检查是否重复讲解", max(0, start), max(0, start) + len(label))

    findings.sort(key=lambda item: (item.start, item.code, item.message))
    return MechanicalQualityReport(
        body_sha256=_sha256(body),
        findings=tuple(findings),
        heading_counts=tuple((level, heading_counts[level]) for level in range(1, 7)),
        figure_references=tuple(sorted(figure_counts.items())),
        formula_references=tuple(sorted(formula_counts.items())),
        placeholder_count=placeholder_count,
        planning_jargon_count=jargon_count,
        untranslated_english_count=len(english_words),
        untranslated_english_words=tuple(dict.fromkeys(english_words)),
    )


def _normalize_duplicate_text(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[\[(?:FIG|MATH):[^\]]+\]\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[`*_>#|~\-]+", "", text)
    return re.sub(r"[^\w一-鿿]+", "", text).casefold()


def _paragraphs(body: str, min_chars: int) -> list[_Paragraph]:
    paragraphs: list[_Paragraph] = []
    current_start: int | None = None
    current_end = 0
    in_fence = False
    fence_token = ""
    offset = 0

    def flush() -> None:
        nonlocal current_start, current_end
        if current_start is None:
            return
        raw = body[current_start:current_end]
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw.rstrip())
        start = current_start + leading
        end = current_start + trailing
        text = body[start:end]
        normalized = _normalize_duplicate_text(text)
        lines = [line for line in text.splitlines() if line.strip()]
        table_like = bool(lines) and sum(line.count("|") >= 2 for line in lines) / len(lines) >= 0.5
        marker_only = bool(text.strip()) and all(
            _MARKER_RE.fullmatch(line.strip()) for line in lines
        )
        if len(normalized) >= min_chars and not table_like and not marker_only:
            paragraphs.append(_Paragraph(
                start=start,
                end=end,
                line=_line_number(body, start),
                text=text,
                normalized=normalized,
            ))
        current_start = None
        current_end = 0

    for line in body.splitlines(keepends=True):
        stripped = line.strip()
        fence = _FENCE_RE.match(line)
        if fence:
            flush()
            token = fence.group(1)[0]
            if not in_fence:
                in_fence = True
                fence_token = token
            elif token == fence_token:
                in_fence = False
                fence_token = ""
            offset += len(line)
            continue
        if in_fence or not stripped or _HEADING_RE.match(line.rstrip("\r\n")):
            flush()
            offset += len(line)
            continue
        if current_start is None:
            current_start = offset
        current_end = offset + len(line.rstrip("\r\n"))
        offset += len(line)
    flush()
    return paragraphs


def _shingles(text: str, size: int = 3) -> set[str]:
    if len(text) < size:
        return {text} if text else set()
    return {text[index:index + size] for index in range(len(text) - size + 1)}


def duplicate_candidates(
    body: str,
    *,
    min_chars: int = DEFAULT_DUPLICATE_MIN_CHARS,
    similarity_threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
    max_candidates: int = 20,
) -> tuple[DuplicateCandidate, ...]:
    """按段落返回高置信 exact/near duplicate 候选；候选不等同于应删除的问题。"""
    body = _require_body(body)
    if type(min_chars) is not int or min_chars < 20:
        raise V9ValidationError("min_chars 必须是至少 20 的整数")
    if not isinstance(similarity_threshold, (int, float)) or isinstance(similarity_threshold, bool):
        raise V9ValidationError("similarity_threshold 必须是数字")
    threshold = float(similarity_threshold)
    if not 0.5 <= threshold <= 1.0:
        raise V9ValidationError("similarity_threshold 必须位于 0.5～1.0")
    if type(max_candidates) is not int or max_candidates < 0:
        raise V9ValidationError("max_candidates 必须是非负整数")

    paragraphs = _paragraphs(body, min_chars)
    fingerprints = [_shingles(item.normalized) for item in paragraphs]
    found: list[DuplicateCandidate] = []
    for first_index, first in enumerate(paragraphs):
        for second_index in range(first_index + 1, len(paragraphs)):
            second = paragraphs[second_index]
            short_len = min(len(first.normalized), len(second.normalized))
            long_len = max(len(first.normalized), len(second.normalized))
            if short_len / max(1, long_len) < 0.55:
                continue
            if first.normalized == second.normalized:
                score = 1.0
                kind = "exact"
            else:
                a_set = fingerprints[first_index]
                b_set = fingerprints[second_index]
                containment = len(a_set & b_set) / max(1, min(len(a_set), len(b_set)))
                if containment < max(0.55, threshold - 0.18):
                    continue
                score = SequenceMatcher(
                    None, first.normalized, second.normalized, autojunk=False
                ).ratio()
                if score < threshold and not (containment >= 0.92 and short_len / long_len >= 0.75):
                    continue
                score = max(score, min(1.0, containment * 0.98))
                kind = "near"
            found.append(DuplicateCandidate(
                first_start=first.start,
                first_end=first.end,
                first_line=first.line,
                first_text=first.text,
                second_start=second.start,
                second_end=second.end,
                second_line=second.line,
                second_text=second.text,
                similarity=round(score, 4),
                kind=kind,
            ))
    found.sort(key=lambda item: (-item.similarity, item.first_start, item.second_start))
    return tuple(found[:max_candidates])


def _all_occurrences(body: str, needle: str) -> list[int]:
    starts: list[int] = []
    offset = 0
    while True:
        found = body.find(needle, offset)
        if found < 0:
            return starts
        starts.append(found)
        offset = found + max(1, len(needle))


def _scope_allows(text: str, scope: str) -> bool:
    blank_breaks = len(re.findall(r"\n[ \t]*\n", text))
    if scope == "sentence":
        return blank_breaks == 0 and "\n" not in text
    if scope == "paragraph":
        return blank_breaks == 0
    return blank_breaks <= 1


def _card_id(issue: AuditIssue, target: str) -> str:
    material = f"{issue.audit_kind}|{issue.issue_id}|{issue.operation}|{target}"
    return "card-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _reader_target(issue: AuditIssue) -> tuple[str, str]:
    if issue.kind != "duplication":
        return issue.span_a, issue.span_b
    if issue.keep == "a":
        return issue.span_b, issue.span_a
    return issue.span_a, issue.span_b


def _resolve_reader_anchor(body: str, quote: str) -> tuple[str, str]:
    """把审查器的近似摘录归一为正文中的唯一原文锚点。

    审查模型有时会复述段落而不是逐字复制。先保留精确匹配；只有精确命中为
    0 时才使用 reading_graph 的三级定位（精确/规范化/局部对齐），且最终必须
    得到正文中唯一出现的实际片段。无法唯一定位时仍 fail closed。
    """
    text = str(quote or "").strip()
    if not text:
        return "", "空摘录"
    if len(_all_occurrences(body, text)) == 1:
        return text, "exact"
    resolved = resolve_anchor(body, text)
    if resolved.resolved:
        actual = str(resolved.quote or "").strip()
        if actual and len(_all_occurrences(body, actual)) == 1:
            return actual, "fuzzy"

    # resolve_anchor 的滑窗步长对短中文句可能跨过真实起点；再按段落/句子边界
    # 做一次局部对齐。仍要求高相似度、最佳候选与次佳候选有明显差距且原文唯一。
    normalized_quote = _normalize_duplicate_text(text)
    candidates: list[tuple[float, str]] = []
    for paragraph in _paragraphs(body, min_chars=8):
        raw_candidates = [paragraph.text]
        raw_candidates.extend(
            item.strip()
            for item in re.findall(r"[^。！？!?；;\n]+(?:[。！？!?；;]|$)", paragraph.text)
            if item.strip()
        )
        for raw in dict.fromkeys(raw_candidates):
            normalized = _normalize_duplicate_text(raw)
            if not normalized:
                continue
            length_ratio = min(len(normalized_quote), len(normalized)) / max(
                1, max(len(normalized_quote), len(normalized))
            )
            if length_ratio < 0.65:
                continue
            score = SequenceMatcher(
                None, normalized_quote, normalized, autojunk=False,
            ).ratio()
            if score >= 0.82:
                candidates.append((score, raw))
    candidates.sort(key=lambda item: item[0], reverse=True)
    if not candidates:
        return "", resolved.reason or "局部对齐低于相似度阈值"
    best_score, best_raw = candidates[0]
    second_score = candidates[1][0] if len(candidates) > 1 else 0.0
    if best_score - second_score < 0.04:
        return "", "候选差距不足，歧义"
    if len(_all_occurrences(body, best_raw)) != 1:
        return "", "归一化锚点在正文中不唯一"
    return best_raw, "fuzzy"


def _fact_quote_anchors(quote: str) -> tuple[str, ...]:
    """从非连续审核引文中提取仍可逐字核对的长句锚点。

    审核模型偶尔会用省略号把同一段的两句拼成 draft_quote。这里不做模糊
    相似匹配，只保留审核引文中原样出现、长度足够的句子片段。
    """
    anchors: list[str] = []
    for chunk in re.split(r"(?:\.{3,}|…+)", str(quote or "")):
        for match in re.finditer(r"[^。！？!?；;\n]+(?:[。！？!?；;]|$)", chunk):
            value = match.group(0).strip().strip('"\'“”‘’')
            visible_chars = re.sub(r"\s+", "", value)
            if len(visible_chars) >= 20:
                anchors.append(value)
    return tuple(dict.fromkeys(anchors))


def _recover_fact_paragraph_target(body: str, quote: str) -> str:
    """用唯一精确句锚点恢复同一完整段落；失败时返回空字符串。

    这条恢复只服务 ``max_scope=paragraph`` 的事实卡。它不会改字、删 Markdown
    或跨段拼接：锚点必须在正文中逐字且唯一出现，所有可用锚点必须落在同一
    个正文段落，该完整段落本身也必须唯一。任何歧义都 fail closed。
    """
    paragraphs = _paragraphs(body, min_chars=1)
    anchors = _fact_quote_anchors(quote)
    if not anchors:
        return ""
    matched: list[_Paragraph] = []
    for anchor in anchors:
        occurrences = _all_occurrences(body, anchor)
        if len(occurrences) != 1:
            # 引文中任一可识别片段缺失或不唯一，都不能凭另一个片段猜段落。
            return ""
        start = occurrences[0]
        end = start + len(anchor)
        owners = [
            paragraph for paragraph in paragraphs
            if paragraph.start <= start and end <= paragraph.end
        ]
        if len(owners) == 1:
            matched.append(owners[0])
        else:
            return ""
    if not matched:
        return ""
    targets = {paragraph.text for paragraph in matched}
    if len(targets) != 1:
        return ""
    target = next(iter(targets))
    if len(_all_occurrences(body, target)) != 1 or not _scope_allows(target, "paragraph"):
        return ""
    return target


def select_edit_cards(
    body: str,
    fact_report: AuditReport,
    reader_report: AuditReport,
    *,
    max_fact_cards: int = DEFAULT_MAX_FACT_CARDS,
    max_reader_cards: int = DEFAULT_MAX_READER_CARDS,
    max_total_cards: int = DEFAULT_MAX_TOTAL_CARDS,
) -> EditCardSelection:
    """事实问题优先，再选少量读者问题；不把一次修订变成批量重写。"""
    body = _require_body(body)
    if not isinstance(fact_report, AuditReport) or fact_report.kind != "fact":
        raise V9ValidationError("fact_report 必须是 kind='fact' 的 AuditReport")
    if not isinstance(reader_report, AuditReport) or reader_report.kind != "reader":
        raise V9ValidationError("reader_report 必须是 kind='reader' 的 AuditReport")
    for name, value, upper in (
        ("max_fact_cards", max_fact_cards, 5),
        ("max_reader_cards", max_reader_cards, 5),
        ("max_total_cards", max_total_cards, 8),
    ):
        if type(value) is not int or not 0 <= value <= upper:
            raise V9ValidationError(f"{name} 必须是 0～{upper} 的整数")

    cards: list[EditCard] = []
    rejected: list[EditCardRejection] = []
    occupied: list[tuple[int, int, str]] = []
    fact_accepted = 0
    reader_accepted = 0
    total_accepted = 0

    def reject(issue: AuditIssue, code: str, reason: str) -> None:
        rejected.append(EditCardRejection(issue.audit_kind, issue.issue_id, code, reason))

    def consider(issue: AuditIssue) -> None:
        nonlocal fact_accepted, reader_accepted, total_accepted
        # 事实问题不因 P 级被静默丢弃：P2 也进入同一有限卡预算，避免系统
        # 已经知道数字/事实可能错误，却既不修订也不向用户说明。预算仍由
        # max_fact_cards/max_total_cards 控制，防止一次审查变成整章重写。
        if issue.audit_kind == "fact" and fact_accepted >= max_fact_cards:
            reject(issue, "fact_budget", f"事实编辑卡已达到上限 {max_fact_cards}")
            return
        if issue.audit_kind == "reader" and reader_accepted >= max_reader_cards:
            reject(issue, "reader_budget", f"读者编辑卡已达到上限 {max_reader_cards}")
            return
        if total_accepted >= max_total_cards:
            reject(issue, "total_budget", f"本次编辑卡已达到总上限 {max_total_cards}")
            return
        if issue.operation == "move":
            reject(issue, "move_not_atomic", "move 缺少单一精确目标，不能作为自动单补丁执行")
            return

        if issue.audit_kind == "fact":
            target = issue.draft_quote
            paired = ""
            problem = issue.problem
            evidence = f"{issue.source_location}\n{issue.source_evidence}".strip()
            keep = "none"
            protected: tuple[str, ...] = ()
        else:
            target, paired = _reader_target(issue)
            problem = issue.reader_harm
            evidence = ""
            keep = issue.keep
            protected = (paired,) if paired else ()

        if issue.operation == "merge" and not paired:
            reject(issue, "paired_target_missing", "merge 必须同时定位两个可合并片段")
            return

        occurrences = _all_occurrences(body, target)
        if (
            not occurrences
            and issue.audit_kind == "fact"
            and issue.max_scope == "paragraph"
        ):
            recovered = _recover_fact_paragraph_target(body, target)
            if recovered:
                target = recovered
                occurrences = _all_occurrences(body, target)
        if not occurrences and issue.audit_kind == "reader":
            resolved_target, _ = _resolve_reader_anchor(body, target)
            if resolved_target:
                target = resolved_target
                occurrences = _all_occurrences(body, target)
        if len(occurrences) != 1:
            code = "target_missing" if not occurrences else "target_ambiguous"
            reject(issue, code, f"target 在正文中精确出现 {len(occurrences)} 次，要求恰好 1 次")
            return
        if paired:
            paired_occurrences = _all_occurrences(body, paired)
            if not paired_occurrences and issue.audit_kind == "reader":
                resolved_paired, _ = _resolve_reader_anchor(body, paired)
                if resolved_paired:
                    paired = resolved_paired
                    protected = (paired,)
                    paired_occurrences = _all_occurrences(body, paired)
            if len(paired_occurrences) != 1:
                reject(
                    issue,
                    "paired_target_ambiguous",
                    f"paired span 在正文中精确出现 {len(paired_occurrences)} 次，要求恰好 1 次",
                )
                return
        if not _scope_allows(target, issue.max_scope):
            reject(issue, "scope_exceeded", f"target 超出声明范围 {issue.max_scope}")
            return
        start = occurrences[0]
        end = start + len(target)
        ranges = [(start, end)]
        if paired:
            paired_start = _all_occurrences(body, paired)[0]
            ranges.append((paired_start, paired_start + len(paired)))
        overlap = next(
            (
                card_id
                for old_start, old_end, card_id in occupied
                for new_start, new_end in ranges
                if new_start < old_end and old_start < new_end
            ),
            "",
        )
        if overlap:
            reject(issue, "overlap", f"target 与已选编辑卡 {overlap} 重叠")
            return

        card = EditCard(
            card_id=_card_id(issue, target),
            issue_ids=(issue.issue_id,),
            audit_kind=issue.audit_kind,
            severity=issue.severity,
            operation=issue.operation,
            target_quote=target,
            required_result=issue.required_outcome,
            max_scope=issue.max_scope,
            problem=problem,
            evidence=evidence,
            protected_content=protected,
            paired_quote=paired,
            keep=keep,
        )
        cards.append(card)
        occupied.extend((range_start, range_end, card.card_id) for range_start, range_end in ranges)
        if issue.audit_kind == "reader":
            reader_accepted += 1
        else:
            fact_accepted += 1
        total_accepted += 1

    if fact_report.status == "unavailable":
        rejected.append(EditCardRejection(
            "fact", "report:fact", "report_unavailable", fact_report.reason
        ))
    else:
        ordered_fact = sorted(
            enumerate(fact_report.issues),
            key=lambda item: ({"P0": 0, "P1": 1, "P2": 2}[item[1].severity], item[0]),
        )
        for _, issue in ordered_fact:
            consider(issue)

    if reader_report.status == "unavailable":
        rejected.append(EditCardRejection(
            "reader", "report:reader", "report_unavailable", reader_report.reason
        ))
    else:
        ordered_reader = sorted(
            enumerate(reader_report.issues),
            key=lambda item: ({"P1": 0, "P2": 1}[item[1].severity], item[0]),
        )
        for _, issue in ordered_reader:
            consider(issue)

    return EditCardSelection(cards=tuple(cards), rejected=tuple(rejected))


def _coerce_edit_payload(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, (str, bytes, bytearray)):
        return decode_json_object(payload, context="revision edits JSON")
    if isinstance(payload, Mapping):
        return payload
    raise V9ValidationError("revision payload 必须是 JSON 对象或对象字典")


def _located_edits(body: str, edits: Sequence[RevisionEdit]) -> list[tuple[int, int, RevisionEdit]]:
    ids: set[str] = set()
    located: list[tuple[int, int, RevisionEdit]] = []
    for edit in edits:
        if edit.edit_id in ids:
            raise V9ValidationError(f"revision edits 的 edit_id 重复：{edit.edit_id}")
        ids.add(edit.edit_id)
        occurrences = _all_occurrences(body, edit.old_text)
        if len(occurrences) != 1:
            raise V9ValidationError(
                f"{edit.edit_id}.old_text 在正文中精确出现 {len(occurrences)} 次，要求恰好 1 次"
            )
        for preserved in edit.preserve:
            if preserved not in body:
                raise V9ValidationError(
                    f"{edit.edit_id}.preserve 在原正文中不存在：{preserved[:40]!r}"
                )
        start = occurrences[0]
        located.append((start, start + len(edit.old_text), edit))
    located.sort(key=lambda item: item[0])
    for previous, current in zip(located, located[1:]):
        if current[0] < previous[1]:
            raise V9ValidationError(
                f"revision edits 重叠：{previous[2].edit_id} 与 {current[2].edit_id}"
            )
    return located


def parse_revision_edits(payload: Any, body: str) -> tuple[RevisionEdit, ...]:
    """严格解析 Writer 的 ``{"edits": [...]}``，并校验唯一 old_text 与非重叠。"""
    body = _require_body(body)
    data = _coerce_edit_payload(payload)
    if set(data) != {"edits"}:
        missing = {"edits"} - set(data)
        unknown = set(data) - {"edits"}
        parts = []
        if missing:
            parts.append("缺少 edits")
        if unknown:
            parts.append(f"未知字段：{', '.join(sorted(str(key) for key in unknown))}")
        raise V9ValidationError("revision payload " + "；".join(parts))
    raw = data["edits"]
    if not isinstance(raw, list):
        raise V9ValidationError("revision payload.edits 必须是数组")
    edits = tuple(RevisionEdit.from_dict(item) for item in raw)
    _located_edits(body, edits)
    return edits


def _replacement_change_cost(old_text: str, new_text: str) -> int:
    cost = 0
    matcher = SequenceMatcher(None, old_text, new_text, autojunk=False)
    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        cost += max(old_end - old_start, new_end - new_start)
    return cost


def _apply_located(body: str, located: Sequence[tuple[int, int, RevisionEdit]]) -> str:
    result = body
    for start, end, edit in reversed(located):
        result = result[:start] + edit.new_text + result[end:]
    return result


def apply_edit_script(
    body: str,
    edits: Sequence[RevisionEdit | Mapping[str, Any]],
    max_change_ratio: float = DEFAULT_MAX_CHANGE_RATIO,
    max_change_chars: int | None = None,
) -> EditScriptResult:
    """按原正文坐标应用非重叠补丁；超出改动预算或保护内容丢失时整批回滚。"""
    body = _require_body(body)
    if isinstance(edits, (str, bytes, bytearray, Mapping)):
        raise V9ValidationError("edits 必须是 RevisionEdit/字典组成的序列；JSON 请先 parse_revision_edits")
    if not isinstance(edits, Sequence):
        raise V9ValidationError("edits 必须是序列")
    if not isinstance(max_change_ratio, (int, float)) or isinstance(max_change_ratio, bool):
        raise V9ValidationError("max_change_ratio 必须是数字")
    ratio_limit = float(max_change_ratio)
    if not 0.0 <= ratio_limit <= 1.0:
        raise V9ValidationError("max_change_ratio 必须位于 0.0～1.0")
    if max_change_chars is not None and (
        type(max_change_chars) is not int or max_change_chars < 0
    ):
        raise V9ValidationError("max_change_chars 必须是非负整数或 None")

    parsed: list[RevisionEdit] = []
    for index, item in enumerate(edits):
        if isinstance(item, RevisionEdit):
            parsed.append(item)
        elif isinstance(item, Mapping):
            parsed.append(RevisionEdit.from_dict(item))
        else:
            raise V9ValidationError(f"edits[{index}] 类型非法")

    original_hash = _sha256(body)
    if not parsed:
        return EditScriptResult(
            body=body,
            accepted=(),
            rejected=(),
            changed_chars=0,
            change_ratio=0.0,
            original_sha256=original_hash,
            result_sha256=original_hash,
        )

    try:
        located = _located_edits(body, parsed)
    except V9ValidationError as exc:
        rejected = tuple(
            PatchRejection(edit.edit_id, "invalid_script", str(exc)) for edit in parsed
        )
        return EditScriptResult(
            body=body,
            accepted=(),
            rejected=rejected,
            changed_chars=0,
            change_ratio=0.0,
            original_sha256=original_hash,
            result_sha256=original_hash,
        )

    changed_chars = sum(
        _replacement_change_cost(edit.old_text, edit.new_text) for edit in parsed
    )
    change_ratio = changed_chars / max(1, len(body))
    if change_ratio > ratio_limit:
        reason = f"总改动比例 {change_ratio:.3%} 超过上限 {ratio_limit:.3%}，整批回滚"
        rejected = tuple(
            PatchRejection(edit.edit_id, "change_ratio_exceeded", reason) for edit in parsed
        )
        return EditScriptResult(
            body=body,
            accepted=(),
            rejected=rejected,
            changed_chars=changed_chars,
            change_ratio=round(change_ratio, 6),
            original_sha256=original_hash,
            result_sha256=original_hash,
        )
    if max_change_chars is not None and changed_chars > max_change_chars:
        reason = (
            f"总改动字符数 {changed_chars} 超过上限 {max_change_chars}，整批回滚"
        )
        rejected = tuple(
            PatchRejection(edit.edit_id, "change_chars_exceeded", reason) for edit in parsed
        )
        return EditScriptResult(
            body=body,
            accepted=(),
            rejected=rejected,
            changed_chars=changed_chars,
            change_ratio=round(change_ratio, 6),
            original_sha256=original_hash,
            result_sha256=original_hash,
        )

    candidate = _apply_located(body, located)
    lost: list[tuple[RevisionEdit, str]] = []
    for edit in parsed:
        for protected in edit.preserve:
            if protected not in candidate:
                lost.append((edit, protected))
                break
    if lost:
        lost_ids = {edit.edit_id: protected for edit, protected in lost}
        rejected = tuple(
            PatchRejection(
                edit.edit_id,
                "protected_content_lost",
                (
                    f"保护内容在候选稿中消失：{lost_ids[edit.edit_id][:60]!r}"
                    if edit.edit_id in lost_ids
                    else "同批其他补丁导致保护内容丢失，整批回滚"
                ),
            )
            for edit in parsed
        )
        return EditScriptResult(
            body=body,
            accepted=(),
            rejected=rejected,
            changed_chars=changed_chars,
            change_ratio=round(change_ratio, 6),
            original_sha256=original_hash,
            result_sha256=original_hash,
        )

    return EditScriptResult(
        body=candidate,
        accepted=tuple(parsed),
        rejected=(),
        changed_chars=changed_chars,
        change_ratio=round(change_ratio, 6),
        original_sha256=original_hash,
        result_sha256=_sha256(candidate),
    )


__all__ = [
    "DEFAULT_DUPLICATE_MIN_CHARS",
    "DEFAULT_DUPLICATE_THRESHOLD",
    "DEFAULT_MAX_CHANGE_RATIO",
    "DuplicateCandidate",
    "EditCardRejection",
    "EditCardSelection",
    "EditScriptResult",
    "MechanicalQualityReport",
    "PatchRejection",
    "QualityFinding",
    "apply_edit_script",
    "duplicate_candidates",
    "mechanical_quality_report",
    "parse_revision_edits",
    "select_edit_cards",
]
