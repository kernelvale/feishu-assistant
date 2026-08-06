"""V9 精读管线的纯数据模型与严格 JSON 校验。

本模块刻意不依赖现有 V7/V8 的 Plan、Debt、IssueLedger 或模型客户端：
- NarrativeBrief 是作者可见的轻量叙事路线；
- AuditReport 只表达事实审查或读者审查的诊断结果，不含 pass/score；
- EditCard 是从审查意见筛出的有限编辑指令，不等同于最终文字补丁；
- RevisionEdit 是 Writer 返回的精确 old_text/new_text 补丁；
- VerificationResult 只验证一个补丁是否修对，不产生新编辑意见。

所有 ``from_dict`` 都拒绝未知字段、缺失字段和错误类型；所有 dataclass 在直接
构造时也执行同一组语义校验。字符串片段（draft_quote/span/old_text/new_text）保留
原始空白，避免破坏后续精确匹配。
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

BRIEF_SCHEMA_VERSION = "reading-v9-brief-v2"
BRIEF_TLDR_MAX_CHARS = 200
AUDIT_SCHEMA_VERSION = "reading-v9-audit-v1"
EDIT_CARD_SCHEMA_VERSION = "reading-v9-edit-card-v1"
REVISION_EDIT_SCHEMA_VERSION = "reading-v9-revision-edit-v1"
VERIFICATION_SCHEMA_VERSION = "reading-v9-verification-v1"

PAPER_TYPES = frozenset({"method", "experiment", "theory", "survey", "system", "other"})
BEAT_LENGTHS = frozenset({"short", "medium", "deep"})
AUDIT_KINDS = frozenset({"fact", "reader"})
AUDIT_STATUSES = frozenset({"complete", "unavailable"})
SEVERITIES = frozenset({"P0", "P1", "P2"})
FACT_ISSUE_KINDS = frozenset({
    "fact", "number", "figure", "formula", "causality", "contradiction", "core_omission",
})
READER_ISSUE_KINDS = frozenset({
    "duplication", "broken_reference", "fragmentation", "mechanical_tone",
    "figure_overuse", "missing_bridge", "missing_takeaway",
})
FACT_OPERATIONS = frozenset({"correct", "delete", "shorten", "bridge"})
READER_OPERATIONS = frozenset({"delete", "merge", "shorten", "move", "bridge"})
EDIT_CARD_OPERATIONS = FACT_OPERATIONS | READER_OPERATIONS
REVISION_OPERATIONS = frozenset({"delete", "replace", "merge", "shorten", "bridge", "move"})
EDIT_SCOPES = frozenset({"sentence", "paragraph", "adjacent_paragraphs"})

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$")


class V9ValidationError(ValueError):
    """V9 结构或语义校验失败。"""


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise V9ValidationError(f"{context} 必须是对象")
    if any(not isinstance(key, str) for key in value):
        raise V9ValidationError(f"{context} 的键必须都是字符串")
    return value


def _strict_keys(
    data: Mapping[str, Any],
    *,
    required: set[str] | frozenset[str],
    optional: set[str] | frozenset[str] = frozenset(),
    context: str,
) -> None:
    keys = set(data)
    missing = sorted(set(required) - keys)
    unknown = sorted(keys - set(required) - set(optional))
    if missing:
        raise V9ValidationError(f"{context} 缺少字段：{', '.join(missing)}")
    if unknown:
        raise V9ValidationError(f"{context} 含未知字段：{', '.join(unknown)}")


def _string(
    value: Any,
    field_name: str,
    *,
    allow_empty: bool = False,
    preserve: bool = False,
    max_chars: int | None = None,
) -> str:
    if not isinstance(value, str):
        raise V9ValidationError(f"{field_name} 必须是字符串")
    checked = value if preserve else value.strip()
    if not allow_empty and not checked.strip():
        raise V9ValidationError(f"{field_name} 不能为空")
    if max_chars is not None and len(checked) > max_chars:
        raise V9ValidationError(f"{field_name} 不能超过 {max_chars} 字符")
    return checked


def _bounded_string(
    value: Any,
    field_name: str,
    *,
    max_chars: int,
    allow_empty: bool = False,
) -> str:
    """校验字符串并确定性截断，避免单个展示字段击穿整个对象。"""
    text = _string(value, field_name, allow_empty=allow_empty)
    if len(text) <= max_chars:
        return text
    prefix = text[: max_chars - 1].rstrip(" ，、；：")
    return prefix + "…"


def _enum(value: Any, field_name: str, allowed: frozenset[str]) -> str:
    text = _string(value, field_name)
    if text not in allowed:
        raise V9ValidationError(
            f"{field_name} 非法：{text!r}；允许值为 {', '.join(sorted(allowed))}"
        )
    return text


def _identifier(value: Any, field_name: str) -> str:
    text = _string(value, field_name)
    if not _ID_RE.fullmatch(text):
        raise V9ValidationError(
            f"{field_name} 必须以字母或数字开头，且只能含字母、数字、._:-，最长 96 字符"
        )
    return text


def _boolean(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise V9ValidationError(f"{field_name} 必须是布尔值")
    return value


def _integer(
    value: Any,
    field_name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise V9ValidationError(f"{field_name} 必须是整数")
    if minimum is not None and value < minimum:
        raise V9ValidationError(f"{field_name} 不能小于 {minimum}")
    if maximum is not None and value > maximum:
        raise V9ValidationError(f"{field_name} 不能大于 {maximum}")
    return value


def _array(value: Any, field_name: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes, bytearray)):
        raise V9ValidationError(f"{field_name} 必须是数组")
    return value


def _string_tuple(
    value: Any,
    field_name: str,
    *,
    allow_empty: bool = True,
    preserve: bool = False,
    max_items: int | None = None,
) -> tuple[str, ...]:
    items = _array(value, field_name)
    if not allow_empty and not items:
        raise V9ValidationError(f"{field_name} 不能为空数组")
    if max_items is not None and len(items) > max_items:
        raise V9ValidationError(f"{field_name} 最多包含 {max_items} 项")
    result = tuple(
        _string(item, f"{field_name}[{index}]", preserve=preserve)
        for index, item in enumerate(items)
    )
    if len(set(result)) != len(result):
        raise V9ValidationError(f"{field_name} 不允许重复项")
    return result


def _normalize_tuple(value: Any, field_name: str, *, preserve: bool = False) -> tuple[str, ...]:
    """直接构造 dataclass 时，把 list/tuple 归一为不可变 tuple。"""
    return _string_tuple(value, field_name, preserve=preserve)


def _reject_json_constant(value: str) -> None:
    raise V9ValidationError(f"JSON 不允许非常数数值：{value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise V9ValidationError(f"JSON 对象含重复字段：{key}")
        result[key] = value
    return result


def decode_json_object(payload: str | bytes | bytearray, *, context: str = "JSON") -> dict[str, Any]:
    """严格解析一个 JSON 对象：拒绝重复键、NaN/Infinity 和非对象顶层值。"""
    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = bytes(payload).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise V9ValidationError(f"{context} 不是 UTF-8") from exc
    if not isinstance(payload, str):
        raise V9ValidationError(f"{context} 必须是 JSON 字符串或 UTF-8 字节")
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except V9ValidationError:
        raise
    except json.JSONDecodeError as exc:
        raise V9ValidationError(
            f"{context} 解析失败：第 {exc.lineno} 行第 {exc.colno} 列：{exc.msg}"
        ) from exc
    return dict(_mapping(value, context))


def encode_json_object(data: Mapping[str, Any], *, indent: int | None = 2) -> str:
    """稳定序列化 JSON 对象，不转义中文。"""
    _mapping(data, "JSON")
    return json.dumps(data, ensure_ascii=False, indent=indent, sort_keys=False, allow_nan=False)


@dataclass(frozen=True, slots=True)
class Beat:
    """Narrative Brief 中的一个读者问题节点。"""

    id: str
    title: str
    question: str
    purpose: str
    reader_takeaway: str
    source_refs: tuple[str, ...] = ()
    figure_refs: tuple[str, ...] = ()
    length: str = "medium"

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _identifier(self.id, "Beat.id"))
        object.__setattr__(self, "title", _string(self.title, "Beat.title", max_chars=80))
        object.__setattr__(self, "question", _string(self.question, "Beat.question"))
        object.__setattr__(self, "purpose", _string(self.purpose, "Beat.purpose"))
        object.__setattr__(
            self,
            "reader_takeaway",
            _string(self.reader_takeaway, "Beat.reader_takeaway", max_chars=360),
        )
        object.__setattr__(
            self, "source_refs", _normalize_tuple(self.source_refs, "Beat.source_refs")
        )
        object.__setattr__(
            self, "figure_refs", _normalize_tuple(self.figure_refs, "Beat.figure_refs")
        )
        object.__setattr__(self, "length", _enum(self.length, "Beat.length", BEAT_LENGTHS))

    @property
    def beat_id(self) -> str:
        return self.id

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Beat":
        data = _mapping(data, "Beat")
        _strict_keys(
            data,
            required={
                "id", "title", "question", "purpose", "reader_takeaway",
                "source_refs", "figure_refs", "length",
            },
            context="Beat",
        )
        return cls(
            id=_identifier(data["id"], "Beat.id"),
            title=_string(data["title"], "Beat.title", max_chars=80),
            question=_string(data["question"], "Beat.question"),
            purpose=_string(data["purpose"], "Beat.purpose"),
            reader_takeaway=_string(
                data["reader_takeaway"], "Beat.reader_takeaway", max_chars=360,
            ),
            source_refs=_string_tuple(data["source_refs"], "Beat.source_refs"),
            figure_refs=_string_tuple(data["figure_refs"], "Beat.figure_refs"),
            length=_enum(data["length"], "Beat.length", BEAT_LENGTHS),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "question": self.question,
            "purpose": self.purpose,
            "reader_takeaway": self.reader_takeaway,
            "source_refs": list(self.source_refs),
            "figure_refs": list(self.figure_refs),
            "length": self.length,
        }


@dataclass(frozen=True, slots=True)
class ConceptHome:
    """核心概念唯一的首次完整解释位置。"""

    concept: str
    beat_id: str
    full_explanation: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "concept", _string(self.concept, "ConceptHome.concept"))
        object.__setattr__(self, "beat_id", _identifier(self.beat_id, "ConceptHome.beat_id"))
        object.__setattr__(
            self,
            "full_explanation",
            _boolean(self.full_explanation, "ConceptHome.full_explanation"),
        )
        if not self.full_explanation:
            raise V9ValidationError("ConceptHome.full_explanation 必须为 true")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConceptHome":
        data = _mapping(data, "ConceptHome")
        _strict_keys(
            data,
            required={"concept", "beat_id", "full_explanation"},
            context="ConceptHome",
        )
        return cls(
            concept=_string(data["concept"], "ConceptHome.concept"),
            beat_id=_identifier(data["beat_id"], "ConceptHome.beat_id"),
            full_explanation=_boolean(
                data["full_explanation"], "ConceptHome.full_explanation"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept": self.concept,
            "beat_id": self.beat_id,
            "full_explanation": self.full_explanation,
        }


@dataclass(frozen=True, slots=True)
class FigureHome:
    """关键图表的一次主讲解位置与可选的一次回指位置。"""

    figure: str
    primary_beat_id: str
    callback_beat_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "figure", _string(self.figure, "FigureHome.figure"))
        object.__setattr__(
            self,
            "primary_beat_id",
            _identifier(self.primary_beat_id, "FigureHome.primary_beat_id"),
        )
        callback = _string(
            self.callback_beat_id,
            "FigureHome.callback_beat_id",
            allow_empty=True,
        )
        if callback:
            callback = _identifier(callback, "FigureHome.callback_beat_id")
        object.__setattr__(self, "callback_beat_id", callback)
        if callback and callback == self.primary_beat_id:
            raise V9ValidationError("FigureHome 的主讲解和回指位置不能相同")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FigureHome":
        data = _mapping(data, "FigureHome")
        _strict_keys(
            data,
            required={"figure", "primary_beat_id", "callback_beat_id"},
            context="FigureHome",
        )
        return cls(
            figure=_string(data["figure"], "FigureHome.figure"),
            primary_beat_id=_identifier(
                data["primary_beat_id"], "FigureHome.primary_beat_id"
            ),
            callback_beat_id=_string(
                data["callback_beat_id"],
                "FigureHome.callback_beat_id",
                allow_empty=True,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "figure": self.figure,
            "primary_beat_id": self.primary_beat_id,
            "callback_beat_id": self.callback_beat_id,
        }


@dataclass(frozen=True, slots=True)
class BriefTarget:
    """叙事简报中的全局篇幅护栏；它不是逐节字数合同。"""

    h2_max: int = 7
    total_heading_max: int = 18
    target_chars: int = 36_000

    def __post_init__(self) -> None:
        object.__setattr__(self, "h2_max", _integer(self.h2_max, "BriefTarget.h2_max", minimum=3, maximum=12))
        object.__setattr__(
            self,
            "total_heading_max",
            _integer(
                self.total_heading_max,
                "BriefTarget.total_heading_max",
                minimum=3,
                maximum=60,
            ),
        )
        object.__setattr__(
            self,
            "target_chars",
            _integer(self.target_chars, "BriefTarget.target_chars", minimum=1_000, maximum=200_000),
        )
        if self.total_heading_max < self.h2_max:
            raise V9ValidationError("BriefTarget.total_heading_max 不能小于 h2_max")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BriefTarget":
        data = _mapping(data, "BriefTarget")
        _strict_keys(
            data,
            required={"h2_max", "total_heading_max", "target_chars"},
            context="BriefTarget",
        )
        return cls(
            h2_max=_integer(data["h2_max"], "BriefTarget.h2_max", minimum=3, maximum=12),
            total_heading_max=_integer(
                data["total_heading_max"],
                "BriefTarget.total_heading_max",
                minimum=3,
                maximum=60,
            ),
            target_chars=_integer(
                data["target_chars"],
                "BriefTarget.target_chars",
                minimum=1_000,
                maximum=200_000,
            ),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "h2_max": self.h2_max,
            "total_heading_max": self.total_heading_max,
            "target_chars": self.target_chars,
        }


@dataclass(frozen=True, slots=True)
class NarrativeBrief:
    """V9 写作前的轻量叙事简报。"""

    schema_version: ClassVar[str] = BRIEF_SCHEMA_VERSION

    title_zh: str
    title_en: str
    paper_type: str
    reader_question: str
    paper_turn: str
    tldr: str
    contributions: tuple[str, ...]
    beats: tuple[Beat, ...]
    deep_dive_ids: tuple[str, ...]
    concept_homes: tuple[ConceptHome, ...]
    figure_homes: tuple[FigureHome, ...]
    exclusions: tuple[str, ...]
    target: BriefTarget

    def __post_init__(self) -> None:
        object.__setattr__(self, "title_zh", _string(self.title_zh, "NarrativeBrief.title_zh", max_chars=120))
        object.__setattr__(
            self,
            "title_en",
            _string(self.title_en, "NarrativeBrief.title_en", allow_empty=True, max_chars=240),
        )
        object.__setattr__(
            self,
            "paper_type",
            _enum(self.paper_type, "NarrativeBrief.paper_type", PAPER_TYPES),
        )
        object.__setattr__(
            self,
            "reader_question",
            _string(self.reader_question, "NarrativeBrief.reader_question"),
        )
        object.__setattr__(
            self,
            "paper_turn",
            _string(self.paper_turn, "NarrativeBrief.paper_turn"),
        )
        object.__setattr__(
            self,
            "tldr",
            _bounded_string(
                self.tldr, "NarrativeBrief.tldr",
                max_chars=BRIEF_TLDR_MAX_CHARS, allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "contributions",
            _normalize_tuple(self.contributions, "NarrativeBrief.contributions"),
        )
        object.__setattr__(self, "beats", tuple(self.beats))
        object.__setattr__(
            self,
            "deep_dive_ids",
            _normalize_tuple(self.deep_dive_ids, "NarrativeBrief.deep_dive_ids"),
        )
        object.__setattr__(self, "concept_homes", tuple(self.concept_homes))
        object.__setattr__(self, "figure_homes", tuple(self.figure_homes))
        object.__setattr__(
            self,
            "exclusions",
            _normalize_tuple(self.exclusions, "NarrativeBrief.exclusions"),
        )
        self.validate()

    def validate(self) -> None:
        if len(self.contributions) > 3:
            raise V9ValidationError("NarrativeBrief.contributions 最多 3 条")
        if not 5 <= len(self.beats) <= 7:
            raise V9ValidationError("NarrativeBrief.beats 必须包含 5～7 个 beat")
        if any(not isinstance(beat, Beat) for beat in self.beats):
            raise V9ValidationError("NarrativeBrief.beats 必须全部是 Beat")
        if any(not isinstance(home, ConceptHome) for home in self.concept_homes):
            raise V9ValidationError("NarrativeBrief.concept_homes 类型非法")
        if any(not isinstance(home, FigureHome) for home in self.figure_homes):
            raise V9ValidationError("NarrativeBrief.figure_homes 类型非法")
        if not isinstance(self.target, BriefTarget):
            raise V9ValidationError("NarrativeBrief.target 必须是 BriefTarget")

        beat_ids = [beat.id for beat in self.beats]
        if len(set(beat_ids)) != len(beat_ids):
            raise V9ValidationError("NarrativeBrief.beats 的 id 不允许重复")
        if len({beat.title for beat in self.beats}) != len(self.beats):
            raise V9ValidationError("NarrativeBrief.beats 的 title 不允许重复")

        if not 1 <= len(self.deep_dive_ids) <= 2:
            raise V9ValidationError("NarrativeBrief.deep_dive_ids 必须包含 1～2 个 beat id")
        unknown_deep = sorted(set(self.deep_dive_ids) - set(beat_ids))
        if unknown_deep:
            raise V9ValidationError(
                f"NarrativeBrief.deep_dive_ids 指向不存在的 beat：{', '.join(unknown_deep)}"
            )
        declared_deep = {beat.id for beat in self.beats if beat.length == "deep"}
        if declared_deep != set(self.deep_dive_ids):
            raise V9ValidationError("length=deep 的 beat 必须与 deep_dive_ids 完全一致")

        concepts = [home.concept.casefold() for home in self.concept_homes]
        if len(set(concepts)) != len(concepts):
            raise V9ValidationError("NarrativeBrief.concept_homes 的 concept 不允许重复")
        for home in self.concept_homes:
            if home.beat_id not in beat_ids:
                raise V9ValidationError(
                    f"概念 {home.concept!r} 指向不存在的 beat：{home.beat_id}"
                )

        figures = [re.sub(r"\s+", "", home.figure).casefold() for home in self.figure_homes]
        if len(set(figures)) != len(figures):
            raise V9ValidationError("NarrativeBrief.figure_homes 的 figure 不允许重复")
        for home in self.figure_homes:
            if home.primary_beat_id not in beat_ids:
                raise V9ValidationError(
                    f"图表 {home.figure!r} 的 primary_beat_id 不存在：{home.primary_beat_id}"
                )
            if home.callback_beat_id and home.callback_beat_id not in beat_ids:
                raise V9ValidationError(
                    f"图表 {home.figure!r} 的 callback_beat_id 不存在：{home.callback_beat_id}"
                )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NarrativeBrief":
        data = _mapping(data, "NarrativeBrief")
        _strict_keys(
            data,
            required={
                "title_zh", "title_en", "paper_type", "reader_question", "paper_turn",
                "tldr", "contributions", "beats", "deep_dive_ids", "concept_homes",
                "figure_homes", "exclusions", "target",
            },
            optional={"schema", "schema_version"},
            context="NarrativeBrief",
        )
        schema = data.get("schema", data.get("schema_version", BRIEF_SCHEMA_VERSION))
        if schema != BRIEF_SCHEMA_VERSION:
            raise V9ValidationError(f"NarrativeBrief schema 不兼容：{schema!r}")
        beats_raw = _array(data["beats"], "NarrativeBrief.beats")
        concept_raw = _array(data["concept_homes"], "NarrativeBrief.concept_homes")
        figure_raw = _array(data["figure_homes"], "NarrativeBrief.figure_homes")
        return cls(
            title_zh=_string(data["title_zh"], "NarrativeBrief.title_zh", max_chars=120),
            title_en=_string(
                data["title_en"], "NarrativeBrief.title_en", allow_empty=True, max_chars=240
            ),
            paper_type=_enum(data["paper_type"], "NarrativeBrief.paper_type", PAPER_TYPES),
            reader_question=_string(data["reader_question"], "NarrativeBrief.reader_question"),
            paper_turn=_string(data["paper_turn"], "NarrativeBrief.paper_turn"),
            tldr=_bounded_string(
                data["tldr"], "NarrativeBrief.tldr",
                max_chars=BRIEF_TLDR_MAX_CHARS, allow_empty=True,
            ),
            contributions=_string_tuple(
                data["contributions"], "NarrativeBrief.contributions", max_items=3
            ),
            beats=tuple(Beat.from_dict(item) for item in beats_raw),
            deep_dive_ids=_string_tuple(
                data["deep_dive_ids"], "NarrativeBrief.deep_dive_ids", allow_empty=False, max_items=2
            ),
            concept_homes=tuple(ConceptHome.from_dict(item) for item in concept_raw),
            figure_homes=tuple(FigureHome.from_dict(item) for item in figure_raw),
            exclusions=_string_tuple(data["exclusions"], "NarrativeBrief.exclusions"),
            target=BriefTarget.from_dict(_mapping(data["target"], "NarrativeBrief.target")),
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> "NarrativeBrief":
        return cls.from_dict(decode_json_object(payload, context="NarrativeBrief JSON"))

    @classmethod
    def fallback(
        cls,
        title_zh: str = "论文精读",
        *,
        title_en: str = "",
    ) -> "NarrativeBrief":
        """返回不编造论文事实的五步兜底路线，供 Brief 服务不可用时继续产草稿。"""
        beats = (
            Beat(
                id="beat-1",
                title="问题从哪里来",
                question="这篇论文要解决的真实问题是什么？",
                purpose="先让读者知道后续方法为何有必要。",
                reader_takeaway="读者应先明确论文要解决的问题，以及旧做法为什么不足。",
                length="medium",
            ),
            Beat(
                id="beat-2",
                title="已有方法为何不够",
                question="现有做法卡在什么地方？",
                purpose="建立论文提出新方案之前的认知缺口。",
                reader_takeaway="读者应能说出论文真正改变了哪一个关键环节，而不只是知道它新增了一个模块。",
                length="medium",
            ),
            Beat(
                id="beat-3",
                title="核心机制如何工作",
                question="论文最关键的机制怎样一步步运作？",
                purpose="用一个连续过程完成全文唯一一次核心机制深讲。",
                reader_takeaway="读者应能复述核心机制如何从输入经过关键步骤走到输出，并理解每一步为何存在。",
                length="deep",
            ),
            Beat(
                id="beat-4",
                title="证据说明了什么",
                question="实验或论证如何支持核心机制？",
                purpose="把证据用于检验前面的解释，而不是重复方法介绍。",
                reader_takeaway="读者应能分清证据支持了哪条判断、没有支持哪条更强的外推。",
                length="medium",
            ),
            Beat(
                id="beat-5",
                title="边界与最终理解",
                question="哪些结论可以成立，哪些仍不能推出？",
                purpose="收束论文贡献、成本与适用范围，不按章节复述全文。",
                reader_takeaway="读者应带走论文贡献成立的范围、主要代价和不能外推的边界。",
                length="short",
            ),
        )
        return cls(
            title_zh=_string(title_zh, "NarrativeBrief.fallback.title_zh", max_chars=120),
            title_en=_string(
                title_en,
                "NarrativeBrief.fallback.title_en",
                allow_empty=True,
                max_chars=240,
            ),
            paper_type="other",
            reader_question="这篇论文解决了什么问题，核心机制为何有效，又有哪些边界？",
            paper_turn="沿着问题、机制、证据与边界建立一条连续阅读主线。",
            tldr="叙事简报不可用，作者将依据论文来源按五步通用路线生成可审查草稿。",
            contributions=(),
            beats=beats,
            deep_dive_ids=("beat-3",),
            concept_homes=(),
            figure_homes=(),
            exclusions=("来源不能确认的旁支、数字和图表不展开",),
            target=BriefTarget(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "title_zh": self.title_zh,
            "title_en": self.title_en,
            "paper_type": self.paper_type,
            "reader_question": self.reader_question,
            "paper_turn": self.paper_turn,
            "tldr": self.tldr,
            "contributions": list(self.contributions),
            "beats": [beat.to_dict() for beat in self.beats],
            "deep_dive_ids": list(self.deep_dive_ids),
            "concept_homes": [home.to_dict() for home in self.concept_homes],
            "figure_homes": [home.to_dict() for home in self.figure_homes],
            "exclusions": list(self.exclusions),
            "target": self.target.to_dict(),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return encode_json_object(self.to_dict(), indent=indent)


@dataclass(frozen=True, slots=True)
class AuditIssue:
    """事实审查或读者审查中的一条精确诊断。"""

    audit_kind: str
    issue_id: str
    severity: str
    kind: str
    operation: str
    required_outcome: str
    max_scope: str
    draft_quote: str = ""
    source_location: str = ""
    source_evidence: str = ""
    problem: str = ""
    span_a: str = ""
    span_b: str = ""
    keep: str = "none"
    reader_harm: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "audit_kind", _enum(self.audit_kind, "AuditIssue.audit_kind", AUDIT_KINDS))
        object.__setattr__(self, "issue_id", _identifier(self.issue_id, "AuditIssue.issue_id"))
        object.__setattr__(self, "severity", _enum(self.severity, "AuditIssue.severity", SEVERITIES))
        object.__setattr__(self, "required_outcome", _string(self.required_outcome, "AuditIssue.required_outcome"))
        object.__setattr__(self, "max_scope", _enum(self.max_scope, "AuditIssue.max_scope", EDIT_SCOPES))
        self.validate()

    def validate(self) -> None:
        if self.audit_kind == "fact":
            _enum(self.kind, "AuditIssue.kind", FACT_ISSUE_KINDS)
            _enum(self.operation, "AuditIssue.operation", FACT_OPERATIONS)
            if self.severity not in SEVERITIES:
                raise V9ValidationError("事实审查 severity 必须是 P0/P1/P2")
            if self.max_scope not in {"sentence", "paragraph"}:
                raise V9ValidationError("事实审查 max_scope 只能是 sentence 或 paragraph")
            _string(self.draft_quote, "AuditIssue.draft_quote", preserve=True)
            _string(self.source_location, "AuditIssue.source_location")
            _string(self.source_evidence, "AuditIssue.source_evidence")
            _string(self.problem, "AuditIssue.problem")
            if any((self.span_a, self.span_b, self.reader_harm)) or self.keep != "none":
                raise V9ValidationError("事实审查 issue 不能携带读者审查字段")
            return

        _enum(self.kind, "AuditIssue.kind", READER_ISSUE_KINDS)
        _enum(self.operation, "AuditIssue.operation", READER_OPERATIONS)
        if self.severity not in {"P1", "P2"}:
            raise V9ValidationError("读者审查 severity 只能是 P1 或 P2")
        _string(self.span_a, "AuditIssue.span_a", preserve=True)
        _string(self.reader_harm, "AuditIssue.reader_harm")
        if self.keep not in {"a", "b", "none"}:
            raise V9ValidationError("AuditIssue.keep 只能是 a、b 或 none")
        if self.kind == "duplication":
            _string(self.span_b, "AuditIssue.span_b", preserve=True)
            if self.keep not in {"a", "b"}:
                raise V9ValidationError("duplication issue 必须明确 keep=a 或 keep=b")
            if self.operation not in {"delete", "merge", "shorten", "move"}:
                raise V9ValidationError("duplication issue 的 operation 非法")
        elif self.keep != "none":
            raise V9ValidationError("非 duplication issue 的 keep 必须为 none")
        if any((self.draft_quote, self.source_location, self.source_evidence, self.problem)):
            raise V9ValidationError("读者审查 issue 不能携带事实审查字段")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, kind: str) -> "AuditIssue":
        data = _mapping(data, "AuditIssue")
        audit_kind = _enum(kind, "AuditIssue.audit_kind", AUDIT_KINDS)
        if audit_kind == "fact":
            _strict_keys(
                data,
                required={
                    "issue_id", "severity", "kind", "draft_quote", "source_location",
                    "source_evidence", "problem", "required_outcome", "operation", "max_scope",
                },
                context="Fact AuditIssue",
            )
            return cls(
                audit_kind="fact",
                issue_id=_identifier(data["issue_id"], "AuditIssue.issue_id"),
                severity=_enum(data["severity"], "AuditIssue.severity", SEVERITIES),
                kind=_enum(data["kind"], "AuditIssue.kind", FACT_ISSUE_KINDS),
                operation=_enum(data["operation"], "AuditIssue.operation", FACT_OPERATIONS),
                required_outcome=_string(data["required_outcome"], "AuditIssue.required_outcome"),
                max_scope=_enum(data["max_scope"], "AuditIssue.max_scope", EDIT_SCOPES),
                draft_quote=_string(data["draft_quote"], "AuditIssue.draft_quote", preserve=True),
                source_location=_string(data["source_location"], "AuditIssue.source_location"),
                source_evidence=_string(data["source_evidence"], "AuditIssue.source_evidence"),
                problem=_string(data["problem"], "AuditIssue.problem"),
            )

        _strict_keys(
            data,
            required={
                "issue_id", "severity", "kind", "span_a", "span_b", "keep", "operation",
                "reader_harm", "required_outcome", "max_scope",
            },
            context="Reader AuditIssue",
        )
        return cls(
            audit_kind="reader",
            issue_id=_identifier(data["issue_id"], "AuditIssue.issue_id"),
            severity=_enum(data["severity"], "AuditIssue.severity", SEVERITIES),
            kind=_enum(data["kind"], "AuditIssue.kind", READER_ISSUE_KINDS),
            operation=_enum(data["operation"], "AuditIssue.operation", READER_OPERATIONS),
            required_outcome=_string(data["required_outcome"], "AuditIssue.required_outcome"),
            max_scope=_enum(data["max_scope"], "AuditIssue.max_scope", EDIT_SCOPES),
            span_a=_string(data["span_a"], "AuditIssue.span_a", preserve=True),
            span_b=_string(
                data["span_b"], "AuditIssue.span_b", allow_empty=True, preserve=True
            ),
            keep=_string(data["keep"], "AuditIssue.keep"),
            reader_harm=_string(data["reader_harm"], "AuditIssue.reader_harm"),
        )

    def to_dict(self) -> dict[str, Any]:
        if self.audit_kind == "fact":
            return {
                "issue_id": self.issue_id,
                "severity": self.severity,
                "kind": self.kind,
                "draft_quote": self.draft_quote,
                "source_location": self.source_location,
                "source_evidence": self.source_evidence,
                "problem": self.problem,
                "required_outcome": self.required_outcome,
                "operation": self.operation,
                "max_scope": self.max_scope,
            }
        return {
            "issue_id": self.issue_id,
            "severity": self.severity,
            "kind": self.kind,
            "span_a": self.span_a,
            "span_b": self.span_b,
            "keep": self.keep,
            "operation": self.operation,
            "reader_harm": self.reader_harm,
            "required_outcome": self.required_outcome,
            "max_scope": self.max_scope,
        }


@dataclass(frozen=True, slots=True)
class AuditReport:
    """一份专项审查报告；kind 由调用方指定，防止空 issues 时猜错类型。"""

    schema_version: ClassVar[str] = AUDIT_SCHEMA_VERSION

    kind: str
    status: str
    issues: tuple[AuditIssue, ...] = ()
    checked_claims: tuple[str, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(self.kind, "AuditReport.kind", AUDIT_KINDS))
        object.__setattr__(self, "status", _enum(self.status, "AuditReport.status", AUDIT_STATUSES))
        object.__setattr__(self, "issues", tuple(self.issues))
        object.__setattr__(
            self,
            "checked_claims",
            _normalize_tuple(self.checked_claims, "AuditReport.checked_claims"),
        )
        object.__setattr__(
            self,
            "reason",
            _string(self.reason, "AuditReport.reason", allow_empty=True),
        )
        self.validate()

    def validate(self) -> None:
        if any(not isinstance(issue, AuditIssue) for issue in self.issues):
            raise V9ValidationError("AuditReport.issues 必须全部是 AuditIssue")
        if any(issue.audit_kind != self.kind for issue in self.issues):
            raise V9ValidationError("AuditReport 中 issue 的 audit_kind 与报告 kind 不一致")
        issue_ids = [issue.issue_id for issue in self.issues]
        if len(set(issue_ids)) != len(issue_ids):
            raise V9ValidationError("AuditReport.issue_id 不允许重复")
        if self.kind == "reader" and len(self.issues) > 5:
            raise V9ValidationError("读者审查最多报告 5 条问题")
        if self.kind == "fact" and len(self.issues) > 30:
            raise V9ValidationError("事实审查最多报告 30 条问题")
        if self.kind == "reader" and self.checked_claims:
            raise V9ValidationError("读者审查不能携带 checked_claims")
        if self.status == "unavailable":
            if self.issues:
                raise V9ValidationError("unavailable 的 AuditReport.issues 必须为空")
            if not self.reason:
                raise V9ValidationError("unavailable 的 AuditReport.reason 不能为空")
        elif self.reason:
            raise V9ValidationError("complete 的 AuditReport.reason 必须为空")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, kind: str) -> "AuditReport":
        data = _mapping(data, "AuditReport")
        audit_kind = _enum(kind, "AuditReport.kind", AUDIT_KINDS)
        if audit_kind == "fact":
            _strict_keys(
                data,
                required={"status", "issues", "checked_claims", "reason"},
                optional={"schema", "schema_version"},
                context="Fact AuditReport",
            )
        else:
            _strict_keys(
                data,
                required={"status", "issues", "reason"},
                optional={"schema", "schema_version"},
                context="Reader AuditReport",
            )
        schema = data.get("schema", data.get("schema_version", AUDIT_SCHEMA_VERSION))
        if schema != AUDIT_SCHEMA_VERSION:
            raise V9ValidationError(f"AuditReport schema 不兼容：{schema!r}")
        status = _enum(data["status"], "AuditReport.status", AUDIT_STATUSES)
        issues_raw = _array(data["issues"], "AuditReport.issues")
        checked = (
            _string_tuple(data["checked_claims"], "AuditReport.checked_claims")
            if audit_kind == "fact"
            else ()
        )
        return cls(
            kind=audit_kind,
            status=status,
            issues=tuple(AuditIssue.from_dict(item, kind=audit_kind) for item in issues_raw),
            checked_claims=checked,
            reason=_string(data["reason"], "AuditReport.reason", allow_empty=True),
        )

    @classmethod
    def from_json(
        cls,
        payload: str | bytes | bytearray,
        *,
        kind: str,
    ) -> "AuditReport":
        return cls.from_dict(
            decode_json_object(payload, context=f"{kind} AuditReport JSON"),
            kind=kind,
        )

    @classmethod
    def unavailable(cls, kind: str, reason: str) -> "AuditReport":
        return cls(
            kind=_enum(kind, "AuditReport.kind", AUDIT_KINDS),
            status="unavailable",
            issues=(),
            checked_claims=(),
            reason=_string(reason, "AuditReport.reason"),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "status": self.status,
            "issues": [issue.to_dict() for issue in self.issues],
        }
        if self.kind == "fact":
            data["checked_claims"] = list(self.checked_claims)
        data["reason"] = self.reason
        return data

    def to_json(self, *, indent: int | None = 2) -> str:
        return encode_json_object(self.to_dict(), indent=indent)


@dataclass(frozen=True, slots=True)
class EditCard:
    """交回唯一作者的有限编辑卡；它定位问题，但不强迫作者重写整章。"""

    schema_version: ClassVar[str] = EDIT_CARD_SCHEMA_VERSION

    card_id: str
    issue_ids: tuple[str, ...]
    audit_kind: str
    severity: str
    operation: str
    target_quote: str
    required_result: str
    max_scope: str
    problem: str = ""
    evidence: str = ""
    protected_content: tuple[str, ...] = ()
    paired_quote: str = ""
    keep: str = "none"
    new_text: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "card_id", _identifier(self.card_id, "EditCard.card_id"))
        object.__setattr__(
            self,
            "issue_ids",
            _normalize_tuple(self.issue_ids, "EditCard.issue_ids"),
        )
        object.__setattr__(self, "audit_kind", _enum(self.audit_kind, "EditCard.audit_kind", AUDIT_KINDS))
        object.__setattr__(self, "severity", _enum(self.severity, "EditCard.severity", SEVERITIES))
        object.__setattr__(self, "operation", _enum(self.operation, "EditCard.operation", EDIT_CARD_OPERATIONS))
        object.__setattr__(
            self,
            "target_quote",
            _string(self.target_quote, "EditCard.target_quote", preserve=True),
        )
        object.__setattr__(
            self,
            "required_result",
            _string(self.required_result, "EditCard.required_result"),
        )
        object.__setattr__(self, "max_scope", _enum(self.max_scope, "EditCard.max_scope", EDIT_SCOPES))
        object.__setattr__(
            self,
            "problem",
            _string(self.problem, "EditCard.problem", allow_empty=True),
        )
        object.__setattr__(
            self,
            "evidence",
            _string(self.evidence, "EditCard.evidence", allow_empty=True),
        )
        object.__setattr__(
            self,
            "protected_content",
            _normalize_tuple(
                self.protected_content,
                "EditCard.protected_content",
                preserve=True,
            ),
        )
        object.__setattr__(
            self,
            "paired_quote",
            _string(
                self.paired_quote,
                "EditCard.paired_quote",
                allow_empty=True,
                preserve=True,
            ),
        )
        if self.keep not in {"a", "b", "none"}:
            raise V9ValidationError("EditCard.keep 只能是 a、b 或 none")
        if self.new_text is not None:
            if not isinstance(self.new_text, str):
                raise V9ValidationError("EditCard.new_text 必须是字符串或 null")
            if self.new_text == self.target_quote:
                raise V9ValidationError("EditCard.new_text 不能与 target_quote 完全相同")
        self.validate()

    @property
    def old_text(self) -> str:
        return self.target_quote

    @property
    def target_text(self) -> str:
        return self.target_quote

    def validate(self) -> None:
        if not self.issue_ids:
            raise V9ValidationError("EditCard.issue_ids 不能为空")
        for issue_id in self.issue_ids:
            _identifier(issue_id, "EditCard.issue_ids[]")
        if self.audit_kind == "fact" and not self.evidence:
            raise V9ValidationError("事实 EditCard.evidence 不能为空")
        if self.audit_kind == "fact" and self.operation not in FACT_OPERATIONS:
            raise V9ValidationError("事实 EditCard.operation 非法")
        if self.audit_kind == "reader" and self.operation not in READER_OPERATIONS:
            raise V9ValidationError("读者 EditCard.operation 非法")
        if self.operation == "merge" and not self.paired_quote:
            raise V9ValidationError("merge EditCard 必须提供 paired_quote")
        if self.operation == "delete" and self.new_text not in (None, ""):
            raise V9ValidationError("delete EditCard 的 new_text 只能为空或 null")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EditCard":
        data = _mapping(data, "EditCard")
        _strict_keys(
            data,
            required={
                "card_id", "issue_ids", "audit_kind", "severity", "operation",
                "required_result", "max_scope",
            },
            optional={
                "schema", "schema_version", "target_quote", "old_text", "problem", "evidence",
                "protected_content", "paired_quote", "keep", "new_text",
            },
            context="EditCard",
        )
        schema = data.get("schema", data.get("schema_version", EDIT_CARD_SCHEMA_VERSION))
        if schema != EDIT_CARD_SCHEMA_VERSION:
            raise V9ValidationError(f"EditCard schema 不兼容：{schema!r}")
        has_target = "target_quote" in data
        has_old = "old_text" in data
        if not has_target and not has_old:
            raise V9ValidationError("EditCard 必须提供 target_quote 或 old_text")
        if has_target and has_old and data["target_quote"] != data["old_text"]:
            raise V9ValidationError("EditCard.target_quote 与 old_text 不一致")
        target = data["target_quote"] if has_target else data["old_text"]
        return cls(
            card_id=_identifier(data["card_id"], "EditCard.card_id"),
            issue_ids=_string_tuple(data["issue_ids"], "EditCard.issue_ids", allow_empty=False),
            audit_kind=_enum(data["audit_kind"], "EditCard.audit_kind", AUDIT_KINDS),
            severity=_enum(data["severity"], "EditCard.severity", SEVERITIES),
            operation=_enum(data["operation"], "EditCard.operation", EDIT_CARD_OPERATIONS),
            target_quote=_string(target, "EditCard.target_quote", preserve=True),
            required_result=_string(data["required_result"], "EditCard.required_result"),
            max_scope=_enum(data["max_scope"], "EditCard.max_scope", EDIT_SCOPES),
            problem=_string(data.get("problem", ""), "EditCard.problem", allow_empty=True),
            evidence=_string(data.get("evidence", ""), "EditCard.evidence", allow_empty=True),
            protected_content=_string_tuple(
                data.get("protected_content", []),
                "EditCard.protected_content",
                preserve=True,
            ),
            paired_quote=_string(
                data.get("paired_quote", ""),
                "EditCard.paired_quote",
                allow_empty=True,
                preserve=True,
            ),
            keep=_string(data.get("keep", "none"), "EditCard.keep"),
            new_text=(
                None
                if data.get("new_text") is None
                else _string(
                    data["new_text"],
                    "EditCard.new_text",
                    allow_empty=True,
                    preserve=True,
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id,
            "issue_ids": list(self.issue_ids),
            "audit_kind": self.audit_kind,
            "severity": self.severity,
            "operation": self.operation,
            "target_quote": self.target_quote,
            "required_result": self.required_result,
            "max_scope": self.max_scope,
            "problem": self.problem,
            "evidence": self.evidence,
            "protected_content": list(self.protected_content),
            "paired_quote": self.paired_quote,
            "keep": self.keep,
            "new_text": self.new_text,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return encode_json_object(self.to_dict(), indent=indent)


@dataclass(frozen=True, slots=True)
class RevisionEdit:
    """Writer 产出的一个精确文字补丁。"""

    schema_version: ClassVar[str] = REVISION_EDIT_SCHEMA_VERSION

    edit_id: str
    operation: str
    old_text: str
    new_text: str
    reason: str
    preserve: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "edit_id", _identifier(self.edit_id, "RevisionEdit.edit_id"))
        object.__setattr__(
            self,
            "operation",
            _enum(self.operation, "RevisionEdit.operation", REVISION_OPERATIONS),
        )
        object.__setattr__(
            self,
            "old_text",
            _string(self.old_text, "RevisionEdit.old_text", preserve=True),
        )
        object.__setattr__(
            self,
            "new_text",
            _string(
                self.new_text,
                "RevisionEdit.new_text",
                allow_empty=True,
                preserve=True,
            ),
        )
        object.__setattr__(self, "reason", _string(self.reason, "RevisionEdit.reason"))
        object.__setattr__(
            self,
            "preserve",
            _normalize_tuple(self.preserve, "RevisionEdit.preserve", preserve=True),
        )
        if self.old_text == self.new_text:
            raise V9ValidationError("RevisionEdit.old_text 与 new_text 不能完全相同")
        if self.operation == "delete" and self.new_text != "":
            raise V9ValidationError("delete 补丁的 new_text 必须为空字符串")
        if self.operation != "delete" and not self.new_text.strip():
            raise V9ValidationError("非 delete 补丁的 new_text 不能为空")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RevisionEdit":
        data = _mapping(data, "RevisionEdit")
        _strict_keys(
            data,
            required={"edit_id", "operation", "old_text", "new_text", "reason", "preserve"},
            optional={"schema", "schema_version"},
            context="RevisionEdit",
        )
        schema = data.get("schema", data.get("schema_version", REVISION_EDIT_SCHEMA_VERSION))
        if schema != REVISION_EDIT_SCHEMA_VERSION:
            raise V9ValidationError(f"RevisionEdit schema 不兼容：{schema!r}")
        return cls(
            edit_id=_identifier(data["edit_id"], "RevisionEdit.edit_id"),
            operation=_enum(data["operation"], "RevisionEdit.operation", REVISION_OPERATIONS),
            old_text=_string(data["old_text"], "RevisionEdit.old_text", preserve=True),
            new_text=_string(
                data["new_text"], "RevisionEdit.new_text", allow_empty=True, preserve=True
            ),
            reason=_string(data["reason"], "RevisionEdit.reason"),
            preserve=_string_tuple(
                data["preserve"], "RevisionEdit.preserve", preserve=True
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "edit_id": self.edit_id,
            "operation": self.operation,
            "old_text": self.old_text,
            "new_text": self.new_text,
            "reason": self.reason,
            "preserve": list(self.preserve),
        }


# 更直观的补丁别名；两者是同一个严格模型。
TextPatch = RevisionEdit


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """一个补丁的定点验证结果。"""

    schema_version: ClassVar[str] = VERIFICATION_SCHEMA_VERSION

    edit_id: str
    target_fixed: bool
    new_fact_error: bool
    protected_content_lost: bool
    broken_reference: bool
    decision: str
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "edit_id", _identifier(self.edit_id, "VerificationResult.edit_id"))
        object.__setattr__(self, "target_fixed", _boolean(self.target_fixed, "VerificationResult.target_fixed"))
        object.__setattr__(self, "new_fact_error", _boolean(self.new_fact_error, "VerificationResult.new_fact_error"))
        object.__setattr__(
            self,
            "protected_content_lost",
            _boolean(self.protected_content_lost, "VerificationResult.protected_content_lost"),
        )
        object.__setattr__(
            self,
            "broken_reference",
            _boolean(self.broken_reference, "VerificationResult.broken_reference"),
        )
        if self.decision not in {"accept", "rollback"}:
            raise V9ValidationError("VerificationResult.decision 只能是 accept 或 rollback")
        object.__setattr__(self, "reason", _string(self.reason, "VerificationResult.reason"))
        safe = (
            self.target_fixed
            and not self.new_fact_error
            and not self.protected_content_lost
            and not self.broken_reference
        )
        if (self.decision == "accept") != safe:
            raise V9ValidationError(
                "VerificationResult.decision 与四项验证布尔值不一致；只有全部安全才能 accept"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VerificationResult":
        data = _mapping(data, "VerificationResult")
        _strict_keys(
            data,
            required={
                "edit_id", "target_fixed", "new_fact_error", "protected_content_lost",
                "broken_reference", "decision", "reason",
            },
            optional={"schema", "schema_version"},
            context="VerificationResult",
        )
        schema = data.get("schema", data.get("schema_version", VERIFICATION_SCHEMA_VERSION))
        if schema != VERIFICATION_SCHEMA_VERSION:
            raise V9ValidationError(f"VerificationResult schema 不兼容：{schema!r}")
        return cls(
            edit_id=_identifier(data["edit_id"], "VerificationResult.edit_id"),
            target_fixed=_boolean(data["target_fixed"], "VerificationResult.target_fixed"),
            new_fact_error=_boolean(data["new_fact_error"], "VerificationResult.new_fact_error"),
            protected_content_lost=_boolean(
                data["protected_content_lost"], "VerificationResult.protected_content_lost"
            ),
            broken_reference=_boolean(
                data["broken_reference"], "VerificationResult.broken_reference"
            ),
            decision=_string(data["decision"], "VerificationResult.decision"),
            reason=_string(data["reason"], "VerificationResult.reason"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "edit_id": self.edit_id,
            "target_fixed": self.target_fixed,
            "new_fact_error": self.new_fact_error,
            "protected_content_lost": self.protected_content_lost,
            "broken_reference": self.broken_reference,
            "decision": self.decision,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """验证服务的整批结果；unavailable 时必须为空并触发上层回滚。"""

    status: str
    results: tuple[VerificationResult, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _enum(self.status, "VerificationReport.status", AUDIT_STATUSES))
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(
            self,
            "reason",
            _string(self.reason, "VerificationReport.reason", allow_empty=True),
        )
        if any(not isinstance(item, VerificationResult) for item in self.results):
            raise V9ValidationError("VerificationReport.results 类型非法")
        ids = [item.edit_id for item in self.results]
        if len(set(ids)) != len(ids):
            raise V9ValidationError("VerificationReport.edit_id 不允许重复")
        if self.status == "unavailable":
            if self.results:
                raise V9ValidationError("unavailable 的 VerificationReport.results 必须为空")
            if not self.reason:
                raise V9ValidationError("unavailable 的 VerificationReport.reason 不能为空")
        elif self.reason:
            raise V9ValidationError("complete 的 VerificationReport.reason 必须为空")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VerificationReport":
        data = _mapping(data, "VerificationReport")
        _strict_keys(
            data,
            required={"status", "results", "reason"},
            context="VerificationReport",
        )
        raw = _array(data["results"], "VerificationReport.results")
        return cls(
            status=_enum(data["status"], "VerificationReport.status", AUDIT_STATUSES),
            results=tuple(VerificationResult.from_dict(item) for item in raw),
            reason=_string(data["reason"], "VerificationReport.reason", allow_empty=True),
        )

    @classmethod
    def unavailable(cls, reason: str) -> "VerificationReport":
        return cls(status="unavailable", results=(), reason=_string(reason, "reason"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "results": [item.to_dict() for item in self.results],
            "reason": self.reason,
        }


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "BRIEF_SCHEMA_VERSION",
    "EDIT_CARD_SCHEMA_VERSION",
    "REVISION_EDIT_SCHEMA_VERSION",
    "VERIFICATION_SCHEMA_VERSION",
    "AuditIssue",
    "AuditReport",
    "Beat",
    "BriefTarget",
    "ConceptHome",
    "EditCard",
    "FigureHome",
    "NarrativeBrief",
    "RevisionEdit",
    "TextPatch",
    "V9ValidationError",
    "VerificationReport",
    "VerificationResult",
    "decode_json_object",
    "encode_json_object",
]
