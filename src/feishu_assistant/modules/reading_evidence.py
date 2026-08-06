"""长论文证据索引：分块只是导航，写作和审核仍回到原文。

这里不生成可发布摘要。每张证据卡都绑定 chunk_id；教学计划引用 chunk_id，Writer
据此重新装载原始 Markdown，避免“根据摘要写精读”造成事实和跨章节关系损失。
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..kimi import FrozenKimiClient, KimiClient, KimiError
from .reading_graph import content_digest
from .reading_workflow import ReadingWorkspace

logger = logging.getLogger("feishu_assistant.reading.evidence")

# 全文直通与长文证据索引必须共用同一个边界。旧实现这里是 120k，
# ``paper_size_tier`` 却是 200k，导致 120k～200k 的论文一边被当成长文建卡，
# 一边又被当短文分配写作/追问预算，预算协议自相矛盾。
LONG_PAPER_CHARS = 200_000
SOURCE_CHUNK_CHARS = 24_000
EVIDENCE_INDEX_WORKERS = 6  # 证据块彼此独立，受控并行建卡；不影响缓存（块间无共享前缀）。
                            # 3→6（08-05）：块间无依赖，唯一风险是网关限流（用户已批准，实测后定）
EVIDENCE_SCHEMA_DIGEST = "evidence-card-v2"  # {statement,quote,chunk_id} 条目结构
SOURCE_PACKET_CHARS = 70_000
# 规划器索引上限。80k 是按「小卡×块数」拍脑袋的时代值；实测 45 图 111 页综述归并后
# 仍需 ~85k。256k 上下文模型读 160k（~45k token）索引毫无压力，超出才需要分层归并。
INDEX_MAX_CHARS = 160_000

# 论文尺寸统一分档：≤200k 字符（约 80 页内）各处全文直通，超出走长文路径。
# 估算宁大勿小（用户裁定）：2603(21页)≈70k、50页论文≈150k 都进 short；
# 2509(120页综述)≈417k 进 long。k3 为 1M token 窗口，200k 字符 ≈ 12 万 token，余量充足。
SHORT_PAPER_CHARS = 200_000
# 长文分层来源包里未命中块的证据卡摘要单块上限
LAYERED_SUMMARY_CHARS = 1_200


@dataclass(frozen=True)
class PaperBudgetProfile:
    """一篇论文从索引到追问共用的尺寸预算。

    ``max_tokens`` 只是上限，不会迫使模型写够；因此预算同时包含输入包上限、
    推荐正文长度和必须达到的正文下限。字符预算按混合中英文约 2 chars/token
    保守估算，并在 256k token 级上下文内预留输出、reasoning 和提示词空间。
    """

    tier: str
    paper_chars: int
    planner_chars: int
    author_packet_chars: int
    fact_packet_chars: int
    followup_retrieval_chars: int
    followup_output_tokens: int
    brief_output_tokens: int
    evidence_output_tokens: int
    author_output_tokens: int
    audit_output_tokens: int
    revision_output_tokens: int
    verify_output_tokens: int
    author_parts: int
    recommended_draft_chars: int
    minimum_draft_chars: int
    # 单次请求全部输入（来源 + 已写正文 + prompt 框架）的软上限。
    context_input_chars: int = 430_000

    @property
    def long_mode(self) -> bool:
        return self.paper_chars > SHORT_PAPER_CHARS

    def source_budget(self, stage: str, *, fixed_chars: int = 0) -> int:
        """扣除已写正文/Brief 后，给当前来源包留下安全预算。"""
        caps = {
            "planner": self.planner_chars,
            "author": self.author_packet_chars,
            "fact": self.fact_packet_chars,
            "followup": self.followup_retrieval_chars,
        }
        if stage not in caps:
            raise ValueError(f"未知预算阶段：{stage}")
        remaining = self.context_input_chars - max(0, int(fixed_chars))
        # 低于 32k 时来源已经不足以支撑长文判断，宁可让上层显式失败/降级，
        # 不把一个很小的前缀伪装成完整来源。
        return max(32_000, min(caps[stage], remaining))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bounded(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))


def paper_budget_profile(paper_chars: int) -> PaperBudgetProfile:
    """按论文尺寸返回统一输入/输出预算与正文深度下限。

    分档不直接把输出长度线性绑定到原文字数，而是给综述/长系统论文更高的
    讲解空间。``minimum_draft_chars`` 是发布前硬下限；模型给出的 Brief target
    可以比推荐值更高，但不能把它压低。
    """
    chars = max(0, int(paper_chars))
    if chars <= 20_000:
        recommended = _bounded(round(chars * 0.25), 3_000, 8_000)
        minimum = _bounded(round(chars * 0.12), 600, 3_500)
        return PaperBudgetProfile(
            tier="compact", paper_chars=chars,
            planner_chars=40_000, author_packet_chars=50_000,
            fact_packet_chars=60_000, followup_retrieval_chars=60_000,
            followup_output_tokens=8_192,
            brief_output_tokens=8_192, evidence_output_tokens=12_288,
            author_output_tokens=16_384, audit_output_tokens=10_240,
            revision_output_tokens=10_240, verify_output_tokens=8_192,
            author_parts=1, recommended_draft_chars=recommended,
            minimum_draft_chars=minimum,
        )
    if chars <= 80_000:
        recommended = _bounded(round(chars * 0.18), 8_000, 18_000)
        minimum = _bounded(round(chars * 0.10), 5_000, 12_000)
        return PaperBudgetProfile(
            tier="short", paper_chars=chars,
            planner_chars=100_000, author_packet_chars=110_000,
            fact_packet_chars=120_000, followup_retrieval_chars=120_000,
            followup_output_tokens=10_240,
            brief_output_tokens=12_288, evidence_output_tokens=14_336,
            author_output_tokens=24_576, audit_output_tokens=12_288,
            revision_output_tokens=12_288, verify_output_tokens=10_240,
            author_parts=1, recommended_draft_chars=recommended,
            minimum_draft_chars=minimum,
        )
    if chars <= SHORT_PAPER_CHARS:
        recommended = _bounded(round(chars * 0.15), 18_000, 32_000)
        minimum = _bounded(round(chars * 0.09), 12_000, 22_000)
        return PaperBudgetProfile(
            tier="standard", paper_chars=chars,
            planner_chars=220_000, author_packet_chars=220_000,
            fact_packet_chars=240_000, followup_retrieval_chars=200_000,
            followup_output_tokens=12_288,
            brief_output_tokens=16_384, evidence_output_tokens=16_384,
            author_output_tokens=32_768, audit_output_tokens=16_384,
            revision_output_tokens=16_384, verify_output_tokens=12_288,
            author_parts=2, recommended_draft_chars=recommended,
            minimum_draft_chars=minimum,
        )
    if chars <= 450_000:
        recommended = _bounded(round(chars * 0.12), 38_000, 52_000)
        minimum = _bounded(round(chars * 0.078), 28_000, 36_000)
        return PaperBudgetProfile(
            tier="long", paper_chars=chars,
            planner_chars=260_000, author_packet_chars=320_000,
            fact_packet_chars=340_000, followup_retrieval_chars=240_000,
            followup_output_tokens=16_384,
            brief_output_tokens=20_480, evidence_output_tokens=20_480,
            author_output_tokens=32_768, audit_output_tokens=20_480,
            revision_output_tokens=20_480, verify_output_tokens=16_384,
            author_parts=3, recommended_draft_chars=recommended,
            minimum_draft_chars=minimum,
        )
    recommended = _bounded(round(chars * 0.10), 52_000, 80_000)
    minimum = _bounded(round(chars * 0.068), 36_000, 52_000)
    return PaperBudgetProfile(
        tier="extra_long", paper_chars=chars,
        planner_chars=340_000, author_packet_chars=360_000,
        fact_packet_chars=370_000, followup_retrieval_chars=300_000,
        followup_output_tokens=20_480,
        brief_output_tokens=24_576, evidence_output_tokens=24_576,
        author_output_tokens=32_768, audit_output_tokens=24_576,
        revision_output_tokens=24_576, verify_output_tokens=16_384,
        author_parts=4, recommended_draft_chars=recommended,
        minimum_draft_chars=minimum,
    )


def paper_size_tier(paper_chars: int) -> str:
    """统一的论文尺寸分档："short" 全文直通 / "long" 各预算走长文路径。"""
    return "long" if paper_budget_profile(paper_chars).long_mode else "short"

_FRAGMENT_SUFFIX_RE = re.compile(r"（片段\s*\d+）\s*$")
_PACK_SUFFIX_RE = re.compile(r"\s+等\s*\d+\s*节\s*$")


def _chunk_base_title(title: str) -> str:
    """块标题的归属基名：剥掉「等 N 节」与「（片段 N）」，用于亲属片段识别。"""
    base = _PACK_SUFFIX_RE.sub("", str(title))
    base = _FRAGMENT_SUFFIX_RE.sub("", base)
    return base.strip()


_PLANNER_ENTRY_FIELDS = (
    "definitions", "claims", "mechanisms", "evidence", "limitations",
)
_PLANNER_ENTRY_CHARS = 180
_PLANNER_CARD_ITEMS = 3
_PLANNER_GROUP_CHARS = 720


def _clip_navigation_text(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip(" ，、；：") + "…"


def _navigation_entry_text(value: Any) -> str:
    if isinstance(value, dict):
        statement = str(value.get("statement") or "").strip()
        quote = str(value.get("quote") or "").strip()
        if statement and quote and quote != statement:
            return f"{statement}（原文：{quote}）"
        return statement or quote
    return str(value or "").strip()


def _entry_statement(value: Any) -> str:
    """证据卡条目只取 statement（不含长引文）；旧字符串条目原样返回。"""
    if isinstance(value, dict):
        return str(value.get("statement") or value.get("quote") or "").strip()
    return str(value or "").strip()


def _compact_evidence_card(card: Any) -> dict[str, Any]:
    """删除重复长引文但保留每块的可规划信号和 chunk_id。"""
    data = asdict(card) if hasattr(card, "__dataclass_fields__") else dict(card or {})
    compact: dict[str, Any] = {
        "chunk_id": str(data.get("chunk_id") or ""),
        "section": str(data.get("section") or ""),
        "orientation": _clip_navigation_text(data.get("orientation"), 260),
    }
    for field_name in _PLANNER_ENTRY_FIELDS:
        values = data.get(field_name) or []
        if not isinstance(values, list):
            values = [values]
        entries = []
        for value in values[:_PLANNER_CARD_ITEMS]:
            entry = _clip_navigation_text(_navigation_entry_text(value), _PLANNER_ENTRY_CHARS)
            if entry:
                entries.append(entry)
        compact[field_name] = entries
    for field_name, limit in (("open_questions", 3), ("relations", 3),
                              ("figures", 8), ("formulas", 8)):
        values = data.get(field_name) or []
        if not isinstance(values, list):
            values = [values]
        compact[field_name] = [
            _clip_navigation_text(value, 120) for value in values[:limit]
            if str(value or "").strip()
        ]
    return compact


def _merge_navigation_cards(
    cards: list[dict[str, Any]],
    *,
    group_size: int,
    signal_chars: int,
) -> list[dict[str, Any]]:
    """把连续证据卡归并为规划目录；所有 chunk_id 都保留，内容按类别压缩。"""
    merged: list[dict[str, Any]] = []
    for start in range(0, len(cards), max(1, group_size)):
        group = cards[start:start + max(1, group_size)]
        signals: dict[str, str] = {}
        for field_name in _PLANNER_ENTRY_FIELDS:
            values = [
                str(value)
                for card in group
                for value in (card.get(field_name) or [])
                if str(value).strip()
            ]
            signals[field_name] = _clip_navigation_text("；".join(values), signal_chars)
        orientations = [
            str(card.get("orientation") or "") for card in group
            if str(card.get("orientation") or "").strip()
        ]
        merged.append({
            "group_id": f"evidence-group-{start // max(1, group_size) + 1:03d}",
            "chunk_ids": [str(card.get("chunk_id") or "") for card in group],
            "sections": [str(card.get("section") or "") for card in group],
            "orientation": _clip_navigation_text("；".join(orientations), signal_chars),
            "signals": signals,
            "figures": list(dict.fromkeys(
                str(value) for card in group for value in (card.get("figures") or [])
            ))[:12],
            "formulas": list(dict.fromkeys(
                str(value) for card in group for value in (card.get("formulas") or [])
            ))[:12],
        })
    return merged


def _planner_json(material: Any) -> str:
    return json.dumps(material, ensure_ascii=False, separators=(",", ":"))


class EvidenceError(Exception):
    """证据索引不完整；调用方必须停止发布，不能静默跳过。"""


@dataclass(frozen=True)
class SourceChunk:
    chunk_id: str
    title: str
    text: str
    order: int
    start: int = -1       # 在 paper_md 中的绝对偏移（切分时生成，禁止 find 推算）
    end: int = -1
    digest: str = ""


@dataclass
class EvidenceCard:
    chunk_id: str
    section: str
    orientation: str = ""
    definitions: list[dict[str, str]] = field(default_factory=list)
    claims: list[dict[str, str]] = field(default_factory=list)
    mechanisms: list[dict[str, str]] = field(default_factory=list)
    evidence: list[dict[str, str]] = field(default_factory=list)
    limitations: list[dict[str, str]] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)
    formulas: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any], chunk: SourceChunk) -> "EvidenceCard":
        def items(name: str, limit: int = 12) -> list[str]:
            raw = data.get(name)
            if not isinstance(raw, list):
                return []
            return [str(v).strip() for v in raw if str(v).strip()][:limit]

        def entries(name: str, limit: int = 12) -> list[dict[str, str]]:
            """结构化条目：{statement, quote, chunk_id}；旧字符串按兼容策略包装。"""
            raw = data.get(name)
            if not isinstance(raw, list):
                return []
            out: list[dict[str, str]] = []
            for entry in raw[:limit]:
                if isinstance(entry, dict):
                    statement = str(entry.get("statement") or "").strip()
                    quote = str(entry.get("quote") or "").strip()
                    if statement and quote:
                        out.append({"statement": statement, "quote": quote,
                                    "chunk_id": str(entry.get("chunk_id") or chunk.chunk_id)})
                elif str(entry).strip():
                    legacy = str(entry).strip()
                    out.append({"statement": legacy, "quote": legacy[:120],
                                "chunk_id": chunk.chunk_id})
            return out

        return cls(
            chunk_id=chunk.chunk_id,
            section=chunk.title,
            orientation=str(data.get("orientation") or "").strip(),
            definitions=entries("definitions"),
            claims=entries("claims"),
            mechanisms=entries("mechanisms"),
            evidence=entries("evidence"),
            limitations=entries("limitations"),
            open_questions=items("open_questions"),
            relations=items("relations"),
            figures=items("figures", 30),
            formulas=items("formulas", 30),
        )


@dataclass(frozen=True)
class SourcePacket:
    """原文包及其覆盖情况。

    ``source_packet`` 过去只返回字符串，调用方无法知道关键词召回或字符预算
    是否把原文静默截掉。V9 只把这个结构化信息用于发布护栏；旧 V4/V8
    调用仍可继续使用字符串接口。
    """

    text: str
    requested_ids: tuple[str, ...] = ()
    selected_ids: tuple[str, ...] = ()
    candidate_ids: tuple[str, ...] = ()
    omitted_ids: tuple[str, ...] = ()
    truncated_ids: tuple[str, ...] = ()
    used_fallback: bool = False

    @property
    def complete(self) -> bool:
        return bool(self.selected_ids) and not (
            self.omitted_ids or self.truncated_ids or self.used_fallback
        )


@dataclass
class EvidenceBundle:
    chunks: list[SourceChunk]
    cards: list[EvidenceCard]
    long_mode: bool
    # 单块失败只降级该块；其余证据卡仍可参与 Brief 规划。
    degraded_chunks: dict[str, str] = field(default_factory=dict)
    index_metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_map(self) -> dict[str, SourceChunk]:
        return {chunk.chunk_id: chunk for chunk in self.chunks}

    def planner_material(self, paper_md: str, max_chars: int = INDEX_MAX_CHARS) -> str:
        if type(max_chars) is not int or max_chars <= 0:
            raise ValueError("max_chars 必须是正整数")
        if not self.long_mode:
            return paper_md
        prefix = (
            "以下是逐章节建立的证据索引。每项保留 source chunk id；这是规划导航，"
            "不是允许脱离原文写作的摘要。\n"
        )
        full = prefix + "\n".join(
            json.dumps(asdict(card), ensure_ascii=False, separators=(",", ":"))
            for card in self.cards
        )
        if len(full) <= max_chars:
            return full

        # 先去掉重复长引文，保留每块的 statement/机制/证据和全部 chunk id。
        compact_cards = [_compact_evidence_card(card) for card in self.cards]
        compact = (
            prefix
            + "本索引因字符预算采用卡片级分层归并；每个 chunk_id 均可用于回装原文。\n"
            + _planner_json(compact_cards)
        )
        if len(compact) <= max_chars:
            logger.info(
                "长文证据索引启用卡片级分层归并：%d → %d 字符",
                len(full), len(compact),
            )
            return compact

        # 仍超限时按原文顺序做第二层归并；组内只保留规划信号，chunk_ids 不丢，
        # Writer 后续仍通过这些 id 回装原文全文，不会因为索引压缩而改走关键词召回。
        for group_size, signal_chars in ((8, _PLANNER_GROUP_CHARS), (32, 260), (128, 120)):
            groups = _merge_navigation_cards(
                compact_cards, group_size=group_size, signal_chars=signal_chars,
            )
            layered = (
                prefix
                + "本索引因字符预算采用分层归并；group 内的 chunk_ids 均可用于回装原文。\n"
                + _planner_json(groups)
            )
            if len(layered) <= max_chars:
                logger.info(
                    "长文证据索引启用分组归并：%d → %d 字符（group_size=%d）",
                    len(full), len(layered), group_size,
                )
                return layered

        raise EvidenceError(
            f"证据索引分层归并后仍过长（{len(layered)} 字符）；"
            f"当前索引预算 {max_chars} 字符，需要增加预算，不能截掉尾部"
        )

    def source_packet_info(
        self,
        refs: list[str] | None,
        hints: list[str] | None = None,
        max_chars: int = SOURCE_PACKET_CHARS,
    ) -> SourcePacket:
        """装载原文并显式报告是否发生召回降级、块省略或块内截断。

        精确 source ref 仍然是首选；无 ref 或无法由上层解析时才做关键词召回。
        关键词召回和任何预算截断都被视为 ``complete=False``，避免长论文在
        依据不完整时被误标成已验证终稿。
        """
        if type(max_chars) is not int or max_chars <= 0:
            raise ValueError("max_chars 必须是正整数")
        refs = list(dict.fromkeys(
            str(v).strip() for v in (refs or []) if str(v).strip()
        ))
        strict_refs = bool(refs)
        mapping = self.chunk_map
        selected = [mapping[ref] for ref in refs if ref in mapping]
        if refs and len(selected) != len(set(refs)):
            missing = sorted(set(refs) - mapping.keys())
            raise EvidenceError("教学计划引用了不存在的原文块：" + "、".join(missing))
        if strict_refs and selected:
            # 大章切片会把一段技术论述撕进多个「片段 N」块；计划只点其中一个时，
            # 把同基名的亲属片段一并打包，避免写手拿到半截论述。
            bases = {_chunk_base_title(chunk.title) for chunk in selected}
            expanded = [chunk for chunk in self.chunks
                        if _chunk_base_title(chunk.title) in bases]
            if len(expanded) > len(selected):
                logger.info("引用片段自动并入亲属块：%d → %d", len(selected), len(expanded))
            selected = expanded

        used_fallback = False
        if not selected:
            used_fallback = True
            terms = {
                term.casefold()
                for hint in (hints or [])
                for term in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,8}", hint)
            }
            ranked: list[tuple[int, int, SourceChunk]] = []
            for chunk in self.chunks:
                haystack = f"{chunk.title}\n{chunk.text}".casefold()
                score = sum(3 if term in chunk.title.casefold() else 1
                            for term in terms if term in haystack)
                ranked.append((score, -chunk.order, chunk))
            selected = [item[2] for item in sorted(ranked, reverse=True) if item[0] > 0]
        if not selected:
            used_fallback = True
            selected = self.chunks[:1]

        requested_ids = tuple(refs)
        candidate_ids = tuple(chunk.chunk_id for chunk in sorted(selected, key=lambda item: item.order))
        omitted_ids: list[str] = []
        truncated_ids: list[str] = []
        included_ids: list[str] = []
        parts: list[str] = []
        used = 0
        for chunk in sorted(selected, key=lambda item: item.order):
            header = f"\n\n--- SOURCE {chunk.chunk_id} · {chunk.title} ---\n"
            remaining = max_chars - used - len(header)
            if remaining <= 0:
                omitted_ids.append(chunk.chunk_id)
                continue
            if remaining < len(chunk.text):
                # 首块即使超预算也保留可见前缀，但必须显式标记为截断；
                # 后续块则整体省略，避免把半截块伪装成完整证据。
                if not parts:
                    parts.append(header + chunk.text[:remaining])
                    used += len(header) + remaining
                    truncated_ids.append(chunk.chunk_id)
                    included_ids.append(chunk.chunk_id)
                else:
                    omitted_ids.append(chunk.chunk_id)
                continue
            parts.append(header + chunk.text)
            used += len(header) + len(chunk.text)
            included_ids.append(chunk.chunk_id)

        packet = "".join(parts).strip()
        if omitted_ids:
            packet += ("\n\n（注：本单元计划引用的部分原文块因单包容量限制未装入："
                       + "、".join(omitted_ids)
                       + "；相关事实缺失时不得补写或推断。）")
        if truncated_ids:
            packet += ("\n\n（注：原文块 " + "、".join(truncated_ids)
                       + " 只装入了前缀；来源覆盖不完整，不得据此断言未出现的细节。）")
        return SourcePacket(
            text=packet,
            requested_ids=requested_ids,
            selected_ids=tuple(included_ids),
            candidate_ids=candidate_ids,
            omitted_ids=tuple(omitted_ids),
            truncated_ids=tuple(truncated_ids),
            used_fallback=used_fallback,
        )

    def source_packet(
        self,
        refs: list[str] | None,
        hints: list[str] | None = None,
        max_chars: int = SOURCE_PACKET_CHARS,
    ) -> str:
        """兼容旧调用的字符串接口；V9 使用 ``source_packet_info`` 获取覆盖状态。"""
        return self.source_packet_info(refs, hints=hints, max_chars=max_chars).text

    def _chunk_summary(self, chunk: SourceChunk, max_chars: int) -> str:
        """分层包中未命中块的摘要：优先证据卡 statements，缺卡时取原文开头摘录。"""
        card = next((item for item in self.cards if item.chunk_id == chunk.chunk_id), None)
        lines: list[str] = []
        if card is not None:
            orientation = str(getattr(card, "orientation", "") or "").strip()
            if orientation:
                lines.append(orientation)
            for field_name, label in (("definitions", "定义"), ("claims", "主张"),
                                      ("mechanisms", "机制"), ("evidence", "证据"),
                                      ("limitations", "局限")):
                for entry in (getattr(card, field_name, None) or [])[:3]:
                    statement = _entry_statement(entry)
                    if statement:
                        lines.append(f"{label}：{statement}")
        if not lines:
            excerpt = re.sub(r"\s+", " ", chunk.text).strip()
            lines.append(f"原文摘录：{excerpt}")
        return _clip_navigation_text("\n".join(lines), max_chars)

    def layered_source_packet_info(
        self,
        refs: list[str] | None,
        hints: list[str] | None = None,
        max_chars: int = SOURCE_PACKET_CHARS,
    ) -> SourcePacket:
        """长文分层来源包：命中块给全文，未命中块给证据卡摘要，绝不截尾丢弃。

        与 ``source_packet_info`` 的差别：旧接口超预算会省略/截断未装入的块，
        长文下（如实测 417k 字符综述）会把覆盖缺口误报成「退回关键词召回」降级。
        这里所有块都以全文或摘要形式出现，预算只决定多少块装得下全文；
        未解析引用/问题提示只影响哪些块提升为全文，不再触发召回降级，
        因此 used_fallback 恒为 False（来源覆盖天然完整）。
        """
        if type(max_chars) is not int or max_chars <= 0:
            raise ValueError("max_chars 必须是正整数")
        refs = list(dict.fromkeys(
            str(v).strip() for v in (refs or []) if str(v).strip()
        ))
        mapping = self.chunk_map
        selected = [mapping[ref] for ref in refs if ref in mapping]
        if refs and len(selected) != len(set(refs)):
            missing = sorted(set(refs) - mapping.keys())
            raise EvidenceError("教学计划引用了不存在的原文块：" + "、".join(missing))
        ordered = sorted(self.chunks, key=lambda item: item.order)
        # 精确引用优先；大章片段的亲属块并入（与 source_packet_info 同一规则）
        ref_ids: list[str] = []
        if selected:
            bases = {_chunk_base_title(chunk.title) for chunk in selected}
            ref_ids = [chunk.chunk_id for chunk in ordered
                       if _chunk_base_title(chunk.title) in bases]
        # 提示词（未解析引用、beat 问题等）重叠的块补为全文候选，按重叠度排序；
        # 预算装不下的候选自动降为摘要块，不省略不截断
        terms = {
            term.casefold()
            for hint in (hints or [])
            for term in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[一-鿿]{2,8}", hint)
        }
        promoted: list[tuple[int, str]] = []
        if terms:
            for chunk in ordered:
                if chunk.chunk_id in ref_ids:
                    continue
                haystack = f"{chunk.title}\n{chunk.text}".casefold()
                score = sum(3 if term in chunk.title.casefold() else 1
                            for term in terms if term in haystack)
                if score > 0:
                    promoted.append((score, chunk.chunk_id))
        promoted.sort(key=lambda item: -item[0])  # 同分保持原文顺序（sort 稳定）
        full_candidates = ref_ids + [chunk_id for _, chunk_id in promoted]

        def header(chunk: SourceChunk, kind: str) -> str:
            suffix = "（证据卡摘要）" if kind == "SUMMARY" else ""
            return f"\n\n--- {kind} {chunk.chunk_id} · {chunk.title}{suffix} ---\n"

        # 摘要单块上限随块数自适应：先给 70% 预算；极端块数下继续压缩，
        # 直到所有 chunk id 与最小导航信号都能完整装入。不能为了声称“覆盖完整”
        # 而悄悄让最终字符串超过 max_chars。
        per_cap = min(
            LAYERED_SUMMARY_CHARS,
            max(40, (max_chars * 7 // 10) // max(1, len(ordered)) - 80),
        )
        summaries: dict[str, str] = {}
        summary_cost = max_chars + 1
        for candidate_cap in dict.fromkeys((per_cap, 120, 80, 40)):
            if candidate_cap > per_cap:
                continue
            summaries = {
                chunk.chunk_id: self._chunk_summary(chunk, candidate_cap)
                for chunk in ordered
            }
            summary_cost = sum(
                len(header(chunk, "SUMMARY")) + len(summaries[chunk.chunk_id])
                for chunk in ordered
            )
            if summary_cost <= max_chars:
                break
        if summary_cost > max_chars:
            raise EvidenceError(
                f"长文分层来源包仅保留最小摘要仍需 {summary_cost} 字符，"
                f"超过预算 {max_chars}；不能截掉尾部块"
            )

        full_budget = max_chars - summary_cost
        full_set: set[str] = set()
        used = 0
        for chunk_id in full_candidates:
            chunk = mapping[chunk_id]
            extra = (len(header(chunk, "SOURCE")) + len(chunk.text)
                     - len(header(chunk, "SUMMARY")) - len(summaries[chunk_id]))
            if used + extra <= full_budget:
                full_set.add(chunk_id)
                used += extra
        parts = [
            (header(chunk, "SOURCE") + chunk.text) if chunk.chunk_id in full_set
            else (header(chunk, "SUMMARY") + summaries[chunk.chunk_id])
            for chunk in ordered
        ]
        text = "".join(parts).strip()
        if len(text) > max_chars:
            raise EvidenceError(
                f"长文分层来源包组装后为 {len(text)} 字符，超过预算 {max_chars}"
            )
        return SourcePacket(
            text=text,
            requested_ids=tuple(refs),
            selected_ids=tuple(
                chunk.chunk_id for chunk in ordered if chunk.chunk_id in full_set
            ),
            candidate_ids=tuple(chunk.chunk_id for chunk in ordered),
            omitted_ids=(),
            truncated_ids=(),
            used_fallback=False,
        )

    def review_material(self, max_chars: int = INDEX_MAX_CHARS) -> str:
        """全局审核使用可追溯证据索引；逐章事实审核使用 source_packet 原文。"""
        if not self.long_mode:
            return ""
        return self.planner_material("", max_chars=max_chars)


def split_markdown_sections(md: str) -> list[tuple[str, str]]:
    """按 H1~H3 切分 Markdown，忽略 ``` / ~~~ 代码围栏中的伪标题。

    围栏闭合遵循 CommonMark：仅当收尾 fence 与开头同字符且长度 ≥ 开头。
    因此 ```` 外层内的 ``` 是内容而非闭合（同字符嵌套不会提前关围栏）。
    """
    headings: list[tuple[int, str]] = []
    offset = 0
    fence_char = ""
    fence_len = 0
    for line in md.splitlines(keepends=True):
        stripped = line.lstrip()
        fence = re.match(r"(`{3,}|~{3,})", stripped)
        if fence:
            marker = fence.group(1)
            if not fence_char:
                fence_char, fence_len = marker[0], len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_len:
                fence_char, fence_len = "", 0
            offset += len(line)
            continue
        if not fence_char:
            match = re.match(r"^#{1,3}\s+(.+?)\s*$", line.rstrip("\r\n"))
            if match:
                headings.append((offset, match.group(1).strip()))
        offset += len(line)
    if not headings:
        return [("全文", md)]
    result: list[tuple[str, str]] = []
    if headings[0][0] > 0 and md[:headings[0][0]].strip():
        result.append(("文首", md[:headings[0][0]]))
    for idx, (start, title) in enumerate(headings):
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(md)
        result.append((title, md[start:end]))
    return result


def split_source(md: str, chunk_chars: int = SOURCE_CHUNK_CHARS) -> list[SourceChunk]:
    """按章节切；超长章节再按段落切；相邻小块归并到 ≤ chunk_chars 的包。

    任何字符都进入某个块，不做尾部截断。只切不合会造成小节密集论文（如综述）
    碎块爆炸：块均几千字符 → 卡数翻倍 → 证据索引超出规划上限。
    """
    raw: list[tuple[str, str]] = []
    for title, section in split_markdown_sections(md):
        if len(section) <= chunk_chars:
            raw.append((title, section))
            continue
        paragraphs = re.split(r"(?=\n\n+)", section)
        current = ""
        part = 1
        for paragraph in paragraphs:
            if current and len(current) + len(paragraph) > chunk_chars:
                raw.append((f"{title}（片段 {part}）", current))
                part += 1
                current = ""
            while len(paragraph) > chunk_chars:
                if current:
                    raw.append((f"{title}（片段 {part}）", current))
                    part += 1
                    current = ""
                raw.append((f"{title}（片段 {part}）", paragraph[:chunk_chars]))
                part += 1
                paragraph = paragraph[chunk_chars:]
            current += paragraph
        if current:
            raw.append((f"{title}（片段 {part}）", current))

    # 归并相邻小块：每包 ≤ chunk_chars，标题取首节（多节附「等 N 节」）。
    merged: list[tuple[str, str]] = []
    buf_title, buf_text, buf_count = "", "", 0
    for title, text in raw:
        if buf_text and len(buf_text) + len(text) > chunk_chars:
            suffix = f" 等{buf_count}节" if buf_count > 1 else ""
            merged.append((buf_title + suffix, buf_text))
            buf_title, buf_text, buf_count = "", "", 0
        if not buf_text:
            buf_title = title
        buf_text += text
        buf_count += 1
    if buf_text:
        suffix = f" 等{buf_count}节" if buf_count > 1 else ""
        merged.append((buf_title + suffix, buf_text))

    # 偏移在切分时生成：片段是源序连续子串，前移游标精确定位（非事后 find 推算）
    chunks: list[SourceChunk] = []
    cursor = 0
    for idx, (title, text) in enumerate(merged, 1):
        if not text.strip():
            continue
        start = md.find(text, cursor)
        if start < 0:
            start = cursor  # 理论上不可达；保底不炸
        end = start + len(text)
        cursor = end
        chunks.append(SourceChunk(
            f"chunk-{idx:04d}", title, text, idx,
            start=start, end=end,
            digest=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        ))
    return chunks


def _json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fenced = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        # 编码向模型常把 JSON 裹在解释文字里：取首尾大括号的最大跨度再试一次
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise EvidenceError("证据卡返回非法 JSON") from exc
        try:
            value = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            raise EvidenceError("证据卡返回非法 JSON") from exc
    if not isinstance(value, dict):
        raise EvidenceError("证据卡必须是 JSON 对象")
    return value


def _source_fallback_card(chunk: SourceChunk) -> EvidenceCard:
    """模型格式连续异常时，从原文生成最保守的导航卡。

    证据卡只负责帮助 Planner 定位；Writer 随后仍会凭 chunk_id 重新读取完整原文。
    因此这里不猜测定义、结论或因果关系，只保留章节名和原文开头的可核对摘录，
    比因为一次非 JSON 输出终止整篇更安全。
    """
    excerpts: list[str] = []
    for block in re.split(r"\n\s*\n", chunk.text):
        text = block.strip()
        if not text:
            continue
        text = re.sub(r"(?m)^#{1,6}\s+.*$", "", text).strip()
        # 图片链接本身不提供导航语义；其图号仍由后续图表资产管线处理。
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 20:
            continue
        excerpts.append(text[:700])
        if len(excerpts) >= 3 or sum(map(len, excerpts)) >= 1_600:
            break
    excerpt = " ".join(excerpts)[:1_800]
    orientation = f"原文导航摘录：{excerpt or chunk.title}"
    return EvidenceCard(
        chunk_id=chunk.chunk_id,
        section=chunk.title,
        orientation=orientation,
    )


class EvidenceIndexer:
    def __init__(self, client: Any, prompts_dir: Path) -> None:
        self.client = client
        self.prompt = (prompts_dir / "reading_evidence.md").read_text(encoding="utf-8")

    def build(
        self,
        paper_md: str,
        *,
        workspace: ReadingWorkspace | None = None,
        progress: Callable[[str], None] = lambda _: None,
    ) -> tuple[EvidenceBundle, int]:
        chunks = split_source(paper_md)
        budget = paper_budget_profile(len(paper_md))
        long_mode = budget.long_mode
        if not long_mode:
            return EvidenceBundle(chunks, [], False), 0

        prompt_digest = content_digest(self.prompt)

        telemetry_lock = threading.Lock()
        active_calls = 0
        max_active_calls = 0
        call_sequence = 0

        def chat_with_telemetry(
            chunk: SourceChunk,
            messages: list[dict[str, Any]],
            *,
            attempt: int,
            heavy: bool,
            reasoning_effort: str,
            max_tokens: int | None = None,
        ) -> str:
            nonlocal active_calls, max_active_calls, call_sequence
            with telemetry_lock:
                call_sequence += 1
                sequence = call_sequence
                active_calls += 1
                max_active_calls = max(max_active_calls, active_calls)
                inflight = active_calls
            started = time.perf_counter()
            logger.info(
                "证据块调用开始：chunk=%s attempt=%d heavy=%s inflight=%d seq=%d",
                chunk.chunk_id, attempt, heavy, inflight, sequence,
            )
            try:
                kwargs: dict[str, Any] = {
                    "heavy": heavy,
                    "json_mode": True,
                    "reasoning_effort": reasoning_effort,
                }
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                try:
                    return self.client.chat(messages, **kwargs)
                except TypeError as exc:
                    # 历史测试桩/旧适配器只接受 low_reasoning 布尔值。只有明确的
                    # “未知关键字”才做本地参数翻译；请求已经真正发出后的业务
                    # TypeError 不能盲目重发。
                    message = str(exc).casefold()
                    if not any(token in message for token in (
                        "unexpected keyword", "got an unexpected keyword",
                        "keyword-only argument",
                    )):
                        raise
                    legacy_kwargs = dict(kwargs)
                    legacy_kwargs.pop("reasoning_effort", None)
                    legacy_kwargs["low_reasoning"] = reasoning_effort == "low"
                    return self.client.chat(messages, **legacy_kwargs)
            finally:
                elapsed = time.perf_counter() - started
                with telemetry_lock:
                    active_calls -= 1
                    remaining = active_calls
                logger.info(
                    "证据块调用结束：chunk=%s attempt=%d elapsed=%.1fs remaining=%d seq=%d",
                    chunk.chunk_id, attempt, elapsed, remaining, sequence,
                )

        def build_one(chunk: SourceChunk) -> tuple[EvidenceCard, int, str]:
            key = f"evidence-{chunk.chunk_id}"
            cached = workspace.get_json(key) if workspace else None
            if isinstance(cached, dict):
                # 卡签名：chunk 内容 + schema + prompt 摘要任一不一致即重建本块
                sig = cached.get("_cache_sig") or {}
                if (sig.get("chunk_digest") == chunk.digest
                        and sig.get("schema") == EVIDENCE_SCHEMA_DIGEST
                        and sig.get("prompt") == prompt_digest):
                    try:
                        card = EvidenceCard.from_dict(cached, chunk)
                        reason = str(cached.get("_degraded_reason") or "")
                        if not reason and card.orientation.startswith("原文导航摘录："):
                            reason = "缓存中的确定性原文导航卡"
                        return card, 0, reason
                    except (TypeError, ValueError, KeyError) as exc:
                        logger.warning(
                            "原文块 %s 证据缓存非法，改为本块重建：%s",
                            chunk.chunk_id, exc,
                        )

            prompt = self.prompt.replace("{{CHUNK_ID}}", chunk.chunk_id)
            prompt = prompt.replace("{{SECTION_TITLE}}", chunk.title)
            prompt = prompt.replace("{{SOURCE_CONTENT}}", chunk.text)
            calls = 0
            reason = ""

            def card_from(text: str) -> EvidenceCard:
                return EvidenceCard.from_dict(_json_object(text), chunk)

            try:
                calls += 1
                raw = chat_with_telemetry(
                    chunk,
                    [{"role": "user", "content": prompt}],
                    attempt=1,
                    heavy=True,
                    reasoning_effort="high",
                    max_tokens=budget.evidence_output_tokens,
                )
                card = card_from(raw)
            except (KimiError, EvidenceError, TypeError, ValueError) as first_error:
                # JSON/结构问题和 reasoning-exhaustion 进入一次低推理档纠错；
                # 明确的 API/网络宕机不对每块重复轰炸，直接单块导航降级。
                first_reason = str(first_error)[:300]
                reason_folded = first_reason.casefold()
                retryable_content_failure = (
                    isinstance(first_error, (EvidenceError, TypeError, ValueError))
                    or any(token in reason_folded for token in (
                        "reasoning", "正文为空", "空内容", "输出预算", "invalid json",
                        "非法 json", "证据卡必须", "证据卡返回",
                    ))
                )
                if not retryable_content_failure:
                    reason = f"证据调用失败，未做逐块重复请求：{first_reason}"
                    logger.warning(
                        "原文块 %s 证据调用不可用，单块降级为原文导航卡：%s",
                        chunk.chunk_id, reason,
                    )
                    card = _source_fallback_card(chunk)
                else:
                    retry_prompt = (
                        prompt + "\n\n## 本次必须修正\n"
                        "上一条证据卡请求失败或输出不可用。请降低推理开销，只输出一个完整 JSON 对象，"
                        "不要解释、不要 markdown 围栏；保留 chunk_id/section 及本块能从原文确认的字段。"
                    )
                    try:
                        calls += 1
                        raw = chat_with_telemetry(
                            chunk,
                            [{"role": "user", "content": retry_prompt}],
                            attempt=2,
                            heavy=False,
                            reasoning_effort="low",
                            max_tokens=max(8_192, budget.evidence_output_tokens // 2),
                        )
                        card = card_from(raw)
                    except (KimiError, EvidenceError, TypeError, ValueError) as second_error:
                        reason = (
                            f"首次证据卡失败：{first_reason}；"
                            f"低推理重试失败：{str(second_error)[:300]}"
                        )
                        logger.warning(
                            "原文块 %s 证据卡单块降级为原文导航卡：%s",
                            chunk.chunk_id, reason,
                        )
                        card = _source_fallback_card(chunk)

            if not any((card.orientation, card.claims, card.mechanisms, card.evidence,
                        card.definitions, card.limitations)):
                reason = reason or "证据卡字段为空"
                logger.warning(
                    "原文块 %s 证据卡字段为空，单块降级为原文导航卡：%s",
                    chunk.chunk_id, reason,
                )
                card = _source_fallback_card(chunk)
            if card.orientation.startswith("原文导航摘录：") and not reason:
                reason = "本块使用确定性原文导航卡"
            if workspace:
                payload = asdict(card)
                payload["_cache_sig"] = {
                    "chunk_digest": chunk.digest,
                    "schema": EVIDENCE_SCHEMA_DIGEST,
                    "prompt": prompt_digest,
                }
                if reason:
                    payload["_degraded_reason"] = reason
                workspace.put_json(key, payload)
            return card, calls, reason

        # 证据块彼此独立：真实客户端受控并行（测试 Fake 保持串行，时序可断言）
        workers = (
            min(EVIDENCE_INDEX_WORKERS, len(chunks))
            if isinstance(self.client, (KimiClient, FrozenKimiClient)) else 1
        )
        cards_by_id: dict[str, EvidenceCard] = {}
        degraded_chunks: dict[str, str] = {}
        calls = 0
        done = 0
        started_at = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(build_one, chunk): chunk for chunk in chunks}
            for future in concurrent.futures.as_completed(futures):
                chunk = futures[future]
                try:
                    card, chunk_calls, reason = future.result()
                except Exception as exc:  # 单块意外异常也隔离为导航卡
                    reason = f"证据块任务异常：{type(exc).__name__}: {str(exc)[:300]}"
                    logger.exception("原文块 %s 证据索引异常，单块降级", chunk.chunk_id)
                    card, chunk_calls = _source_fallback_card(chunk), 0
                cards_by_id[chunk.chunk_id] = card
                if reason:
                    degraded_chunks[chunk.chunk_id] = reason
                calls += chunk_calls
                done += 1
                detail = f"，已降级：{reason[:100]}" if reason else ""
                progress(f"长论文证据索引 {done}/{len(chunks)}：{chunk.title[:28]}{detail}")
        cards = [cards_by_id[chunk.chunk_id] for chunk in chunks]
        elapsed = time.perf_counter() - started_at
        metrics = {
            "workers": workers,
            "chunks": len(chunks),
            "calls": calls,
            "elapsed_seconds": round(elapsed, 3),
            "max_inflight_calls": max_active_calls,
            "degraded_chunks": len(degraded_chunks),
        }
        logger.info(
            "长论文证据索引完成：chunks=%d calls=%d workers=%d elapsed=%.1fs max_inflight=%d degraded=%d",
            len(chunks), calls, workers, elapsed, max_active_calls, len(degraded_chunks),
        )
        return EvidenceBundle(
            chunks, cards, True,
            degraded_chunks=degraded_chunks,
            index_metrics=metrics,
        ), calls
