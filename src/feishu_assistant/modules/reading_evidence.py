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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..kimi import KimiClient, KimiError
from .reading_graph import content_digest
from .reading_workflow import ReadingWorkspace

logger = logging.getLogger("feishu_assistant.reading.evidence")

LONG_PAPER_CHARS = 120_000
SOURCE_CHUNK_CHARS = 24_000
EVIDENCE_INDEX_WORKERS = 3  # 证据块彼此独立，受控并行建卡；不影响缓存（块间无共享前缀）
EVIDENCE_SCHEMA_DIGEST = "evidence-card-v2"  # {statement,quote,chunk_id} 条目结构
SOURCE_PACKET_CHARS = 70_000
# 规划器索引上限。80k 是按「小卡×块数」拍脑袋的时代值；实测 45 图 111 页综述归并后
# 仍需 ~85k。256k 上下文模型读 160k（~45k token）索引毫无压力，超出才需要分层归并。
INDEX_MAX_CHARS = 160_000

_FRAGMENT_SUFFIX_RE = re.compile(r"（片段\s*\d+）\s*$")
_PACK_SUFFIX_RE = re.compile(r"\s+等\s*\d+\s*节\s*$")


def _chunk_base_title(title: str) -> str:
    """块标题的归属基名：剥掉「等 N 节」与「（片段 N）」，用于亲属片段识别。"""
    base = _PACK_SUFFIX_RE.sub("", str(title))
    base = _FRAGMENT_SUFFIX_RE.sub("", base)
    return base.strip()


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


@dataclass
class EvidenceBundle:
    chunks: list[SourceChunk]
    cards: list[EvidenceCard]
    long_mode: bool

    @property
    def chunk_map(self) -> dict[str, SourceChunk]:
        return {chunk.chunk_id: chunk for chunk in self.chunks}

    def planner_material(self, paper_md: str) -> str:
        if not self.long_mode:
            return paper_md
        parts = [
            "以下是逐章节建立的证据索引。每项保留 source chunk id；这是规划导航，"
            "不是允许脱离原文写作的摘要。",
        ]
        for card in self.cards:
            parts.append(json.dumps(asdict(card), ensure_ascii=False))
        text = "\n".join(parts)
        if len(text) > INDEX_MAX_CHARS:
            raise EvidenceError(
                f"证据索引仍过长（{len(text)} 字符）；需要增加分层归并，不能截掉尾部"
            )
        return text

    def source_packet(
        self,
        refs: list[str] | None,
        hints: list[str] | None = None,
        max_chars: int = SOURCE_PACKET_CHARS,
    ) -> str:
        """优先按计划的精确引用取原文；无引用时用标题/正文关键词保守召回。"""
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
            # 把同基名的亲属片段一并打包（按原文顺序），避免写手拿到半截论述。
            bases = {_chunk_base_title(chunk.title) for chunk in selected}
            expanded = [chunk for chunk in self.chunks
                        if _chunk_base_title(chunk.title) in bases]
            if len(expanded) > len(selected):
                logger.info("引用片段自动并入亲属块：%d → %d", len(selected), len(expanded))
            selected = expanded
        if not selected:
            terms = {
                term.casefold()
                for hint in (hints or [])
                for term in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,8}", hint)
            }
            ranked: list[tuple[int, int, SourceChunk]] = []
            for chunk in self.chunks:
                haystack = f"{chunk.title}\n{chunk.text}".casefold()
                score = sum(3 if term in chunk.title.casefold() else 1 for term in terms if term in haystack)
                ranked.append((score, -chunk.order, chunk))
            selected = [item[2] for item in sorted(ranked, reverse=True) if item[0] > 0]
        if not selected:
            selected = self.chunks[:1]
        if strict_refs:
            required = sum(
                len(chunk.text) + len(chunk.title) + len(chunk.chunk_id) + 24
                for chunk in selected
            )
            if required > max_chars:
                raise EvidenceError(
                    f"一次写作引用的原文块过多（约 {required} 字符，单包上限 {max_chars}）；"
                    "应把章节或教学层拆细，不能静默截掉后面的引用"
                )
        parts: list[str] = []
        used = 0
        for chunk in sorted(selected, key=lambda item: item.order):
            header = f"\n\n--- SOURCE {chunk.chunk_id} · {chunk.title} ---\n"
            remaining = max_chars - used - len(header)
            if remaining <= 0:
                break
            body = chunk.text[:remaining]
            parts.append(header + body)
            used += len(header) + len(body)
        return "".join(parts).strip()

    def review_material(self) -> str:
        """全局审核使用可追溯证据索引；逐章事实审核使用 source_packet 原文。"""
        if not self.long_mode:
            return ""
        return self.planner_material("")


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
        long_mode = len(paper_md) > LONG_PAPER_CHARS
        if not long_mode:
            return EvidenceBundle(chunks, [], False), 0

        prompt_digest = content_digest(self.prompt)

        def build_one(chunk: SourceChunk) -> tuple[EvidenceCard, int]:
            key = f"evidence-{chunk.chunk_id}"
            cached = workspace.get_json(key) if workspace else None
            if isinstance(cached, dict):
                # 卡签名：chunk 内容 + schema + prompt 摘要任一不一致即重建本块
                sig = cached.get("_cache_sig") or {}
                if (sig.get("chunk_digest") == chunk.digest
                        and sig.get("schema") == EVIDENCE_SCHEMA_DIGEST
                        and sig.get("prompt") == prompt_digest):
                    return EvidenceCard.from_dict(cached, chunk), 0
            prompt = self.prompt.replace("{{CHUNK_ID}}", chunk.chunk_id)
            prompt = prompt.replace("{{SECTION_TITLE}}", chunk.title)
            prompt = prompt.replace("{{SOURCE_CONTENT}}", chunk.text)
            calls = 0
            try:
                raw = self.client.chat(
                    [{"role": "user", "content": prompt}],
                    heavy=True, json_mode=True, low_reasoning=True, max_tokens=16384,  # 原文块是长输入；流式避免把总生成时间当 read timeout
                )
            except KimiError as exc:
                raise EvidenceError(f"原文块 {chunk.chunk_id} 证据提取失败：{exc}") from exc
            calls += 1

            def _card_from(text: str) -> EvidenceCard:
                return EvidenceCard.from_dict(_json_object(text), chunk)

            try:
                card = _card_from(raw)
            except EvidenceError:
                # 一次非法输出不判死刑：带修正指令重试一次（与简历画像同一模式）
                retry_prompt = (
                    prompt + "\n\n## 本次必须修正\n"
                    "上一次输出无法解析为 JSON。只输出一个 JSON 对象，不要任何解释、"
                    "前后缀或 markdown 围栏；字段与要求保持一致。"
                )
                try:
                    raw = self.client.chat(
                        [{"role": "user", "content": retry_prompt}],
                        heavy=True, json_mode=True, low_reasoning=True, max_tokens=16384,
                    )
                except KimiError as exc:
                    raise EvidenceError(f"原文块 {chunk.chunk_id} 证据提取失败：{exc}") from exc
                calls += 1
                try:
                    card = _card_from(raw)
                except EvidenceError:
                    logger.warning(
                        "原文块 %s 连续返回非 JSON；API 请求已成功，改用原文导航卡继续",
                        chunk.chunk_id,
                    )
                    card = _source_fallback_card(chunk)
            if not any((card.orientation, card.claims, card.mechanisms, card.evidence,
                        card.definitions, card.limitations)):
                raise EvidenceError(f"原文块 {chunk.chunk_id} 没有形成可用证据卡")
            if workspace:
                payload = asdict(card)
                payload["_cache_sig"] = {
                    "chunk_digest": chunk.digest,
                    "schema": EVIDENCE_SCHEMA_DIGEST,
                    "prompt": prompt_digest,
                }
                workspace.put_json(key, payload)
            return card, calls

        # 证据块彼此独立：真实客户端受控并行（测试 Fake 保持串行，时序可断言）
        workers = (
            min(EVIDENCE_INDEX_WORKERS, len(chunks))
            if isinstance(self.client, KimiClient) else 1
        )
        cards_by_id: dict[str, EvidenceCard] = {}
        calls = 0
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(build_one, chunk): chunk for chunk in chunks}
            for future in concurrent.futures.as_completed(futures):
                chunk = futures[future]
                card, chunk_calls = future.result()
                cards_by_id[chunk.chunk_id] = card
                calls += chunk_calls
                done += 1
                progress(f"长论文证据索引 {done}/{len(chunks)}：{chunk.title[:28]}")
        cards = [cards_by_id[chunk.chunk_id] for chunk in chunks]
        return EvidenceBundle(chunks, cards, True), calls
