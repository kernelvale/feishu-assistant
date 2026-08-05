"""正文块清单（V8 第 6 步）：block manifest、审核标记注入、变更检测与块级机械操作。

设计契约（与 V8 规划裁定一致）：
- 每单元一份 BlockManifest，随正文落盘 workspace key `blocks-{unit_id}`（json）。
  切块只依赖正文内容：标题行（markdown 各级标题）与空行为块边界——同文同块同 ID
  （block_id 按位置序 `{unit_id}#b{序号:02d}`）；digest 复用 reading_graph.content_digest。
- covered_debt_ids / source_span_ids 是程序启发式挂载：与注册表债务的
  learner_action/success_test/subject statement、span 原文的包含度相似度，
  阈值以上才挂，可空（现实张力裁定 3：只用于失效范围与对账，审核纠正优先）。
- render_for_review 在审核材料中注入 block 标记（行首 `<!-- block:xxx#bNN -->` 注释），
  **只给审核器看，绝不进发布稿**（发布稿始终取 workspace 落盘的未标注正文）。
- 变更检测：同单元新旧 manifest 按 block_id 对 digest——变化/新增/消失的块集合，
  供「只复核改动块+相邻句」。
- delete_blocks / move_blocks 是零模型机械操作：审核 issue 带 body_block_ids 时
  程序直接删/移对应块，不经 Writer。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .reading_graph import content_digest

BLOCK_MANIFEST_SCHEMA = "block-manifest-v1"

# 启发式挂载阈值（字符 bigram 包含度，见 _containment；两个方向取较大者，
# 覆盖「短债务文本 vs 长正文块」与「正文块引原文 span」两种不对称）：
COVER_MIN_CONTAINMENT = 0.6   # 块 ↔ 债务挂载
SPAN_MIN_CONTAINMENT = 0.6    # 块 ↔ source span 挂载

_HEADING_RE = re.compile(r"^#{1,6}\s+\S")


def _norm(text: str) -> str:
    return re.sub(r"[^\w一-鿿]+", "", str(text or "")).casefold()


def _bigrams(text: str) -> set[str]:
    value = _norm(text)
    return {value[i:i + 2] for i in range(len(value) - 1)} or ({value} if value else set())


def _containment(short_text: str, long_text: str) -> float:
    """短文本 bigram 集合被长文本覆盖的比例（0~1）；空文本为 0。"""
    short, long = _bigrams(short_text), _bigrams(long_text)
    if not short or not long:
        return 0.0
    return len(short & long) / len(short)


@dataclass
class Block:
    """一个正文块：标题段或空行分隔的段落。"""

    block_id: str                              # {unit_id}#b{序号:02d}
    unit_id: str
    text: str
    digest: str = ""                           # content_digest(text)
    covered_debt_ids: list[str] = field(default_factory=list)
    source_span_ids: list[str] = field(default_factory=list)


@dataclass
class BlockManifest:
    """单元 block 清单：审核定位、变更检测与精准失效的事实源。"""

    unit_id: str
    blocks: list[Block] = field(default_factory=list)

    def get(self, block_id: str) -> Block | None:
        for block in self.blocks:
            if block.block_id == block_id:
                return block
        return None

    def changed_block_ids(self, old: "BlockManifest") -> list[str]:
        """相对旧 manifest 的变更集合：digest 不同 + 新增 + 消失的块（保序去重）。"""
        old_map = {b.block_id: b.digest for b in old.blocks}
        new_ids = {b.block_id for b in self.blocks}
        changed = [b.block_id for b in self.blocks
                   if old_map.get(b.block_id) != b.digest]
        changed += [bid for bid in old_map if bid not in new_ids]
        return list(dict.fromkeys(changed))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BLOCK_MANIFEST_SCHEMA,
            "unit_id": self.unit_id,
            "blocks": [asdict(b) for b in self.blocks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BlockManifest":
        if str(data.get("schema")) != BLOCK_MANIFEST_SCHEMA:
            raise ValueError(f"block manifest schema 不兼容：{data.get('schema')}")
        return cls(
            unit_id=str(data.get("unit_id") or ""),
            blocks=[Block(**dict(raw)) for raw in data.get("blocks") or []],
        )


def split_blocks(
    unit_id: str,
    body: str,
    *,
    registry: Any | None = None,
    graph: Any | None = None,
) -> BlockManifest:
    """正文 → BlockManifest：标题行/空行为块边界，序号位置 ID。

    registry/graph 给定时做启发式挂载（covered_debt_ids / source_span_ids）；
    缺失时挂载为空（不判死）。"""
    raw_blocks: list[str] = []
    current: list[str] = []

    def _flush() -> None:
        text = "\n".join(current).strip()
        if text:
            raw_blocks.append(text)

    for line in str(body or "").split("\n"):
        if _HEADING_RE.match(line):
            _flush()
            current = [line]
        elif not line.strip():
            # 块内只有标题时空行不切块：标题与随后段落同块（避免孤标题块）
            if any(l.strip() and not _HEADING_RE.match(l) for l in current):
                _flush()
                current = []
        else:
            current.append(line)
    _flush()

    debts = getattr(registry, "debts", None) or {}
    blocks: list[Block] = []
    for idx, text in enumerate(raw_blocks, 1):
        # 挂载相似度用剔除标题行的块内容：标题属单元结构，不属于段落内容
        content_text = "\n".join(
            l for l in text.split("\n") if not _HEADING_RE.match(l))
        covered: list[str] = []
        for did, debt in debts.items():
            subject = ""
            if graph is not None and debt.subject_node_id in getattr(graph, "nodes", {}):
                subject = graph.nodes[debt.subject_node_id].statement
            probe = "\n".join([debt.learner_action, debt.success_test, subject])
            if max(_containment(probe, content_text),
                   _containment(content_text, probe)) >= COVER_MIN_CONTAINMENT:
                covered.append(did)
        span_ids: list[str] = []
        if graph is not None:
            for sid, span in getattr(graph, "spans", {}).items():
                if max(_containment(content_text, span.text),
                       _containment(span.text, content_text)) >= SPAN_MIN_CONTAINMENT:
                    span_ids.append(sid)
        blocks.append(Block(
            block_id=f"{unit_id}#b{idx:02d}",
            unit_id=unit_id,
            text=text,
            digest=content_digest(text),
            covered_debt_ids=covered,
            source_span_ids=span_ids,
        ))
    return BlockManifest(unit_id=unit_id, blocks=blocks)


def render_for_review(body: str, manifest: BlockManifest) -> str:
    """审核材料：每个块前注入行首定位注释。**只给审核器看，不进发布稿**。"""
    return "\n\n".join(
        f"<!-- block:{block.block_id} -->\n{block.text}"
        for block in manifest.blocks
    )


def delete_blocks(body: str, manifest: BlockManifest,
                  block_ids: Iterable[str]) -> str:
    """机械删除指定块（零模型）：按 manifest 过滤后重拼正文。"""
    drop = {str(bid).strip() for bid in block_ids}
    kept = [b.text for b in manifest.blocks if b.block_id not in drop]
    return "\n\n".join(kept)


def move_blocks(body: str, manifest: BlockManifest, block_ids: Iterable[str],
                *, after_block_id: str | None = None) -> str:
    """机械移动指定块（零模型）：移到 after_block_id 之后；缺省移到单元末尾。"""
    moving_ids = {str(bid).strip() for bid in block_ids}
    moving = [b for b in manifest.blocks if b.block_id in moving_ids]
    rest = [b for b in manifest.blocks if b.block_id not in moving_ids]
    if after_block_id:
        for idx, block in enumerate(rest):
            if block.block_id == after_block_id:
                rest[idx + 1:idx + 1] = moving
                break
        else:
            rest.extend(moving)  # 锚点不存在：落到末尾（防御，不判死）
    else:
        rest.extend(moving)
    return "\n\n".join(b.text for b in rest)


def deletion_neighbor_context(manifest: BlockManifest,
                              deleted_ids: Iterable[str],
                              *, max_chars: int = 2_000) -> str:
    """删除点相邻块上下文（前一块+后一块）：供补丁复核只审改动区域。"""
    deleted = {str(bid).strip() for bid in deleted_ids}
    keep: set[int] = set()
    for idx, block in enumerate(manifest.blocks):
        if block.block_id in deleted:
            keep.update({idx - 1, idx + 1})
    neighbors = [b.text for idx, b in enumerate(manifest.blocks)
                 if idx in keep and b.block_id not in deleted]
    return "\n\n".join(neighbors)[:max_chars]
