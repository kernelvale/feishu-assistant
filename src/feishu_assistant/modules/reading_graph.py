"""语义图基础设施（v1）：稳定 ID、source_span、锚点解析、产物签名、依赖失效。

设计契约（与评审通过的 v1 边界一致）：
- 节点仅三类：source_span（程序解析的原文片段）、concept（前置概念）、
  assertion（核心主张/待讲清命题）；mechanism/evidence/limitation 暂作标签扩展。
- 边仅两类：supported_by（→source_span）、depends_on（→concept）。
- section/layer 属于后续 Plan，不属于本图；Plan 通过 concept_ids/assertion_ids 绑定。
- 稳定 ID = 规范化内容哈希；顺序号只用于展示，绝不参与身份。
- 锚点定位程序负责：模型只交 chunk + 近似摘录 + 语义陈述；程序三级匹配
  （精确 → 规范化 → 局部对齐），歧义或低于阈值标 unresolved_anchor，绝不静默取最像。
- 产物签名五要素 + input_revisions；semantic_contract_version 由确定性输入生成。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "semantic-graph-v1"
PRODUCER_VERSION = "reading-graph/1.0"

# 锚点局部对齐阈值：先用 2603/2509 实测分布校准后再冻结
ANCHOR_MIN_SIMILARITY = 0.86
ANCHOR_MIN_GAP = 0.04


def _normalize_text(text: str) -> str:
    """身份哈希用规范化：去空白/全半角标点差异，不改变语义。"""
    value = re.sub(r"\s+", "", str(text or ""))
    value = re.sub(r"[，。；：？！、（）「」【】《》]", "", value)
    return value.casefold()


def content_digest(text: str) -> str:
    return hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()[:16]


def stable_id(kind: str, content: str) -> str:
    """稳定 ID：规范化内容哈希。改名/移动/换序不改变 ID。"""
    return f"{kind}-{content_digest(content)}"


# ---------------------------------------------------------------------------
# source_span：程序解析的稳定原文片段
# ---------------------------------------------------------------------------

@dataclass
class SourceSpan:
    span_id: str
    text: str          # 逐字原文
    start: int         # 在 paper_md 中的偏移
    end: int
    digest: str        # 规范化内容哈希

    @classmethod
    def from_text(cls, text: str, start: int, end: int) -> "SourceSpan":
        return cls(
            span_id=stable_id("span", text),
            text=text, start=start, end=end,
            digest=content_digest(text),
        )


def split_source_spans(paper_md: str, *, max_chars: int = 4_000) -> list[SourceSpan]:
    """把原文切成稳定片段：按二级/三级标题分组，超长段落按段再切。
    只依赖内容，不依赖顺序——原文换序后未变化片段 ID 不变。"""
    spans: list[SourceSpan] = []
    cursor = 0
    for match in re.finditer(r"^#{1,3}\s+[^\n]+$", paper_md, re.MULTILINE):
        # 标题行本身并入下一段（标题是片段的身份的一部分）
        cursor = cursor  # 保持可读性
        _emit_span(spans, paper_md, cursor, match.start(), max_chars)
        cursor = match.start()
    _emit_span(spans, paper_md, cursor, len(paper_md), max_chars)
    return [s for s in spans if s.text.strip()]


def _emit_span(spans: list[SourceSpan], text: str, start: int, end: int,
               max_chars: int) -> None:
    if end <= start:
        return
    segment = text[start:end]
    if len(segment) <= max_chars:
        spans.append(SourceSpan.from_text(segment, start, end))
        return
    # 超长段按空行切；仍超长则硬切（边界处不影响身份，因为 ID 来自内容）
    pos = start
    while pos < end:
        nxt = text.find("\n\n", pos, min(pos + max_chars, end))
        cut = nxt if 0 < nxt <= pos + max_chars else min(pos + max_chars, end)
        spans.append(SourceSpan.from_text(text[pos:cut], pos, cut))
        pos = cut + 2 if nxt > 0 else cut


# ---------------------------------------------------------------------------
# 锚点解析：模型给近似摘录，程序三级定位
# ---------------------------------------------------------------------------

@dataclass
class AnchorResult:
    resolved: bool
    quote: str = ""        # 原文逐字摘录
    start: int = -1
    end: int = -1
    digest: str = ""
    reason: str = ""       # unresolved 时说明


def resolve_anchor(source: str, approx_quote: str, *,
                   min_similarity: float = ANCHOR_MIN_SIMILARITY,
                   min_gap: float = ANCHOR_MIN_GAP) -> AnchorResult:
    """三级定位：精确 → 规范化 → 局部对齐（带相似度与候选差距阈值）。
    歧义或低于阈值：unresolved_anchor，绝不静默取最像。"""
    quote = str(approx_quote or "").strip()
    if not quote:
        return AnchorResult(False, reason="空摘录")
    # 一级：精确匹配
    idx = source.find(quote)
    if idx >= 0 and source.find(quote, idx + 1) < 0:
        return AnchorResult(True, quote=quote, start=idx, end=idx + len(quote),
                            digest=content_digest(quote))
    if idx >= 0:
        return AnchorResult(False, reason="摘录在原文出现多次，候选歧义")
    # 二级：规范化匹配（去空白/标点差异）
    norm_quote = _normalize_text(quote)
    norm_hits: list[tuple[int, int, str]] = []
    if norm_quote:
        for m in re.finditer(r"\S(?:[^\n]*\S)?", source):
            raw = m.group(0)
            if _normalize_text(raw) == norm_quote:
                norm_hits.append((m.start(), m.end(), raw))
    if len(norm_hits) == 1:
        start, end, raw = norm_hits[0]
        return AnchorResult(True, quote=raw, start=start, end=end,
                            digest=content_digest(raw))
    if len(norm_hits) > 1:
        return AnchorResult(False, reason="规范化后仍有多个候选，歧义")
    # 三级：局部对齐（滑窗 + 相似度 + 候选差距）
    window = max(len(quote), 16)
    candidates: list[tuple[float, int, int, str]] = []
    step = max(1, window // 4)
    for pos in range(0, max(1, len(source) - window + 1), step):
        frag = source[pos:pos + window]
        sim = _similarity(_normalize_text(quote), _normalize_text(frag))
        if sim >= min_similarity:
            candidates.append((sim, pos, pos + window, frag))
    if not candidates:
        return AnchorResult(False, reason="局部对齐低于相似度阈值")
    candidates.sort(reverse=True)
    best = candidates[0]
    second = candidates[1][0] if len(candidates) > 1 else 0.0
    if best[0] - second < min_gap:
        return AnchorResult(False, reason="候选差距不足，歧义")
    sim, start, end, frag = best
    return AnchorResult(True, quote=frag, start=start, end=end,
                        digest=content_digest(frag))


def _similarity(a: str, b: str) -> float:
    """规范化文本的字符级 Jaccard 近似（避免引外部依赖）。"""
    if not a or not b:
        return 0.0
    set_a = {a[i:i + 2] for i in range(len(a) - 1)} or {a}
    set_b = {b[i:i + 2] for i in range(len(b) - 1)} or {b}
    inter = len(set_a & set_b)
    return inter / max(1, len(set_a | set_b))


# ---------------------------------------------------------------------------
# SemanticGraph：concept/assertion 节点 + supported_by/depends_on 边
# ---------------------------------------------------------------------------

NODE_KINDS = ("source_span", "concept", "assertion")
EDGE_RULES = {
    "supported_by": {"concept", "assertion"},   # → source_span
    "depends_on": {"concept", "assertion"},     # → concept 或 assertion（跨类型允许）
}


@dataclass
class GraphEdge:
    """可审核的依赖边：origin 区分局部提取/跨 chunk 连接，rationale 供 Plan Reviewer 判真。"""
    kind: str                    # supported_by | depends_on
    from_id: str
    to_id: str
    origin: str = "local_extract"        # local_extract | cross_chunk_link
    rationale: str = ""
    evidence_node_ids: list[str] = field(default_factory=list)
    edge_id: str = ""

    def __post_init__(self) -> None:
        if not self.edge_id:
            self.edge_id = stable_id(
                "edge", f"{self.kind}|{self.from_id}|{self.to_id}|{self.rationale}")
        if not self.evidence_node_ids:
            self.evidence_node_ids = [self.from_id, self.to_id]


@dataclass
class GraphNode:
    node_id: str
    kind: str                    # concept | assertion（source_span 单独成表）
    statement: str
    anchor: dict[str, Any] = field(default_factory=dict)  # quote/start/end/digest
    labels: list[str] = field(default_factory=list)       # mechanism/evidence/limitation 扩展位
    unresolved_anchor: bool = False


class GraphError(Exception):
    pass


class SemanticGraph:
    """最小语义图：source_span 表 + concept/assertion 节点 + 类型化边。"""

    def __init__(self) -> None:
        self.spans: dict[str, SourceSpan] = {}
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []
        self.issues: list[dict[str, Any]] = []  # graph_issues：未决锚点等结构化记录

    # ---- 构建 ----

    def add_source_text(self, paper_md: str) -> list[str]:
        for span in split_source_spans(paper_md):
            self.spans[span.span_id] = span
        return list(self.spans)

    def add_node(self, kind: str, statement: str, *,
                 approx_quote: str = "", source: str = "",
                 labels: Iterable[str] = ()) -> GraphNode:
        if kind not in ("concept", "assertion"):
            raise GraphError(f"非法节点类型：{kind}")
        if not str(statement).strip():
            raise GraphError("statement 不能为空")
        node_id = stable_id(kind, statement)
        if node_id in self.nodes:           # 幂等：同陈述同节点
            return self.nodes[node_id]
        anchor: dict[str, Any] = {}
        unresolved = False
        if approx_quote and source:
            result = resolve_anchor(source, approx_quote)
            if result.resolved:
                anchor = {"quote": result.quote, "start": result.start,
                          "end": result.end, "digest": result.digest}
            else:
                unresolved = True
                logger.warning("锚点未决（%s）：%s", node_id, result.reason)
        node = GraphNode(node_id, kind, statement.strip(), anchor,
                         list(labels), unresolved)
        self.nodes[node_id] = node
        return node

    def add_edge(self, kind: str, src_id: str, dst_id: str, *,
                 origin: str = "local_extract", rationale: str = "") -> GraphEdge:
        if kind not in EDGE_RULES:
            raise GraphError(f"非法边类型：{kind}")
        src = self.nodes.get(src_id)
        if src is None:
            raise GraphError(f"边起点不存在：{src_id}")
        if src.kind not in EDGE_RULES[kind]:
            raise GraphError(f"{kind} 边不允许起点类型 {src.kind}")
        if src_id == dst_id:
            raise GraphError("不允许自依赖")
        if kind == "supported_by":
            if dst_id not in self.spans:
                raise GraphError(f"supported_by 目标必须是 source_span：{dst_id}")
        else:
            dst = self.nodes.get(dst_id)
            if dst is None:
                raise GraphError(f"depends_on 目标不存在：{dst_id}")
            if dst.kind not in ("concept", "assertion"):
                raise GraphError(f"depends_on 目标类型非法：{dst.kind}")
        edge = GraphEdge(kind, src_id, dst_id, origin=origin, rationale=rationale)
        if all(e.edge_id != edge.edge_id for e in self.edges):
            self.edges.append(edge)
        return edge

    def has_cycle(self) -> bool:
        """统一判环：全部 concept/assertion 与所有 depends_on 边一张图，拓扑排序。"""
        outgoing: dict[str, list[str]] = {}
        nodes = set(self.nodes)
        for e in self.edges:
            if e.kind != "depends_on":
                continue
            outgoing.setdefault(e.from_id, []).append(e.to_id)
            nodes.add(e.from_id)
            nodes.add(e.to_id)
        indegree: dict[str, int] = {n: 0 for n in nodes}
        for src, dsts in outgoing.items():
            for dst in dsts:
                indegree[dst] = indegree.get(dst, 0) + 1
        queue = [n for n in nodes if indegree.get(n, 0) == 0]
        seen = 0
        while queue:
            node = queue.pop()
            seen += 1
            for dst in outgoing.get(node, []):
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    queue.append(dst)
        return seen < len(nodes)

    # ---- 校验 ----

    def invariants(self) -> list[str]:
        """全局不变量：未决锚点、悬空边。"""
        problems = [
            f"节点 {n.node_id} 锚点未决" for n in self.nodes.values() if n.unresolved_anchor
        ]
        for e in self.edges:
            if e.from_id not in self.nodes:
                problems.append(f"边 {e.kind} 起点悬空：{e.from_id}")
            if e.kind == "supported_by" and e.to_id not in self.spans:
                problems.append(f"边 supported_by 终点悬空：{e.to_id}")
            if e.kind == "depends_on" and e.to_id not in self.nodes:
                problems.append(f"边 depends_on 终点悬空：{e.to_id}")
        if self.has_cycle():
            problems.append("depends_on 存在环（统一 DAG 拓扑排序未通过）")
        return problems

    # ---- 持久化 ----

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "spans": [asdict(s) for s in self.spans.values()],
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges],
            "issues": self.issues,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticGraph":
        version = str(data.get("schema_version") or "")
        if version != SCHEMA_VERSION:
            raise GraphError(f"语义图 schema 版本不兼容：{version or '缺失'}")
        graph = cls()
        for raw in data.get("spans") or []:
            span = SourceSpan(**raw)
            graph.spans[span.span_id] = span
        for raw in data.get("nodes") or []:
            node = GraphNode(**raw)
            graph.nodes[node.node_id] = node
        graph.edges = [
            e if isinstance(e, GraphEdge) else GraphEdge(**e) if isinstance(e, dict)
            else GraphEdge(e[0], e[1], e[2])
            for e in data.get("edges") or []
        ]
        graph.issues = list(data.get("issues") or [])
        return graph

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SemanticGraph":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# 目录渲染：prompt 注入用 compact 图目录（DebtCompiler/PlanCompiler 用）
# ---------------------------------------------------------------------------

def render_graph_catalog(graph: "SemanticGraph", max_chars: int = 6000) -> str:
    """渲染 compact 图目录文本，供 DebtCompiler/PlanCompiler 注入 prompt。

    - 每个 concept/assertion 一行：
      node_id | kind | labels | statement(截断) | supported_by→span_id | depends_on→[ids]
    - source_span 单独 compact 列出：span_id | 起始偏移 | 首行截断
    - 确定性输出：组内按 ID 排序，与图的构建/遍历顺序无关
    - 超 max_chars 时按 assertion → concept → span 优先级保留，低位组先省略
      并在末尾标注各组省略条数
    """
    def _node_line(node: GraphNode) -> str:
        labels = ",".join(node.labels) if node.labels else "-"
        statement = re.sub(r"\s+", " ", node.statement).strip()
        if len(statement) > 100:
            statement = statement[:100] + "…"
        supports = sorted(e.to_id for e in graph.edges
                          if e.kind == "supported_by" and e.from_id == node.node_id)
        depends = sorted(e.to_id for e in graph.edges
                         if e.kind == "depends_on" and e.from_id == node.node_id)
        return (f"{node.node_id} | {node.kind} | {labels} | {statement} | "
                f"supported_by→{','.join(supports) or '-'} | "
                f"depends_on→[{','.join(depends)}]")

    def _span_line(span: SourceSpan) -> str:
        first = re.sub(r"\s+", " ", span.text).strip()
        if len(first) > 60:
            first = first[:60] + "…"
        return f"{span.span_id} | {span.start} | {first}"

    # 优先级即保留顺序：assertion 最高，span 最低
    prioritized: list[tuple[str, list[str]]] = [
        ("assertion", [_node_line(n) for n in sorted(
            (n for n in graph.nodes.values() if n.kind == "assertion"),
            key=lambda n: n.node_id)]),
        ("concept", [_node_line(n) for n in sorted(
            (n for n in graph.nodes.values() if n.kind == "concept"),
            key=lambda n: n.node_id)]),
        ("source_span", [_span_line(s) for s in sorted(
            graph.spans.values(), key=lambda s: s.span_id)]),
    ]
    out: list[str] = []
    used = 0
    cut = False            # 预算耗尽后，低位组整体省略
    omitted: list[str] = []
    for title, group_lines in prioritized:
        header = f"## {title}（共 {len(group_lines)} 条）"
        if cut or used + len(header) + 1 > max_chars:
            omitted.append(f"{title} {len(group_lines)} 条")
            cut = True
            continue
        out.append(header)
        used += len(header) + 1
        shown = 0
        for line in group_lines:
            if used + len(line) + 1 > max_chars:
                cut = True
                break
            out.append(line)
            used += len(line) + 1
            shown += 1
        if shown < len(group_lines):
            omitted.append(f"{title} {len(group_lines) - shown} 条")
    if omitted:
        out.append("…（目录超出预算，已按优先级省略：" + "；".join(omitted) + "）")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 产物签名与依赖失效
# ---------------------------------------------------------------------------

@dataclass
class ArtifactSignature:
    artifact_id: str
    revision: int
    prompt_digest: str
    schema_digest: str
    producer_version: str
    input_revisions: dict[str, int] = field(default_factory=dict)
    semantic_contract_version: str = ""

    def __post_init__(self) -> None:
        # 语义契约版本由确定性输入生成，不允许手填
        material = json.dumps({
            "prompt": self.prompt_digest, "schema": self.schema_digest,
            "producer": self.producer_version,
        }, sort_keys=True)
        self.semantic_contract_version = hashlib.sha256(
            material.encode()).hexdigest()[:12]

    def matches(self, other: "ArtifactSignature") -> bool:
        return (
            self.artifact_id == other.artifact_id
            and self.revision == other.revision
            and self.semantic_contract_version == other.semantic_contract_version
            and self.input_revisions == other.input_revisions
        )


class DependencyManifest:
    """产物依赖清单：artifact → {input_id: revision}；脏闭包 = 反向依赖传播。"""

    def __init__(self) -> None:
        self._inputs: dict[str, dict[str, int]] = {}

    def register(self, artifact_id: str, inputs: dict[str, int]) -> None:
        self._inputs[artifact_id] = dict(inputs)

    def remove(self, artifact_id: str) -> None:
        self._inputs.pop(artifact_id, None)

    def dirty_closure(self, changed: dict[str, int]) -> set[str]:
        """changed: {input_id: 新 revision}。产物的输入 revision 不一致即脏；
        脏产物本身作为下游输入时视为必然失效（其新 revision 未定，保守传播）。"""
        dirty: set[str] = set()
        queue = list(changed)
        while queue:
            current = queue.pop(0)
            for artifact, inputs in self._inputs.items():
                if artifact in dirty or current not in inputs:
                    continue
                if current in changed:
                    stale = inputs[current] != changed[current]
                else:
                    stale = True  # current 是脏产物：下游必脏
                if stale:
                    dirty.add(artifact)
                    queue.append(artifact)
        return dirty


# ---------------------------------------------------------------------------
# 构建器：EvidenceIndexer 之后构建/加载 SemanticGraph（v1 阶段一）
# ---------------------------------------------------------------------------

def validate_graph_for_plan(graph: "SemanticGraph") -> list[str]:
    """进入 Plan 前的图验收（fail-closed）：锚点已决、supported_by 齐备、
    无悬空/自依赖/环、非空图。核心绑定判断在 Plan 阶段（层深度属于 Plan 语义）。"""
    gaps: list[str] = []
    if not graph.nodes:
        gaps.append("语义图为空")
    for node in graph.nodes.values():
        if node.unresolved_anchor or not node.anchor.get("quote"):
            gaps.append(f"节点 {node.node_id} 锚点未决")
        if not any(e.kind == "supported_by" and e.from_id == node.node_id
                   for e in graph.edges):
            gaps.append(f"节点 {node.node_id} 缺少 supported_by")
    gaps.extend(graph.invariants())
    return gaps


def _span_for_offset(spans: dict[str, SourceSpan], start: int, end: int) -> str | None:
    """找包含锚点区间的 span（允许跨片时取首个覆盖者）。"""
    best: tuple[int, str] | None = None
    for span in spans.values():
        if span.start <= start and span.end >= end:
            overlap = min(span.end, end) - max(span.start, start)
            if best is None or overlap > best[0]:
                best = (overlap, span.span_id)
    return best[1] if best else None


def build_semantic_graph(
    paper_md: str,
    evidence: Any,
    client: Any,
    prompts_dir: Path,
    workspace: Any | None = None,
) -> tuple["SemanticGraph", dict[str, int]]:
    """原文证据 → SemanticGraph（两阶段：局部提取 + 跨 chunk link pass）。
    锚点规则：模型只交近似摘录，程序三级定位（chunk 内解析→绝对偏移）；
    未决节点写入 graph_issues 结构化记录，不进图也不静默。
    返回 (graph, stats)：extract_calls/link_calls/anchor_dropped 等统计。"""
    if workspace is not None:
        cached = workspace.get_json("semantic-graph")
        if isinstance(cached, dict) and cached.get("schema_version") == SCHEMA_VERSION:
            try:
                graph = SemanticGraph.from_dict(cached)
                logger.info("语义图命中检查点：%d 节点 %d 片段",
                            len(graph.nodes), len(graph.spans))
                return graph, {"extract_calls": 0, "link_calls": 0,
                               "anchor_dropped": 0}
            except (GraphError, TypeError, KeyError):
                logger.warning("语义图检查点损坏，重建")

    graph = SemanticGraph()
    graph.add_source_text(paper_md)
    stats = {"extract_calls": 0, "link_calls": 0, "anchor_dropped": 0}

    # 节点注册：临时键 → 稳定 ID 映射在收集完后再建边
    key_to_id: dict[str, str] = {}
    pending_deps: list[tuple[str, str]] = []  # (from_key, to_key)

    def _register(kind: str, statement: str, quote: str,
                  labels: Iterable[str] = (), temp_key: str = "",
                  chunk: Any | None = None) -> None:
        """锚点在指定 chunk 内解析（长文）或全文（短文）；换算绝对偏移后绑 span。
        未决节点丢弃并写入 graph_issues（结构化，不只日志）。"""
        if not str(statement).strip():
            return
        search_text = chunk.text if chunk is not None else paper_md
        offset_base = chunk.start if chunk is not None else 0
        result = resolve_anchor(search_text, quote) if quote else None
        if result is None or not result.resolved:
            graph.issues.append({
                "kind": "unresolved_anchor", "statement": statement[:80],
                "chunk_id": getattr(chunk, "chunk_id", ""),
                "reason": getattr(result, "reason", "无摘录") if result else "无摘录",
            })
            return
        abs_start, abs_end = offset_base + result.start, offset_base + result.end
        node = graph.add_node(kind, statement, labels=labels)
        node.anchor = {"quote": result.quote, "start": abs_start, "end": abs_end,
                       "digest": result.digest}
        node.unresolved_anchor = False
        span_id = _span_for_offset(graph.spans, abs_start, abs_end)
        if span_id:
            graph.add_edge("supported_by", node.node_id, span_id)
        else:
            graph.issues.append({"kind": "span_unbound", "node_id": node.node_id})
            del graph.nodes[node.node_id]
            return
        if temp_key:
            key_to_id[temp_key] = node.node_id

    if evidence is not None and getattr(evidence, "long_mode", False):
        # 长文：复用证据卡；条目兼容新 {statement,quote,chunk_id} 与旧字符串
        chunk_by_id = {c.chunk_id: c for c in evidence.chunks}
        for card in evidence.cards:
            chunk = chunk_by_id.get(card.chunk_id)

            def _items(value: Any) -> list[dict[str, str]]:
                out: list[dict[str, str]] = []
                for entry in value or []:
                    if isinstance(entry, dict):
                        out.append({"statement": str(entry.get("statement") or ""),
                                    "quote": str(entry.get("quote") or "")})
                    elif str(entry).strip():
                        out.append({"statement": str(entry), "quote": str(entry)[:120]})
                return out

            for entry in _items(card.definitions):
                _register("concept", entry["statement"], entry["quote"], chunk=chunk)
            for entry in _items(card.claims):
                _register("assertion", entry["statement"], entry["quote"], chunk=chunk)
            for entry in _items(card.mechanisms):
                _register("assertion", entry["statement"], entry["quote"],
                          ["mechanism"], chunk=chunk)
            for entry in _items(card.evidence):
                _register("assertion", entry["statement"], entry["quote"],
                          ["evidence"], chunk=chunk)
            for entry in _items(card.limitations):
                _register("assertion", entry["statement"], entry["quote"],
                          ["limitation"], chunk=chunk)
    else:
        from .reading import _json_object, _load, _render  # 延迟导入防环
        prompt = _render(
            _load(prompts_dir / "reading_concept_extract.md"),
            SOURCE_CONTENT=paper_md,
        )
        raw = client.chat(
            [{"role": "user", "content": prompt}],
            heavy=True, json_mode=True, low_reasoning=True, max_tokens=16384,
        )
        stats["extract_calls"] += 1
        payload: dict[str, Any] | None = None
        for _attempt in range(2):
            try:
                payload = _json_object(raw)
                break
            except Exception:
                raw = client.chat(
                    [{"role": "user", "content": prompt + "\n\n## 本次必须修正\n"
                     "上一次输出无法解析为 JSON。只输出一个 JSON 对象。"}],
                    heavy=True, json_mode=True, low_reasoning=True, max_tokens=16384,
                )
        if payload is None:
            # 末次重试的结果同样要解析：直接丢弃会白烧一次调用还误判失败
            try:
                payload = _json_object(raw)
            except Exception as exc:
                raise GraphError("语义节点提取连续返回非法 JSON") from exc
        staged: list[tuple[str, dict[str, Any]]] = []
        for item in payload.get("concepts") or []:
            if isinstance(item, dict):
                staged.append(("concept", item))
        for item in payload.get("assertions") or []:
            if isinstance(item, dict):
                staged.append(("assertion", item))
        for kind, item in staged:
            key = str(item.get("key") or "").strip()
            _register(kind, str(item.get("statement") or ""),
                      str(item.get("quote") or ""),
                      [str(v) for v in (item.get("labels") or [])],
                      temp_key=key)
            for dep in item.get("depends_on") or []:
                if key and str(dep).strip():
                    pending_deps.append((key, str(dep).strip()))
        # 临时键 → 正式 depends_on 边
        for from_key, to_key in pending_deps:
            src, dst = key_to_id.get(from_key), key_to_id.get(to_key)
            if not src or not dst:
                graph.issues.append({"kind": "dangling_temp_ref",
                                     "from": from_key, "to": to_key})
                continue
            try:
                graph.add_edge("depends_on", src, dst, origin="local_extract")
            except GraphError as exc:
                graph.issues.append({"kind": "edge_rejected",
                                     "from": src, "to": dst, "reason": str(exc)})

    # 跨 chunk link pass：多 chunk 时一次有界调用，只补边不建点
    if evidence is not None and getattr(evidence, "long_mode", False) \
            and len(evidence.chunks) > 1 and graph.nodes:
        from .reading import _json_object, _load, _render
        catalog = "\n".join(
            f"{n.node_id} | {n.kind} | {n.statement[:60]} | {n.anchor.get('quote', '')[:30]}"
            for n in graph.nodes.values()
        )
        link_prompt = _render(
            _load(prompts_dir / "reading_graph_link.md"),
            NODE_CATALOG=catalog,
        )
        raw = client.chat(
            [{"role": "user", "content": link_prompt}],
            heavy=True, json_mode=True, low_reasoning=True, max_tokens=8192,
        )
        stats["link_calls"] += 1
        try:
            link_data = _json_object(raw)
            for edge in link_data.get("edges") or []:
                if not isinstance(edge, dict):
                    continue
                src, dst = (str(edge.get("from_id") or "").strip(),
                            str(edge.get("to_id") or "").strip())
                if src not in graph.nodes or dst not in graph.nodes:
                    continue  # 引用清单外 ID：丢弃
                try:
                    graph.add_edge("depends_on", src, dst,
                                   origin="cross_chunk_link",
                                   rationale=str(edge.get("rationale") or "")[:200])
                except GraphError:
                    pass
        except Exception:
            logger.warning("跨 chunk link pass 返回非法 JSON，跳过补边")

    stats["anchor_dropped"] = len(graph.issues)
    if graph.issues:
        logger.warning("语义图构建：%d 条 graph_issues（未决锚点/悬空引用等）",
                       len(graph.issues))
    if workspace is not None:
        workspace.put_json("semantic-graph", graph.to_dict())
    return graph, stats
