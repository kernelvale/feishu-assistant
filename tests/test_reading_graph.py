"""语义图基础设施测试矩阵（v1 验收项驱动）。"""

import json

import pytest

from feishu_assistant.modules.reading_graph import (
    ArtifactSignature,
    DependencyManifest,
    GraphError,
    SemanticGraph,
    content_digest,
    resolve_anchor,
    split_source_spans,
    stable_id,
)

PAPER = (
    "# 论文标题\n\n## 1 引言\n\n本文提出两道校验机制，先校验输入再校验输出。\n\n"
    "## 2 方法\n\n方法细节如 Figure 1 所示，包含三个环节：筛选、验证、合成。"
    "树搜索在程序空间进行，Thompson 采样用于挑选候选。\n\n"
    "## 3 实验\n\n145 个游戏上合法动作率达到 100%。\n"
)


# ---- 稳定 ID ----

def test_stable_id_survives_rename_and_move():
    a = stable_id("layer", "树搜索在程序空间进行")
    b = stable_id("layer", "树搜索在程序空间进行")
    assert a == b
    assert a != stable_id("layer", "另一段内容")
    # 空白/标点差异不改变身份
    assert stable_id("layer", "树搜索， 在程序空间进行") == a


def test_span_ids_stable_under_source_reorder():
    spans_a = {s.span_id for s in split_source_spans(PAPER)}
    reordered = (
        "# 论文标题\n\n## 3 实验\n\n145 个游戏上合法动作率达到 100%。\n\n"
        "## 1 引言\n\n本文提出两道校验机制，先校验输入再校验输出。\n\n"
        "## 2 方法\n\n方法细节如 Figure 1 所示，包含三个环节：筛选、验证、合成。"
        "树搜索在程序空间进行，Thompson 采样用于挑选候选。\n"
    )
    spans_b = {s.span_id for s in split_source_spans(reordered)}
    assert spans_a == spans_b  # 换序不变身份


# ---- 锚点解析 ----

def test_anchor_exact_match():
    r = resolve_anchor(PAPER, "先校验输入再校验输出")
    assert r.resolved and PAPER[r.start:r.end] == r.quote


def test_anchor_ambiguous_is_unresolved_not_silent():
    text = "他说了同一句话。她说也了同一句话。" + "同一句话" + "。"
    r = resolve_anchor(text, "同一句话")
    assert not r.resolved and "歧义" in r.reason


def test_anchor_low_similarity_is_unresolved():
    r = resolve_anchor(PAPER, "一段完全不存在于原文的表述需要被定位")
    assert not r.resolved


# ---- SemanticGraph ----

def _graph():
    g = SemanticGraph()
    g.add_source_text(PAPER)
    return g


def test_graph_nodes_edges_and_invariants():
    g = _graph()
    concept = g.add_node("concept", "合法动作必须由代码定义",
                         approx_quote="先校验输入再校验输出", source=PAPER)
    assert not concept.unresolved_anchor and concept.anchor["start"] >= 0
    claim = g.add_node("assertion", "harness 使合法动作率达到 100%",
                       approx_quote="145 个游戏上合法动作率达到 100%", source=PAPER)
    g.add_edge("depends_on", claim.node_id, concept.node_id)
    span_id = next(iter(g.spans))
    g.add_edge("supported_by", claim.node_id, span_id)
    assert g.invariants() == []


def test_graph_rejects_bad_edges_and_types():
    g = _graph()
    c = g.add_node("concept", "概念甲")
    with pytest.raises(GraphError, match="非法节点类型"):
        g.add_node("unit", "不允许的类型")
    with pytest.raises(GraphError, match="起点不存在"):
        g.add_edge("depends_on", "concept-幽灵", c.node_id)
    a = g.add_node("assertion", "某个论断")
    with pytest.raises(GraphError, match="自依赖"):
        g.add_edge("depends_on", a.node_id, a.node_id)
    b = g.add_node("assertion", "另一个论断")
    g.add_edge("depends_on", a.node_id, b.node_id)  # assertion→assertion 显式允许


def test_graph_anchor_unresolved_marked_not_silent():
    g = _graph()
    node = g.add_node("concept", "无法定位的概念", approx_quote="原文根本没有这句话", source=PAPER)
    assert node.unresolved_anchor
    assert any("锚点未决" in p for p in g.invariants())


def test_graph_persistence_roundtrip_and_version(tmp_path):
    g = _graph()
    g.add_node("concept", "概念甲")
    path = tmp_path / "semantic-graph.json"
    g.save(path)
    loaded = SemanticGraph.load(path)
    assert set(loaded.nodes) == set(g.nodes)
    assert set(loaded.spans) == set(g.spans)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema_version"] = "semantic-graph-v0"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(GraphError, match="schema 版本不兼容"):
        SemanticGraph.load(path)


# ---- 产物签名 ----

def _sig(artifact="layer-x", rev=1, inputs=None, prompt="p1", schema="s1"):
    return ArtifactSignature(
        artifact_id=artifact, revision=rev,
        prompt_digest=prompt, schema_digest=schema,
        producer_version="reading-graph/1.0",
        input_revisions=inputs or {},
    )


def test_signature_semantic_version_derived_not_handwritten():
    a = _sig()
    assert a.semantic_contract_version
    b = _sig(prompt="p2")
    assert a.semantic_contract_version != b.semantic_contract_version
    assert a.matches(_sig())
    assert not a.matches(_sig(rev=2))
    assert not a.matches(_sig(inputs={"source:x": 2}))


# ---- 依赖失效闭包 ----

def test_dirty_closure_reverse_propagates():
    m = DependencyManifest()
    m.register("section-a", {"span-1": 1})
    m.register("layer-a", {"concept-1": 1, "section-a": 1})
    m.register("final", {"layer-a": 1, "layer-b": 1})
    m.register("layer-b", {"span-9": 1})
    assert m.dirty_closure({"span-1": 2}) == {"section-a", "layer-a", "final"}
    assert m.dirty_closure({"span-9": 2}) == {"layer-b", "final"}
    assert m.dirty_closure({"span-1": 1}) == set()  # revision 未变不脏


def test_unknown_targets_do_not_pollute():
    m = DependencyManifest()
    m.register("layer-a", {"concept-1": 1})
    assert m.dirty_closure({"concept-幽灵": 9}) == set()


# ---- 阶段一接线：构建器 ----

_EXTRACT_PAPER = (
    "# 论文\n\n## 1 方法\n\n我们把合法性约束搬进可执行代码，"
    "由 propose_action 与 is_legal_action 两个函数承载。"
    "实验在 145 个游戏上合法率达到 100%。\n"
)


class _ExtractKimi:
    def chat(self, messages, **kwargs):
        return json.dumps({
            "concepts": [{"key": "c1", "statement": "合法性由代码承载",
                          "quote": "合法性约束搬进可执行代码，"}],
            "assertions": [{"key": "a1", "statement": "145 个游戏合法率 100%",
                            "quote": "实验在 145 个游戏上合法率达到 100%",
                            "labels": ["evidence"], "depends_on": ["c1"]},
                           {"key": "a2", "statement": "幽灵主张",
                            "quote": "原文根本没有这句话", "labels": []}],
        }, ensure_ascii=False)


def test_build_semantic_graph_short_paper(tmp_path):
    from feishu_assistant.modules.reading import ReadingWorkspace
    from feishu_assistant.modules.reading_graph import build_semantic_graph
    ws = ReadingWorkspace(tmp_path)
    g, _ = build_semantic_graph(_EXTRACT_PAPER, None, _ExtractKimi(),
                             __import__("pathlib").Path(__file__).resolve().parents[1] / "prompts", workspace=ws)
    kinds = {n.kind for n in g.nodes.values()}
    assert kinds == {"concept", "assertion"}
    # 幽灵主张锚点未决被丢弃
    assert all(not n.unresolved_anchor for n in g.nodes.values())
    assert all(n.anchor.get("quote") for n in g.nodes.values())
    # supported_by 指向真实 span 且包含锚点区间
    for edge in [e for e in g.edges if e.kind == "supported_by"]:
        span = g.spans[edge.to_id]
        anchor = g.nodes[edge.from_id].anchor
        assert span.start <= anchor["start"] and span.end >= anchor["end"]
    # 持久化 + 检查点复用（第二次构建零调用）
    g2, _ = build_semantic_graph(_EXTRACT_PAPER, None, _ExtractKimi(),
                              __import__("pathlib").Path(__file__).resolve().parents[1] / "prompts", workspace=ws)
    assert set(g2.nodes) == set(g.nodes)


# ---- 棒一补充：依赖图/判环/卡签名/图验收 ----

def test_depends_on_temp_keys_become_real_edges(tmp_path):
    from feishu_assistant.modules.reading import ReadingWorkspace
    from feishu_assistant.modules.reading_graph import build_semantic_graph
    ws = ReadingWorkspace(tmp_path)
    g, _ = build_semantic_graph(_EXTRACT_PAPER, None, _ExtractKimi(),
                             __import__("pathlib").Path(__file__).resolve().parents[1] / "prompts",
                             workspace=ws)
    deps = [e for e in g.edges if e.kind == "depends_on"]
    assert len(deps) == 1
    assert deps[0].from_id.startswith("assertion-") and deps[0].to_id.startswith("concept-")


def test_unified_cycle_detection_catches_cross_type_cycle():
    g = _graph()
    a = g.add_node("assertion", "主张甲")
    c = g.add_node("concept", "概念乙")
    g.add_edge("depends_on", a.node_id, c.node_id)
    g.add_edge("depends_on", c.node_id, a.node_id)  # assertion↔concept 混合环
    assert g.has_cycle() is True
    assert any("环" in p for p in g.invariants())


def test_validate_graph_for_plan_fail_closed():
    g = SemanticGraph()
    assert "语义图为空" in validate_graph_for_plan(g)
    g2 = _graph()
    node = g2.add_node("concept", "孤立概念")  # 无锚点无 supported_by
    gaps = validate_graph_for_plan(g2)
    assert any("锚点未决" in g_ for g_ in gaps) or any("supported_by" in g_ for g_ in gaps)


def test_evidence_card_cache_signature(tmp_path):
    """卡签名：chunk digest 不一致 → 重建；一致 → 命中零调用。"""
    from feishu_assistant.modules.reading import ReadingWorkspace
    source = "# 综述\n\n" + "研究内容。" * 25_000 + "\n\n## 结论\n\n尾部结论"
    ws = ReadingWorkspace(tmp_path)
    client = EvidenceKimi()
    bundle1, calls1 = EvidenceIndexer(client, PROJECT_ROOT / "prompts").build(
        source, workspace=ws)
    assert calls1 > 0
    client2 = EvidenceKimi()
    bundle2, calls2 = EvidenceIndexer(client2, PROJECT_ROOT / "prompts").build(
        source, workspace=ws)
    assert calls2 == 0  # 全部命中签名缓存
    assert len(bundle2.cards) == len(bundle1.cards)


from feishu_assistant.config import PROJECT_ROOT
from feishu_assistant.modules.reading_graph import validate_graph_for_plan
from feishu_assistant.modules.reading_evidence import EvidenceIndexer
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from test_reading_evidence import EvidenceKimi


def test_long_paper_cards_resolve_quotes_within_chunk(tmp_path):
    """长文：statement≠quote（归纳≠原文摘录）也能在所属 chunk 内定位并换算绝对偏移。"""
    from feishu_assistant.modules.reading_evidence import (
        EvidenceBundle, EvidenceCard, split_source,
    )
    from feishu_assistant.modules.reading_graph import build_semantic_graph

    paper = "## 引言\n\n" + "背景介绍。" * 30 + "\n\n## 方法\n\n" + \
        "我们提出约束解码方法，用校验器过滤非法动作。" + "方法细节展开。" * 20
    chunks = split_source(paper, chunk_chars=400)
    long_chunks = [c for c in chunks if "约束解码" in c.text]
    assert long_chunks and long_chunks[0].start >= 0
    card = EvidenceCard(
        chunk_id=long_chunks[0].chunk_id, section="方法",
        claims=[{"statement": "校验器能过滤非法动作（归纳）",
                 "quote": "用校验器过滤非法动作"}],
    )
    bundle = EvidenceBundle(chunks, [card], True)
    g, stats = build_semantic_graph(
        paper, bundle, client=None,
        prompts_dir=__import__("pathlib").Path(__file__).resolve().parents[1] / "prompts",
    )
    assert stats["extract_calls"] == 0
    assert g.nodes, "长文卡应产出节点"
    for node in g.nodes.values():
        assert not node.unresolved_anchor
        chunk = long_chunks[0]
        assert chunk.start <= node.anchor["start"] <= chunk.end  # 绝对偏移落在块内


def test_structured_card_roundtrip_and_quote_resolution():
    """结构化卡：往返保持对象结构；quote 在 chunk 内可定位；statement 不冒充 quote。"""
    from feishu_assistant.modules.reading_evidence import (
        EvidenceCard, split_source,
    )
    from feishu_assistant.modules.reading_graph import build_semantic_graph

    paper = "## 方法\n\n我们提出约束解码方法，用校验器过滤非法动作。其余展开。"
    chunk = split_source(paper)[0]
    card = EvidenceCard(
        chunk_id=chunk.chunk_id, section="方法",
        claims=[{"statement": "校验器负责过滤", "quote": "用校验器过滤非法动作",
                 "chunk_id": chunk.chunk_id}],
    )
    loaded = EvidenceCard.from_dict(json.loads(json.dumps(asdict_for(card))), chunk)
    assert isinstance(loaded.claims[0], dict)
    assert loaded.claims[0]["quote"] == "用校验器过滤非法动作"
    assert loaded.claims[0]["statement"] == "校验器负责过滤"

    from feishu_assistant.modules.reading_evidence import EvidenceBundle
    bundle = EvidenceBundle([chunk], [card], True)
    g, stats = build_semantic_graph(paper, bundle, client=None,
                                    prompts_dir=PROJECT_ROOT / "prompts")
    assert g.nodes and not g.issues
    node = next(iter(g.nodes.values()))
    assert node.anchor["quote"] == "用校验器过滤非法动作"
    assert chunk.start <= node.anchor["start"] < chunk.end


def asdict_for(card):
    from dataclasses import asdict
    return asdict(card)
