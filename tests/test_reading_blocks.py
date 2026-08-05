"""V8 第 6 步：block manifest 模块级测试。

覆盖：切块稳定（同文同块 ID）、digest 变更检测、启发式挂载、render_for_review
标记只进审核材料、delete/move 机械操作、manifest 落盘往返。
"""

from feishu_assistant.modules.reading_blocks import (
    BlockManifest,
    delete_blocks,
    deletion_neighbor_context,
    move_blocks,
    render_for_review,
    split_blocks,
)
from feishu_assistant.modules.reading_debts import DebtRegistry, DebtSpec
from feishu_assistant.modules.reading_graph import SemanticGraph

_BODY = """## 两道校验如何接力

第一道校验负责合法性检查，把不合法的输入挡在门外。

第二道校验负责边界确认，防止结论外推到不适用场景。

### 小节标题

两段合在一起不再切分。
"""


def test_split_blocks_stable_ids_and_boundaries():
    """切块稳定：同文同块同 ID；标题行与空行都是块边界。"""
    m1 = split_blocks("section-x", _BODY)
    m2 = split_blocks("section-x", _BODY)
    assert [b.block_id for b in m1.blocks] == [b.block_id for b in m2.blocks]
    assert [b.digest for b in m1.blocks] == [b.digest for b in m2.blocks]
    ids = [b.block_id for b in m1.blocks]
    assert ids == [f"section-x#b{i:02d}" for i in range(1, len(ids) + 1)]
    assert m1.blocks[0].text.startswith("## 两道校验")
    assert "### 小节标题" in m1.blocks[-1].text
    assert "两段合在一起" in m1.blocks[-1].text  # 标题与其段落同块
    # 空行多寡不影响切块（身份只依赖内容结构）
    airy = _BODY.replace("。\n\n第二道", "。\n\n\n\n第二道")
    assert [b.block_id for b in split_blocks("section-x", airy).blocks] == ids


def test_changed_block_ids_digest_detection():
    """变更检测：digest 变化/新增/消失的块都在集合里，未变的不在。"""
    old = split_blocks("section-x", _BODY)
    modified = _BODY.replace("第一道校验负责合法性检查", "第一道校验负责语法检查")
    new = split_blocks("section-x", modified)
    changed = new.changed_block_ids(old)
    assert changed == ["section-x#b01"]          # 只有首块 digest 变了
    assert "section-x#b02" not in changed
    # 消失块：删一段 → 位置序号重排，自删除点起都算变更（保守对账）
    shorter = "## 两道校验如何接力\n\n第一道校验负责合法性检查，把不合法的输入挡在门外。\n"
    new2 = split_blocks("section-x", shorter)
    changed2 = new2.changed_block_ids(old)
    assert "section-x#b02" in changed2           # 原 b02 消失
    assert "section-x#b01" not in changed2       # 首块逐字未变


def _debt_registry(graph: SemanticGraph) -> DebtRegistry:
    node = graph.add_node("assertion", "合法性校验如何把不合法输入挡在门外")
    registry = DebtRegistry()
    registry.add(DebtSpec.from_payload({
        "kind": "mechanism",
        "subject_node_id": node.node_id,
        "relation": "transforms",
        "object_node_ids": [],
        "premise_node_ids": [],
        "evidence_span_ids": [],
        "learner_action": "合法性校验负责检查输入并挡下不合法内容",
        "success_test": "闭卷说出合法性校验的作用",
        "necessity_reason": "不理解校验就无法跟上机制链",
    }))
    return registry


def test_covered_debt_ids_heuristic_attach_and_skip():
    """启发式挂载：高度重叠的块挂上债务，无关块不挂；无 registry 时挂载为空。"""
    graph = SemanticGraph()
    registry = _debt_registry(graph)
    did = registry.order[0]
    manifest = split_blocks("section-x", _BODY, registry=registry, graph=graph)
    attached = {b.block_id: b.covered_debt_ids for b in manifest.blocks}
    assert did in attached["section-x#b01"]      # 「第一道校验…合法性…不合法输入」块
    assert attached["section-x#b02"] == []       # 边界确认块与本债务无关
    bare = split_blocks("section-x", _BODY)      # 无 registry/graph：挂载全空
    assert all(b.covered_debt_ids == [] and b.source_span_ids == []
               for b in bare.blocks)


def test_render_for_review_marks_blocks():
    """审核材料：每块前有行首定位注释；标记文本不含正文改写。"""
    manifest = split_blocks("section-x", _BODY)
    marked = render_for_review(_BODY, manifest)
    for block in manifest.blocks:
        assert f"<!-- block:{block.block_id} -->" in marked
    assert marked.count("<!-- block:") == len(manifest.blocks)
    # 发布稿始终取未标注正文：render 只生成审核材料，不改 body 本身
    assert "<!-- block:" not in _BODY


def test_delete_and_move_blocks_mechanical():
    """delete/move 机械执行：目标块被移除/换位，其余块逐字保留。"""
    manifest = split_blocks("section-x", _BODY)
    victim = manifest.blocks[1]
    deleted = delete_blocks(_BODY, manifest, [victim.block_id])
    assert victim.text not in deleted
    assert manifest.blocks[0].text in deleted
    assert manifest.blocks[2].text in deleted
    # move：把 b02 移到末尾
    moved = move_blocks(_BODY, manifest, ["section-x#b02"])
    new_manifest = split_blocks("section-x", moved)
    assert new_manifest.blocks[-1].text == victim.text
    assert new_manifest.blocks[0].text == manifest.blocks[0].text
    # move 到指定块之后
    moved2 = move_blocks(_BODY, manifest, ["section-x#b01"],
                         after_block_id="section-x#b03")
    ids2 = [b.text for b in split_blocks("section-x", moved2).blocks]
    assert ids2[1] == manifest.blocks[2].text    # b03 提前
    assert ids2[2] == manifest.blocks[0].text    # 标题块落到其后


def test_deletion_neighbor_context():
    """删块复核上下文：只含删除点相邻块，不含被删块。"""
    manifest = split_blocks("section-x", _BODY)
    ctx = deletion_neighbor_context(manifest, ["section-x#b02"])
    assert manifest.blocks[1].text not in ctx    # 被删块不在上下文
    assert manifest.blocks[0].text in ctx        # 前一块在
    assert manifest.blocks[2].text in ctx        # 后一块在


def test_manifest_persistence_roundtrip():
    """落盘往返：to_dict/from_dict 完全保留；schema 不兼容拒收。"""
    manifest = split_blocks("section-x", _BODY)
    restored = BlockManifest.from_dict(manifest.to_dict())
    assert [b.block_id for b in restored.blocks] == [b.block_id for b in manifest.blocks]
    assert [b.digest for b in restored.blocks] == [b.digest for b in manifest.blocks]
    import pytest
    with pytest.raises(ValueError):
        BlockManifest.from_dict({"schema": "block-manifest-v0", "blocks": []})
