"""理解债务基础设施测试（V8 第 1/2 步验收项驱动）。"""

import json

import pytest

from feishu_assistant.kimi import KimiError
from feishu_assistant.modules.reading import PROMPTS_DIR, ReadingEngine
from feishu_assistant.modules.reading_debts import (
    DEBT_CHECK_ITEMS,
    DEBT_KINDS,
    DEBT_REGISTRY_SCHEMA,
    DEBT_RELATIONS,
    DebtAuditor,
    DebtRegistry,
    DebtRepairAction,
    DebtRepairEngine,
    DebtSpec,
    RepairRejected,
    compute_debt_id,
)
from feishu_assistant.modules.reading_graph import (
    SemanticGraph,
    render_graph_catalog,
)
from feishu_assistant.modules.reading_workflow import ReadingWorkspace

PAPER = (
    "# 论文标题\n\n## 1 引言\n\n本文提出两道校验机制，先校验输入再校验输出。\n\n"
    "## 2 方法\n\n方法细节如 Figure 1 所示，包含三个环节：筛选、验证、合成。"
    "树搜索在程序空间进行，Thompson 采样用于挑选候选。\n\n"
    "## 3 实验\n\n145 个游戏上合法动作率达到 100%。\n"
)


def _graph() -> SemanticGraph:
    g = SemanticGraph()
    g.add_source_text(PAPER)
    return g


def _nodes(g: SemanticGraph):
    concept = g.add_node("concept", "合法动作必须由代码定义")
    claim = g.add_node("assertion", "harness 使合法动作率达到 100%")
    mech = g.add_node("assertion", "Thompson 采样用于挑选候选")
    return concept, claim, mech


def _payload(subject: str, **overrides):
    payload = {
        "kind": "definition",
        "subject_node_id": subject,
        "relation": "defines",
        "object_node_ids": [],
        "premise_node_ids": [],
        "evidence_span_ids": [],
        "learner_action": "用自己的话复述合法动作的定义",
        "success_test": "能指出输入校验失败时的行为",
        "necessity_reason": "不理解定义就无法跟进机制层",
    }
    payload.update(overrides)
    return payload


# ---- 确定性身份：措辞不参与 ----

def test_same_semantics_different_wording_same_debt_id():
    g = _graph()
    concept, claim, _ = _nodes(g)
    base = _payload(claim.node_id, object_node_ids=[concept.node_id])
    a = DebtSpec.from_payload({**base,
                               "learner_action": "复述定义",
                               "success_test": "测试甲：举例说明"})
    b = DebtSpec.from_payload({**base,
                               "learner_action": "完全不同的措辞：向同事解释该定义并给出一个反例",
                               "success_test": "测试乙：不看原文写出定义要点"})
    assert a.debt_id == b.debt_id


def test_non_identity_fields_do_not_change_debt_id():
    g = _graph()
    concept, claim, _ = _nodes(g)
    span_id = next(iter(g.spans))
    base = _payload(claim.node_id, object_node_ids=[concept.node_id])
    original = DebtSpec.from_payload(base)
    for overrides in (
        {"learner_action": "另一套措辞"},
        {"success_test": "另一个判据"},
        {"evidence_span_ids": [span_id]},
        {"learner_action": "改", "success_test": "改", "evidence_span_ids": [span_id]},
        {"necessity_reason": "连 necessity 措辞也不参与身份"},
        {"revision": 3},
    ):
        variant = DebtSpec.from_payload({**base, **overrides})
        assert variant.debt_id == original.debt_id, overrides
    # 直接验算：与程序计算一致
    assert original.debt_id == compute_debt_id(
        "definition", claim.node_id, "defines", [concept.node_id], [])


def test_list_order_does_not_change_debt_id():
    # object/premise 内部顺序不参与身份（sorted 后入哈希）
    a = compute_debt_id("mechanism", "assertion-x", "transforms",
                        ["concept-a", "concept-b"], ["concept-c"])
    b = compute_debt_id("mechanism", "assertion-x", "transforms",
                        ["concept-b", "concept-a"], ["concept-c"])
    assert a == b


def test_identity_fields_change_debt_id():
    g = _graph()
    concept, claim, mech = _nodes(g)
    base = _payload(claim.node_id, object_node_ids=[concept.node_id],
                    premise_node_ids=[mech.node_id])
    original = DebtSpec.from_payload(base).debt_id
    variants = [
        {**base, "kind": "mechanism"},                        # 改 kind
        {**base, "subject_node_id": mech.node_id},            # 改 subject
        {**base, "relation": "derives"},                      # 改 relation
        {**base, "object_node_ids": [mech.node_id]},          # 改 object
        {**base, "object_node_ids": [concept.node_id, mech.node_id]},  # object 增员
        {**base, "premise_node_ids": [concept.node_id]},      # 改 premise
        {**base, "premise_node_ids": []},                     # premise 减员
    ]
    for overrides in variants:
        changed = DebtSpec.from_payload(overrides).debt_id
        assert changed != original, overrides


# ---- DebtRegistry：去重 / 克隆 / 持久化 ----

def _valid_registry(g: SemanticGraph) -> DebtRegistry:
    concept, claim, mech = _nodes(g)
    span_id = next(iter(g.spans))
    registry = DebtRegistry()
    registry.add(DebtSpec.from_payload(_payload(
        claim.node_id, kind="definition", relation="defines",
        object_node_ids=[concept.node_id], evidence_span_ids=[span_id])))
    registry.add(DebtSpec.from_payload(_payload(
        mech.node_id, kind="mechanism", relation="transforms",
        object_node_ids=[claim.node_id], premise_node_ids=[concept.node_id],
        evidence_span_ids=[span_id])))
    return registry


def test_registry_add_dedupes_same_debt_id():
    g = _graph()
    concept, claim, _ = _nodes(g)
    registry = DebtRegistry()
    spec = DebtSpec.from_payload(_payload(claim.node_id, object_node_ids=[concept.node_id]))
    assert registry.add(spec) is True
    # 同身份不同措辞：幂等去重，不覆盖、不报错、计数
    again = DebtSpec.from_payload(_payload(
        claim.node_id, object_node_ids=[concept.node_id],
        learner_action="另一套措辞", success_test="另一个判据"))
    assert registry.add(again) is False
    assert len(registry.debts) == 1
    assert registry.order == [spec.debt_id]
    assert registry.duplicate_count == 1
    assert registry.debts[spec.debt_id].learner_action == spec.learner_action  # 保留先入册者


def test_registry_clone_roundtrip_byte_identical():
    g = _graph()
    registry = _valid_registry(g)
    registry.admission_status = "clean"
    clone = registry.clone()
    dump = lambda r: json.dumps(r.to_dict(), ensure_ascii=False, sort_keys=True)
    assert dump(clone) == dump(registry)
    # 深拷贝：改克隆不影响原件
    clone.debts[clone.order[0]].learner_action = "篡改"
    assert registry.debts[registry.order[0]].learner_action != "篡改"


def test_registry_from_dict_rejects_bad_schema():
    g = _graph()
    data = _valid_registry(g).to_dict()
    data["schema"] = "debt-registry-v0"
    with pytest.raises(ValueError, match="schema 不兼容"):
        DebtRegistry.from_dict(data)
    assert DEBT_REGISTRY_SCHEMA == "debt-registry-v1"


# ---- validate：程序门禁（返回 gaps，不抛异常） ----

def test_validate_detects_fabricated_node_and_span_ids():
    g = _graph()
    concept, claim, _ = _nodes(g)
    registry = DebtRegistry()
    registry.add(DebtSpec.from_payload(_payload(
        "assertion-幽灵节点", relation="defines",
        object_node_ids=[concept.node_id, "concept-幽灵客体"],
        premise_node_ids=["concept-幽灵前提"],
        evidence_span_ids=["span-幽灵证据"])))
    gaps = registry.validate(g)
    assert any("assertion-幽灵节点" in gap for gap in gaps)
    assert any("concept-幽灵客体" in gap for gap in gaps)
    assert any("concept-幽灵前提" in gap for gap in gaps)
    assert any("span-幽灵证据" in gap for gap in gaps)


def test_validate_detects_empty_required_fields():
    g = _graph()
    concept, claim, _ = _nodes(g)
    registry = DebtRegistry()
    registry.add(DebtSpec.from_payload(_payload(
        claim.node_id, object_node_ids=[concept.node_id],
        learner_action="", success_test="  ", necessity_reason="")))
    gaps = registry.validate(g)
    assert any("necessity_reason 为空" in gap for gap in gaps)
    assert any("learner_action 为空" in gap for gap in gaps)
    assert any("success_test 为空" in gap for gap in gaps)


def test_validate_detects_illegal_kind_relation_and_tampered_id():
    g = _graph()
    concept, claim, _ = _nodes(g)
    registry = DebtRegistry()
    bad = DebtSpec.from_payload(_payload(
        claim.node_id, kind="essay", relation="mentions",
        object_node_ids=[concept.node_id]))
    assert bad.kind not in DEBT_KINDS and bad.relation not in DEBT_RELATIONS
    registry.add(bad)
    tampered = DebtSpec.from_payload(_payload(
        concept.node_id, relation="defines"))
    tampered.debt_id = "debt-手改的ID"  # 身份漂移必须被检出
    registry.add(tampered)
    gaps = registry.validate(g)
    assert any("kind 非法" in gap for gap in gaps)
    assert any("relation 非法" in gap for gap in gaps)
    assert any("身份字段不一致" in gap for gap in gaps)


def test_validate_clean_registry_has_no_gaps():
    g = _graph()
    registry = _valid_registry(g)
    assert registry.validate(g) == []


# ---- render_graph_catalog ----

def test_render_graph_catalog_deterministic_and_contains_ids():
    g = _graph()
    concept, claim, mech = _nodes(g)
    span_id = next(iter(g.spans))
    g.add_edge("supported_by", claim.node_id, span_id)
    g.add_edge("depends_on", claim.node_id, concept.node_id)
    g.add_edge("depends_on", mech.node_id, concept.node_id)
    first = render_graph_catalog(g)
    second = render_graph_catalog(g)
    assert first == second  # 确定性：两次渲染逐字一致
    # 关键 ID 与边关系都在目录里
    assert claim.node_id in first
    assert concept.node_id in first
    assert mech.node_id in first
    assert span_id in first
    assert f"supported_by→{span_id}" in first
    assert f"depends_on→[{concept.node_id}]" in first


def test_render_graph_catalog_truncates_by_priority_with_omission_note():
    g = _graph()
    _, claim, _ = _nodes(g)
    for i in range(30):
        g.add_node("concept", f"第 {i} 号背景概念，陈述内容足够长以占用渲染预算" * 2)
    catalog = render_graph_catalog(g, max_chars=600)
    assert "省略" in catalog
    # assertion 优先级最高：预算紧张时仍保留
    assert claim.node_id in catalog
    # 主体行被预算截断（省略标注本身允许少量余量）
    body = catalog.split("…（目录超出预算")[0]
    assert len(body) <= 600
    again = render_graph_catalog(g, max_chars=600)
    assert again == catalog


# ===========================================================================
# 第 2 步：DebtCompiler / DebtAuditor / DebtRepairEngine
# ===========================================================================

def _first_span(g: SemanticGraph) -> str:
    return next(iter(g.spans))


def _three_debt_registry(g: SemanticGraph) -> DebtRegistry:
    """三条真实债务（revision=4）：definition + mechanism + boundary。"""
    concept, claim, mech = _nodes(g)
    span = _first_span(g)
    registry = DebtRegistry()
    registry.add(DebtSpec.from_payload(_payload(
        claim.node_id, kind="definition", relation="defines",
        object_node_ids=[concept.node_id], evidence_span_ids=[span])))
    registry.add(DebtSpec.from_payload(_payload(
        mech.node_id, kind="mechanism", relation="transforms",
        object_node_ids=[claim.node_id], premise_node_ids=[concept.node_id],
        evidence_span_ids=[span])))
    registry.add(DebtSpec.from_payload(_payload(
        claim.node_id, kind="boundary", relation="limited_by",
        object_node_ids=[mech.node_id], evidence_span_ids=[span],
        learner_action="说出核心结论的适用边界",
        success_test="闭卷列出两条失效情形",
        necessity_reason="不知道边界会误用核心结论")))
    return registry


def _dump(registry: DebtRegistry) -> str:
    return json.dumps(registry.to_dict(), ensure_ascii=False, sort_keys=True)


def _action(op: str, registry: DebtRegistry, **params) -> DebtRepairAction:
    return DebtRepairAction.from_dict({
        "op": op,
        "expected_registry_revision": registry.registry_revision,
        "params": params,
    })


class QueueClient:
    """可编程 fake：按队列返回文本或抛异常，记录全部 prompt。"""

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def chat(self, messages, **kwargs):
        self.prompts.append(str(messages[-1]["content"]))
        if not self.outputs:
            raise AssertionError("QueueClient 调用次数超预算")
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def _checks(debt_ids, overrides=None):
    """六准入全 pass（或被 overrides 覆盖）的 checks 数组。"""
    overrides = overrides or {}
    out = []
    for item in DEBT_CHECK_ITEMS:
        outcome, evidence = overrides.get(item, ("pass", "目录与附录核对一致"))
        out.append({"item": item, "outcome": outcome,
                    "debt_ids": list(debt_ids), "evidence": evidence})
    return out


def _audit_json(checks, issues=(), repairs=()):
    return json.dumps({
        "status": "clean" if not issues else "issues_found",
        "checks": checks, "issues": list(issues), "repairs": list(repairs),
    }, ensure_ascii=False)


# ---- 编译期去重（验收 1、3）----

def test_compile_payload_dedupes_same_assertion_relation_object():
    g = _graph()
    concept, claim, _ = _nodes(g)
    span = _first_span(g)
    base = _payload(claim.node_id, object_node_ids=[concept.node_id],
                    evidence_span_ids=[span])
    payload = {"core_assertion_ids": [claim.node_id], "debts": [
        {**base, "learner_action": "复述定义", "success_test": "闭卷举例"},
        {**base, "learner_action": "完全不同措辞的动作", "success_test": "另一个判据"},
    ]}
    registry, gaps = ReadingEngine._debts_from_payload(payload, g)
    assert len(registry.debts) == 1          # 同 assertion/relation/object 不产生多个债务
    assert registry.duplicate_count == 1
    assert gaps == []


def test_same_evidence_different_relation_forms_distinct_debts():
    g = _graph()
    concept, claim, _ = _nodes(g)
    span = _first_span(g)
    registry, gaps = ReadingEngine._debts_from_payload({"debts": [
        _payload(claim.node_id, kind="definition", relation="defines",
                 object_node_ids=[concept.node_id], evidence_span_ids=[span]),
        _payload(claim.node_id, kind="evidence", relation="supported_by",
                 object_node_ids=[concept.node_id], evidence_span_ids=[span]),
    ]}, g)
    assert gaps == []
    assert len(registry.debts) == 2           # 同证据不同 relation 允许形成不同债务
    assert registry.duplicate_count == 0


# ---- DebtRepairEngine：八类 issue 的真实修复路径 ----

def test_repair_canonicalize_debt_retires_semantic_duplicate():
    g = _graph()
    concept, claim, mech = _nodes(g)
    span = _first_span(g)
    registry = DebtRegistry()
    keep = DebtSpec.from_payload(_payload(
        claim.node_id, object_node_ids=[concept.node_id], evidence_span_ids=[span]))
    dupe = DebtSpec.from_payload(_payload(
        claim.node_id, object_node_ids=[concept.node_id, mech.node_id],
        evidence_span_ids=[span]))  # 身份不同 ID 不同的语义重复
    registry.add(keep)
    registry.add(dupe)
    before = _dump(registry)
    new_registry, stats = DebtRepairEngine().apply(registry, g, [
        _action("canonicalize_debt", registry,
                keep_debt_id=keep.debt_id, retire_debt_ids=[dupe.debt_id])])
    assert dupe.debt_id not in new_registry.debts
    assert keep.debt_id in new_registry.debts
    assert stats.retired == [dupe.debt_id]
    assert new_registry.registry_revision == registry.registry_revision + 1
    assert _dump(registry) == before  # 事务不触碰原 registry


def test_repair_replace_debt_fragment_split_and_merge():
    g = _graph()
    concept, claim, mech = _nodes(g)
    span = _first_span(g)
    registry = _three_debt_registry(g)
    target = registry.order[1]  # mechanism 债务：拆成 transforms + derives 两条
    split_payloads = [
        _payload(mech.node_id, kind="mechanism", relation="transforms",
                 object_node_ids=[claim.node_id], evidence_span_ids=[span],
                 learner_action="复述采样如何变换候选", success_test="闭卷画出变换流程",
                 necessity_reason="机制第一环"),
        _payload(mech.node_id, kind="derivation", relation="derives",
                 premise_node_ids=[concept.node_id], evidence_span_ids=[span],
                 learner_action="推导候选分布的来源", success_test="闭卷写出推导链",
                 necessity_reason="机制第二环"),
    ]
    new_registry, stats = DebtRepairEngine().apply(registry, g, [
        _action("replace_debt_fragment", registry,
                retire_debt_ids=[target], creates=split_payloads)])
    assert target not in new_registry.debts
    assert len(new_registry.debts) == 4  # 退 1 拆 2
    assert len(stats.created) == 2 and stats.retired == [target]
    assert new_registry.validate(g) == []

    # 合：把刚拆出的两条并回一条（identity 可与被退者相同）
    created_ids = list(stats.created)
    merge_payload = _payload(
        mech.node_id, kind="mechanism", relation="transforms",
        object_node_ids=[claim.node_id], premise_node_ids=[concept.node_id],
        evidence_span_ids=[span],
        learner_action="完整复述机制", success_test="闭卷复述两环",
        necessity_reason="机制整体")
    merged, stats2 = DebtRepairEngine().apply(new_registry, g, [
        _action("replace_debt_fragment", new_registry,
                retire_debt_ids=created_ids, creates=[merge_payload])])
    assert len(merged.debts) == 3
    assert merged.debts[target].learner_action == "完整复述机制"  # 同身份重新入册
    assert merged.validate(g) == []


def test_repair_add_debt_from_graph():
    g = _graph()
    concept, claim, mech = _nodes(g)
    span = _first_span(g)
    registry = _three_debt_registry(g)
    payload = _payload(mech.node_id, kind="comparison", relation="contrasts_with",
                       object_node_ids=[claim.node_id], premise_node_ids=[concept.node_id],
                       evidence_span_ids=[span],
                       learner_action="对比采样与穷举", success_test="闭卷说两条差异",
                       necessity_reason="对比是核心链的一环")
    new_registry, stats = DebtRepairEngine().apply(registry, g, [
        _action("add_debt_from_graph", registry, creates=[payload])])
    assert len(new_registry.debts) == 4
    assert stats.created and stats.created[0] in new_registry.debts


def test_repair_add_debt_from_graph_rejects_fabricated_ids():
    g = _graph()
    registry = _three_debt_registry(g)
    before = _dump(registry)
    with pytest.raises(RepairRejected, match="不存在的图节点"):
        DebtRepairEngine().apply(registry, g, [
            _action("add_debt_from_graph", registry, creates=[
                _payload("assertion-幽灵", evidence_span_ids=[_first_span(g)])])])
    assert _dump(registry) == before  # 非法 creates 拒收，原 registry 字节不变


def test_repair_rebind_debt_evidence():
    g = _graph()
    registry = _three_debt_registry(g)
    spans = list(g.spans)
    debt_id = registry.order[0]
    new_registry, stats = DebtRepairEngine().apply(registry, g, [
        _action("rebind_debt_evidence", registry,
                debt_id=debt_id, evidence_span_ids=[spans[1]])])
    rebound = new_registry.debts[debt_id]
    assert rebound.evidence_span_ids == [spans[1]]
    assert rebound.debt_id == debt_id   # 证据非身份字段：ID 不变
    assert rebound.revision == 2
    assert stats.touched == [debt_id]


def test_repair_rebind_rejects_ghost_span():
    g = _graph()
    registry = _three_debt_registry(g)
    before = _dump(registry)
    with pytest.raises(RepairRejected, match="不存在的 span"):
        DebtRepairEngine().apply(registry, g, [
            _action("rebind_debt_evidence", registry,
                    debt_id=registry.order[0], evidence_span_ids=["span-幽灵"])])
    assert _dump(registry) == before


def test_repair_retire_debt():
    g = _graph()
    registry = _three_debt_registry(g)
    debt_id = registry.order[2]
    new_registry, stats = DebtRepairEngine().apply(registry, g, [
        _action("retire_debt", registry, debt_id=debt_id)])
    assert debt_id not in new_registry.debts
    assert stats.retired == [debt_id]
    assert new_registry.validate(g) == []


def test_repair_update_debt_relation_recomputes_identity():
    g = _graph()
    registry = _three_debt_registry(g)
    debt_id = registry.order[0]
    old_position = registry.order.index(debt_id)
    new_registry, stats = DebtRepairEngine().apply(registry, g, [
        _action("update_debt_relation", registry, debt_id=debt_id,
                kind="mechanism", relation="transforms",
                learner_action="复述机制", success_test="闭卷画出流程")])
    assert debt_id not in new_registry.debts  # 身份字段变化 → debt_id 重算
    (new_id,) = stats.touched
    assert new_id != debt_id
    assert new_registry.order.index(new_id) == old_position  # order 位置保持
    updated = new_registry.debts[new_id]
    assert updated.kind == "mechanism" and updated.relation == "transforms"
    assert updated.learner_action == "复述机制"
    assert updated.revision == 2
    assert new_registry.validate(g) == []


def test_repair_update_debt_relation_rejects_identity_collision():
    g = _graph()
    concept, claim, mech = _nodes(g)
    span = _first_span(g)
    registry = _three_debt_registry(g)
    # 与 order[1]（mechanism/mech/transforms/[claim]/[concept]）仅 kind/relation 不同
    twin = DebtSpec.from_payload(_payload(
        mech.node_id, kind="evidence", relation="supported_by",
        object_node_ids=[claim.node_id], premise_node_ids=[concept.node_id],
        evidence_span_ids=[span]))
    registry.add(twin)
    before = _dump(registry)
    with pytest.raises(RepairRejected, match="身份重算后与在册债务冲突"):
        DebtRepairEngine().apply(registry, g, [
            _action("update_debt_relation", registry, debt_id=twin.debt_id,
                    kind="mechanism", relation="transforms",
                    learner_action="x", success_test="y")])
    assert _dump(registry) == before


def test_repair_reassign_debt_owner():
    g = _graph()
    registry = _three_debt_registry(g)
    debt_id = registry.order[0]
    new_registry, stats = DebtRepairEngine().apply(registry, g, [
        _action("reassign_debt_owner", registry,
                debt_id=debt_id, owner_layer_id="layer-机制层")])
    assert new_registry.debts[debt_id].owner_layer_id == "layer-机制层"
    assert new_registry.debts[debt_id].debt_id == debt_id  # owner 不参与身份
    assert new_registry.validate(g) == []
    with pytest.raises(RepairRejected, match="owner_layer_id"):
        DebtRepairEngine().apply(registry, g, [
            _action("reassign_debt_owner", registry,
                    debt_id=debt_id, owner_layer_id="")])


def test_repair_revision_conflict_keeps_original_bytes():
    g = _graph()
    registry = _three_debt_registry(g)
    before = _dump(registry)
    stale = DebtRepairAction.from_dict({
        "op": "retire_debt",
        "expected_registry_revision": registry.registry_revision + 1,  # 过期版本
        "params": {"debt_id": registry.order[0]},
    })
    with pytest.raises(RepairRejected, match="revision 冲突"):
        DebtRepairEngine().apply(registry, g, [stale])
    assert _dump(registry) == before


def test_repair_rejects_creates_with_empty_required_fields():
    g = _graph()
    registry = _three_debt_registry(g)
    before = _dump(registry)
    with pytest.raises(RepairRejected, match="为空"):
        DebtRepairEngine().apply(registry, g, [
            _action("add_debt_from_graph", registry, creates=[
                _payload(registry.order[0], learner_action="")])])
    assert _dump(registry) == before


def test_repair_empty_actions_rejected():
    g = _graph()
    registry = _three_debt_registry(g)
    with pytest.raises(RepairRejected, match="空 repairs"):
        DebtRepairEngine().apply(registry, g, [])


# ---- DebtAuditor：六准入判定（验收 4）----

def test_auditor_clean_when_all_checks_pass():
    g = _graph()
    registry = _three_debt_registry(g)
    client = QueueClient(_audit_json(_checks(registry.order)))
    audited, report = DebtAuditor().audit(registry, g, client, PROMPTS_DIR)
    assert audited.admission_status == "clean"
    assert report["status"] == "clean"
    assert report["audit_calls"] == 1
    assert report["open_gaps"] == []


def test_auditor_atomic_fail_for_two_independently_failable_tests():
    g = _graph()
    registry = _three_debt_registry(g)
    checks = _checks(registry.order, overrides={
        "atomic": ("fail", "success_test 含「写出定义」与「推导性质」两个可独立失败部分")})
    client = QueueClient(_audit_json(checks, issues=[{
        "kind": "overbroad_debt", "severity": "medium",
        "debt_ids": [registry.order[0]],
        "problem": "一条债务承担定义与推导两个认知动作", "evidence": "success_test 原文"}]))
    audited, report = DebtAuditor().audit(registry, g, client, PROMPTS_DIR)
    assert audited.admission_status == "degraded"  # fail 不得 clean
    assert any("atomic=fail" in gap for gap in report["open_gaps"])


def test_auditor_necessary_fail_when_reason_points_nowhere():
    g = _graph()
    registry = _three_debt_registry(g)
    checks = _checks(registry.order, overrides={
        "necessary": ("fail", "necessity_reason 是「有助于理解全文」式空话，未指向核心链")})
    client = QueueClient(_audit_json(checks))
    audited, report = DebtAuditor().audit(registry, g, client, PROMPTS_DIR)
    assert audited.admission_status == "degraded"
    assert any("necessary=fail" in gap for gap in report["open_gaps"])


def test_auditor_testable_fail_for_vague_success_test():
    g = _graph()
    registry = _three_debt_registry(g)
    checks = _checks(registry.order, overrides={
        "testable": ("fail", "success_test 是「理解本层」类空话，闭卷不可判定")})
    client = QueueClient(_audit_json(checks))
    audited, report = DebtAuditor().audit(registry, g, client, PROMPTS_DIR)
    assert audited.admission_status == "degraded"
    assert any("testable=fail" in gap for gap in report["open_gaps"])


def test_auditor_unknown_outcome_cannot_be_clean():
    g = _graph()
    registry = _three_debt_registry(g)
    checks = _checks(registry.order, overrides={
        "source_backed": ("unknown", "附录未收录所引 span，无法核对")})
    client = QueueClient(_audit_json(checks))
    audited, report = DebtAuditor().audit(registry, g, client, PROMPTS_DIR)
    assert audited.admission_status == "degraded"  # unknown 同样不得 clean
    assert any("source_backed=unknown" in gap for gap in report["open_gaps"])


# ---- DebtAuditor + RepairEngine：修复循环（验收 8、12）----

def test_audit_canonicalizes_duplicate_and_rechecks_to_clean():
    g = _graph()
    concept, claim, mech = _nodes(g)
    span = _first_span(g)
    registry = DebtRegistry()
    keep = DebtSpec.from_payload(_payload(
        claim.node_id, object_node_ids=[concept.node_id], evidence_span_ids=[span]))
    dupe = DebtSpec.from_payload(_payload(
        claim.node_id, object_node_ids=[concept.node_id, mech.node_id],
        evidence_span_ids=[span]))
    registry.add(keep)
    registry.add(dupe)
    full = _audit_json(
        _checks(registry.order, overrides={
            "non_duplicate": ("fail", "两债务偿还同一认知动作（仅措辞与覆盖不同）")}),
        issues=[{"kind": "duplicate_debt", "severity": "high",
                 "debt_ids": [keep.debt_id, dupe.debt_id],
                 "problem": "语义重复", "evidence": "learner_action 等价"}],
        repairs=[{"op": "canonicalize_debt",
                  "expected_registry_revision": registry.registry_revision,
                  "params": {"keep_debt_id": keep.debt_id,
                             "retire_debt_ids": [dupe.debt_id]}}])
    scoped = _audit_json(_checks([keep.debt_id]))  # 修后复核：keep 全 pass
    client = QueueClient(full, scoped)
    audited, report = DebtAuditor().audit(registry, g, client, PROMPTS_DIR)
    assert audited.admission_status == "clean"
    assert dupe.debt_id not in audited.debts
    assert report["repair_rounds"] == 1 and report["repairs_applied"] == 1
    # 修后复核被执行：第二次调用是轻量复核，范围只含受影响的 keep
    assert len(client.prompts) == 2
    assert "修后轻量复核" in client.prompts[1]
    assert keep.debt_id in client.prompts[1]
    assert dupe.debt_id not in client.prompts[1].split("修后轻量复核")[1].split("\n\n## 六条")[0]


def test_audit_retires_unfounded_debt_and_rechecks_to_clean():
    g = _graph()
    concept, claim, mech = _nodes(g)
    registry = _three_debt_registry(g)
    ghost = DebtSpec.from_payload(_payload(
        mech.node_id, kind="evidence", relation="supported_by",
        object_node_ids=[claim.node_id], evidence_span_ids=["span-幽灵"]))
    registry.add(ghost)
    real_ids = [did for did in registry.order if did != ghost.debt_id]
    full = _audit_json(
        _checks(real_ids)
        + [{"item": "source_backed", "outcome": "fail", "debt_ids": [ghost.debt_id],
            "evidence": "span-幽灵 不存在于目录"}]
        + [{"item": item, "outcome": "pass", "debt_ids": [ghost.debt_id],
            "evidence": "其余维度无问题"} for item in DEBT_CHECK_ITEMS
           if item != "source_backed"],
        issues=[{"kind": "unfounded_debt", "severity": "high",
                 "debt_ids": [ghost.debt_id], "problem": "证据不存在",
                 "evidence": "span-幽灵"}],
        repairs=[{"op": "retire_debt",
                  "expected_registry_revision": registry.registry_revision,
                  "params": {"debt_id": ghost.debt_id}}])
    scoped = _audit_json([])  # 受影响债务已全部退休：checks 空数组
    client = QueueClient(full, scoped)
    audited, report = DebtAuditor().audit(registry, g, client, PROMPTS_DIR)
    assert audited.admission_status == "clean"  # 程序 validate gap 消除 + 复核通过
    assert ghost.debt_id not in audited.debts
    assert report["repair_rounds"] == 1
    assert "受影响债务已全部退休" in client.prompts[1]


def test_audit_adds_missing_debt_from_graph():
    g = _graph()
    concept, claim, mech = _nodes(g)
    span = _first_span(g)
    registry = DebtRegistry()
    registry.add(DebtSpec.from_payload(_payload(
        claim.node_id, object_node_ids=[concept.node_id], evidence_span_ids=[span])))
    existing = registry.order[0]
    missing_payload = _payload(
        mech.node_id, kind="mechanism", relation="transforms",
        object_node_ids=[claim.node_id], premise_node_ids=[concept.node_id],
        evidence_span_ids=[span],
        learner_action="复述采样机制", success_test="闭卷画出机制流程",
        necessity_reason="核心链第二环")
    missing_id = DebtSpec.from_payload(missing_payload).debt_id
    full = _audit_json(
        _checks([existing], overrides={
            "granularity": ("fail", "核心链上采样机制无对应理解债务，覆盖不全")}),
        issues=[{"kind": "missing_debt", "severity": "high", "debt_ids": [],
                 "problem": "核心链缺少机制理解债务", "evidence": "目录 assertion 闭环"}],
        repairs=[{"op": "add_debt_from_graph",
                  "expected_registry_revision": registry.registry_revision,
                  "params": {"creates": [missing_payload]}}])
    scoped = _audit_json(_checks([existing, missing_id]))
    client = QueueClient(full, scoped)
    audited, report = DebtAuditor().audit(registry, g, client, PROMPTS_DIR)
    assert audited.admission_status == "clean"
    assert missing_id in audited.debts and len(audited.debts) == 2
    assert report["repair_rounds"] == 1


def test_audit_repair_loop_bounded_two_rounds_then_degraded():
    g = _graph()
    registry = _three_debt_registry(g)
    spans = list(g.spans)
    debt_id = registry.order[0]
    fail_checks = lambda: _checks([debt_id], overrides={
        "atomic": ("fail", "仍含两个可独立失败部分")})
    full = _audit_json(fail_checks(), repairs=[{
        "op": "rebind_debt_evidence",
        "expected_registry_revision": 4,
        "params": {"debt_id": debt_id, "evidence_span_ids": [spans[1]]}}])
    scoped1 = _audit_json(fail_checks(), repairs=[{
        "op": "rebind_debt_evidence",
        "expected_registry_revision": 5,
        "params": {"debt_id": debt_id, "evidence_span_ids": [spans[0]]}}])
    scoped2 = _audit_json(fail_checks())  # 不再给修复动作
    client = QueueClient(full, scoped1, scoped2)
    audited, report = DebtAuditor().audit(registry, g, client, PROMPTS_DIR)
    assert audited.admission_status == "degraded"
    assert report["repair_rounds"] == 2           # 语义修复至多两轮
    assert report["audit_calls"] == 3             # 全量 1 + 修后复核 2
    assert any("修复 2 轮" in note for note in report["notes"])
    assert any("atomic=fail" in gap for gap in report["open_gaps"])


def test_audit_rolls_back_last_valid_when_recheck_unavailable():
    g = _graph()
    registry = _three_debt_registry(g)
    spans = list(g.spans)
    debt_id = registry.order[0]
    original_evidence = list(registry.debts[debt_id].evidence_span_ids)
    full = _audit_json(
        _checks([debt_id], overrides={
            "source_backed": ("fail", "span 原文不支撑该债务")}),
        repairs=[{"op": "rebind_debt_evidence",
                  "expected_registry_revision": registry.registry_revision,
                  "params": {"debt_id": debt_id, "evidence_span_ids": [spans[1]]}}])
    client = QueueClient(full, KimiError("审核服务挂了"))
    audited, report = DebtAuditor().audit(registry, g, client, PROMPTS_DIR)
    assert audited.admission_status == "degraded"
    # 回滚 last_valid：修复不留痕，注册表内容回到修复前
    assert audited.debts[debt_id].evidence_span_ids == original_evidence
    assert audited.registry_revision == registry.registry_revision
    assert report["repairs_applied"] == 0
    assert any("回滚 last_valid" in note for note in report["notes"])
    assert any("回滚 last_valid" in issue for issue in audited.issues)


def test_auditor_unavailable_marks_degraded_and_keeps_registry():
    g = _graph()
    registry = _three_debt_registry(g)
    before = _dump(registry)
    client = QueueClient(KimiError("额度耗尽"))
    audited, report = DebtAuditor().audit(registry, g, client, PROMPTS_DIR)
    assert audited.admission_status == "degraded"
    assert report["audit_calls"] == 0
    # 债务内容原样保留（仅状态与批注变化），不判死
    assert set(audited.debts) == set(registry.debts)
    assert _dump(registry) == before  # 入参 registry 不被触碰
    assert any("不可用" in note for note in report["notes"])


def test_auditor_format_correction_once_then_clean():
    g = _graph()
    registry = _three_debt_registry(g)
    client = QueueClient("这不是 JSON", _audit_json(_checks(registry.order)))
    audited, report = DebtAuditor().audit(registry, g, client, PROMPTS_DIR)
    assert audited.admission_status == "clean"
    assert report["audit_calls"] == 2  # 一次格式纠正


def test_auditor_double_invalid_json_degrades():
    g = _graph()
    registry = _three_debt_registry(g)
    client = QueueClient("坏输出", "还是坏输出")
    audited, report = DebtAuditor().audit(registry, g, client, PROMPTS_DIR)
    assert audited.admission_status == "degraded"
    assert report["audit_calls"] == 2  # 纠正预算只有一次
    assert any("仍非法" in note for note in report["notes"])


def test_auditor_empty_registry_degrades_without_call():
    g = _graph()
    registry = DebtRegistry()
    client = QueueClient()  # 任何调用都会 AssertionError
    audited, report = DebtAuditor().audit(registry, g, client, PROMPTS_DIR)
    assert audited.admission_status == "degraded"
    assert report["audit_calls"] == 0
    assert any("注册表为空" in note for note in report["notes"])


# ---- ReadingEngine._compile_debts ----

def _compile_debt_payload(claim, concept, span, **overrides):
    payload = {
        "kind": "definition",
        "subject_node_id": claim.node_id,
        "relation": "defines",
        "object_node_ids": [concept.node_id],
        "premise_node_ids": [],
        "evidence_span_ids": [span],
        "learner_action": "复述合法动作的定义",
        "success_test": "闭卷指出校验失败时的行为",
        "necessity_reason": "核心链第一步",
    }
    payload.update(overrides)
    return payload


def test_compile_debts_happy_path_dedupes_and_persists(tmp_path):
    g = _graph()
    concept, claim, _ = _nodes(g)
    span = _first_span(g)
    good = _compile_debt_payload(claim, concept, span)
    dupe = {**good, "learner_action": "换一套措辞的动作"}  # 同身份：去重
    client = QueueClient(json.dumps({
        "core_assertion_ids": [claim.node_id], "debts": [good, dupe]}))
    workspace = ReadingWorkspace(tmp_path)
    registry, stats = ReadingEngine(client)._compile_debts(g, "原文材料", client, workspace)
    assert len(registry.debts) == 1
    assert stats["dedup_removed"] == 1
    assert stats["compile_calls"] == 1
    assert stats["notes"] == []
    assert registry.admission_status == "pending"  # 落盘 pending，待审
    assert workspace.get_json("debt-registry")["schema"] == DEBT_REGISTRY_SCHEMA
    # 目录与原文包都在 CACHE_BREAK 前（随源缓存）
    prompt = client.prompts[0]
    assert prompt.index("原文包") < prompt.index("<!--CACHE_BREAK-->")
    assert prompt.index("语义图目录") < prompt.index("<!--CACHE_BREAK-->")
    assert claim.node_id in prompt  # 目录注入真实 ID


def test_compile_debts_correction_fixes_fabricated_ids():
    g = _graph()
    concept, claim, _ = _nodes(g)
    span = _first_span(g)
    ghost = _compile_debt_payload(claim, concept, span,
                                  subject_node_id="assertion-幽灵")
    good = _compile_debt_payload(claim, concept, span)
    client = QueueClient(
        json.dumps({"core_assertion_ids": [], "debts": [ghost]}),
        json.dumps({"core_assertion_ids": [claim.node_id], "debts": [good]}),
    )
    registry, stats = ReadingEngine(client)._compile_debts(g, "原文材料", client, None)
    assert stats["compile_calls"] == 2           # 一次 correction
    assert len(registry.debts) == 1              # 采用修正版
    assert stats["notes"] == []                  # 修正成功无需剔除
    assert "上一次输出存在非法债务项" in client.prompts[1]
    assert "assertion-幽灵" in client.prompts[1]  # 缺口回灌


def test_compile_debts_prunes_illegal_items_when_correction_still_bad():
    g = _graph()
    concept, claim, mech = _nodes(g)
    span = _first_span(g)
    good1 = _compile_debt_payload(claim, concept, span)
    ghost1 = _compile_debt_payload(claim, concept, span,
                                   subject_node_id="assertion-幽灵甲")
    good2 = _compile_debt_payload(claim, concept, span,
                                  kind="mechanism", relation="transforms",
                                  subject_node_id=mech.node_id)
    ghost2 = _compile_debt_payload(claim, concept, span,
                                   subject_node_id="assertion-幽灵乙")
    client = QueueClient(
        json.dumps({"core_assertion_ids": [], "debts": [good1, ghost1]}),
        json.dumps({"core_assertion_ids": [], "debts": [good2, ghost2]}),
    )
    registry, stats = ReadingEngine(client)._compile_debts(g, "原文材料", client, None)
    assert stats["compile_calls"] == 2
    # 两次各有一个幽灵（缺口数相同，保留首版）→ 剔除幽灵甲，保留 good1
    assert len(registry.debts) == 1
    assert registry.debts[registry.order[0]].kind == "definition"
    assert any("剔除非法债务" in note for note in stats["notes"])
    assert registry.admission_status == "pending"  # 剔除后继续待审，不判死


def test_compile_debts_invalid_json_three_times_degrades():
    g = _graph()
    _nodes(g)
    client = QueueClient("不是 JSON", "还不是 JSON", "依然不是 JSON")
    registry, stats = ReadingEngine(client)._compile_debts(g, "原文材料", client, None)
    assert registry.admission_status == "degraded"
    assert len(registry.debts) == 0
    assert stats["compile_calls"] == 3            # 首发 + 两次格式重试，末次也解析
    assert any("连续返回非法 JSON" in note for note in stats["notes"])


def test_compile_debts_kimi_error_degrades_without_dying():
    g = _graph()
    _nodes(g)
    client = QueueClient(KimiError("限流"))
    registry, stats = ReadingEngine(client)._compile_debts(g, "原文材料", client, None)
    assert registry.admission_status == "degraded"
    assert stats["compile_calls"] == 0
    assert any("调用失败" in note for note in stats["notes"])


def test_compile_debts_weak_graph_skips_model_call():
    g = _graph()
    g.add_node("concept", "只有概念没有主张")      # 零 assertion：弱图
    client = QueueClient()                         # 任何调用都会 AssertionError
    registry, stats = ReadingEngine(client)._compile_debts(g, "原文材料", client, None)
    assert registry.admission_status == "degraded"
    assert stats["compile_calls"] == 0
    assert any("弱图" in note for note in stats["notes"])


def test_compile_debts_checkpoint_reused_without_calls(tmp_path):
    g = _graph()
    concept, claim, _ = _nodes(g)
    span = _first_span(g)
    good = _compile_debt_payload(claim, concept, span)
    workspace = ReadingWorkspace(tmp_path)
    first_client = QueueClient(json.dumps({"core_assertion_ids": [], "debts": [good]}))
    first, _ = ReadingEngine(first_client)._compile_debts(g, "原文材料", first_client, workspace)
    second_client = QueueClient()  # 空队列：命中检查点则零调用
    second, stats = ReadingEngine(second_client)._compile_debts(
        g, "原文材料", second_client, workspace)
    assert stats["compile_calls"] == 0
    assert _dump(second) == _dump(first)
    assert any("检查点" in note for note in stats["notes"])
