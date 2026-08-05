"""RepairEngine 专项测试。"""

import pytest

from feishu_assistant.modules.reading_plan import PlanUnit, ReadingPlan
from feishu_assistant.modules.reading_repair import (
    RepairAction,
    RepairEngine,
    RepairRejected,
)


def _plan() -> ReadingPlan:
    plan = ReadingPlan()
    for uid, kind in (("section-a", "section"), ("section-b", "section"),
                      ("layer-x", "layer")):
        unit = PlanUnit(unit_id=uid, kind=kind,
                        data={"title": uid, "name": uid, "purpose": "evidence",
                              "teaching_mode": "process", "question": uid + "问题"})
        unit.semantic_digest = unit.compute_semantic_digest()
        unit.display_digest = unit.compute_display_digest()
        plan.units[uid] = unit
    plan.order = {"sections": ["section-a", "section-b"], "layers": ["layer-x"]}
    plan.preparations = [{
        "concept": "合法动作", "concept_id": "concept-1",
        "owner_section_id": "section-a", "target_layer_ids": ["layer-x"],
    }]
    return plan


def _action(op, params, rev=1, plan_rev=1, expected=None):
    return RepairAction(issue_id="i1", op=op, params=params,
                        expected_plan_revision=plan_rev,
                        expected_revisions=expected or {})


def test_reassign_preparation_moves_owner_and_bumps_both():
    plan = _plan()
    new_plan, stats = RepairEngine().apply(plan, [_action(
        "reassign_preparation",
        {"concept_id": "concept-1", "from_section_id": "section-a",
         "to_section_id": "section-b"},
        expected={"section-a": 1, "section-b": 1},
    )])
    prep = new_plan.preparations[0]
    assert prep["owner_section_id"] == "section-b"
    assert new_plan.units["section-a"].semantic_revision == 2
    assert new_plan.units["section-b"].semantic_revision == 2
    assert new_plan.units["layer-x"].semantic_revision == 1  # 未变
    assert new_plan.plan_revision == 2
    assert plan.preparations[0]["owner_section_id"] == "section-a"  # 原件未动


def test_revision_conflict_keeps_plan_untouched():
    plan = _plan()
    raw = plan.to_dict()
    with pytest.raises(RepairRejected, match="revision 冲突"):
        RepairEngine().apply(plan, [_action(
            "update_layer_mode", {"layer_id": "layer-x", "mode": "comparison"},
            expected={"layer-x": 99},
        )])
    assert plan.to_dict() == raw


def test_merge_returns_unsupported_not_applied():
    with pytest.raises(RepairRejected, match="unsupported_repair"):
        RepairAction.from_dict({"op": "merge_layers", "params": {}})


def test_update_unit_fields_whitelist_and_forbidden():
    plan = _plan()
    with pytest.raises(RepairRejected, match="禁止修改字段"):
        RepairEngine().apply(plan, [_action(
            "update_unit_fields",
            {"unit_id": "layer-x", "fields": {"semantic_revision": 9}})])
    with pytest.raises(RepairRejected, match="不允许修改字段"):
        RepairEngine().apply(plan, [_action(
            "update_unit_fields",
            {"unit_id": "section-a", "fields": {"teaching_mode": "process"}})])
    new_plan, _ = RepairEngine().apply(plan, [_action(
        "update_unit_fields",
        {"unit_id": "layer-x", "fields": {"question": "新的核心问题"}})])
    assert new_plan.units["layer-x"].data["question"] == "新的核心问题"
    assert new_plan.units["layer-x"].semantic_revision == 2


def test_batch_transaction_single_plan_revision_bump():
    plan = _plan()
    new_plan, stats = RepairEngine().apply(plan, [
        _action("remove_preparation", {"concept_id": "concept-1"}),
        _action("update_layer_mode",
                {"layer_id": "layer-x", "mode": "comparison"}),
    ])
    assert stats.applied == 2
    assert new_plan.plan_revision == 2  # 整批只 +1
    assert not new_plan.preparations
    assert new_plan.units["layer-x"].data["teaching_mode"] == "comparison"


def test_digest_unchanged_means_no_revision_bump():
    plan = _plan()
    new_plan, stats = RepairEngine().apply(plan, [_action(
        "update_unit_fields",
        {"unit_id": "layer-x",
         "fields": {"question": plan.units["layer-x"].data["question"]}},  # 同值写入
    )])
    assert new_plan.units["layer-x"].semantic_revision == 1
    assert new_plan.plan_revision == 1
    assert stats.changed_units == []


# ---------------------------------------------------------------------------
# V8 第 7 步：replace_plan_fragment（四形态 + 失败回滚 + 无实质变化）
# ---------------------------------------------------------------------------

def _frag_graph():
    from feishu_assistant.modules.reading_graph import SemanticGraph
    g = SemanticGraph()
    g.add_source_text("第一段原文。\n\n第二段原文。")
    c1 = g.add_node("concept", "概念甲")
    a1 = g.add_node("assertion", "主张一")
    a2 = g.add_node("assertion", "主张二")
    a3 = g.add_node("assertion", "主张三")
    g.add_edge("depends_on", a1.node_id, c1.node_id)
    g.add_edge("depends_on", a2.node_id, c1.node_id)
    g.add_edge("depends_on", a3.node_id, c1.node_id)
    return g, c1, a1, a2, a3


def _frag_debts(c1, a1, a2, a3, span_id):
    from feishu_assistant.modules.reading_debts import DebtRegistry, DebtSpec
    reg = DebtRegistry()

    def _debt(kind, subject, relation, action):
        return DebtSpec.from_payload({
            "kind": kind, "subject_node_id": subject, "relation": relation,
            "object_node_ids": [], "premise_node_ids": [],
            "evidence_span_ids": [span_id],
            "learner_action": action, "success_test": f"闭卷完成：{action}",
            "necessity_reason": "核心链必需"})

    d1 = _debt("mechanism", a1.node_id, "transforms", "复述主张一机制")
    d2 = _debt("definition", c1.node_id, "defines", "定义概念甲")
    d3 = _debt("boundary", a3.node_id, "limited_by", "说明主张三边界")
    d4 = _debt("evidence", a2.node_id, "supported_by", "说明主张二证据")
    for d in (d1, d2, d3, d4):
        reg.add(d)
    reg.admission_status = "clean"
    return reg, d1, d2, d3, d4


def _layer(uid, name, primary, debt_ids, span_id, debts):
    unit = PlanUnit(
        unit_id=uid, kind="layer",
        data={"name": name, "why": f"承接{name}", "question": f"{name}如何成立",
              "primary_assertion_id": primary, "debt_ids": list(debt_ids),
              "supporting_node_ids": [], "prerequisite_concept_ids": [],
              "source_span_ids": [span_id],
              "must_explain": [debts[d].learner_action for d in debt_ids],
              "success_test": [debts[d].success_test for d in debt_ids]},
        assertion_ids=[primary])
    unit.semantic_digest = unit.compute_semantic_digest()
    unit.display_digest = unit.compute_display_digest()
    return unit


def _frag_plan(layer_b_debts=("d2",)):
    """基础夹具：节一 + 层A(primary a1, [d1]) + 层B(primary a2, 默认 [d2])；
    prep 概念甲 → 层B。返回 (graph, registry, plan, 节点组, 债务组)。"""
    g, c1, a1, a2, a3 = _frag_graph()
    span_id = next(iter(g.spans))
    reg, d1, d2, d3, d4 = _frag_debts(c1, a1, a2, a3, span_id)
    debts = {"d1": d1, "d2": d2, "d3": d3, "d4": d4}
    plan = ReadingPlan()
    section = PlanUnit(unit_id="section-a", kind="section",
                       data={"title": "节一", "purpose": "orientation",
                             "purposes": ["orientation"]})
    section.semantic_digest = section.compute_semantic_digest()
    section.display_digest = section.compute_display_digest()
    plan.units["section-a"] = section
    plan.units["layer-a"] = _layer("layer-a", "层A", a1.node_id, [d1.debt_id],
                                   span_id, {d.debt_id: d for d in debts.values()})
    plan.units["layer-b"] = _layer(
        "layer-b", "层B", a2.node_id,
        [debts[k].debt_id for k in layer_b_debts],
        span_id, {d.debt_id: d for d in debts.values()})
    plan.order = {"sections": ["section-a"], "layers": ["layer-a", "layer-b"]}
    plan.preparations = [{
        "concept": "概念甲", "concept_id": c1.node_id,
        "owner_section_id": "section-a", "target_layer_ids": ["layer-b"],
    }]
    plan.precompute_boundaries()
    nodes = type("N", (), {"c1": c1, "a1": a1, "a2": a2, "a3": a3})
    return g, reg, plan, nodes, debts


def _frag(params, plan_rev=1):
    return RepairAction(issue_id="i1", op="replace_plan_fragment",
                        target_ids=list(params.get("target_ids") or []),
                        expected_plan_revision=plan_rev, params=params)


def test_fragment_op_accepted_bare_merge_still_rejected():
    action = RepairAction.from_dict({"op": "replace_plan_fragment",
                                     "issue_id": "i1", "params": {}})
    assert action.op == "replace_plan_fragment"
    with pytest.raises(RepairRejected, match="unsupported_repair"):
        RepairAction.from_dict({"op": "merge_sections", "params": {}})


def test_fragment_merge_absorbs_retires_and_writes_tombstone():
    """形态一（吸收）：保留 A、B 退休——债务/prep 迁入 A，tombstone 写入，
    owner 重绑，原 Plan 字节不变。"""
    g, reg, plan, _n, debts = _frag_plan()
    d1, d2 = debts["d1"], debts["d2"]
    raw = plan.to_dict()
    new_plan, stats = RepairEngine(g, registry=reg).apply(plan, [_frag({
        "target_ids": ["layer-a", "layer-b"],
        "keeps": ["layer-a"],
        "retires": [{"unit_id": "layer-b", "replaced_by": ["layer-a"]}],
    })])
    assert new_plan.order["layers"] == ["layer-a"]
    assert "layer-b" not in new_plan.units
    assert new_plan.tombstones["layer-b"] == "layer-a"
    unit = new_plan.units["layer-a"]
    assert unit.data["debt_ids"] == [d1.debt_id, d2.debt_id]  # B 的债务迁入 A
    # 派生值按 Registry 重派生（唯一事实源）
    assert unit.data["must_explain"] == [d1.learner_action, d2.learner_action]
    assert unit.data["success_test"] == [d1.success_test, d2.success_test]
    assert new_plan.preparations[0]["target_layer_ids"] == ["layer-a"]  # prep 重定向
    assert stats.retired == ["layer-b"] and stats.created == []
    assert new_plan.plan_revision == 2
    assert new_plan.boundaries == []  # 单层无相邻对（boundary 已重算）
    assert stats.registry.debts[d2.debt_id].owner_layer_id == "layer-a"
    assert plan.to_dict() == raw  # 原 Plan 字节不变


def test_fragment_two_into_one_creates_and_retires():
    """形态二（二合一）：创建 C、A/B 退休——债务按 replaced_by 迁入 C，
    creates 按既有稳定 ID 规则生成 unit_id（类型+语义角色+绑定 ID）。"""
    from feishu_assistant.modules.reading_plan import layer_unit_id
    g, reg, plan, nodes, debts = _frag_plan()
    d1, d2 = debts["d1"], debts["d2"]
    c_id = layer_unit_id("合并主张一与二", "两者如何接力", [nodes.a1.node_id])
    new_plan, stats = RepairEngine(g, registry=reg).apply(plan, [_frag({
        "target_ids": ["layer-a", "layer-b"],
        "creates": [{"kind": "layer",
                     "data": {"name": "层C", "why": "合并主张一与二",
                              "question": "两者如何接力",
                              "teaching_mode": "process"},
                     "primary_assertion_id": nodes.a1.node_id, "debt_ids": []}],
        # replaced_by 用 creates 的 name 别名（程序生成 ID 调用方无法预知）
        "retires": [{"unit_id": "layer-a", "replaced_by": ["层C"]},
                    {"unit_id": "layer-b", "replaced_by": ["层C"]}],
    })])
    assert new_plan.order["layers"] == [c_id]  # 插入首个退休位置
    unit = new_plan.units[c_id]
    assert unit.data["debt_ids"] == [d1.debt_id, d2.debt_id]
    assert unit.data["must_explain"] == [d1.learner_action, d2.learner_action]
    assert new_plan.tombstones == {"layer-a": c_id, "layer-b": c_id}
    assert new_plan.preparations[0]["target_layer_ids"] == [c_id]
    assert stats.created == [c_id]
    assert stats.retired == ["layer-a", "layer-b"]
    assert new_plan.plan_revision == 2
    assert unit.semantic_revision == 1  # 新建单元 revision 从 1 起


def test_fragment_one_into_two_splits_with_debt_changes():
    """形态三（一分二）：创建 C/D、A 退休——A 的债务先自动迁入 C，
    debt_changes 显式把 d2 分给 D；B 不受影响。"""
    from feishu_assistant.modules.reading_plan import layer_unit_id
    g, reg, plan, nodes, debts = _frag_plan()
    d1, d2 = debts["d1"], debts["d2"]
    # A 改为持两条债务（d1+d2），B 持有 d3（primary 换 a3，腾出 a2 给 D）
    plan.units["layer-a"].data["debt_ids"] = [d1.debt_id, d2.debt_id]
    plan.units["layer-a"].data["must_explain"] = [
        d1.learner_action, d2.learner_action]
    plan.units["layer-a"].data["success_test"] = [
        d1.success_test, d2.success_test]
    plan.units["layer-b"] = _layer("layer-b", "层B", nodes.a3.node_id,
                                   [debts["d3"].debt_id], next(iter(g.spans)),
                                   {d.debt_id: d for d in debts.values()})
    plan.precompute_boundaries()
    c_id = layer_unit_id("拆分上半", "主张一如何成立", [nodes.a1.node_id])
    d_id = layer_unit_id("拆分下半", "概念甲如何支撑", [nodes.a2.node_id])
    new_plan, stats = RepairEngine(g, registry=reg).apply(plan, [_frag({
        "target_ids": ["layer-a"],
        "creates": [{"kind": "layer",
                     "data": {"name": "层C", "why": "拆分上半",
                              "question": "主张一如何成立"},
                     "primary_assertion_id": nodes.a1.node_id,
                     "debt_ids": [d1.debt_id]},
                    {"kind": "layer",
                     "data": {"name": "层D", "why": "拆分下半",
                              "question": "概念甲如何支撑"},
                     "primary_assertion_id": nodes.a2.node_id, "debt_ids": []}],
        "retires": [{"unit_id": "layer-a", "replaced_by": ["层C", "层D"]}],
        "debt_changes": [{"debt_id": d2.debt_id, "owner_layer_id": "层D"}],
    })])
    assert new_plan.order["layers"] == [c_id, d_id, "layer-b"]  # 占据 A 的位置
    assert new_plan.tombstones["layer-a"] == c_id  # replaced_by[0]
    assert new_plan.units[c_id].data["debt_ids"] == [d1.debt_id]
    assert new_plan.units[d_id].data["debt_ids"] == [d2.debt_id]
    assert new_plan.units["layer-b"].data["debt_ids"] == [debts["d3"].debt_id]
    assert stats.registry.debts[d2.debt_id].owner_layer_id == d_id
    assert new_plan.plan_revision == 2


def test_fragment_dependency_only_adjustments_keep_node_count():
    """形态四（仅调依赖）：edge_changes 迁债务 + preparation_changes 改目标，
    节点数量不变，无 creates/retires。"""
    g, reg, plan, nodes, debts = _frag_plan(layer_b_debts=("d2", "d4"))
    d1, d2 = debts["d1"], debts["d2"]
    new_plan, stats = RepairEngine(g, registry=reg).apply(plan, [_frag({
        "target_ids": ["layer-a", "layer-b"],
        "edge_changes": [{"from_debt_id": d2.debt_id,
                          "to_owner_layer_id": "layer-a"}],
        "preparation_changes": [{"concept_id": nodes.c1.node_id,
                                 "target_layer_ids": ["layer-a"]}],
    })])
    assert set(new_plan.order["layers"]) == {"layer-a", "layer-b"}  # 节点数不变
    assert stats.created == [] and stats.retired == []
    assert new_plan.units["layer-a"].data["debt_ids"] == [d1.debt_id, d2.debt_id]
    assert new_plan.units["layer-b"].data["debt_ids"] == [debts["d4"].debt_id]
    assert new_plan.preparations[0]["target_layer_ids"] == ["layer-a"]
    assert stats.registry.debts[d2.debt_id].owner_layer_id == "layer-a"
    assert new_plan.plan_revision == 2


def test_fragment_rejected_rolls_back_byte_identical():
    """失败回滚：creates 引用不存在 ID / retires 缺 replaced_by /
    validate_plan 出 gaps——原 Plan 字节不变 + RepairRejected。"""
    g, reg, plan, nodes, debts = _frag_plan()
    engine = RepairEngine(g, registry=reg)
    # ① creates 的 primary assertion 编造
    raw = plan.to_dict()
    with pytest.raises(RepairRejected, match="primary assertion 不存在"):
        engine.apply(plan, [_frag({
            "target_ids": ["layer-a"],
            "creates": [{"kind": "layer", "data": {"name": "层C"},
                         "primary_assertion_id": "assertion-幽灵"}],
            "retires": [{"unit_id": "layer-a", "replaced_by": ["层C"]}],
        })])
    assert plan.to_dict() == raw
    # ② retires 缺 replaced_by
    with pytest.raises(RepairRejected, match="replaced_by"):
        engine.apply(plan, [_frag({
            "target_ids": ["layer-a"],
            "retires": [{"unit_id": "layer-a"}],
        })])
    assert plan.to_dict() == raw
    # ③ 迁移后 validate_plan 出 gaps（d1 subject 不在 layer-b primary 闭包内）
    with pytest.raises(RepairRejected, match="不变量未通过"):
        engine.apply(plan, [_frag({
            "target_ids": ["layer-a", "layer-b"],
            "keeps": ["layer-b"],
            "retires": [{"unit_id": "layer-a", "replaced_by": ["layer-b"]}],
        })])
    assert plan.to_dict() == raw


def test_fragment_noop_does_not_bump_plan_revision():
    """无实质变化（债务原地指派）不增 plan_revision，created/retired/changed 全空。"""
    g, reg, plan, _n, debts = _frag_plan()
    d2 = debts["d2"]
    new_plan, stats = RepairEngine(g, registry=reg).apply(plan, [_frag({
        "target_ids": ["layer-b"],
        "debt_changes": [{"debt_id": d2.debt_id, "owner_layer_id": "layer-b"}],
    })])
    assert new_plan.plan_revision == 1
    assert stats.created == [] and stats.retired == []
    assert stats.changed_units == []
