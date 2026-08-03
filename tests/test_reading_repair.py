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
