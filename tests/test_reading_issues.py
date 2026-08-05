"""V8 第 4 步：统一 ReviewIssue / RepairAction v2 / IssueLedger / RepairRegistry 测试。

覆盖：from_legacy 归一映射与 issue_id 稳定去重；台账状态机（禁静默丢失、
非法迁移拒收、落盘往返）；路由表全 kind 覆盖与重点路由值；程序保证函数
（validate_review_outcome / check_action_coverage / check_action_evidence）；
以及 run() 端到端接线（issue-ledger 落盘、全部终态、metadata 透出）。
"""

import json

import pytest

from feishu_assistant.kimi import KimiError
from feishu_assistant.modules.reading import ReadingEngine
from feishu_assistant.modules.reading_graph import SemanticGraph
from feishu_assistant.modules.reading_issues import (
    ISSUE_KINDS,
    ISSUE_LEDGER_SCHEMA,
    ISSUE_STAGES,
    REPAIR_ROUTES,
    ROOT_SCOPES,
    UNIFIED_REPAIR_OPS,
    IssueLedger,
    IssueRouteError,
    RepairAction,
    ReviewIssue,
    check_action_coverage,
    check_action_evidence,
    route,
    validate_review_outcome,
)
from feishu_assistant.modules.reading_workflow import ReadingWorkspace


def _issue(stage="unit", kind="general", scope="body", severity="high",
           targets=("u1",), invariant="", problem="问题"):
    return ReviewIssue(stage=stage, kind=kind, root_scope=scope, severity=severity,
                       target_ids=list(targets), violated_invariant=invariant,
                       problem=problem)


# ---------------------------------------------------------------------------
# from_legacy：旧自由 dict → 归一映射
# ---------------------------------------------------------------------------

def test_from_legacy_severity_chinese_mapping():
    assert ReviewIssue.from_legacy({"severity": "高", "problem": "p"}, "unit").severity == "high"
    assert ReviewIssue.from_legacy({"severity": "中", "problem": "p"}, "unit").severity == "medium"
    assert ReviewIssue.from_legacy({"severity": "低", "problem": "p"}, "unit").severity == "low"
    # 已是枚举直通；缺省/未知按 high（与局审闸门「缺省 severity 按高」同口径）
    assert ReviewIssue.from_legacy({"severity": "medium", "problem": "p"}, "unit").severity == "medium"
    assert ReviewIssue.from_legacy({"problem": "p"}, "unit").severity == "high"
    assert ReviewIssue.from_legacy({"severity": "未知", "problem": "p"}, "unit").severity == "high"


def test_from_legacy_structural_duplication_prefix():
    issue = ReviewIssue.from_legacy(
        {"severity": "低", "problem": "结构重复：两节复述同一机制"}, "unit")
    assert issue.kind == "structural_duplication"
    # 结构重复硬判高严重度，模型降级无效（与 _local_review_passed 硬规则一致）
    assert issue.severity == "high"


def test_from_legacy_location_kept_in_problem():
    issue = ReviewIssue.from_legacy(
        {"location": "层A", "problem": "数字对不上"}, "final")
    assert issue.problem == "层A：数字对不上"
    # location 不进身份字段（violated_invariant 保持空）
    assert issue.violated_invariant == ""


def test_from_legacy_kind_and_scope_mapping():
    # 显式合法 kind 保留
    issue = ReviewIssue.from_legacy({"kind": "fact_error", "problem": "p"}, "unit")
    assert issue.kind == "fact_error"
    # 非法 kind → general，原文 kind 记入 violated_invariant 可审计
    issue = ReviewIssue.from_legacy({"kind": "断链", "problem": "p"}, "plan")
    assert issue.kind == "general"
    assert issue.violated_invariant == "断链"
    # stage → 默认 root_scope
    assert ReviewIssue.from_legacy({"problem": "p"}, "plan").root_scope == "plan"
    assert ReviewIssue.from_legacy({"problem": "p"}, "unit").root_scope == "body"
    assert ReviewIssue.from_legacy({"problem": "p"}, "figure").root_scope == "asset"
    assert ReviewIssue.from_legacy({"problem": "p"}, "final").root_scope == "body"
    # 显式合法 root_scope 优先于 stage 默认
    assert ReviewIssue.from_legacy(
        {"problem": "p", "root_scope": "graph"}, "final").root_scope == "graph"
    # kind_override（review_unavailable 口径接线用）
    issue = ReviewIssue.from_legacy(
        {"severity": "中", "problem": "审核服务不可用"}, "unit",
        kind_override="review_unavailable")
    assert issue.kind == "review_unavailable"


def test_from_legacy_target_ids_merge():
    issue = ReviewIssue.from_legacy(
        {"problem": "p", "target_id": "layer-a", "target_ids": ["section-a"]},
        "unit", default_target_ids=["layer-a", "section-b"])
    assert issue.target_ids == ["section-a", "layer-a", "section-b"]


def test_from_legacy_rejects_non_dict():
    with pytest.raises(ValueError):
        ReviewIssue.from_legacy("不是对象", "unit")
    with pytest.raises(ValueError):
        ReviewIssue.from_legacy({"problem": "p"}, "不存在阶段")


# ---------------------------------------------------------------------------
# issue_id：稳定身份与幂等去重
# ---------------------------------------------------------------------------

def test_issue_id_stable_and_dedup():
    a = ReviewIssue.from_legacy({"severity": "高", "location": "层A", "problem": "事实错误"}, "unit")
    b = ReviewIssue.from_legacy({"severity": "高", "location": "层A", "problem": "事实错误"}, "unit")
    assert a.issue_id == b.issue_id
    ledger = IssueLedger()
    assert ledger.register([a, b, a]) == [a.issue_id] * 3
    assert len(ledger.entries) == 1  # 同一问题重复报告同 ID，幂等去重


def test_issue_id_ignores_wording_and_target_order():
    a = _issue(problem="措辞一", targets=("u1", "u2"))
    b = _issue(problem="措辞二完全不同", targets=("u2", "u1"))
    assert a.issue_id == b.issue_id  # problem 与 target 顺序不参与身份


def test_issue_id_changes_with_identity_fields():
    base = _issue()
    assert _issue(stage="plan").issue_id != base.issue_id
    assert _issue(kind="fact_error").issue_id != base.issue_id
    assert _issue(scope="plan").issue_id != base.issue_id
    assert _issue(targets=("u2",)).issue_id != base.issue_id
    assert _issue(invariant="债务唯一归属").issue_id != base.issue_id


def test_issue_kind_enum_covers_debt_eight():
    from feishu_assistant.modules.reading_debts import DEBT_ISSUE_KINDS
    for kind in DEBT_ISSUE_KINDS:
        assert kind in ISSUE_KINDS
    for kind in ("structural_duplication", "fact_error", "number_inconsistency",
                 "strong_causality", "scope_violation", "contract_gap",
                 "debt_uncovered", "debt_duplicated", "boundary_fail",
                 "preparation_illegal", "figure_misread", "format_gap",
                 "review_unavailable", "general"):
        assert kind in ISSUE_KINDS


# ---------------------------------------------------------------------------
# RepairAction v2
# ---------------------------------------------------------------------------

def test_repair_action_v2_requires_resolves():
    with pytest.raises(ValueError):
        RepairAction.from_dict({"op": "retire_debt", "target_ids": ["d1"]})
    with pytest.raises(ValueError):
        RepairAction(op="", resolves_issue_ids=["i1"])
    with pytest.raises(ValueError):
        RepairAction(op="retire_debt", resolves_issue_ids=[])


def test_repair_action_v2_legacy_issue_id_compatible():
    action = RepairAction.from_dict({
        "issue_id": "i1", "op": "update_unit_fields", "target_ids": ["u1"],
        "params": {"unit_id": "u1", "fields": {}}})
    assert action.resolves_issue_ids == ["i1"]  # v1 单值兼容解析
    both = RepairAction.from_dict({
        "issue_id": "i1", "resolves_issue_ids": ["i2", "i1"],
        "op": "retire_debt", "target_ids": ["d1"]})
    assert both.resolves_issue_ids == ["i2", "i1"]  # 合并去重


def test_repair_action_id_stable_and_roundtrip():
    payload = {"resolves_issue_ids": ["i1"], "op": "update_layer_mode",
               "target_ids": ["layer-a", "layer-b"],
               "params": {"layer_id": "layer-a", "mode": "comparison"}}
    a = RepairAction.from_dict(payload)
    payload2 = dict(payload, target_ids=["layer-b", "layer-a"])
    b = RepairAction.from_dict(payload2)
    assert a.action_id == b.action_id  # 同内容同 ID（target 顺序不参与）
    c = RepairAction.from_dict(dict(payload, op="update_unit_fields",
                                    params={"unit_id": "u", "fields": {}}))
    assert c.action_id != a.action_id
    assert RepairAction.from_dict(a.to_dict()).action_id == a.action_id


# ---------------------------------------------------------------------------
# IssueLedger：状态机与禁静默丢失
# ---------------------------------------------------------------------------

def test_ledger_state_machine_open_to_resolved():
    ledger = IssueLedger()
    issue = _issue()
    ledger.register([issue])
    assert [i.issue_id for i in ledger.open_issues()] == [issue.issue_id]
    assert not ledger.all_terminal()
    assert [i.issue_id for i in ledger.unterminated()] == [issue.issue_id]
    ledger.mark_resolved([issue.issue_id], "action-x")
    assert ledger.all_terminal()
    assert ledger.unterminated() == []
    assert ledger.entries[issue.issue_id].resolved_by == "action-x"


def test_ledger_rejects_illegal_transitions():
    ledger = IssueLedger()
    a, b = _issue(problem="a"), _issue(targets=("u2",), problem="b")
    ledger.register([a, b])
    ledger.mark_resolved([a.issue_id], "x")
    with pytest.raises(ValueError):  # resolved→resolved 非法
        ledger.mark_resolved([a.issue_id], "y")
    with pytest.raises(ValueError):  # resolved→degraded 非法
        ledger.mark_degraded([a.issue_id], "原因")
    ledger.mark_degraded([b.issue_id], "带原因降级")
    with pytest.raises(ValueError):  # degraded→resolved 非法
        ledger.mark_resolved([b.issue_id], "x")
    with pytest.raises(ValueError):  # 未注册不得迁移（禁静默丢失的反面）
        ledger.mark_resolved(["issue-幽灵"], "x")
    with pytest.raises(ValueError):  # resolved 必须携带凭据
        ledger.mark_resolved([], "  ")
    with pytest.raises(ValueError):  # degraded 必须携带原因
        ledger.mark_degraded([b.issue_id], "")


def test_ledger_open_issues_stage_filter():
    ledger = IssueLedger()
    plan_issue = _issue(stage="plan", scope="plan")
    unit_issue = _issue()
    ledger.register([plan_issue, unit_issue])
    assert {i.issue_id for i in ledger.open_issues(stage="plan")} == {plan_issue.issue_id}
    assert len(ledger.open_issues()) == 2


def test_ledger_persist_roundtrip():
    ledger = IssueLedger()
    a, b = _issue(problem="a"), _issue(targets=("u2",), problem="b")
    ledger.register([a, b])
    ledger.mark_resolved([a.issue_id], "action-x")
    ledger.mark_degraded([b.issue_id], "带原因降级")
    restored = IssueLedger.from_dict(json.loads(json.dumps(ledger.to_dict())))
    assert restored.summary() == ledger.summary()
    assert restored.entries[a.issue_id].resolved_by == "action-x"
    assert restored.entries[b.issue_id].degraded_reason == "带原因降级"
    assert restored.entries[a.issue_id].issue.problem == "a"
    with pytest.raises(ValueError):
        IssueLedger.from_dict({"schema": "issue-ledger-v0", "entries": []})
    assert ledger.to_dict()["schema"] == ISSUE_LEDGER_SCHEMA


# ---------------------------------------------------------------------------
# RepairRegistry 路由表
# ---------------------------------------------------------------------------

def test_route_table_covers_every_kind():
    """每 issue kind（除 review_unavailable）必须有确定路由；表可被遍历。"""
    for kind in ISSUE_KINDS:
        entries = [(k, s) for (k, s) in REPAIR_ROUTES if k == kind]
        assert entries, f"{kind} 无任何路由条目"
        for _k, scope in entries:
            issue = _issue(kind=kind,
                           scope=scope if scope != "*" else "plan")
            ops = route(issue)
            if kind == "review_unavailable":
                assert ops is None  # 无 op：走标注交付，不属于程序错误
            else:
                assert ops, f"{kind}@{scope} 路由为空"
                assert all(op in UNIFIED_REPAIR_OPS for op in ops)


def test_route_structural_duplication_three_way_split():
    """结构重复三分流的三条路径都在表里：plan 债务侧 / body 正文侧 / 两 debt 实同。"""
    plan_ops = route(_issue(kind="structural_duplication", scope="plan"))
    assert plan_ops == ("canonicalize_debt", "reassign_debt_owner", "replace_plan_fragment")
    body_ops = route(_issue(kind="structural_duplication", scope="body"))
    assert body_ops == ("delete_blocks", "move_blocks")
    assert route(_issue(kind="debt_duplicated", scope="plan")) == (
        "canonicalize_debt", "replace_debt_fragment")


def test_route_key_kinds():
    for kind in ("fact_error", "number_inconsistency", "strong_causality", "scope_violation"):
        assert route(_issue(kind=kind, scope="body")) == ("block_patch",)
    assert route(_issue(kind="fact_error", scope="graph")) == (
        "rebind_source_spans", "reextract_graph_fragment")
    for kind in ("contract_gap", "debt_uncovered", "boundary_fail"):
        assert route(_issue(kind=kind, scope="plan")) == ("replace_plan_fragment",)
    assert route(_issue(kind="preparation_illegal", scope="plan")) == (
        "reassign_preparation", "remove_preparation")
    assert route(_issue(kind="figure_misread", scope="asset")) == ("redo_figure",)
    assert route(_issue(kind="format_gap", scope="body")) == ("mechanical_fix",)


def test_route_debt_kinds_to_debt_ops():
    assert route(_issue(kind="duplicate_debt", scope="plan")) == ("canonicalize_debt",)
    for kind in ("overbroad_debt", "fragmented_debt", "wrong_granularity"):
        assert route(_issue(kind=kind, scope="plan")) == ("replace_debt_fragment",)
    assert route(_issue(kind="missing_debt", scope="plan")) == ("add_debt_from_graph",)
    assert route(_issue(kind="unfounded_debt", scope="plan")) == (
        "rebind_debt_evidence", "retire_debt")
    assert route(_issue(kind="wrong_debt_kind", scope="plan")) == ("update_debt_relation",)
    assert route(_issue(kind="wrong_owner", scope="plan")) == ("reassign_debt_owner",)


def test_route_missing_is_program_error():
    with pytest.raises(IssueRouteError):
        route(_issue(kind="preparation_illegal", scope="body"))  # 该 kind 无 body 路由
    with pytest.raises(IssueRouteError):
        route(_issue(kind="boundary_fail", scope="asset"))


# ---------------------------------------------------------------------------
# validate_review_outcome / check_action_coverage / check_action_evidence
# ---------------------------------------------------------------------------

def _action(op="update_unit_fields", resolves=("i1",), targets=("u1",), params=None):
    if params is None:
        params = {"unit_id": "u1", "fields": {}}
    return RepairAction(op=op, resolves_issue_ids=list(resolves),
                        target_ids=list(targets), params=params)


def test_validate_clean_requires_all_pass_no_issues_no_actions():
    assert validate_review_outcome("clean", [{"item": "x", "outcome": "pass"}], [], []) == []
    # fail/unknown check 不得 clean
    assert validate_review_outcome("clean", [{"item": "x", "outcome": "fail"}], [], [])
    assert validate_review_outcome("clean", [{"item": "x", "outcome": "unknown"}], [], [])
    # clean 不得携带 issues / actions
    assert validate_review_outcome("clean", [], [_issue()], [])
    assert validate_review_outcome("clean", [], [], [_action()])
    # 旧式 checks（无 outcome 键，result 举证）不由本函数评判
    assert validate_review_outcome(
        "clean", [{"item": "断链", "result": "成立", "evidence": "原文"}], [], []) == []


def test_validate_repairable_needs_action_per_issue():
    issue = _issue()
    gaps = validate_review_outcome("repairable", [], [issue], [])
    assert any("无修复动作" in g or "未给出任何修复动作" in g for g in gaps)
    action = _action(resolves=(issue.issue_id,))
    assert validate_review_outcome("repairable", [], [issue], [action]) == []


def test_validate_repairable_rejects_unknown_issue_reference():
    issue = _issue()
    action = _action(resolves=("issue-幽灵",))
    gaps = validate_review_outcome("repairable", [], [issue], [action])
    assert any("引用不存在的 issue" in g for g in gaps)


def test_validate_repairable_rejects_bad_targets():
    issue = _issue()
    # target_ids 为空
    action = RepairAction(op="update_unit_fields", resolves_issue_ids=[issue.issue_id],
                          target_ids=[], params={"unit_id": "u1", "fields": {}})
    gaps = validate_review_outcome("repairable", [], [issue], [action])
    assert any("target_ids 为空" in g for g in gaps)
    # op 参数与 target_ids 不一致
    action = RepairAction(op="update_unit_fields", resolves_issue_ids=[issue.issue_id],
                          target_ids=["u1"], params={"unit_id": "u2", "fields": {}})
    gaps = validate_review_outcome("repairable", [], [issue], [action])
    assert any("不一致" in g for g in gaps)
    # 映射表参数缺失
    action = RepairAction(op="update_unit_fields", resolves_issue_ids=[issue.issue_id],
                          target_ids=["u1"], params={"fields": {}})
    gaps = validate_review_outcome("repairable", [], [issue], [action])
    assert any("缺少参数" in g for g in gaps)


def test_validate_accepts_raw_dict_actions_and_flags_illegal():
    issue = _issue()
    raw = {"resolves_issue_ids": [issue.issue_id], "op": "update_unit_fields",
           "target_ids": ["u1"], "params": {"unit_id": "u1", "fields": {}}}
    assert validate_review_outcome("repairable", [], [issue], [raw]) == []
    gaps = validate_review_outcome("repairable", [], [issue], [{"target_ids": ["u1"]}])
    assert any("协议非法" in g for g in gaps)


def test_action_coverage_touched_equals_target_ids():
    action = _action(targets=("u1", "u2"))
    assert check_action_coverage([action], ["u2", "u1"]) == []
    # touched 多出：执行了未声明的修改
    assert any("未声明" in g for g in check_action_coverage([action], ["u1", "u2", "u3"]))
    # target 多出：声明未生效
    assert any("未生效" in g for g in check_action_coverage([action], ["u1"]))
    # 经 validate_review_outcome 的 touched_ids 参数触发（模拟 Engine 执行后比对）
    issue = _issue()
    action = _action(resolves=(issue.issue_id,))
    assert validate_review_outcome(
        "repairable", [], [issue], [action], touched_ids=["u1"]) == []
    assert validate_review_outcome(
        "repairable", [], [issue], [action], touched_ids=["u9"])


def _graph_with_node():
    graph = SemanticGraph()
    graph.add_source_text("原文片段一：方法与实验。原文片段二：结果与边界。")
    node = graph.add_node("concept", "合法性校验概念")
    return graph, node


def test_check_action_evidence_real_and_fabricated():
    graph, node = _graph_with_node()
    span_id = next(iter(graph.spans))
    real = RepairAction(op="reassign_preparation", resolves_issue_ids=["i1"],
                        target_ids=["s1", "s2"],
                        params={"concept_id": node.node_id,
                                "from_section_id": "s1", "to_section_id": "s2"})
    assert check_action_evidence(real, graph) == []
    # 编造节点：evidence_node_ids 与 params 两处都检出
    fake = RepairAction(op="reassign_preparation", resolves_issue_ids=["i1"],
                        target_ids=["s1", "s2"],
                        evidence_node_ids=["concept-幽灵节点"],
                        params={"concept_id": "concept-幽灵节点"})
    gaps = check_action_evidence(fake, graph)
    assert any("concept-幽灵节点" in g for g in gaps)
    # creates[] 内一层嵌套也校验（债务拆/合与补全）
    nested = RepairAction(op="add_debt_from_graph", resolves_issue_ids=["i1"],
                          target_ids=["d1"],
                          params={"creates": [{"subject_node_id": node.node_id,
                                               "evidence_span_ids": [span_id]}]})
    assert check_action_evidence(nested, graph) == []
    nested_bad = RepairAction(op="add_debt_from_graph", resolves_issue_ids=["i1"],
                              target_ids=["d1"],
                              params={"creates": [{"subject_node_id": "assertion-编造",
                                                   "evidence_span_ids": ["span-编造"]}]})
    gaps = check_action_evidence(nested_bad, graph)
    assert any("assertion-编造" in g for g in gaps)
    assert any("span-编造" in g for g in gaps)
    # 无 node/span 参数的 op（机械修复）天然无 gap
    mech = RepairAction(op="mechanical_fix", resolves_issue_ids=["i1"],
                        target_ids=["b1"], params={})
    assert check_action_evidence(mech, graph) == []


def test_review_issue_roundtrip_via_dict():
    issue = ReviewIssue.from_legacy(
        {"severity": "中", "location": "层B", "kind": "fact_error",
         "problem": "数字错", "debt_ids": ["debt-x"], "node_ids": ["concept-y"]},
        "final")
    restored = ReviewIssue.from_dict(json.loads(json.dumps(issue.to_dict())))
    assert restored == issue


# ---------------------------------------------------------------------------
# E2E：run() 接线（issue-ledger 落盘 + 全部终态 + metadata 透出）
# ---------------------------------------------------------------------------

def _load_ledger(workspace):
    raw = workspace.get_json("issue-ledger")
    assert isinstance(raw, dict) and raw.get("schema") == ISSUE_LEDGER_SCHEMA
    return IssueLedger.from_dict(raw)


def test_e2e_issue_ledger_persisted_and_all_terminal(tmp_path):
    from test_reading import FakeKimiReading, _PAPER

    workspace = ReadingWorkspace(tmp_path)
    result = ReadingEngine(FakeKimiReading()).run(
        _PAPER, [], arxiv_id="2603.03329", progress=lambda _: None,
        workspace=workspace)
    assert result.body_md
    summary = result.metadata["issue_ledger"]
    assert summary["unterminated"] == 0
    ledger = _load_ledger(workspace)
    assert ledger.all_terminal()  # 禁静默丢失：register 过的 issue 全部终态
    assert ledger.summary() == summary  # metadata 透出与落盘一致


def test_e2e_plan_repairable_issue_resolved_in_ledger(tmp_path):
    from test_reading import FakeKimiReading, _PAPER, _first_layer_id, _plan_review_payload

    class RepairKimi(FakeKimiReading):
        def chat(self, messages, *, heavy=False, json_mode=False, low_reasoning=False,
                 max_tokens=None):
            text = str(messages[-1]["content"])
            if "任务：Plan 审核" in text:
                n = len([c for c in self.calls if c[0] == "plan-review"])
                self.calls.append(("plan-review", heavy))
                if n == 0:
                    return _plan_review_payload("repairable", repairs=[{
                        "issue_id": "i1", "op": "update_layer_mode",
                        "target_ids": [], "evidence_node_ids": [],
                        "expected_plan_revision": 1, "expected_revisions": {},
                        "params": {"layer_id": _first_layer_id(tmp_path),
                                   "mode": "comparison"},
                    }], issues=[{"kind": "mode", "location": "层A",
                                 "problem": "该层应按对比讲"}], text=text)
                return _plan_review_payload("clean", text=text)
            return super().chat(messages, heavy=heavy, json_mode=json_mode,
                                low_reasoning=low_reasoning, max_tokens=max_tokens)

    workspace = ReadingWorkspace(tmp_path)
    result = ReadingEngine(RepairKimi()).run(
        _PAPER, [], progress=lambda _: None, workspace=workspace)
    assert result.body_md
    ledger = _load_ledger(workspace)
    plan_entries = [e for e in ledger.entries.values() if e.issue.stage == "plan"]
    assert plan_entries, "repairable 审核的 issue 必须入册"
    assert all(e.status == "resolved" for e in plan_entries)
    assert all(e.resolved_by == "plan-review:clean" for e in plan_entries)
    assert result.metadata["issue_ledger"]["unterminated"] == 0


def test_e2e_carried_minor_issue_degraded_with_reason(tmp_path):
    from test_reading import FakeKimiReading, _PAPER

    class MinorIssueKimi(FakeKimiReading):
        def chat(self, messages, *, heavy=False, json_mode=False, low_reasoning=False,
                 max_tokens=None):
            text = str(messages[-1]["content"])
            if "精读章节事实与教学合同审核" in text:
                self.calls.append(("local-review", heavy))
                return json.dumps({
                    "pass": True,
                    "issues": [{"severity": "中", "location": "本节",
                                "problem": "措辞可更简洁"}],
                    "continuity_update": {"defined_terms": [],
                                          "established_claims": [],
                                          "open_questions": []},
                }, ensure_ascii=False)
            return super().chat(messages, heavy=heavy, json_mode=json_mode,
                                low_reasoning=low_reasoning, max_tokens=max_tokens)

    workspace = ReadingWorkspace(tmp_path)
    result = ReadingEngine(MinorIssueKimi()).run(
        _PAPER, [], progress=lambda _: None, workspace=workspace)
    assert result.body_md
    ledger = _load_ledger(workspace)
    unit_entries = [e for e in ledger.entries.values() if e.issue.stage == "unit"]
    assert unit_entries, "局审携带意见必须入册"
    # 中低严重度意见按局审口径放行：degraded_with_reason 且带原因（非静默丢失）
    assert all(e.status == "degraded_with_reason" for e in unit_entries)
    assert all("放行" in e.degraded_reason for e in unit_entries)
    summary = result.metadata["issue_ledger"]
    assert summary["degraded"] >= 1
    assert summary["unterminated"] == 0


def test_e2e_final_review_unavailable_synthesized_and_degraded(tmp_path):
    from test_reading import FakeKimiReading, _PAPER

    class FinalDownKimi(FakeKimiReading):
        def chat(self, messages, *, heavy=False, json_mode=False, low_reasoning=False,
                 max_tokens=None):
            text = str(messages[-1]["content"])
            if "论文精读发布前终审" in text:
                self.calls.append(("final-review", heavy))
                raise KimiError("终审服务宕机")
            return super().chat(messages, heavy=heavy, json_mode=json_mode,
                                low_reasoning=low_reasoning, max_tokens=max_tokens)

    workspace = ReadingWorkspace(tmp_path)
    result = ReadingEngine(FinalDownKimi()).run(
        _PAPER, [], progress=lambda _: None, workspace=workspace)
    assert result.body_md  # 产出优先：终审不可用横幅交付，不判死
    ledger = _load_ledger(workspace)
    final_entries = [e for e in ledger.entries.values() if e.issue.stage == "final"]
    # 终审 unavailable 协议 issues 为空：程序合成一条 review_unavailable 入册
    assert len(final_entries) == 1
    assert final_entries[0].issue.kind == "review_unavailable"
    assert final_entries[0].status == "degraded_with_reason"
    assert "横幅交付" in final_entries[0].degraded_reason
    assert result.metadata["issue_ledger"]["unterminated"] == 0


def test_e2e_review_unavailable_marked_and_degraded(tmp_path):
    from test_reading import FakeKimiReading, _PAPER

    class ReviewerDownKimi(FakeKimiReading):
        def chat(self, messages, *, heavy=False, json_mode=False, low_reasoning=False,
                 max_tokens=None):
            text = str(messages[-1]["content"])
            if "精读章节事实与教学合同审核" in text:
                self.calls.append(("local-review", heavy))
                raise KimiError("审核服务宕机")
            return super().chat(messages, heavy=heavy, json_mode=json_mode,
                                low_reasoning=low_reasoning, max_tokens=max_tokens)

    workspace = ReadingWorkspace(tmp_path)
    result = ReadingEngine(ReviewerDownKimi()).run(
        _PAPER, [], progress=lambda _: None, workspace=workspace)
    assert result.body_md  # 产出优先：审核不可用不判死
    ledger = _load_ledger(workspace)
    unavailable = [e for e in ledger.entries.values()
                   if e.issue.kind == "review_unavailable"]
    assert unavailable, "review_unavailable 必须以对应 kind 入册"
    # 无修复路由（route 为 None）→ 走标注交付：degraded_with_reason
    assert all(e.status == "degraded_with_reason" for e in unavailable)
    assert all("标注交付" in e.degraded_reason for e in unavailable)
    assert route(unavailable[0].issue) is None
    assert result.metadata["issue_ledger"]["unterminated"] == 0
