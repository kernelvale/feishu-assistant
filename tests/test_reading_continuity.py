"""ContinuityLedger 单元测试（V8 第 8 步）：真实依赖登记、装包规则、精准失效。"""

from __future__ import annotations

from feishu_assistant.modules.reading_continuity import (
    ContinuityLedger,
    entry_id_for,
    unit_probe_text,
)


def _update(terms=(), claims=(), questions=()):
    return {
        "defined_terms": list(terms),
        "established_claims": list(claims),
        "open_questions": list(questions),
    }


# ---- 登记：身份、归属、revision ----

def test_entry_id_stable_by_kind_and_statement():
    """措辞即身份：同 kind 同 statement 同 ID；换措辞/换 kind 必变。"""
    e1 = entry_id_for("term", "校验器是合法性守门员")
    e2 = entry_id_for("term", "校验器是合法性守门员")
    assert e1 == e2
    assert e1 != entry_id_for("term", "校验器是别的什么")
    assert e1 != entry_id_for("claim", "校验器是合法性守门员")


def test_register_unit_assigns_kind_and_owner():
    ledger = ContinuityLedger()
    changed = ledger.register_unit("u1", _update(
        terms=["校验器是合法性守门员"],
        claims=["先校验输入再校验输出"],
        questions=["域外泛化尚未验证"],
    ))
    assert changed == {}  # 首次登记无「变化」（消费者尚不存在）
    entries = ledger.entries()
    assert [e.kind for e in entries] == ["term", "claim", "state"]
    assert all(e.established_by == "u1" for e in entries)
    assert all(e.established_revision == 1 for e in entries)


def test_register_unit_replace_reports_dropped_and_recreated():
    """单元返工：撤下的条目报 0；措辞不变的条目 revision 不动、不报变化。"""
    ledger = ContinuityLedger()
    ledger.register_unit("u1", _update(terms=["术语甲是第一版", "术语乙保留"]))
    dropped_id = entry_id_for("term", "术语甲是第一版")
    kept_id = entry_id_for("term", "术语乙保留")
    changed = ledger.register_unit("u1", _update(terms=["术语乙保留", "术语丙新建立"]))
    assert changed == {dropped_id: 0}
    assert ledger.get(kept_id).established_revision == 1  # 措辞不变不惊动下游
    assert ledger.get(dropped_id) is None
    # 消失后重建（含易主）：revision +1 → 消费者 manifest 边失配
    changed2 = ledger.register_unit("u2", _update(terms=["术语甲是第一版"]))
    assert changed2 == {dropped_id: 2}
    assert ledger.get(dropped_id).established_by == "u2"
    assert ledger.get(dropped_id).established_revision == 2


def test_first_establisher_wins_for_identical_statement():
    """同身份条目首建者保留，后来者不夺主也不重复登记。"""
    ledger = ContinuityLedger()
    ledger.register_unit("u1", _update(terms=["同一术语"]))
    changed = ledger.register_unit("u2", _update(terms=["同一术语"]))
    assert changed == {}
    assert len(ledger.entries()) == 1
    assert ledger.entries()[0].established_by == "u1"


def test_register_unit_none_drops_all_old_entries():
    """审核降级/缺失：该单元旧条目全部撤下并报变化（不静默留旧账）。"""
    ledger = ContinuityLedger()
    ledger.register_unit("u1", _update(terms=["术语甲"]))
    changed = ledger.register_unit("u1", None)
    assert changed == {entry_id_for("term", "术语甲"): 0}
    assert ledger.entries() == []


def test_persistence_roundtrip_keeps_revision_and_retired():
    ledger = ContinuityLedger()
    ledger.register_unit("u1", _update(terms=["术语甲", "术语乙"]))
    ledger.register_unit("u1", _update(terms=["术语乙"]))  # 术语甲撤下（retired=1）
    restored = ContinuityLedger.from_dict(ledger.to_dict())
    assert [e.statement for e in restored.entries()] == ["术语乙"]
    # 重建术语甲：revision 从退休版本 +1（易主/重建记忆不丢）
    changed = restored.register_unit("u2", _update(terms=["术语甲"]))
    assert changed == {entry_id_for("term", "术语甲"): 2}


def test_from_dict_rejects_incompatible_schema():
    import pytest
    with pytest.raises(ValueError):
        ContinuityLedger.from_dict({"schema": "continuity-ledger-v0", "entries": []})


# ---- 装包：真实依赖全注入，无依赖不注入 ----

def test_pack_injects_only_probe_relevant_entries():
    ledger = ContinuityLedger()
    ledger.register_unit("u1", _update(
        terms=["校验器负责把非法动作挡在输出之前"],
        claims=["训练曲线在第三 epoch 收敛"],
    ))
    probe = "本层讲校验器怎样把非法动作挡在输出之前"  # 只命中术语条目
    injected, input_ids = ledger.pack(probe)
    assert injected["defined_terms"] == ["校验器负责把非法动作挡在输出之前"]
    assert injected["established_claims"] == []
    assert input_ids == [(entry_id_for("term", "校验器负责把非法动作挡在输出之前"), 1)]


def test_pack_empty_probe_injects_nothing():
    ledger = ContinuityLedger()
    ledger.register_unit("u1", _update(terms=["校验器"]))
    injected, input_ids = ledger.pack("")
    assert all(not v for v in injected.values())
    assert input_ids == []


def test_pack_budget_caps_total_chars():
    ledger = ContinuityLedger()
    ledger.register_unit("u1", _update(
        terms=["探针相关术语甲" + "长" * 50, "探针相关术语乙" + "长" * 50]))
    injected, input_ids = ledger.pack("探针相关术语", max_chars=80)
    assert len(input_ids) == 1  # 预算耗尽：早建立的优先，截断不判死
    assert injected["defined_terms"] == ["探针相关术语甲" + "长" * 50]


# ---- 探针：合同文本 + prerequisite/debt 闭包 + 绑定节点 ----

class _FakeNode:
    def __init__(self, statement):
        self.statement = statement


class _FakeDebt:
    def __init__(self, subject, objects=(), premises=()):
        self.subject_node_id = subject
        self.object_node_ids = list(objects)
        self.premise_node_ids = list(premises)


class _FakeUnit:
    def __init__(self, unit_id, data, concept_ids=(), assertion_ids=()):
        self.unit_id = unit_id
        self.data = data
        self.concept_ids = list(concept_ids)
        self.assertion_ids = list(assertion_ids)


class _FakeGraph:
    def __init__(self, nodes):
        self.nodes = nodes


class _FakeRegistry:
    def __init__(self, debts):
        self.debts = debts


class _FakePlan:
    def __init__(self, preparations=()):
        self.preparations = list(preparations)


def test_unit_probe_includes_debt_closure_and_prerequisite_nodes():
    graph = _FakeGraph({
        "n1": _FakeNode("前提概念：动作空间约束"),
        "n2": _FakeNode("核心断言：校验器挡住非法动作"),
        "n3": _FakeNode("证据节点：消融实验"),
        "n4": _FakeNode("前提节点：策略梯度"),
    })
    registry = _FakeRegistry({
        "d1": _FakeDebt("n2", objects=["n3"], premises=["n4"]),
    })
    unit = _FakeUnit(
        "u1",
        {"name": "层A", "question": "校验器怎样工作？",
         "prerequisite_concept_ids": ["n1"], "debt_ids": ["d1"]},
        concept_ids=["n1"],
    )
    probe = unit_probe_text(unit, plan=_FakePlan(), registry=registry, graph=graph)
    for expected in ("校验器怎样工作？", "前提概念：动作空间约束",
                     "核心断言：校验器挡住非法动作", "证据节点：消融实验",
                     "前提节点：策略梯度"):
        assert expected in probe


def test_unit_probe_includes_preparations_text():
    unit = _FakeUnit("sec-1", {"title": "导论"})
    plan = _FakePlan(preparations=[
        {"concept": "校验器", "owner_section_id": "sec-1", "target_layer_ids": []},
        {"concept": "无关概念", "owner_section_id": "sec-2", "target_layer_ids": []},
    ])
    probe = unit_probe_text(unit, plan=plan)
    assert "校验器" in probe and "无关概念" not in probe
