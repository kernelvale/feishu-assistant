"""RepairEngine：结构化修复动作的事务化执行（v1）。

边界（评审定稿）：
- 只支持四种结构操作：reassign_preparation / remove_preparation /
  update_layer_mode / update_unit_fields；merge 类返回 unsupported_repair，
  由 needs_local_replan 通道处理（不静默删除重建）。
- 逐目标版本校验：expected_plan_revision + expected_revisions 全部匹配才执行；
  任一冲突，Plan 字节不变，只重试 Reviewer。
- revision 不硬编码：整批执行后按前后 digest 变化递增；同事务每种最多 +1；
  plan_revision 确有变化只 +1。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .reading_plan import ReadingPlan, validate_plan
from .reading_graph import SemanticGraph

logger = logging.getLogger(__name__)

SUPPORTED_OPS = frozenset({
    "reassign_preparation", "remove_preparation",
    "update_layer_mode", "update_unit_fields",
})
MERGE_OPS = frozenset({"merge_layers", "merge_sections", "split_layer"})

# update_unit_fields 白名单（按 kind 分；unit_id/revision/kind/order/绑定禁改）
FIELD_WHITELIST: dict[str, set[str]] = {
    "section": {"purpose", "title", "purpose_reason", "source_refs"},
    "layer": {"name", "why", "question", "prerequisite", "input", "output",
              "teaching_mode", "depth", "must_explain", "success_test",
              "source_hints", "source_refs", "formula_hints"},
}
FORBIDDEN_FIELDS = frozenset({
    "unit_id", "revision", "semantic_revision", "display_revision",
    "kind", "order", "concept_ids", "assertion_ids",
})


class RepairRejected(Exception):
    """动作非法/版本冲突/证据缺失：只重试 Reviewer，不动 Plan。"""


@dataclass
class RepairAction:
    issue_id: str
    op: str
    target_ids: list[str] = field(default_factory=list)
    evidence_node_ids: list[str] = field(default_factory=list)
    expected_plan_revision: int = 0
    expected_revisions: dict[str, int] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RepairAction":
        if not isinstance(data, dict):
            raise RepairRejected("RepairAction 必须是对象")
        op = str(data.get("op") or "").strip()
        if op in MERGE_OPS:
            raise RepairRejected(f"unsupported_repair:{op}")
        if op not in SUPPORTED_OPS:
            raise RepairRejected(f"非法操作：{op or '缺失'}")
        return cls(
            issue_id=str(data.get("issue_id") or "").strip(),
            op=op,
            target_ids=[str(v) for v in (data.get("target_ids") or [])],
            evidence_node_ids=[str(v) for v in (data.get("evidence_node_ids") or [])],
            expected_plan_revision=int(data.get("expected_plan_revision") or 0),
            expected_revisions={str(k): int(v) for k, v in
                                (data.get("expected_revisions") or {}).items()},
            params=dict(data.get("params") or {}),
        )


@dataclass
class RepairStats:
    applied: int = 0
    targets: list[str] = field(default_factory=list)
    changed_units: list[str] = field(default_factory=list)


class RepairEngine:
    """克隆 → 校验 → 副本执行 → digest 重算 → 不变量 → 由调用方原子提交。"""

    def __init__(self, graph: SemanticGraph | None = None,
                 evidence_chunks: set[str] | None = None) -> None:
        self.graph = graph
        self.evidence_chunks = evidence_chunks

    def apply(self, plan: ReadingPlan,
              actions: list[RepairAction]) -> tuple[ReadingPlan, RepairStats]:
        if not actions:
            raise RepairRejected("空 repairs")
        work = ReadingPlan.from_dict(plan.to_dict())  # 克隆
        stats = RepairStats()
        touched: set[str] = set()

        for action in actions:
            self._validate(work, action)
            self._apply_one(work, action, touched)
            stats.applied += 1
            stats.targets.extend(action.target_ids)

        # 统一 revision 规则：按前后 digest 变化递增，不按操作名硬编码
        changed_units: list[str] = []
        for uid in touched:
            unit = work.units.get(uid)
            if unit is None:
                continue
            view = None
            if unit.kind == "section":
                view = {"preparations": [
                    p for p in work.preparations
                    if p.get("owner_section_id") == uid
                ]}
            sem_changed, disp_changed = unit.refresh_digests(view)
            if sem_changed or disp_changed:
                changed_units.append(uid)
        if changed_units:
            work.plan_revision += 1
        stats.changed_units = changed_units

        gaps = validate_plan(work, self.graph, evidence_chunks=self.evidence_chunks)
        if gaps:
            raise RepairRejected("修复后 Plan 不变量未通过：" + "；".join(gaps))
        return work, stats

    # ---- 校验 ----

    def _validate(self, plan: ReadingPlan, action: RepairAction) -> None:
        if action.expected_plan_revision != plan.plan_revision:
            raise RepairRejected(
                f"plan revision 冲突：期望 {action.expected_plan_revision}，"
                f"实际 {plan.plan_revision}")
        for uid, expected in action.expected_revisions.items():
            unit = plan.units.get(uid)
            if unit is None:
                raise RepairRejected(f"expected_revisions 指向不存在单元：{uid}")
            if unit.semantic_revision != expected:
                raise RepairRejected(
                    f"单元 {uid} revision 冲突：期望 {expected}，"
                    f"实际 {unit.semantic_revision}")
        if self.graph is not None:
            for nid in action.evidence_node_ids:
                if nid not in self.graph.nodes and nid not in self.graph.spans:
                    raise RepairRejected(f"证据节点不存在于语义图：{nid}")

    # ---- 操作执行 ----

    def _apply_one(self, plan: ReadingPlan, action: RepairAction,
                   touched: set[str]) -> None:
        params = action.params
        if action.op == "reassign_preparation":
            concept_id = str(params.get("concept_id") or "").strip()
            if not concept_id:
                raise RepairRejected("reassign 拒绝空 concept_id（防止误改全部未绑定 prep）")
            from_section = str(params.get("from_section_id") or "").strip()
            to_section = str(params.get("to_section_id") or "").strip()
            if to_section not in plan.units or plan.units[to_section].kind != "section":
                raise RepairRejected(f"reassign 目标节不存在：{to_section}")
            hit = None
            for prep in plan.preparations:
                if prep.get("concept_id") == concept_id:
                    hit = prep
                    break
            if hit is None:
                raise RepairRejected(f"reassign 概念不存在：{concept_id}")
            old_owner = str(hit.get("owner_section_id") or "")
            if not from_section or from_section != old_owner:
                raise RepairRejected(
                    f"reassign from_section_id 与当前归属不符："
                    f"{from_section or '空'} ≠ {old_owner or '空'}")
            hit["owner_section_id"] = to_section
            touched.update(u for u in (old_owner, to_section) if u)

        elif action.op == "remove_preparation":
            concept_id = str(params.get("concept_id") or "").strip()
            if not concept_id:
                raise RepairRejected("remove 拒绝空 concept_id（防止误删全部未绑定 prep）")
            before = len(plan.preparations)
            removed = [p for p in plan.preparations
                       if p.get("concept_id") == concept_id]
            plan.preparations = [p for p in plan.preparations
                                 if p.get("concept_id") != concept_id]
            if len(plan.preparations) == before:
                raise RepairRejected(f"remove 概念不存在：{concept_id}")
            for prep in removed:
                owner = str(prep.get("owner_section_id") or "")
                if owner:
                    touched.add(owner)
                for layer_id in prep.get("target_layer_ids") or []:
                    if layer_id in plan.units:
                        touched.add(layer_id)

        elif action.op == "update_layer_mode":
            layer_id = str(params.get("layer_id") or "").strip()
            unit = plan.units.get(layer_id)
            if unit is None or unit.kind != "layer":
                raise RepairRejected(f"update_layer_mode 目标层不存在：{layer_id}")
            mode = str(params.get("mode") or "").strip()
            if mode not in {"process", "comparison", "derivation", "taxonomy", "evidence"}:
                raise RepairRejected(f"非法 teaching_mode：{mode}")
            unit.data["teaching_mode"] = mode
            if params.get("mode_reason"):
                unit.data["mode_reason"] = str(params["mode_reason"])
            touched.add(layer_id)

        elif action.op == "update_unit_fields":
            unit_id = str(params.get("unit_id") or "").strip()
            unit = plan.units.get(unit_id)
            if unit is None:
                raise RepairRejected(f"update_unit_fields 目标不存在：{unit_id}")
            fields = params.get("fields")
            if not isinstance(fields, dict) or not fields:
                raise RepairRejected("update_unit_fields 缺少 fields")
            allowed = FIELD_WHITELIST.get(unit.kind, set())
            for key, value in fields.items():
                if key in FORBIDDEN_FIELDS:
                    raise RepairRejected(f"禁止修改字段：{key}")
                if key not in allowed:
                    raise RepairRejected(f"{unit.kind} 不允许修改字段：{key}")
                unit.data[key] = value
            touched.add(unit_id)
