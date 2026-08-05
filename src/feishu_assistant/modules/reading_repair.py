"""RepairEngine：结构化修复动作的事务化执行（v1 四操作 + V8 第 7 步拓扑修复）。

边界（评审定稿）：
- 结构操作：reassign_preparation / remove_preparation /
  update_layer_mode / update_unit_fields + replace_plan_fragment
  （第 7 步：承担所有合并/拆分/重排拓扑修复）；裸 merge 类 op 仍返回
  unsupported_repair（防止无结构的合并），由审核器/调用方翻译成
  replace_plan_fragment 或走 needs_local_replan 通道（不静默删除重建）。
- 逐目标版本校验：expected_plan_revision + expected_revisions 全部匹配才执行；
  任一冲突，Plan 字节不变，只重试 Reviewer。
- revision 不硬编码：整批执行后按前后 digest 变化递增；同事务每种最多 +1；
  plan_revision 确有变化（digest 变化/新建/退休）只 +1，无实质变化不增。
- replace_plan_fragment 事务语义：克隆 → 校验（target 存在、creates 引用真实、
  retires 的 replaced_by 指向幸存单元）→ 应用（units/order/preparations 一致
  更新，退休单元写 tombstones）→ boundary 重算 → 债务 owner 重绑 →
  validate_plan 无 gaps → 调用方原子提交；任何一步失败 RepairRejected，
  克隆丢弃，原 Plan 字节不变。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .reading_plan import (
    PlanUnit,
    ReadingPlan,
    layer_unit_id,
    section_unit_id,
    validate_plan,
)
from .reading_graph import SemanticGraph

logger = logging.getLogger(__name__)

SUPPORTED_OPS = frozenset({
    "reassign_preparation", "remove_preparation",
    "update_layer_mode", "update_unit_fields",
    "replace_plan_fragment",
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
    created: list[str] = field(default_factory=list)   # fragment 新建单元
    retired: list[str] = field(default_factory=list)   # fragment 退休单元
    registry: Any | None = None   # 修复后债务 owner 重绑的注册表克隆（调用方落盘）


class RepairEngine:
    """克隆 → 校验 → 副本执行 → digest 重算 → 不变量 → 由调用方原子提交。

    registry（Canonical DebtRegistry，V8 第 5 步接线）随 validate_plan 参与
    修复后校验：共享债务/独占债务/subject 闭包缺口在修复提交前被拦下。"""

    def __init__(self, graph: SemanticGraph | None = None,
                 evidence_chunks: set[str] | None = None,
                 registry: Any | None = None) -> None:
        self.graph = graph
        self.evidence_chunks = evidence_chunks
        self.registry = registry

    def apply(self, plan: ReadingPlan,
              actions: list[RepairAction]) -> tuple[ReadingPlan, RepairStats]:
        if not actions:
            raise RepairRejected("空 repairs")
        work = ReadingPlan.from_dict(plan.to_dict())  # 克隆
        stats = RepairStats()
        touched: set[str] = set()

        for action in actions:
            self._validate(work, action)
            if action.op == "replace_plan_fragment":
                self._apply_fragment(work, action, touched, stats)
            else:
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
        # 拓扑变化（新建/退休）也是实质变化；digest 全未变且无拓扑变化不增 revision
        if changed_units or stats.created or stats.retired:
            work.plan_revision += 1
        stats.changed_units = changed_units

        # 债务 owner 重绑（registry 可用时）：修复后各层 debt_ids 写回 owner，
        # 重绑结果随 stats 交给调用方落盘；校验也面对重绑后的注册表
        final_registry = None
        if self.registry is not None:
            final_registry = work.bind_debt_owners(self.registry)
        gaps = validate_plan(work, self.graph, evidence_chunks=self.evidence_chunks,
                             registry=final_registry if final_registry is not None
                             else self.registry)
        if gaps:
            raise RepairRejected("修复后 Plan 不变量未通过：" + "；".join(gaps))
        stats.registry = final_registry
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
            self._update_fields(
                plan, str(params.get("unit_id") or "").strip(),
                params.get("fields"), touched)

    # ---- update_unit_fields 共用实现（fragment 的 updates 复用同一白名单）----

    def _update_fields(self, plan: ReadingPlan, unit_id: str, fields: Any,
                       touched: set[str]) -> None:
        unit = plan.units.get(unit_id)
        if unit is None:
            raise RepairRejected(f"update_unit_fields 目标不存在：{unit_id}")
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

    # ---- replace_plan_fragment（V8 第 7 步：承担所有拓扑修复）----

    def _apply_fragment(self, plan: ReadingPlan, action: RepairAction,
                        touched: set[str], stats: RepairStats) -> None:
        """Plan 片段替换：四形态统一事务（保留吸收/二合一/一分二/仅调依赖）。

        执行顺序：解析校验 → creates 入册 → updates（禁改将退休单元）→
        retires（债务/preparations/绑定按 replaced_by[0] 迁移 → tombstones →
        删除）→ order 重排（新建占据首个退休位置，无退休追加）→
        debt_changes/edge_changes（显式迁移，覆盖自动迁移结果）→
        preparation_changes → evidence_bindings → 派生值重算 → 新建单元
        初始 digest → boundary 重算。不变量由 apply() 收尾的 validate_plan
        统一把关，任何失败整体回滚（克隆丢弃，原 Plan 字节不变）。"""
        params = action.params
        target_ids = _str_list(params.get("target_ids"))
        keeps = _str_list(params.get("keeps"))
        updates = self._dict_list(params.get("updates"), "updates")
        creates = self._dict_list(params.get("creates"), "creates")
        retires = self._dict_list(params.get("retires"), "retires")
        edge_changes = self._dict_list(params.get("edge_changes"), "edge_changes")
        debt_changes = self._dict_list(params.get("debt_changes"), "debt_changes")
        prep_changes = self._dict_list(
            params.get("preparation_changes"), "preparation_changes")
        evidence_bindings = self._dict_list(
            params.get("evidence_bindings"), "evidence_bindings")
        for uid in target_ids:
            if uid not in plan.units:
                raise RepairRejected(f"replace_plan_fragment target 不存在：{uid}")
        for uid in keeps:
            if uid not in plan.units:
                raise RepairRejected(f"replace_plan_fragment keeps 单元不存在：{uid}")

        # ---- retires 预检：replaced_by 非空且指向 keeps/creates/未退休现存 ----
        retire_ids: list[str] = []
        replaced_map: dict[str, list[str]] = {}
        for entry in retires:
            uid = str(entry.get("unit_id") or "").strip()
            replaced_by = _str_list(entry.get("replaced_by"))
            if uid not in plan.units:
                raise RepairRejected(f"retires 单元不存在：{uid or '空'}")
            if not replaced_by:
                raise RepairRejected(f"retires 缺少 replaced_by：{uid}")
            retire_ids.append(uid)
            replaced_map[uid] = replaced_by
        if len(set(retire_ids)) != len(retire_ids):
            raise RepairRejected("retires 含重复单元")
        conflict = set(retire_ids) & set(keeps)
        if conflict:
            raise RepairRejected("retires 与 keeps 冲突：" + "、".join(sorted(conflict)))

        # ---- creates 预检并构建（kind/必填字段/引用真实 ID/稳定 ID 无冲突）----
        created_units = [self._build_created_unit(entry) for entry in creates]
        created_ids = [u.unit_id for u in created_units]
        if len(set(created_ids)) != len(created_ids):
            raise RepairRejected("creates 内部稳定 ID 重复（语义角色+绑定全同）")
        clash = [uid for uid in created_ids if uid in plan.units]
        if clash:
            raise RepairRejected("creates 与现存单元稳定 ID 冲突：" + "、".join(clash))
        # creates 别名：unit_id 由程序生成，调用方无法预知——事务内 replaced_by /
        # debt_changes / preparation_changes / evidence_bindings 可用创建单元的
        # name/title 精确引用（不是模糊匹配：只认本事务 creates 声明的展示名）
        aliases: dict[str, str] = {}
        for unit in created_units:
            for key in ("name", "title"):
                alias = str(unit.data.get(key) or "").strip()
                if alias and alias not in aliases and alias not in plan.units:
                    aliases[alias] = unit.unit_id

        def _resolve(uid: str) -> str:
            return aliases.get(uid, uid)

        survivors = (set(plan.units) - set(retire_ids)) | set(created_ids)
        resolved_map: dict[str, list[str]] = {}
        for uid, replaced_by in replaced_map.items():
            resolved = [_resolve(t) for t in replaced_by]
            bad = [t for t in resolved if t not in survivors]
            if bad:
                raise RepairRejected(
                    f"retires[{uid}] replaced_by 必须指向 keeps/creates/未退休单元："
                    + "、".join(bad))
            resolved_map[uid] = resolved

        # ---- 应用：creates 入册 → updates（禁改将退休单元）----
        for unit in created_units:
            plan.units[unit.unit_id] = unit
        for entry in updates:
            uid = _resolve(str(entry.get("unit_id") or "").strip())
            if uid in replaced_map:
                raise RepairRejected(f"updates 指向将被退休的单元：{uid}")
            self._update_fields(plan, uid, entry.get("fields"), touched)

        # ---- retires：迁移 → tombstones 接线（被替换 ID → 继承者）→ 删除 ----
        first_retire_pos: dict[str, int] = {}
        for bucket in ("sections", "layers"):
            order_list = plan.order.get(bucket) or []
            positions = [i for i, uid in enumerate(order_list) if uid in replaced_map]
            if positions:
                first_retire_pos[bucket] = min(positions)
        derived_dirty: set[str] = set()
        for uid in retire_ids:
            unit = plan.units[uid]
            heir_id = resolved_map[uid][0]
            heir = plan.units[heir_id]
            self._absorb_unit(plan, unit, heir)
            plan.tombstones[uid] = heir_id
            del plan.units[uid]
            touched.discard(uid)
            touched.add(heir_id)
            derived_dirty.add(heir_id)

        # ---- order 一致更新：退休删除，新建插入首个退休位置（无退休追加末尾）----
        for bucket in ("sections", "layers"):
            order_list = [u for u in (plan.order.get(bucket) or [])
                          if u in plan.units]
            new_ids = [u.unit_id for u in created_units if u.kind == bucket[:-1]]
            if new_ids:
                pos = min(first_retire_pos.get(bucket, len(order_list)),
                          len(order_list))
                order_list[pos:pos] = new_ids
            plan.order[bucket] = order_list

        # ---- debt_changes/edge_changes：显式债务迁移（最后胜出）----
        for entry in debt_changes:
            self._move_debt(
                plan, str(entry.get("debt_id") or "").strip(),
                _resolve(str(entry.get("owner_layer_id") or "").strip()),
                "debt_changes", touched, derived_dirty)
        for entry in edge_changes:
            self._move_debt(
                plan, str(entry.get("from_debt_id") or "").strip(),
                _resolve(str(entry.get("to_owner_layer_id") or "").strip()),
                "edge_changes", touched, derived_dirty)

        # ---- preparation_changes：改 owner/目标层（不动节点数）----
        for entry in prep_changes:
            concept_id = str(entry.get("concept_id") or "").strip()
            if not concept_id:
                raise RepairRejected("preparation_changes 缺少 concept_id")
            hit = next((p for p in plan.preparations
                        if p.get("concept_id") == concept_id), None)
            if hit is None:
                raise RepairRejected(f"preparation_changes 概念不存在：{concept_id}")
            old_owner = str(hit.get("owner_section_id") or "")
            new_owner = str(entry.get("owner_section_id") or "").strip()
            if new_owner:
                owner_unit = plan.units.get(new_owner)
                if owner_unit is None or owner_unit.kind != "section":
                    raise RepairRejected(
                        f"preparation_changes owner 必须是现存 section：{new_owner}")
                hit["owner_section_id"] = new_owner
            if "target_layer_ids" in entry:
                targets = [_resolve(t) for t in _str_list(entry.get("target_layer_ids"))]
                if not targets:
                    raise RepairRejected("preparation_changes target_layer_ids 为空")
                for tid in targets:
                    target_unit = plan.units.get(tid)
                    if target_unit is None:
                        raise RepairRejected(
                            f"preparation_changes target 不存在：{tid}")
                    if target_unit.kind != "layer":
                        raise RepairRejected(
                            f"preparation_changes target 必须是 layer：{tid}")
                hit["target_layer_ids"] = targets
            if old_owner:
                touched.add(old_owner)
            touched.add(str(hit.get("owner_section_id") or ""))

        # ---- evidence_bindings：重绑 source_span_ids（引用必须真实）----
        for entry in evidence_bindings:
            uid = _resolve(str(entry.get("unit_id") or "").strip())
            unit = plan.units.get(uid)
            if unit is None:
                raise RepairRejected(f"evidence_bindings 单元不存在：{uid or '空'}")
            span_ids = _str_list(entry.get("source_span_ids"))
            if self.graph is not None:
                unknown = [s for s in span_ids if s not in self.graph.spans]
                if unknown:
                    raise RepairRejected(
                        "evidence_bindings 指向不存在的 span：" + "、".join(unknown))
            unit.data["source_span_ids"] = span_ids
            touched.add(uid)

        # ---- 派生值重算：债务归属变化的层按 Registry 重派生（唯一事实源）----
        debts = getattr(self.registry, "debts", None) or {}
        if debts:
            for uid in derived_dirty:
                unit = plan.units.get(uid)
                if unit is None or unit.kind != "layer":
                    continue
                dids = [d for d in (unit.data.get("debt_ids") or []) if d in debts]
                unit.data["must_explain"] = [debts[d].learner_action for d in dids]
                unit.data["success_test"] = [debts[d].success_test for d in dids]

        # ---- 新建单元初始 digest（迁移全部落定后计算，revision 从 1 起）----
        for unit in created_units:
            unit.semantic_digest = unit.compute_semantic_digest()
            unit.display_digest = unit.compute_display_digest()
            touched.add(unit.unit_id)
        # 拓扑变化影响相邻层对：boundary 交集确定性重算，outcome 复位待复核
        plan.precompute_boundaries()
        stats.created.extend(created_ids)
        stats.retired.extend(retire_ids)

    def _build_created_unit(self, entry: dict[str, Any]) -> PlanUnit:
        """creates 项 → PlanUnit：必填字段与引用真实性校验 + 既有稳定 ID 规则。

        模型给的 unit_id/revision 一律丢弃：ID 由程序按「类型+语义角色+绑定
        ID」生成（与 from_skeleton 同一规则），revision 从 1 起。"""
        kind = str(entry.get("kind") or "").strip()
        if kind not in ("layer", "section"):
            raise RepairRejected(f"creates kind 非法：{kind or '空'}")
        data = dict(entry.get("data") or {})
        for key in ("unit_id", "revision", "semantic_revision", "display_revision",
                    "kind", "order", "concept_ids", "assertion_ids"):
            data.pop(key, None)
        graph = self.graph
        debts = getattr(self.registry, "debts", None) or {}
        if kind == "section":
            title = str(data.get("title") or "").strip()
            if not title:
                raise RepairRejected("creates section 缺少 title")
            purposes = _str_list(data.get("purposes"))
            if not purposes:
                single = str(data.get("purpose") or "").strip()
                purposes = [single] if single else []
            if not purposes:
                raise RepairRejected(f"creates section「{title}」缺少 purposes")
            data["purposes"] = purposes
            data["purpose"] = purposes[0]
            data.pop("preparations", None)  # 顶层 preparations 为唯一事实源
            return PlanUnit(
                unit_id=section_unit_id(purposes, data.get("purpose_reason")),
                kind="section", data=data)
        # kind == "layer"
        name = str(data.get("name") or "").strip()
        if not name:
            raise RepairRejected("creates layer 缺少 name")
        primary = str(entry.get("primary_assertion_id")
                      or data.get("primary_assertion_id") or "").strip()
        if not primary:
            raise RepairRejected(f"creates layer「{name}」缺少 primary_assertion_id")
        if graph is not None:
            node = graph.nodes.get(primary)
            if node is None:
                raise RepairRejected(
                    f"creates layer「{name}」primary assertion 不存在：{primary}")
            if node.kind != "assertion":
                raise RepairRejected(
                    f"creates layer「{name}」primary 指向非 assertion 节点：{primary}")
        supporting = _str_list(entry.get("supporting_node_ids")
                               or data.get("supporting_node_ids"))
        prereq = _str_list(entry.get("prerequisite_concept_ids")
                           or data.get("prerequisite_concept_ids"))
        span_ids = _str_list(entry.get("source_span_ids")
                             or data.get("source_span_ids"))
        if graph is not None:
            unknown_nodes = [n for n in (*supporting, *prereq)
                             if n not in graph.nodes]
            if unknown_nodes:
                raise RepairRejected(
                    f"creates layer「{name}」引用不存在的图节点："
                    + "、".join(unknown_nodes))
            unknown_spans = [s for s in span_ids if s not in graph.spans]
            if unknown_spans:
                raise RepairRejected(
                    f"creates layer「{name}」引用不存在的 span："
                    + "、".join(unknown_spans))
        debt_ids = _str_list(entry.get("debt_ids") or data.get("debt_ids"))
        if debts:
            unknown_debts = [d for d in debt_ids if d not in debts]
            if unknown_debts:
                raise RepairRejected(
                    f"creates layer「{name}」引用注册表外的债务："
                    + "、".join(unknown_debts))
        # must_explain/success_test 程序派生（Registry 唯一事实源；缺失兜底派生）
        if debt_ids and debts:
            data["must_explain"] = [debts[d].learner_action for d in debt_ids]
            data["success_test"] = [debts[d].success_test for d in debt_ids]
        else:
            if primary and graph is not None:
                fallback_explain = [graph.nodes[primary].statement]
            elif str(data.get("question") or "").strip():
                fallback_explain = [str(data["question"]).strip()]
            else:
                fallback_explain = ["论文的核心机制是什么"]
            data.setdefault("must_explain", fallback_explain)
            question = str(data.get("question") or "").strip()
            data.setdefault("success_test",
                            [f"能闭卷回答：{question}"] if question
                            else ["能复述论文的核心方法"])
        assertion_ids = list(dict.fromkeys(
            ([primary] if primary else [])
            + [n for n in supporting
               if graph is not None and graph.nodes[n].kind == "assertion"]))
        concept_ids = list(dict.fromkeys(
            n for n in (*supporting, *prereq)
            if graph is not None and graph.nodes[n].kind == "concept"))
        for key, default in (("why", ""), ("question", ""), ("prerequisite", ""),
                             ("input", ""), ("output", ""),
                             ("teaching_mode", "process"), ("mode_reason", ""),
                             ("break_if_merged", {}), ("depth", "core"),
                             ("source_hints", []), ("source_refs", []),
                             ("formula_hints", [])):
            data.setdefault(key, default)
        data["primary_assertion_id"] = primary
        data["debt_ids"] = debt_ids
        data["supporting_node_ids"] = supporting
        data["prerequisite_concept_ids"] = prereq
        data["source_span_ids"] = span_ids
        return PlanUnit(
            unit_id=layer_unit_id(data.get("why"), data.get("question"),
                                  assertion_ids),
            kind="layer", data=data,
            assertion_ids=assertion_ids, concept_ids=concept_ids)

    def _absorb_unit(self, plan: ReadingPlan, unit: PlanUnit, heir: PlanUnit) -> None:
        """退休单元 → 接收单元的自动迁移：债务并入、绑定并入、preparations 重定向。

        primary_assertion_id 不迁移（跨层唯一），退休单元的 primary 也不自动
        降为 supporting（继承者的合同边界由审核器经 creates/updates 显式声明）。"""
        if unit.kind == "layer" and heir.kind == "layer":
            heir_debts = list(heir.data.get("debt_ids") or [])
            for did in unit.data.get("debt_ids") or []:
                if did not in heir_debts:
                    heir_debts.append(did)
            heir.data["debt_ids"] = heir_debts
            for key in ("supporting_node_ids", "prerequisite_concept_ids",
                        "source_span_ids", "source_refs"):
                merged = list(heir.data.get(key) or [])
                for value in unit.data.get(key) or []:
                    if value not in merged:
                        merged.append(value)
                heir.data[key] = merged
            if self.graph is not None:
                primary = str(heir.data.get("primary_assertion_id") or "")
                supporting = list(heir.data.get("supporting_node_ids") or [])
                prereq = list(heir.data.get("prerequisite_concept_ids") or [])
                heir.assertion_ids = list(dict.fromkeys(
                    ([primary] if primary else [])
                    + [n for n in supporting
                       if self.graph.nodes.get(n) is not None
                       and self.graph.nodes[n].kind == "assertion"]))
                heir.concept_ids = list(dict.fromkeys(
                    n for n in (*supporting, *prereq)
                    if self.graph.nodes.get(n) is not None
                    and self.graph.nodes[n].kind == "concept"))
        for prep in plan.preparations:
            targets = [heir.unit_id if t == unit.unit_id else t
                       for t in (prep.get("target_layer_ids") or [])]
            prep["target_layer_ids"] = list(dict.fromkeys(targets))
            if prep.get("owner_section_id") == unit.unit_id:
                prep["owner_section_id"] = heir.unit_id

    def _move_debt(self, plan: ReadingPlan, debt_id: str, owner_id: str,
                   label: str, touched: set[str],
                   derived_dirty: set[str]) -> None:
        """显式债务迁移：从全部层移除后挂到目标层（唯一 owner）；原地指派为 no-op。"""
        if not debt_id:
            raise RepairRejected(f"{label} 缺少 debt_id")
        debts = getattr(self.registry, "debts", None) or {}
        if debts and debt_id not in debts:
            raise RepairRejected(f"{label} 债务不存在于注册表：{debt_id}")
        owner = plan.units.get(owner_id)
        if owner is None:
            raise RepairRejected(f"{label} 目标层不存在：{owner_id or '空'}")
        if owner.kind != "layer":
            raise RepairRejected(f"{label} 目标必须是 layer：{owner_id}")
        current = [uid for uid in plan.order.get("layers") or []
                   if uid in plan.units
                   and debt_id in (plan.units[uid].data.get("debt_ids") or [])]
        if current == [owner_id]:
            return  # 原地指派：无实质变化（no-op 判定依赖此短路）
        for uid in current:
            unit = plan.units[uid]
            unit.data["debt_ids"] = [d for d in (unit.data.get("debt_ids") or [])
                                     if d != debt_id]
            touched.add(uid)
            derived_dirty.add(uid)
        owner_debts = list(owner.data.get("debt_ids") or [])
        if debt_id not in owner_debts:
            owner_debts.append(debt_id)
        owner.data["debt_ids"] = owner_debts
        touched.add(owner_id)
        derived_dirty.add(owner_id)

    @staticmethod
    def _dict_list(value: Any, label: str) -> list[dict[str, Any]]:
        """fragment 子协议数组解析：缺省为 []，非数组/非对象项拒收。"""
        if value is None:
            return []
        if not isinstance(value, list):
            raise RepairRejected(f"{label} 必须是数组")
        for entry in value:
            if not isinstance(entry, dict):
                raise RepairRejected(f"{label} 项必须是对象")
        return value


def _str_list(value: Any) -> list[str]:
    return [str(v).strip() for v in value or [] if str(v).strip()]
