"""ReadingPlan：唯一 canonical Plan 存储（plan.json）。

规则（评审定稿）：
- 稳定 ID 排除 name/title 等展示字段，也不直接用会被修复的 question/must_explain
  作永久身份：初次 ID 由 unit 类型 + 绑定的 concept/assertion IDs 确定，
  创建后永久保留；字段变化只递增 revision 并更新 content_digest。
- plan.json 承载 plan revision、单元 revision、稳定 ID、绑定关系和顺序数组；
  Planner 原始文本只作审计，不参与运行时修复或缓存命中。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .reading_graph import SemanticGraph, stable_id

logger = logging.getLogger(__name__)

PLAN_SCHEMA_VERSION = "reading-plan-v1"


class PlanCollisionError(Exception):
    """稳定 ID 碰撞：同 seed 产生同 ID，禁止覆盖/序号后缀，交由有界重规划。"""


def _norm(text: Any) -> str:
    return re.sub(r"[^\w一-鿿]+", "", str(text or "")).casefold()


def _bigram_sim(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    sa = {a[i:i + 2] for i in range(len(a) - 1)} or {a}
    sb = {b[i:i + 2] for i in range(len(b) - 1)} or {b}
    return len(sa & sb) / max(1, len(sa | sb))


@dataclass
class PlanUnit:
    unit_id: str
    kind: str                      # section | layer
    semantic_revision: int = 1     # 实质字段变化才递增
    display_revision: int = 1      # name/title 等展示字段变化递增
    semantic_digest: str = ""
    display_digest: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    concept_ids: list[str] = field(default_factory=list)
    assertion_ids: list[str] = field(default_factory=list)

    DISPLAY_FIELDS = ("title", "name")

    def compute_semantic_digest(self, semantic_view: dict[str, Any] | None = None) -> str:
        """实质字段哈希；semantic_view 传入顶层 preparations 派生的义务视图时使用。"""
        meaningful = {k: v for k, v in self.data.items()
                      if k not in self.DISPLAY_FIELDS}
        if semantic_view is not None:
            meaningful["_preparations_view"] = semantic_view
        raw = json.dumps(meaningful, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def compute_display_digest(self) -> str:
        raw = json.dumps({k: self.data.get(k) for k in self.DISPLAY_FIELDS},
                         ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def refresh_digests(self, semantic_view: dict[str, Any] | None = None) -> tuple[bool, bool]:
        """按 digest 变化递增 revision（同事务每种最多 +1）；ID 不动。
        返回 (semantic_changed, display_changed)。"""
        new_sem = self.compute_semantic_digest(semantic_view)
        new_disp = self.compute_display_digest()
        sem_changed = new_sem != self.semantic_digest
        disp_changed = new_disp != self.display_digest
        if sem_changed:
            self.semantic_digest = new_sem
            self.semantic_revision += 1
        if disp_changed:
            self.display_digest = new_disp
            self.display_revision += 1
        return sem_changed, disp_changed

    # 兼容旧引用
    @property
    def revision(self) -> int:
        return self.semantic_revision

    @property
    def content_digest(self) -> str:
        return self.semantic_digest


@dataclass
class ReadingPlan:
    plan_revision: int = 1
    order: dict[str, list[str]] = field(default_factory=lambda: {"sections": [], "layers": []})
    units: dict[str, PlanUnit] = field(default_factory=dict)
    prompt_digest: str = ""        # 骨架/审核提示词摘要（semantic contract 的一部分）
    schema_digest: str = PLAN_SCHEMA_VERSION
    meta: dict[str, Any] = field(default_factory=dict)   # deep_dive_title/paper_type/running_example/metadata 等
    preparations: list[dict[str, Any]] = field(default_factory=list)  # 唯一事实源
    tombstones: dict[str, str] = field(default_factory=dict)          # 被替换 ID → alias

    # ---- 构建 ----

    @classmethod
    def from_skeleton(cls, skeleton: Any, graph: SemanticGraph,
                      prompt_digest: str,
                      meta: dict[str, Any] | None = None) -> "ReadingPlan":
        plan = cls(prompt_digest=prompt_digest, meta=dict(meta or {}))
        plan.meta.setdefault("deep_dive_title", skeleton.deep_dive_title)
        plan.meta.setdefault("paper_type", skeleton.paper_type)
        plan.meta.setdefault("running_example", skeleton.running_example)
        plan.meta.setdefault("needs_running_example", skeleton.needs_running_example)
        plan.meta.setdefault("top3", skeleton.top3)
        assertions = [n for n in graph.nodes.values() if n.kind == "assertion"]
        concepts = [n for n in graph.nodes.values() if n.kind == "concept"]

        for section in skeleton.sections:
            concept_ids = _bind_concepts(section.get("preparations") or [], concepts)
            role = _norm(section.get("purpose")) + "|" + _norm(section.get("purpose_reason"))
            seed = "|".join([role, *sorted(concept_ids)])
            unit_id = stable_id("section", seed)
            if unit_id in plan.units:
                raise PlanCollisionError(
                    f"section「{section.get('title')}」与已有单元同 ID：{unit_id}")
            data = dict(section)
            data.pop("preparations", None)  # 顶层 preparations 为唯一事实源
            unit = PlanUnit(
                unit_id=unit_id,
                kind="section",
                data=data,
                concept_ids=concept_ids,
            )
            unit.semantic_digest = unit.compute_semantic_digest()
            unit.display_digest = unit.compute_display_digest()
            plan.units[unit.unit_id] = unit
            plan.order["sections"].append(unit.unit_id)
            for prep in section.get("preparations") or []:
                if not isinstance(prep, dict) or not str(prep.get("concept") or "").strip():
                    continue
                bound = _bind_concepts([prep], concepts)
                plan.preparations.append({
                    "concept": str(prep.get("concept") or "").strip(),
                    "for_layer": str(prep.get("for_layer") or "").strip(),
                    "concept_id": bound[0] if bound else "",
                    "owner_section_id": unit.unit_id,
                    "target_layer_ids": [],  # 层建完后统一解析
                })

        for layer in skeleton.layers:
            assertion_ids = _bind_assertions(layer, assertions)
            role = _norm(layer.why) + "|" + _norm(layer.question)
            seed = "|".join([role, *sorted(assertion_ids)])
            unit_id = stable_id("layer", seed)
            if unit_id in plan.units:
                raise PlanCollisionError(
                    f"layer「{layer.name}」与已有单元同 ID：{unit_id}")
            unit = PlanUnit(
                unit_id=unit_id,
                kind="layer",
                data={
                    "name": layer.name, "why": layer.why, "question": layer.question,
                    "prerequisite": layer.prerequisite, "input": layer.input,
                    "output": layer.output, "teaching_mode": layer.teaching_mode,
                    "mode_reason": layer.mode_reason,
                    "break_if_merged": layer.break_if_merged,
                    "depth": layer.depth, "must_explain": layer.must_explain,
                    "success_test": layer.success_test,
                    "source_hints": layer.source_hints,
                    "source_refs": layer.source_refs,
                    "formula_hints": layer.formula_hints,
                },
                assertion_ids=assertion_ids,
            )
            unit.semantic_digest = unit.compute_semantic_digest()
            unit.display_digest = unit.compute_display_digest()
            plan.units[unit.unit_id] = unit
            plan.order["layers"].append(unit.unit_id)
        # for_layer 文本 → 真实层 unit_id（归一匹配；匹配失败记入 meta issues）
        name_to_uid = {plan.units[uid].data.get("name"): uid
                       for uid in plan.order["layers"]}
        unresolved: list[str] = []
        for prep in plan.preparations:
            target = str(prep.get("for_layer") or "").strip()
            uid = name_to_uid.get(target)
            if uid is None and target:
                norm = _norm(target)
                uid = next((u for n, u in name_to_uid.items()
                            if _norm(n) == norm), None)
            if uid:
                prep["target_layer_ids"] = [uid]
            elif target:
                unresolved.append(target)
        if unresolved:
            plan.meta.setdefault("issues", []).append(
                "preparations 目标层未解析：" + "、".join(unresolved))
        return plan

    # ---- 运行时还原（Skeleton 形状不变，供现有 run() 使用）----

    def to_skeleton(self) -> Any:
        from .reading import KeyLayer, Skeleton  # 延迟导入防环
        uid_to_layer_name = {uid: self.units[uid].data.get("name", "")
                             for uid in self.order.get("layers", []) if uid in self.units}
        sections = []
        for uid in self.order["sections"]:
            data = dict(self.units[uid].data)
            data["preparations"] = [
                {"for_layer": (uid_to_layer_name.get(p["target_layer_ids"][0], "")
                              if p.get("target_layer_ids") else p.get("for_layer", "")),
                 "concept": p.get("concept", "")}
                for p in self.preparations
                if p.get("owner_section_id") == uid
            ]
            sections.append(data)
        layers = []
        for uid in self.order["layers"]:
            d = self.units[uid].data
            layers.append(KeyLayer(
                name=d["name"], why=d.get("why", ""), question=d.get("question", ""),
                prerequisite=d.get("prerequisite", ""), input=d.get("input", ""),
                output=d.get("output", ""), source_hints=list(d.get("source_hints") or []),
                formula_hints=list(d.get("formula_hints") or []),
                teaching_mode=d.get("teaching_mode", "process"),
                mode_reason=d.get("mode_reason", ""),
                break_if_merged=dict(d.get("break_if_merged") or {}),
                depth=d.get("depth", "core"),
                must_explain=list(d.get("must_explain") or []),
                success_test=list(d.get("success_test") or []),
                source_refs=list(d.get("source_refs") or []),
            ))
        metadata = dict(self.meta.get("metadata") or {})
        return Skeleton(
            "",
            layers,
            str(self.meta.get("top3") or ""),
            running_example=str(self.meta.get("running_example") or ""),
            paper_type=str(self.meta.get("paper_type") or "method"),
            needs_running_example=bool(self.meta.get("needs_running_example")),
            sections=sections,
            deep_dive_title=str(self.meta.get("deep_dive_title") or "核心内容精讲"),
        ), metadata

    # ---- 持久化 ----

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema_digest,
            "plan_revision": self.plan_revision,
            "prompt_digest": self.prompt_digest,
            "order": json.loads(json.dumps(self.order)),
            "meta": json.loads(json.dumps(self.meta)),
            "preparations": json.loads(json.dumps(self.preparations)),
            "tombstones": json.loads(json.dumps(self.tombstones)),
            "units": {uid: asdict(u) for uid, u in self.units.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReadingPlan":
        if str(data.get("schema")) != PLAN_SCHEMA_VERSION:
            raise ValueError(f"plan schema 不兼容：{data.get('schema')}")
        plan = cls(plan_revision=int(data.get("plan_revision") or 1),
                   prompt_digest=str(data.get("prompt_digest") or ""),
                   meta=dict(data.get("meta") or {}),
                   preparations=list(data.get("preparations") or []),
                   tombstones=dict(data.get("tombstones") or {}))
        plan.order = {k: list(v) for k, v in (data.get("order") or {}).items()}
        for uid, raw in (data.get("units") or {}).items():
            raw = dict(raw)
            if "revision" in raw and "semantic_revision" not in raw:
                raw["semantic_revision"] = raw.pop("revision")
            raw.pop("revision", None)
            raw.pop("content_digest", None)
            plan.units[uid] = PlanUnit(**raw)
        return plan


def _bind_concepts(preparations: list[dict], concepts: list[Any]) -> list[str]:
    """preparations 的 concept 文本 → 图中 concept 节点（归一匹配）。"""
    bound: list[str] = []
    by_norm = {_norm(n.statement): n.node_id for n in concepts}
    for prep in preparations:
        if not isinstance(prep, dict):
            continue
        key = _norm(prep.get("concept"))
        if not key:
            continue
        if key in by_norm:
            bound.append(by_norm[key])
            continue
        # 近似匹配：归一化 bigram 相似度 ≥ 0.6 取最优
        best = max(
            ((n, _bigram_sim(key, n.statement)) for n in concepts),
            key=lambda x: x[1], default=None,
        )
        if best and best[1] >= 0.6:
            bound.append(best[0].node_id)
    return list(dict.fromkeys(bound))


def _bind_assertions(layer: Any, assertions: list[Any]) -> list[str]:
    """layer 的 question+must_explain → 图中 assertion 节点（相似度阈值 0.5，至多 3 个）。"""
    text = " ".join([layer.question, *layer.must_explain])
    scored = sorted(
        ((n, _bigram_sim(text, n.statement)) for n in assertions),
        key=lambda x: x[1], reverse=True,
    )
    return [n.node_id for n, sim in scored[:3] if sim >= 0.5]


def validate_plan(plan: ReadingPlan, graph: SemanticGraph | None = None,
                  evidence_chunks: set[str] | None = None) -> list[str]:
    """Plan 提交前不变量：order 唯一/存在/无孤儿/kind 一致/绑定存在于图；
    source_refs 指向真实 chunk/span；preparations 绑定与归属身份合法。"""
    gaps: list[str] = []
    for bucket, kind in (("sections", "section"), ("layers", "layer")):
        ids = plan.order.get(bucket) or []
        if len(set(ids)) != len(ids):
            gaps.append(f"{bucket} 顺序数组 ID 重复（疑似碰撞）")
        for uid in ids:
            unit = plan.units.get(uid)
            if unit is None:
                gaps.append(f"{bucket} 引用了不存在的 unit：{uid}")
            elif unit.kind != kind:
                gaps.append(f"unit {uid} kind={unit.kind} 与 {bucket} 不一致")
    referenced = set(plan.order.get("sections") or []) | set(plan.order.get("layers") or [])
    orphans = set(plan.units) - referenced
    if orphans:
        gaps.append("存在未被 order 引用的孤儿单元：" + "、".join(sorted(orphans)))
    if graph is not None:
        for uid in referenced:
            unit = plan.units.get(uid)
            if unit is None:
                continue
            for nid in (*unit.concept_ids, *unit.assertion_ids):
                if nid not in graph.nodes:
                    gaps.append(f"unit {uid} 绑定了不存在的图节点：{nid}")
    # source_refs 真实性：修复通道可改写 refs，必须指向真实存在的 chunk/span
    valid_refs: set[str] = set()
    if graph is not None:
        valid_refs |= set(graph.spans)
    if evidence_chunks:
        valid_refs |= set(evidence_chunks)
    if valid_refs:
        for uid in referenced:
            unit = plan.units.get(uid)
            if unit is None:
                continue
            for ref in unit.data.get("source_refs") or []:
                ref_text = str(ref).strip()
                if ref_text and ref_text not in valid_refs:
                    gaps.append(f"unit {uid} source_refs 指向不存在的块/span：{ref_text}")
    # preparations 身份：空绑定记 gap（不静默存在）；owner 必须指向真实 section
    for prep in plan.preparations:
        label = str(prep.get("concept") or "")[:30] or "未命名概念"
        if not str(prep.get("concept_id") or "").strip():
            gaps.append(f"preparation「{label}」concept_id 空绑定")
        owner = str(prep.get("owner_section_id") or "")
        owner_unit = plan.units.get(owner)
        if owner_unit is None or owner_unit.kind != "section":
            gaps.append(f"preparation「{label}」owner_section_id 非法：{owner or '空'}")
    return gaps
