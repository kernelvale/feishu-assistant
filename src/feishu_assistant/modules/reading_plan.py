"""ReadingPlan：唯一 canonical Plan 存储（plan.json）。

规则（评审定稿）：
- 稳定 ID 排除 name/title 等展示字段，也不直接用会被修复的 question/must_explain
  作永久身份：初次 ID 由 unit 类型 + 绑定的 concept/assertion IDs 确定，
  创建后永久保留；字段变化只递增 revision 并更新 content_digest。
- plan.json 承载 plan revision、单元 revision、稳定 ID、绑定关系和顺序数组；
  Planner 原始文本只作审计，不参与运行时修复或缓存命中。

V8 第 3 步（PlanCompiler 接线）：
- 层绑定直接消费 Planner 直出的 graph-ID / debt-ID（primary_assertion_id、
  supporting_node_ids、prerequisite_concept_ids、debt_ids、source_span_ids），
  未知 ID 收集进 plan.meta["issues"]，不再做任何 bigram 模糊绑定。
- must_explain/success_test 由程序从 Canonical DebtRegistry 派生
  （learner_action/success_test 是唯一事实源）；注册表缺失/为空/degraded 时
  按 degraded 标注兜底派生（primary 主张 → question 兜底），绝不判死。
- boundary（相邻层对）由程序预计算：outcome=unknown 待审，lost_* 从债务/断言
  交集确定性预填；第 5 步 PlanAuditor 消费并逐项判定。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .reading_graph import SemanticGraph, stable_id

logger = logging.getLogger(__name__)

PLAN_SCHEMA_VERSION = "reading-plan-v2"

# Plan 准入状态：pending（待审）| clean（审核通过）| degraded（带批注继续）
PLAN_ADMISSION_STATUSES = ("pending", "clean", "degraded")
# boundary 审核结论：unknown（初始待审）| pass | fail（第 5 步 PlanAuditor 写入）
BOUNDARY_OUTCOMES = ("unknown", "pass", "fail")


class PlanCollisionError(Exception):
    """稳定 ID 碰撞：同 seed 产生同 ID，禁止覆盖/序号后缀，交由有界重规划。"""


def _norm(text: Any) -> str:
    return re.sub(r"[^\w一-鿿]+", "", str(text or "")).casefold()


def section_unit_id(purposes: list[str], purpose_reason: Any) -> str:
    """节稳定 ID（类型+语义角色）：与 from_skeleton 同一 seed 规则，单一事实源。"""
    seed = "|".join([*sorted(_norm(p) for p in purposes), _norm(purpose_reason)])
    return stable_id("section", seed)


def layer_unit_id(why: Any, question: Any, assertion_ids: Iterable[str]) -> str:
    """层稳定 ID（类型+语义角色+绑定 assertion）：与 from_skeleton 同一 seed 规则。"""
    role = _norm(why) + "|" + _norm(question)
    seed = "|".join([role, *sorted(assertion_ids)])
    return stable_id("layer", seed)


def graph_content_revision(graph: SemanticGraph) -> str:
    """图内容摘要：作为 plan.graph_revision 记录构建/审核时面对的图版本。
    只含身份性内容（节点/边/span ID 与陈述），与遍历顺序无关。"""
    material = json.dumps({
        "nodes": sorted((n.node_id, n.kind, n.statement)
                        for n in graph.nodes.values()),
        "edges": sorted((e.kind, e.from_id, e.to_id) for e in graph.edges),
        "spans": sorted(graph.spans),
    }, ensure_ascii=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


def depends_on_closure(graph: SemanticGraph, root_id: str) -> set[str]:
    """root 节点的 depends_on 传递闭包（不含 root 自身）；图上无环由图校验保证，
    这里仍用 visited 防御。"""
    closure: set[str] = set()
    stack = [root_id]
    while stack:
        current = stack.pop()
        for edge in graph.edges:
            if edge.kind == "depends_on" and edge.from_id == current \
                    and edge.to_id not in closure:
                closure.add(edge.to_id)
                stack.append(edge.to_id)
    return closure


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
    # ---- V8 第 3 步：准入与版本记录（第 5 步 PlanAuditor 消费）----
    admission_status: str = "pending"     # PLAN_ADMISSION_STATUSES 之一
    reviewed_revision: int = 0            # 最近一次完成审核时的 plan_revision（0=从未审核）
    graph_revision: str = ""              # 构建时的图内容摘要（graph_content_revision）
    debt_registry_revision: int = 0       # 构建时消费的债务注册表 revision（0=无注册表）
    boundaries: list[dict[str, Any]] = field(default_factory=list)  # 相邻层对边界（程序预计算）

    # ---- 构建 ----

    @classmethod
    def from_skeleton(cls, skeleton: Any, graph: SemanticGraph,
                      prompt_digest: str,
                      meta: dict[str, Any] | None = None,
                      registry: Any | None = None) -> "ReadingPlan":
        """骨架 → canonical Plan：直出 ID 绑定 + 债务派生 + boundary 预计算。

        registry 为 Canonical DebtRegistry（reading_debts.DebtRegistry）或 None；
        缺失/为空/degraded 时按 degraded 标注兜底派生，绝不判死。"""
        plan = cls(prompt_digest=prompt_digest, meta=dict(meta or {}))
        plan.meta.setdefault("deep_dive_title", skeleton.deep_dive_title)
        plan.meta.setdefault("paper_type", skeleton.paper_type)
        plan.meta.setdefault("running_example", skeleton.running_example)
        plan.meta.setdefault("needs_running_example", skeleton.needs_running_example)
        plan.meta.setdefault("top3", skeleton.top3)
        if getattr(skeleton, "coverage", None):
            plan.meta.setdefault("coverage", dict(skeleton.coverage))
        plan.graph_revision = graph_content_revision(graph)
        debts = getattr(registry, "debts", None) or {}
        registry_usable = bool(debts) and getattr(registry, "admission_status", "") != "degraded"
        if registry is not None:
            plan.debt_registry_revision = int(getattr(registry, "registry_revision", 0) or 0)
        issues: list[str] = plan.meta.setdefault("issues", [])
        if not registry_usable:
            # 产出优先：债务注册表不可用 → 层按无债务模式兜底派生（degraded 标注）
            issues.append(
                "债务注册表缺失/为空/degraded：层按无债务模式规划，"
                "must_explain/success_test 由程序兜底派生")

        def _unknown(kind_label: str, raw_id: str, owner: str) -> None:
            note = f"{owner} 引用不存在的{kind_label}：{raw_id}"
            if note not in issues:
                issues.append(note)

        for section in skeleton.sections:
            purposes = _section_purposes(section)
            # 稳定 ID seed：排序拼接 purposes + purpose_reason（第 3 步裁定）
            unit_id = section_unit_id(purposes, section.get("purpose_reason"))
            if unit_id in plan.units:
                raise PlanCollisionError(
                    f"section「{section.get('title')}」与已有单元同 ID：{unit_id}")
            concept_ids: list[str] = []
            prep_entries: list[dict[str, Any]] = []
            for prep in section.get("preparations") or []:
                if not isinstance(prep, dict):
                    continue
                concept_id = str(prep.get("concept_id") or "").strip()
                if concept_id:
                    if concept_id in graph.nodes:
                        concept_ids.append(concept_id)
                    else:
                        _unknown("概念节点", concept_id,
                                 f"章节「{section.get('title')}」preparations")
                        concept_id = ""
                concept_text = str(prep.get("concept") or "").strip()
                if not concept_text and concept_id in graph.nodes:
                    concept_text = graph.nodes[concept_id].statement
                if not concept_text:
                    continue
                prep_entries.append({
                    "concept": concept_text,
                    "for_layer": str(prep.get("for_layer") or "").strip(),
                    "concept_id": concept_id,
                    "owner_section_id": unit_id,
                    "target_layer_ids": [],  # 层建完后统一解析
                })
            data = dict(section)
            data.pop("preparations", None)  # 顶层 preparations 为唯一事实源
            data["purposes"] = purposes      # 全量保留；下游兼容取 purposes[0]
            data["purpose"] = purposes[0]
            unit = PlanUnit(
                unit_id=unit_id,
                kind="section",
                data=data,
                concept_ids=list(dict.fromkeys(concept_ids)),
            )
            unit.semantic_digest = unit.compute_semantic_digest()
            unit.display_digest = unit.compute_display_digest()
            plan.units[unit.unit_id] = unit
            plan.order["sections"].append(unit.unit_id)
            plan.preparations.extend(prep_entries)

        for layer in skeleton.layers:
            name = str(getattr(layer, "name", "") or "")
            # ---- 直出 ID 绑定（未知 ID 记 issues，不做 bigram 模糊绑定）----
            primary = str(getattr(layer, "primary_assertion_id", "") or "").strip()
            if primary and primary not in graph.nodes:
                _unknown("primary assertion", primary, f"层「{name}」")
                primary = ""
            supporting = [str(v).strip()
                          for v in (getattr(layer, "supporting_node_ids", None) or [])
                          if str(v).strip()]
            known_supporting: list[str] = []
            for nid in supporting:
                if nid in graph.nodes:
                    known_supporting.append(nid)
                else:
                    _unknown("图节点", nid, f"层「{name}」supporting_node_ids")
            prereq_concepts = [str(v).strip()
                               for v in (getattr(layer, "prerequisite_concept_ids", None) or [])
                               if str(v).strip()]
            known_prereq: list[str] = []
            for nid in prereq_concepts:
                if nid in graph.nodes:
                    known_prereq.append(nid)
                else:
                    _unknown("图节点", nid, f"层「{name}」prerequisite_concept_ids")
            span_ids = [str(v).strip()
                        for v in (getattr(layer, "source_span_ids", None) or [])
                        if str(v).strip()]
            known_spans: list[str] = []
            for sid in span_ids:
                if sid in graph.spans:
                    known_spans.append(sid)
                else:
                    _unknown("source span", sid, f"层「{name}」source_span_ids")
            assertion_ids = list(dict.fromkeys(
                ([primary] if primary else [])
                + [nid for nid in known_supporting
                   if graph.nodes[nid].kind == "assertion"]))
            concept_ids = list(dict.fromkeys(
                [nid for nid in (*known_supporting, *known_prereq)
                 if graph.nodes[nid].kind == "concept"]))
            # ---- 债务归属：只能引 Registry 真实债务；must_explain/success_test 程序派生 ----
            debt_ids: list[str] = []
            for did in (getattr(layer, "debt_ids", None) or []):
                did = str(did).strip()
                if not did:
                    continue
                if did in debts:
                    debt_ids.append(did)
                else:
                    _unknown("债务", did, f"层「{name}」debt_ids")
            debt_ids = list(dict.fromkeys(debt_ids))
            if debt_ids:
                must_explain = [debts[did].learner_action for did in debt_ids]
                success_test = [debts[did].success_test for did in debt_ids]
            elif list(getattr(layer, "must_explain", None) or []):
                # 兼容：代码构造的骨架（确定性降级 Plan）自带债务文本，原样保留
                must_explain = list(layer.must_explain)
                success_test = list(getattr(layer, "success_test", None) or [])
            else:
                # 兜底派生（注册表不可用）：本层核心主张/核心问题，绝不留空
                if primary:
                    must_explain = [graph.nodes[primary].statement]
                elif str(getattr(layer, "question", "") or "").strip():
                    must_explain = [str(layer.question).strip()]
                else:
                    must_explain = ["论文的核心机制是什么"]
                question = str(getattr(layer, "question", "") or "").strip()
                success_test = ([f"能闭卷回答：{question}"] if question
                                else ["能复述论文的核心方法"])
            unit_id = layer_unit_id(layer.why, layer.question, assertion_ids)
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
                    # 兼容占位：断链判定改由 boundaries + 第 5 步审核承担
                    "break_if_merged": dict(getattr(layer, "break_if_merged", None) or {}),
                    "depth": layer.depth,
                    "must_explain": must_explain,
                    "success_test": success_test,
                    "source_hints": list(getattr(layer, "source_hints", None) or []),
                    "source_refs": list(getattr(layer, "source_refs", None) or []),
                    "formula_hints": list(getattr(layer, "formula_hints", None) or []),
                    "primary_assertion_id": primary,
                    "debt_ids": debt_ids,
                    "supporting_node_ids": known_supporting,
                    "prerequisite_concept_ids": known_prereq,
                    "source_span_ids": known_spans,
                },
                assertion_ids=assertion_ids,
                concept_ids=concept_ids,
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
            issues.append("preparations 目标层未解析：" + "、".join(unresolved))
        if not issues:
            plan.meta.pop("issues", None)
        plan.precompute_boundaries()
        return plan

    # ---- boundary 预计算（第 5 步 PlanAuditor 消费）----

    def precompute_boundaries(self) -> None:
        """相邻层对生成 boundary 条目：outcome=unknown 待审；
        lost_* 从两层债务/断言/证据 span 交集确定性预填，reason 空待审。"""
        self.boundaries = []
        layers = [self.units[uid] for uid in self.order.get("layers") or []
                  if uid in self.units]
        for left, right in zip(layers, layers[1:]):
            shared_debts = sorted(set(left.data.get("debt_ids") or [])
                                  & set(right.data.get("debt_ids") or []))
            shared_assertions = sorted(set(left.assertion_ids) & set(right.assertion_ids))
            shared_spans = sorted(set(left.data.get("source_span_ids") or [])
                                  & set(right.data.get("source_span_ids") or []))
            self.boundaries.append({
                "left_layer_id": left.unit_id,
                "right_layer_id": right.unit_id,
                "outcome": "unknown",
                "lost_debt_ids": shared_debts,
                "lost_assertion_ids": shared_assertions,
                "evidence_span_ids": shared_spans,
                "reason": "",
            })

    # ---- 债务 owner 写回（克隆上改，不落盘；落盘由审核后统一提交）----

    def bind_debt_owners(self, registry: Any) -> Any:
        """把各层 unit.data["debt_ids"] 写回 debt.owner_layer_id（registry 克隆）。

        owner 不参与债务身份，初始绑定不递增 revision；输入 registry 不被触碰。"""
        work = registry.clone()
        for uid in self.order.get("layers") or []:
            unit = self.units.get(uid)
            if unit is None:
                continue
            for did in unit.data.get("debt_ids") or []:
                debt = work.debts.get(did)
                if debt is not None:
                    debt.owner_layer_id = uid
        return work

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
                break_if_merged=dict(d.get("break_if_merged") or {}),  # 兼容空值
                depth=d.get("depth", "core"),
                must_explain=list(d.get("must_explain") or []),   # 派生值
                success_test=list(d.get("success_test") or []),   # 派生值
                source_refs=list(d.get("source_refs") or []),
                primary_assertion_id=str(d.get("primary_assertion_id") or ""),
                debt_ids=list(d.get("debt_ids") or []),
                supporting_node_ids=list(d.get("supporting_node_ids") or []),
                prerequisite_concept_ids=list(d.get("prerequisite_concept_ids") or []),
                source_span_ids=list(d.get("source_span_ids") or []),
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
            coverage=dict(self.meta.get("coverage") or {}),
        ), metadata

    # ---- 持久化 ----

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema_digest,
            "plan_revision": self.plan_revision,
            "prompt_digest": self.prompt_digest,
            "admission_status": self.admission_status,
            "reviewed_revision": self.reviewed_revision,
            "graph_revision": self.graph_revision,
            "debt_registry_revision": self.debt_registry_revision,
            "boundaries": json.loads(json.dumps(self.boundaries)),
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
        status = str(data.get("admission_status") or "pending")
        plan = cls(plan_revision=int(data.get("plan_revision") or 1),
                   prompt_digest=str(data.get("prompt_digest") or ""),
                   meta=dict(data.get("meta") or {}),
                   preparations=list(data.get("preparations") or []),
                   tombstones=dict(data.get("tombstones") or {}),
                   admission_status=(status if status in PLAN_ADMISSION_STATUSES
                                     else "pending"),
                   reviewed_revision=int(data.get("reviewed_revision") or 0),
                   graph_revision=str(data.get("graph_revision") or ""),
                   debt_registry_revision=int(data.get("debt_registry_revision") or 0),
                   boundaries=list(data.get("boundaries") or []))
        plan.order = {k: list(v) for k, v in (data.get("order") or {}).items()}
        for uid, raw in (data.get("units") or {}).items():
            raw = dict(raw)
            if "revision" in raw and "semantic_revision" not in raw:
                raw["semantic_revision"] = raw.pop("revision")
            raw.pop("revision", None)
            raw.pop("content_digest", None)
            plan.units[uid] = PlanUnit(**raw)
        return plan


def _section_purposes(section: dict[str, Any]) -> list[str]:
    """章节目的数组：新协议 purposes；兼容旧单值 purpose。空值给 "overview"。"""
    raw = section.get("purposes")
    if isinstance(raw, list):
        purposes = [str(v).strip() for v in raw if str(v).strip()]
        if purposes:
            return purposes
    single = str(section.get("purpose") or "").strip()
    return [single] if single else ["overview"]


def validate_plan(plan: ReadingPlan, graph: SemanticGraph | None = None,
                  evidence_chunks: set[str] | None = None,
                  registry: Any | None = None) -> list[str]:
    """Plan 提交前不变量：order 唯一/存在/无孤儿/kind 一致/绑定存在于图；
    source_refs 指向真实 chunk/span；preparations 绑定与归属身份合法。
    第 3 步新增：primary assertion 真实且跨层唯一、每层 ≥1 个独占债务、
    同层债务 subject 落在 primary 闭包内、preparations 闭包/顺序合法。
    只产 gaps 不抛死；registry 缺失/degraded 时跳过债务相关检查。"""
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
    layer_uids = [uid for uid in plan.order.get("layers") or [] if uid in plan.units]
    graph_has_assertions = bool(graph is not None and any(
        n.kind == "assertion" for n in graph.nodes.values()))
    # ---- primary assertion：真实、跨层唯一（图有 assertion 可引时才检查）----
    if graph_has_assertions:
        primary_owner: dict[str, str] = {}
        for uid in layer_uids:
            unit = plan.units[uid]
            label = str(unit.data.get("name") or uid)
            primary = str(unit.data.get("primary_assertion_id") or "").strip()
            if not primary:
                gaps.append(f"层「{label}」缺少 primary_assertion_id")
                continue
            node = graph.nodes.get(primary)
            if node is None:
                gaps.append(f"层「{label}」primary assertion 不存在：{primary}")
            elif node.kind != "assertion":
                gaps.append(f"层「{label}」primary 指向非 assertion 节点：{primary}")
            elif primary in primary_owner:
                gaps.append(f"层「{label}」与层「{primary_owner[primary]}」"
                            f"共用同一 primary assertion：{primary}")
            else:
                primary_owner[primary] = label
    # ---- 债务归属：两层不得引用相同 debt_id；每层 ≥1 个独占债务；
    #      同层债务 subject == 本层 primary（宽松：或在其 depends_on 闭包内）----
    debts = getattr(registry, "debts", None) or {}
    debt_checkable = bool(debts) and getattr(registry, "admission_status", "") != "degraded"
    if debt_checkable:
        use_count: dict[str, int] = {}
        for uid in layer_uids:
            for did in plan.units[uid].data.get("debt_ids") or []:
                use_count[did] = use_count.get(did, 0) + 1
        for did in sorted(did for did, n in use_count.items() if n > 1):
            gaps.append(f"债务 {did} 被 {use_count[did]} 个层引用"
                        "（两层不得引用相同 debt_id，不得进入 canonical Plan）")
        for uid in layer_uids:
            unit = plan.units[uid]
            label = str(unit.data.get("name") or uid)
            dids = list(unit.data.get("debt_ids") or [])
            for did in dids:
                if did not in debts:
                    gaps.append(f"层「{label}」引用注册表外的债务：{did}")
            if not any(use_count.get(did) == 1 and did in debts for did in dids):
                gaps.append(f"层「{label}」无独占债务（每层至少拥有 1 条独有债务）")
            primary = str(unit.data.get("primary_assertion_id") or "").strip()
            if graph is not None and primary in graph.nodes:
                closure = depends_on_closure(graph, primary)
                for did in dids:
                    debt = debts.get(did)
                    if debt is None:
                        continue
                    subject = debt.subject_node_id
                    if subject != primary and subject not in closure:
                        gaps.append(
                            f"债务 {did} 的 subject（{subject}）不在层「{label}」"
                            f" primary assertion 的 depends_on 闭包内")
    elif any(plan.units[uid].data.get("debt_ids") for uid in layer_uids) \
            and registry is not None:
        # 注册表缺失/为空/degraded 但 Plan 携带债务引用：跳过债务检查并注明
        gaps.append("债务注册表缺失/为空/degraded：跳过债务归属检查"
                    "（Plan 中的 debt_ids 未校验）")
    # ---- preparations：身份/归属/顺序/闭包合法 ----
    doc_position: dict[str, int] = {}
    for idx, uid in enumerate(plan.order.get("sections") or []):
        doc_position[uid] = idx
    base = len(plan.order.get("sections") or [])
    for idx, uid in enumerate(plan.order.get("layers") or []):
        doc_position[uid] = base + idx
    for prep in plan.preparations:
        label = str(prep.get("concept") or "")[:30] or "未命名概念"
        concept_id = str(prep.get("concept_id") or "").strip()
        if not concept_id:
            gaps.append(f"preparation「{label}」concept_id 空绑定")
        owner = str(prep.get("owner_section_id") or "")
        owner_unit = plan.units.get(owner)
        if owner_unit is None or owner_unit.kind != "section":
            gaps.append(f"preparation「{label}」owner_section_id 非法：{owner or '空'}")
        targets = [str(t) for t in (prep.get("target_layer_ids") or []) if str(t)]
        if not targets:
            gaps.append(f"preparation「{label}」target_layer_ids 为空")
        for tid in targets:
            target_unit = plan.units.get(tid)
            if target_unit is None:
                gaps.append(f"preparation「{label}」target 指向不存在单元：{tid}")
                continue
            # owner 必须在正文顺序中位于使用者之前
            if owner in doc_position and tid in doc_position \
                    and doc_position[owner] >= doc_position[tid]:
                gaps.append(f"preparation「{label}」owner 未位于使用者之前：{tid}")
            # concept 必须落在目标层 primary assertion 的 depends_on 闭包内
            if concept_id and graph is not None:
                primary = str(target_unit.data.get("primary_assertion_id") or "").strip()
                if primary in graph.nodes:
                    closure = depends_on_closure(graph, primary) | {primary}
                    if concept_id not in closure:
                        gaps.append(
                            f"preparation「{label}」concept 不在目标层"
                            f"「{target_unit.data.get('name') or tid}」 primary assertion"
                            " 的 depends_on 闭包内")
    return gaps
