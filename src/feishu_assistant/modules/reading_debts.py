"""理解债务基础设施（V8 第 1/2 步）：DebtSpec、确定性身份、Canonical DebtRegistry、
DebtRepairEngine（修复八操作）、DebtAuditor（六准入审核 + 有界修复循环）。

设计契约（与 V8 规划裁定一致）：
- 债务在分层之前独立生成、去重、审核（DebtCompiler → DebtAuditor → 本注册表）；
  层只是债务的容器，must_explain/success_test 后续由程序从债务派生。
- debt_id 由程序计算，只含语义身份字段（kind/subject/relation/object/premise）；
  措辞、learner_action、success_test、evidence_span_ids、owner、展示顺序绝不参与身份。
  模型输出不含 ID（from_payload 入口），程序算，禁止模型自报。
- kind 是债务操作类型，不新增图节点类型（图仍 concept/assertion/source_span 三类）。
- validate 只返回 gaps 列表，绝不抛异常：DebtAuditor 的程序门禁直接消费。
- 修复通道：clone → 逐项校验 → 应用 → validate 重审 → 原子提交；任何一步失败
  RepairRejected，原 registry 字节不变。审核侧任何异常路径 degraded 批注继续，不判死。
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

DEBT_REGISTRY_SCHEMA = "debt-registry-v1"

# 债务操作类型：读者要偿还的是哪一类理解动作
DEBT_KINDS = (
    "definition",    # 定义：某概念/记号指什么
    "mechanism",     # 机制：某方法如何运转
    "derivation",    # 推导：从前提如何得到结论
    "comparison",    # 对比：与对照物的差异
    "taxonomy",      # 分类：在 taxonomy 中的位置
    "evidence",      # 证据：支撑某主张的实验/数据
    "boundary",      # 边界：适用条件与失效情形
)

# 债务关系：subject 以何种关系关联 object（均为图内真实节点）
DEBT_RELATIONS = (
    "defines",         # subject 定义 object
    "transforms",      # subject 把 premise/object 变换为新形态
    "derives",         # subject 由 premise 推导而来
    "contrasts_with",  # subject 与 object 对比
    "classifies",      # subject 把 object 归入类别
    "supported_by",    # subject 被 object（证据节点）支撑
    "limited_by",      # subject 的适用性被 object 限制
)

# 准入状态：pending（待审）| clean（全部门禁通过）| degraded（带批注继续）
ADMISSION_STATUSES = ("pending", "clean", "degraded")


def compute_debt_id(
    kind: str,
    subject_node_id: str,
    relation: str,
    object_node_ids: Iterable[str] = (),
    premise_node_ids: Iterable[str] = (),
) -> str:
    """债务身份：仅由语义身份字段决定。

    措辞/learner_action/success_test/evidence_span_ids/owner/展示顺序不参与；
    object/premise 内部排序也不参与（sorted 后入哈希）。"""
    material = "|".join([
        str(kind).strip(),
        str(subject_node_id).strip(),
        str(relation).strip(),
        ",".join(sorted(str(v).strip() for v in object_node_ids)),
        ",".join(sorted(str(v).strip() for v in premise_node_ids)),
    ])
    return stable_id("debt", material)


@dataclass
class DebtSpec:
    """一项理解债务：读者为跟上核心链条必须完成的认知动作。"""

    debt_id: str                       # 程序计算的身份（见 compute_debt_id）
    kind: str                          # DEBT_KINDS 之一
    subject_node_id: str               # 债务主题：图内真实节点
    relation: str                      # DEBT_RELATIONS 之一
    object_node_ids: list[str] = field(default_factory=list)    # 关系客体（图节点）
    premise_node_ids: list[str] = field(default_factory=list)   # 理解前提（图节点）
    evidence_span_ids: list[str] = field(default_factory=list)  # 原文证据 span
    learner_action: str = ""           # 读者要完成的动作（派生 must_explain 的事实源）
    success_test: str = ""             # 偿还判据（派生 success_test 的事实源）
    necessity_reason: str = ""         # 必要性：指向哪条核心链，删之为何断
    owner_layer_id: str = ""           # 归属层（第 3 步 Plan 接线后才有意义；不参与身份）
    revision: int = 1                  # 实质修复递增；措辞变化不递增

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "DebtSpec":
        """模型输出入口：payload 不含 ID，程序从身份字段计算。

        不做合法性拒绝（编造 ID/空字段由 DebtRegistry.validate 检出），
        保证「模型直出 → 程序算 ID → 校验」的流水线不被解析层打断。"""
        kind = str(payload.get("kind") or "").strip()
        subject = str(payload.get("subject_node_id") or "").strip()
        relation = str(payload.get("relation") or "").strip()
        objects = _str_list(payload.get("object_node_ids"))
        premises = _str_list(payload.get("premise_node_ids"))
        return cls(
            debt_id=compute_debt_id(kind, subject, relation, objects, premises),
            kind=kind,
            subject_node_id=subject,
            relation=relation,
            object_node_ids=objects,
            premise_node_ids=premises,
            evidence_span_ids=_str_list(payload.get("evidence_span_ids")),
            learner_action=str(payload.get("learner_action") or "").strip(),
            success_test=str(payload.get("success_test") or "").strip(),
            necessity_reason=str(payload.get("necessity_reason") or "").strip(),
            owner_layer_id=str(payload.get("owner_layer_id") or "").strip(),
            revision=int(payload.get("revision") or 1),
        )

    def identity_material(self) -> tuple[str, str, str, list[str], list[str]]:
        """身份字段快照：validate 用它重算 debt_id，检出手改/漂移。"""
        return (self.kind, self.subject_node_id, self.relation,
                sorted(self.object_node_ids), sorted(self.premise_node_ids))


def _str_list(value: Any) -> list[str]:
    return [str(v).strip() for v in value or [] if str(v).strip()]


class DebtRegistry:
    """Canonical 债务注册表：debt 字典 + order 数组 + revision + 准入状态。

    修复通道约定（第 2 步接线）：clone → 应用修复 → validate 重审 → 原子提交；
    本类只保证存储、去重与校验，不做任何模型调用。"""

    def __init__(self) -> None:
        self.debts: dict[str, DebtSpec] = {}
        self.order: list[str] = []              # 展示顺序，不参与身份
        self.registry_revision: int = 1         # 实质内容变化才递增
        self.admission_status: str = "pending"  # ADMISSION_STATUSES 之一
        self.duplicate_count: int = 0           # add 去重命中计数（对账用）
        self.issues: list[str] = []             # 降级/剔除批注（透明状态，随落盘可审计）

    # ---- 构建 ----

    def add(self, spec: DebtSpec) -> bool:
        """加入债务：同 debt_id 幂等去重（计数，不覆盖、不报错、不增 revision）。

        返回 True 表示新入册；False 表示命中已有身份被去重。"""
        if spec.debt_id in self.debts:
            self.duplicate_count += 1
            return False
        self.debts[spec.debt_id] = spec
        self.order.append(spec.debt_id)
        self.registry_revision += 1
        return True

    def remove(self, debt_id: str) -> bool:
        """退休债务：从字典与 order 中移除。返回 False 表示目标不存在（不报错）。"""
        if debt_id not in self.debts:
            return False
        del self.debts[debt_id]
        self.order = [did for did in self.order if did != debt_id]
        self.registry_revision += 1
        return True

    # ---- 克隆 ----

    def clone(self) -> "DebtRegistry":
        """经 to_dict/from_dict 往返的深拷贝：修复通道的克隆→应用→原子提交用。"""
        return self.from_dict(json.loads(json.dumps(self.to_dict(), ensure_ascii=False)))

    # ---- 校验 ----

    def validate(self, graph: SemanticGraph) -> list[str]:
        """对图校验注册表，返回 gaps 列表（空 = 通过），绝不抛异常。

        检查项：kind/relation 合法；subject/object/premise 是图内真实节点；
        evidence_span_ids 是真实 span；无重复/孤儿 debt_id；debt_id 与身份字段一致；
        necessity_reason/learner_action/success_test 非空。"""
        gaps: list[str] = []
        if len(set(self.order)) != len(self.order):
            gaps.append("order 数组存在重复 debt_id")
        orphans = set(self.order) - set(self.debts)
        if orphans:
            gaps.append("order 引用了不存在的债务：" + "、".join(sorted(orphans)))
        missing = set(self.debts) - set(self.order)
        if missing:
            gaps.append("存在未被 order 引用的孤儿债务：" + "、".join(sorted(missing)))
        if self.admission_status not in ADMISSION_STATUSES:
            gaps.append(f"非法 admission_status：{self.admission_status}")
        for debt_id in self.order:
            spec = self.debts.get(debt_id)
            if spec is None:
                continue  # 已在孤儿检查记录
            label = f"债务 {debt_id}"
            if spec.kind not in DEBT_KINDS:
                gaps.append(f"{label} kind 非法：{spec.kind or '空'}")
            if spec.relation not in DEBT_RELATIONS:
                gaps.append(f"{label} relation 非法：{spec.relation or '空'}")
            if not spec.subject_node_id:
                gaps.append(f"{label} subject_node_id 为空")
            elif spec.subject_node_id not in graph.nodes:
                gaps.append(f"{label} subject 指向不存在的图节点：{spec.subject_node_id}")
            for nid in spec.object_node_ids:
                if nid not in graph.nodes:
                    gaps.append(f"{label} object 指向不存在的图节点：{nid}")
            for nid in spec.premise_node_ids:
                if nid not in graph.nodes:
                    gaps.append(f"{label} premise 指向不存在的图节点：{nid}")
            for sid in spec.evidence_span_ids:
                if sid not in graph.spans:
                    gaps.append(f"{label} evidence 指向不存在的 span：{sid}")
            recomputed = compute_debt_id(spec.kind, spec.subject_node_id, spec.relation,
                                         spec.object_node_ids, spec.premise_node_ids)
            if recomputed != spec.debt_id:
                gaps.append(f"{label} debt_id 与身份字段不一致（疑似手改 ID）："
                            f"应为 {recomputed}")
            if not spec.necessity_reason:
                gaps.append(f"{label} necessity_reason 为空")
            if not spec.learner_action:
                gaps.append(f"{label} learner_action 为空")
            if not spec.success_test:
                gaps.append(f"{label} success_test 为空")
        return gaps

    # ---- 持久化 ----

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DEBT_REGISTRY_SCHEMA,
            "registry_revision": self.registry_revision,
            "admission_status": self.admission_status,
            "duplicate_count": self.duplicate_count,
            "order": list(self.order),
            "debts": {did: asdict(d) for did, d in self.debts.items()},
            "issues": list(self.issues),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DebtRegistry":
        if str(data.get("schema")) != DEBT_REGISTRY_SCHEMA:
            raise ValueError(f"debt registry schema 不兼容：{data.get('schema')}")
        registry = cls()
        registry.registry_revision = int(data.get("registry_revision") or 1)
        status = str(data.get("admission_status") or "pending")
        registry.admission_status = status if status in ADMISSION_STATUSES else "pending"
        registry.duplicate_count = int(data.get("duplicate_count") or 0)
        registry.order = [str(v) for v in data.get("order") or []]
        for did, raw in (data.get("debts") or {}).items():
            registry.debts[str(did)] = DebtSpec(**dict(raw))
        registry.issues = [str(v) for v in data.get("issues") or []]
        return registry


# ---------------------------------------------------------------------------
# 第 2 步：准入检查项、issue 路由枚举、修复八操作（DebtRepairEngine）
# ---------------------------------------------------------------------------

# 六条准入检查：DebtAuditor 逐项判定，程序门禁要求全部 pass 才可 clean
DEBT_CHECK_ITEMS = (
    "source_backed",   # 有据：ID 真实且 span 原文确实支撑
    "necessary",       # 必要：指向具体核心链，删之即断
    "atomic",          # 原子：一项债务一个主要认知动作
    "non_duplicate",   # 不重复：比身份与能力不比文字
    "testable",        # 可测：success_test 闭卷可执行，拒空话
    "granularity",     # 粒度落地：不拆碎、不装筐
)

CHECK_OUTCOMES = ("pass", "fail", "unknown")

# 八类 issue（修复路由表；每类必有确定修复路径）
DEBT_ISSUE_KINDS = (
    "duplicate_debt",     # 语义重复（身份不同但认知动作相同）
    "overbroad_debt",     # 过宽：一条债务装了多个认知动作
    "fragmented_debt",    # 过碎：同一认知动作被拆成多条
    "wrong_granularity",  # 粒度错误但不宜简单拆/合
    "missing_debt",       # 核心链上的真实理解难点缺失
    "unfounded_debt",     # 无据：证据不支撑，或必要性落空
    "wrong_debt_kind",    # kind/relation 与真实语义关系不符
    "wrong_owner",        # 归属理解单元错位
)

# 七个修复操作（覆盖八类 issue；unfounded_debt 有 rebind/retire 两条路径）
DEBT_REPAIR_OPS = (
    "canonicalize_debt",     # duplicate_debt：保留 canonical，退休其余
    "replace_debt_fragment", # overbroad/fragmented/wrong_granularity：拆/合
    "add_debt_from_graph",   # missing_debt：补全（必须给全真实 ID）
    "rebind_debt_evidence",  # unfounded_debt：换绑真实 span 证据
    "retire_debt",           # unfounded_debt：无法补救，退休
    "update_debt_relation",  # wrong_debt_kind：关系类型+动作+判据同步更新
    "reassign_debt_owner",   # wrong_owner：改归属层（第 3 步 Plan 接线后生效）
)

SEVERITY_RANK = {"high": 3, "medium": 2, "low": 1}
_SEVERITY_ALIASES = {"高": "high", "中": "medium", "低": "low"}


class RepairRejected(Exception):
    """修复动作非法/版本冲突/证据缺失：事务整体拒绝，原 registry 字节不变。"""


@dataclass
class DebtRepairAction:
    """一条债务修复动作：op + 乐观并发版本 + 操作参数。"""

    op: str
    expected_registry_revision: int
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DebtRepairAction":
        if not isinstance(data, dict):
            raise RepairRejected("DebtRepairAction 必须是对象")
        op = str(data.get("op") or "").strip()
        if op not in DEBT_REPAIR_OPS:
            raise RepairRejected(f"非法修复操作：{op or '缺失'}")
        try:
            revision = int(data.get("expected_registry_revision"))
        except (TypeError, ValueError):
            raise RepairRejected("expected_registry_revision 缺失或非法")
        params = data.get("params")
        if not isinstance(params, dict):
            raise RepairRejected("params 缺失或不是对象")
        return cls(op=op, expected_registry_revision=revision, params=dict(params))


@dataclass
class DebtRepairStats:
    applied: int = 0
    retired: list[str] = field(default_factory=list)
    created: list[str] = field(default_factory=list)
    touched: list[str] = field(default_factory=list)

    @property
    def affected_ids(self) -> set[str]:
        return set(self.retired) | set(self.created) | set(self.touched)


class DebtRepairEngine:
    """债务修复事务：clone → 逐项校验 → 应用 → 重算 ID → validate → 原子提交。

    任何一步失败抛 RepairRejected，输入 registry 不被触碰（字节不变）。
    本类不做任何模型调用。"""

    def apply(
        self,
        registry: DebtRegistry,
        graph: SemanticGraph,
        actions: list[DebtRepairAction],
    ) -> tuple[DebtRegistry, DebtRepairStats]:
        if not actions:
            raise RepairRejected("空 repairs")
        work = registry.clone()
        base_revision = registry.registry_revision
        stats = DebtRepairStats()
        for action in actions:
            if action.expected_registry_revision != base_revision:
                raise RepairRejected(
                    f"registry revision 冲突：动作期望 {action.expected_registry_revision}，"
                    f"当前 {base_revision}")
            self._apply_one(work, graph, action, stats)
            stats.applied += 1
        # 统一 revision 口径：每应用一个动作递增一次（覆盖 add/remove 的内部自增，
        # 最终值只取决于基准与动作数，确定性可断言）
        work.registry_revision = base_revision + stats.applied
        gaps = work.validate(graph)
        if gaps:
            raise RepairRejected("修复后校验未过：" + "；".join(gaps[:6]))
        return work, stats

    # ---- 单项应用 ----

    def _apply_one(self, work: DebtRegistry, graph: SemanticGraph,
                   action: DebtRepairAction, stats: DebtRepairStats) -> None:
        op, params = action.op, action.params
        if op == "canonicalize_debt":
            keep = self._param_str(params, "keep_debt_id")
            retires = self._param_str_list(params, "retire_debt_ids")
            self._require_debts(work, [keep] + retires)
            if keep in retires:
                raise RepairRejected("canonical 不得同时出现在退休列表")
            for did in retires:
                work.remove(did)
                stats.retired.append(did)
        elif op == "replace_debt_fragment":
            retires = self._param_str_list(params, "retire_debt_ids")
            self._require_debts(work, retires)
            specs = self._build_specs(graph, params, op)
            retiring = set(retires)
            self._check_collisions(work, specs, allow=retiring)
            self._check_batch_distinct(specs)
            for did in retires:
                work.remove(did)
                stats.retired.append(did)
            for spec in specs:
                work.add(spec)
                stats.created.append(spec.debt_id)
        elif op == "add_debt_from_graph":
            specs = self._build_specs(graph, params, op)
            self._check_collisions(work, specs, allow=set())
            self._check_batch_distinct(specs)
            for spec in specs:
                work.add(spec)
                stats.created.append(spec.debt_id)
        elif op == "rebind_debt_evidence":
            debt_id = self._param_str(params, "debt_id")
            self._require_debts(work, [debt_id])
            spans = self._param_str_list(params, "evidence_span_ids")
            unknown = [sid for sid in spans if sid not in graph.spans]
            if unknown:
                raise RepairRejected("evidence 指向不存在的 span：" + "、".join(unknown))
            spec = work.debts[debt_id]
            spec.evidence_span_ids = spans  # 非身份字段：debt_id 不变
            spec.revision += 1
            stats.touched.append(debt_id)
        elif op == "retire_debt":
            debt_id = self._param_str(params, "debt_id")
            self._require_debts(work, [debt_id])
            work.remove(debt_id)
            stats.retired.append(debt_id)
        elif op == "update_debt_relation":
            debt_id = self._param_str(params, "debt_id")
            self._require_debts(work, [debt_id])
            kind = self._param_str(params, "kind")
            relation = self._param_str(params, "relation")
            learner_action = self._param_str(params, "learner_action")
            success_test = self._param_str(params, "success_test")
            if kind not in DEBT_KINDS:
                raise RepairRejected(f"kind 非法：{kind or '空'}")
            if relation not in DEBT_RELATIONS:
                raise RepairRejected(f"relation 非法：{relation or '空'}")
            spec = work.debts[debt_id]
            spec.kind, spec.relation = kind, relation
            spec.learner_action, spec.success_test = learner_action, success_test
            spec.revision += 1
            # 身份字段变化 → 重算 debt_id 并保持 order 位置
            new_id = compute_debt_id(kind, spec.subject_node_id, relation,
                                     spec.object_node_ids, spec.premise_node_ids)
            if new_id != debt_id:
                if new_id in work.debts:
                    raise RepairRejected(f"身份重算后与在册债务冲突：{new_id}")
                del work.debts[debt_id]
                spec.debt_id = new_id
                work.debts[new_id] = spec
                work.order = [new_id if did == debt_id else did for did in work.order]
            stats.touched.append(new_id)
        elif op == "reassign_debt_owner":
            debt_id = self._param_str(params, "debt_id")
            self._require_debts(work, [debt_id])
            owner = self._param_str(params, "owner_layer_id")
            spec = work.debts[debt_id]
            spec.owner_layer_id = owner  # 非身份字段：debt_id 不变
            spec.revision += 1
            stats.touched.append(debt_id)
        else:  # from_dict 已挡，防御兜底
            raise RepairRejected(f"非法修复操作：{op}")

    # ---- 参数与校验工具 ----

    @staticmethod
    def _param_str(params: dict[str, Any], key: str) -> str:
        value = str(params.get(key) or "").strip()
        if not value:
            raise RepairRejected(f"params 缺少 {key}")
        return value

    @staticmethod
    def _param_str_list(params: dict[str, Any], key: str) -> list[str]:
        value = params.get(key)
        if not isinstance(value, list):
            raise RepairRejected(f"params.{key} 必须是数组")
        items = [str(v).strip() for v in value if str(v).strip()]
        if not items:
            raise RepairRejected(f"params.{key} 为空")
        return items

    @staticmethod
    def _require_debts(registry: DebtRegistry, debt_ids: list[str]) -> None:
        missing = [did for did in debt_ids if did not in registry.debts]
        if missing:
            raise RepairRejected("目标债务不存在：" + "、".join(missing))

    def _build_specs(self, graph: SemanticGraph, params: dict[str, Any],
                     op: str) -> list[DebtSpec]:
        raw = params.get("creates")
        if not isinstance(raw, list) or not raw:
            raise RepairRejected(f"{op} 缺少 creates（拆/合与补全都必须给出新债务）")
        specs: list[DebtSpec] = []
        for entry in raw:
            if not isinstance(entry, dict):
                raise RepairRejected(f"{op} 的 creates 项必须是对象")
            spec = DebtSpec.from_payload(entry)
            # 新 ID 真实性与字段完整性：复用单债务探针 validate（fail fast，
            # 最终还有全表 validate 兜底）
            probe = DebtRegistry()
            probe.add(spec)
            gaps = probe.validate(graph)
            if gaps:
                raise RepairRejected(f"{op} 新债务非法：" + "；".join(gaps[:4]))
            if not spec.evidence_span_ids:
                raise RepairRejected(f"{op} 新债务缺少 evidence_span_ids")
            if not spec.object_node_ids and not spec.premise_node_ids:
                raise RepairRejected(f"{op} 新债务 object/premise 至少一个非空")
            specs.append(spec)
        return specs

    @staticmethod
    def _check_collisions(registry: DebtRegistry, specs: list[DebtSpec],
                          allow: set[str]) -> None:
        for spec in specs:
            if spec.debt_id in registry.debts and spec.debt_id not in allow:
                raise RepairRejected(f"新债务与在册债务身份冲突：{spec.debt_id}")

    @staticmethod
    def _check_batch_distinct(specs: list[DebtSpec]) -> None:
        seen: set[str] = set()
        for spec in specs:
            if spec.debt_id in seen:
                raise RepairRejected(f"creates 内部身份重复：{spec.debt_id}")
            seen.add(spec.debt_id)


# ---------------------------------------------------------------------------
# 第 2 步：DebtAuditor —— 六准入审核 + 有界修复循环
# ---------------------------------------------------------------------------

class DebtAuditor:
    """债务准入审核：模型逐项六检查 + 程序门禁 + 修复循环（格式纠正一次 +
    语义修复至多两轮），任何异常路径 degraded 批注继续，绝不抛异常判死。

    程序门禁：clean ⇔ 注册表每条债务 × 六条检查全部 outcome=pass
    且 validate(graph) 无 gaps；fail/unknown/gaps 一律不得 clean。
    修后必须轻量复核受影响债务项；复核不可用 → 回滚 last_valid + 批注。
    """

    MAX_REPAIR_ROUNDS = 2

    def audit(
        self,
        registry: DebtRegistry,
        graph: SemanticGraph,
        client: Any,
        prompts_dir: Any,
    ) -> tuple[DebtRegistry, dict[str, Any]]:
        report: dict[str, Any] = {
            "status": "pending", "audit_calls": 0, "repair_rounds": 0,
            "repairs_applied": 0, "checks": [], "issues": [], "notes": [],
            "open_gaps": [],
        }
        working = registry.clone()
        working.admission_status = "pending"
        last_valid = registry.clone()
        accum: dict[tuple[str, str], tuple[str, str]] = {}  # (debt_id, item) → (outcome, evidence)
        format_budget = 1  # 全流程共用的一次格式纠正
        engine = DebtRepairEngine()

        def _degraded(reason: str) -> None:
            working.admission_status = "degraded"
            note = f"债务审核 degraded：{reason}"
            if note not in working.issues:
                working.issues.append(note)
            report["notes"].append(note)
            report["status"] = "degraded"
            logger.warning(note)

        if not working.debts:
            _degraded("注册表为空，无债务可审")
            return working, report

        out, format_budget = self._call_and_parse(
            client, prompts_dir, graph, working, None, "", format_budget, report)
        if out is None:
            _degraded("审核调用不可用或输出连续非法")
            return working, report
        self._accumulate(accum, out["checks"])
        report["checks"], report["issues"] = out["checks"], out["issues"]
        latest = out

        gaps = self._program_gaps(working, graph, accum)
        prev_max, prev_count = self._max_severity(gaps), len(gaps)
        prev_digest = self._digest(working)
        seen_action_sigs: set[str] = set()
        repair_round = 0

        while gaps:
            if repair_round >= self.MAX_REPAIR_ROUNDS:
                _degraded(f"修复 {repair_round} 轮后仍有 {len(gaps)} 项门禁未过")
                break
            try:
                actions = [DebtRepairAction.from_dict(a) for a in latest["repairs"]]
                parse_error: Exception | None = None
            except RepairRejected as exc:
                actions, parse_error = [], exc
            if not actions:
                if parse_error is not None and format_budget:
                    # 一次格式纠正：修复动作协议错误回灌，不消耗语义修复轮次
                    format_budget -= 1
                    retry, format_budget = self._call_and_parse(
                        client, prompts_dir, graph, working, None,
                        f"修复动作非法：{parse_error}", format_budget, report)
                    if retry is not None:
                        self._accumulate(accum, retry["checks"])
                        report["checks"], report["issues"] = retry["checks"], retry["issues"]
                        latest = retry
                        continue
                _degraded("存在未过门禁项但无合法修复动作"
                          + (f"（{parse_error}）" if parse_error else ""))
                break
            sigs = [self._action_sig(a) for a in actions]
            if all(sig in seen_action_sigs for sig in sigs):
                _degraded("修复动作与历史完全重复（无进展熔断）")
                break
            try:
                candidate, stats = engine.apply(working, graph, actions)
            except RepairRejected as exc:
                if not format_budget:
                    _degraded(f"修复动作被拒：{exc}")
                    break
                # 一次格式纠正：把程序拒绝原因回灌，要求重出修复动作
                format_budget -= 1
                retry, format_budget = self._call_and_parse(
                    client, prompts_dir, graph, working, None,
                    f"修复动作被程序拒绝：{exc}", format_budget, report)
                if retry is None:
                    _degraded("修复被拒且纠正调用不可用")
                    break
                self._accumulate(accum, retry["checks"])
                report["checks"], report["issues"] = retry["checks"], retry["issues"]
                try:
                    actions = [DebtRepairAction.from_dict(a) for a in retry["repairs"]]
                    candidate, stats = engine.apply(working, graph, actions)
                    sigs = [self._action_sig(a) for a in actions]
                except RepairRejected as exc2:
                    _degraded(f"修复动作连续被拒：{exc2}")
                    break
            # 修后必须重新审核（轻量：只复核受影响债务项；程序 validate 已在 apply 内完成）
            # 复核范围 = 修复触及的债务 ∪ 当前被 fail/unknown 点名且仍存在的债务
            # （fail 指向的债务若不经复核改判，其判定永不消除，循环会空转）
            open_ids = {did for (did, _item), (outcome, _ev) in accum.items()
                        if outcome != "pass" and did in candidate.debts}
            scoped, format_budget = self._call_and_parse(
                client, prompts_dir, graph, candidate,
                sorted(stats.affected_ids | open_ids), "", format_budget, report)
            if scoped is None:
                # 不可用 → 回滚 last_valid Registry + 批注继续
                working = last_valid.clone()
                _degraded("修后复核不可用，已回滚 last_valid 注册表")
                break
            repair_round += 1
            report["repair_rounds"] = repair_round
            report["repairs_applied"] += stats.applied
            working = candidate
            self._accumulate(accum, scoped["checks"])
            self._purge(accum, working)
            report["checks"], report["issues"] = scoped["checks"], scoped["issues"]
            gaps = self._program_gaps(working, graph, accum)
            # 继续条件三选一：issue 集合或最高严重度下降 / digest 实际变化 / 无重复相同 action
            improved = self._max_severity(gaps) < prev_max or len(gaps) < prev_count
            changed = self._digest(working) != prev_digest
            no_repeat = any(sig not in seen_action_sigs for sig in sigs)
            seen_action_sigs.update(sigs)
            if not (improved or changed or no_repeat):
                _degraded("修复无实质进展（issue 未降 / digest 未变 / 动作重复）")
                break
            prev_max, prev_count = self._max_severity(gaps), len(gaps)
            prev_digest = self._digest(working)
            latest = scoped
            last_valid = working.clone()

        report["open_gaps"] = [gap["detail"] for gap in gaps]
        if not gaps and report["status"] == "pending":
            working.admission_status = "clean"
            report["status"] = "clean"
            logger.info("债务准入审核 clean：%d 条债务全部门禁通过", len(working.debts))
        elif report["status"] == "pending":
            _degraded("审核流程异常收口")  # 防御兜底：正常路径各 break 均已批注
        return working, report

    # ---- 模型调用与协议解析 ----

    def _call_and_parse(
        self,
        client: Any,
        prompts_dir: Any,
        graph: SemanticGraph,
        registry: DebtRegistry,
        scope_ids: list[str] | None,
        correction: str,
        format_budget: int,
        report: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, int]:
        """一次审核调用（消耗共享的一次性格式纠正预算）。失败返回 (None, 剩余预算)。"""
        from ..kimi import KimiError
        from .reading import LOCAL_REVIEW_OUTPUT_TOKENS  # 延迟导入防环

        prompt = self._build_prompt(prompts_dir, graph, registry, scope_ids, correction)
        try:
            raw = client.chat(
                [{"role": "user", "content": prompt}],
                heavy=True, json_mode=True, low_reasoning=True,
                max_tokens=LOCAL_REVIEW_OUTPUT_TOKENS,
            )
        except KimiError as exc:
            report["notes"].append(f"DebtAuditor 调用失败：{str(exc)[:160]}")
            return None, format_budget
        report["audit_calls"] += 1
        try:
            return self._parse_output(raw, registry, scope_ids), format_budget
        except ValueError as exc:
            if not format_budget:
                report["notes"].append(f"DebtAuditor 输出非法且无纠正预算：{exc}")
                return None, format_budget
            format_budget -= 1
            fix_prompt = (prompt + "\n\n## 本次必须修正\n"
                          f"上一次输出未通过程序校验：{exc}\n"
                          "只输出一个符合协议的 JSON 对象，不要任何解释或围栏。")
            try:
                raw = client.chat(
                    [{"role": "user", "content": fix_prompt}],
                    heavy=True, json_mode=True, low_reasoning=True,
                    max_tokens=LOCAL_REVIEW_OUTPUT_TOKENS,
                )
            except KimiError as exc2:
                report["notes"].append(f"DebtAuditor 纠正调用失败：{str(exc2)[:160]}")
                return None, format_budget
            report["audit_calls"] += 1
            try:
                return self._parse_output(raw, registry, scope_ids), format_budget
            except ValueError as exc2:
                report["notes"].append(f"DebtAuditor 纠正后输出仍非法：{exc2}")
                return None, format_budget

    def _build_prompt(
        self,
        prompts_dir: Any,
        graph: SemanticGraph,
        registry: DebtRegistry,
        scope_ids: list[str] | None,
        correction: str,
    ) -> str:
        from .reading import _load, _render  # 延迟导入防环
        from .reading_graph import render_graph_catalog

        scope = sorted(s for s in (scope_ids or []) if s in registry.debts)
        if scope_ids is None:
            scope_text = ("## 审核范围\n\n全量审核：注册表内每条债务 × 六条检查都必须给出判定。")
        else:
            scope_text = (
                "## 审核范围\n\n本次为修后轻量复核：**只复核下列债务项**（其余债务此前已"
                "复核，不得改动其判定，也不得将其列入 checks/issues/repairs）：\n"
                + ("、".join(scope) if scope else "（受影响债务已全部退休：确认无遗留问题即可，"
                                                 "checks 给空数组）"))
        view = {
            "registry_revision": registry.registry_revision,
            "debts": [asdict(registry.debts[did]) for did in registry.order
                      if scope_ids is None or did in scope],
        }
        prompt = _render(
            _load(prompts_dir / "reading_debt_audit.md"),
            GRAPH_CATALOG=render_graph_catalog(graph, max_chars=9000),
            EVIDENCE_APPENDIX=self._evidence_appendix(graph, registry, scope or None),
            AUDIT_SCOPE=scope_text,
            DEBT_REGISTRY_JSON=json.dumps(view, ensure_ascii=False, indent=2),
        )
        if correction:
            prompt += ("\n\n## 上次输出未通过程序校验\n" + correction
                       + "\n请按协议重新输出。")
        return prompt

    @staticmethod
    def _evidence_appendix(
        graph: SemanticGraph,
        registry: DebtRegistry,
        scope_ids: list[str] | None,
        max_chars: int = 8000,
    ) -> str:
        """确定性证据附录：审核范围内债务引用的 span 原文摘录（超预算省略并标注）。"""
        span_ids: set[str] = set()
        for did in registry.order:
            if scope_ids is not None and did not in scope_ids:
                continue
            span_ids.update(registry.debts[did].evidence_span_ids)
        lines: list[str] = []
        used = omitted = 0
        for sid in sorted(span_ids):
            span = graph.spans.get(sid)
            if span is None:
                continue  # 幽灵 span 由 validate 检出，附录不造内容
            text = re.sub(r"\s+", " ", span.text).strip()
            if len(text) > 600:
                text = text[:600] + "…"
            block = f"### {sid}\n{text}"
            if used + len(block) + 1 > max_chars:
                omitted += 1
                continue
            lines.append(block)
            used += len(block) + 1
        if omitted:
            lines.append(f"…（{omitted} 条 span 超出附录预算已省略，"
                         "对应债务的 source_backed 请判 unknown）")
        return "\n\n".join(lines) or "（审核范围未引用任何 span）"

    def _parse_output(
        self,
        raw: str,
        registry: DebtRegistry,
        scope_ids: list[str] | None,
    ) -> dict[str, Any]:
        """解析审核输出；协议违反（未知枚举/引用不存在债务/缺举证据）→ ValueError。"""
        from .reading import _json_object  # 延迟导入防环

        payload = _json_object(raw)
        in_scope = [did for did in registry.order if scope_ids is None or did in scope_ids]
        checks = self._parse_checks(payload.get("checks"), registry,
                                    require=bool(in_scope))
        issues = self._parse_issues(payload.get("issues"), registry)
        repairs = payload.get("repairs") or []
        if not isinstance(repairs, list):
            raise ValueError("repairs 必须是数组")
        return {"status": str(payload.get("status") or "").strip(),
                "checks": checks, "issues": issues, "repairs": repairs}

    @staticmethod
    def _parse_checks(value: Any, registry: DebtRegistry, *,
                      require: bool) -> list[dict[str, Any]]:
        if not isinstance(value, list) or (require and not value):
            raise ValueError("checks 缺失或为空")
        checks: list[dict[str, Any]] = []
        for entry in value:
            if not isinstance(entry, dict):
                raise ValueError("checks 含非对象项")
            item = str(entry.get("item") or "").strip()
            if item not in DEBT_CHECK_ITEMS:
                raise ValueError(f"非法检查项：{item or '空'}")
            outcome = str(entry.get("outcome") or "").strip()
            if outcome not in CHECK_OUTCOMES:
                raise ValueError(f"非法判定：{outcome or '空'}")
            raw_ids = entry.get("debt_ids")
            if not isinstance(raw_ids, list) or not raw_ids:
                raise ValueError("check 缺少 debt_ids")
            debt_ids = [str(v) for v in raw_ids]
            unknown = [d for d in debt_ids if d not in registry.debts]
            if unknown:
                raise ValueError("check 引用不存在的债务：" + "、".join(unknown[:3]))
            evidence = str(entry.get("evidence") or "").strip()
            if not evidence:
                raise ValueError(f"check {item} 缺少 evidence")
            checks.append({"item": item, "outcome": outcome,
                           "debt_ids": debt_ids, "evidence": evidence})
        return checks

    @staticmethod
    def _parse_issues(value: Any, registry: DebtRegistry) -> list[dict[str, Any]]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("issues 必须是数组")
        issues: list[dict[str, Any]] = []
        for entry in value:
            if not isinstance(entry, dict):
                raise ValueError("issues 含非对象项")
            kind = str(entry.get("kind") or "").strip()
            if kind not in DEBT_ISSUE_KINDS:
                raise ValueError(f"非法 issue kind：{kind or '空'}")
            severity = str(entry.get("severity") or "medium").strip()
            severity = _SEVERITY_ALIASES.get(severity, severity)
            if severity not in SEVERITY_RANK:
                raise ValueError(f"非法 severity：{severity}")
            raw_ids = entry.get("debt_ids") or []
            if not isinstance(raw_ids, list):
                raise ValueError("issue debt_ids 必须是数组")
            debt_ids = [str(v) for v in raw_ids]
            unknown = [d for d in debt_ids if d not in registry.debts]
            if unknown:
                raise ValueError("issue 引用不存在的债务：" + "、".join(unknown[:3]))
            problem = str(entry.get("problem") or "").strip()
            if not problem:
                raise ValueError("issue 缺少 problem")
            issues.append({"kind": kind, "severity": severity, "debt_ids": debt_ids,
                           "problem": problem,
                           "evidence": str(entry.get("evidence") or "").strip()})
        return issues

    # ---- 程序门禁与循环状态 ----

    @staticmethod
    def _accumulate(accum: dict[tuple[str, str], tuple[str, str]],
                    checks: list[dict[str, Any]]) -> None:
        for check in checks:
            for did in check["debt_ids"]:
                accum[(did, check["item"])] = (check["outcome"], check["evidence"])

    @staticmethod
    def _purge(accum: dict[tuple[str, str], tuple[str, str]],
               registry: DebtRegistry) -> None:
        for key in [k for k in accum if k[0] not in registry.debts]:
            del accum[key]

    @staticmethod
    def _program_gaps(registry: DebtRegistry, graph: SemanticGraph,
                      accum: dict[tuple[str, str], tuple[str, str]],
                      ) -> list[dict[str, str]]:
        """程序门禁：validate gaps（high）+ 每债务 × 六检查的覆盖与判定
        （fail=high，unknown/未覆盖=medium）。"""
        gaps = [{"severity": "high", "detail": detail}
                for detail in registry.validate(graph)]
        for did in registry.order:
            for item in DEBT_CHECK_ITEMS:
                outcome, evidence = accum.get((did, item), ("unknown", ""))
                if outcome == "pass":
                    continue
                gaps.append({
                    "severity": "high" if outcome == "fail" else "medium",
                    "detail": f"债务 {did} 准入门禁 {item}={outcome}"
                              + (f"（{evidence[:80]}）" if evidence else ""),
                })
        return gaps

    @staticmethod
    def _max_severity(gaps: list[dict[str, str]]) -> int:
        return max((SEVERITY_RANK.get(g["severity"], 0) for g in gaps), default=0)

    @staticmethod
    def _digest(registry: DebtRegistry) -> str:
        material = json.dumps({"debts": registry.to_dict()["debts"],
                               "order": registry.order},
                              ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _action_sig(action: DebtRepairAction) -> str:
        return action.op + "|" + json.dumps(action.params, ensure_ascii=False,
                                            sort_keys=True)
