"""统一问题/修复/台账基础设施（V8 第 4 步）：ReviewIssue、RepairAction v2、
IssueLedger、RepairRegistry 路由表与程序保证函数。

设计契约（与 V8 规划裁定一致）：
- 本模块是统一 schema 与路由层：只定义问题/修复的数据结构、状态机与路由表；
  op 的实际执行仍由 reading_repair.RepairEngine（Plan 结构四操作）与
  reading_debts.DebtRepairEngine（债务七操作）承担，本模块不重复实现执行逻辑。
- issue_id 由程序计算（stage+kind+root_scope+sorted(target_ids)+violated_invariant），
  同一问题重复报告同 ID——IssueLedger 幂等去重靠它；措辞/problem 不参与身份。
- 禁静默丢失：任何 register 过的 issue 在流程结束必须到达终态
  （resolved | degraded_with_reason）；open 以外的迁移一律拒绝。
- 过渡期兼容：各审核器仍输出自由 dict，经 ReviewIssue.from_legacy 归一入册；
  判定逻辑第 5/6 步才改引 checks 协议与 debt_id/block_id。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .reading_debts import DEBT_ISSUE_KINDS, DEBT_REPAIR_OPS, SEVERITY_RANK
from .reading_graph import SemanticGraph, stable_id

ISSUE_LEDGER_SCHEMA = "issue-ledger-v1"

# 审核阶段：Plan 审核 | 单元（节/层）局审与补丁复核 | 图表讲解验收 | 发布前终审
ISSUE_STAGES = ("plan", "unit", "figure", "final")
# 问题根因落点：语义图 | 计划（含债务注册表）| 正文 | 资产（图表/公式）
ROOT_SCOPES = ("graph", "plan", "body", "asset")
# 严重度（与 reading_debts.SEVERITY_RANK 同口径，单一事实源）
SEVERITIES = tuple(SEVERITY_RANK)

# 审核类 issue kind（债务八类从 reading_debts 引入统一，见 ISSUE_KINDS）
REVIEW_ISSUE_KINDS = (
    "structural_duplication",  # 结构重复：同一债务被多处偿还
    "fact_error",              # 事实错误：与原文不符
    "number_inconsistency",    # 数字/读数不一致
    "strong_causality",        # 强因果：超出原文证据的因果断言
    "scope_violation",         # 越界：完成了他人合同的内容
    "contract_gap",            # 合同缺口：应偿还的债务/覆盖义务未兑现
    "debt_uncovered",          # 债务无 owner：Plan 未分配偿还单元
    "debt_duplicated",         # 两 debt 实同：语义相同的重复债务
    "boundary_fail",           # 层间边界审核 fail（第 5 步 boundary 门禁）
    "preparation_illegal",     # preparations 重复建立/欠分配/下放错误
    "figure_misread",          # 图表误读：读数/比较/结论与图不符
    "format_gap",              # 格式缺口：可机械修复（无模型）
    "review_unavailable",      # 审核不可用：标注交付，不属于程序错误
    "general",                 # 过渡期兜底：无法归类的问题
)
# 全量 kind 枚举：审核类 + 债务八类（路由表对每类必须有确定路由）
ISSUE_KINDS = REVIEW_ISSUE_KINDS + DEBT_ISSUE_KINDS

# 台账状态机：open → resolved | degraded_with_reason（禁其他迁移）
LEDGER_STATUSES = ("open", "resolved", "degraded_with_reason")

# stage → 默认 root_scope（from_legacy 归一用；显式 root_scope 优先）
STAGE_DEFAULT_SCOPE = {"plan": "plan", "unit": "body", "figure": "asset", "final": "body"}

# 统一修复操作全集：v1 Plan 结构四操作 + 债务七操作 + 后续步骤的拓扑/正文/图/资产操作
UNIFIED_REPAIR_OPS = (
    # reading_repair.RepairEngine 执行（白名单见 SUPPORTED_OPS）
    "reassign_preparation", "remove_preparation",
    "update_layer_mode", "update_unit_fields",
    # reading_debts.DebtRepairEngine 执行（七操作覆盖债务八类 issue）
    *DEBT_REPAIR_OPS,
    # 第 7 步：Plan 拓扑修复（承担所有合并/拆分/重排）
    "replace_plan_fragment",
    # 第 6 步：正文块级操作（block manifest 落地后执行）
    "block_patch", "delete_blocks", "move_blocks",
    # 第 6 步：图级修复（证据重绑/局部重抽取，有界一次）
    "rebind_source_spans", "reextract_graph_fragment",
    # 资产操作：重做图表讲解
    "redo_figure",
    # 程序机械修复（无模型）：占位符/标记/标题归一
    "mechanical_fix",
)

# RepairRegistry 路由表：(kind, root_scope) → 候选 op 模板（按子情形选择其一）。
# 空元组 = 无 op（review_unavailable 走标注交付，不属于程序错误）；
# ("*", kind) 为该 kind 的兜底 scope。每 issue kind（除 review_unavailable）
# 必须有确定路由；查表缺失即抛 IssueRouteError（程序错误，不容忍静默）。
REPAIR_ROUTES: dict[tuple[str, str], tuple[str, ...]] = {
    # 结构重复三分流（第 6 步）：①同一 debt 两 owner → canonicalize/reassign/plan 片段；
    # ②Plan 唯一 owner、Writer 越界 → 正文删/移块；③两 debt 实同（debt_duplicated）→ 债务合并
    ("structural_duplication", "plan"): (
        "canonicalize_debt", "reassign_debt_owner", "replace_plan_fragment"),
    ("structural_duplication", "body"): ("delete_blocks", "move_blocks"),
    # 两 debt 实同：审核阶段带 creates 走 replace_debt_fragment 合并；
    # 第 6 步正文回流无模型供给 creates，机械合并走 canonicalize_debt（保留一条退休其余）
    ("debt_duplicated", "plan"): ("canonicalize_debt", "replace_debt_fragment"),
    # 正文语义错误 → 块级补丁一次
    ("fact_error", "body"): ("block_patch",),
    ("number_inconsistency", "body"): ("block_patch",),
    ("strong_causality", "body"): ("block_patch",),
    ("scope_violation", "body"): ("block_patch",),
    # 事实错误的根因在图（抽取错/证据绑错）→ 证据重绑或局部重抽取
    ("fact_error", "graph"): ("rebind_source_spans", "reextract_graph_fragment"),
    # 计划合同缺口 / 债务无 owner / 边界 fail → Plan 片段替换
    ("contract_gap", "plan"): ("replace_plan_fragment",),
    ("debt_uncovered", "plan"): ("replace_plan_fragment",),
    ("boundary_fail", "plan"): ("replace_plan_fragment",),
    # preparations 违法 → 改派/移除（v1 白名单操作）
    ("preparation_illegal", "plan"): ("reassign_preparation", "remove_preparation"),
    # 图表误读 → 重做该图讲解
    ("figure_misread", "asset"): ("redo_figure",),
    # 格式缺口 → 程序机械修复（无模型）
    ("format_gap", "body"): ("mechanical_fix",),
    # 审核不可用 → 无 op：标注交付（重写 0 次），不属于程序错误
    ("review_unavailable", "*"): (),
    # 债务八类 → 七操作（与 reading_debts 第 2 步路由一致）
    ("duplicate_debt", "plan"): ("canonicalize_debt",),
    ("overbroad_debt", "plan"): ("replace_debt_fragment",),
    ("fragmented_debt", "plan"): ("replace_debt_fragment",),
    ("wrong_granularity", "plan"): ("replace_debt_fragment",),
    ("missing_debt", "plan"): ("add_debt_from_graph",),
    ("unfounded_debt", "plan"): ("rebind_debt_evidence", "retire_debt"),
    ("wrong_debt_kind", "plan"): ("update_debt_relation",),
    ("wrong_owner", "plan"): ("reassign_debt_owner",),
    # 过渡期兜底：general 按落点给默认通道
    ("general", "plan"): ("replace_plan_fragment",),
    ("general", "body"): ("block_patch",),
    ("general", "asset"): ("redo_figure",),
    ("general", "graph"): ("rebind_source_spans", "reextract_graph_fragment"),
}

# op → 必须与 target_ids 一致的 params 键（值可为字符串或字符串数组）。
# remove_preparation 的 concept_id 是图节点而非单元，不在此列（target 是 owner 节）。
_OP_TARGET_PARAMS: dict[str, tuple[str, ...]] = {
    "reassign_preparation": ("from_section_id", "to_section_id"),
    "update_layer_mode": ("layer_id",),
    "update_unit_fields": ("unit_id",),
    "replace_plan_fragment": ("target_ids",),
    "canonicalize_debt": ("keep_debt_id", "retire_debt_ids"),
    "replace_debt_fragment": ("retire_debt_ids",),
    "rebind_debt_evidence": ("debt_id",),
    "retire_debt": ("debt_id",),
    "update_debt_relation": ("debt_id",),
    "reassign_debt_owner": ("debt_id",),
    "block_patch": ("block_ids",),
    "delete_blocks": ("block_ids",),
    "move_blocks": ("block_ids",),
    "redo_figure": ("label",),
}

# 语义修复证据键：带这些参数的 op 必须引用真实图节点/span（check_action_evidence）
_EVIDENCE_NODE_KEYS = (
    "node_id", "node_ids", "concept_id", "concept_ids",
    "assertion_id", "assertion_ids", "primary_assertion_id",
    "supporting_node_ids", "prerequisite_concept_ids",
    "subject_node_id", "object_node_ids", "premise_node_ids",
)
_EVIDENCE_SPAN_KEYS = ("span_id", "span_ids", "source_span_ids", "evidence_span_ids")

_SEVERITY_ALIASES = {"高": "high", "中": "medium", "低": "low"}


class IssueRouteError(Exception):
    """issue kind 无确定修复路由：程序错误，不容忍静默兜底。"""


def _str_list(value: Any) -> list[str]:
    return [str(v).strip() for v in value or [] if str(v).strip()]


def compute_issue_id(
    stage: str,
    kind: str,
    root_scope: str,
    target_ids: Iterable[str] = (),
    violated_invariant: str = "",
) -> str:
    """问题身份：同一问题重复报告同 ID（ledger 幂等去重的事实源）。

    措辞/problem/debt_ids/node_ids/severity 不参与身份；target_ids 内部顺序
    不参与（sorted 后入哈希）。"""
    material = "|".join([
        str(stage).strip(),
        str(kind).strip(),
        str(root_scope).strip(),
        ",".join(sorted(str(v).strip() for v in target_ids)),
        str(violated_invariant).strip(),
    ])
    return stable_id("issue", material)


@dataclass
class ReviewIssue:
    """一条审核问题：全字段结构化，跨 Plan/单元/图表/终审统一。"""

    issue_id: str = ""                 # 程序计算的身份（为空时 __post_init__ 计算）
    stage: str = ""                    # ISSUE_STAGES 之一
    kind: str = ""                     # ISSUE_KINDS 之一
    root_scope: str = ""               # ROOT_SCOPES 之一
    severity: str = "medium"           # SEVERITIES 之一
    target_ids: list[str] = field(default_factory=list)      # 落点：unit/block/figure 等
    debt_ids: list[str] = field(default_factory=list)        # 涉及债务（第 6 步接线）
    body_block_ids: list[str] = field(default_factory=list)  # 涉及正文块（第 6 步接线）
    node_ids: list[str] = field(default_factory=list)        # 涉及图节点
    source_span_ids: list[str] = field(default_factory=list)  # 涉及原文 span
    violated_invariant: str = ""       # 被违反的不变量（参与身份）
    problem: str = ""                  # 问题描述（含旧 location，不参与身份）

    def __post_init__(self) -> None:
        if self.stage not in ISSUE_STAGES:
            raise ValueError(f"非法 stage：{self.stage or '空'}")
        if self.kind not in ISSUE_KINDS:
            raise ValueError(f"非法 issue kind：{self.kind or '空'}")
        if self.root_scope not in ROOT_SCOPES:
            raise ValueError(f"非法 root_scope：{self.root_scope or '空'}")
        if self.severity not in SEVERITIES:
            raise ValueError(f"非法 severity：{self.severity or '空'}")
        if not self.issue_id:
            self.issue_id = compute_issue_id(
                self.stage, self.kind, self.root_scope,
                self.target_ids, self.violated_invariant)

    @classmethod
    def from_legacy(
        cls,
        item: dict[str, Any],
        stage: str,
        *,
        kind_override: str | None = None,
        default_target_ids: Iterable[str] = (),
    ) -> "ReviewIssue":
        """旧自由 dict（location/problem/severity 中文）→ 归一 ReviewIssue。

        映射口径：severity 高/中/低→high/medium/low（缺省按 high，与局审闸门
        「缺省 severity 按高处理」同口径）；problem 含「结构重复：」→
        kind=structural_duplication 且 severity 硬判 high（模型降级无效，与
        _local_review_passed 的硬规则一致）；kind 显式合法则保留，非法/缺失
        归 general 并把原文 kind 记入 violated_invariant；location 保留进
        problem 文本（不参与身份）。"""
        if not isinstance(item, dict):
            raise ValueError("legacy issue 必须是对象")
        problem = str(item.get("problem") or "").strip()
        location = str(item.get("location") or "").strip()
        if location:
            problem = f"{location}：{problem}" if problem else location
        structural = "结构重复：" in problem
        raw_kind = str(item.get("kind") or "").strip()
        if kind_override is not None:
            kind = kind_override
        elif raw_kind in ISSUE_KINDS:
            kind = raw_kind
        elif structural:
            kind = "structural_duplication"
        else:
            kind = "general"
        raw_severity = str(item.get("severity") or "").strip()
        severity = _SEVERITY_ALIASES.get(raw_severity, raw_severity)
        if severity not in SEVERITIES:
            severity = "high"  # 缺省/未知严重度按高处理（保守，防漏标绕过闸门）
        if structural:
            severity = "high"  # 结构重复硬判高严重度
        scope = str(item.get("root_scope") or "").strip()
        if scope not in ROOT_SCOPES:
            scope = STAGE_DEFAULT_SCOPE.get(stage)
        if scope is None:
            raise ValueError(f"非法 stage：{stage or '空'}")
        target_ids = _str_list(item.get("target_ids"))
        scalar_target = str(item.get("target_id") or "").strip()
        if scalar_target and scalar_target not in target_ids:
            target_ids.append(scalar_target)
        for tid in default_target_ids:
            tid = str(tid).strip()
            if tid and tid not in target_ids:
                target_ids.append(tid)
        invariant = str(item.get("violated_invariant") or "").strip()
        if not invariant and raw_kind and raw_kind not in ISSUE_KINDS:
            invariant = raw_kind  # 旧 kind 文本多为被违反的规则名，保留可审计
        return cls(
            stage=stage, kind=kind, root_scope=scope, severity=severity,
            target_ids=target_ids,
            debt_ids=_str_list(item.get("debt_ids")),
            body_block_ids=_str_list(item.get("body_block_ids")),
            node_ids=_str_list(item.get("node_ids")),
            source_span_ids=_str_list(item.get("source_span_ids")),
            violated_invariant=invariant,
            problem=problem,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReviewIssue":
        if not isinstance(data, dict):
            raise ValueError("ReviewIssue 必须是对象")
        return cls(
            issue_id=str(data.get("issue_id") or "").strip(),
            stage=str(data.get("stage") or "").strip(),
            kind=str(data.get("kind") or "").strip(),
            root_scope=str(data.get("root_scope") or "").strip(),
            severity=str(data.get("severity") or "medium").strip(),
            target_ids=_str_list(data.get("target_ids")),
            debt_ids=_str_list(data.get("debt_ids")),
            body_block_ids=_str_list(data.get("body_block_ids")),
            node_ids=_str_list(data.get("node_ids")),
            source_span_ids=_str_list(data.get("source_span_ids")),
            violated_invariant=str(data.get("violated_invariant") or "").strip(),
            problem=str(data.get("problem") or "").strip(),
        )


def compute_action_id(op: str, target_ids: Iterable[str], params: dict[str, Any]) -> str:
    """修复动作身份：同一 op + 同一批目标 + 同一参数 = 同一动作（防重复执行对账用）。"""
    material = "|".join([
        str(op).strip(),
        ",".join(sorted(str(v).strip() for v in target_ids)),
        json.dumps(params or {}, ensure_ascii=False, sort_keys=True, default=str),
    ])
    return stable_id("action", material)


@dataclass
class RepairAction:
    """统一修复动作 v2：在 v1 基础上增 action_id（稳定哈希）与
    resolves_issue_ids（必填非空：每个动作必须声明它偿还哪些 issue）。

    op 白名单不在本层校验：各 Engine（RepairEngine/DebtRepairEngine）各自
    拒绝不支持的操作；本层是跨 Engine 的统一载体。"""

    action_id: str = ""                    # 程序计算（为空时 __post_init__ 计算）
    resolves_issue_ids: list[str] = field(default_factory=list)  # 必填非空
    op: str = ""
    target_ids: list[str] = field(default_factory=list)
    evidence_node_ids: list[str] = field(default_factory=list)
    expected_plan_revision: int = 0
    expected_revisions: dict[str, int] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.resolves_issue_ids = _str_list(self.resolves_issue_ids)
        if not self.resolves_issue_ids:
            raise ValueError("RepairAction v2 缺少 resolves_issue_ids（必填非空）")
        if not str(self.op or "").strip():
            raise ValueError("RepairAction v2 缺少 op")
        if not self.action_id:
            self.action_id = compute_action_id(self.op, self.target_ids, self.params)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RepairAction":
        """兼容解析：v2（resolves_issue_ids 数组）与 v1（issue_id 单值）都接受；
        两者合并去重后仍为空则拒收。"""
        if not isinstance(data, dict):
            raise ValueError("RepairAction 必须是对象")
        resolves = _str_list(data.get("resolves_issue_ids"))
        legacy = str(data.get("issue_id") or "").strip()
        if legacy and legacy not in resolves:
            resolves.append(legacy)
        return cls(
            action_id=str(data.get("action_id") or "").strip(),
            resolves_issue_ids=resolves,
            op=str(data.get("op") or "").strip(),
            target_ids=_str_list(data.get("target_ids")),
            evidence_node_ids=_str_list(data.get("evidence_node_ids")),
            expected_plan_revision=int(data.get("expected_plan_revision") or 0),
            expected_revisions={str(k): int(v) for k, v in
                                (data.get("expected_revisions") or {}).items()},
            params=dict(data.get("params") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def route(issue: ReviewIssue) -> tuple[str, ...] | None:
    """RepairRegistry 路由：按 (kind, root_scope) 查表返回候选 op 模板。

    返回 None 表示无 op（review_unavailable：走标注交付，不属于程序错误）。
    除 review_unavailable 外每 issue kind 必须有确定路由；查表缺失即抛
    IssueRouteError（程序错误，不容忍静默兜底）。"""
    ops = REPAIR_ROUTES.get((issue.kind, issue.root_scope))
    if ops is None:
        ops = REPAIR_ROUTES.get((issue.kind, "*"))
    if ops is None:
        raise IssueRouteError(
            f"issue kind 无确定修复路由：{issue.kind}@{issue.root_scope}")
    return ops or None


@dataclass
class LedgerEntry:
    """台账条目：issue + 终态机状态 + 审计字段（谁偿还/为何降级）。"""

    issue: ReviewIssue
    status: str = "open"          # LEDGER_STATUSES 之一
    resolved_by: str = ""         # mark_resolved 的 by_action_id
    degraded_reason: str = ""     # mark_degraded 的原因（必填非空）

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue": self.issue.to_dict(),
            "status": self.status,
            "resolved_by": self.resolved_by,
            "degraded_reason": self.degraded_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LedgerEntry":
        status = str(data.get("status") or "open").strip()
        if status not in LEDGER_STATUSES:
            raise ValueError(f"非法台账状态：{status or '空'}")
        return cls(
            issue=ReviewIssue.from_dict(dict(data.get("issue") or {})),
            status=status,
            resolved_by=str(data.get("resolved_by") or ""),
            degraded_reason=str(data.get("degraded_reason") or ""),
        )


class IssueLedger:
    """统一问题台账：register 幂等去重 + 终态机 + 落盘。

    禁静默丢失：任何 register 过的 issue 在流程结束必须到达终态
    （resolved | degraded_with_reason）；只有 open 允许迁移，其他迁移一律
    ValueError。本类不做任何模型调用。"""

    def __init__(self) -> None:
        self.entries: dict[str, LedgerEntry] = {}

    # ---- 入册 ----

    def register(self, issues: Iterable[ReviewIssue]) -> list[str]:
        """按 issue_id 幂等去重入册（已入册的不覆盖、不重置终态）。

        返回本次输入对应的 issue_id 列表（含已入册的，供调用方对账）。"""
        ids: list[str] = []
        for issue in issues:
            if not isinstance(issue, ReviewIssue):
                raise ValueError("register 只接受 ReviewIssue")
            if issue.issue_id not in self.entries:
                self.entries[issue.issue_id] = LedgerEntry(issue=issue)
            ids.append(issue.issue_id)
        return ids

    # ---- 终态迁移（只有 open→resolved|degraded_with_reason）----

    def mark_resolved(self, issue_ids: Iterable[str], by_action_id: str) -> None:
        if not str(by_action_id or "").strip():
            raise ValueError("mark_resolved 必须携带 by_action_id（审计追溯）")
        for issue_id in dict.fromkeys(issue_ids):
            entry = self._require(issue_id)
            if entry.status != "open":
                raise ValueError(
                    f"非法状态迁移：{entry.status}→resolved（{issue_id}）")
            entry.status = "resolved"
            entry.resolved_by = str(by_action_id).strip()

    def mark_degraded(self, issue_ids: Iterable[str], reason: str) -> None:
        reason = str(reason or "").strip()
        if not reason:
            raise ValueError("mark_degraded 必须携带原因（degraded_with_reason）")
        for issue_id in dict.fromkeys(issue_ids):
            entry = self._require(issue_id)
            if entry.status != "open":
                raise ValueError(
                    f"非法状态迁移：{entry.status}→degraded_with_reason（{issue_id}）")
            entry.status = "degraded_with_reason"
            entry.degraded_reason = reason

    def _require(self, issue_id: str) -> LedgerEntry:
        entry = self.entries.get(issue_id)
        if entry is None:
            raise ValueError(f"未注册的 issue 不得迁移状态：{issue_id}")
        return entry

    # ---- 查询 ----

    def open_issues(self, stage: str | None = None) -> list[ReviewIssue]:
        """仍 open 的 issue；stage 给定时只返回该阶段的。"""
        return [
            entry.issue for entry in self.entries.values()
            if entry.status == "open" and (stage is None or entry.issue.stage == stage)
        ]

    def unterminated(self) -> list[ReviewIssue]:
        """流程结束必须为空：任何 register 过的 issue 不得停留在 open。"""
        return self.open_issues()

    def all_terminal(self) -> bool:
        return all(entry.status != "open" for entry in self.entries.values())

    def summary(self) -> dict[str, int]:
        """终态汇总（metadata 透出）：total/resolved/degraded/unterminated。"""
        resolved = sum(1 for e in self.entries.values() if e.status == "resolved")
        degraded = sum(
            1 for e in self.entries.values() if e.status == "degraded_with_reason")
        open_count = len(self.entries) - resolved - degraded
        return {
            "total": len(self.entries),
            "resolved": resolved,
            "degraded": degraded,
            "unterminated": open_count,
        }

    # ---- 持久化 ----

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ISSUE_LEDGER_SCHEMA,
            "entries": [entry.to_dict() for entry in self.entries.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IssueLedger":
        if str(data.get("schema")) != ISSUE_LEDGER_SCHEMA:
            raise ValueError(f"issue ledger schema 不兼容：{data.get('schema')}")
        ledger = cls()
        for raw in data.get("entries") or []:
            entry = LedgerEntry.from_dict(dict(raw))
            if entry.issue.issue_id in ledger.entries:
                raise ValueError(f"台账落盘存在重复 issue_id：{entry.issue.issue_id}")
            ledger.entries[entry.issue.issue_id] = entry
        return ledger


# ---------------------------------------------------------------------------
# 程序保证函数（供各审核循环调用；违反返回 gaps，调用方走 correction/降级，不判死）
# ---------------------------------------------------------------------------

def validate_review_outcome(
    status: Any,
    checks: Any,
    issues: Any,
    actions: Any,
    *,
    touched_ids: Iterable[str] | None = None,
) -> list[str]:
    """审核输出一致性校验：返回 gaps 列表（空 = 通过），绝不抛异常。

    口径：
    - clean ⇔ 所有带显式 outcome 的 check 全 pass 且 issues 空且不得携带 actions；
      fail/unknown check 不得 clean（无 outcome 键的旧式 checks 不由本函数评判，
      其举证门禁仍由各审核器既有程序校验承担）。
    - repairable ⇒ 每个可识别 issue 必有 resolves 它的 action；action 不得引用
      不存在的 issue；action.target_ids 非空且与 op 参数一致（_OP_TARGET_PARAMS）。
    - touched_ids（Engine 执行后比对）：执行实际触及的 ID 集合必须与各 action
      声明的 target_ids 并集一致（touched==target_ids）。
    issues/actions 接受 ReviewIssue/RepairAction 实例或原始 dict（dict 经
    from_dict 解析，解析失败记 gap）。"""
    gaps: list[str] = []
    status = str(status or "").strip()
    check_list = [c for c in (checks or []) if isinstance(c, dict)] if isinstance(checks, list) else []

    norm_issues: list[tuple[str, Any]] = []  # (可识别 issue_id 或 "", 原对象)
    for item in (issues or []) if isinstance(issues, list) else []:
        if isinstance(item, ReviewIssue):
            norm_issues.append((item.issue_id, item))
        elif isinstance(item, dict):
            norm_issues.append((str(item.get("issue_id") or "").strip(), item))

    norm_actions: list[RepairAction] = []
    for item in (actions or []) if isinstance(actions, list) else []:
        if isinstance(item, RepairAction):
            norm_actions.append(item)
        elif isinstance(item, dict):
            try:
                norm_actions.append(RepairAction.from_dict(item))
            except (ValueError, TypeError) as exc:
                gaps.append(f"修复动作协议非法：{exc}")

    if status == "clean":
        if norm_actions:
            gaps.append(f"clean 状态不得携带 repairs/actions（{len(norm_actions)} 个）")
        if norm_issues:
            gaps.append(f"clean 状态不得携带 issues（{len(norm_issues)} 条）")
        for check in check_list:
            outcome = check.get("outcome")
            if outcome is None:
                continue  # 旧式 checks（result 举证）不由本函数评判
            if str(outcome).strip() != "pass":
                gaps.append(
                    f"clean 但检查项 outcome={outcome}：{check.get('item') or '未命名'}")

    if status == "repairable":
        if not norm_actions:
            gaps.append("repairable 状态未给出任何修复动作")
        known_ids = {iid for iid, _ in norm_issues if iid}
        resolved_ids = {iid for action in norm_actions
                        for iid in action.resolves_issue_ids}
        if known_ids:
            for iid, _item in norm_issues:
                if iid and iid not in resolved_ids:
                    gaps.append(f"repairable 但 issue 无修复动作：{iid}")
            for action in norm_actions:
                unknown = [iid for iid in action.resolves_issue_ids
                           if iid not in known_ids]
                if unknown:
                    gaps.append(
                        f"修复动作 {action.action_id} 引用不存在的 issue："
                        + "、".join(unknown[:4]))
        for action in norm_actions:
            if not action.target_ids:
                gaps.append(f"修复动作 {action.action_id} target_ids 为空")
            gaps.extend(_op_target_gaps(action))

    if touched_ids is not None:
        gaps.extend(check_action_coverage(norm_actions, touched_ids))
    return gaps


def _op_target_gaps(action: RepairAction) -> list[str]:
    """op 参数与 target_ids 一致性：映射表内的 params 键值必须 ⊆ target_ids。"""
    gaps: list[str] = []
    for key in _OP_TARGET_PARAMS.get(action.op, ()):
        raw = action.params.get(key)
        values = [str(v).strip() for v in raw] if isinstance(raw, list) else [
            str(raw).strip()] if raw is not None else []
        values = [v for v in values if v]
        if not values:
            gaps.append(f"修复动作 {action.action_id}（{action.op}）缺少参数 {key}")
            continue
        outside = [v for v in values if v not in action.target_ids]
        if outside:
            gaps.append(
                f"修复动作 {action.action_id}（{action.op}）参数 {key} 与 target_ids "
                f"不一致：{'、'.join(outside[:4])}")
    return gaps


def check_action_coverage(
    actions: Iterable[RepairAction],
    touched_ids: Iterable[str],
) -> list[str]:
    """Engine 执行后比对：touched（实际触及）必须等于各 action 声明的 target_ids 并集。

     touched 多出 = 执行了未声明的修改（禁越权）；target_ids 多出 = 声明未生效
    （防空转）。返回 gaps（空 = 一致）。"""
    declared: set[str] = set()
    for action in actions:
        declared.update(action.target_ids)
    touched = set(touched_ids)
    gaps: list[str] = []
    undeclared = sorted(touched - declared)
    if undeclared:
        gaps.append("执行触及了未声明 target_ids 的对象：" + "、".join(undeclared[:6]))
    ineffective = sorted(declared - touched)
    if ineffective:
        gaps.append("声明的 target_ids 未生效（执行未触及）：" + "、".join(ineffective[:6]))
    return gaps


def check_action_evidence(action: RepairAction, graph: SemanticGraph) -> list[str]:
    """语义修复证据校验：凡带 node/span 参数的 op 必须引用真实图节点/span。

    覆盖 action.evidence_node_ids、params 顶层证据键，以及 creates[] 数组项
    内一层（债务拆/合与补全的新债务参数）。返回 gaps（空 = 通过）。"""
    gaps: list[str] = []
    for nid in action.evidence_node_ids:
        if nid not in graph.nodes and nid not in graph.spans:
            gaps.append(f"证据节点不存在于语义图：{nid}")

    def _collect(params: dict[str, Any]) -> None:
        for key in _EVIDENCE_NODE_KEYS:
            for nid in _param_values(params.get(key)):
                if nid not in graph.nodes:
                    gaps.append(f"修复参数 {key} 指向不存在的图节点：{nid}")
        for key in _EVIDENCE_SPAN_KEYS:
            for sid in _param_values(params.get(key)):
                if sid not in graph.spans:
                    gaps.append(f"修复参数 {key} 指向不存在的 span：{sid}")

    _collect(action.params)
    creates = action.params.get("creates")
    if isinstance(creates, list):
        for entry in creates:
            if isinstance(entry, dict):
                _collect(entry)
    return gaps


def _param_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []
