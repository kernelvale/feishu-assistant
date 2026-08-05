"""连续状态台账（V8 第 8 步）：ContinuityLedger —— 真实 term/claim/state 依赖。

设计契约（与 V8 规划裁定一致）：
- 每篇一份 ContinuityLedger，落盘 workspace key `continuity-ledger`（json）。
- entry：{entry_id, kind(term|claim|state), statement, established_by(unit_id),
  established_revision}。entry_id = stable_id("cont", kind + 归一化 statement)：
  措辞即身份——同一表述换哪个单元建立都是同一条；established_revision 在条目
  消失后重建（含建立者易主）时 +1，消费者 manifest 边 `continuity:{entry_id}@revision`
  随之失配——生产者变化才传播失效，措辞不变的返工不惊动下游。
- 填充来源与旧 continuity dict 相同：每单元局审（未降级）review JSON 的
  continuity_update 字段。register_unit 原子替换该单元建立的全部条目，返回变化集
  {entry_id: 新 revision | 0（消失）}，供 DependencyManifest.dirty_closure 精准失效。
- pack：Writer 装包只注入与本单元有真实依赖的条目（真实依赖全注入、无依赖不注入）：
  条目 statement 的 bigram 被本单元探针（合同文本 + prerequisite_concept_ids /
  debt subject·object·premise 闭包 / 绑定节点的 statement）覆盖 ≥ 阈值即相关。
  返回（注入 dict, continuity_input_ids）——实际注入什么登记什么。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .reading_graph import stable_id

CONTINUITY_LEDGER_SCHEMA = "continuity-ledger-v1"

# 装包相关度阈值：条目 statement 的 bigram 被单元探针覆盖的比例（召回取向——
# 宁多勿漏，「真实依赖全注入」；探针覆盖不到的条目不注入，不是「尽量少注入」）
PACK_MIN_CONTAINMENT = 0.35
# 单次装包注入条目的总字符预算（防爆 prompt；足够装下全部真实依赖条目）
PACK_MAX_CHARS = 6_000

# review continuity_update 字段 → entry kind
_KIND_BY_FIELD = {
    "defined_terms": "term",
    "established_claims": "claim",
    "open_questions": "state",
}
# entry kind → 注入 dict 字段（与旧 continuity dict 同构，prompt 模板不变）
_FIELD_BY_KIND = {v: k for k, v in _KIND_BY_FIELD.items()}


def _norm(text: str) -> str:
    return re.sub(r"[^\w一-鿿]+", "", str(text or "")).casefold()


def _bigrams(text: str) -> set[str]:
    value = _norm(text)
    return {value[i:i + 2] for i in range(len(value) - 1)} or ({value} if value else set())


def _containment(short_text: str, long_text: str) -> float:
    """短文本 bigram 集合被长文本覆盖的比例（0~1）；空文本为 0。"""
    short, long = _bigrams(short_text), _bigrams(long_text)
    if not short or not long:
        return 0.0
    return len(short & long) / len(short)


@dataclass
class ContinuityEntry:
    """一条已建立的连续状态：哪个单元真实建立了哪个 term/claim/state。"""

    entry_id: str                    # stable_id("cont", kind + 归一化 statement)
    kind: str                        # term | claim | state
    statement: str
    established_by: str              # 建立者 unit_id
    established_revision: int = 1    # 消失后重建（含易主）+1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContinuityEntry":
        return cls(
            entry_id=str(data.get("entry_id") or ""),
            kind=str(data.get("kind") or ""),
            statement=str(data.get("statement") or ""),
            established_by=str(data.get("established_by") or ""),
            established_revision=int(data.get("established_revision") or 1),
        )


def entry_id_for(kind: str, statement: str) -> str:
    """条目身份：kind + 归一化 statement 的稳定哈希（措辞即身份）。"""
    return stable_id("cont", f"{kind}|{_norm(statement)}")


class ContinuityLedger:
    """连续状态台账：登记/查询前序单元真实建立的术语/论断/状态。"""

    def __init__(self) -> None:
        self._entries: dict[str, ContinuityEntry] = {}
        self._order: list[str] = []          # 建立顺序（装包保序）
        self._retired: dict[str, int] = {}   # 已消失条目的最后 revision（重建时 +1）

    # ---- 查询 ----

    def get(self, entry_id: str) -> ContinuityEntry | None:
        return self._entries.get(entry_id)

    def entries(self) -> list[ContinuityEntry]:
        """按建立顺序返回全部条目。"""
        return [self._entries[eid] for eid in self._order if eid in self._entries]

    def established_by(self, unit_id: str) -> list[ContinuityEntry]:
        return [e for e in self.entries() if e.established_by == unit_id]

    # ---- 登记 ----

    @staticmethod
    def _parse_update(update: Any) -> list[tuple[str, str]]:
        """review continuity_update → [(kind, statement)]（保序去重，非法项跳过）。"""
        out: list[tuple[str, str]] = []
        if not isinstance(update, dict):
            return out
        for field_name, kind in _KIND_BY_FIELD.items():
            values = update.get(field_name)
            if not isinstance(values, list):
                continue
            for value in values:
                text = str(value).strip()
                if text:
                    out.append((kind, text))
        return out

    def register_unit(self, unit_id: str, update: Any) -> dict[str, int]:
        """原子替换该单元建立的全部条目；返回变化集 {entry_id: 新 revision|0}。

        update 为 None（审核降级/缺失）视为该单元本轮无建立：旧条目全部撤下。
        变化集供 manifest dirty_closure 精准失效消费者；无变化返回空 dict。"""
        old_ids = [eid for eid in self._order
                   if self._entries[eid].established_by == unit_id]
        parsed = self._parse_update(update)
        new_ids: list[str] = []
        changed: dict[str, int] = {}
        # 先撤旧：建立者本轮没再声明的条目消失（消费者必须失效）
        keep: set[str] = set()
        for kind, statement in parsed:
            keep.add(entry_id_for(kind, statement))
        for eid in old_ids:
            if eid not in keep:
                entry = self._entries.pop(eid)
                self._order.remove(eid)
                self._retired[eid] = entry.established_revision
                changed[eid] = 0
        # 再登记：已存在（本单元未撤）原样保留；被其他单元建立的同身份条目不夺主；
        # 全新/重建条目 revision = 退休版本 + 1（易主/重建 → 消费者失配失效）
        for kind, statement in parsed:
            eid = entry_id_for(kind, statement)
            existing = self._entries.get(eid)
            if existing is not None:
                if existing.established_by == unit_id:
                    new_ids.append(eid)
                # 其他单元已建立同身份条目：首建者保留，本单元不重复登记
                continue
            revision = self._retired.pop(eid, 0) + 1
            self._entries[eid] = ContinuityEntry(
                entry_id=eid, kind=kind, statement=statement,
                established_by=unit_id, established_revision=revision,
            )
            self._order.append(eid)
            new_ids.append(eid)
            if revision > 1:
                changed[eid] = revision
        return changed

    # ---- 装包 ----

    def pack(self, probe_text: str, *, max_chars: int = PACK_MAX_CHARS
             ) -> tuple[dict[str, list[str]], list[tuple[str, int]]]:
        """Writer 装包：只注入探针覆盖的真实依赖条目。

        返回（与旧 continuity dict 同构的注入 dict, continuity_input_ids）；
        探针为空或无相关条目时注入空 dict（Writer 看到的即「无相关前序条目」）。"""
        injected: dict[str, list[str]] = {name: [] for name in _KIND_BY_FIELD}
        input_ids: list[tuple[str, int]] = []
        used = 0
        for entry in self.entries():
            if _containment(entry.statement, probe_text) < PACK_MIN_CONTAINMENT:
                continue
            if used + len(entry.statement) > max_chars:
                break  # 预算耗尽：早建立的条目优先（截断不判死）
            injected[_FIELD_BY_KIND[entry.kind]].append(entry.statement)
            input_ids.append((entry.entry_id, entry.established_revision))
            used += len(entry.statement)
        return injected, input_ids

    # ---- 持久化 ----

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTINUITY_LEDGER_SCHEMA,
            "entries": [self._entries[eid].to_dict() for eid in self._order
                        if eid in self._entries],
            "retired": dict(self._retired),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContinuityLedger":
        if str(data.get("schema")) != CONTINUITY_LEDGER_SCHEMA:
            raise ValueError(f"continuity-ledger schema 不兼容：{data.get('schema')}")
        ledger = cls()
        for raw in data.get("entries") or []:
            entry = ContinuityEntry.from_dict(dict(raw))
            if entry.entry_id:
                ledger._entries[entry.entry_id] = entry
                ledger._order.append(entry.entry_id)
        ledger._retired = {str(k): int(v)
                           for k, v in (data.get("retired") or {}).items()}
        return ledger


# ---------------------------------------------------------------------------
# 单元探针：装包相关度判定的事实源
# ---------------------------------------------------------------------------

# 合同文本字段（展示字段也计入：改名不影响 statement 匹配，命中即相关；
# must_explain/success_test 是债务派生文本，与之相关的条目同属真实依赖）
_CONTRACT_TEXT_KEYS = (
    "title", "name", "question", "why", "prerequisite", "input", "output",
    "purpose", "purposes", "mode_reason", "must_explain", "success_test",
)


def unit_probe_text(unit: Any, *, plan: Any = None, registry: Any = None,
                    graph: Any = None) -> str:
    """本单元装包探针：合同文本 + prerequisite/debt 闭包 + 绑定节点的 statement。

    探针越全召回越高（真实依赖全注入）：debt_ids → subject/object/premise 闭包、
    prerequisite_concept_ids、concept/assertion 绑定、preparations 义务文本。"""
    data = getattr(unit, "data", None) or {}
    parts: list[str] = []
    for key in _CONTRACT_TEXT_KEYS:
        value = data.get(key)
        if isinstance(value, list):
            parts.extend(str(v) for v in value if str(v).strip())
        elif str(value or "").strip():
            parts.append(str(value))
    # preparations 建立义务文本（本单元承担的 + 指向本单元的）
    preparations = getattr(plan, "preparations", None) or []
    unit_id = str(getattr(unit, "unit_id", "") or "")
    for prep in preparations:
        if not isinstance(prep, dict):
            continue
        if prep.get("owner_section_id") == unit_id \
                or unit_id in (prep.get("target_layer_ids") or []):
            text = str(prep.get("concept") or "").strip()
            if text:
                parts.append(text)
    # 节点 statement：绑定 + 合同直出 + 债务闭包
    node_ids: list[str] = [
        *(getattr(unit, "concept_ids", None) or []),
        *(getattr(unit, "assertion_ids", None) or []),
        *(str(v) for v in (data.get("prerequisite_concept_ids") or [])),
        *(str(v) for v in (data.get("supporting_node_ids") or [])),
    ]
    debts = getattr(registry, "debts", None) or {}
    for did in (data.get("debt_ids") or []):
        debt = debts.get(str(did))
        if debt is None:
            continue
        node_ids.append(str(getattr(debt, "subject_node_id", "") or ""))
        node_ids.extend(str(v) for v in (getattr(debt, "object_node_ids", None) or []))
        node_ids.extend(str(v) for v in (getattr(debt, "premise_node_ids", None) or []))
    nodes = getattr(graph, "nodes", None) or {}
    for nid in dict.fromkeys(nid for nid in node_ids if nid):
        node = nodes.get(nid)
        statement = str(getattr(node, "statement", "") or "").strip()
        if statement:
            parts.append(statement)
    return "\n".join(parts)
