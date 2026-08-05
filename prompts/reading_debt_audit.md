## 语义图目录（程序生成：node/span ID 的唯一合法来源）

{{GRAPH_CATALOG}}

## 债务证据附录（注册表所引用 span 的原文片段）

{{EVIDENCE_APPENDIX}}

<!--CACHE_BREAK-->

# 任务：债务准入审核（DebtAuditor）

下面是一份候选理解债务注册表（DebtRegistry）。你要逐项执行六条准入检查，
判定每条债务是否值得进入 Canonical 注册表。你的判决必须引用目录 ID 或
附录 span 原文作为证据。只输出一个 JSON 对象，不要任何前后缀或围栏。

{{AUDIT_SCOPE}}

## 六条准入检查（每条债务 × 每条检查都要有判定）

1. `source_backed`（有据）：subject/object/premise 是目录真实节点；
   evidence_span_ids 是真实 span，且附录中该 span 的原文确实支撑这条债务。
   反例：span 原文讲的是另一件事 → fail；附录未收录所引 span、无法核对
   → unknown。
2. `necessary`（必要）：necessity_reason 指向具体核心链——哪个核心
   assertion 的哪一步，删掉这条债务就断。反例：「有助于理解全文」
   「对后续有帮助」式空话 → fail；目录被截断、核心链不可见 → unknown。
3. `atomic`（原子）：一条债务只承担一个主要认知动作。success_test 含两个
   可独立失败的部分（会 A 不会 B、会 B 不会 A 都可能）→ 必须拆分，判 fail。
   反例：「能写出定义，并推导出三条性质」→ fail。
4. `non_duplicate`（不重复）：比身份与能力，不比文字——两条债务若偿还的是
   同一认知动作（只是措辞不同）→ 重复，判 fail 并指明保留哪条。正例（不算
   重复）：同一份证据支撑 relation 不同的两条债务（supported_by 与
   limited_by 回答的是不同问题）。
5. `testable`（可测）：success_test 闭卷可执行、结果可判定对错。反例：
   「理解本层」「掌握机制」「熟悉流程」→ fail。
6. `granularity`（粒度落地）：同一 assertion + relation + 认知动作 + 判据
   不得拆成多条；也不得把多个相互独立的认知动作塞进一条。粒度过粗或过细
   都判 fail。

判定取值：`pass`（明确通过）/ `fail`（明确违反）/ `unknown`（证据不足以
判定）。**fail 与 unknown 都不得当作通过**；拿不准就 unknown，禁止猜 pass。

## issues 协议（对每条 fail/unknown 给出结构化发现）

`kind` 八选一：

- `duplicate_debt`：语义重复（身份不同但认知动作相同）；
- `overbroad_debt`：过宽，一条债务装了多个认知动作；
- `fragmented_debt`：过碎，同一认知动作被拆成多条；
- `wrong_granularity`：粒度错误但不宜简单拆/合；
- `missing_debt`：核心链上存在真实理解难点但注册表缺失；
- `unfounded_debt`：无据——证据不支撑，或必要性落空；
- `wrong_debt_kind`：kind/relation 与真实语义关系不符（常伴随
  learner_action/success_test 需要同步重写）；
- `wrong_owner`：债务应归属的理解单元明显错位。

`severity`：high（核心链断裂级）/ medium（影响偿还质量）/ low（瑕疵）。

## repairs 协议（每个可修复 issue 给一个动作；无法修复就 retire）

- `canonicalize_debt`：params {keep_debt_id, retire_debt_ids}——语义重复时
  保留一条 canonical，退休其余；
- `replace_debt_fragment`：params {retire_debt_ids, creates[]}——拆/合，
  creates 是完整 DebtSpec 字段（真实 ID，给全 subject/object 或
  premise/evidence/learner_action/success_test/necessity_reason）；
- `add_debt_from_graph`：params {creates[]}——补缺失债务，必须给全真实
  subject/object/premise/span ID；
- `rebind_debt_evidence`：params {debt_id, evidence_span_ids}——换绑真实
  span 证据；
- `retire_debt`：params {debt_id}——无据且无法补救，退休；
- `update_debt_relation`：params {debt_id, kind, relation, learner_action,
  success_test}——修正关系类型，动作与判据同步更新；
- `reassign_debt_owner`：params {debt_id, owner_layer_id}——改归属单元。

每个 repair 必填 `expected_registry_revision`：取待审核注册表 JSON 中的
registry_revision 当前值。params 引用的 debt_id 必须在注册表中存在；
creates 引用的 node/span ID 必须来自目录/附录，禁止编造。

## 输出协议

```json
{
  "status": "clean | issues_found",
  "checks": [
    {"item": "atomic", "outcome": "fail",
     "debt_ids": ["debt-……"],
     "evidence": "success_test 含「写出定义」与「推导性质」两个可独立失败部分"}
  ],
  "issues": [
    {"kind": "overbroad_debt", "severity": "medium",
     "debt_ids": ["debt-……"], "problem": "一条债务承担定义与推导两个动作",
     "evidence": "注册表 learner_action 与 success_test 原文"}
  ],
  "repairs": [
    {"op": "replace_debt_fragment", "expected_registry_revision": 3,
     "params": {"retire_debt_ids": ["debt-……"], "creates": []}}
  ]
}
```

- `status=clean` 仅当全部 checks 为 pass 且 issues 为空；否则 `issues_found`。
- checks 可用一条覆盖多个 debt_ids，但注册表内每条债务 × 六条检查都必须
  有判定；缺判定会被程序视为输出不完整而要求重出。
- 没有可执行修复时 repairs 给空数组，禁止输出协议外操作。

## 待审核债务注册表

{{DEBT_REGISTRY_JSON}}
