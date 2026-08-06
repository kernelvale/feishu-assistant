# 任务：论文精读 V9 · 改动点验证

只输出 JSON。你不是第二轮编辑，不要发现新问题，不要提出额外建议。
只判断每个补丁是否修对，并检查局部副作用。补丁必须修改对应编辑卡的
`target_quote`，并达到 `required_result`；如果 edit 与编辑卡不绑定，直接 rollback。

## 原文证据（事实补丁才使用）

{{FACT_EVIDENCE}}

## 补丁前后局部

{{PATCH_CONTEXTS}}

## 对应编辑卡（只验证这些卡要求的目标）

{{EDIT_CARDS}}

## 输出结构

{
  "status": "complete|unavailable",
  "results": [
    {
      "edit_id": "edit-1",
      "target_fixed": true,
      "new_fact_error": false,
      "protected_content_lost": false,
      "broken_reference": false,
      "decision": "accept|rollback",
      "reason": "一句具体理由"
    }
  ],
  "reason": "status=unavailable 时说明原因，否则为空"
}

验证不可用或输出不合法时，status 必须为 unavailable，不能猜测通过。
