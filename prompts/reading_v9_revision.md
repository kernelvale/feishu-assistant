# 任务：论文精读 V9 · 同一作者的有限编辑

你仍是原文的唯一作者。不要重新生成全文，不要重写章节，不要输出解释。
只根据下面的编辑卡输出 JSON 补丁列表。每个补丁必须使用待修改正文中的精确 old_text，且补丁之间不能重叠。
优先删除、合并、移动和句子修正，最后才增加一句必要的桥接。

## 风格

{{STYLE}}

## 原始正文

{{BODY}}

## 已接受的编辑卡（不是原始审核意见）

{{EDIT_CARDS}}

## 原文证据（只为事实修正提供依据）

{{FACT_EVIDENCE}}

## 保护内容

不要删除核心机制、具体过程、关键反例、唯一证据数字或图表的主要解释，除非编辑卡明确定位它们。
不要引入规划黑话：debt、must_explain、success_test、root_scope、scope_violation、有限判断、证据支持什么、合同缺口。

## 输出结构

{
  "edits": [
    {
      "edit_id": "edit-1",
      "operation": "delete|replace|merge|shorten|bridge|move",
      "old_text": "正文中连续且唯一的原文片段",
      "new_text": "替换后的片段；删除时为空",
      "reason": "对应哪张编辑卡",
      "preserve": ["必须保留的事实或功能"]
    }
  ]
}

没有可靠修改时输出 {"edits": []}。不要返回整篇文章。
`reason` 必须包含对应编辑卡的 `card_id` 或其中的 `issue_ids`，`old_text` 必须逐字等于该卡的 `target_quote`；不能修改编辑卡之外的正文。
`preserve` 只能填写逐字存在于 `old_text`/原正文中的短片段；无法明确保留时填空数组，
不要把“应保留的功能”改写成原文不存在的概念名。
