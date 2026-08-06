# 任务：论文精读 V9 · 真实读者审查

只输出 JSON。把自己当成第一次接触这篇论文、只懂 LLM 基础的研究生。
你只看文章本身，不看论文原文、Narrative Brief、Debt Registry、章节合同或其他审核意见。
不要按模板打分，也不要用工程或审核黑话。

## 待读文章

{{BODY}}

<!--CACHE_BREAK-->

## 机械候选（只作为定位线索，不等于问题）

{{CANDIDATES}}

## 语义落点清单（只核验是否自然推出）

{{TAKEAWAY_CHECKLIST}}

其中 `takeaway_checklist` 只用于检查“读者是否自然得到 Brief 预设的语义落点”；它不是合同，也不要求正文逐字出现这些句子。

## 检查问题

1. 读到文章前段时，是否知道论文真正要解决什么问题和关键转向？
2. 核心机制出现时，必要的前置知识是否已经自然建立？
3. 是否有同一个认知被完整解释两次？
4. 是否出现“前文已经建立/前面已经说明”，但前文实际上没有建立？
5. 图表、数字或结论是否被重复搬运，却没有新解释功能？
6. 哪些段落明显像规划合同或审核报告，而不像自然讲解？
7. 对 `takeaway_checklist` 中的 medium/deep beat，正文是否通过具体机制、证据或边界自然推出了对应判断？如果只罗列事实、只复述步骤，或判断与证据脱节，且能定位到具体段落，报告 `missing_takeaway`；不要因为没有出现原句就报告。

## 重复的判定规则

只有“同一认知 + 同一阅读功能”才算有害重复。
同一概念在后文用于新证据、新比较或新局限时可以保留，但回顾必须压缩。
每条重复意见必须给出两段精确片段、保留哪一段，以及删除/合并/压缩操作。
`span_a`、`span_b` 必须是从“待读文章”中逐字复制的连续原文片段，不能概括、改写、拼接或补标点；非重复问题的 `span_b` 为空。输出前逐条检查：正文中每个非空 span 必须能用字符串精确找到且只出现一次。若无法逐字定位，不要报告该问题。

## 输出结构

{
  "status": "complete|unavailable",
  "issues": [
    {
      "issue_id": "reader-1",
      "severity": "P1|P2",
      "kind": "duplication|broken_reference|fragmentation|mechanical_tone|figure_overuse|missing_bridge|missing_takeaway",
      "span_a": "精确片段 A",
      "span_b": "精确片段 B；非重复问题可为空",
      "keep": "a|b|none",
      "operation": "delete|merge|shorten|move|bridge",
      "reader_harm": "读者为什么会被影响",
      "required_outcome": "修改后应保留什么新的阅读功能",
      "max_scope": "sentence|paragraph|adjacent_paragraphs"
    }
  ],
  "reason": "status=unavailable 时说明原因，否则为空"
}

最多报告 5 条最影响理解的问题。没有明确定位的问题不要报告。
