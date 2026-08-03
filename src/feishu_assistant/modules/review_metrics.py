"""精读文档的确定性排版诊断：只报告信号，不作为发布闸门。

与离线主观回归评审（scripts/review_doc.py 加载 review_student.md）互补：
- 这些指标回答"形式上达不达标"（可复现、可对比迭代前后）
- 主观评审回答"初学者读不读得懂"

指标清单（全部论文通用，不针对任何具体论文）：
- latex_rough：LaTeX 生肉计数（$...$ / \\命令{...}）
- para_walls：段落墙计数（单段 >5 个物理行）
- long_sentence_ratio：长句占比（>60 字的句子 / 全部句子）
- figure_note_total / figure_note_with_meaning：图表讲解数 / 含"意义"要素的比例
- en_common_words：该汉化的英文普通词命中数（黑名单，可扩充）
- subsections：三级小节的长度清单，仅帮助人工定位，不判断内容是否充分

论文类型、章节结构和教学合同都是动态的，不能从固定标题或关键词推断“讲清了”。
线上发布闸门由 reading_final_review 统一执行来源事实、初学者可读性、论文专属解释债务
与闭卷复述检查；review_student.md 刻意保留为发布后的独立离线评测，避免同一生成管线
同时运行两套高度重叠的模型闸门。
"""

from __future__ import annotations

import re

_LATEX_RE = re.compile(r"\$[^$\n]{2,}\$|\\[a-zA-Z]{2,}\{")
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?])")
_SUBSECTION_RE = re.compile(r"^###\s+(.+)$", re.MULTILINE)

# 有通行中文说法、却常被拿来充术语的英文普通词（黑名单按需扩充）
EN_COMMON_WORDS = [
    "ground truth", "checker", "pass/fail", "model-based", "sampling-based",
    "meta-reward", "credit assignment", "reward hacking", "false negative",
    "false positive", "rollout ", "warmup", "baseline model", "return ",
]


def _sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if len(p.strip()) >= 10]


def _subsections(text: str) -> list[tuple[str, int]]:
    """(三级标题, 正文字符数) 清单；图表总览条目不计入技术小节。"""
    matches = [m for m in _SUBSECTION_RE.finditer(text)
               if not re.match(r"(?:Figure|Table|图|表)\s*\d+", m.group(1), re.IGNORECASE)]
    sections: list[tuple[str, int]] = []
    for i, m in enumerate(matches):
        next_layer = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        next_h2_match = re.search(r"^##\s+", text[m.end():], re.MULTILINE)
        next_h2 = m.end() + next_h2_match.start() if next_h2_match else len(text)
        end = min(next_layer, next_h2)
        title = m.group(1).strip()
        sections.append((title, end - m.start()))
    return sections


def compute_metrics(text: str) -> dict:
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    walls = [p for p in paras if p.count("\n") >= 5]
    sentences = _sentences(text)
    long_sentences = [s for s in sentences if len(s) > 60]

    # 新图表讲解以固定教学问题开头，避免把正文里的“怎么读”误算成图表数。
    _ms = list(re.finditer(r"【要回答的问题】", text))
    note_blocks = [
        text[m.start(): (_ms[i + 1].start() if i + 1 < len(_ms) else m.start() + 400)]
        for i, m in enumerate(_ms)
    ]
    with_meaning = [b for b in note_blocks if re.search(
        r"意义|回答了什么|论证|角色|基准|立论|能说明什么|数据解释|现象解释|关键节点与流向",
        b,
    )]

    en_hits = sum(text.lower().count(w.lower()) for w in EN_COMMON_WORDS)
    subsections = _subsections(text)

    return {
        "latex_rough": len(_LATEX_RE.findall(text)),
        "para_walls": len(walls),
        "long_sentence_ratio": round(len(long_sentences) / max(1, len(sentences)), 3),
        "figure_note_total": len(note_blocks),
        "figure_note_with_meaning": f"{len(with_meaning)}/{len(note_blocks)}",
        "en_common_words": en_hits,
        "subsections": [{"title": t, "chars": c} for t, c in subsections],
    }


def format_metrics(m: dict) -> str:
    lines = [
        f"  LaTeX 生肉：{m['latex_rough']} 处",
        f"  段落墙（>5 行）：{m['para_walls']} 段",
        f"  长句占比（>60 字）：{m['long_sentence_ratio']:.1%}",
        f"  图表讲解：{m['figure_note_total']} 段，含意义要素 {m['figure_note_with_meaning']}",
        f"  英文普通词（黑名单）：{m['en_common_words']} 处",
        "  以下长度只用于定位，不能据此判定讲解是否充分：",
    ]
    for section in m["subsections"]:
        lines.append(f"    · {section['title'][:26]}：{section['chars']} 字")
    return "\n".join(lines)
