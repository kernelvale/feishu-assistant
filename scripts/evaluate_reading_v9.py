#!/usr/bin/env python3
"""离线评估精读成稿与修订净效果，不调用模型、不上传论文。

示例：
  .venv/bin/python scripts/evaluate_reading_v9.py draft.md final.md
  .venv/bin/python scripts/evaluate_reading_v9.py --json draft.md final.md

本工具只提供机械与重复候选信号，不把分数冒充阅读质量。成稿替换仍需盲读判断。
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path
from typing import Any

from feishu_assistant.modules.reading_v9_quality import (
    duplicate_candidates,
    mechanical_quality_report,
)

JARGON = (
    "有限判断", "证据支持什么", "关键转向", "本章", "本层",
    "前文已经建立", "概念已在前文建立", "must_explain", "root_scope",
)


def _paragraphs(text: str) -> list[str]:
    return [
        part.strip() for part in re.split(r"\n\s*\n", text)
        if part.strip() and not part.lstrip().startswith("#")
    ]


def analyze(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    mechanical = mechanical_quality_report(text)
    duplicates = duplicate_candidates(text)
    paragraphs = _paragraphs(text)
    heading_counts = dict(mechanical.heading_counts)
    return {
        "path": str(path),
        "chars": len(text),
        "lines": text.count("\n") + 1,
        "paragraphs": len(paragraphs),
        "mean_paragraph_chars": round(
            sum(map(len, paragraphs)) / max(1, len(paragraphs)), 1
        ),
        "short_paragraphs_under_100": sum(len(item) < 100 for item in paragraphs),
        "h1": heading_counts.get(1, 0),
        "h2": heading_counts.get(2, 0),
        "h3": heading_counts.get(3, 0),
        "total_headings": sum(heading_counts.values()),
        "duplicate_candidates": len(duplicates),
        "top_duplicate_pairs": [item.to_dict() for item in duplicates[:5]],
        "planning_jargon_count": mechanical.planning_jargon_count,
        "untranslated_english_count": mechanical.untranslated_english_count,
        "untranslated_english_words": list(mechanical.untranslated_english_words),
        "placeholder_count": mechanical.placeholder_count,
        "mechanical_errors": [item.to_dict() for item in mechanical.errors],
        "mechanical_warnings": [item.to_dict() for item in mechanical.warnings],
        "jargon": {term: text.count(term) for term in JARGON if text.count(term)},
        "figure_mentions": {
            label: count for label, count in mechanical.figure_references if count
        },
    }


def compare(before: dict[str, Any], after: dict[str, Any], before_text: str, after_text: str) -> dict[str, Any]:
    matcher = difflib.SequenceMatcher(None, before_text, after_text, autojunk=False)
    changed = sum(
        max(i2 - i1, j2 - j1)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    )
    ratio = changed / max(1, len(before_text))
    regressions: list[str] = []
    if after["mechanical_errors"] and len(after["mechanical_errors"]) > len(before["mechanical_errors"]):
        regressions.append("新增机械错误")
    if after["placeholder_count"] > before["placeholder_count"]:
        regressions.append("新增占位符")
    if after["planning_jargon_count"] > before["planning_jargon_count"]:
        regressions.append("规划/审核黑话增加")
    if after["untranslated_english_count"] > before["untranslated_english_count"]:
        regressions.append("未翻译英文残留增加")
    if after["duplicate_candidates"] > before["duplicate_candidates"]:
        regressions.append("高相似重复候选增加")
    if after["total_headings"] > before["total_headings"] and after["total_headings"] > 18:
        regressions.append("标题碎片化加重")
    if ratio > 0.20:
        regressions.append("总改动比例超过 20%")
    return {
        "changed_chars_proxy": changed,
        "change_ratio": round(ratio, 6),
        "delta": {
            "chars": after["chars"] - before["chars"],
            "total_headings": after["total_headings"] - before["total_headings"],
            "duplicate_candidates": after["duplicate_candidates"] - before["duplicate_candidates"],
            "planning_jargon_count": after["planning_jargon_count"] - before["planning_jargon_count"],
            "untranslated_english_count": (
                after["untranslated_english_count"] - before["untranslated_english_count"]
            ),
            "placeholder_count": after["placeholder_count"] - before["placeholder_count"],
        },
        "deterministic_regressions": regressions,
        "safe_by_mechanical_gate": not regressions,
        "note": "机械门禁只检查已知回归；是否更好读仍需盲读。",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", help="输出完整 JSON")
    args = parser.parse_args()
    reports = [analyze(path) for path in args.paths]
    output: dict[str, Any] = {"documents": reports}
    if len(args.paths) == 2:
        before_text = args.paths[0].read_text(encoding="utf-8")
        after_text = args.paths[1].read_text(encoding="utf-8")
        output["comparison"] = compare(reports[0], reports[1], before_text, after_text)
    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    for item in reports:
        print(
            f"{item['path']}: chars={item['chars']} headings={item['total_headings']} "
            f"duplicates={item['duplicate_candidates']} jargon={item['planning_jargon_count']} "
            f"english={item['untranslated_english_count']} placeholders={item['placeholder_count']}"
        )
        if item["jargon"]:
            print("  jargon:", json.dumps(item["jargon"], ensure_ascii=False))
    if "comparison" in output:
        comp = output["comparison"]
        print("comparison:", json.dumps(comp, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
