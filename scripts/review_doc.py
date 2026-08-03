#!/usr/bin/env python3
"""初学者评审 agent：拉精读文档 → 模拟初学者研究生审查 → 输出结构化问题单。

用法：
    uv run python scripts/review_doc.py <doc_id_or_url> [--save]
    uv run python scripts/review_doc.py --arxiv 2509.08827   # 从论文库主记录取最新版文档
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feishu_assistant.config import load_settings
from feishu_assistant.feishu import build_client
from feishu_assistant.feishu.bitable import BitableDAO
from feishu_assistant.feishu.docx import DocxClient
from feishu_assistant.kimi import KimiClient, KimiError

REVIEW_PROMPT = (ROOT / "prompts" / "review_student.md").read_text(encoding="utf-8")
DOC_CAP = 60_000  # 评审输入预算（字符），超出截中段保头尾


def doc_id_from_arg(arg: str) -> str:
    m = re.search(r"docx/(\w+)", arg)
    return m.group(1) if m else arg


def load_doc_by_arxiv(dao: BitableDAO, arxiv_id: str) -> tuple[str, str]:
    fields = dao.find_record_fields("论文库", "arXivID", arxiv_id) or {}
    link = fields.get("文档链接")
    url = link.get("link", "") if isinstance(link, dict) else str(link or "")
    if not url:
        raise SystemExit(f"论文库里 {arxiv_id} 没有文档链接")
    return doc_id_from_arg(url), str(fields.get("版本") or "v?")


def cap_middle(text: str, cap: int = DOC_CAP) -> str:
    if len(text) <= cap:
        return text
    head = cap * 2 // 3
    return text[:head] + f"\n\n……（中段省略 {len(text) - cap} 字）……\n\n" + text[-(cap - head):]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", nargs="?", help="doc_id 或文档 URL")
    parser.add_argument("--arxiv", help="按 arXiv ID 从论文库取最新版")
    parser.add_argument("--save", action="store_true", help="存 data/reviews/")
    args = parser.parse_args()

    settings = load_settings()
    lark = build_client(settings)
    version = ""
    if args.arxiv:
        dao = BitableDAO(lark, settings.bitable_app_token, settings.bitable_tables)
        doc_id, version = load_doc_by_arxiv(dao, args.arxiv)
    elif args.target:
        doc_id = doc_id_from_arg(args.target)
    else:
        parser.error("给 doc_id/URL 或 --arxiv")

    text = DocxClient(lark).export_text(doc_id)
    print(f"📄 文档 {doc_id} {version}：{len(text)} 字符")

    from feishu_assistant.modules.review_metrics import compute_metrics, format_metrics
    metrics = compute_metrics(text)
    print("\n===== 客观指标（论文通用） =====")
    print(format_metrics(metrics))

    print(f"\n评审中（heavy，约 1~2 分钟）……")

    kimi = KimiClient(settings)
    prompt = REVIEW_PROMPT.replace("{{DOC_CONTENT}}", cap_middle(text))
    t0 = time.time()
    try:
        report = kimi.chat_json([{"role": "user", "content": prompt}], heavy=True)
    except KimiError as exc:
        print(f"评审调用失败：{exc}")
        return 1

    print(f"\n===== 初学者评审（{time.time() - t0:.0f}s）=====")
    print(f"评分 {report.get('score')}/10 · verdict {report.get('verdict')}")
    print(f"总评：{report.get('summary', '')}")
    strengths = report.get("strengths") or []
    if strengths:
        print("\n值得保留：")
        for s in strengths:
            print(f"  + {s}")
    for i, issue in enumerate(report.get("global_issues") or [], 1):
        print(f"\nG{i}. [{issue.get('severity')}|全局] {issue.get('scope')}")
        print(f"   卡点：{issue.get('problem')}")
        print(f"   期望：{issue.get('expect')}")
    for i, issue in enumerate(report.get("local_issues") or [], 1):
        print(f"\nL{i}. [{issue.get('severity')}|局部] {issue.get('location')}")
        print(f"   原文：{issue.get('evidence')}")
        print(f"   卡点：{issue.get('problem')}")
        print(f"   期望：{issue.get('expect')}")

    if args.save:
        out_dir = ROOT / "data" / "reviews"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{args.arxiv or doc_id}_{version}_{int(t0)}.json"
        payload = {"metrics": metrics, "review": report}
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n已存 {out}")
    return 0 if report.get("verdict") == "pass" else 2


if __name__ == "__main__":
    sys.exit(main())
