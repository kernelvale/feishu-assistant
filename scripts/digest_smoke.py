#!/usr/bin/env python3
"""P4 推送模块真实冒烟脚本（真实 arXiv API + .env 飞书凭证 + 真实模型）。

默认只读：真实抓 arXiv → 解析 → 排序（关键词兜底，无需向量权重）→ 打印 top 5，
不发任何推送、不写任何表。--tldr 调真实 light 模型写推荐语；--push CHAT_ID 真发卡片。

用法：
    uv run python scripts/digest_smoke.py                 # 抓取+解析+排序（只读，安全）
    uv run python scripts/digest_smoke.py --tldr          # 加真实 TL;DR 生成
    uv run python scripts/digest_smoke.py --tldr --push oc_xxx   # 真发日报卡片到指定会话
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from feishu_assistant.config import load_settings  # noqa: E402
from feishu_assistant.feishu import build_client  # noqa: E402
from feishu_assistant.feishu.bitable import BitableDAO  # noqa: E402
from feishu_assistant.feishu.career_io import CareerIO  # noqa: E402
from feishu_assistant.kimi import KimiClient  # noqa: E402
from feishu_assistant.modules.digest import (  # noqa: E402
    DigestService,
    build_arxiv_card,
    build_arxiv_query,
    fetch_arxiv_entries,
    keyword_score,
    rank_entries,
    taste_keywords_from_papers,
)


class CliSender:
    """--push 时的真实推送器（复用 bot 的 REST 发送）。"""

    def __init__(self, settings):
        from feishu_assistant.feishu.bot import FeishuBot
        from feishu_assistant.modules.router import Router

        self.bot = FeishuBot(settings, Router())

    def send_text(self, chat_id, text):
        return self.bot.send_text(chat_id, text)

    def send_card(self, chat_id, card):
        return self.bot.send_card(chat_id, card)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tldr", action="store_true", help="调真实 light 模型生成 TL;DR")
    parser.add_argument("--push", metavar="CHAT_ID", help="真发日报卡片到指定 chat_id")
    args = parser.parse_args()

    settings = load_settings()

    print("1/4 构造 arXiv 查询并真实抓取……")
    print(f"    query: {build_arxiv_query()[:100]}…")
    try:
        entries = fetch_arxiv_entries()
    except Exception as exc:
        print(f"    ❌ 抓取失败：{exc}")
        return 1
    print(f"    抓到 {len(entries)} 篇 ✓")
    if not entries:
        print("    近 2 天无新论文（周末/节假日正常），冒烟到此为止。")
        return 0

    print("2/4 读取论文库口味（只读）并排序……")
    keywords: list[str] = []
    if settings.bitable_ready() and settings.feishu_app_id:
        io = CareerIO(BitableDAO(build_client(settings), settings.bitable_app_token, settings.bitable_tables))
        keywords = taste_keywords_from_papers(io.list_all("论文库"))
        print(f"    口味关键词 {len(keywords)} 个：{'、'.join(keywords[:8])}…" if keywords else "    论文库为空，按零分排序")
    else:
        print("    多维表格未初始化，按零分排序（日期序）")
    ranked = rank_entries(entries, lambda e: keyword_score(e, keywords))
    top = [e for e, s in ranked[:5]]
    for i, (e, s) in enumerate(ranked[:5], 1):
        print(f"    {i}. [{s:.0f}分] {e.title[:50]}（{e.published}）")

    if not args.tldr:
        print("3/4 跳过（--tldr 才调模型）")
        print("4/4 跳过（--push 才发送）")
        print("\n✅ 只读冒烟通过：抓取 → 解析 → 排序")
        return 0

    print("3/4 调真实 light 模型生成 TL;DR……")
    kimi = KimiClient(settings)
    svc = DigestService(settings, kimi, io=io if keywords else None)
    items = svc._summarize_entries(top)
    for i, item in enumerate(items, 1):
        print(f"    {i}. {item['tldr'][:60]}")
        if item["reason"]:
            print(f"       💡 {item['reason'][:50]}")

    if not args.push:
        print("4/4 跳过（--push CHAT_ID 才发送）")
        print("\n✅ 冒烟通过：抓取 → 排序 → TL;DR（未发送）")
        return 0

    print(f"4/4 真发日报卡片到 {args.push}……")
    sender = CliSender(settings)
    card = build_arxiv_card(items, scanned=len(entries), days=2)
    if sender.send_card(args.push, card):
        print("    已发送 ✓")
        print("\n✅ 冒烟全部通过：抓取 → 排序 → TL;DR → 推送")
        return 0
    print("    ❌ 发送失败（检查 im:message 权限与 chat_id）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
