#!/usr/bin/env python3
"""生产管线同步实跑一篇 arXiv 精读（完整 UX：进度行 + 文档 + 完成卡 + 归档）。

用法：uv run python scripts/produce_paper.py <arxiv_id>
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feishu_assistant.config import load_settings
from feishu_assistant.feishu import build_client
from feishu_assistant.feishu.bitable import BitableDAO
from feishu_assistant.feishu.bot import FeishuBot
from feishu_assistant.kimi import KimiClient
from feishu_assistant.main import setup_logging
from feishu_assistant.modules.paper import MessageContext, PaperReadingService, build_result_card
from feishu_assistant.modules.router import Router
from feishu_assistant.store import PaperStore
from feishu_assistant.vectors import PaperVectorIndex

CHAT_ID = "oc_221a0d161254e1c4eab2ac052f604fdb"


def main() -> int:
    setup_logging()
    if len(sys.argv) < 2:
        print("用法：produce_paper.py <arxiv_id>")
        return 1
    arxiv_id = sys.argv[1].strip()
    settings = load_settings()
    lark_client = build_client(settings)
    kimi = KimiClient(settings)
    bot = FeishuBot(settings, Router())
    svc = PaperReadingService(settings, kimi, lark_client, PaperStore(), sender=bot,
                              vector_index=PaperVectorIndex(ROOT / "data" / "papers.db"))
    svc.bitable_dao = BitableDAO(lark_client, settings.bitable_app_token, settings.bitable_tables)

    ctx = MessageContext(chat_id=CHAT_ID, open_id="", message_id="")
    from feishu_assistant.modules.status import StatusMessenger
    status = StatusMessenger(bot, CHAT_ID, f"精读 {arxiv_id}").start()

    def job(context, **kwargs):
        return svc._job_arxiv(context, **kwargs)

    try:
        # 瞬时故障（网关超时/断网/DNS）按 2/5/10 分钟退避自动重进，检查点续跑
        metadata, doc_url, archived = svc._run_with_transient_retry(
            ctx, f"arXiv:{arxiv_id}", job, {"arxiv_id": arxiv_id}, status)
    except Exception as exc:
        print(f"管线失败：{type(exc).__name__}: {exc}")
        bot.send_text(CHAT_ID, f"❌ 精读失败：arXiv:{arxiv_id}\n{type(exc).__name__}: {exc}")
        return 1
    bot.send_card(CHAT_ID, build_result_card(
        metadata, doc_url, arxiv_id, archived,
        timing_text=str(metadata.get("_timing", "")),
    ))
    print(f"文档：{doc_url}")
    print(f"版本：v{metadata.get('_version')}")
    print(f"归档：{archived}")
    print(f"计时：{metadata.get('_timing', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
