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
from feishu_assistant.modules.paper import MessageContext, PaperReadingService
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
    from feishu_assistant.modules.status import TaskCard
    from feishu_assistant.modules.reading_v9 import V9ReadingEngine
    from feishu_assistant.modules.paper import _card_state_path
    stages = (
        ["解析 PDF", "来源导航", "叙事简报", "连续正文", "专项审查", "有限修订", "定点验证", "发布归档"]
        if isinstance(svc.reading_engine, V9ReadingEngine) else
        ["解析 PDF", "证据索引", "规划（债务+Plan）", "正文生成", "图表讲解", "终审", "发布归档"]
    )
    status = TaskCard(bot, CHAT_ID, f"精读 {arxiv_id}", stages,
                      state_path=_card_state_path(f"arxiv:{arxiv_id}")).start()
    ctx.status = status  # 进度回调按 context.status 分发，不挂上就永远不更新

    def job(context, **kwargs):
        return svc._job_arxiv(context, **kwargs)

    try:
        # 瞬时故障（网关超时/断网/DNS）按 2/5/10 分钟退避自动重进，检查点续跑
        metadata, doc_url, archived = svc._run_with_transient_retry(
            ctx, f"arXiv:{arxiv_id}", job, {"arxiv_id": arxiv_id}, status)
    except Exception as exc:
        print(f"管线失败：{type(exc).__name__}: {exc}")
        try:
            bot.send_text(CHAT_ID, f"❌ 精读失败：arXiv:{arxiv_id}\n{type(exc).__name__}: {exc}")
        except Exception as notify_exc:
            # 断网期失败通知本身也发不出去：保底打印，不再二次崩溃（真实教训）
            print(f"失败通知发送失败（网络不可达）：{notify_exc}")
        return 1
    from feishu_assistant.modules.paper import _result_markdown
    release_status = str(metadata.get("release_status") or "").strip()
    status_note = ("精读完成，两项审查已完成；自动改动仅采用已验证补丁"
                   if release_status == "verified_final" else "精读完成")
    status.stop(ok=True, note=status_note,
                result_md=_result_markdown(
                    metadata, doc_url, arxiv_id, archived,
                    timing_text=str(metadata.get("_timing", ""))))
    print(f"文档：{doc_url}")
    print(f"版本：v{metadata.get('_version')}")
    print(f"归档：{archived}")
    print(f"计时：{metadata.get('_timing', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
