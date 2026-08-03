"""消息幂等去重测试：DedupStore 状态机 + bot 事件入口去重。"""

from __future__ import annotations

import json

from feishu_assistant.config import Settings
from feishu_assistant.feishu.bot import FeishuBot
from feishu_assistant.modules.router import Router
from feishu_assistant.store import DedupStore
from lark_oapi.api.im.v1 import P2ImMessageReceiveV1

# ---- DedupStore ----

def test_first_seen_true_second_false(tmp_path):
    dedup = DedupStore(tmp_path / "t.db")
    assert dedup.mark_processing("m1") is True
    assert dedup.mark_processing("m1") is False  # 重投 → 跳过


def test_done_still_seen(tmp_path):
    dedup = DedupStore(tmp_path / "t.db")
    dedup.mark_processing("m1")
    dedup.mark_done("m1")
    assert dedup.status("m1") == "done"
    assert dedup.mark_processing("m1") is False  # 已完成也跳过


def test_crash_residual_processing_treated_as_seen(tmp_path):
    """上次崩溃留下 processing 记录：宁可漏答不可重复答。"""
    dedup = DedupStore(tmp_path / "t.db")
    dedup.mark_processing("m1")  # 没有 mark_done，模拟崩溃残留
    assert dedup.status("m1") == "processing"
    assert dedup.mark_processing("m1") is False


def test_survives_reopen(tmp_path):
    """启动时不清表：换连接/重启后仍然记得。"""
    dedup1 = DedupStore(tmp_path / "t.db")
    dedup1.mark_processing("m1")
    dedup1.mark_done("m1")
    dedup2 = DedupStore(tmp_path / "t.db")  # 模拟重启
    assert dedup2.mark_processing("m1") is False


# ---- bot 事件入口 ----

def _make_event(message_id: str, text: str, chat_type: str = "p2p") -> P2ImMessageReceiveV1:
    return P2ImMessageReceiveV1({
        "event": {
            "sender": {"sender_type": "user", "sender_id": {"open_id": "ou_x"}},
            "message": {
                "message_id": message_id, "chat_id": "oc_chat", "chat_type": chat_type,
                "message_type": "text", "content": json.dumps({"text": text}),
            },
        }
    })


def _make_bot(tmp_path):
    replies: list[str] = []
    bot = FeishuBot(Settings(feishu_app_id="x", feishu_app_secret="y"),
                    Router(chat_fn=lambda t, c: "回复:" + t))
    bot.reply_text = lambda mid, text: replies.append(text) or True  # type: ignore[method-assign]
    bot.dedup = DedupStore(tmp_path / "t.db")
    return bot, replies


def test_bot_dedup_skips_redelivery(tmp_path):
    bot, replies = _make_bot(tmp_path)
    event = _make_event("m1", "你好")
    bot._on_message_receive(event)
    bot._on_message_receive(event)  # 长连接重投
    assert replies == ["回复:你好"]
    assert bot.dedup.status("m1") == "done"


def test_bot_dedup_marks_done_after_success(tmp_path):
    bot, _ = _make_bot(tmp_path)
    bot._on_message_receive(_make_event("m9", "你好"))
    assert bot.dedup.status("m9") == "done"


def test_bot_dedup_only_p2p(tmp_path):
    """群聊消息不参与去重（chat_type 非 p2p 时同 id 也处理）。"""
    bot, replies = _make_bot(tmp_path)
    event = _make_event("m2", "你好", chat_type="group")
    bot._on_message_receive(event)
    bot._on_message_receive(event)
    assert replies == ["回复:你好", "回复:你好"]


# ---- TTL 清理与死信 ----

def test_ttl_cleanup(tmp_path):
    """过期（>7 天）的 seen 记录在下次 mark_processing 时被清理，同 id 可重新处理。"""
    import sqlite3, time
    from feishu_assistant.store import DEDUP_TTL_SECONDS
    db = tmp_path / "t.db"
    dedup = DedupStore(db)
    # 手工塞一条 8 天前的记录
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO seen_messages (message_id, status, first_seen) VALUES (?, 'done', ?)",
            ("old_msg", time.time() - DEDUP_TTL_SECONDS - 3600),
        )
    assert dedup.mark_processing("old_msg") is True  # 过期已清，按新消息处理


def test_fresh_record_not_cleaned(tmp_path):
    dedup = DedupStore(tmp_path / "t.db")
    dedup.mark_processing("fresh")
    dedup.mark_done("fresh")
    assert dedup.mark_processing("fresh") is False  # 未过期，仍然去重


def test_dead_letter_written_on_skip(tmp_path):
    """bot 跳过重复消息时写 dead_letter，含原因与摘要。"""
    bot, replies = _make_bot(tmp_path)
    event = _make_event("m1", "精读这篇论文")
    bot._on_message_receive(event)
    bot._on_message_receive(event)  # 重投 → 跳过 + 死信
    letters = bot.dedup.list_dead_letters("oc_chat")
    assert len(letters) == 1
    assert letters[0].message_id == "m1"
    assert letters[0].content_text == "精读这篇论文"
    assert "重复投递" in letters[0].reason


def test_dead_letter_reason_for_crash_residual(tmp_path):
    """崩溃残留（processing 状态）跳过时原因不同。"""
    bot, _ = _make_bot(tmp_path)
    bot.dedup.mark_processing("m_crash")  # 模拟上次崩溃残留
    bot._on_message_receive(_make_event("m_crash", "你好"))
    letters = bot.dedup.list_dead_letters("oc_chat")
    assert len(letters) == 1 and "崩溃残留" in letters[0].reason
