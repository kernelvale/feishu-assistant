"""feishu/bot.py 生命周期测试：回复送达语义、死信、worker 分发、状态卡序号/孤儿实体。

与 test_dedup.py 的分工：那边测「去重状态机」，这边测「处理完成 vs 回复未送达」
的完整生命周期与 WS 线程模型。
"""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock

from feishu_assistant.config import Settings
from feishu_assistant.feishu.bot import FeishuBot
from feishu_assistant.modules.router import Router
from feishu_assistant.store import DedupStore
from lark_oapi.api.im.v1 import P2ImMessageReceiveV1


def _make_bot(tmp_path, reply_text_result=True, chat_fn=None):
    replies: list[str] = []
    bot = FeishuBot(Settings(feishu_app_id="x", feishu_app_secret="y"),
                    Router(chat_fn=chat_fn or (lambda t, c: "回复:" + t)))
    bot.reply_text = lambda mid, text: replies.append(text) or reply_text_result  # type: ignore[method-assign]
    bot.dedup = DedupStore(tmp_path / "t.db")
    return bot, replies


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


# ---- 回复传输层异常收口 ----

def test_reply_text_transport_exception_returns_false(tmp_path):
    """传输层异常（断网/超时）不逃逸：记日志返回 False，兑现 docstring 的「不抛」承诺。"""
    bot, _ = _make_bot(tmp_path)
    del bot.reply_text  # 去掉 _make_bot 的替身，测真实方法
    bot.client = MagicMock()
    bot.client.im.v1.message.reply.side_effect = ConnectionError("boom")
    assert bot.reply_text("om_1", "hi") is False
    # API 错误码路径同样返回 False
    resp = MagicMock()
    resp.success.return_value = False
    bot.client.im.v1.message.reply.side_effect = None
    bot.client.im.v1.message.reply.return_value = resp
    assert bot.reply_text("om_1", "hi") is False


def test_send_transport_exception_returns_false(tmp_path):
    """主动推送（send_text/send_card）的传输层异常同样只记日志返回 False。"""
    bot, _ = _make_bot(tmp_path)
    bot.client = MagicMock()
    bot.client.im.v1.message.create.side_effect = TimeoutError("boom")
    assert bot.send_text("oc_chat", "hi") is False


# ---- dispatch / mark_done / 死信 生命周期 ----

def test_reply_failure_marks_done_but_writes_dead_letter(tmp_path):
    """LLM 算完但回复没送达：标 done（不重复答）+ 写死信（可 /重放），不是判死。"""
    bot, replies = _make_bot(tmp_path, reply_text_result=False)
    bot._on_message_receive(_make_event("m1", "你好"))
    assert replies == ["回复:你好"]  # 尝试过回复
    assert bot.dedup.status("m1") == "done"  # 终态：已处理
    letters = bot.dedup.list_dead_letters("oc_chat")
    assert len(letters) == 1 and "回复未送达" in letters[0].reason


def test_successful_delivery_writes_no_dead_letter(tmp_path):
    """回复送达：标 done，不写死信。"""
    bot, _ = _make_bot(tmp_path)
    bot._on_message_receive(_make_event("m2", "你好"))
    assert bot.dedup.status("m2") == "done"
    assert bot.dedup.list_dead_letters("oc_chat") == []


def test_router_exception_falls_back_to_error_reply(tmp_path):
    """路由抛异常 → 中文错误回复（送达即完成，不写死信），异常不抛给 WS 层。"""
    def boom(t, c):
        raise RuntimeError("x")

    bot, replies = _make_bot(tmp_path, chat_fn=boom)
    bot._on_message_receive(_make_event("m3", "你好"))
    assert replies and "出错" in replies[-1]
    assert bot.dedup.status("m3") == "done"
    assert bot.dedup.list_dead_letters("oc_chat") == []


# ---- WS 线程模型：事件循环只排队，分发在 worker 线程 ----

def test_enqueue_message_dispatches_in_worker_thread(tmp_path):
    """WS 入口只排队：dedup + 分发 + 回复全在 worker 线程完成，事件循环不被 LLM 阻塞。"""
    bot, replies = _make_bot(tmp_path)
    seen_threads: list[str] = []
    orig = bot._on_message_receive

    def spy(data):
        seen_threads.append(threading.current_thread().name)
        return orig(data)

    bot._on_message_receive = spy  # type: ignore[method-assign]
    bot._enqueue_message(_make_event("m4", "你好"))  # 立即返回，不阻塞
    for _ in range(100):
        if replies:
            break
        time.sleep(0.02)
    assert replies == ["回复:你好"]
    assert seen_threads and seen_threads[0].startswith("feishu-msg")
    assert bot.dedup.status("m4") == "done"


def test_enqueue_falls_back_to_sync_without_executor(tmp_path):
    """__new__ 构造的 bot 没有 executor：退化为同步执行（语义一致）。"""
    bot, replies = _make_bot(tmp_path)
    del bot._msg_executor
    bot._enqueue_message(_make_event("m5", "你好"))
    assert replies == ["回复:你好"]  # 同步完成，无需等待


# ---- 卡片回调：防连点与异步回刷 ----

def _card_event(value: dict, msg_id: str = ""):
    d = MagicMock()
    d.event.action.value = value
    d.event.context.open_message_id = msg_id
    return d


def test_read_paper_button_dedup_blocks_double_click(tmp_path):
    """卡片「精读这篇」防连点：同一篇已有在飞任务 → toast 提示，不再起管线。"""
    bot, _ = _make_bot(tmp_path)
    svc = MagicMock()
    svc._start_async.return_value = None  # 在飞
    bot.paper_service = svc
    resp = bot._on_card_action(_card_event(
        {"act": "read_paper", "arxiv_id": "2603.03329", "title": "T"}))
    assert resp.toast.type == "warning" and "已在精读中" in resp.toast.content
    _, kwargs = svc._start_async.call_args
    assert kwargs["dedup_key"] == "arxiv:2603.03329"

    svc._start_async.return_value = "om_status"  # 正常启动
    resp2 = bot._on_card_action(_card_event(
        {"act": "read_paper", "arxiv_id": "2603.03329", "title": "T"}))
    assert resp2.toast.type == "success"


def test_mark_read_sync_state_change_and_async_restyle(tmp_path):
    """mark_read：状态变更（查库+更新）同步完成支撑 toast；卡片回刷挪 worker。"""
    bot, _ = _make_bot(tmp_path)
    bot.client = MagicMock()
    del bot._card_executor  # 无 executor：回刷退化为同步执行，便于断言
    dao = MagicMock()
    dao.find_record_by_field.return_value = "rec1"
    svc = MagicMock()
    svc.bitable_dao = dao
    bot.paper_service = svc
    resp = bot._on_card_action(_card_event(
        {"act": "mark_read", "arxiv_id": "x", "title": "T"}))  # msg_id 空：回刷跳过
    assert resp.toast.type == "success"
    dao.update_record.assert_called_once_with("精选库", "rec1", {"状态": "已读"})


def test_refresh_card_read_patches_message(tmp_path):
    """已读回刷（worker）：读原卡片、把目标按钮换成 ✓ 已读、PATCH 回原消息。"""
    bot, _ = _make_bot(tmp_path)
    bot.client = MagicMock()
    card = {"elements": [{"tag": "action", "actions": [
        {"tag": "button", "text": {"tag": "plain_text", "content": "已读"},
         "type": "default", "value": {"act": "mark_read", "arxiv_id": "x"}},
    ]}]}
    get_resp = MagicMock()
    get_resp.success.return_value = True
    item = MagicMock()
    item.body.content = json.dumps(card, ensure_ascii=False)
    get_resp.data.items = [item]
    bot.client.im.v1.message.get.return_value = get_resp
    upd_resp = MagicMock()
    upd_resp.success.return_value = True
    bot.client.im.v1.message.update.return_value = upd_resp

    bot._refresh_card_read(_card_event({}, msg_id="om_card"), "x")
    req = bot.client.im.v1.message.update.call_args.args[0]
    body = json.loads(req.request_body.content)
    btn = body["elements"][0]["actions"][0]
    assert btn["text"]["content"] == "✓ 已读" and btn["type"] == "default"


# ---- 状态卡：序号锁 / 序号清理 / 孤儿实体废弃 ----

def _bot_with_card_client(tmp_path):
    bot, _ = _make_bot(tmp_path)
    bot.client = MagicMock()
    ok = MagicMock()
    ok.success.return_value = True
    bot.client.cardkit.v1.card_element.content.return_value = ok
    return bot


def test_status_card_seqs_increment_and_close_cleans(tmp_path):
    """流式序号按 card_id 单调递增；status_card_close 清理对应序号（防只增不清）。"""
    bot = _bot_with_card_client(tmp_path)
    bot.status_card_update("c1|e", "f1")
    bot.status_card_update("c1|e", "f2")
    bot.status_card_update("c2|e", "g1")
    assert bot._status_card_seqs == {"c1": 2, "c2": 1}
    bot.status_card_close("c1|e")
    assert bot._status_card_seqs == {"c2": 1}
    bot.status_card_update("c1|e", "f3")  # 重新从 1 计
    assert bot._status_card_seqs["c1"] == 1


def test_status_card_seqs_monotonic_under_threads(tmp_path):
    """序号锁内分配：多线程并发更新同一卡，序号不撞号不少号。"""
    bot = _bot_with_card_client(tmp_path)

    def bump():
        for _ in range(50):
            bot.status_card_update("c1|e", "f")

    threads = [threading.Thread(target=bump) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert bot._status_card_seqs["c1"] == 200


def test_status_card_create_send_failure_discards_orphan(tmp_path):
    """卡实体创建成功但发送失败：写墓碑 + 清序号，返回 ""（调用方据此降级文本）。"""
    bot = _bot_with_card_client(tmp_path)
    create_resp = MagicMock()
    create_resp.success.return_value = True
    create_resp.data.card_id = "card_orphan"
    bot.client.cardkit.v1.card.create.return_value = create_resp
    bot._send = lambda *a, **k: False  # type: ignore[method-assign]  # 发送失败

    handle = bot.status_card_create("oc_chat", "初始帧")
    assert handle == ""
    bot.client.cardkit.v1.card_element.content.assert_called_once()  # 写了墓碑
    assert bot._status_card_seqs == {}  # 序号已清，不泄漏


# ---- 事件通知外发箱补发线程（outbox flusher）----

def _make_bot_with_outbox(tmp_path, send_result=True):
    """bot + 真 NotificationOutbox（tmp 库）+ 可控 send_text 替身。"""
    from feishu_assistant.store import NotificationOutbox
    outbox = NotificationOutbox(tmp_path / "o.db")
    bot, _ = _make_bot(tmp_path)
    bot.outbox = outbox
    sent: list[tuple[str, str]] = []
    bot.send_text = lambda chat_id, text: sent.append((chat_id, text)) or send_result  # type: ignore[method-assign]
    return bot, outbox, sent


def test_flush_outbox_delivers_and_marks(tmp_path):
    """补发成功：逐条 send_text，送达 mark_delivered，pending 清空。"""
    bot, outbox, sent = _make_bot_with_outbox(tmp_path)
    outbox.enqueue("oc_chat", "❌ 精读L失败：网络断")
    outbox.enqueue("oc_chat", "⚠️ 另一条")

    assert bot.flush_outbox_once() == 2
    assert [t for _, t in sent] == ["❌ 精读L失败：网络断", "⚠️ 另一条"]
    assert outbox.pending() == []
    bot.flush_outbox_once()  # 无 pending：不再发
    assert len(sent) == 2


def test_flush_outbox_failure_retries_next_round(tmp_path):
    """断网一轮：send 失败 mark_attempt（仍 pending）；网络恢复下轮送达。"""
    bot, outbox, sent = _make_bot_with_outbox(tmp_path, send_result=False)
    outbox.enqueue("oc_chat", "🔁 恢复通知")

    assert bot.flush_outbox_once() == 0  # 断网：全部失败
    assert len(sent) == 1
    msgs = outbox.pending()
    assert len(msgs) == 1 and msgs[0].attempts == 1 and msgs[0].last_attempt_ts is not None

    bot.send_text = lambda chat_id, text: sent.append((chat_id, text)) or True  # type: ignore[method-assign]
    assert bot.flush_outbox_once() == 1  # 恢复：补发成功
    assert outbox.pending() == []


def test_flush_outbox_attempts_capped(tmp_path):
    """尝试封顶：attempts 到上限的通知不再补发（不无限骚扰）。"""
    from feishu_assistant.store import OUTBOX_MAX_ATTEMPTS
    bot, outbox, sent = _make_bot_with_outbox(tmp_path, send_result=False)
    msg_id = outbox.enqueue("oc_chat", "送不出去")
    for _ in range(OUTBOX_MAX_ATTEMPTS):
        outbox.mark_attempt(msg_id)
    assert bot.flush_outbox_once() == 0
    assert sent == []  # 封顶后 send_text 一次都没调


def test_flush_outbox_skips_expired(tmp_path):
    """超 24h TTL 的积压通知放弃补发。"""
    import sqlite3 as _sqlite3
    bot, outbox, sent = _make_bot_with_outbox(tmp_path)
    old_id = outbox.enqueue("oc_chat", "隔夜通知")
    with _sqlite3.connect(outbox.db_path) as conn:
        conn.execute(
            "UPDATE notification_outbox SET created_ts = ? WHERE id = ?",
            (time.time() - 25 * 3600, old_id),
        )
    assert bot.flush_outbox_once() == 0 and sent == []


def test_flush_outbox_without_outbox_noop(tmp_path):
    """未注入 outbox：flush 返回 0，start_outbox_flusher 不起线程。"""
    bot, _ = _make_bot(tmp_path)
    assert bot.outbox is None
    assert bot.flush_outbox_once() == 0
    bot.start_outbox_flusher(interval=0.01)
    assert bot._outbox_flusher_thread is None


def test_outbox_flush_loop_exception_contained(tmp_path):
    """单轮 flush 抛异常被循环兜住：线程活着继续跑（不炸主流程）。"""
    bot, outbox, sent = _make_bot_with_outbox(tmp_path)
    outbox.enqueue("oc_chat", "必达通知")
    real_pending = outbox.pending
    calls = []

    def flaky_pending(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("sqlite 抖动")
        return real_pending(*a, **k)

    outbox.pending = flaky_pending  # type: ignore[method-assign]
    bot.start_outbox_flusher(interval=0.02)
    try:
        for _ in range(200):  # 第一轮炸、第二轮送达
            if sent:
                break
            time.sleep(0.01)
        assert sent == [("oc_chat", "必达通知")]
        assert bot._outbox_flusher_thread is not None and bot._outbox_flusher_thread.is_alive()
    finally:
        bot.stop_outbox_flusher()  # 清理：daemon 线程不外泄污染后续测试
