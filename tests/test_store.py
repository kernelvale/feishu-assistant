"""store.py 检索层测试：FTS/LIKE 查询转义、exclude_session_id 下推。全部离线。

另含 ReadingTaskStore（精读任务登记）生命周期：登记→完成→interrupted→purge；
NotificationOutbox（事件通知外发箱）生命周期：enqueue 去重→pending 过滤→
mark_delivered/mark_attempt→purge。
"""

from __future__ import annotations

import sqlite3
import time

from feishu_assistant.store import (
    OUTBOX_DEDUP_WINDOW_SECONDS,
    OUTBOX_MAX_ATTEMPTS,
    NotificationOutbox,
    ReadingTaskStore,
    SessionStore,
)


def make_store(tmp_path) -> SessionStore:
    return SessionStore(tmp_path / "s.db")


def seed_two_sessions(store: SessionStore, chat_id: str = "c1"):
    """建两个会话（旧/新），返回 (旧, 新)。"""
    old = store.create_session(chat_id, "旧会话")
    new = store.create_session(chat_id, "新会话")
    return old, new


# ---- FTS5 查询转义 ----

def test_fts_search_returns_session_title(tmp_path):
    store = make_store(tmp_path)
    old, _ = seed_two_sessions(store)
    store.add_message(old.id, "c1", "user", "马尔可夫决策过程")
    hits = store.search_messages("c1", "马尔可夫")
    assert len(hits) == 1
    msg, title = hits[0]
    assert "马尔可夫" in msg.content and title == "旧会话"


def test_fts_query_with_quotes_no_crash(tmp_path):
    """用户输入带双引号：内嵌引号翻倍转义，不再 OperationalError。"""
    store = make_store(tmp_path)
    old, _ = seed_two_sessions(store)
    store.add_message(old.id, "c1", "user", '他说"你好"然后走了')
    hits = store.search_messages("c1", '上次说的"你好"')
    assert len(hits) == 1 and "你好" in hits[0][0].content


def test_fts_special_chars_never_crash(tmp_path):
    """FTS5 语法字符全部字面化：任何用户输入都不该炸掉召回。"""
    store = make_store(tmp_path)
    old, _ = seed_two_sessions(store)
    store.add_message(old.id, "c1", "user", "普通内容")
    for q in ('"', '"""', "a OR b", "NOT x", "NEAR(", "*", "(", ")", "^", ":", "-", "50%"):
        store.search_messages("c1", q)  # 不抛 sqlite3.OperationalError


def test_empty_query_returns_empty(tmp_path):
    store = make_store(tmp_path)
    old, _ = seed_two_sessions(store)
    store.add_message(old.id, "c1", "user", "内容")
    assert store.search_messages("c1", "   ") == []


# ---- LIKE 退化路径转义 ----

def test_like_fallback_escapes_wildcards(tmp_path):
    """无 FTS5 时 LIKE 的 % _ 是字面量不是通配符（修复前 "_" 能匹配全部）。"""
    store = make_store(tmp_path)
    store.fts_available = False  # 强制走 LIKE 退化路径
    old, _ = seed_two_sessions(store)
    store.add_message(old.id, "c1", "user", "进度 50% 了")
    store.add_message(old.id, "c1", "user", "随便别的内容")
    hits = store.search_messages("c1", "50%")
    assert [m.content for m, _ in hits] == ["进度 50% 了"]
    assert store.search_messages("c1", "_") == []


def test_like_fallback_matches_literal_underscore(tmp_path):
    store = make_store(tmp_path)
    store.fts_available = False
    old, _ = seed_two_sessions(store)
    store.add_message(old.id, "c1", "user", "snake_case 命名")
    store.add_message(old.id, "c1", "user", "snakeXcase 不算")
    hits = store.search_messages("c1", "snake_case")
    assert [m.content for m, _ in hits] == ["snake_case 命名"]


# ---- exclude_session_id 下推到 SQL ----

def test_exclude_session_applied_before_limit(tmp_path):
    """先排除再 LIMIT：当前会话占满 top3 时，旧会话命中仍能返回（修复前召回变空）。"""
    store = make_store(tmp_path)
    old, new = seed_two_sessions(store)
    store.add_message(old.id, "c1", "user", "旧会话的马尔可夫")
    for _ in range(3):
        store.add_message(new.id, "c1", "user", "新会话的马尔可夫")
    # 不排除时 top3 全是新会话（id 倒序）
    hits_all = store.search_messages("c1", "马尔可夫", limit=3)
    assert len(hits_all) == 3 and all(m.session_id == new.id for m, _ in hits_all)
    # SQL 层排除后旧会话命中浮出
    hits = store.search_messages("c1", "马尔可夫", limit=3, exclude_session_id=new.id)
    assert len(hits) == 1 and hits[0][0].session_id == old.id


def test_exclude_session_in_like_fallback(tmp_path):
    store = make_store(tmp_path)
    store.fts_available = False
    old, new = seed_two_sessions(store)
    store.add_message(old.id, "c1", "user", "旧会话的马尔可夫")
    store.add_message(new.id, "c1", "user", "新会话的马尔可夫")
    hits = store.search_messages("c1", "马尔可夫", exclude_session_id=new.id)
    assert [m.session_id for m, _ in hits] == [old.id]


# ---- ReadingTaskStore：精读任务登记（断线/重启恢复）----

def _make_task_store(tmp_path) -> ReadingTaskStore:
    return ReadingTaskStore(tmp_path / "t.db")


def _backdate_created(store: ReadingTaskStore, dedup_key: str, created_ts: float) -> None:
    """把登记时间改到过去（模拟老任务/过期任务）。"""
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE reading_tasks SET created_ts = ? WHERE dedup_key = ?",
            (created_ts, dedup_key),
        )


def test_reading_tasks_register_and_finish(tmp_path):
    """登记 → running；结束回写 done/failed 并带 finished_ts。"""
    store = _make_task_store(tmp_path)
    store.upsert_start("arxiv:1", "arxiv", "1", "arXiv:1", "c1", open_id="ou_x", message_id="om_1")
    task = store.get("arxiv:1")
    assert task is not None and task.status == "running"
    assert (task.kind, task.ref, task.label, task.chat_id) == ("arxiv", "1", "arXiv:1", "c1")
    assert task.open_id == "ou_x" and task.message_id == "om_1"
    assert task.finished_ts is None

    store.mark_finished("arxiv:1", ok=True)
    task = store.get("arxiv:1")
    assert task is not None and task.status == "done" and task.finished_ts is not None

    store.upsert_start("url:https://x", "url", "https://x", "链接:x", "c1")
    store.mark_finished("url:https://x", ok=False)
    assert store.get("url:https://x").status == "failed"  # type: ignore[union-attr]


def test_reading_tasks_reregister_resets_to_running(tmp_path):
    """同一查重键重读：upsert 重置回 running（新一轮时间戳，finished_ts 清空）。"""
    store = _make_task_store(tmp_path)
    store.upsert_start("arxiv:1", "arxiv", "1", "arXiv:1", "c1")
    store.mark_finished("arxiv:1", ok=False)
    assert store.get("arxiv:1").status == "failed"  # type: ignore[union-attr]
    store.upsert_start("arxiv:1", "arxiv", "1", "arXiv:1", "c2")  # 换个 chat 复读
    task = store.get("arxiv:1")
    assert task is not None
    assert task.status == "running" and task.finished_ts is None and task.chat_id == "c2"


def test_reading_tasks_interrupted_only_old_running(tmp_path):
    """interrupted 只捞 status=running 且早于阈值：done/failed 与新登记的都排除。"""
    store = _make_task_store(tmp_path)
    store.upsert_start("arxiv:old", "arxiv", "old", "老任务", "c1")
    store.upsert_start("arxiv:done", "arxiv", "done", "已完成", "c1")
    store.upsert_start("arxiv:failed", "arxiv", "failed", "已失败", "c1")
    store.mark_finished("arxiv:done", ok=True)
    store.mark_finished("arxiv:failed", ok=False)
    now = time.time()
    _backdate_created(store, "arxiv:old", now - 3600)
    store.upsert_start("arxiv:new", "arxiv", "new", "新任务", "c1")  # 阈值之后才登记

    hits = store.interrupted(older_than=now)
    assert [t.dedup_key for t in hits] == ["arxiv:old"]
    # 默认阈值=调用时刻：old 与新登记的 new 都早于它，全部捞出（按登记时间正序）
    assert [t.dedup_key for t in store.interrupted()] == ["arxiv:old", "arxiv:new"]


def test_reading_tasks_purge_old(tmp_path):
    """purge_old 只删超过 TTL 的记录（含僵死 running），返回删除行数。"""
    store = _make_task_store(tmp_path)
    store.upsert_start("arxiv:stale", "arxiv", "s", "僵死任务", "c1")
    store.upsert_start("arxiv:fresh", "arxiv", "f", "新任务", "c1")
    _backdate_created(store, "arxiv:stale", time.time() - 8 * 24 * 3600)

    assert store.purge_old() == 1
    assert store.get("arxiv:stale") is None
    assert store.get("arxiv:fresh") is not None
    assert store.purge_old() == 0


# ---- NotificationOutbox：事件通知外发箱（失败/卡死/恢复必达，断网排队补发）----

def _make_outbox(tmp_path) -> NotificationOutbox:
    return NotificationOutbox(tmp_path / "o.db")


def _backdate_outbox(store: NotificationOutbox, msg_id: int, created_ts: float) -> None:
    """把入队时间改到过去（模拟老通知/超窗通知）。"""
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE notification_outbox SET created_ts = ? WHERE id = ?",
            (created_ts, msg_id),
        )


def test_outbox_enqueue_and_pending_fifo(tmp_path):
    """入队 → pending 按先入先出返回，字段默认值正确。"""
    outbox = _make_outbox(tmp_path)
    id1 = outbox.enqueue("c1", "第一条")
    id2 = outbox.enqueue("c1", "第二条")
    assert id1 is not None and id2 is not None and id2 > id1

    msgs = outbox.pending()
    assert [(m.id, m.text) for m in msgs] == [(id1, "第一条"), (id2, "第二条")]
    assert msgs[0].chat_id == "c1" and msgs[0].attempts == 0
    assert msgs[0].delivered_ts is None and msgs[0].last_attempt_ts is None


def test_outbox_enqueue_dedup_same_text_in_window(tmp_path):
    """同 chat_id+text 10 分钟内不重复入队；换文案/换 chat 或超窗则可再入队。"""
    outbox = _make_outbox(tmp_path)
    assert outbox.enqueue("c1", "❌ 精读L失败：网络断") is not None
    assert outbox.enqueue("c1", "❌ 精读L失败：网络断") is None  # 同文去重
    assert outbox.enqueue("c1", "⚠️ 另一条") is not None  # 不同文案
    assert outbox.enqueue("c2", "❌ 精读L失败：网络断") is not None  # 不同 chat
    assert len(outbox.pending()) == 3


def test_outbox_enqueue_allowed_after_dedup_window(tmp_path):
    """同文通知超过去重窗口后再次入队（同一失败隔天重演仍要告诉用户）。"""
    outbox = _make_outbox(tmp_path)
    old_id = outbox.enqueue("c1", "❌ 精读L失败：网络断")
    _backdate_outbox(outbox, old_id, time.time() - OUTBOX_DEDUP_WINDOW_SECONDS - 1)
    assert outbox.enqueue("c1", "❌ 精读L失败：网络断") is not None
    assert len(outbox.pending()) == 2


def test_outbox_mark_delivered_excludes_pending(tmp_path):
    """送达确认后不再出现在 pending。"""
    outbox = _make_outbox(tmp_path)
    id1 = outbox.enqueue("c1", "第一条")
    outbox.enqueue("c1", "第二条")
    outbox.mark_delivered(id1)
    msgs = outbox.pending()
    assert [m.text for m in msgs] == ["第二条"]


def test_outbox_attempts_cap_excludes_pending(tmp_path):
    """补发尝试达到上限的通知放弃投递（不无限骚扰）。"""
    outbox = _make_outbox(tmp_path)
    msg_id = outbox.enqueue("c1", "送不出去")
    for _ in range(OUTBOX_MAX_ATTEMPTS):
        outbox.mark_attempt(msg_id)
    assert outbox.pending() == []


def test_outbox_ttl_excludes_old_pending(tmp_path):
    """入队超 24h 未送达：pending 放弃（用户早已不在等这条）。"""
    outbox = _make_outbox(tmp_path)
    old_id = outbox.enqueue("c1", "隔夜通知")
    outbox.enqueue("c1", "新鲜通知")
    _backdate_outbox(outbox, old_id, time.time() - 25 * 3600)
    assert [m.text for m in outbox.pending()] == ["新鲜通知"]


def test_outbox_pending_limit(tmp_path):
    """pending 尊重 limit（一轮最多补发 limit 条）。"""
    outbox = _make_outbox(tmp_path)
    for i in range(5):
        outbox.enqueue("c1", f"通知{i}")
    assert len(outbox.pending(limit=3)) == 3


def test_outbox_purge_old(tmp_path):
    """purge_old 删除超 TTL 记录（含送不出去的），返回行数；enqueue 也顺带清理。"""
    outbox = _make_outbox(tmp_path)
    stale_id = outbox.enqueue("c1", "老通知")
    outbox.enqueue("c1", "新通知")
    _backdate_outbox(outbox, stale_id, time.time() - 8 * 24 * 3600)

    assert outbox.purge_old() == 1
    assert [m.text for m in outbox.pending()] == ["新通知"]
    assert outbox.purge_old() == 0
