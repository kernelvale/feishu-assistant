"""本地 sqlite 存储层（data/papers.db，五个 store 共用一库）。

- PaperStore：精读论文全文与最近精读记录（追问/绑定会话用）
- DedupStore：消息幂等去重（飞书长连接"至少一次"投递，重启会重投近期消息）
- ReadingTaskStore：精读任务登记（进程死在中途后，启动时自动恢复重跑）
- SessionStore：会话系统（sessions + messages + messages_fts + memories 长期记忆）
- NotificationOutbox：事件通知外发箱（失败/卡死/恢复必达，断网排队补发）

只依赖标准库 sqlite3，全部可离线测试。多线程并发写：WAL + busy_timeout。
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .config import PROJECT_ROOT

DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "papers.db"
FOLLOWUP_WINDOW_HOURS = 24  # 精读完成后该时间窗内允许追问


def _open(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def estimate_tokens(text: str) -> int:
    """粗略 token 估算（CJK 约 1 字 1 token、英文约 4 字符 1 token，取折中）。"""
    return max(1, (len(text) + 1) // 2)


# ---------------------------------------------------------------------------
# 论文全文存储
# ---------------------------------------------------------------------------

_PAPERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL,
    arxiv_id TEXT NOT NULL,
    title TEXT NOT NULL,
    markdown TEXT NOT NULL,
    association_text TEXT NOT NULL DEFAULT '',
    doc_url TEXT NOT NULL DEFAULT '',
    pdf_path TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_papers_chat_time ON papers (chat_id, created_at DESC);
"""


@dataclass
class PaperRecord:
    id: int
    chat_id: str
    arxiv_id: str
    title: str
    markdown: str
    doc_url: str
    pdf_path: str
    created_at: float
    association_text: str = ""


class PaperStore:
    """papers 表的最小封装。线程安全：每次操作开新连接（sqlite 推荐用法）。"""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with _open(self.db_path) as conn:
            conn.executescript(_PAPERS_SCHEMA)
            # 兼容已经运行过的本地库：CREATE TABLE IF NOT EXISTS 不会补新列。
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(papers)")}
            if "association_text" not in columns:
                conn.execute(
                    "ALTER TABLE papers ADD COLUMN association_text TEXT NOT NULL DEFAULT ''"
                )

    def _conn(self) -> sqlite3.Connection:
        return _open(self.db_path)

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> PaperRecord:
        return PaperRecord(
            id=row["id"], chat_id=row["chat_id"], arxiv_id=row["arxiv_id"],
            title=row["title"], markdown=row["markdown"], doc_url=row["doc_url"],
            pdf_path=row["pdf_path"], created_at=row["created_at"],
            association_text=row["association_text"],
        )

    def save_paper(
        self,
        chat_id: str,
        arxiv_id: str,
        title: str,
        markdown: str,
        doc_url: str = "",
        pdf_path: str = "",
        created_at: float | None = None,
        association_text: str = "",
    ) -> int:
        """保存一篇精读论文，返回行 id。"""
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO papers"
                " (chat_id, arxiv_id, title, markdown, association_text, doc_url, pdf_path, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (chat_id, arxiv_id, title, markdown, association_text, doc_url, pdf_path,
                 created_at if created_at is not None else time.time()),
            )
            return int(cur.lastrowid)

    def latest_paper(self, chat_id: str, within_hours: float | None = None) -> PaperRecord | None:
        """取该 chat 最近一篇精读论文；within_hours 限制时间窗（超时返回 None）。"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM papers WHERE chat_id = ? ORDER BY created_at DESC LIMIT 1",
                (chat_id,),
            ).fetchone()
        if row is None:
            return None
        if within_hours is not None and time.time() - row["created_at"] > within_hours * 3600:
            return None
        return self._row_to_record(row)

    def recent_papers(
        self,
        chat_id: str,
        within_hours: float | None = None,
        limit: int = 8,
    ) -> list[PaperRecord]:
        """取该 chat 最近的不同论文，按时间倒序。

        同一 arXiv/链接论文复读会生成多条全文记录；关联追问时只保留每个查重键
        最新的一条，避免候选列表里出现多个同名版本。
        """
        params: list[object] = [chat_id]
        sql = "SELECT * FROM papers WHERE chat_id = ?"
        if within_hours is not None:
            sql += " AND created_at >= ?"
            params.append(time.time() - within_hours * 3600)
        sql += " ORDER BY created_at DESC"
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        papers: list[PaperRecord] = []
        seen: set[str] = set()
        for row in rows:
            key = str(row["arxiv_id"] or row["title"]).strip()
            if key in seen:
                continue
            seen.add(key)
            papers.append(self._row_to_record(row))
            if len(papers) >= limit:
                break
        return papers

    def get_paper_by_arxiv(self, arxiv_id: str, chat_id: str = "") -> PaperRecord | None:
        """按 arXiv id 取最近一篇精读；可限制 chat，防跨会话串文。"""
        where = "arxiv_id = ?"
        params: list[object] = [arxiv_id]
        if chat_id:
            where += " AND chat_id = ?"
            params.append(chat_id)
        with self._conn() as conn:
            row = conn.execute(
                f"SELECT * FROM papers WHERE {where} ORDER BY created_at DESC LIMIT 1",
                params,
            ).fetchone()
        return self._row_to_record(row) if row else None

    def clear_pdf_paths_under(self, directory: Path | str) -> int:
        """清空指向某个待删除缓存目录的路径，返回更新行数。"""
        prefix = str(Path(directory).resolve(strict=False))
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, pdf_path FROM papers WHERE pdf_path != ''"
            ).fetchall()
            ids = []
            for row in rows:
                try:
                    Path(row["pdf_path"]).resolve(strict=False).relative_to(prefix)
                except (OSError, ValueError):
                    continue
                ids.append(int(row["id"]))
            if ids:
                conn.executemany("UPDATE papers SET pdf_path = '' WHERE id = ?", [(i,) for i in ids])
            return len(ids)

    def clear_missing_pdf_paths(self) -> int:
        """崩溃恢复对账：清空已经不存在或不再是普通文件的本地 PDF 路径。"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, pdf_path FROM papers WHERE pdf_path != ''"
            ).fetchall()
            ids = [
                int(row["id"]) for row in rows
                if not Path(row["pdf_path"]).is_file() or Path(row["pdf_path"]).is_symlink()
            ]
            if ids:
                conn.executemany("UPDATE papers SET pdf_path = '' WHERE id = ?", [(i,) for i in ids])
            return len(ids)


# ---------------------------------------------------------------------------
# 消息幂等去重
# ---------------------------------------------------------------------------

_DEDUP_SCHEMA = """
CREATE TABLE IF NOT EXISTS seen_messages (
    message_id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'processing',  -- processing | done
    first_seen REAL NOT NULL,
    done_at REAL
);
CREATE TABLE IF NOT EXISTS dead_letters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL,
    chat_id TEXT NOT NULL,
    open_id TEXT NOT NULL DEFAULT '',
    msg_type TEXT NOT NULL DEFAULT 'text',
    content_text TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL,
    created_at REAL NOT NULL,
    replayed INTEGER NOT NULL DEFAULT 0,
    replayed_at REAL
);
"""

DEDUP_TTL_SECONDS = 7 * 24 * 3600  # seen_messages 保留 7 天（处理时顺带清理过期）


@dataclass
class DeadLetter:
    id: int
    message_id: str
    chat_id: str
    open_id: str
    msg_type: str
    content_text: str
    reason: str
    created_at: float
    replayed: bool


class DedupStore:
    """消息去重 + 死信：先标记"处理中"，完成后更新"已完成"。

    时序约定：事件入口先 mark_processing——插入成功（首次见）才处理；
    已存在（无论 processing 还是 done）一律跳过并写死信表（可 /重放）。
    进程崩溃残留的 processing 记录同样按已处理对待：宁可漏答，不可重复答。
    seen_messages 带 7 天 TTL，mark_processing 时顺带清理。启动时不清表。
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with _open(self.db_path) as conn:
            conn.executescript(_DEDUP_SCHEMA)

    def mark_processing(self, message_id: str) -> bool:
        """首次见返回 True（并已占位）；已见过返回 False。顺带清理过期记录。"""
        with _open(self.db_path) as conn:
            conn.execute(
                "DELETE FROM seen_messages WHERE first_seen < ?",
                (time.time() - DEDUP_TTL_SECONDS,),
            )
            cur = conn.execute(
                "INSERT OR IGNORE INTO seen_messages (message_id, status, first_seen)"
                " VALUES (?, 'processing', ?)",
                (message_id, time.time()),
            )
            return cur.rowcount == 1

    def mark_done(self, message_id: str) -> None:
        with _open(self.db_path) as conn:
            conn.execute(
                "UPDATE seen_messages SET status = 'done', done_at = ? WHERE message_id = ?",
                (time.time(), message_id),
            )

    def status(self, message_id: str) -> str | None:
        """返回 processing / done / None（未见），主要给测试与排查用。"""
        with _open(self.db_path) as conn:
            row = conn.execute(
                "SELECT status FROM seen_messages WHERE message_id = ?", (message_id,)
            ).fetchone()
        return row["status"] if row else None

    # ---- 死信 ----

    def add_dead_letter(
        self,
        message_id: str,
        chat_id: str,
        reason: str,
        open_id: str = "",
        msg_type: str = "text",
        content_text: str = "",
    ) -> int:
        """被跳过的消息留档（含崩溃残留按已处理跳过的），供排查与重放。"""
        with _open(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO dead_letters"
                " (message_id, chat_id, open_id, msg_type, content_text, reason, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (message_id, chat_id, open_id, msg_type, content_text[:500], reason, time.time()),
            )
            return int(cur.lastrowid)

    # list_dead_letters 保留：test_bot.py / test_dedup.py 用它核验生产死信写入（add_dead_letter，活）。
    def list_dead_letters(
        self, chat_id: str, include_replayed: bool = False, limit: int = 20
    ) -> list[DeadLetter]:
        """死信列表（新的在前，序号即 /重放 N 的 N）。"""
        sql = "SELECT * FROM dead_letters WHERE chat_id = ?"
        if not include_replayed:
            sql += " AND replayed = 0"
        sql += " ORDER BY id DESC LIMIT ?"
        with _open(self.db_path) as conn:
            rows = conn.execute(sql, (chat_id, limit)).fetchall()
        return [self._to_dead_letter(r) for r in rows]

    @staticmethod
    def _to_dead_letter(row: sqlite3.Row) -> DeadLetter:
        return DeadLetter(
            id=row["id"], message_id=row["message_id"], chat_id=row["chat_id"],
            open_id=row["open_id"], msg_type=row["msg_type"],
            content_text=row["content_text"], reason=row["reason"],
            created_at=row["created_at"], replayed=bool(row["replayed"]),
        )


# ---------------------------------------------------------------------------
# 精读任务登记（断线/重启恢复）
# ---------------------------------------------------------------------------

_READING_TASKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS reading_tasks (
    dedup_key TEXT PRIMARY KEY,
    kind TEXT NOT NULL,                    -- arxiv | url | file
    ref TEXT NOT NULL,                     -- arxiv_id / 归一化链接 / 附件 file_key
    label TEXT NOT NULL,
    chat_id TEXT NOT NULL,
    open_id TEXT NOT NULL DEFAULT '',
    message_id TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'running',  -- running | done | failed
    created_ts REAL NOT NULL,
    finished_ts REAL
);
"""

READING_TASK_TTL_SECONDS = 7 * 24 * 3600  # reading_tasks 保留 7 天（登记时顺带清理）


@dataclass
class ReadingTask:
    dedup_key: str
    kind: str
    ref: str
    label: str
    chat_id: str
    open_id: str
    message_id: str
    status: str
    created_ts: float
    finished_ts: float | None


class ReadingTaskStore:
    """精读任务的 durable 登记：进程死在中途（status 停在 running）重启可恢复。

    时序约定：_start_async 启动时 upsert_start（同一查重键重读会重置回 running），
    _run_guarded 结束时 mark_finished（成功 done / 失败 failed）。
    语义边界：interrupted 只取 running——已 failed 的任务（重试耗尽走过失败路径）
    不恢复，防止恢复死循环。7 天 TTL，upsert_start 时顺带清理。启动时不清表。
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with _open(self.db_path) as conn:
            conn.executescript(_READING_TASKS_SCHEMA)

    def upsert_start(
        self,
        dedup_key: str,
        kind: str,
        ref: str,
        label: str,
        chat_id: str,
        open_id: str = "",
        message_id: str = "",
    ) -> None:
        """登记任务进入 running；同键重读重置为新一轮（状态/时间戳重来）。顺带清理过期。"""
        with _open(self.db_path) as conn:
            conn.execute(
                "DELETE FROM reading_tasks WHERE created_ts < ?",
                (time.time() - READING_TASK_TTL_SECONDS,),
            )
            conn.execute(
                "INSERT INTO reading_tasks"
                " (dedup_key, kind, ref, label, chat_id, open_id, message_id,"
                "  status, created_ts, finished_ts)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?, NULL)"
                " ON CONFLICT(dedup_key) DO UPDATE SET"
                " kind=excluded.kind, ref=excluded.ref, label=excluded.label,"
                " chat_id=excluded.chat_id, open_id=excluded.open_id,"
                " message_id=excluded.message_id, status='running',"
                " created_ts=excluded.created_ts, finished_ts=NULL",
                (dedup_key, kind, ref, label, chat_id, open_id, message_id, time.time()),
            )

    def mark_finished(self, dedup_key: str, ok: bool) -> None:
        """任务结束：成功 done / 失败 failed。failed 的不再被 interrupted 捞出。"""
        with _open(self.db_path) as conn:
            conn.execute(
                "UPDATE reading_tasks SET status = ?, finished_ts = ? WHERE dedup_key = ?",
                ("done" if ok else "failed", time.time(), dedup_key),
            )

    def interrupted(self, older_than: float | None = None) -> list[ReadingTask]:
        """进程死在中途的任务：status=running 且登记早于 older_than（默认现在）。

        older_than 传本次启动时间，可排除当前进程自己刚登记的在飞任务。
        """
        cutoff = older_than if older_than is not None else time.time()
        with _open(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM reading_tasks WHERE status = 'running' AND created_ts < ?"
                " ORDER BY created_ts ASC",
                (cutoff,),
            ).fetchall()
        return [self._to_task(r) for r in rows]

    def get(self, dedup_key: str) -> ReadingTask | None:
        """按查重键取一条（测试与排查用）。"""
        with _open(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM reading_tasks WHERE dedup_key = ?", (dedup_key,)
            ).fetchone()
        return self._to_task(row) if row else None

    def purge_old(self, ttl_seconds: float = READING_TASK_TTL_SECONDS) -> int:
        """删除超过 TTL 的记录（含僵死的 running：真实任务跑不过 1 小时），返回行数。"""
        with _open(self.db_path) as conn:
            cur = conn.execute(
                "DELETE FROM reading_tasks WHERE created_ts < ?",
                (time.time() - ttl_seconds,),
            )
            return cur.rowcount

    @staticmethod
    def _to_task(row: sqlite3.Row) -> ReadingTask:
        return ReadingTask(
            dedup_key=row["dedup_key"], kind=row["kind"], ref=row["ref"],
            label=row["label"], chat_id=row["chat_id"], open_id=row["open_id"],
            message_id=row["message_id"], status=row["status"],
            created_ts=row["created_ts"], finished_ts=row["finished_ts"],
        )


# ---------------------------------------------------------------------------
# 事件通知外发箱（失败/卡死/恢复通知必达，断网排队补发）
# ---------------------------------------------------------------------------

_OUTBOX_SCHEMA = """
CREATE TABLE IF NOT EXISTS notification_outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL,
    text TEXT NOT NULL,
    created_ts REAL NOT NULL,
    delivered_ts REAL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_attempt_ts REAL
);
"""

OUTBOX_DEDUP_WINDOW_SECONDS = 600  # 同 chat_id+text 在该窗口内不重复入队（事件级去抖）
OUTBOX_MAX_ATTEMPTS = 50  # 补发尝试上限（30s 一轮约 25 分钟轮询量，实际跨断网可多日）
OUTBOX_TTL_SECONDS = 24 * 3600  # 入队超过 24h 未送达：放弃（用户早已不在等这条）
OUTBOX_PURGE_SECONDS = 7 * 24 * 3600  # 记录保留 7 天（enqueue 时顺带清理）


@dataclass
class OutboxMessage:
    id: int
    chat_id: str
    text: str
    created_ts: float
    delivered_ts: float | None
    attempts: int
    last_attempt_ts: float | None


class NotificationOutbox:
    """事件通知的 durable 外发箱：先落库再补发，断网/重启不丢通知。

    语义：任务失败/卡死/恢复这类"用户必须知道"的事件先 enqueue 落库（进程死掉
    通知也在），后台 flusher 线程周期取出 pending 逐条 send_text——送达才
    mark_delivered，失败 mark_attempt 下轮再试；24h 未送达或 50 次尝试后放弃。
    enqueue 带 10 分钟同文去重（卡死报警/连续失败不刷屏），7 天 TTL 顺带清理。
    与卡片推送的关系：卡片是 best-effort 的"富结果"，外发箱是事件级的"必达文本"，
    两者互补不替代。启动时不清表：上次进程没发完的通知重启后继续补。
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with _open(self.db_path) as conn:
            conn.executescript(_OUTBOX_SCHEMA)

    def enqueue(self, chat_id: str, text: str) -> int | None:
        """通知入队，返回行 id；同 chat_id+text 在 10 分钟窗口内已入队过返回 None。
        顺带清理 7 天前的旧记录。"""
        now = time.time()
        with _open(self.db_path) as conn:
            conn.execute(
                "DELETE FROM notification_outbox WHERE created_ts < ?",
                (now - OUTBOX_PURGE_SECONDS,),
            )
            dup = conn.execute(
                "SELECT id FROM notification_outbox"
                " WHERE chat_id = ? AND text = ? AND created_ts > ?",
                (chat_id, text, now - OUTBOX_DEDUP_WINDOW_SECONDS),
            ).fetchone()
            if dup is not None:
                return None
            cur = conn.execute(
                "INSERT INTO notification_outbox (chat_id, text, created_ts)"
                " VALUES (?, ?, ?)",
                (chat_id, text, now),
            )
            return int(cur.lastrowid)

    def pending(self, limit: int = 20) -> list[OutboxMessage]:
        """待补发：未送达、尝试未封顶、未超 24h TTL；按入队顺序（先老后新）。"""
        with _open(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM notification_outbox"
                " WHERE delivered_ts IS NULL AND attempts < ? AND created_ts > ?"
                " ORDER BY id ASC LIMIT ?",
                (OUTBOX_MAX_ATTEMPTS, time.time() - OUTBOX_TTL_SECONDS, limit),
            ).fetchall()
        return [self._to_message(r) for r in rows]

    def mark_delivered(self, msg_id: int) -> None:
        """送达确认：记录 delivered_ts，不再补发。"""
        with _open(self.db_path) as conn:
            conn.execute(
                "UPDATE notification_outbox SET delivered_ts = ? WHERE id = ?",
                (time.time(), msg_id),
            )

    def mark_attempt(self, msg_id: int) -> None:
        """补发失败一次：attempts+1 并记时间（下一轮再试）。"""
        with _open(self.db_path) as conn:
            conn.execute(
                "UPDATE notification_outbox"
                " SET attempts = attempts + 1, last_attempt_ts = ? WHERE id = ?",
                (time.time(), msg_id),
            )

    def purge_old(self, ttl_seconds: float = OUTBOX_PURGE_SECONDS) -> int:
        """删除超过 TTL 的记录（含送不出去的），返回行数。"""
        with _open(self.db_path) as conn:
            cur = conn.execute(
                "DELETE FROM notification_outbox WHERE created_ts < ?",
                (time.time() - ttl_seconds,),
            )
            return cur.rowcount

    @staticmethod
    def _to_message(row: sqlite3.Row) -> OutboxMessage:
        return OutboxMessage(
            id=row["id"], chat_id=row["chat_id"], text=row["text"],
            created_ts=row["created_ts"], delivered_ts=row["delivered_ts"],
            attempts=row["attempts"], last_attempt_ts=row["last_attempt_ts"],
        )


# ---------------------------------------------------------------------------
# 会话系统
# ---------------------------------------------------------------------------

_SESSION_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL,
    title TEXT NOT NULL,
    bind_type TEXT NOT NULL DEFAULT 'general',   -- general | paper | career
    bind_ref TEXT NOT NULL DEFAULT '',           -- arxiv_id 等绑定对象
    status TEXT NOT NULL DEFAULT 'active',       -- active | archived
    summary TEXT NOT NULL DEFAULT '',            -- 滚动摘要
    custom_title INTEGER NOT NULL DEFAULT 0,     -- 1 = 用户改过名，不再自动起标题
    created_at REAL NOT NULL,
    last_active REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_chat ON sessions (chat_id, status, last_active DESC);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    chat_id TEXT NOT NULL,
    role TEXT NOT NULL,                          -- user | assistant
    content TEXT NOT NULL,
    tokens INTEGER NOT NULL DEFAULT 0,
    provider TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session_id, id);

CREATE TABLE IF NOT EXISTS chat_state (
    chat_id TEXT PRIMARY KEY,
    active_session_id INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS interview_states (
    session_id INTEGER PRIMARY KEY,
    state TEXT NOT NULL DEFAULT 'active',        -- active | ending | ended
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT '备忘',       -- 偏好 | 事实 | 项目 | 决定 | 备忘（口令粗判）
    content TEXT NOT NULL,
    created_at REAL NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 0,
    last_hit REAL
);
CREATE INDEX IF NOT EXISTS idx_memories_chat ON memories (chat_id, created_at DESC);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(content);
"""

_CJK_RE = re.compile(r"([一-鿿])")


def fts_tokenize(text: str) -> str:
    """CJK 字间加空格，让 FTS5 默认分词器能按字命中中文。"""
    return _CJK_RE.sub(r" \1 ", text)


def _fts_quote(term: str) -> str:
    """FTS5 字符串引用：双引号包围、内嵌引号翻倍（FTS5 唯一转义规则）。

    引用串内其他特殊字符（OR/AND/NOT、*、括号、^、: 等）全部字面化，
    用户输入无法改写 MATCH 表达式结构。
    """
    return '"' + term.replace('"', '""') + '"'


def _like_escape(text: str) -> str:
    """LIKE 模式转义（配合 ESCAPE '\\'）：\\、%、_ 字面化，用户输入成不了通配符。"""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


@dataclass
class Session:
    id: int
    chat_id: str
    title: str
    bind_type: str
    bind_ref: str
    status: str
    summary: str
    custom_title: bool
    created_at: float
    last_active: float
    message_count: int = 0  # list 时填充

    @property
    def code(self) -> str:
        """短码（日志/展示用）。"""
        return f"S{self.id}"


@dataclass
class Message:
    id: int
    session_id: int
    role: str
    content: str
    tokens: int
    provider: str
    model: str
    created_at: float


@dataclass
class Memory:
    id: int
    chat_id: str
    category: str
    content: str
    created_at: float
    hit_count: int
    last_hit: float | None


class SessionStore:
    """sessions / messages / chat_state / messages_fts / memories 的封装。"""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with _open(self.db_path) as conn:
            conn.executescript(_SESSION_SCHEMA)
            try:
                conn.executescript(_FTS_SCHEMA)
                self.fts_available = True
            except sqlite3.OperationalError:
                self.fts_available = False  # 极端环境无 FTS5，召回退化为 LIKE

    def _conn(self) -> sqlite3.Connection:
        return _open(self.db_path)

    # -- 行映射 --

    @staticmethod
    def _to_session(row: sqlite3.Row, message_count: int = 0) -> Session:
        return Session(
            id=row["id"], chat_id=row["chat_id"], title=row["title"],
            bind_type=row["bind_type"], bind_ref=row["bind_ref"], status=row["status"],
            summary=row["summary"], custom_title=bool(row["custom_title"]),
            created_at=row["created_at"], last_active=row["last_active"],
            message_count=message_count,
        )

    @staticmethod
    def _to_message(row: sqlite3.Row) -> Message:
        return Message(
            id=row["id"], session_id=row["session_id"], role=row["role"],
            content=row["content"], tokens=row["tokens"], provider=row["provider"],
            model=row["model"], created_at=row["created_at"],
        )

    # -- 会话 CRUD --

    def create_session(
        self,
        chat_id: str,
        title: str,
        bind_type: str = "general",
        bind_ref: str = "",
    ) -> Session:
        now = time.time()
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO sessions (chat_id, title, bind_type, bind_ref, created_at, last_active)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (chat_id, title, bind_type, bind_ref, now, now),
            )
            session_id = int(cur.lastrowid)
        return Session(session_id, chat_id, title, bind_type, bind_ref, "active", "", False, now, now)

    def get_session(self, session_id: int) -> Session | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return self._to_session(row, self.message_count(session_id)) if row else None

    def list_sessions(self, chat_id: str, include_archived: bool = False) -> list[Session]:
        """按最后活跃倒序（即「切换 N」的序号语义）。"""
        sql = (
            "SELECT s.*, (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS mc"
            " FROM sessions s WHERE s.chat_id = ?"
        )
        if not include_archived:
            sql += " AND s.status = 'active'"
        sql += " ORDER BY s.last_active DESC"
        with self._conn() as conn:
            rows = conn.execute(sql, (chat_id,)).fetchall()
        return [self._to_session(r, r["mc"]) for r in rows]

    def _touch(self, conn: sqlite3.Connection, session_id: int) -> None:
        conn.execute("UPDATE sessions SET last_active = ? WHERE id = ?", (time.time(), session_id))

    def rename_session(self, session_id: int, title: str, custom: bool = True) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE sessions SET title = ?, custom_title = ? WHERE id = ?",
                (title, 1 if custom else 0, session_id),
            )

    def archive_session(self, session_id: int) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE sessions SET status = 'archived' WHERE id = ?", (session_id,))

    def delete_session(self, session_id: int) -> None:
        """删除会话及其消息（含 FTS 索引）。"""
        with self._conn() as conn:
            if self.fts_available:
                conn.execute(
                    "DELETE FROM messages_fts WHERE rowid IN"
                    " (SELECT id FROM messages WHERE session_id = ?)",
                    (session_id,),
                )
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM interview_states WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.execute("DELETE FROM chat_state WHERE active_session_id = ?", (session_id,))

    def update_summary(self, session_id: int, summary: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE sessions SET summary = ? WHERE id = ?", (summary, session_id))

    # -- 活跃会话指针 --

    def get_active_session(self, chat_id: str) -> Session | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT s.* FROM chat_state cs JOIN sessions s ON s.id = cs.active_session_id"
                " WHERE cs.chat_id = ? AND s.status = 'active'",
                (chat_id,),
            ).fetchone()
        return self._to_session(row, self.message_count(row["id"])) if row else None

    def set_active_session(self, chat_id: str, session_id: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO chat_state (chat_id, active_session_id) VALUES (?, ?)"
                " ON CONFLICT(chat_id) DO UPDATE SET active_session_id = excluded.active_session_id",
                (chat_id, session_id),
            )
            self._touch(conn, session_id)

    # -- 模拟面试持久状态机 --

    def ensure_interview_state(self, session_id: int) -> None:
        """为 career 会话幂等建立持久状态；已有状态绝不覆盖。"""
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO interview_states (session_id, state, updated_at)"
                " VALUES (?, 'active', ?)",
                (session_id, time.time()),
            )

    def interview_state(self, session_id: int) -> str:
        """返回 active / ending / ended；旧会话默认 active。"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT state FROM interview_states WHERE session_id = ?", (session_id,)
            ).fetchone()
        return str(row["state"]) if row else "active"

    def begin_interview_end(self, session_id: int) -> bool:
        """原子抢占结束权；仅第一个 active→ending 的请求返回 True。"""
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO interview_states (session_id, state, updated_at)"
                " VALUES (?, 'active', ?)",
                (session_id, time.time()),
            )
            cur = conn.execute(
                "UPDATE interview_states SET state = 'ending', updated_at = ?"
                " WHERE session_id = ? AND state = 'active'",
                (time.time(), session_id),
            )
            return cur.rowcount == 1

    def reset_interview_end(self, session_id: int) -> None:
        """复盘失败后允许用户重试结束；已完成状态不回退。"""
        with self._conn() as conn:
            conn.execute(
                "UPDATE interview_states SET state = 'active', updated_at = ?"
                " WHERE session_id = ? AND state = 'ending'",
                (time.time(), session_id),
            )

    def complete_interview_end(self, session: Session, content: str) -> None:
        """在同一 SQLite 事务中持久化结束回复与 ended 状态。"""
        tokens = estimate_tokens(content)
        now = time.time()
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO messages"
                " (session_id, chat_id, role, content, tokens, provider, model, created_at)"
                " VALUES (?, ?, 'assistant', ?, ?, '', '', ?)",
                (session.id, session.chat_id, content, tokens, now),
            )
            msg_id = int(cur.lastrowid)
            if self.fts_available:
                conn.execute(
                    "INSERT INTO messages_fts (rowid, content) VALUES (?, ?)",
                    (msg_id, fts_tokenize(content)),
                )
            conn.execute(
                "INSERT INTO interview_states (session_id, state, updated_at)"
                " VALUES (?, 'ended', ?)"
                " ON CONFLICT(session_id) DO UPDATE SET state = 'ended', updated_at = excluded.updated_at",
                (session.id, now),
            )
            self._touch(conn, session.id)

    def get_or_create_default_session(self, chat_id: str) -> Session:
        """自由对话默认落点：每 chat 一个「日常」会话，首次自动建。

        活跃指针悬空时（如 /删除 当前会话后），优先复用最近活跃的既有会话，
        避免「日常」会话身份漂移、用户以为历史被清空。
        """
        active = self.get_active_session(chat_id)
        if active is not None:
            return active
        existing = self.list_sessions(chat_id)
        if existing:
            self.set_active_session(chat_id, existing[0].id)
            return existing[0]
        session = self.create_session(chat_id, "日常")
        self.set_active_session(chat_id, session.id)
        return session

    # -- 消息 --

    def add_message(
        self,
        session_id: int,
        chat_id: str,
        role: str,
        content: str,
        provider: str = "",
        model: str = "",
    ) -> int:
        tokens = estimate_tokens(content)
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO messages (session_id, chat_id, role, content, tokens, provider, model, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, chat_id, role, content, tokens, provider, model, time.time()),
            )
            msg_id = int(cur.lastrowid)
            if self.fts_available:
                conn.execute(
                    "INSERT INTO messages_fts (rowid, content) VALUES (?, ?)",
                    (msg_id, fts_tokenize(content)),
                )
            self._touch(conn, session_id)
        return msg_id

    def recent_messages(self, session_id: int, limit: int = 20) -> list[Message]:
        """最近 limit 条，按时间正序返回。"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM (SELECT * FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?)"
                " ORDER BY id ASC",
                (session_id, limit),
            ).fetchall()
        return [self._to_message(r) for r in rows]

    def all_messages(self, session_id: int) -> list[Message]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,)
            ).fetchall()
        return [self._to_message(r) for r in rows]

    def message_count(self, session_id: int) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM messages WHERE session_id = ?", (session_id,)
            ).fetchone()
        return int(row["c"])

    # -- 长期记忆 --

    @staticmethod
    def _to_memory(row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"], chat_id=row["chat_id"], category=row["category"],
            content=row["content"], created_at=row["created_at"],
            hit_count=row["hit_count"], last_hit=row["last_hit"],
        )

    def add_memory(self, chat_id: str, category: str, content: str) -> int:
        """写入长期记忆，返回行 id。同 chat 同文去重：已有则刷新 created_at（类别跟随最新），不新增。"""
        now = time.time()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id FROM memories WHERE chat_id = ? AND content = ?",
                (chat_id, content),
            ).fetchone()
            if row is not None:
                conn.execute(
                    "UPDATE memories SET created_at = ?, category = ? WHERE id = ?",
                    (now, category, row["id"]),
                )
                return int(row["id"])
            cur = conn.execute(
                "INSERT INTO memories (chat_id, category, content, created_at)"
                " VALUES (?, ?, ?, ?)",
                (chat_id, category, content, now),
            )
            return int(cur.lastrowid)

    def list_memories(self, chat_id: str, limit: int = 30) -> list[Memory]:
        """该 chat 的长期记忆，按创建时间倒序（新的在前；id 兜底同时间戳次序）。"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE chat_id = ?"
                " ORDER BY created_at DESC, id DESC LIMIT ?",
                (chat_id, limit),
            ).fetchall()
        return [self._to_memory(r) for r in rows]

    def delete_memory(self, chat_id: str, keyword: str) -> int:
        """按内容模糊匹配删除，返回删了几条。keyword 经转义，% _ 等通配符不生效。"""
        like = f"%{_like_escape(keyword)}%"
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM memories WHERE chat_id = ? AND content LIKE ? ESCAPE '\\'",
                (chat_id, like),
            )
            return cur.rowcount

    def touch_memory(self, memory_id: int) -> None:
        """命中计数 +1 并记时间（记忆被用上时调用）。"""
        with self._conn() as conn:
            conn.execute(
                "UPDATE memories SET hit_count = hit_count + 1, last_hit = ? WHERE id = ?",
                (time.time(), memory_id),
            )

    # -- 跨会话召回（FTS5）--

    def search_messages(
        self,
        chat_id: str,
        query: str,
        limit: int = 5,
        exclude_session_id: int | None = None,
    ) -> list[tuple[Message, str]]:
        """跨该 chat 全部会话检索，返回 (消息, 来源会话标题)。空查询/无索引返回 []。

        exclude_session_id 在 SQL 层先排除再 LIMIT，避免名额被当前会话命中占满。
        用户输入绝不直接拼进 SQL：值一律参数绑定；FTS MATCH 表达式与 LIKE 模式
        这两处必须拼字符串的，分别经 _fts_quote / _like_escape 转义。
        """
        terms = [t for t in fts_tokenize(query).split() if len(t) >= 1][:8]
        if not terms:
            return []
        with self._conn() as conn:
            if self.fts_available:
                match = " OR ".join(_fts_quote(t) for t in terms)
                sql = (
                    "SELECT m.*, s.title AS session_title FROM messages_fts f"
                    " JOIN messages m ON m.id = f.rowid"
                    " JOIN sessions s ON s.id = m.session_id"
                    " WHERE messages_fts MATCH ? AND m.chat_id = ?"
                )
                params: list = [match, chat_id]
            else:  # 无 FTS5 的退化路径：LIKE 逐词过滤
                like = f"%{_like_escape(query[:20])}%"
                sql = (
                    "SELECT m.*, s.title AS session_title FROM messages m"
                    " JOIN sessions s ON s.id = m.session_id"
                    " WHERE m.chat_id = ? AND m.content LIKE ? ESCAPE '\\'"
                )
                params = [chat_id, like]
            if exclude_session_id is not None:
                sql += " AND m.session_id != ?"
                params.append(exclude_session_id)
            sql += " ORDER BY m.id DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
        return [(self._to_message(r), r["session_title"]) for r in rows]
