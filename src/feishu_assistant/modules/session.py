"""会话系统（P2b）：多会话、隔离、滚动摘要、跨会话召回、任务绑定。

设计要点：
- 每 chat 一个默认「日常」会话（首次自动建）；自由对话落在当前活跃会话
- 上下文装配：系统人设(≤persona_chars) + 滚动摘要(≤summary_chars)
  + 最近 N 轮原文(≤recent_tokens，含已落库的当前消息；超出增量压缩进摘要，light 模型)
  + 跨会话召回（仅命中召回意图，≤recall_chars）；各层预算可配。
  消息历史唯一事实源是 store：chat 先落库当前消息再装配，不会重复进上下文
- 隔离：上下文只取活跃会话的消息（装配器只看 session_id）
- 召回：sqlite FTS5（CJK 按字索引），意图规则命中才检索，注入标注来源会话名
- 绑定会话：精读完成后自动建「《标题》精读」（bind_type=paper）并切换，
  该会话内所有消息都带论文全文上下文回答
- 标题：第 2 轮后 light 模型自动起 ≤10 字标题；「改名」覆盖后不再自动起
- 长期记忆：压缩时 light 模型顺带提炼「类别|内容」入 memories 表（解析失败只丢记忆，
  不影响摘要）；装配时 persona 之后、摘要之前全量注入（最新 30 条、≤memory_chars 字符，
  v1 不做检索）；口令「记住 xxx / 忘掉 xxx / 我的记忆」直接管理
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..kimi import CACHE_BREAK, KimiClient, KimiError, is_heavy_task
from ..store import Session, SessionStore
from .chat import load_system_prompt
from .paper import MessageContext

logger = logging.getLogger("feishu_assistant.session")

DEFAULT_SESSION_TITLE = "日常"
DELETE_CONFIRM_TTL = 120  # 删除二次确认有效期（秒）

# 召回意图规则：「上次聊的 xx」「之前那个结论」类表达
_RECALL_INTENT_RE = re.compile(r"上次|之前|以前|早先|刚才聊|那个结论|那段|聊过|说过|记得吗|提过")
_RECALL_STRIP_RE = re.compile(r"上次|之前|以前|早先|刚才|那个|那段|聊的|聊过|说过|记得吗|提过|我们|你|吗|呢|吧|的")

_SUMMARIZE_PROMPT = (
    "以下是对话的历史片段，请把它增量合并进已有摘要（保留关键事实、结论、用户偏好与待办，"
    "丢弃寒暄与过程性内容，摘要不超过 400 字）。\n\n"
    "【已有摘要】\n{summary}\n\n【历史片段】\n{history}\n\n"
    "【输出格式】\n"
    "先输出摘要正文。若历史片段中有值得长期记住的事项（用户的偏好、重要事实、项目进展、"
    "明确决定），在摘要之后另起一行输出「##长期记忆」，其下每行一条，格式「类别|内容」，"
    "类别限：偏好/事实/项目/决定；没有则不输出该部分。"
)
_TITLE_PROMPT = (
    "为以下对话起一个不超过 10 个字的标题（只输出标题本身，不要标点、不要书名号）：\n\n{dialog}"
)

_MEMORY_SECTION_RE = re.compile(r"#+\s*长期记忆")
_MEMORY_CATEGORIES = ("偏好", "事实", "项目", "决定")
MEMORY_INJECT_LIMIT = 30  # 注入条数上限（个人记忆量小，v1 全量注入不做检索）


def _split_summary_memories(text: str) -> tuple[str, list[tuple[str, str]]]:
    """拆压缩输出：「##长期记忆」之前是摘要（主输出），之后每行一条「类别|内容」。

    摘要是主输出、记忆是附加区：没有标记 / 行格式非法 / 类别不在白名单都只丢记忆，
    摘要原样保留。
    """
    m = _MEMORY_SECTION_RE.search(text)
    if m is None:
        return text.strip(), []
    items: list[tuple[str, str]] = []
    for line in text[m.end():].splitlines():
        line = line.strip().lstrip("-•*· ").strip()
        if "|" not in line:
            continue
        category, _, content = line.partition("|")
        category, content = category.strip(), content.strip()
        if category in _MEMORY_CATEGORIES and content:
            items.append((category, content))
    return text[: m.start()].strip(), items


def _guess_memory_category(content: str) -> str:
    """「记住」口令的粗判分类：命中偏好/决定关键词，否则落「备忘」。"""
    if re.search(r"偏好|我喜欢|我爱|我讨厌|不喜欢|习惯", content):
        return "偏好"
    if re.search(r"我决定|决定了|打算|不再", content):
        return "决定"
    return "备忘"


@dataclass
class ContextBudget:
    """上下文装配各层预算（字符/token），可在 config.yaml 的 session 段覆盖。"""

    persona_chars: int = 500
    summary_chars: int = 1500
    recent_rounds: int = 10
    recent_tokens: int = 6000
    recall_chars: int = 2000
    memory_chars: int = 1500

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ContextBudget":
        budget = cls()
        for key in ("persona_chars", "summary_chars", "recent_rounds", "recent_tokens",
                    "recall_chars", "memory_chars"):
            if key in d:
                setattr(budget, key, int(d[key]))
        return budget


@dataclass
class AssembledContext:
    """装配结果（messages 直接给模型；各层来源供测试断言）。"""

    messages: list[dict[str, str]]
    used_summary: bool
    used_recall: bool
    recent_count: int
    used_memory: bool = False


class SessionService:
    def __init__(
        self,
        kimi: KimiClient,
        store: SessionStore,
        budget: ContextBudget | None = None,
        persona: str | None = None,
    ) -> None:
        self.kimi = kimi
        self.store = store
        self.budget = budget or ContextBudget()
        self.persona = persona if persona is not None else load_system_prompt()
        # 论文绑定会话的问答回调（main.py 注入 paper_service.answer_in_session）
        self.paper_qa_fn: Callable[[Session, str], str] | None = None
        # 删除二次确认：chat_id → (session_id, title, 过期时间)；
        # 取消策略在 router 入口（任何非「确认删除」输入都调 cancel_pending_delete）
        self._pending_delete: dict[str, tuple[int, str, float]] = {}

    # ------------------------------------------------------------------
    # 命令（router 注册表调用）
    # ------------------------------------------------------------------

    def cmd_new_session(self, arg: str, ctx: MessageContext) -> str:
        title = arg.strip() or "新会话"
        session = self.store.create_session(ctx.chat_id, title)
        self.store.set_active_session(ctx.chat_id, session.id)
        logger.info("新会话：chat=%s session=%s「%s」", ctx.chat_id, session.code, title)
        return f"✅ 已创建并切换到会话「{title}」（{session.code}）。发「会话列表」查看全部会话。"

    def cmd_list(self, ctx: MessageContext) -> str:
        sessions = self.store.list_sessions(ctx.chat_id)
        if not sessions:
            return "还没有会话，随便聊一句就会自动建立「日常」会话。"
        active = self.store.get_active_session(ctx.chat_id)
        lines = []
        for i, s in enumerate(sessions, 1):
            mark = "👉 " if active and s.id == active.id else ""
            bind = {"general": "", "paper": "📄论文", "career": "💼求职"}.get(s.bind_type, s.bind_type)
            bind_text = f" · {bind}" if bind else ""
            lines.append(
                f"{mark}{i}. 「{s.title}」({s.code}) · {s.message_count} 条消息{bind_text}"
                f" · 最后活跃 {_fmt_time(s.last_active)}"
            )
        body = "\n".join(lines)
        return (
            f"📋 会话列表（共 {len(sessions)} 个）\n{body}\n\n"
            "管理：切换 N / 归档 N / 删除 N；新建：新会话 [标题]"
        )

    def cmd_switch(self, arg: str, ctx: MessageContext) -> str:
        n = _parse_index(arg)
        sessions = self.store.list_sessions(ctx.chat_id)
        if n is None or not (1 <= n <= len(sessions)):
            return f"没有第 {arg} 个会话。发「会话列表」查看序号。"
        target = sessions[n - 1]
        self.store.set_active_session(ctx.chat_id, target.id)
        logger.info("切换会话：chat=%s → %s「%s」", ctx.chat_id, target.code, target.title)
        return f"✅ 已切换到会话「{target.title}」（{target.code}）。"

    def cmd_rename(self, arg: str, ctx: MessageContext) -> str:
        new_title = arg.strip()
        if not new_title:
            return "用法：改名 新标题（重命名当前会话）"
        session = self.store.get_active_session(ctx.chat_id)
        if session is None:
            return "当前没有活跃会话，先聊一句或「新会话」。"
        self.store.rename_session(session.id, new_title[:20], custom=True)
        return f"✅ 当前会话已改名为「{new_title[:20]}」。"

    def cmd_archive(self, arg: str, ctx: MessageContext) -> str:
        n = _parse_index(arg)
        sessions = self.store.list_sessions(ctx.chat_id)
        if n is None or not (1 <= n <= len(sessions)):
            return f"没有第 {arg} 个会话。发「会话列表」查看序号。"
        target = sessions[n - 1]
        active = self.store.get_active_session(ctx.chat_id)  # 归档前先记录当前
        was_active = active is not None and active.id == target.id
        self.store.archive_session(target.id)
        note = ""
        if was_active:
            # 归档的是当前会话：切回日常
            default = self._ensure_default(ctx.chat_id, exclude=target.id)
            self.store.set_active_session(ctx.chat_id, default.id)
            note = f"\n已自动切换回「{default.title}」。"
        logger.info("归档会话：chat=%s %s「%s」", ctx.chat_id, target.code, target.title)
        return f"🗄 已归档会话「{target.title}」（发「会话列表」不再显示，数据保留）。{note}"

    def cmd_delete(self, arg: str, ctx: MessageContext) -> str:
        n = _parse_index(arg)
        sessions = self.store.list_sessions(ctx.chat_id)
        if n is None or not (1 <= n <= len(sessions)):
            return f"没有第 {arg} 个会话。发「会话列表」查看序号。"
        target = sessions[n - 1]
        self._pending_delete[ctx.chat_id] = (
            target.id, target.title, time.time() + DELETE_CONFIRM_TTL
        )
        return (
            f"⚠️ 确认删除会话「{target.title}」（{target.message_count} 条消息，不可恢复）？\n"
            f"回复「确认删除」生效（{DELETE_CONFIRM_TTL // 60} 分钟内），回复其他内容取消。"
        )

    def cmd_delete_confirm(self, ctx: MessageContext) -> str:
        pending = self._pending_delete.pop(ctx.chat_id, None)
        if pending is None:
            return "没有待确认的删除操作。先发「删除 N」选择要删除的会话。"
        session_id, title, deadline = pending
        if time.time() > deadline:
            return "删除确认已过期，请重新发「删除 N」。"
        self.store.delete_session(session_id)
        active = self.store.get_active_session(ctx.chat_id)
        note = ""
        if active is None:  # 删掉的正是当前会话：回日常
            default = self._ensure_default(ctx.chat_id)
            self.store.set_active_session(ctx.chat_id, default.id)
            note = f"\n已自动切换回「{default.title}」。"
        logger.info("删除会话：chat=%s session_id=%s「%s」", ctx.chat_id, session_id, title)
        return f"🗑 已删除会话「{title}」及其全部消息。{note}"

    def cancel_pending_delete(self, chat_id: str) -> str | None:
        """取消该 chat 待确认的删除，返回被取消的会话标题；没有待确认返回 None。

        由 router 在消息入口统一调用：「删除 N」后任何非「确认删除」的输入都视为取消。
        """
        pending = self._pending_delete.pop(chat_id, None)
        return pending[1] if pending is not None else None

    # -- 长期记忆口令（「记住」≠ inbox 的「记一下」，后者写备忘库不动）--

    def cmd_remember(self, arg: str, ctx: MessageContext) -> str:
        content = arg.strip()
        if not content:
            return "用法：记住 内容（写入长期记忆，之后对话会自动带上）。"
        category = _guess_memory_category(content)
        self.store.add_memory(ctx.chat_id, category, content)
        logger.info("记住：chat=%s [%s] %s", ctx.chat_id, category, content[:30])
        return f"✅ 已记住（{category}）：{content}\n发「我的记忆」查看全部，「忘掉 关键词」删除。"

    def cmd_forget(self, arg: str, ctx: MessageContext) -> str:
        keyword = arg.strip()
        if not keyword:
            return "用法：忘掉 关键词（按内容模糊匹配删除长期记忆）。"
        n = self.store.delete_memory(ctx.chat_id, keyword)
        if n == 0:
            return f"没有找到含「{keyword}」的记忆。发「我的记忆」查看全部。"
        logger.info("忘掉：chat=%s 关键词「%s」删除 %d 条", ctx.chat_id, keyword, n)
        return f"🗑 已忘掉 {n} 条含「{keyword}」的记忆。"

    def cmd_memories(self, ctx: MessageContext) -> str:
        memories = self.store.list_memories(ctx.chat_id, limit=100)
        if not memories:
            return "还没有长期记忆。发「记住 xxx」写入第一条。"
        lines = [f"{i}. 【{m.category}】{m.content}" for i, m in enumerate(memories, 1)]
        return (
            f"🧠 长期记忆（共 {len(memories)} 条，新的在前）\n" + "\n".join(lines)
            + "\n管理：记住 xxx / 忘掉 关键词"
        )

    # ------------------------------------------------------------------
    # 自由对话入口（替换原 chat_fn）
    # ------------------------------------------------------------------

    def chat(self, text: str, ctx: MessageContext | None) -> str:
        if ctx is None or not ctx.chat_id:
            return self._plain_chat(text)  # 无上下文（测试/异常路径）退化为单轮
        session = self.store.get_or_create_default_session(ctx.chat_id)
        # 当前消息先落库：装配器只从 store 读历史（唯一事实源），不再单独追加
        self.store.add_message(session.id, ctx.chat_id, "user", text)

        try:
            if session.bind_type == "paper" and self.paper_qa_fn is not None:
                # 绑定精读问答：传入最近消息供装配对话记录（当前问题已落库，
                # 尾随未答的用户消息不会成对，天然被排除在记录外）
                reply = self.paper_qa_fn(
                    session, text,
                    history_msgs=self.store.recent_messages(session.id, limit=12))
                provider, model = "", ""
            else:
                recall = self.recall(ctx.chat_id, text, exclude_session_id=session.id)
                assembled = self.assemble_context(session, recall)
                result = self.kimi.chat_full(assembled.messages, heavy=is_heavy_task(text))
                reply, provider, model = result.content, result.provider, result.model
        except KimiError as exc:
            reply = f"调用大模型出错了：{exc}"
            provider, model = "", ""

        self.store.add_message(session.id, ctx.chat_id, "assistant", reply,
                               provider=provider, model=model)
        self.maybe_compress(session)
        self.maybe_autotitle(session)
        return reply

    def _plain_chat(self, text: str) -> str:
        messages = [
            {"role": "system", "content": self.persona[: self.budget.persona_chars]},
            {"role": "user", "content": text},
        ]
        try:
            return self.kimi.chat(messages, heavy=is_heavy_task(text))
        except KimiError as exc:
            return f"调用大模型出错了：{exc}"

    # ------------------------------------------------------------------
    # 上下文装配
    # ------------------------------------------------------------------

    def assemble_context(
        self, session: Session, recall_text: str | None = None
    ) -> AssembledContext:
        """系统人设 + 长期记忆 + 滚动摘要 + 最近 N 轮（token 预算内，含已落库的当前消息）+ 召回片段。

        消息历史唯一事实源是 store：调用方须先落库当前消息，装配器不再单独追加，
        模型看到的历史与库内完全一致（不重复、不遗漏）。
        """
        b = self.budget
        system_parts = [self.persona[: b.persona_chars]]
        if session.bind_type == "paper":
            system_parts.append(f"当前会话「{session.title}」绑定论文精读，回答优先基于论文内容。")
        memory_block = self._memory_block(session.chat_id)
        if memory_block:
            system_parts.append(memory_block)
        if session.summary:
            system_parts.append(
                f"当前会话「{session.title}」前情摘要：\n{session.summary[: b.summary_chars]}"
            )
        if recall_text:
            system_parts.append(f"跨会话相关记忆（供参考，已标注来源会话）：\n{recall_text[: b.recall_chars]}")

        messages: list[dict[str, str]] = [{"role": "system", "content": "\n\n".join(system_parts)}]
        kept = self._recent_within_budget(session)
        messages += [{"role": m.role, "content": m.content} for m in kept]
        # 增量缓存断点：哨兵标记「稳定前缀 | 当前问题」分界。多轮对话的历史是
        # 只增不改的，前轮缓存的前缀本轮继续命中——每轮只有最新一轮问答按原价计算。
        # 人设/记忆/摘要在前缀区：它们变化频率低，变化后下一轮自动重缓存。
        anchor = messages[-2] if len(messages) > 1 and messages[-1]["role"] == "user" \
            else messages[0]
        anchor["content"] = anchor["content"] + "\n" + CACHE_BREAK
        return AssembledContext(messages, bool(session.summary), bool(recall_text), len(kept),
                                memory_block is not None)

    def _memory_block(self, chat_id: str) -> str | None:
        """长期记忆注入区：最新 N 条按时间倒序全量注入，总预算 memory_chars（超出丢最旧的）。

        与召回同理：存储异常只记日志按无记忆处理，不拖垮整轮装配。
        """
        try:
            memories = self.store.list_memories(chat_id, limit=MEMORY_INJECT_LIMIT)
        except sqlite3.Error as exc:
            logger.warning("长期记忆读取失败（按无记忆继续）：%s", exc)
            return None
        if not memories:
            return None
        lines: list[str] = []
        total = 0
        for m in memories:  # 新的在前：预算耗尽时最旧的先被丢
            line = f"【{m.category}】{m.content}"
            if lines and total + len(line) > self.budget.memory_chars:
                break
            lines.append(line)
            total += len(line)
        return (
            "## 长期记忆\n以下是关于用户的长期记忆，供参考，不要逐条复述给用户：\n"
            + "\n".join(lines)
        )

    def _recent_within_budget(self, session: Session):
        """最近 recent_rounds 轮且总 tokens ≤ recent_tokens（从最新往回装）。"""
        msgs = self.store.recent_messages(session.id, self.budget.recent_rounds * 2)
        kept: list = []
        total = 0
        for m in reversed(msgs):
            if total + m.tokens > self.budget.recent_tokens and kept:
                break
            kept.append(m)
            total += m.tokens
        kept.reverse()
        return kept

    # ------------------------------------------------------------------
    # 增量压缩（旧消息 → 滚动摘要，light 模型）
    # ------------------------------------------------------------------

    def maybe_compress(self, session: Session) -> bool:
        """旧消息（最近窗口之外）tokens 超预算时，增量压缩进 summary。返回是否触发。"""
        msgs = self.store.all_messages(session.id)
        window = self.budget.recent_rounds * 2
        if len(msgs) <= window:
            return False
        old = msgs[:-window]
        if sum(m.tokens for m in old) < self.budget.recent_tokens:
            return False
        history = "\n".join(
            f"{'用户' if m.role == 'user' else '助理'}: {m.content}" for m in old
        )[:12_000]
        fresh = self.store.get_session(session.id) or session
        try:
            raw = self.kimi.chat(
                [{"role": "user", "content": _SUMMARIZE_PROMPT.format(
                    summary=fresh.summary or "（空）", history=history)}],
                heavy=False,
            )
        except KimiError as exc:
            logger.warning("摘要压缩失败（下轮再试）：%s", exc)
            return False
        # 摘要是主输出、记忆是附加区：拆不出摘要正文时保留旧摘要，记忆照入
        new_summary, memory_items = _split_summary_memories(raw)
        if new_summary:
            self.store.update_summary(session.id, new_summary)
        else:
            logger.warning("压缩输出缺摘要正文，保留旧摘要（会话 %s）", session.code)
        for category, content in memory_items:
            try:
                self.store.add_memory(session.chat_id, category, content)
            except sqlite3.Error as exc:
                logger.warning("长期记忆入库失败（跳过该条）：%s", exc)
        logger.info("会话 %s 摘要已压缩更新（%d 条旧消息 → %d 字符，提炼记忆 %d 条）",
                    session.code, len(old), len(new_summary), len(memory_items))
        return True

    # ------------------------------------------------------------------
    # 标题自动生成（第 2 轮后，light 模型，≤10 字）
    # ------------------------------------------------------------------

    def maybe_autotitle(self, session: Session) -> bool:
        fresh = self.store.get_session(session.id) or session
        if fresh.custom_title or fresh.bind_type != "general":
            return False  # 用户改过名 / 绑定会话（标题已有意义）不动
        if fresh.title != DEFAULT_SESSION_TITLE and not fresh.title.startswith("新会话"):
            return False
        msgs = self.store.recent_messages(session.id, 6)
        if len(msgs) < 4:  # 第 2 轮（一问一答 ×2）后再起
            return False
        dialog = "\n".join(f"{'用户' if m.role == 'user' else '助理'}: {m.content[:200]}" for m in msgs)
        try:
            title = self.kimi.chat(
                [{"role": "user", "content": _TITLE_PROMPT.format(dialog=dialog)}], heavy=False
            )
        except KimiError as exc:
            logger.warning("自动起标题失败（下轮再试）：%s", exc)
            return False
        title = title.strip().strip("《》\"'，。！？ ")[:10]
        if not title or title == fresh.title:
            return False
        self.store.rename_session(session.id, title, custom=False)
        logger.info("会话 %s 自动起名：「%s」", session.code, title)
        return True

    # ------------------------------------------------------------------
    # 跨会话召回（FTS5，意图命中才检索）
    # ------------------------------------------------------------------

    def recall_intent(self, text: str) -> bool:
        return bool(_RECALL_INTENT_RE.search(text))

    def recall(self, chat_id: str, text: str, exclude_session_id: int | None = None) -> str | None:
        """命中召回意图时跨全部会话检索，返回带来源标注的片段；否则 None。

        exclude_session_id 下推到 SQL 层（先排除再取 top N），当前会话命中不占名额。
        召回只是增强：底层检索异常只记日志并返回 None，绝不让整轮对话崩掉。
        """
        if not self.recall_intent(text):
            return None
        query = _RECALL_STRIP_RE.sub(" ", text).strip() or text
        try:
            hits = self.store.search_messages(
                chat_id, query, limit=3, exclude_session_id=exclude_session_id)
            if not hits and query != text:
                hits = self.store.search_messages(
                    chat_id, text, limit=3, exclude_session_id=exclude_session_id)
        except sqlite3.Error as exc:
            logger.warning("跨会话召回检索失败（按无召回继续）：%s", exc)
            return None
        snippets = []
        for msg, session_title in hits:
            who = "用户" if msg.role == "user" else "助理"
            snippets.append(f"【会话「{session_title}」·{who}】{msg.content[:200]}")
        if not snippets:
            return None
        logger.info("召回命中：chat=%s 片段=%d", chat_id, len(snippets))
        return "\n".join(snippets)

    # ------------------------------------------------------------------
    # 任务绑定会话
    # ------------------------------------------------------------------

    def bind_paper_session(self, ctx: MessageContext, title: str, arxiv_id: str) -> str:
        """精读完成后：建绑定会话「《标题》精读」并切换，返回给用户的提示语。"""
        session = self.store.create_session(
            ctx.chat_id, f"《{title}》精读", bind_type="paper", bind_ref=arxiv_id
        )
        self.store.set_active_session(ctx.chat_id, session.id)
        logger.info("绑定精读会话：chat=%s %s「%s」", ctx.chat_id, session.code, session.title)
        return (
            f"💬 已切换到新会话「{session.title}」，追问本文细节直接问即可；"
            "发「会话列表」→「切换 N」可回到之前的会话。"
        )

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _ensure_default(self, chat_id: str, exclude: int | None = None) -> Session:
        """找或建「日常」会话（不经过活跃指针）。"""
        for s in self.store.list_sessions(chat_id):
            if s.title == DEFAULT_SESSION_TITLE and s.id != exclude:
                return s
        return self.store.create_session(chat_id, DEFAULT_SESSION_TITLE)


def _parse_index(arg: str) -> int | None:
    m = re.fullmatch(r"\s*(\d+)\s*", arg or "")
    return int(m.group(1)) if m else None


def _fmt_time(ts: float) -> str:
    return time.strftime("%m-%d %H:%M", time.localtime(ts))
