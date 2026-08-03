"""会话系统测试：命令、隔离、装配器预算、压缩、标题、FTS 召回、绑定会话。全部离线。"""

from __future__ import annotations

import sqlite3
import time

import pytest

from feishu_assistant.modules.paper import MessageContext
from feishu_assistant.modules.router import Router
from feishu_assistant.modules.session import (
    DEFAULT_SESSION_TITLE,
    ContextBudget,
    SessionService,
)
from feishu_assistant.store import SessionStore

CTX = MessageContext(chat_id="c1", open_id="ou_x", message_id="m1")


class FakeKimi:
    """chat_full 返回带 provider/model 的结果；chat 用于压缩/起标题。"""

    def __init__(self, reply="模型回复", title="自动标题"):
        self.reply = reply
        self.title = title
        self.full_calls: list[dict] = []
        self.chat_calls: list[dict] = []

    def chat_full(self, messages, *, heavy=False, json_mode=False):
        self.full_calls.append({"messages": messages, "heavy": heavy})
        from feishu_assistant.kimi import ChatResult
        return ChatResult(self.reply, "", "free-tokens", "kimi-k3-highspeed")

    def chat(self, messages, *, heavy=False, json_mode=False):
        self.chat_calls.append({"messages": messages, "heavy": heavy})
        return self.title


def make_service(tmp_path, kimi=None, **budget_kw) -> tuple[SessionService, FakeKimi, SessionStore]:
    kimi = kimi or FakeKimi()
    store = SessionStore(tmp_path / "s.db")
    budget = ContextBudget(**budget_kw) if budget_kw else ContextBudget()
    svc = SessionService(kimi, store, budget=budget, persona="你是个人助理。" * 100)
    return svc, kimi, store


# ---- 命令流程 ----

def test_default_session_auto_created_on_first_chat(tmp_path):
    svc, kimi, _ = make_service(tmp_path)
    assert svc.chat("你好", CTX) == "模型回复"
    active = svc.store.get_active_session("c1")
    assert active is not None and active.title == DEFAULT_SESSION_TITLE
    assert active.message_count == 2  # 一问一答都落库


def test_new_session_and_switch(tmp_path):
    svc, _, _ = make_service(tmp_path)
    svc.chat("先聊一句", CTX)  # 首次聊天自动建「日常」
    assert svc.store.get_active_session("c1").title == DEFAULT_SESSION_TITLE
    assert "已创建并切换" in svc.cmd_new_session("论文讨论", CTX)
    assert svc.store.get_active_session("c1").title == "论文讨论"

    sessions = svc.store.list_sessions("c1")
    assert [s.title for s in sessions] == ["论文讨论", DEFAULT_SESSION_TITLE]  # 最后活跃在前
    assert "已切换" in svc.cmd_switch("2", CTX)  # 切回日常
    assert svc.store.get_active_session("c1").title == DEFAULT_SESSION_TITLE
    assert "没有第" in svc.cmd_switch("99", CTX)


def test_list_display(tmp_path):
    svc, _, _ = make_service(tmp_path)
    svc.chat("你好", CTX)
    svc.cmd_new_session("项目A", CTX)
    reply = svc.cmd_list(CTX)
    assert "项目A" in reply and DEFAULT_SESSION_TITLE in reply
    assert "条消息" in reply and "👉" in reply  # 当前会话有标记


def test_rename_sets_custom_title(tmp_path):
    svc, _, _ = make_service(tmp_path)
    svc.chat("你好", CTX)
    assert "已改名" in svc.cmd_rename("我的专题", CTX)
    s = svc.store.get_active_session("c1")
    assert s.title == "我的专题" and s.custom_title


def test_archive_and_fallback_to_default(tmp_path):
    svc, _, _ = make_service(tmp_path)
    svc.chat("你好", CTX)
    svc.cmd_new_session("临时", CTX)
    sessions = svc.store.list_sessions("c1")
    assert sessions[0].title == "临时"
    reply = svc.cmd_archive("1", CTX)  # 归档当前会话
    assert "已归档" in reply and "日常" in reply  # 自动切回日常
    assert svc.store.get_active_session("c1").title == DEFAULT_SESSION_TITLE
    assert len(svc.store.list_sessions("c1")) == 1  # 归档的不显示
    assert len(svc.store.list_sessions("c1", include_archived=True)) == 2  # 数据保留


def test_delete_with_confirm_flow(tmp_path):
    svc, _, _ = make_service(tmp_path)
    svc.chat("你好", CTX)
    svc.cmd_new_session("要删的", CTX)
    reply = svc.cmd_delete("1", CTX)
    assert "确认删除" in reply and "要删的" in reply
    assert svc.cmd_delete_confirm(CTX).startswith("🗑")
    assert all(s.title != "要删的" for s in svc.store.list_sessions("c1", include_archived=True))


def test_delete_confirm_expired(tmp_path):
    svc, _, _ = make_service(tmp_path)
    svc.chat("你好", CTX)
    svc.cmd_delete("1", CTX)
    svc._pending_delete["c1"] = (svc._pending_delete["c1"][0],
                                 svc._pending_delete["c1"][1], time.time() - 1)  # 强制过期
    assert "已过期" in svc.cmd_delete_confirm(CTX)


def test_delete_cancelled_by_other_message(tmp_path):
    """「删除 N」后回复其他内容 = 取消删除；取消策略在 router 入口统一执行。"""
    svc, _, _ = make_service(tmp_path)
    router = Router(chat_fn=svc.chat, session_svc=svc)
    svc.chat("你好", CTX)
    svc.cmd_delete("1", CTX)
    reply = router.handle("随便聊聊", CTX)
    assert "已取消删除" in reply
    assert svc.cmd_delete_confirm(CTX).startswith("没有待确认")


def test_delete_active_session_falls_back(tmp_path):
    svc, _, _ = make_service(tmp_path)
    svc.chat("你好", CTX)  # 建日常
    svc.cmd_new_session("要删的", CTX)
    svc.cmd_delete("1", CTX)
    svc.cmd_delete_confirm(CTX)
    assert svc.store.get_active_session("c1").title == DEFAULT_SESSION_TITLE


# ---- 隔离 ----

def test_sessions_isolated(tmp_path):
    svc, kimi, store = make_service(tmp_path)
    svc.chat("A会话的独有话题：量子退火", CTX)  # 日常
    svc.cmd_new_session("B会话", CTX)
    svc.chat("B会话的内容", CTX)

    b = store.get_active_session("c1")
    assembled_b = svc.assemble_context(b)
    all_text = str(assembled_b.messages)
    assert "B会话的内容" in all_text
    assert "量子退火" not in all_text  # A 的消息不泄漏进 B

    svc.cmd_switch("2", CTX)  # 切回日常（A）
    a = store.get_active_session("c1")
    assembled_a = svc.assemble_context(a)
    assert "量子退火" in str(assembled_a.messages)
    assert "B会话的内容" not in str(assembled_a.messages)


# ---- 装配器预算 ----

def test_assembler_layers_and_persona_budget(tmp_path):
    svc, _, store = make_service(tmp_path, persona_chars=10)
    svc.chat("你好", CTX)
    session = store.get_active_session("c1")
    store.add_message(session.id, "c1", "user", "新问题")  # 当前消息先落库（与 chat 流程一致）
    assembled = svc.assemble_context(session, recall_text="召回片段")
    system = assembled.messages[0]["content"]
    assert len("你是个人助理。" * 100) > 10
    assert system.startswith("你是个人助理。"[:5])  # 被预算截断
    assert "召回片段" in system and assembled.used_recall
    assert assembled.messages[-1] == {"role": "user", "content": "新问题"}
    contents = [m["content"] for m in assembled.messages]
    assert contents.count("新问题") == 1  # 只出现一次（修复前给模型发两遍）


def test_assembler_summary_included(tmp_path):
    svc, _, store = make_service(tmp_path)
    session = store.get_or_create_default_session("c1")
    store.update_summary(session.id, "用户是Agent方向研究生" * 100)
    session = store.get_active_session("c1")
    assembled = svc.assemble_context(session)
    assert "前情摘要" in assembled.messages[0]["content"] and assembled.used_summary


def test_assembler_recent_window_and_token_budget(tmp_path):
    svc, _, store = make_service(tmp_path, recent_rounds=2, recent_tokens=50)
    session = store.get_or_create_default_session("c1")
    for i in range(6):
        store.add_message(session.id, "c1", "user", f"第{i}条" + "x" * 30)
    store.add_message(session.id, "c1", "user", "现在")  # 当前消息先落库（与 chat 流程一致）
    assembled = svc.assemble_context(session)
    contents = [m["content"] for m in assembled.messages[1:-1]]
    assert "第5条" + "x" * 30 in contents
    assert "第0条" + "x" * 30 not in contents  # 旧消息被窗口/预算裁掉
    assert assembled.recent_count <= 4  # 最多 recent_rounds*2 条


# ---- 增量压缩 ----

def test_compress_triggered_over_budget(tmp_path):
    svc, kimi, store = make_service(tmp_path, recent_rounds=1, recent_tokens=20)
    session = store.get_or_create_default_session("c1")
    for i in range(6):
        store.add_message(session.id, "c1", "user" if i % 2 == 0 else "assistant",
                          "历史内容" + "x" * 30)
    assert svc.maybe_compress(session) is True
    assert kimi.chat_calls and kimi.chat_calls[0]["heavy"] is False  # 压缩走 light
    assert store.get_session(session.id).summary == "自动标题"


def test_compress_not_triggered_under_budget(tmp_path):
    svc, kimi, store = make_service(tmp_path)
    session = store.get_or_create_default_session("c1")
    store.add_message(session.id, "c1", "user", "短")
    assert svc.maybe_compress(session) is False
    assert kimi.chat_calls == []


# ---- 标题自动生成 ----

def test_autotitle_after_second_round(tmp_path):
    kimi = FakeKimi(title="论文讨论")
    svc, _, store = make_service(tmp_path, kimi=kimi)
    svc.chat("聊聊 Transformer", CTX)   # 第 1 轮，不起名
    s = store.get_active_session("c1")
    assert s.title == DEFAULT_SESSION_TITLE
    svc.chat("注意力机制呢", CTX)          # 第 2 轮，触发起名
    assert store.get_active_session("c1").title == "论文讨论"
    assert kimi.chat_calls  # light 模型被调用


def test_autotitle_skips_custom_and_bound(tmp_path):
    svc, _, store = make_service(tmp_path)
    svc.chat("你好", CTX)
    svc.cmd_rename("我自己起的", CTX)
    svc.chat("继续", CTX)
    assert store.get_active_session("c1").title == "我自己起的"  # 不被覆盖


# ---- 召回（FTS5）----

def test_recall_across_sessions(tmp_path):
    svc, _, store = make_service(tmp_path)
    svc.chat("给我讲讲马尔可夫决策过程", CTX)  # 日常会话聊过
    svc.cmd_new_session("全新话题", CTX)        # 切走
    svc.chat("今天天气怎么样", CTX)
    recall = svc.recall("c1", "上次聊的马尔可夫是什么", exclude_session_id=None)
    assert recall is not None
    assert "马尔可夫" in recall
    assert f"会话「{DEFAULT_SESSION_TITLE}」" in recall  # 标注来源会话名


def test_recall_only_on_intent(tmp_path):
    svc, _, _ = make_service(tmp_path)
    svc.chat("给我讲讲马尔可夫", CTX)
    assert svc.recall("c1", "马尔可夫是什么") is None  # 无意图词，不召回
    assert svc.recall("c1", "之前那个马尔可夫") is not None


def test_recall_excludes_current_session(tmp_path):
    svc, _, store = make_service(tmp_path)
    svc.chat("讲讲贝叶斯", CTX)
    current = store.get_active_session("c1")
    assert svc.recall("c1", "上次说的贝叶斯", exclude_session_id=current.id) is None


def test_fts_tokenize_chinese():
    from feishu_assistant.store import fts_tokenize
    assert fts_tokenize("马尔可夫") == " 马  尔  可  夫 "
    assert "abc" in fts_tokenize("abc")  # 英文不受影响


# ---- 绑定会话 ----

def test_bind_paper_session(tmp_path):
    svc, _, store = make_service(tmp_path)
    svc.chat("日常开场", CTX)
    note = svc.bind_paper_session(CTX, "AutoHarness", "2603.03329")
    assert "《AutoHarness》精读" in note
    active = store.get_active_session("c1")
    assert active.bind_type == "paper" and active.bind_ref == "2603.03329"
    assert active.title == "《AutoHarness》精读"


def test_bound_session_chat_uses_paper_qa(tmp_path):
    svc, kimi, store = make_service(tmp_path)
    qa_calls = []
    svc.paper_qa_fn = lambda session, text: qa_calls.append(text) or "论文问答回复"
    svc.bind_paper_session(CTX, "AutoHarness", "2603.03329")
    reply = svc.chat("第三章怎么算的", CTX)
    assert reply == "论文问答回复"
    assert qa_calls == ["第三章怎么算的"]
    assert kimi.full_calls == []  # 没走普通对话路径
    # 消息仍落库到绑定会话
    assert store.get_active_session("c1").message_count == 2


def test_general_session_chat_path_with_provider_model(tmp_path):
    svc, kimi, store = make_service(tmp_path)
    svc.chat("你好", CTX)
    msgs = store.all_messages(store.get_active_session("c1").id)
    assert msgs[1].role == "assistant" and msgs[1].provider == "free-tokens"
    assert msgs[1].model == "kimi-k3-highspeed"


# ---- 当前消息单一事实源（不重复进上下文）----

def test_current_message_sent_to_model_exactly_once(tmp_path):
    """当前消息先落库再装配：模型入参里它只出现一次（修复前发两遍）。"""
    svc, kimi, _ = make_service(tmp_path)
    svc.chat("第一问", CTX)
    svc.chat("第二问", CTX)
    contents = [m["content"] for m in kimi.full_calls[-1]["messages"]]
    assert contents.count("第二问") == 1
    assert contents.count("第一问") == 1  # 历史正常带入
    assert contents[-1] == "第二问"  # 当前消息在末尾


# ---- 召回：排除下推 + 失败契约 ----

def test_recall_exclude_pushed_down_to_query(tmp_path):
    """排除在 SQL 层完成：当前会话占满 top3 时，旧会话命中仍能召回。"""
    svc, _, store = make_service(tmp_path)
    svc.chat("给我讲讲马尔可夫", CTX)   # 日常会话的旧命中
    svc.cmd_new_session("新话题", CTX)
    for _ in range(3):
        svc.chat("再聊聊马尔可夫", CTX)  # 当前会话 3 条更新命中，占满 top3
    current = store.get_active_session("c1")
    recall = svc.recall("c1", "上次说的马尔可夫", exclude_session_id=current.id)
    assert recall is not None and "马尔可夫" in recall
    assert f"会话「{DEFAULT_SESSION_TITLE}」" in recall


def test_recall_failure_degrades_to_no_recall(tmp_path):
    """召回检索异常只记日志返回 None，不炸掉整轮对话。"""
    svc, _, store = make_service(tmp_path)
    svc.chat("你好", CTX)

    def boom(*args, **kwargs):
        raise sqlite3.OperationalError("fts5: syntax error near '\"'")

    store.search_messages = boom  # 模拟底层检索故障
    assert svc.recall("c1", "上次说的你好") is None
    assert svc.chat("上次说的你好", CTX) == "模型回复"  # 整轮对话不受影响


# ---- 删除二次确认：取消策略在 router 入口 ----

def test_delete_pending_cancelled_by_command(tmp_path):
    """「删除 N」后发命令（非「确认删除」）也取消待确认删除（修复前只有闲聊取消）。"""
    svc, _, _ = make_service(tmp_path)
    router = Router(chat_fn=svc.chat, session_svc=svc)
    svc.chat("你好", CTX)
    svc.cmd_new_session("要删的", CTX)
    router.handle("删除 1", CTX)
    assert "c1" in svc._pending_delete
    reply = router.handle("会话列表", CTX)
    assert "已取消删除" in reply and "要删的" in reply
    assert "c1" not in svc._pending_delete
    assert "没有待确认" in router.handle("确认删除", CTX)


def test_delete_confirm_via_router_not_cancelled(tmp_path):
    """「确认删除」本身不触发取消：router 直接放行给确认逻辑。"""
    svc, _, _ = make_service(tmp_path)
    router = Router(chat_fn=svc.chat, session_svc=svc)
    svc.chat("你好", CTX)
    svc.cmd_new_session("要删的", CTX)
    router.handle("删除 1", CTX)
    reply = router.handle("确认删除", CTX)
    assert reply.startswith("🗑")
    assert all(s.title != "要删的" for s in svc.store.list_sessions("c1", include_archived=True))
