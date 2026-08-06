"""长期记忆测试：CRUD、压缩提炼、装配注入、口令。全部离线。"""

from __future__ import annotations

from feishu_assistant.modules.paper import MessageContext
from feishu_assistant.modules.router import Router
from feishu_assistant.modules.session import (
    MEMORY_INJECT_LIMIT,
    ContextBudget,
    SessionService,
)
from feishu_assistant.store import SessionStore

CTX = MessageContext(chat_id="c1", open_id="ou_x", message_id="m1")


class FakeKimi:
    """chat_full 走自由对话；chat 返回固定文本（压缩/起标题共用）。"""

    def __init__(self, chat_reply="摘要正文"):
        self.chat_reply = chat_reply
        self.chat_calls: list[dict] = []

    def chat_full(self, messages, *, heavy=False, json_mode=False):
        from feishu_assistant.kimi import ChatResult
        return ChatResult("模型回复", "", "free-tokens", "kimi-k3-highspeed")

    def chat(self, messages, *, heavy=False, json_mode=False):
        self.chat_calls.append({"messages": messages, "heavy": heavy})
        return self.chat_reply


def make_service(tmp_path, kimi=None, **budget_kw) -> tuple[SessionService, FakeKimi, SessionStore]:
    kimi = kimi or FakeKimi()
    store = SessionStore(tmp_path / "s.db")
    budget = ContextBudget(**budget_kw) if budget_kw else ContextBudget()
    svc = SessionService(kimi, store, budget=budget, persona="你是个人助理。")
    return svc, kimi, store


def make_router(tmp_path, **budget_kw):
    """带记录型 inbox_fn 的 Router：验证「记住」系列口令不落到收件箱。"""
    svc, kimi, store = make_service(tmp_path, **budget_kw)
    inbox_calls: list[str] = []

    def inbox_fn(text, ctx):
        inbox_calls.append(text)
        return "已入收件箱"

    router = Router(chat_fn=svc.chat, session_svc=svc, inbox_fn=inbox_fn)
    return router, svc, store, inbox_calls


def fill_to_compress(store, session, n=6):
    """塞够触发压缩的历史消息（对齐 test_session 的压缩触发条件）。"""
    for i in range(n):
        store.add_message(session.id, "c1", "user" if i % 2 == 0 else "assistant",
                          "历史内容" + "x" * 30)


# ---- CRUD ----

def test_add_and_list_memories(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    store.add_memory("c1", "偏好", "喜欢简洁回答")
    store.add_memory("c1", "项目", "在做 Agent 评测")
    store.add_memory("c2", "备忘", "别的 chat 的不算")
    mems = store.list_memories("c1")
    assert [m.content for m in mems] == ["在做 Agent 评测", "喜欢简洁回答"]  # 新的在前
    assert mems[0].category == "项目" and mems[0].hit_count == 0 and mems[0].last_hit is None


def test_add_memory_dedup_same_content(tmp_path):
    """同文记忆不新增：刷新 created_at（类别跟随最新），行 id 不变。"""
    store = SessionStore(tmp_path / "s.db")
    id1 = store.add_memory("c1", "备忘", "周三交周报")
    before = store.list_memories("c1")[0].created_at
    id2 = store.add_memory("c1", "偏好", "周三交周报")
    assert id1 == id2
    mems = store.list_memories("c1")
    assert len(mems) == 1
    assert mems[0].created_at >= before and mems[0].category == "偏好"


def test_delete_memory_fuzzy(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    store.add_memory("c1", "备忘", "周三交周报")
    store.add_memory("c1", "备忘", "周五开组会")
    assert store.delete_memory("c1", "周报") == 1  # 模糊匹配，返回删除条数
    assert [m.content for m in store.list_memories("c1")] == ["周五开组会"]
    assert store.delete_memory("c1", "不存在") == 0
    store.add_memory("c1", "备忘", "百分之五十")
    assert store.delete_memory("c1", "%") == 0  # 通配符被转义，按字面匹配


def test_touch_memory(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    mid = store.add_memory("c1", "备忘", "x")
    store.touch_memory(mid)
    store.touch_memory(mid)
    m = store.list_memories("c1")[0]
    assert m.hit_count == 2 and m.last_hit is not None


# ---- 压缩提炼 ----

def test_compress_extracts_memories(tmp_path):
    """压缩输出带「##长期记忆」区：摘要照旧更新，合法记忆行入库，坏行跳过。"""
    kimi = FakeKimi(chat_reply=(
        "用户在做 Agent 评测，偏好简洁回答。\n"
        "##长期记忆\n"
        "偏好|喜欢简洁直接的回答\n"
        "- 项目|在做 Agent Harness 评测\n"
        "坏行没有分隔符\n"
        "心情|非法类别跳过\n"
        "事实|\n"
    ))
    svc, _, store = make_service(tmp_path, kimi=kimi, recent_rounds=1, recent_tokens=20)
    session = store.get_or_create_default_session("c1")
    fill_to_compress(store, session)
    assert svc.maybe_compress(session) is True
    assert store.get_session(session.id).summary == "用户在做 Agent 评测，偏好简洁回答。"
    mems = store.list_memories("c1")
    assert {(m.category, m.content) for m in mems} == {
        ("偏好", "喜欢简洁直接的回答"),
        ("项目", "在做 Agent Harness 评测"),
    }


def test_compress_without_memory_section_keeps_behavior(tmp_path):
    """没有记忆区时与旧行为完全一致：整个输出即摘要，不落记忆。"""
    svc, kimi, store = make_service(tmp_path, recent_rounds=1, recent_tokens=20)
    session = store.get_or_create_default_session("c1")
    fill_to_compress(store, session)
    assert svc.maybe_compress(session) is True
    assert store.get_session(session.id).summary == "摘要正文"
    assert store.list_memories("c1") == []
    assert kimi.chat_calls and kimi.chat_calls[0]["heavy"] is False  # 压缩走 light


def test_compress_memory_section_only_keeps_old_summary(tmp_path):
    """模型只回了记忆区（摘要正文为空）：保留旧摘要，记忆照入——摘要是主输出不可丢。"""
    kimi = FakeKimi(chat_reply="## 长期记忆\n事实|用户姓王")  # 带空格的标记也认
    svc, _, store = make_service(tmp_path, kimi=kimi, recent_rounds=1, recent_tokens=20)
    session = store.get_or_create_default_session("c1")
    store.update_summary(session.id, "旧摘要保留")
    fill_to_compress(store, session)
    assert svc.maybe_compress(session) is True
    assert store.get_session(session.id).summary == "旧摘要保留"
    assert [(m.category, m.content) for m in store.list_memories("c1")] == [("事实", "用户姓王")]


# ---- 装配注入 ----

def test_assemble_injects_memory_block(tmp_path):
    svc, _, store = make_service(tmp_path)
    session = store.get_or_create_default_session("c1")
    store.update_summary(session.id, "旧摘要")
    store.add_memory("c1", "偏好", "喜欢简洁回答")
    session = store.get_session(session.id)
    assembled = svc.assemble_context(session)
    system = assembled.messages[0]["content"]
    assert "## 长期记忆" in system and "【偏好】喜欢简洁回答" in system
    assert "不要逐条复述给用户" in system
    assert assembled.used_memory
    # 装配顺序：persona → 长期记忆 → 摘要
    assert system.index("你是个人助理") < system.index("## 长期记忆") < system.index("前情摘要")


def test_assemble_no_memories_no_block(tmp_path):
    svc, _, store = make_service(tmp_path)
    session = store.get_or_create_default_session("c1")
    assembled = svc.assemble_context(session)
    assert "## 长期记忆" not in assembled.messages[0]["content"]
    assert not assembled.used_memory


def test_assemble_memory_budget_drops_oldest(tmp_path):
    """超 memory_chars 预算时从最旧的开始丢（新的必保）。"""
    svc, _, store = make_service(tmp_path, memory_chars=50)
    session = store.get_or_create_default_session("c1")
    store.add_memory("c1", "备忘", "旧" + "x" * 40)
    store.add_memory("c1", "备忘", "新" + "x" * 40)
    system = svc.assemble_context(session).messages[0]["content"]
    assert "新" + "x" * 40 in system
    assert "旧" + "x" * 40 not in system


def test_assemble_memory_count_capped(tmp_path):
    """条目超上限时只注入最近 N 条。"""
    svc, _, store = make_service(tmp_path)
    session = store.get_or_create_default_session("c1")
    for i in range(MEMORY_INJECT_LIMIT + 5):
        store.add_memory("c1", "备忘", f"条目{i:02d}")
    system = svc.assemble_context(session).messages[0]["content"]
    assert f"条目{MEMORY_INJECT_LIMIT + 4:02d}" in system  # 最新的在
    assert "条目00" not in system  # 最旧的被条数上限裁掉


# ---- 口令 ----

def test_cmd_remember_via_router(tmp_path):
    router, _, store, inbox_calls = make_router(tmp_path)
    reply = router.handle("记住 我喜欢喝美式", CTX)
    assert "已记住" in reply and "偏好" in reply
    mems = store.list_memories("c1")
    assert len(mems) == 1 and mems[0].category == "偏好" and mems[0].content == "我喜欢喝美式"
    assert inbox_calls == []  # 「记住」≠「记一下」：不进收件箱备忘库


def test_cmd_remember_category_guess(tmp_path):
    router, _, store, _ = make_router(tmp_path)
    router.handle("记住 我决定投身 Agent 方向", CTX)
    router.handle("记住 周三要交周报", CTX)
    by_content = {m.content: m.category for m in store.list_memories("c1")}
    assert by_content["我决定投身 Agent 方向"] == "决定"
    assert by_content["周三要交周报"] == "备忘"  # 粗判不中的落「备忘」


def test_cmd_remember_without_arg(tmp_path):
    router, _, store, _ = make_router(tmp_path)
    assert "用法" in router.handle("记住", CTX)
    assert store.list_memories("c1") == []


def test_cmd_forget_via_router(tmp_path):
    router, _, store, _ = make_router(tmp_path)
    router.handle("记住 周三要交周报", CTX)
    router.handle("记住 周五开组会", CTX)
    reply = router.handle("忘掉 周报", CTX)
    assert "已忘掉 1 条" in reply
    assert [m.content for m in store.list_memories("c1")] == ["周五开组会"]
    assert "没有找到" in router.handle("忘掉 周报", CTX)  # 已删，再删报 0 条
    assert "用法" in router.handle("忘掉", CTX)


def test_cmd_memories_list_via_router(tmp_path):
    router, _, store, _ = make_router(tmp_path)
    assert "还没有长期记忆" in router.handle("我的记忆", CTX)
    router.handle("记住 我喜欢喝美式", CTX)
    router.handle("记住 周三要交周报", CTX)
    reply = router.handle("我的记忆", CTX)
    assert "1. 【备忘】周三要交周报" in reply  # 新的在前
    assert "2. 【偏好】我喜欢喝美式" in reply
    assert "共 2 条" in reply


def test_jiyixia_still_goes_to_inbox(tmp_path):
    """「记一下」维持原样走收件箱备忘库，绝不落长期记忆。"""
    router, _, store, inbox_calls = make_router(tmp_path)
    assert router.handle("记一下 买牛奶", CTX) == "已入收件箱"
    assert inbox_calls == ["记一下 买牛奶"]
    assert store.list_memories("c1") == []
