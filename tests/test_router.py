"""router.py 命令注册表测试：精确匹配、中文口令、占位口令、带参命令、防误伤、删除确认取消。"""

from __future__ import annotations

import inspect

from feishu_assistant.modules.paper import MessageContext
from feishu_assistant.modules.router import COMMANDS, HELP_TEXT, PANEL_URL, Router

CTX = MessageContext(chat_id="c1", open_id="ou_x", message_id="m1")


class FakeSessionSvc:
    """记录会话命令调用，返回固定回复；带最小 pending 删除状态以模拟取消语义。"""

    def __init__(self):
        self.calls: list[tuple] = []
        self.pending_title: str | None = None

    def cmd_new_session(self, arg, ctx):
        self.calls.append(("new", arg))
        return f"新会话 {arg}"

    def cmd_list(self, ctx):
        self.calls.append(("list",))
        return "会话列表回复"

    def cmd_switch(self, arg, ctx):
        self.calls.append(("switch", arg))
        return f"切换到 {arg}"

    def cmd_rename(self, arg, ctx):
        self.calls.append(("rename", arg))
        return f"改名 {arg}"

    def cmd_archive(self, arg, ctx):
        self.calls.append(("archive", arg))
        return f"归档 {arg}"

    def cmd_delete(self, arg, ctx):
        self.calls.append(("delete", arg))
        self.pending_title = f"会话{arg}"
        return f"删除确认 {arg}"

    def cmd_delete_confirm(self, ctx):
        self.calls.append(("confirm",))
        self.pending_title = None
        return "已删除"

    def cancel_pending_delete(self, chat_id):
        self.calls.append(("cancel", chat_id))
        title, self.pending_title = self.pending_title, None
        return title


def make_router():
    calls = {"chat": [], "init": 0, "paper": [], "followup": []}

    def chat_fn(text, ctx):
        calls["chat"].append(text)
        return "对话回复"

    def init_fn():
        calls["init"] += 1
        return "初始化完成"

    def paper_fn(text, ctx):
        calls["paper"].append(text)
        return None  # 默认不接管

    def followup_fn(text, ctx):
        calls["followup"].append(text)
        return None

    svc = FakeSessionSvc()
    router = Router(chat_fn=chat_fn, init_fn=init_fn, paper_fn=paper_fn,
                    followup_fn=followup_fn, session_svc=svc)
    return router, calls, svc


# ---- 帮助与 init ----

def test_help_variants():
    router, _, _ = make_router()
    for cmd in ("/help", "help", "帮助", "帮助。", "帮助！"):
        assert router.handle(cmd) == HELP_TEXT


def test_help_generated_from_registry():
    for cmd in COMMANDS:
        assert cmd.help.split("：")[0].split(" ")[0] in HELP_TEXT
    for name in ("论文库", "收藏库", "灵感库", "求职看板", "面经库", "简历库", "备忘库", "精选库"):
        assert name in HELP_TEXT  # init 帮助行保留 8 张表名


def test_command_context_contract_matches_handler_signatures():
    """上下文需求属于命令本身；注册表与 handler 签名漂移时立即失败。"""
    for cmd in COMMANDS:
        parameters = inspect.signature(getattr(Router, cmd.handler)).parameters
        expected = 1 + int(bool(cmd.arg_pattern)) + int(cmd.needs_context)
        assert len(parameters) == expected, cmd.handler


def test_init_command():
    router, calls, _ = make_router()
    assert router.handle("/init") == "初始化完成"
    assert router.handle("初始化") == "初始化完成"
    assert calls["init"] == 2


def test_case_insensitive_ascii_command():
    router, _, _ = make_router()
    assert router.handle("/HELP") == HELP_TEXT


# ---- 菜单按钮中文口令 ----

def test_paper_hint_commands():
    router, calls, _ = make_router()
    for text in ("精读", "精读论文", "精读。"):
        reply = router.handle(text)
        assert "arXiv" in reply and "PDF" in reply
    assert calls["chat"] == []  # 未落入自由对话


def test_today_without_digest_service():
    router, _, _ = make_router()
    reply = router.handle("今日", object())
    assert "未配置" in reply  # P4 已接线：无 digest 服务时给友好提示而非占位
    assert router.handle("今日提醒", object()) == reply


def test_panel_command():
    router, _, _ = make_router()
    assert PANEL_URL in router.handle("面板")


def test_panel_command_lists_table_deep_links():
    """注入表映射后：面板列出各表深链（?table= 直达），含简历库等求职表。"""
    router, _, _ = make_router()
    router.panel_tables = {"论文库": "tblAAA", "简历库": "tblBBB", "求职看板": "tblCCC"}
    reply = router.handle("面板")
    assert f"{PANEL_URL}?table=tblAAA" in reply
    assert f"{PANEL_URL}?table=tblBBB" in reply
    assert "简历库" in reply and "求职看板" in reply


def test_p3_p4_commands_wired_not_unknown():
    """P3/P4 口令已正式接线：未注入服务时给「未配置」提示，绝不报未知命令。"""
    router, calls, _ = make_router()
    ctx = object()
    for text in ("arXiv日报", "今日", "组会周报", "周报"):
        reply = router.handle(text, ctx)
        assert "推送功能未配置" in reply, f"{text} 应委托 digest 服务"
    for text in ("押题", "押题 字节", "求职看板", "看板", "看板 新增 字节 后端",
                 "进展", "记面经 x", "面经", "简历", "模拟面试 字节"):
        reply = router.handle(text, ctx)
        assert "求职功能未配置" in reply, f"{text} 应委托 career 服务"
    reply = router.handle("复盘", ctx)
    assert "录音" in reply and "记面经" in reply  # 复盘用法提示
    assert "未知命令" not in reply and calls["chat"] == []


# ---- 会话命令（带参）----

def test_session_commands():
    router, _, svc = make_router()
    ctx = object()  # FakeSessionSvc 不关心内容，非 None 即可
    assert router.handle("会话列表", ctx) == "会话列表回复"
    assert router.handle("列表", ctx) == "会话列表回复"
    assert router.handle("新会话", ctx) == "新会话 "
    assert router.handle("新会话 论文讨论", ctx) == "新会话 论文讨论"
    assert router.handle("切换 2", ctx) == "切换到 2"
    assert router.handle("切换2", ctx) == "切换到 2"
    assert router.handle("改名 我的会话", ctx) == "改名 我的会话"
    assert router.handle("归档 3", ctx) == "归档 3"
    assert router.handle("删除 1", ctx) == "删除确认 1"
    assert router.handle("确认删除", ctx) == "已删除"
    assert ("new", "论文讨论") in svc.calls and ("switch", "2") in svc.calls


def test_session_commands_without_service():
    router = Router(chat_fn=lambda t, c: "x", session_svc=None)
    assert "未配置" in router.handle("会话列表")
    assert "未配置" in router.handle("切换 1")


# ---- 防误伤：整句精确匹配 ----

def test_no_hijack_normal_chat():
    router, calls, svc = make_router()
    for text in (
        "我今天想吃面", "今日天气怎么样", "帮我精读一下论文", "复盘一下今天学的知识",
        "列表推导式怎么用", "模型压缩的原理是什么", "简历怎么写", "改名需要什么材料",
    ):
        assert router.handle(text) == "对话回复", f"「{text}」不应被命令劫持"
    assert len(calls["chat"]) == 8
    assert not any(call[0] == "rename" for call in svc.calls)


def test_exact_blocklist_command_wins_over_argument_prefix():
    class Digest:
        def __init__(self):
            self.calls = []

        def cmd_blocklist(self, ctx):
            self.calls.append(("list", ctx))
            return "别推列表为空"

        def cmd_block(self, arg, ctx):
            self.calls.append(("block", arg, ctx))
            return "已屏蔽"

    digest = Digest()
    router = Router(chat_fn=lambda text, ctx: "对话回复", digest_svc=digest)
    ctx = object()

    assert router.handle("别推列表", ctx) == "别推列表为空"
    assert digest.calls == [("list", ctx)]


def test_natural_language_commands_require_explicit_argument_boundary():
    router, _, svc = make_router()
    ctx = object()

    assert router.handle("改名：论文讨论", ctx) == "改名 论文讨论"
    assert router.handle("新会话：实验记录", ctx) == "新会话 实验记录"
    assert ("rename", "论文讨论") in svc.calls


def test_trailing_punctuation_still_matches():
    router, _, _ = make_router()
    assert "求职功能未配置" in router.handle("押题！", object())
    assert "推送功能未配置" in router.handle("今日提醒。", object())


# ---- 未知命令与自由对话兜底 ----

def test_unknown_slash_command():
    router, calls, _ = make_router()
    reply = router.handle("/paper 标题")
    assert "未知命令" in reply and "/paper" in reply
    assert calls["chat"] == []


def test_paper_hook_called_before_chat():
    router, calls, _ = make_router()

    def paper_fn(text, ctx):
        if "arxiv.org" in text:
            return "精读 ack"
        return None

    router.paper_fn = paper_fn
    assert router.handle("https://arxiv.org/abs/2603.03329") == "精读 ack"
    assert calls["chat"] == []


def test_followup_hook_called_before_chat():
    router, calls, _ = make_router()
    router.followup_fn = lambda t, c: "追问回答" if "第三章" in t else None
    assert router.handle("第三章怎么算的") == "追问回答"
    assert router.handle("今天天气") == "对话回复"


def test_empty_text_returns_empty():
    router, _, _ = make_router()
    assert router.handle("   ") == ""


def test_chat_unavailable_message():
    router = Router(chat_fn=None)
    reply = router.handle("你好")
    assert "暂时不可用" in reply and "/help" in reply


def test_model_undo_command_wired():
    """「撤销切换」正式接线：未注入模型服务时给提示，绝不报未知命令。"""
    router, _, _ = make_router()
    assert "未配置" in router.handle("撤销切换")
    assert "未配置" in router.handle("撤销模型切换")


def test_model_command_switches_one_global_selection():
    """新命令不再要求 heavy/light；旧写法也只能切同一个全局选择。"""
    router, _, _ = make_router()

    class FakeModelSvc:
        def __init__(self):
            self.calls = []

        def switch(self, scope, target):
            self.calls.append((scope, target))
            return f"已切换 {target}"

    svc = FakeModelSvc()
    router.model_svc = svc
    assert router.handle("模型 provider-a/model-x") == "已切换 provider-a/model-x"
    assert router.handle("模型 heavy provider-b/model-y") == "已切换 provider-b/model-y"
    assert svc.calls == [
        ("all", "provider-a/model-x"),
        ("all", "provider-b/model-y"),
    ]


# ---- 删除二次确认：router 入口统一取消 ----

def test_pending_delete_cancelled_by_any_command():
    """「删除 N」后发任何命令（非「确认删除」）都取消待确认删除。"""
    router, _, svc = make_router()
    svc.pending_title = "要删的"
    reply = router.handle("会话列表", CTX)
    assert reply.startswith("（已取消删除会话「要删的」）")
    assert "会话列表回复" in reply
    assert svc.pending_title is None
    assert ("cancel", "c1") in svc.calls


def test_pending_delete_not_cancelled_by_confirm_itself():
    """「确认删除」不触发入口取消，直接走确认逻辑。"""
    router, _, svc = make_router()
    svc.pending_title = "要删的"
    reply = router.handle("确认删除", CTX)
    assert reply == "已删除"  # 无取消前缀
    assert ("cancel", "c1") not in svc.calls


def test_pending_delete_cancelled_by_chat_message():
    """自由对话消息同样经入口取消（原 session.chat 内的取消逻辑已上移）。"""
    router, calls, svc = make_router()
    svc.pending_title = "要删的"
    reply = router.handle("随便聊聊", CTX)
    assert reply.startswith("（已取消删除会话「要删的」）")
    assert svc.pending_title is None
    assert calls["chat"] == ["随便聊聊"]  # 对话本身照常进行


def test_new_delete_replaces_pending_with_note():
    """连续「删除 N」：旧目标取消并提示，新目标挂起。"""
    router, _, svc = make_router()
    svc.pending_title = "旧的"
    reply = router.handle("删除 2", CTX)
    assert reply.startswith("（已取消删除会话「旧的」）")
    assert "删除确认 2" in reply
    assert svc.pending_title == "会话2"


def test_pending_delete_cancel_skipped_without_chat_context():
    """无会话上下文（无 chat_id）时不触碰 pending 状态。"""
    router, _, svc = make_router()
    svc.pending_title = "要删的"
    assert router.handle("会话列表", object()) == "会话列表回复"
    assert svc.pending_title == "要删的"
