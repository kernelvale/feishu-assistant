"""消息路由：命令注册表 + 功能钩子分发。

匹配优先级：
1. 命令注册表（整句精确匹配，忽略首尾空白和结尾标点；带参命令按正则）
2. 未知 / 开头 → 提示
3. paper_fn：arXiv 链接 / 精读触发 → 精读管线（异步 ack）
4. 显式归档意图：在模型问答前走 inbox_fn，只有真实写库成功才确认
5. career_fn：P3 求职（career 绑定会话的模拟面试消息、JD 粘贴识别）
6. inbox_fn：P4 万能收件箱（链接 / 「记一下/提醒我/收藏/灵感」开头 → 三库归档）
7. 论文绑定会话：直接交给会话服务，绑定论文优先于任何语义匹配
8. followup_fn：无绑定会话的论文追问（近 N 小时 + 追问关键词）
9. chat_fn：自由对话（会话系统，含记忆）

口令匹配刻意保守（整句精确匹配），避免误伤正常聊天（如「我今天想吃面」不会命中「今日」）。
/help 由注册表自动生成。各 hook 以 callable/对象注入，测试可传 fake。

「删除 N」的待确认状态由 session 服务持有；任何非「确认删除」的输入在 handle
入口统一调 session_svc.cancel_pending_delete 取消（提示语承诺"回复其他内容取消"）。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger("feishu_assistant.router")

PANEL_URL = "https://feishu.cn/base/VoZHbXaz5aMAqzsiRHacfWzinsQ"  # 多维表格（全部表同库）

# 面板命令展示的表及顺序（均为同一 base 内的表深链）
PANEL_TABLES = ("论文库", "精选库", "简历库", "求职看板", "面经库", "收藏库", "灵感库", "备忘库")

UNKNOWN_COMMAND_HINT = "未知命令「{cmd}」。发 /help 或点菜单「帮助」查看可用命令。"

# hook 签名：fn(text, context) -> str | None；返回 None 表示"不归我管，继续往下路由"
# chat_fn 同样收 (text, context)（会话系统需要上下文），返回 str
Hook = Callable[[str, Any], "str | None"]
ChatHook = Callable[[str, Any], str]
IntentPredicate = Callable[[str], bool]

_TRAILING_PUNCT = "。！？!?.，,、~～… "


@dataclass
class Command:
    """注册表条目。names 为整句精确口令；arg_pattern 为带参命令正则（fullmatch）。"""

    names: tuple[str, ...]
    handler: str           # Router 方法名
    help: str
    arg_pattern: str | None = None
    needs_context: bool = False

    def exact_match(self, text: str) -> bool:
        return text.lower() in (n.lower() for n in self.names)

    def argument_match(self, text: str) -> re.Match | None:
        return re.fullmatch(self.arg_pattern, text) if self.arg_pattern else None


# 命令注册表：顺序即优先级
COMMANDS: list[Command] = [
    Command(("帮助", "/help", "help", "使用帮助"), "_cmd_help",
            "帮助：显示本帮助"),
    Command(("/init", "init", "初始化", "初始化表格"), "_cmd_init",
            "/init：初始化多维表格（自动创建 8 张业务表；已有 app 则幂等补建缺表与缺字段）"),
    Command(("精读", "精读论文"), "_cmd_paper",
            "精读 / 精读 N：发 arXiv 链接或 PDF 即精读；「精读 2」精读今日精选库第 2 篇",
            arg_pattern=r"(?:精读|精读论文)\s*(\d+)", needs_context=True),
    Command(("今日", "今日提醒"), "_cmd_today",
            "今日 / 今日提醒：今日待办汇总（备忘 DDL + 求职跟进 + 收藏翻坟，每日 9:00 自动推送）",
            needs_context=True),
    Command(("面板",), "_cmd_panel",
            "面板：多维表格入口（论文库/精选库/简历库/求职看板等）"),
    Command(("会话列表", "列表"), "_cmd_session_list",
            "会话列表 / 列表：列出全部会话", needs_context=True),
    Command(("新会话",), "_cmd_new_session", "新会话 [标题]：创建并切换到新会话",
            arg_pattern=r"新会话(?:\s+|[：:])(.+)", needs_context=True),
    Command(("切换",), "_cmd_switch", "切换 N：切换到会话列表中的第 N 个",
            arg_pattern=r"切换\s*(\d+)", needs_context=True),
    Command(("改名",), "_cmd_rename", "改名 新标题：重命名当前会话",
            arg_pattern=r"改名(?:\s+|[：:])(.+)", needs_context=True),
    Command(("归档",), "_cmd_archive", "归档 N：归档第 N 个会话（列表不再显示，数据保留）",
            arg_pattern=r"归档\s*(\d+)", needs_context=True),
    Command(("删除",), "_cmd_delete", "删除 N：删除第 N 个会话（需二次确认）",
            arg_pattern=r"删除\s*(\d+)", needs_context=True),
    Command(("确认删除",), "_cmd_delete_confirm", "确认删除：执行待确认的会话删除",
            needs_context=True),
    # ---- 长期记忆（「记住」≠ inbox 的「记一下」，后者写备忘库不动）----
    Command(("记住",), "_cmd_remember", "记住 xxx：写入长期记忆（之后对话自动带上）",
            arg_pattern=r"记住(?:\s+|[：:])(.+)", needs_context=True),
    Command(("忘掉",), "_cmd_forget", "忘掉 xxx：按关键词模糊删除长期记忆",
            arg_pattern=r"忘掉(?:\s+|[：:])(.+)", needs_context=True),
    Command(("我的记忆",), "_cmd_memories", "我的记忆：列出全部长期记忆",
            needs_context=True),
    # ---- P4 自动化推送 ----
    Command(("arXiv日报", "arxiv日报"), "_cmd_arxiv_daily",
            "arXiv日报：立即生成今日 arXiv 精选（每日 8:30 自动推送）", needs_context=True),
    Command(("组会周报", "周报"), "_cmd_weekly",
            "组会周报 / 周报：本周精读+收藏+灵感汇总成组会大纲（每周一 9:00 自动推送）",
            needs_context=True),
    Command(("别推",), "_cmd_block",
            "别推 关键词：日报淘汰含该词的论文（「别推列表」查看，「恢复推」撤销）",
            arg_pattern=r"别推(?:\s+|[：:])(.+)", needs_context=True),
    Command(("恢复推",), "_cmd_unblock", "恢复推 关键词：从别推列表移除",
            arg_pattern=r"恢复推(?:\s+|[：:])(.+)", needs_context=True),
    Command(("别推列表",), "_cmd_blocklist", "别推列表：查看当前负推词",
            needs_context=True),
    Command(("模型", "模型列表", "models", "模型路由"), "_cmd_models",
            "模型：查看当前模型与 provider；「模型 provider/model」全局切换",
            arg_pattern=r"(?:模型路由|模型|models)(?:\s+|[：:])(.+)", needs_context=True),
    Command(("撤销切换", "撤销模型切换"), "_cmd_model_undo",
            "撤销切换：回滚最近一次模型首选切换"),
    # ---- P3 求职面试 ----
    Command(("看板", "求职看板"), "_cmd_career_board",
            "看板 [操作]：投递进度汇总；自然语言增删改查（如「看板 字节改一面过了」）",
            arg_pattern=r"(?:求职)?看板(?:\s+|[：:])(.+)", needs_context=True),
    Command(("进展", "进展如何"), "_cmd_career_progress",
            "进展 / 进展如何：求职看板汇总 + 该跟进清单", needs_context=True),
    Command(("押题",), "_cmd_career_predict",
            "押题 公司：基于 JD + 简历 + 库内面经按岗位能力出题",
            arg_pattern=r"押题(?:\s+|[：:])(.+)", needs_context=True),
    Command(("记面经",), "_cmd_career_save_exp",
            "记面经 <内容>：面经结构化入库（公司/轮次/问题/答砸点）",
            arg_pattern=r"记面经(?:\s+|[：:])(.+)", needs_context=True),
    Command(("面经", "面经查询"), "_cmd_career_search_exp",
            "面经 [公司] [轮次]：检索面经库",
            arg_pattern=r"面经(?:查询)?(?:\s+|[：:])(.+)", needs_context=True),
    Command(("简历", "简历列表"), "_cmd_career_resume",
            "简历 [岗位/能力关键词]：查询多标签画像；「简历 更新画像」补全旧记录（发 PDF/docx 即解析归档）",
            arg_pattern=r"简历(?:列表)?(?:\s+|[：:])(.+)", needs_context=True),
    Command(("定制简历",), "_cmd_career_tailor_resume",
            "定制简历 求职ID/公司岗位：基于JD和基础版生成保持原DOCX格式的岗位分支",
            arg_pattern=r"定制简历(?:\s+|[：:])(.+)", needs_context=True),
    Command(("确认定制简历",), "_cmd_career_confirm_resume",
            "确认定制简历 求职ID：确认草稿并设为该岗位推荐投递版本",
            arg_pattern=r"确认定制简历(?:\s+|[：:])(.+)", needs_context=True),
    Command(("模拟面试",), "_cmd_career_mock",
            "模拟面试 公司：文字模拟面试（多轮追问，发「结束面试」出复盘）",
            arg_pattern=r"模拟面试(?:\s+|[：:])(.+)", needs_context=True),
    Command(("面试复盘", "复盘"), "_cmd_career_review_hint",
            "面试复盘 / 复盘：把面试录音或语音发给我即自动复盘"),
]


def _normalize(text: str) -> str:
    """整句规范化：去首尾空白与结尾标点。"""
    return text.strip().rstrip(_TRAILING_PUNCT).strip()


def _is_delete_confirm(normalized: str) -> bool:
    """该输入是否将分发为「确认删除」（与注册表精确匹配同规则）。"""
    return any(
        cmd.handler == "_cmd_delete_confirm" and cmd.exact_match(normalized)
        for cmd in COMMANDS
    )


def _generate_help() -> str:
    lines = ["🤖 飞书个人助理使用帮助", "", "【口令】菜单按钮或手动输入均可："]
    lines += [f"• {cmd.help}" for cmd in COMMANDS]
    lines += [
        "",
        "【直接聊】",
        "• 发 arXiv 链接（abs/pdf）或把论文 PDF 发给我：立即开始论文精读",
        "• 精读完可直接追问论文细节（会自动切换到该论文的精读会话）",
        "• 发链接或「记一下/提醒我/收藏/灵感」开头：自动归档收件箱（收藏库/灵感库/备忘库）",
        "• 发简历文件（PDF/docx）：自动入简历库；发面试录音/语音：自动复盘；音频文件名含“面经”则仅结构化入面经库",
        "• 粘贴 JD 全文：自动拆解（硬性要求/加分项/潜台词/简历匹配）",
        "• 其他消息：自由对话（带会话记忆，可「新会话」隔离话题、「会话列表」管理）",
    ]
    return "\n".join(lines)


HELP_TEXT = _generate_help()


class Router:
    """命令注册表 + 钩子路由器：输入用户文本（+消息上下文），输出回复文本。"""

    def __init__(
        self,
        chat_fn: ChatHook | None = None,
        init_fn: Callable[[], str] | None = None,
        paper_fn: Hook | None = None,
        followup_fn: Hook | None = None,
        session_svc: Any = None,
        career_svc: Any = None,
        digest_svc: Any = None,
        career_fn: Hook | None = None,
        inbox_fn: Hook | None = None,
        inbox_priority_fn: IntentPredicate | None = None,
        paper_svc: Any = None,
        model_svc: Any = None,
        panel_tables: dict[str, str] | None = None,
    ) -> None:
        self.chat_fn = chat_fn
        self.init_fn = init_fn
        self.paper_fn = paper_fn
        self.followup_fn = followup_fn
        self.session_svc = session_svc
        self.career_svc = career_svc
        self.digest_svc = digest_svc
        self.paper_svc = paper_svc
        self.model_svc = model_svc
        self.career_fn = career_fn
        self.inbox_fn = inbox_fn
        self.inbox_priority_fn = inbox_priority_fn
        self.panel_tables = panel_tables or {}

    # ------------------------------------------------------------------

    def handle(self, text: str, context: Any = None) -> str:
        text = text.strip()
        if not text:
            return ""

        normalized = _normalize(text)
        # 「确认删除」之外的任何输入都先取消待确认的删除（提示语承诺"回复其他内容取消"）
        prefix = "" if _is_delete_confirm(normalized) else self._cancel_pending_delete(context)
        return prefix + self._route(text, normalized, context)

    def _route(self, text: str, normalized: str, context: Any) -> str:
        """按优先级分发：命令注册表 → 未知 / 命令 → 各功能钩子 → 自由对话。"""
        # 无参精确口令永远优先于带参前缀，避免「别推列表」被「别推」吞掉。
        for cmd in COMMANDS:
            if not cmd.exact_match(normalized):
                continue
            logger.info("路由命中：命令 %s", cmd.handler)
            return self._dispatch(cmd, None, context)

        for cmd in COMMANDS:
            m = cmd.argument_match(normalized)
            if m is None:
                continue
            logger.info("路由命中：命令 %s", cmd.handler)
            return self._dispatch(cmd, m, context)

        if text.startswith("/"):
            cmd_name = text.split(maxsplit=1)[0]
            logger.info("路由命中：未知命令 %s", cmd_name)
            return UNKNOWN_COMMAND_HINT.format(cmd=cmd_name)

        if self.paper_fn is not None:
            reply = self.paper_fn(text, context)
            if reply is not None:
                logger.info("路由命中：论文精读")
                return reply

        # 明确的「写进灵感库/存到收藏库」必须在所有模型问答路径之前执行；
        # 确认文案只能来自 inbox_fn 的真实写库返回值。
        inbox_prechecked = False
        if self.inbox_fn is not None and self.inbox_priority_fn is not None \
                and self.inbox_priority_fn(text):
            inbox_prechecked = True
            reply = self.inbox_fn(text, context)
            if reply is not None:
                logger.info("路由命中：万能收件箱（显式归档意图）")
                return reply

        active_bind_type = self._active_bind_type(context)

        # 论文绑定关系比 JD 猜测和全库向量都可靠。仍先放行普通 inbox 规则，保证
        # 用户在论文会话里发「收藏：链接」或普通网页链接时能正常归档。
        if active_bind_type == "paper" and self.chat_fn is not None:
            if self.inbox_fn is not None and not inbox_prechecked:
                inbox_prechecked = True
                reply = self.inbox_fn(text, context)
                if reply is not None:
                    logger.info("路由命中：万能收件箱（论文绑定会话内）")
                    return reply
            logger.info("路由命中：论文绑定会话")
            return self.chat_fn(text, context)

        if self.career_fn is not None:
            reply = self.career_fn(text, context)
            if reply is not None:
                logger.info("路由命中：求职模块")
                return reply

        if self.inbox_fn is not None and not inbox_prechecked:
            reply = self.inbox_fn(text, context)
            if reply is not None:
                logger.info("路由命中：万能收件箱")
                return reply

        if self.followup_fn is not None:
            reply = self.followup_fn(text, context)
            if reply is not None:
                logger.info("路由命中：论文追问")
                return reply

        if self.chat_fn is None:
            return "对话功能暂时不可用，请稍后再试。发 /help 查看可用命令。"
        logger.info("路由命中：自由对话")
        return self.chat_fn(text, context)

    def _active_bind_type(self, context: Any) -> str:
        """读取当前活跃会话绑定类型；存储异常时按无绑定降级。"""
        chat_id = getattr(context, "chat_id", "") if context is not None else ""
        store = getattr(self.session_svc, "store", None)
        getter = getattr(store, "get_active_session", None)
        if not chat_id or getter is None:
            return ""
        try:
            session = getter(chat_id)
        except Exception:
            logger.exception("读取活跃会话失败（按无绑定继续）")
            return ""
        return str(getattr(session, "bind_type", "") or "") if session is not None else ""

    def _dispatch(self, cmd: Command, match: re.Match | None, context: Any) -> str:
        handler = getattr(self, cmd.handler)
        if cmd.arg_pattern:
            arg = match.group(1) if match is not None else ""
            return handler(arg, context) if cmd.needs_context else handler(arg)
        if cmd.needs_context:
            return handler(context)
        return handler()

    def _cancel_pending_delete(self, context: Any) -> str:
        """取消待确认的「删除 N」，返回取消提示前缀（无待确认/无会话上下文时为空串）。

        pending 状态归 session 服务所有；router 只负责在消息入口统一触发取消。
        """
        svc = self.session_svc
        chat_id = getattr(context, "chat_id", "") if context is not None else ""
        cancel = getattr(svc, "cancel_pending_delete", None) if svc is not None else None
        if not chat_id or cancel is None:
            return ""
        title = cancel(chat_id)
        return f"（已取消删除会话「{title}」）\n" if title else ""

    # ------------------------------------------------------------------
    # 命令实现
    # ------------------------------------------------------------------

    def _cmd_help(self) -> str:
        return HELP_TEXT

    def _cmd_init(self) -> str:
        if self.init_fn is None:
            return "多维表格初始化功能未配置，请检查服务启动方式。"
        return self.init_fn()

    def _cmd_paper(self, arg: str, context: Any) -> str:
        """「精读」=用法提示；「精读 N」=精选库最新一批第 N 篇。"""
        arg = (arg or "").strip()
        if not arg:
            return (
                "想精读论文的话，直接发我 arXiv 链接（abs/pdf 均可）或把 PDF 文件发给我，"
                "我就开始精读～完成后会发你精读文档，还能继续追问论文细节。"
                "也可以说「精读 2」，精读今日精选库第 2 篇。"
            )
        if self.paper_svc is None:
            return "精读服务未配置，请检查服务启动方式。"
        if context is None:
            return "「精读 N」需要消息上下文。"
        try:
            n = int(arg)
        except (TypeError, ValueError):
            return "用法：精读 N（如「精读 2」，指今日精选库第 2 篇）。"
        return self.paper_svc.read_pick(n, context)

    def _cmd_models(self, arg: str, context: Any) -> str:
        """「模型」=交互面板；「模型 provider/model」=切换全局唯一模型。"""
        if self.model_svc is None:
            return "模型服务未配置，请检查服务启动方式。"
        arg = (arg or "").strip()
        if not arg:
            sender = getattr(self.model_svc, "sender", None)
            chat_id = getattr(context, "chat_id", "") if context is not None else ""
            if sender is not None and chat_id:
                sender.send_card(chat_id, self.model_svc.build_card())
                return "🎛 模型面板见上方卡片：下拉选 provider/模型 立即切换。"
            return self.model_svc.status_text()
        parts = arg.split(None, 1)
        # 兼容旧命令「模型 heavy provider/model」，但 heavy/light 不再分别选模。
        if parts[0] in ("heavy", "light") and len(parts) == 2:
            return self.model_svc.switch("all", parts[1].strip())
        if len(parts) == 1 and "/" in parts[0]:
            return self.model_svc.switch("all", parts[0])
        return "用法：模型（面板）/ 模型 provider/model（全局切换）。"

    def _cmd_model_undo(self) -> str:
        if self.model_svc is None:
            return "模型服务未配置，请检查服务启动方式。"
        return self.model_svc.undo_last()

    def _cmd_panel(self) -> str:
        """面板：列出各表深链（同库不同表，?table= 直达）；未配置时退化为论文库总链。"""
        links = [
            f"• {name}：{PANEL_URL}?table={self.panel_tables[name]}"
            for name in PANEL_TABLES
            if self.panel_tables.get(name)
        ]
        if not links:
            return f"📚 论文库面板（多维表格）：{PANEL_URL}"
        return "📚 多维表格面板：\n" + "\n".join(links)

    # -- P4 推送命令（委托给 digest 服务）--

    def _digest_required(self) -> str | None:
        if self.digest_svc is None:
            return "推送功能未配置，请检查服务启动方式。"
        return None

    def _cmd_today(self, context: Any) -> str:
        if (err := self._digest_required()) or context is None:
            return err or "「今日」需要消息上下文。"
        return self.digest_svc.cmd_today(context)

    def _cmd_arxiv_daily(self, context: Any) -> str:
        if (err := self._digest_required()) or context is None:
            return err or "「arXiv日报」需要消息上下文。"
        return self.digest_svc.cmd_arxiv_daily(context)

    def _cmd_weekly(self, context: Any) -> str:
        if (err := self._digest_required()) or context is None:
            return err or "「组会周报」需要消息上下文。"
        return self.digest_svc.cmd_weekly(context)

    def _cmd_block(self, arg: str, context: Any) -> str:
        if (err := self._digest_required()) or context is None:
            return err or "「别推」需要消息上下文。"
        return self.digest_svc.cmd_block(arg, context)

    def _cmd_unblock(self, arg: str, context: Any) -> str:
        if (err := self._digest_required()) or context is None:
            return err or "「恢复推」需要消息上下文。"
        return self.digest_svc.cmd_unblock(arg, context)

    def _cmd_blocklist(self, context: Any) -> str:
        if (err := self._digest_required()) or context is None:
            return err or "「别推列表」需要消息上下文。"
        return self.digest_svc.cmd_blocklist(context)

    # -- P3 求职命令（委托给 career 服务）--

    def _career_required(self) -> str | None:
        if self.career_svc is None:
            return "求职功能未配置，请检查服务启动方式。"
        return None

    def _cmd_career_board(self, arg: str, context: Any) -> str:
        if err := self._career_required():
            return err
        return self.career_svc.cmd_board(arg, context)

    def _cmd_career_progress(self, context: Any) -> str:
        if err := self._career_required():
            return err
        return self.career_svc.cmd_board("", context)

    def _cmd_career_predict(self, arg: str, context: Any) -> str:
        if err := self._career_required():
            return err
        return self.career_svc.cmd_predict(arg, context)

    def _cmd_career_save_exp(self, arg: str, context: Any) -> str:
        if err := self._career_required():
            return err
        return self.career_svc.cmd_save_exp(arg, context)

    def _cmd_career_search_exp(self, arg: str, context: Any) -> str:
        if err := self._career_required():
            return err
        return self.career_svc.cmd_search_exp(arg, context)

    def _cmd_career_resume(self, arg: str, context: Any) -> str:
        if err := self._career_required():
            return err
        return self.career_svc.cmd_resume(arg, context)

    def _cmd_career_tailor_resume(self, arg: str, context: Any) -> str:
        if err := self._career_required():
            return err
        return self.career_svc.cmd_tailor_resume(arg, context)

    def _cmd_career_confirm_resume(self, arg: str, context: Any) -> str:
        if err := self._career_required():
            return err
        return self.career_svc.cmd_confirm_tailored_resume(arg, context)

    def _cmd_career_mock(self, arg: str, context: Any) -> str:
        if err := self._career_required():
            return err
        return self.career_svc.cmd_mock_interview(arg, context)

    def _cmd_career_review_hint(self) -> str:
        return (
            "把面试录音文件或语音直接发给我，我就开始复盘：转写 → 复盘报告（飞书文档）"
            "→ 问题自动入面经库。只想把语音结构化入库，可把音频文件名加上“面经”后上传。"
            "文字面经发「记面经 <内容>」；想看库存发「面经 公司名」。"
        )

    # -- 会话命令（委托给 session 服务）--

    def _session_required(self) -> str | None:
        if self.session_svc is None:
            return "会话功能未配置，请检查服务启动方式。"
        return None

    def _cmd_new_session(self, arg: str, context: Any) -> str:
        if (err := self._session_required()) or context is None:
            return err or "会话命令需要消息上下文。"
        return self.session_svc.cmd_new_session(arg, context)

    def _cmd_session_list(self, context: Any) -> str:
        if (err := self._session_required()) or context is None:
            return err or "会话命令需要消息上下文。"
        return self.session_svc.cmd_list(context)

    def _cmd_switch(self, arg: str, context: Any) -> str:
        if (err := self._session_required()) or context is None:
            return err or "会话命令需要消息上下文。"
        return self.session_svc.cmd_switch(arg, context)

    def _cmd_rename(self, arg: str, context: Any) -> str:
        if (err := self._session_required()) or context is None:
            return err or "会话命令需要消息上下文。"
        return self.session_svc.cmd_rename(arg, context)

    def _cmd_archive(self, arg: str, context: Any) -> str:
        if (err := self._session_required()) or context is None:
            return err or "会话命令需要消息上下文。"
        return self.session_svc.cmd_archive(arg, context)

    def _cmd_delete(self, arg: str, context: Any) -> str:
        if (err := self._session_required()) or context is None:
            return err or "会话命令需要消息上下文。"
        return self.session_svc.cmd_delete(arg, context)

    def _cmd_delete_confirm(self, context: Any) -> str:
        if (err := self._session_required()) or context is None:
            return err or "会话命令需要消息上下文。"
        return self.session_svc.cmd_delete_confirm(context)

    def _cmd_remember(self, arg: str, context: Any) -> str:
        if (err := self._session_required()) or context is None:
            return err or "记忆命令需要消息上下文。"
        return self.session_svc.cmd_remember(arg, context)

    def _cmd_forget(self, arg: str, context: Any) -> str:
        if (err := self._session_required()) or context is None:
            return err or "记忆命令需要消息上下文。"
        return self.session_svc.cmd_forget(arg, context)

    def _cmd_memories(self, context: Any) -> str:
        if (err := self._session_required()) or context is None:
            return err or "记忆命令需要消息上下文。"
        return self.session_svc.cmd_memories(context)
