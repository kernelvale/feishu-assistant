"""死信重放：/重放 列出死信，/重放 N 重新处理第 N 条。

被幂等跳过的消息（重复投递、崩溃残留）都会写 dead_letter 表。
重放 = 用死信里存的原文重新走一次 router 分发（不再经过去重闸）。

注意：重放只支持文本类死信（文件消息缺少可复用的 file_key 上下文，跳过并提示）。
接线（命令注册）见 P2b 收尾报告——本模块只提供服务与测试。
"""

# ---- [V7.1 清理] 已停用：死信重放模块从未接线（路由表无「重放」命令），未接线停用。经全项目核对生产链路零引用，保留备查 ----
# from __future__ import annotations
#
# import logging
# import time
# from typing import Any, Callable
#
# from ..store import DedupStore
# from .paper import MessageContext
#
# logger = logging.getLogger("feishu_assistant.replay")
#
# # dispatch_fn 签名：(text, context) -> str，即 router.handle
# DispatchFn = Callable[[str, Any], str]
#
#
# class ReplayService:
#     def __init__(self, dedup: DedupStore, dispatch_fn: DispatchFn | None = None) -> None:
#         self.dedup = dedup
#         self.dispatch_fn = dispatch_fn
#
#     def cmd_list(self, ctx: MessageContext) -> str:
#         letters = self.dedup.list_dead_letters(ctx.chat_id)
#         if not letters:
#             return "📭 没有待重放的死信（被跳过的消息会在这里出现）。"
#         lines = []
#         for i, letter in enumerate(letters, 1):
#             summary = letter.content_text.replace("\n", " ")[:30] or f"[{letter.msg_type}]"
#             when = time.strftime("%m-%d %H:%M", time.localtime(letter.created_at))
#             lines.append(f"{i}. [{when}] {summary}（{letter.reason}）")
#         return (
#             f"📬 死信列表（共 {len(letters)} 条，新的在前）\n" + "\n".join(lines)
#             + "\n\n回复「重放 N」重新处理第 N 条。"
#         )
#
#     def cmd_replay(self, arg: str, ctx: MessageContext) -> str:
#         try:
#             n = int(str(arg).strip())
#         except ValueError:
#             return "用法：重放 N（N 为死信列表里的序号）"
#         letters = self.dedup.list_dead_letters(ctx.chat_id)
#         if not (1 <= n <= len(letters)):
#             return f"没有第 {n} 条死信。发「重放」查看列表。"
#         letter = letters[n - 1]
#         if letter.msg_type != "text" or not letter.content_text:
#             self.dedup.mark_replayed(letter.id)
#             return f"第 {n} 条是 {letter.msg_type} 消息，暂不支持重放（已标记）。文本重发一遍即可。"
#         if self.dispatch_fn is None:
#             return "重放功能未接线（dispatch_fn 缺失），请联系开发者。"
#         logger.info("重放死信：id=%s message_id=%s", letter.id, letter.message_id)
#         reply = self.dispatch_fn(letter.content_text, ctx)
#         self.dedup.mark_replayed(letter.id)
#         return f"🔁 已重放第 {n} 条死信：\n{reply}"
