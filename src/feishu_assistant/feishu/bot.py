"""飞书 bot：WebSocket 长连接接收消息 + REST 回复/推送消息。

WS 长连接模式无需公网 IP / 域名，适合跑在个人 Mac 上。
lark-oapi 的 ws.Client.start() 是阻塞式的（内部自带断线重连），
因此定时任务要用 APScheduler 的 BackgroundScheduler（见 main.py）。

P2 新增：
- file 消息（PDF 附件）入口，转交精读服务
- send_text / send_card：按 chat_id 主动推送（精读异步结果用）
- 收到消息、路由结果全程 INFO 日志

P3 新增：
- 文件消息按类型分流：音频扩展 / audio 语音消息 → 求职复盘；
  docx 与文件名像简历的 PDF → 简历库；其余 PDF → 精读

线程模型：WS 回调（事件循环线程）只排队不干活——消息分发在单 worker 线程跑
（router.handle 里的 LLM 调用可达数分钟，同步跑会饿死 ping_loop 并阻塞后续
所有消息）；卡片回调必须同步返回 toast，仅把慢副作用（卡片回刷的串行 REST）
挪到 card worker。两个 worker 都是单线程：回复/副作用顺序与同步时代一致。

事件通知外发箱（NotificationOutbox）：失败/卡死/恢复类「用户必须知道」的
通知先落 sqlite，outbox flusher 后台线程（daemon，30s 一轮）取 pending 逐条
send_text——送达才 mark_delivered，失败 mark_attempt 下轮再试。断网期间
通知排队不丢，网络恢复后自动补发；flusher 异常绝不外溢到主流程。
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import threading
from typing import Any

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
    P2ImMessageReceiveV1,
    ReplyMessageRequest,
    ReplyMessageRequestBody,
)

from ..config import Settings
from ..modules.paper import MessageContext, PaperReadingService
from ..modules.router import Router
from ..store import DedupStore, NotificationOutbox

logger = logging.getLogger("feishu_assistant.bot")

# 群聊中 @机器人 会在文本里留下 @_user_N 占位符，剔除
_MENTION_PATTERN = re.compile(r"@_user_\d+")


class _NullLock:
    """空锁：__new__ 构造的 bot（测试）没有真锁时的退化和语义。"""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> bool:
        return False


_NULL_LOCK = _NullLock()

# 文件分流：音频扩展 → 求职复盘；docx → 简历库；文件名像简历的 PDF → 简历库
# （裸 PDF 默认仍是论文精读——文件名不含简历特征时按论文处理）
_AUDIO_EXTENSIONS = (".opus", ".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg", ".amr", ".webm")
_RESUME_DOC_EXTENSIONS = (".docx", ".doc")
_RESUME_NAME_RE = re.compile(r"简历|resume|cv", re.IGNORECASE)
_EXP_AUDIO_NAME_RE = re.compile(r"面经|interview[_ -]?experience", re.IGNORECASE)


def extract_text(data: P2ImMessageReceiveV1) -> str | None:
    """从消息事件中提取纯文本。非文本消息返回 None。"""
    msg = data.event.message
    if msg.message_type != "text":
        return None
    try:
        content = json.loads(msg.content or "{}")
    except json.JSONDecodeError:
        logger.warning("消息 content 不是合法 JSON：%r", msg.content)
        return ""
    return _MENTION_PATTERN.sub("", content.get("text", "")).strip()


def extract_file(data: P2ImMessageReceiveV1) -> tuple[str, str] | None:
    """从消息事件中提取文件附件信息 (file_name, file_key)。非文件消息返回 None。"""
    msg = data.event.message
    if msg.message_type != "file":
        return None
    try:
        content = json.loads(msg.content or "{}")
    except json.JSONDecodeError:
        logger.warning("文件消息 content 不是合法 JSON：%r", msg.content)
        return None
    return content.get("file_name", ""), content.get("file_key", "")


def extract_audio(data: P2ImMessageReceiveV1) -> str | None:
    """从语音（audio）消息中提取 file_key；非语音消息返回 None。"""
    msg = data.event.message
    if msg.message_type != "audio":
        return None
    try:
        content = json.loads(msg.content or "{}")
    except json.JSONDecodeError:
        logger.warning("语音消息 content 不是合法 JSON：%r", msg.content)
        return None
    return content.get("file_key", "")


def _sender_open_id(data: P2ImMessageReceiveV1) -> str:
    """取发送者 open_id（文档授权用）；异常时返回空串。"""
    try:
        return data.event.sender.sender_id.open_id or ""
    except AttributeError:
        return ""


# ---- 回复与推送辅助 ----

def _flatten_table_block(block: list[str]) -> list[str]:
    """管道表格块 → 键值/斜杠行（飞书卡片不支持表格语法）。

    两列 → 「• **键**：值」；三列及以上 → 首行加粗列名，数据行「• a / b / c」。
    无分隔行（|---|）时不按表格处理，原样返回。
    """
    rows: list[list[str]] = []
    for line in block:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    if len(rows) < 2 or not all(re.fullmatch(r":?-{2,}:?", c or "-") for c in rows[1]):
        return block  # 不是标准表格，原样
    header, data = rows[0], rows[2:]
    if len(header) == 2:
        out = []
        for r in data:
            key = r[0] if len(r) > 0 else ""
            val = r[1] if len(r) > 1 else ""
            out.append(f"• **{key}**：{val}")
        return out
    out = ["**" + " / ".join(header) + "**"]
    for r in data:
        out.append("• " + " / ".join(r))
    return out


def md_for_card(markdown: str) -> str:
    """把通用 markdown 转成飞书卡片 lark_md 可渲染的形态。

    - `## 标题`（1~4 级）→ `**标题**`
    - 管道表格块 → 键值/斜杠行（见 _flatten_table_block）
    - 其余（加粗、链接、列表、行内代码）原样保留
    """
    out: list[str] = []
    table_buf: list[str] = []

    def flush_table() -> None:
        nonlocal table_buf
        if table_buf:
            out.extend(_flatten_table_block(table_buf))
            table_buf = []

    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|") and "|" in stripped[1:-1]:
            table_buf.append(line)
            continue
        flush_table()
        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            out.append(f"**{m.group(2)}**")
        else:
            out.append(line)
    flush_table()
    return "\n".join(out)


class FeishuBot:
    """事件分发 + 文本/文件消息收发 + 主动推送。所有用户可见文案均为中文。"""

    def __init__(self, settings: Settings, router: Router,
                 outbox: NotificationOutbox | None = None) -> None:
        self.settings = settings
        self.router = router
        self.paper_service: PaperReadingService | None = None  # main.py 接线后注入
        self.career_service: Any = None  # main.py 接线后注入（P3 简历/音频分流）
        self.dedup: DedupStore | None = None  # main.py 接线后注入（私聊消息幂等）
        # 事件通知外发箱：失败/卡死/恢复必达（None = 不启用补发）
        self.outbox = outbox
        self._outbox_flusher_thread: threading.Thread | None = None
        self._outbox_flush_stop = threading.Event()
        self.client = (
            lark.Client.builder()
            .app_id(settings.feishu_app_id)
            .app_secret(settings.feishu_app_secret)
            .log_level(lark.LogLevel.INFO)
            .build()
        )
        # 消息分发 worker：单线程保序（回复顺序与同步时代一致），LLM 分钟级调用不占事件循环
        self._msg_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="feishu-msg")
        # 卡片回调慢副作用 worker（卡片回刷等串行 REST）
        self._card_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="feishu-card")
        # Card Kit 流式序号：按 card_id 单调递增、锁内分配；status.stop 经 status_card_close 清理
        self._status_card_seqs: dict[str, int] = {}
        self._status_card_lock = threading.Lock()

    # ---- 事件处理 ----

    def _on_card_action(self, data):
        """卡片按钮回调（card.action.trigger）：处理精选库「已读 / 精读这篇」。

        按钮 value 约定（digest 卡片注入）：{"act": "mark_read"|"read_paper",
        "arxiv_id": ..., "title": ...}。返回 toast 即时反馈。
        toast 依赖查库结果的动作（mark_read）其状态变更必须同步完成；
        卡片回刷（读原卡片 + PATCH）是串行 REST，挪 card worker 不占事件循环。
        """
        from lark_oapi.event.callback.model.p2_card_action_trigger import (
            CallBackCard,
            CallBackToast,
            P2CardActionTriggerResponse,
        )

        def _resp(text: str, ok: bool = True) -> P2CardActionTriggerResponse:
            toast = CallBackToast()
            toast.type = "success" if ok else "warning"
            toast.content = text
            resp = P2CardActionTriggerResponse()
            resp.toast = toast
            return resp

        try:
            value = (data.event.action.value or {}) if data and data.event and data.event.action else {}
            act = str(value.get("act") or "")
            arxiv_id = str(value.get("arxiv_id") or "").strip()
            title = str(value.get("title") or "").strip() or arxiv_id
            logger.info("卡片回调：act=%s arxiv=%s title=%s", act, arxiv_id, title[:30])

            # 模型路由面板：下拉切换 / 恢复默认 / 添加 provider 表单翻页（不需要 paper_service）
            if act in ("model_switch", "model_reset", "model_add_provider",
                       "model_add_form", "model_panel", "model_confirm_provider"):
                msvc = getattr(self, "model_service", None)
                if msvc is None:
                    return _resp("模型服务未就绪，稍后再试", ok=False)
                option = str(getattr(data.event.action, "option", "") or "")
                form_value = dict(getattr(data.event.action, "form_value", None) or {})
                chain = str(value.get("chain") or "")
                toast, ok, card = msvc.handle_card_action(
                    act, chain=chain, option=option, form_value=form_value)
                resp = _resp(toast, ok)
                if card is not None:  # None = 不刷新卡片（表单已填内容保留）
                    cb = CallBackCard()
                    cb.type = "raw"
                    cb.data = card
                    resp.card = cb
                if act == "model_switch" and ok:  # 切换回执（防误触，可撤销）
                    chat_id = (getattr(data.event.context, "open_chat_id", "")
                               if data.event.context else "")
                    if chat_id:
                        try:
                            self.send_text(
                                chat_id,
                                f"✅ 已切换 {chain} 首选 → {option}；回复「撤销切换」可回滚。")
                        except Exception:
                            logger.exception("模型切换回执发送失败")
                return resp

            svc = self.paper_service
            if svc is None or svc.bitable_dao is None:
                return _resp("服务未就绪，稍后再试", ok=False)

            if act == "mark_read":
                record_id = svc.bitable_dao.find_record_by_field("精选库", "arXivID", arxiv_id)
                if not record_id:
                    return _resp("精选库里没找到这篇（可能不是日报来源）", ok=False)
                svc.bitable_dao.update_record("精选库", record_id, {"状态": "已读"})
                # 卡片回刷：把这篇的「已读」按钮换成 ✓ 已读，点没点过一眼可辨。
                # 读原卡片 + PATCH 是串行 REST，挪 worker 线程；toast 不依赖回刷结果。
                self._submit_card_work(lambda: self._refresh_card_read(data, arxiv_id))
                return _resp(f"✅ 已标记已读：《{title[:40]}》")

            if act == "read_paper":
                ctx = None
                # CallBackContext 的会话字段是 open_chat_id（不是 chat_id）
                chat_id = getattr(data.event.context, "open_chat_id", "") if data.event.context else ""
                if chat_id:
                    from ..modules.paper import MessageContext
                    ctx = MessageContext(chat_id=chat_id,
                                         open_id=getattr(data.event.operator, "open_id", "") or "",
                                         message_id="")
                if ctx is None:
                    return _resp("拿不到会话上下文，直接把链接发给我即可", ok=False)
                r = svc._start_async(ctx, label=f"arXiv:{arxiv_id}", job=svc._job_arxiv,
                                     dedup_key=f"arxiv:{arxiv_id}", arxiv_id=arxiv_id)
                if r is None:  # 防连点：同一篇已有在飞精读任务
                    return _resp("这篇已在精读中，骨架出来后会推送", ok=False)
                return _resp(f"📄 已开始精读《{title[:40]}》，骨架约 2 分钟到")

            return _resp(f"暂不支持的按钮动作：{act or '空'}", ok=False)
        except Exception as exc:  # 回调绝不炸事件循环
            logger.exception("卡片回调处理失败：%s", exc)
            return _resp("处理出错了，去日志看看", ok=False)

    def _restyle_card_read(self, data: Any, arxiv_id: str) -> dict | None:
        """取回调消息原卡片，把目标论文的「已读」按钮换成 ✓ 已读样式。失败返回 None。"""
        try:
            msg_id = getattr(getattr(data.event, "context", None), "open_message_id", "") or ""
            if not msg_id:
                return None
            from lark_oapi.api.im.v1 import GetMessageRequest
            resp = self.client.im.v1.message.get(
                GetMessageRequest.builder().message_id(msg_id).build())
            if not resp.success():
                return None
            card = json.loads(resp.data.items[0].body.content)

            def _walk(node: Any) -> bool:
                if isinstance(node, dict):
                    if (node.get("tag") == "button"
                            and isinstance(node.get("value"), dict)
                            and node["value"].get("act") == "mark_read"
                            and str(node["value"].get("arxiv_id")) == arxiv_id):
                        node["text"] = {"tag": "plain_text", "content": "✓ 已读"}
                        node["type"] = "default"
                        return True
                    return any(_walk(v) for v in node.values())
                if isinstance(node, list):
                    return any(_walk(v) for v in node)
                return False

            return card if _walk(card.get("elements", card)) else None
        except Exception:
            logger.exception("已读卡片回刷失败（不影响状态更新）")
            return None

    def _submit_card_work(self, fn) -> None:
        """卡片回调的慢副作用（串行 REST）挪到 card worker，事件循环只留同步应答。
        测试里 __new__ 构造的 bot 没有 executor：退化为同步执行（语义一致）。"""
        ex = getattr(self, "_card_executor", None)
        if ex is None:
            fn()
            return
        ex.submit(self._run_card_work, fn)

    @staticmethod
    def _run_card_work(fn) -> None:
        """worker 里跑卡片副作用：异常只记日志（futures 会静默吞异常，必须主动兜）。"""
        try:
            fn()
        except Exception:
            logger.exception("卡片后台副作用执行失败（不影响已应答的 toast）")

    def _refresh_card_read(self, data: Any, arxiv_id: str) -> None:
        """worker 线程：取原卡片把目标论文的「已读」按钮换成 ✓ 已读，再 PATCH 回原消息。
        失败仅记日志——状态变更（精选库已读）早已同步完成，这里只是样式回刷。"""
        msg_id = getattr(getattr(data.event, "context", None), "open_message_id", "") or ""
        if not msg_id:
            return
        new_card = self._restyle_card_read(data, arxiv_id)
        if new_card is None:
            return
        from lark_oapi.api.im.v1 import UpdateMessageRequest, UpdateMessageRequestBody
        req = (
            UpdateMessageRequest.builder()
            .message_id(msg_id)
            .request_body(
                UpdateMessageRequestBody.builder()
                .msg_type("interactive")
                .content(json.dumps(new_card, ensure_ascii=False))
                .build()
            )
            .build()
        )
        try:
            resp = self.client.im.v1.message.update(req)
        except Exception:
            logger.exception("已读卡片回刷 PATCH 传输失败")
            return
        if not resp.success():
            logger.warning(
                "已读卡片回刷失败：code=%s, msg=%s, log_id=%s", resp.code, resp.msg, resp.get_log_id()
            )

    def _enqueue_message(self, data: P2ImMessageReceiveV1) -> None:
        """WS 事件入口（事件循环线程）：只排队，分发挪到 worker 线程。

        router.handle 里的 LLM 调用可达数分钟，在事件循环上同步跑会饿死
        ping_loop（长连接掉线）并阻塞后续所有消息。入队失败仅记日志——此时
        dedup 尚未占位，飞书重投仍会重新处理，不丢消息。
        """
        ex = getattr(self, "_msg_executor", None)
        if ex is None:  # 测试里 __new__ 构造的 bot：同步执行（语义一致）
            self._run_message_safe(data)
            return
        try:
            ex.submit(self._run_message_safe, data)
        except Exception:
            logger.exception("消息入队失败（dedup 未占位，等飞书重投）")

    def _run_message_safe(self, data: P2ImMessageReceiveV1) -> None:
        """worker 线程里跑消息分发：异常绝不外溢（futures 会静默吞掉，必须主动兜）。"""
        try:
            self._on_message_receive(data)
        except Exception:
            logger.exception("消息处理未捕获异常")

    def _on_message_receive(self, data: P2ImMessageReceiveV1) -> None:
        """消息事件分发入口（worker 线程；测试直接同步调用）。

        生命周期约定：mark_processing 占位 → _dispatch 处理并回复 → mark_done 记终态。
        「处理完成」= 回复已送达或本就无需回复；LLM 算完但回复没送达不算完成——
        写死信（可 /重放），不能悄无声息地标 done 判死。任何异常都不抛给 WS 层
        （SDK 会把异常变成 500 应答导致飞书重投，与 dedup 语义重复）。
        """
        sender_type = getattr(data.event.sender, "sender_type", "")
        if sender_type and sender_type != "user":
            return  # 忽略其他 bot / 系统消息，避免自回复死循环

        msg = data.event.message
        # 私聊文本/文件消息幂等：飞书长连接"至少一次"投递，重启会重投近期消息。
        # 先标记处理中；已见过（含上次崩溃残留的"处理中"）一律跳过——宁可漏答不可重复答。
        is_p2p = msg.chat_type == "p2p"
        msg_id = msg.message_id or ""
        if self.dedup is not None and is_p2p and msg_id:
            if not self.dedup.mark_processing(msg_id):
                # 重复投递或崩溃残留：跳过 + 写死信（可 /重放）
                prev_status = self.dedup.status(msg_id)
                reason = ("重复投递（已处理过）" if prev_status == "done"
                          else "崩溃残留（上次处理未完成，按已处理跳过）")
                logger.info("重复消息已跳过：%s（%s）", msg_id, reason)
                self._write_dead_letter(data, msg_id, reason)
                return
            delivered = False
            try:
                delivered = self._dispatch(data)
            except Exception:
                # _dispatch 各分支已兜底成中文错误回复，这里是最后防线
                logger.exception("消息分发未捕获异常：%s", msg_id)
            finally:
                self.dedup.mark_done(msg_id)
            if not delivered:
                # 处理完但回复没送达：不是"处理完成"，留死信给用户 /重放
                self._write_dead_letter(data, msg_id, "已处理但回复未送达（可 /重放 找回这条）")
            return
        try:
            self._dispatch(data)
        except Exception:
            logger.exception("消息分发未捕获异常")

    def _write_dead_letter(self, data: P2ImMessageReceiveV1, msg_id: str, reason: str) -> None:
        """被跳过的消息留档 dead_letter；失败仅记日志。"""
        try:
            msg = data.event.message
            file_info = extract_file(data)
            if file_info is not None:
                msg_type, summary = "file", file_info[0]
            else:
                msg_type = msg.message_type or "text"
                summary = (extract_text(data) or "")[:80]
            self.dedup.add_dead_letter(
                msg_id, msg.chat_id or "", reason,
                open_id=_sender_open_id(data), msg_type=msg_type, content_text=summary,
            )
        except Exception:
            logger.exception("写死信失败：%s", msg_id)

    def _dispatch(self, data: P2ImMessageReceiveV1) -> bool:
        """分发一条消息。返回 True = 处理完成（回复已送达或本就无需回复）；
        False = 处理完但回复没送达（调用方据此写死信，/重放 可找回）。
        各分支的业务异常已兜底成中文错误回复；回复动作本身永不抛异常。"""
        msg = data.event.message
        context = MessageContext(
            chat_id=msg.chat_id or "",
            open_id=_sender_open_id(data),
            message_id=msg.message_id or "",
        )
        logger.info(
            "收到消息：chat=%s type=%s user=%s",
            context.chat_id, msg.message_type, context.open_id[-6:] or "?",
        )

        # 语音消息（audio → 求职录音复盘/语音面经）
        audio_key = extract_audio(data)
        if audio_key is not None:
            if self.career_service is None:
                return self.reply_text(context.message_id, "语音复盘功能未就绪，请稍后再试。")
            logger.info("路由命中：语音消息 → 录音复盘")
            try:
                reply = self.career_service.handle_audio_file("语音消息.opus", audio_key, context)
            except Exception:
                logger.exception("处理语音消息出错")
                reply = "抱歉，处理这条语音时出错了，请稍后再试。"
            if reply:
                return self.reply_text(context.message_id, reply)
            return True

        # 文件附件（按类型分流：音频→复盘 / docx·简历PDF→简历库 / 其余 PDF→精读）
        file_info = extract_file(data)
        if file_info is not None:
            file_name, file_key = file_info
            try:
                reply = self._route_file(file_name, file_key, context)
            except Exception:
                logger.exception("处理文件消息出错")
                reply = "抱歉，处理这个文件时出错了，请稍后再试。"
            if reply:
                return self.reply_text(context.message_id, reply)
            return True

        text = extract_text(data)
        if text is None:
            return self.reply_text(
                context.message_id,
                "目前支持文本、文件（PDF/简历/音频）和语音消息，其他类型后续版本再支持～")
        if not text:
            return True

        logger.info("消息文本：%s", text[:80])
        try:
            reply = self.router.handle(text, context)
        except Exception:
            logger.exception("处理消息出错")
            reply = "抱歉，处理这条消息时出错了，请稍后再试。也可以发 /help 查看用法。"
        if reply:
            return self.reply_text(context.message_id, reply)
        return True

    # ---- 回复与推送 ----

    def _route_file(self, file_name: str, file_key: str, context: MessageContext) -> str:
        """文件消息分流：音频→录音复盘；docx/简历名 PDF→简历库；其余 PDF→精读。"""
        lower = file_name.lower()
        if lower.endswith(_AUDIO_EXTENSIONS):
            if self.career_service is None:
                return "语音复盘功能未就绪，请稍后再试。"
            mode = "exp" if _EXP_AUDIO_NAME_RE.search(file_name) else "review"
            logger.info(
                "路由命中：音频文件《%s》→ %s",
                file_name, "语音面经" if mode == "exp" else "录音复盘",
            )
            return self.career_service.handle_audio_file(file_name, file_key, context, mode=mode)
        is_resume = lower.endswith(_RESUME_DOC_EXTENSIONS) or (
            lower.endswith(".pdf") and _RESUME_NAME_RE.search(file_name)
        )
        if is_resume:
            if self.career_service is not None:
                logger.info("路由命中：简历文件《%s》→ 简历库", file_name)
                return self.career_service.handle_resume_file(file_name, file_key, context)
            if not lower.endswith(".pdf"):
                return "简历入库功能未就绪，请稍后再试。"
            # career 缺位时的 PDF：兜底走精读（paper 只能接 PDF）
        if self.paper_service is None:
            return "精读功能未就绪，请稍后再试。"
        logger.info("路由命中：PDF 附件《%s》→ 精读", file_name)
        return self.paper_service.handle_file(file_name, file_key, context)

    # ---- 回复与推送 ----

    def reply_text(self, message_id: str, text: str) -> bool:
        """引用回复一条文本消息。失败仅记日志不抛异常（避免影响事件循环）。
        返回是否送达：False 时调用方可把消息写死信（处理完但没送达 ≠ 处理完成）。"""
        req = (
            ReplyMessageRequest.builder()
            .message_id(message_id)
            .request_body(
                ReplyMessageRequestBody.builder()
                .msg_type("text")
                .content(json.dumps({"text": text}, ensure_ascii=False))
                .build()
            )
            .build()
        )
        try:
            resp = self.client.im.v1.message.reply(req)
        except Exception:  # 传输层异常（断网/超时/SDK 内部错误）与 API 错误码同等对待
            logger.exception("回复消息传输失败：message_id=%s", message_id)
            return False
        if not resp.success():
            logger.error(
                "回复消息失败：code=%s, msg=%s, log_id=%s", resp.code, resp.msg, resp.get_log_id()
            )
            return False
        return True

    def _send(self, chat_id: str, msg_type: str, content: dict, what: str) -> bool:
        """按 chat_id 主动发消息（异步结果推送用）。失败仅记日志返回 False，不抛异常。"""
        req = (
            CreateMessageRequest.builder()
            .receive_id_type("chat_id")
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(chat_id)
                .msg_type(msg_type)
                .content(json.dumps(content, ensure_ascii=False))
                .build()
            )
            .build()
        )
        try:
            resp = self.client.im.v1.message.create(req)
        except Exception:  # 传输层异常（断网/超时/SDK 内部错误）与 API 错误码同等对待
            logger.exception("推送%s传输失败：chat=%s", what, chat_id)
            return False
        if not resp.success():
            logger.error(
                "推送%s失败：code=%s, msg=%s, log_id=%s", what, resp.code, resp.msg, resp.get_log_id()
            )
            return False
        logger.info("已推送%s：chat=%s", what, chat_id)
        return True

    def send_text(self, chat_id: str, text: str) -> bool:
        return self._send(chat_id, "text", {"text": text}, "文本")

    # ---- 状态条：Card Kit 流式卡（无普通消息的编辑次数上限，可高频刷新）----

    def status_card_create(self, chat_id: str, text: str) -> str:
        """创建 Card Kit 流式状态卡并发到会话；返回句柄 "card_id|element_id"，失败 ""。"""
        from lark_oapi.api.cardkit.v1 import CreateCardRequest, CreateCardRequestBody

        element_id = "status_line"
        card_json = {
            "schema": "2.0",
            "config": {"streaming_mode": True, "update_multi": True},
            "body": {"elements": [
                {"tag": "markdown", "element_id": element_id, "content": text},
            ]},
        }
        body = (CreateCardRequestBody.builder().type("card_json")
                .data(json.dumps(card_json, ensure_ascii=False)).build())
        try:
            resp = self.client.cardkit.v1.card.create(
                CreateCardRequest.builder().request_body(body).build())
        except Exception:  # 传输层异常与 API 错误码同等对待：返回 "" 让调用方降级
            logger.exception("创建状态卡传输失败")
            return ""
        if not resp.success():
            logger.error(
                "创建状态卡失败：code=%s, msg=%s, log_id=%s", resp.code, resp.msg, resp.get_log_id()
            )
            return ""
        card_id = str(resp.data.card_id or "")
        if not card_id:
            return ""
        # msg_type 仍是 interactive，content 里用 card_id 引用卡片实体
        if not self._send(chat_id, "interactive",
                          {"type": "card", "data": {"card_id": card_id}}, "状态卡"):
            self._discard_orphan_card(card_id, element_id)
            return ""
        return f"{card_id}|{element_id}"

    def rich_card_create(self, chat_id: str, card_json: dict) -> str:
        """发送完整结构的流式卡片（header/多元素，streaming+update_multi 持续更新）。
        返回 card_id（失败 ""）；元素级更新走 status_card_update("card_id|element_id", text)。"""
        from lark_oapi.api.cardkit.v1 import CreateCardRequest, CreateCardRequestBody

        card_json = dict(card_json)
        config = dict(card_json.get("config") or {})
        config.update({"streaming_mode": True, "update_multi": True})
        card_json["config"] = config
        card_json.setdefault("schema", "2.0")
        body = (CreateCardRequestBody.builder().type("card_json")
                .data(json.dumps(card_json, ensure_ascii=False)).build())
        try:
            resp = self.client.cardkit.v1.card.create(
                CreateCardRequest.builder().request_body(body).build())
        except Exception:
            logger.exception("创建富文本流式卡传输失败")
            return ""
        if not resp.success():
            logger.error(
                "创建富文本流式卡失败：code=%s, msg=%s, log_id=%s",
                resp.code, resp.msg, resp.get_log_id())
            return ""
        card_id = str(resp.data.card_id or "")
        if not card_id:
            return ""
        if not self._send(chat_id, "interactive",
                          {"type": "card", "data": {"card_id": card_id}}, "任务卡"):
            logger.error("任务卡消息发送失败：card_id=%s", card_id)
            return ""
        return card_id

    def _discard_orphan_card(self, card_id: str, element_id: str) -> None:
        """发送失败的卡片实体就地废弃：平台没有删除卡片实体的 API（CardKit 仅支持
        删组件），写墓碑内容（万一浮出可见）+ 释放本地流式序号，card_id 留日志供排查。"""
        logger.error("状态卡发送失败，卡片实体就地废弃（平台无删除实体 API）：card_id=%s", card_id)
        handle = f"{card_id}|{element_id}"
        try:
            self.status_card_update(handle, "⚠️ 状态卡发送失败，此卡片已废弃")
        except Exception:
            logger.exception("废弃卡片实体写墓碑失败")
        self.status_card_close(handle)

    def _card_seq_state(self) -> tuple[dict, Any]:
        """流式序号字典 + 锁（__new__ 构造的 bot 没有这两样：惰性补建，退化为无锁）。"""
        seqs = getattr(self, "_status_card_seqs", None)
        if seqs is None:
            seqs = self._status_card_seqs = {}
        return seqs, getattr(self, "_status_card_lock", None) or _NULL_LOCK

    def status_card_update(self, handle: str, text: str) -> str:
        """流式更新状态卡文本（无编辑次数上限）。返回 "" 成功，否则错误码字符串。"""
        import uuid as _uuid
        from lark_oapi.api.cardkit.v1 import (
            ContentCardElementRequest, ContentCardElementRequestBody,
        )

        card_id, _, element_id = handle.partition("|")
        seqs, lock = self._card_seq_state()
        with lock:  # 序号单调递增、锁内分配：stop 定格帧与刷新线程并发不撞号
            seqs[card_id] = seqs.get(card_id, 0) + 1
            seq = seqs[card_id]
        body = (ContentCardElementRequestBody.builder()
                .content(text).sequence(seq).uuid(str(_uuid.uuid4())).build())
        req = (ContentCardElementRequest.builder()
               .card_id(card_id).element_id(element_id).request_body(body).build())
        resp = self.client.cardkit.v1.card_element.content(req)
        if not resp.success():
            logger.warning(
                "更新状态卡失败：code=%s, msg=%s, log_id=%s", resp.code, resp.msg, resp.get_log_id()
            )
            return str(resp.code)
        return ""

    def status_card_reactivate(self, handle: str) -> bool:
        """重新激活流式卡的 streaming_mode：平台在最后一次激活 ~10 分钟后（滑动）
        自动关闭流式窗口，关闭前调本方法给同一张卡续命，可继续流式更新。
        成功 True；传输异常/API 失败 False 记日志（调用方下帧重试）。
        序号与内容更新共用同一计数：重激活后 seq 继续单调递增（不重置）。"""
        import uuid as _uuid
        from lark_oapi.api.cardkit.v1 import (
            SettingsCardRequest, SettingsCardRequestBody,
        )

        card_id, _, _ = handle.partition("|")
        if not card_id:
            return False
        seqs, lock = self._card_seq_state()
        with lock:  # 与内容更新同一序号空间、锁内分配：settings 与 content 不撞号
            seqs[card_id] = seqs.get(card_id, 0) + 1
            seq = seqs[card_id]
        body = (SettingsCardRequestBody.builder()
                .settings(json.dumps({"config": {"streaming_mode": True}},
                                     ensure_ascii=False))
                .sequence(seq).uuid(str(_uuid.uuid4())).build())
        req = (SettingsCardRequest.builder()
               .card_id(card_id).request_body(body).build())
        try:
            resp = self.client.cardkit.v1.card.settings(req)
        except Exception:  # 传输层异常与 API 错误码同等对待：返回 False 让调用方重试
            logger.exception("重激活状态卡传输失败：card_id=%s", card_id)
            return False
        if not resp.success():
            logger.warning(
                "重激活状态卡失败：code=%s, msg=%s, log_id=%s",
                resp.code, resp.msg, resp.get_log_id()
            )
            return False
        return True

    def status_card_append(self, card_id: str, content: str) -> str:
        """向流式卡追加一个 markdown 元素（完成后追加结果区）。
        返回 "" 成功，否则错误码字符串。元素 id 自动分配（append-result-{seq}）。"""
        import uuid as _uuid
        from lark_oapi.api.cardkit.v1 import (
            CreateCardElementRequest, CreateCardElementRequestBody,
        )

        seqs, lock = self._card_seq_state()
        with lock:
            seqs[card_id] = seqs.get(card_id, 0) + 1
            seq = seqs[card_id]
        element = {"tag": "markdown", "element_id": f"append_result_{seq}",
                   "content": content}
        body = (CreateCardElementRequestBody.builder()
                .type("append").uuid(str(_uuid.uuid4())).sequence(seq)
                .elements(json.dumps([element], ensure_ascii=False)).build())
        req = (CreateCardElementRequest.builder()
               .card_id(card_id).request_body(body).build())
        try:
            resp = self.client.cardkit.v1.card_element.create(req)
        except Exception:
            logger.exception("追加卡片元素传输失败")
            return "exception"
        if not resp.success():
            logger.warning(
                "追加卡片元素失败：code=%s, msg=%s, log_id=%s",
                resp.code, resp.msg, resp.get_log_id())
            return str(resp.code)
        return ""

    def status_card_close(self, handle: str) -> None:
        """状态卡句柄终结：清理按 card_id 累积的流式序号（status.stop 调用，防只增不清的泄漏）。"""
        card_id, _, _ = handle.partition("|")
        if not card_id:
            return
        seqs, lock = self._card_seq_state()
        with lock:
            seqs.pop(card_id, None)

    def status_card_current_seq(self, card_id: str) -> int:
        """读取某卡当前流式序号（任务卡接管时持久化用）。"""
        card_id = card_id.partition("|")[0]
        seqs, lock = self._card_seq_state()
        with lock:
            return int(seqs.get(card_id, 0))

    def status_card_seed_seq(self, card_id: str, seq: int) -> None:
        """接管旧卡时回填流式序号基线（继续单调递增，不与历史帧撞号）。"""
        card_id = card_id.partition("|")[0]
        if not card_id:
            return
        seqs, lock = self._card_seq_state()
        with lock:
            seqs[card_id] = max(int(seq), seqs.get(card_id, 0))

    def send_text_id(self, chat_id: str, text: str) -> str:
        """发文本并返回 message_id（状态条后续 PATCH 用）；失败返回 ""。"""
        req = (
            CreateMessageRequest.builder()
            .receive_id_type("chat_id")
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(chat_id)
                .msg_type("text")
                .content(json.dumps({"text": text}, ensure_ascii=False))
                .build()
            )
            .build()
        )
        resp = self.client.im.v1.message.create(req)
        if not resp.success():
            logger.error(
                "发送状态条失败：code=%s, msg=%s, log_id=%s", resp.code, resp.msg, resp.get_log_id()
            )
            return ""
        return str(resp.data.message_id or "")

    def update_text(self, message_id: str, text: str) -> str:
        """PATCH 更新一条文本消息（状态条逐帧刷新用）。返回 "" 成功，否则错误码字符串。
        230072 = 单条消息编辑次数耗尽（状态条据此换新消息续命）。"""
        from lark_oapi.api.im.v1 import UpdateMessageRequest, UpdateMessageRequestBody
        req = (
            UpdateMessageRequest.builder()
            .message_id(message_id)
            .request_body(
                UpdateMessageRequestBody.builder()
                .msg_type("text")
                .content(json.dumps({"text": text}, ensure_ascii=False))
                .build()
            )
            .build()
        )
        resp = self.client.im.v1.message.update(req)
        if not resp.success():
            logger.warning(
                "更新消息失败：code=%s, msg=%s, log_id=%s", resp.code, resp.msg, resp.get_log_id()
            )
            return str(resp.code)
        return ""

    def send_card(self, chat_id: str, card: dict) -> bool:
        return self._send(chat_id, "interactive", card, "卡片")

    # ---- 启动 ----

    # ---- 事件通知外发箱补发 ----

    def start_outbox_flusher(self, interval: float = 30.0) -> None:
        """启动外发箱补发线程（daemon，默认 30s 一轮）。无 outbox 或已启动则跳过。"""
        if self.outbox is None or self._outbox_flusher_thread is not None:
            return
        self._outbox_flush_stop.clear()  # 允许 stop 后重新启动
        self._outbox_flusher_thread = threading.Thread(
            target=self._outbox_flush_loop, args=(interval,),
            name="outbox-flusher", daemon=True,
        )
        self._outbox_flusher_thread.start()
        logger.info("事件通知外发箱补发线程已启动（%.0fs 一轮）", interval)

    def stop_outbox_flusher(self) -> None:
        """停补发线程（测试清理用）；未启动时静默跳过。"""
        self._outbox_flush_stop.set()
        thread = self._outbox_flusher_thread
        if thread is not None:
            thread.join(timeout=2)
            self._outbox_flusher_thread = None

    def _outbox_flush_loop(self, interval: float) -> None:
        """补发循环：启动先补一轮（崩溃残留通知尽快送达），之后每 interval 一轮。
        单轮异常只记日志（线程不退出、不炸主流程）。用 Event.wait 而非
        time.sleep：可被 stop_outbox_flusher 即时唤醒，也不受测试对
        time.sleep 的进程级打桩影响。"""
        while not self._outbox_flush_stop.is_set():
            try:
                self.flush_outbox_once()
            except Exception:
                logger.exception("外发箱补发一轮失败（下轮继续）")
            self._outbox_flush_stop.wait(interval)

    def flush_outbox_once(self) -> int:
        """补发一轮：pending 逐条 send_text，成功 mark_delivered，失败 mark_attempt
        下轮再试；单条异常不中断整轮。返回本轮送达条数。"""
        outbox = self.outbox
        if outbox is None:
            return 0
        delivered = 0
        for msg in outbox.pending():
            try:
                ok = self.send_text(msg.chat_id, msg.text)
            except Exception:  # send_text 自身不抛，这里兜 fake/未知实现
                logger.exception("外发箱补发异常：id=%s", msg.id)
                ok = False
            try:
                if ok:
                    outbox.mark_delivered(msg.id)
                    delivered += 1
                else:
                    outbox.mark_attempt(msg.id)
            except Exception:
                logger.exception("外发箱状态回写失败（下轮按未送达重试）：id=%s", msg.id)
        return delivered

    def build_ws_client(self) -> lark.ws.Client:
        handler = (
            lark.EventDispatcherHandler.builder("", "")  # WS 模式无需 encrypt_key / token
            .register_p2_im_message_receive_v1(self._enqueue_message)  # 只排队：分发在 worker 线程
            .register_p2_card_action_trigger(self._on_card_action)
            .build()
        )
        return lark.ws.Client(
            self.settings.feishu_app_id,
            self.settings.feishu_app_secret,
            event_handler=handler,
            log_level=lark.LogLevel.INFO,
        )

    def run(self) -> None:
        """启动 WS 长连接（阻塞，自带断线重连）。"""
        self.build_ws_client().start()
