"""任务状态条：一条会"自己更新"的消息，替代刷屏式的阶段推送。

- 长任务开始时发一条消息，后台线程每 interval 秒 PATCH 一次：
  `⠙ 精读 arXiv:xxx · 03:41｜阶段：分层精讲｜活动：LLM·free-tokens/kimi-k3（45s）`
- 转圈只在**真有在途操作**（activity.REGISTRY 非空）时转动；
  无操作（排队等待/网络空等/卡死）帧停住，改显 💤 和静默时长——一眼分辨"在跑"还是"卡了"。
- 卡死看门狗：无在途活动持续 STUCK_ALERT_AFTER 秒且注入了 alert_fn 时，
  回调报警一次（paper 侧接到 NotificationOutbox，必达补发）；活动恢复后复位。
- stop(ok=True/False) 定格成 ✅/❌ 并停线程。

sender 需要两个方法（FeishuBot 提供）：
    send_text_id(chat_id, text) -> str        发消息并返回 message_id
    update_text(message_id, text) -> str      PATCH 更新；返回 "" 成功，否则错误码
                                              （230072 编辑次数耗尽 → 定格，绝不发新消息）
sender 若另有 Card Kit 流式方法则优先走流式卡（无编辑上限）：
    status_card_create(chat_id, text) -> str  返回句柄 "card_id|element_id"，失败 ""
                                              （返回空时降级为上面的文本消息路径）
    status_card_update(handle, text) -> str   流式更新；返回 "" 成功，否则错误码
    status_card_close(handle)                 可选：句柄终结时清理按卡累积的流式序号

帧写纪律：所有帧写（刷新线程的 _patch、stop 的定格帧）都过 _frame_lock；
stop 先停线程再写定格帧，定格帧必是最后一帧（_final_written 挡住迟到帧）。
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any

from ..activity import REGISTRY, ActivityRegistry

logger = logging.getLogger(__name__)

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_IDLE_WARN_AFTER = 60  # 无活动超过该秒数，状态条显式提示"疑似卡死"
STUCK_ALERT_AFTER = 300  # 无在途活动超过该秒数：alert_fn 报警一次（外发箱必达）
_LIVE_WINDOW = 540  # Card Kit 流式窗口实测 ~10 分钟硬死：9 分钟时主动写定格说明（仅卡片路径）
_STAGE_BUDGET = 16  # 单行三段各自截断：总长恒定，聊天窗不跳动
_ACT_BUDGET = 14
_EMOJI_PREFIX_RE = re.compile(r"^[\U0001F300-\U0001FAFF✅❌📄📝🧠⏰🔎🖼️]+\s*")


def _fmt_duration(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def _trunc(text: str, budget: int) -> str:
    return text if len(text) <= budget else text[: budget - 1] + "…"


def _short_activity(label: str) -> str:
    """活动标签缩成短名：LLM·free-tokens/kimi-k3 → kimi-k3；MinerU 解析 PDF → MinerU。"""
    if label.startswith("LLM·"):
        return label.rsplit("/", 1)[-1]
    return label.split()[0]


def _fmt_duration(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


class StatusMessenger:
    """一个长任务一条状态消息；set_stage 改阶段文案，stop 定格收尾。"""

    def __init__(
        self,
        sender: Any,
        chat_id: str,
        title: str,
        *,
        interval: float = 3.0,
        registry: ActivityRegistry = REGISTRY,
        alert_fn: Any = None,
    ) -> None:
        self.sender = sender
        self.chat_id = chat_id
        self.title = title
        self.interval = interval
        self.registry = registry
        # 卡死看门狗：alert_fn(text) 由调用方接到事件外发箱；None = 不启用
        self.alert_fn = alert_fn
        self._stuck_since: float | None = None  # 本轮静默起点（有活动即清零）
        self._stuck_alerted = False  # 本轮静默已报过警（一次静默只报一次）
        self._stage = "启动中"
        self._t0 = time.time()
        self._frame = 0
        self._idle_since: float | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._msg_id = ""
        self._thread: threading.Thread | None = None
        self._last_text = ""
        self._last_sig: tuple | None = None
        self._last_patch_ts = 0.0
        self._frozen = False  # 文本兜底路径编辑次数耗尽后定格（卡片路径无此问题）
        self._use_card = False
        self._frame_lock = threading.Lock()  # 帧写串行化：定格帧与刷新线程不打架
        self._final_written = False  # 定格帧已写：迟到帧不得覆盖

    # ---- 生命周期 ----

    def start(self) -> "StatusMessenger":
        try:
            if hasattr(self.sender, "status_card_create"):  # 首选：流式卡（无编辑上限）
                self._use_card = True
                self._msg_id = self.sender.status_card_create(self.chat_id, self._render())
                if not self._msg_id:  # 流式卡创建失败：降级文本消息（有编辑上限，耗尽定格）
                    logger.warning("流式状态卡创建失败，降级为文本状态条")
                    self._use_card = False
                    self._msg_id = self.sender.send_text_id(self.chat_id, self._render())
            else:  # 兜底：普通文本消息（有编辑上限，耗尽定格）
                self._use_card = False
                self._msg_id = self.sender.send_text_id(self.chat_id, self._render())
        except Exception:
            logger.exception("状态条首发失败（任务继续，仅无状态条）")
            return self
        if not self._msg_id:
            return self
        self._thread = threading.Thread(
            target=self._loop, name=f"status-{int(self._t0)}", daemon=True)
        self._thread.start()
        return self

    def stop(self, ok: bool, note: str = "") -> None:
        """定格最终帧并停线程。ok=True ✅ / False ❌。
        先停刷新线程再写定格帧（帧写锁挡住超时漏网的迟到帧），定格帧必是最后一帧。
        编辑次数已耗尽时最终帧可能写不进去（结果卡兜底，信息不丢）。"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        mark = "✅" if ok else "❌"
        tail = note or ("完成" if ok else "失败")
        final = (f"{mark} {_trunc(self.title, _STAGE_BUDGET)} · "
                 f"全程 {_fmt_duration(time.time() - self._t0)} · {tail}")
        with self._frame_lock:
            self._final_written = True
            if self._frozen:  # 窗口已关：尽力写最后一帧（写不进也不影响任务）
                try:
                    if self._use_card:
                        self.sender.status_card_update(self._msg_id, final)
                    else:
                        self.sender.update_text(self._msg_id, final)
                except Exception:
                    pass
            else:
                self._patch_locked(final)
        self._close_card_seq()

    def _close_card_seq(self) -> None:
        """句柄生命周期结束：通知 sender 清理按 card_id 累积的流式序号（防只增不清）。"""
        if not (self._use_card and self._msg_id):
            return
        close = getattr(self.sender, "status_card_close", None)
        if close is None:
            return
        try:
            close(self._msg_id)
        except Exception:
            logger.exception("状态卡序号清理失败")

    # ---- 运行中 ----

    def set_stage(self, stage: str) -> None:
        with self._lock:
            self._stage = stage

    def _render(self) -> str:
        """单行三段：⠙ 阶段 · 活动 · 计时——各段截断定长，气泡永不跳动。"""
        activities = self.registry.snapshot()
        with self._lock:
            stage = self._stage
        stage_short = _trunc(_EMOJI_PREFIX_RE.sub("", stage), _STAGE_BUDGET)
        elapsed = _fmt_duration(time.time() - self._t0)
        if activities:
            self._idle_since = None
            self._frame = (self._frame + 1) % len(SPINNER_FRAMES)
            icon = SPINNER_FRAMES[self._frame]
            first_label, first_dur = activities[0]
            act = _trunc(f"{_short_activity(first_label)} {_fmt_duration(first_dur)}",
                         _ACT_BUDGET)
            if len(activities) > 1:
                act = _trunc(f"{act}+{len(activities) - 1}", _ACT_BUDGET)
        else:
            if self._idle_since is None:
                self._idle_since = time.time()
            idle = time.time() - self._idle_since
            icon = "⏸"
            act = _trunc(f"💤静默 {_fmt_duration(idle)}", _ACT_BUDGET)
            if idle >= _IDLE_WARN_AFTER:
                act = _trunc("💤疑似卡死", _ACT_BUDGET)
        return f"{icon} {stage_short} · {act} · {elapsed}"

    def _signature(self, activities: list[tuple[str, float]]) -> tuple:
        """变化指纹：阶段文案 + 在途活动标签集 + 有无活动 + 静默卡死警告位。
        时长不计（静止不刷）；卡死警告属于实质状态，翻转时触发刷新。"""
        with self._lock:
            stage = self._stage
        if activities:
            self._idle_since = None
            idle_warn = False
        else:
            if self._idle_since is None:
                self._idle_since = time.time()
            idle_warn = (time.time() - self._idle_since) >= _IDLE_WARN_AFTER
        return (stage, tuple(label for label, _ in activities), bool(activities), idle_warn)

    def _should_patch(self, sig: tuple, now: float) -> bool:
        """纯实质变化驱动：阶段/活动集合/静默迁移变化才刷。
        计时数字静止态不单独刷新（用户定：状态没变就不用刷）；10 分钟窗口
        定格帧由 _tick 强制写出，不经过本判定。"""
        return sig != self._last_sig

    def _tick(self) -> bool:
        """一帧：窗口内按策略刷新；超过流式窗口写「定格说明」并冻结。
        返回 False 表示刷新循环应结束（任务本身继续，结果卡兜底）。
        流式窗口硬死是 Card Kit 平台的限制，只守卫卡片路径；文本兜底路径
        没有 10 分钟窗口（它的寿命由 230072 编辑上限决定），超窗照常刷新。"""
        if self._use_card and time.time() - self._t0 >= _LIVE_WINDOW and not self._frozen:
            self._patch(self._render_window_closed())
            self._frozen = True
            return False
        activities = self.registry.snapshot()
        self._maybe_stuck_alert(activities, time.time())
        sig = self._signature(activities)
        now = time.time()
        if self._should_patch(sig, now):
            self._patch(self._render())
            self._last_sig = sig
            self._last_patch_ts = now
        return True

    def _maybe_stuck_alert(self, activities: list[tuple[str, float]], now: float) -> None:
        """卡死看门狗：无在途活动持续 STUCK_ALERT_AFTER 秒 → alert_fn 报一次警。

        一次静默只报一次（不刷屏）；活动恢复即复位，再卡住可再报。alert_fn
        异常只记日志且不置标志（下一轮重试）；未注入 alert_fn 时只跟踪静默
        起点，不做事。LLM/MinerU 等长生成经 activity.REGISTRY 可见，不算卡死。
        """
        if activities:
            self._stuck_since = None
            self._stuck_alerted = False
            return
        if self._stuck_since is None:
            self._stuck_since = now
        if self.alert_fn is None or self._stuck_alerted:
            return
        if now - self._stuck_since < STUCK_ALERT_AFTER:
            return
        try:
            self.alert_fn(
                f"⏰ {self.title} 已 {STUCK_ALERT_AFTER // 60} 分钟无进展，"
                "疑似卡死；我继续盯着，有变化会发消息"
            )
        except Exception:
            logger.exception("卡死报警回调异常（下轮重试）")
            return
        self._stuck_alerted = True

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            if not self._tick():
                break

    def _render_window_closed(self) -> str:
        """流式窗口关闭前的最后一帧：明确告诉用户是定格，不是死了。"""
        with self._lock:
            stage = self._stage
        stage_short = _trunc(_EMOJI_PREFIX_RE.sub("", stage), _STAGE_BUDGET)
        return (f"⏳ {stage_short} · 已运行 {_fmt_duration(time.time() - self._t0)} · "
                "状态定格（飞书窗口限制，完成会发结果卡）")

    def _patch(self, text: str) -> None:
        """帧写入口：帧写锁串行化；定格帧写完后，迟到帧直接丢弃（不得覆盖 ✅/❌）。"""
        with self._frame_lock:
            if self._final_written:
                return
            self._patch_locked(text)

    def _patch_locked(self, text: str) -> None:
        """帧写本体（调用方须持 _frame_lock；stop 的定格帧也走这里）。"""
        if not self._msg_id or text == self._last_text or self._frozen:
            return
        try:
            if self._use_card:  # Card Kit 流式：无编辑上限，放心刷
                code = self.sender.status_card_update(self._msg_id, text)
            else:
                code = self.sender.update_text(self._msg_id, text)
        except Exception:
            logger.exception("状态条更新失败（下一帧继续）")
            return
        if not code:
            self._last_text = text
        elif code == "230072" and not self._use_card:
            # 文本路径兜底才有编辑上限：定格，绝不发新消息（用户裁决）
            self._frozen = True
            logger.warning("状态条消息编辑次数耗尽，定格不再更新（任务继续）：%s", self._msg_id)
        # 其他错误码（限流/网络）：下一帧再试
