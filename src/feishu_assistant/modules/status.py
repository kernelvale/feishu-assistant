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
    status_card_reactivate(handle) -> bool    可选：流式窗口 ~10 分钟（滑动）关闭前
                                              重激活同一张卡续命；序号继续单调递增
    status_card_close(handle)                 可选：句柄终结时清理按卡累积的流式序号

流式窗口真相：平台在「最后一次激活」~10 分钟后自动关闭（滑动窗口），不是硬死。
因此 _LIVE_WINDOW 是「重激活周期」而非定格点：到周期调 status_card_reactivate
续命并重置窗口计时，绝不写「窗口关闭定格」文案；更新返回「窗口关闭」类错误码
（非空且非 230072）时同样先重激活再原样重试一次。

TaskCard（任务进度卡）：TODO 清单逐项打勾 + 最近动态（关键事件流）
+ 当前活动 + 耗时统计的多行卡片，
与单行 StatusMessenger 并存（求职类等长任务沿用旧类）。复用同一套帧写锁、
签名变化检测（实质变化才 PATCH，计时数字不驱动）、卡死看门狗、重激活续命
与定格帧最后写纪律；仅走 Card Kit 流式卡，首发失败静默降级（任务继续）。

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
_LIVE_WINDOW = 480  # Card Kit 流式窗口「重激活周期」：最后一次激活 ~10 分钟（滑动）关闭，
                    # 8 分钟时主动重激活同一张卡续命——不是定格点（旧版 9 分钟定格是错误假设）
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
        self._window_start = self._t0  # 流式窗口最近一次激活时刻（start 发卡后重置）
        self._reactivate_fails = 0  # 连续重激活失败次数（≥3 退避为低频重试）
        self._next_reactivate_ts = 0.0  # 重激活退避到的时刻（0 = 不退避，下帧即重试）

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
        self._window_start = time.time()  # 卡创建即一次激活：窗口计时从这里起算
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
        计时数字静止态不单独刷新（用户定：状态没变就不用刷）。"""
        return sig != self._last_sig

    def _tick(self) -> bool:
        """一帧：先给流式卡续命（到重激活周期才调），再按策略刷新。
        返回 False 表示刷新循环应结束（任务本身继续，结果卡兜底）。
        流式窗口是平台的滑动窗口（最后一次激活 ~10 分钟自动关闭）：到周期主动
        重激活续命，绝不写「窗口关闭定格」文案——那是旧版的错误假设。
        文本兜底路径没有流式窗口（它的寿命由 230072 编辑上限决定），不参与续命。"""
        if self._use_card and not self._frozen:
            self._maybe_reactivate(time.time())
        activities = self.registry.snapshot()
        self._maybe_stuck_alert(activities, time.time())
        sig = self._signature(activities)
        now = time.time()
        if self._should_patch(sig, now):
            self._patch(self._render())
            self._last_sig = sig
            self._last_patch_ts = now
        return True

    def _maybe_reactivate(self, now: float) -> None:
        """流式窗口续命（仅卡片路径）：每 _LIVE_WINDOW 秒重激活一次同一张卡。
        失败只记日志、下帧重试；连续失败退避 30s 低频重试，绝不打扰用户、绝不定格。"""
        if now - self._window_start < _LIVE_WINDOW or now < self._next_reactivate_ts:
            return
        if self._reactivate_card():
            logger.info("状态卡流式窗口已重激活续命：%s", self._msg_id)
            return
        self._reactivate_fails += 1
        logger.warning("状态卡重激活失败（第 %d 次，下帧重试）：%s",
                       self._reactivate_fails, self._msg_id)
        if self._reactivate_fails >= 3:
            self._next_reactivate_ts = now + 30  # 连续失败退避：不每帧打飞书

    def _reactivate_card(self) -> bool:
        """调 sender.status_card_reactivate 重激活同一张卡；成功重置窗口计时与失败计数。
        sender 无该方法 / 传输异常 / API 失败 → False（下帧重试）。"""
        reactivate = getattr(self.sender, "status_card_reactivate", None)
        if reactivate is None or not self._msg_id:
            return False
        try:
            ok = bool(reactivate(self._msg_id))
        except Exception:
            logger.exception("状态卡重激活异常（下帧重试）")
            return False
        if ok:
            self._window_start = time.time()
            self._reactivate_fails = 0
            self._next_reactivate_ts = 0.0
        return ok

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

    def _patch(self, text: str) -> None:
        """帧写入口：帧写锁串行化；定格帧写完后，迟到帧直接丢弃（不得覆盖 ✅/❌）。"""
        with self._frame_lock:
            if self._final_written:
                return
            self._patch_locked(text)

    def _fatal_on_edit_cap(self) -> bool:
        """230072（编辑次数耗尽）是否致命定格：文本兜底路径才有编辑上限；
        卡片路径无此限制（TaskCard 除外：它只有卡通道，230072 视为致命）。"""
        return not self._use_card

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
        elif code == "230072":
            if self._fatal_on_edit_cap():
                # 编辑次数耗尽类致命错误：定格，绝不发新消息（用户裁决）
                self._frozen = True
                logger.warning("状态条消息编辑次数耗尽，定格不再更新（任务继续）：%s", self._msg_id)
            # 卡片路径无编辑上限：230072 按普通错误下帧重试（不触发重激活）
        elif self._use_card:
            # 疑似流式窗口已被平台关闭（非 230072 的错误码）：重激活后原样重试一次；
            # 仍失败则下一帧照常再来（_last_text 未变，文本会重发）
            if self._reactivate_card():
                try:
                    code = self.sender.status_card_update(self._msg_id, text)
                except Exception:
                    logger.exception("状态卡重激活后重试更新失败（下一帧继续）")
                    return
                if not code:
                    self._last_text = text
        # 其他错误码（限流/网络）：下一帧再试


# ---- 任务进度卡（TaskCard）：TODO 清单逐项打勾的单卡片实时进度 ----

_DETAIL_BUDGET = 40  # 阶段 detail 截断：卡片行宽恒定，聊天窗不跳动
_EVENT_BUDGET = 50  # 事件文本截断：一行一条，卡片行宽恒定
_MAX_EVENTS = 3  # 「最近动态」只留最近 3 条（滚动）
# 关键事件文本标记（✓放行/⚠降级/✎修复/⤴回流）：apply_progress 先于阶段规则识别，
# 只进「最近动态」区，不落 TODO detail（落了就与事件区重复）
_EVENT_PREFIXES = ("✓", "⚠", "✎", "⤴")
# 周期性进度文本：已在 TODO detail 实时刷新，进事件区就是重复
_PERIODIC_NEEDLES = ("正文生成", "精讲 层", "图表讲解")


def _fmt_tokens(value: int) -> str:
    """token 数人类可读：>=1000 转 X.Xk（45200 → 45.2k），不足千原样。"""
    return f"{value / 1000:.1f}k" if value >= 1000 else str(value)

# 精读管线进度文本 → TODO 阶段映射（首个命中生效；索引对应 paper 创建卡时的 stages）。
# 顺序即优先级：完成态消息（更长更具体）先于进行态（「教学计划完成」先于「教学计划」）；
# 「写入」先于「终审」（「已通过终审并写入」是发布归档完成的信号）。
_PAPER_PROGRESS_RULES: tuple[tuple[str, int, str], ...] = (
    ("解析完成", 0, "done"),        # ✅ 解析完成（N 页，M 张图表…）
    ("开始解析 PDF", 0, "current"),
    # V9：来源导航 → 轻量 Brief → 连续正文 → 两项专项审查 → 有限修订/定点验证。
    ("来源导航", 1, "current"),
    ("轻量叙事简报", 2, "current"),
    ("单一作者正在写", 3, "current"),
    ("事实审查与真实读者审查", 4, "current"),
    ("有限修订", 5, "current"),
    ("验证实际改动", 5, "current"),
    ("证据索引", 1, "current"),
    ("教学计划完成", 2, "done"),    # ✅ 教学计划完成（识别出 N 个技术层）
    ("教学计划", 2, "current"),
    ("正文生成", 3, "current"),     # 正文生成 x/y
    ("精讲完成", 3, "done"),        # ✅ 精讲完成（Ns，N 层已通过验收）
    ("精讲 层", 3, "current"),      # 精讲 层i/N
    ("图表讲解", 4, "current"),     # 图表讲解 x/y
    ("写入", 6, "done"),            # 精读文档已通过终审并写入
    ("文档已创建", 6, "done"),
    ("终审", 5, "current"),
    ("返工", 5, "current"),
)

_STAGE_MARKS = {"done": "✅", "current": "🔄", "pending": "⏳"}

# 进行中痕迹：阶段打勾后 detail 里出现这些词属于自相矛盾（「✅ …生成中……」），一律清除
_INPROGRESS_MARKERS = ("生成中", "进行中", "编译中", "索引中", "讲解中", "返工")
_INCOMPLETE_FRACTION_RE = re.compile(r"\d+/\d+")


def _done_detail(detail: str) -> str:
    """阶段完成时的 detail 清理：进行中措辞/未完成分数（x/y 且 x<y）清空，结果类保留。"""
    text = detail.strip().rstrip("…").strip()
    if not text:
        return ""
    if any(marker in text for marker in _INPROGRESS_MARKERS):
        return ""
    for frac in _INCOMPLETE_FRACTION_RE.findall(text):
        done, _, total = frac.partition("/")
        if total.isdigit() and done.isdigit() and int(done) < int(total):
            return ""
    return text


def _completion_detail(text: str) -> str:
    """完成消息的终态文案：优先取首个括号内容（「识别出 N 个技术层」），否则清理全文。"""
    match = re.search(r"（([^（）]+)）", text)
    if match:
        return match.group(1).strip()
    return _done_detail(_EMOJI_PREFIX_RE.sub("", text).strip())


class TaskCard(StatusMessenger):
    """任务进度卡：TODO 清单逐项打勾 + 当前活动 + 耗时统计，单卡片实时更新。

    与 StatusMessenger 并存（求职类等长任务沿用旧类）。复用父类的帧写锁、
    签名变化检测（实质变化才 PATCH，计时数字不驱动）、卡死看门狗、流式窗口
    重激活续命与定格帧最后写纪律；差异只在渲染（多行 TODO 卡）与通道
    （仅 Card Kit 流式卡：首发失败静默降级，任务继续，仅无进度卡）。

    阶段状态机：pending → current（带 detail）→ done；current 唯一，单调前进
    （返工/回流产生的早阶段文本只更新当前 detail，TODO 不倒退）。

    「最近动态」区：TODO 与页脚之间的三行关键事件流（add_event 写入，
    周期性进度文本与 TODO detail 已有的内容不进，避免重复展示）。
    """

    def __init__(
        self,
        sender: Any,
        chat_id: str,
        title: str,
        stages: list[str],
        *,
        registry: ActivityRegistry = REGISTRY,
        alert_fn: Any = None,
        interval: float = 3.0,
        calls_fn: Any = None,
        cached_fn: Any = None,
    ) -> None:
        super().__init__(sender, chat_id, title,
                         interval=interval, registry=registry, alert_fn=alert_fn)
        self._stages = [{"name": str(name), "state": "pending", "detail": ""}
                        for name in stages]
        # 可选：返回累计调用次数的 callable（统计行追加「· 调用 N 次」）
        self.calls_fn = calls_fn
        # 可选：返回累计缓存 token 数的 callable（统计行追加「· 缓存 X.Xk」，0 省略）
        self.cached_fn = cached_fn
        self._events: list[dict[str, Any]] = []  # 最近动态：{text, count, ts}

    # ---- 生命周期 ----

    def start(self) -> "TaskCard":
        """仅走 Card Kit 流式卡（整卡布局：header + TODO 元素 + 页脚元素）；
        首发失败/无卡通道静默降级（任务继续，仅无进度卡）。"""
        create = getattr(self.sender, "rich_card_create", None)
        if create is None:
            logger.warning("sender 无富文本卡通道，任务进度卡静默降级：%s", self.title)
            return self
        try:
            card = self._render()
            self._msg_id = create(self.chat_id, card)
        except Exception:
            logger.exception("任务进度卡首发失败（任务继续，仅无进度卡）")
            return self
        if not self._msg_id:
            logger.warning("任务进度卡创建失败，静默降级（任务继续）：%s", self.title)
            return self
        self._use_card = True
        todo = card["body"]["elements"][0]["content"]
        events = card["body"]["elements"][2]["content"]
        footer = card["body"]["elements"][4]["content"]
        self._last_todo, self._last_events, self._last_footer = todo, events, footer
        self._last_sig = self._signature(self.registry.snapshot())
        self._window_start = time.time()  # 卡创建即一次激活：窗口计时从这里起算
        self._thread = threading.Thread(
            target=self._loop, name=f"taskcard-{int(self._t0)}", daemon=True)
        self._thread.start()
        return self

    def finish(self, ok: bool, note: str = "") -> None:
        """定格最终帧并停线程：TODO 终态 + ✅/❌ 全程时长 + note。
        成功时剩余阶段全部打勾；失败保留终态（死在哪个阶段一眼可见）。
        事件区原样保留（终态可见最后三条动态）。
        先停线程再写定格帧（帧写锁挡住迟到帧），定格帧必是最后一帧。"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if ok:
            with self._lock:
                for stage in self._stages:
                    stage["state"] = "done"
        todo, events, footer = self._render_final(ok, note)
        with self._frame_lock:
            self._final_written = True
            self._patch_elements(todo, events, footer, best_effort=self._frozen)
        self._close_card_seq()

    # ---- 帧写（元素级：TODO/事件区/页脚分开 PATCH）----

    def _patch_elements(self, todo: str, events: str, footer: str,
                        *, best_effort: bool = False) -> None:
        """三元素按需更新；卡片通道错误码触发重激活重试一次，230072 定格不再刷。
        best_effort=True（定格后最后一帧）：尽力写，写不进也不影响任务。"""
        if not self._msg_id:
            return
        for element_id, content, attr in (
                (self._TODO_ELEMENT, todo, "_last_todo"),
                (self._EVENT_ELEMENT, events, "_last_events"),
                (self._FOOTER_ELEMENT, footer, "_last_footer")):
            if getattr(self, attr, None) == content:
                continue
            handle = f"{self._msg_id}|{element_id}"
            try:
                code = self.sender.status_card_update(handle, content)
            except Exception:
                logger.exception("任务卡元素更新失败（下帧继续）：%s", element_id)
                if best_effort:
                    continue
                return
            if not code:
                setattr(self, attr, content)
                continue
            if code == "230072":
                self._frozen = True
                logger.warning("任务卡编辑次数耗尽，定格不再更新（任务继续）")
                if not best_effort:
                    return
                continue
            # 疑似流式窗口超时关闭：重激活后原样重试一次
            self._reactivate_card()
            try:
                code = self.sender.status_card_update(handle, content)
            except Exception:
                code = "exception"
            if not code:
                setattr(self, attr, content)
            elif not best_effort:
                return  # 下帧再试

    def _patch_locked(self, text: Any) -> None:
        """父类帧写入口覆盖：忽略 text，内容由状态现算（TODO/事件区/页脚分开 PATCH）。"""
        if self._frozen:
            return
        activities = self.registry.snapshot()
        with self._lock:
            stages = [(s["name"], s["state"], s["detail"]) for s in self._stages]
            events = [dict(e) for e in self._events]
        self._patch_elements(self._todo_content(stages), self._events_content(events),
                             self._footer_content(activities))

    def stop(self, ok: bool, note: str = "") -> None:
        """与 StatusMessenger 同协议的定格入口（paper._run_guarded 调 stop）。"""
        self.finish(ok, note)

    # ---- 阶段状态机 ----

    def _resolve_stage(self, name_or_index: Any) -> int | None:
        """阶段定位：int 直接校验；str 按阶段名精确匹配；未知文本返回 None。"""
        if isinstance(name_or_index, int):
            return name_or_index if 0 <= name_or_index < len(self._stages) else None
        for i, stage in enumerate(self._stages):
            if stage["name"] == name_or_index:
                return i
        return None

    def _current_index_locked(self) -> int | None:
        """当前 current 阶段序号（调用方须持 _lock）；无 → None。"""
        for i, stage in enumerate(self._stages):
            if stage["state"] == "current":
                return i
        return None

    @staticmethod
    def _clean_detail(text: Any) -> str:
        """detail 归一：剥 emoji 前缀 + 40 字截断。"""
        return _trunc(_EMOJI_PREFIX_RE.sub("", str(text)).strip(), _DETAIL_BUDGET)

    def set_stage(self, name_or_index: Any, detail: str = "") -> None:
        """推进阶段：目标置 current（原 current 置 done），detail 随行。
        未知文本（如「网络中断，N 分钟后自动重试」）不推进，只更新当前阶段 detail；
        倒退文本（返工回流产生的早阶段进度）同样只更新当前 detail——TODO 单调前进。"""
        idx = self._resolve_stage(name_or_index)
        detail = self._clean_detail(detail)
        with self._lock:
            current = self._current_index_locked()
            if idx is None:  # 未命名文本：当前阶段 detail（瞬时重试/降级提示等）
                if current is not None:
                    self._stages[current]["detail"] = self._clean_detail(name_or_index)
                return
            if current is not None and idx < current:  # 倒退：只更新当前 detail
                if detail:
                    self._stages[current]["detail"] = detail
                return
            if current == idx:
                if detail:
                    self._stages[idx]["detail"] = detail
                return
            if current is not None:
                self._stages[current]["state"] = "done"
                self._stages[current]["detail"] = _done_detail(
                    self._stages[current]["detail"])  # 打勾后清掉进行中痕迹
            self._stages[idx]["state"] = "current"
            self._stages[idx]["detail"] = detail

    def mark_done(self, name_or_index: Any, detail: str = "") -> None:
        """直接打勾（不经过 current）：用于「解析完成」「写入」这类完成即过的事件。
        当前 current 阶段序号 ≤ 目标时一并打勾（线性管线，跳过不留尾巴）。
        终态 detail：完成消息取括号内容（「识别出 N 个技术层」），被打勾阶段清进行中痕迹。"""
        idx = self._resolve_stage(name_or_index)
        if idx is None:
            return
        with self._lock:
            current = self._current_index_locked()
            if current is not None and idx < current:  # 倒退：忽略
                return
            if current is not None:
                self._stages[current]["state"] = "done"
                self._stages[current]["detail"] = _done_detail(
                    self._stages[current]["detail"])
            self._stages[idx]["state"] = "done"
            self._stages[idx]["detail"] = self._clean_detail(
                _completion_detail(detail) if detail else "")

    def apply_progress(self, text: str) -> None:
        """精读管线进度文本 → TODO 阶段映射（paper._progress 注入）。
        事件标记（✓/⚠/✎/⤴）开头的文本是「关键事件」：只进最近动态区，
        不参与阶段映射、不落 TODO detail（落了就与事件区重复）。
        未命中规则的其余文本作为当前阶段 detail 展示（降级/重试提示等），不推进阶段。"""
        if text.lstrip().startswith(_EVENT_PREFIXES):
            self.add_event(text)
            return
        for needle, idx, action in _PAPER_PROGRESS_RULES:
            if needle in text:
                if action == "done":
                    self.mark_done(idx, text)
                else:
                    self.set_stage(idx, text)
                return
        self.set_stage(text)  # 未命中：未知阶段名 → 当前阶段 detail

    # ---- 最近动态（关键事件流：TODO 与页脚之间，最多 3 条滚动）----

    def add_event(self, text: str) -> None:
        """关键事件入「最近动态」区：剥 emoji 前缀 + 50 字截断。
        与上一条相同合并为 ×N 后缀；与 TODO 当前 detail 相同跳过（不重复展示）；
        周期性进度文本（正文生成 x/y 等）不进——它们已在 TODO detail 实时刷新。"""
        clean = _trunc(_EMOJI_PREFIX_RE.sub("", str(text)).strip(), _EVENT_BUDGET)
        if not clean or any(needle in clean for needle in _PERIODIC_NEEDLES):
            return
        with self._lock:
            current = self._current_index_locked()
            if current is not None and clean == self._stages[current]["detail"]:
                return  # TODO detail 已在展示同一内容：跳过
            now = time.strftime("%H:%M")
            if self._events and self._events[-1]["text"] == clean:
                self._events[-1]["count"] += 1  # 连续重复合并 ×N
                self._events[-1]["ts"] = now
            else:
                self._events.append({"text": clean, "count": 1, "ts": now})
                del self._events[:-_MAX_EVENTS]  # 上限 3 条滚动

    def _events_content(self, events: list[dict[str, Any]]) -> str:
        """最近动态区：📌 标题行 + 每事件一行「· HH:MM 文本」（×N 合并后缀）。"""
        lines = ["📌 **最近动态**"]
        for event in events:
            suffix = f" ×{event['count']}" if event["count"] > 1 else ""
            lines.append(f"· {event['ts']} {event['text']}{suffix}")
        return "\n".join(lines)

    # ---- 渲染（卡片布局：header 模板色 + TODO + 最近动态 + 灰色页脚元素）----

    _TODO_ELEMENT = "todo_body"
    _EVENT_ELEMENT = "recent_events"
    _FOOTER_ELEMENT = "footer_line"
    _HEADER_TEMPLATE = "blue"

    def _todo_content(self, stages: list[tuple[str, str, str]]) -> str:
        """TODO 主元素：📋 计数行（加粗）+ 图标体系（🌟完成 / 🚀进行中加粗 / 🕐灰色待办）。"""
        done = sum(1 for _, state, _ in stages if state == "done")
        lines = [f"📋 **TODO ({done}/{len(stages)})**", ""]
        for name, state, detail in stages:
            if state == "done":
                line = f"🌟 {name}" + (f" · {detail}" if detail else "")
            elif state == "current":
                line = f"🚀 **{name}**" + (f" · {detail}" if detail else "")
            else:
                line = f"🕐 <font color='grey'>{name}</font>"
            lines.append(line)
        return "\n".join(lines)

    def _footer_content(self, activities: list[tuple[str, float]]) -> str:
        """页脚元素（灰色小字风）：当前活动 + 耗时统计。"""
        return (f"<font color='grey'>{self._activity_line(activities)}　"
                f"{self._stats_line()}</font>")

    def _card_json(self, todo: str, events: str, footer: str, *, final_mark: str = "") -> dict:
        """整卡结构：header（模板色标题栏）+ TODO 元素 + 最近动态元素 + 页脚元素。"""
        title = _trunc(f"{final_mark}{self.title}" if final_mark else self.title,
                       _DETAIL_BUDGET)
        return {
            "schema": "2.0",
            "config": {"streaming_mode": True, "update_multi": True},
            "header": {
                "template": self._HEADER_TEMPLATE,
                "title": {"tag": "plain_text", "content": title},
            },
            "body": {"elements": [
                {"tag": "markdown", "element_id": self._TODO_ELEMENT, "content": todo},
                {"tag": "hr"},
                {"tag": "markdown", "element_id": self._EVENT_ELEMENT, "content": events},
                {"tag": "hr"},
                {"tag": "markdown", "element_id": self._FOOTER_ELEMENT, "content": footer},
            ]},
        }

    def _render(self) -> dict:
        """运行帧（整卡 JSON）：TODO 区 + 最近动态 + 页脚活动/统计。"""
        activities = self.registry.snapshot()
        with self._lock:
            stages = [(s["name"], s["state"], s["detail"]) for s in self._stages]
            events = [dict(e) for e in self._events]
        return self._card_json(self._todo_content(stages), self._events_content(events),
                               self._footer_content(activities))

    def _render_final(self, ok: bool, note: str) -> tuple[str, str, str]:
        """定格帧内容（todo, events, footer）：TODO 终态 + ✅/❌ 全程时长 + note。
        事件区原样带出（内容不变不触发 PATCH；此前漏写的帧借此尽力补齐）。"""
        mark = "✅" if ok else "❌"
        tail = note or ("完成" if ok else "失败")
        with self._lock:
            stages = [(s["name"], s["state"], s["detail"]) for s in self._stages]
            events = [dict(e) for e in self._events]
        footer = (f"<font color='grey'>{mark} {self._stats_line(prefix='全程')}"
                  f" · {tail}</font>")
        return self._todo_content(stages), self._events_content(events), footer

    def _tick_icon(self, activities: list[tuple[str, float]]) -> str:
        """标题行图标：有在途活动转圈，无活动停住（⏸）。"""
        if activities:
            self._idle_since = None
            self._frame = (self._frame + 1) % len(SPINNER_FRAMES)
            return SPINNER_FRAMES[self._frame]
        if self._idle_since is None:
            self._idle_since = time.time()
        return "⏸"

    def _activity_line(self, activities: list[tuple[str, float]]) -> str:
        """当前活动行：⚙️ kimi-k3 · 45s；无活动 💤 静默时长（≥60s 疑似卡死）。"""
        if activities:
            first_label, first_dur = activities[0]
            line = f"⚙️ {_short_activity(first_label)} · {int(first_dur)}s"
            if len(activities) > 1:
                line += f" +{len(activities) - 1}"
            return line
        idle = time.time() - (self._idle_since or time.time())
        if idle >= _IDLE_WARN_AFTER:
            return f"💤 疑似卡死 · 静默 {_fmt_duration(idle)}"
        return f"💤 静默 {_fmt_duration(idle)}"

    def _stats_line(self, prefix: str = "已用") -> str:
        """统计行：⏱ 已用 12:30（注入后追加「· 调用 N 次 · 缓存 X.Xk」，缓存为 0 省略）。"""
        line = f"⏱ {prefix} {_fmt_duration(time.time() - self._t0)}"
        calls = self._calls_count()
        if calls is not None:
            line += f" · 调用 {calls} 次"
        cached = self._cached_tokens()
        if cached:
            line += f" · 缓存 {_fmt_tokens(cached)}"
        return line

    def _calls_count(self) -> int | None:
        """calls_fn 安全求值：未注入/异常 → None（统计行省略调用数，不炸状态卡）。"""
        if self.calls_fn is None:
            return None
        try:
            return int(self.calls_fn())
        except Exception:
            return None

    def _cached_tokens(self) -> int | None:
        """cached_fn 安全求值：未注入/异常 → None（统计行省略缓存项，不炸状态卡）。"""
        if self.cached_fn is None:
            return None
        try:
            return int(self.cached_fn())
        except Exception:
            return None

    def _signature(self, activities: list[tuple[str, float]]) -> tuple:
        """变化指纹：TODO 状态+detail + 事件区 + 活动集合 + 有无活动 + 卡死警告位
        + 调用/缓存计数。计时数字与 spinner 帧不计（静止不刷）；卡死警告翻转触发刷新。"""
        with self._lock:
            stage_sig = tuple((s["state"], s["detail"]) for s in self._stages)
            event_sig = tuple((e["text"], e["count"]) for e in self._events)
        if activities:
            self._idle_since = None
            idle_warn = False
        else:
            if self._idle_since is None:
                self._idle_since = time.time()
            idle_warn = (time.time() - self._idle_since) >= _IDLE_WARN_AFTER
        return (stage_sig, event_sig, tuple(label for label, _ in activities),
                bool(activities), idle_warn, self._calls_count(), self._cached_tokens())

    def _fatal_on_edit_cap(self) -> bool:
        """TaskCard 只有卡通道：230072（编辑次数耗尽类）视为致命，定格不再更新（任务继续）。"""
        return True
