"""状态条（StatusMessenger）与活动登记（activity.track）测试。"""

from __future__ import annotations

import time

from feishu_assistant.activity import ActivityRegistry
from feishu_assistant.modules.status import StatusMessenger, _IDLE_WARN_AFTER


class FakeSender:
    def __init__(self):
        self.sent: list[str] = []
        self.patches: list[str] = []
        self.next_id = "om_status_1"
        self.fail_code = ""  # 设成 "230072" 模拟编辑次数耗尽

    def send_text_id(self, chat_id, text):
        self.sent.append(text)
        return self.next_id

    def update_text(self, message_id, text):
        if self.fail_code:
            return self.fail_code
        assert message_id == self.next_id
        self.patches.append(text)
        return ""


def test_registry_track_and_snapshot():
    reg = ActivityRegistry()
    assert reg.snapshot() == []
    with reg.track("LLM·k3"):
        snap = reg.snapshot()
        assert len(snap) == 1 and snap[0][0] == "LLM·k3" and snap[0][1] >= 0
        with reg.track("MinerU 解析"):
            assert len(reg.snapshot()) == 2
    assert reg.snapshot() == []


def test_render_active_single_line_compact():
    reg = ActivityRegistry()
    sender = FakeSender()
    sm = StatusMessenger(sender, "oc_x", "精读 arXiv:1", registry=reg)
    with reg.track("LLM·free-tokens/kimi-k3"):
        text = sm._render()
        assert "\n" not in text  # 单行
        assert text.split(" ")[0] in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        assert "启动中" in text
        assert "kimi-k3" in text and "free-tokens" not in text  # 活动缩成短模型名


def test_render_idle_freezes_spinner_and_warns():
    reg = ActivityRegistry()
    sender = FakeSender()
    sm = StatusMessenger(sender, "oc_x", "精读 arXiv:1", registry=reg)
    t1 = sm._render()
    t2 = sm._render()
    assert t1.startswith("⏸") and t2.startswith("⏸")  # 无活动：帧停住
    assert "💤静默" in t1
    # 静默超阈值后提示疑似卡死
    sm._idle_since = time.time() - 120
    assert "疑似卡死" in sm._render()


def test_render_truncates_long_segments():
    reg = ActivityRegistry()
    sm = StatusMessenger(FakeSender(), "oc_x", "精读 arXiv:1", registry=reg)
    sm.set_stage("✅ 下载完成（4.2 MB），开始解析 PDF（约 1~2 分钟）……")
    with reg.track("MinerU 解析 PDF"):
        text = sm._render()
        assert len(text) <= 45  # 三段截断后总长恒定
        assert "✅" not in text  # 阶段前缀 emoji 被剥掉
        assert "MinerU" in text


def test_set_stage_and_stop_final_frame():
    sender = FakeSender()
    sm = StatusMessenger(sender, "oc_x", "精读 arXiv:1", interval=999)
    sm.set_stage("分层精讲（heavy 流式）")
    assert "分层精讲" in sm._render()
    sm.start()
    assert len(sender.sent) == 1
    sm.stop(ok=True, note="精读完成，文档与详情见下方结果卡")
    assert sender.patches[-1].startswith("✅")
    assert "全程" in sender.patches[-1] and "结果卡" in sender.patches[-1]


def test_stop_failure_marks_cross():
    sender = FakeSender()
    sm = StatusMessenger(sender, "oc_x", "精读 arXiv:1", interval=999).start()
    sm.stop(ok=False, note="失败：ReadTimeout")
    assert sender.patches[-1].startswith("❌") and "ReadTimeout" in sender.patches[-1]


def test_patch_skips_identical_text():
    sender = FakeSender()
    sm = StatusMessenger(sender, "oc_x", "精读 arXiv:1", interval=999).start()
    sm._patch("同一段文字")
    sm._patch("同一段文字")  # 重复帧不发 PATCH
    assert sender.patches.count("同一段文字") == 1


def test_start_without_message_id_degrades_silently():
    class BadSender(FakeSender):
        def send_text_id(self, chat_id, text):
            return ""

    sm = StatusMessenger(BadSender(), "oc_x", "精读 arXiv:1", interval=0.01).start()
    sm.set_stage("骨架生成中")
    sm.stop(ok=True)  # 不炸线程、不抛异常


def test_edit_cap_freezes_without_new_message():
    """230072（编辑次数耗尽）→ 定格：绝不发新消息，后续 PATCH 静默跳过。"""
    sender = FakeSender()
    sm = StatusMessenger(sender, "oc_x", "精读 arXiv:1", interval=999).start()
    assert sm._msg_id == "om_status_1"
    sender.fail_code = "230072"
    sm._patch("新一帧")
    assert sm._frozen and len(sender.sent) == 1  # 没有新消息
    sm._patch("又一帧")  # 定格后不再 PATCH
    assert sender.patches == []
    # 最终 stop 仍会尝试写最后一帧（失败也不影响任务）
    sm.stop(ok=True)


def test_other_error_keeps_updating():
    sender = FakeSender()
    sm = StatusMessenger(sender, "oc_x", "精读 arXiv:1", interval=999).start()
    sender.fail_code = "230099"  # 非编辑上限错误：不定格，下一帧再试
    sm._patch("某帧")
    assert not sm._frozen and sm._msg_id == "om_status_1"


class CardSender(FakeSender):
    """带 Card Kit 流式通道的 sender。"""

    def __init__(self):
        super().__init__()
        self.card_updates: list[str] = []

    def status_card_create(self, chat_id, text):
        self.sent.append(text)
        return "card_1|status_line"

    def status_card_update(self, handle, text):
        assert handle == "card_1|status_line"
        self.card_updates.append(text)
        return ""


def test_card_channel_preferred_when_available():
    """sender 有流式卡通道 → 走卡片不走文本消息，PATCH 全流式。"""
    sender = CardSender()
    sm = StatusMessenger(sender, "oc_x", "精读 arXiv:1", interval=999).start()
    assert sm._use_card and sm._msg_id == "card_1|status_line"
    sm._patch("帧一")
    sm._patch("帧二")
    assert sender.card_updates == ["帧一", "帧二"]
    assert sender.patches == []  # 文本 PATCH 一次没走


def test_card_channel_never_freezes_on_edit_cap():
    """卡片路径没有编辑上限：即使上游误回 230072 也不定格（按普通错误下帧重试）。"""
    sender = CardSender()
    sm = StatusMessenger(sender, "oc_x", "精读 arXiv:1", interval=999).start()
    sender.status_card_update = lambda h, t: "230072"  # type: ignore[assignment]
    sm._patch("某帧")
    assert not sm._frozen


def test_window_close_writes_final_note_and_freezes():
    """超过流式窗口（默认 9 分钟）：写「状态定格」说明帧后冻结，不再刷新。"""
    import feishu_assistant.modules.status as status_mod
    sender = CardSender()
    sm = StatusMessenger(sender, "oc_x", "精读 arXiv:1", interval=999).start()
    sm._t0 = time.time() - status_mod._LIVE_WINDOW - 1  # 假装已超窗
    alive = sm._tick()
    assert alive is False and sm._frozen
    assert sender.card_updates[-1].startswith("⏳")
    assert "状态定格" in sender.card_updates[-1] and "结果卡" in sender.card_updates[-1]
    # 冻结后 stop 仍尽力写 ✅ 终帧
    sm.stop(ok=True, note="完成")
    assert sender.card_updates[-1].startswith("✅")


def test_should_patch_pure_substance_driven():
    """纯实质变化驱动：签名不变（含卡死警告位）→ 不刷；阶段/活动/卡死警告变化 → 即刷。"""
    sm = StatusMessenger(FakeSender(), "oc_x", "精读 arXiv:1")
    sm._last_sig = ("阶段A", ("LLM·k3",), True, False)
    sm._last_patch_ts = 1000.0
    # 签名完全不变 → 不刷（无论过了多久：计时静止不单独刷新）
    assert not sm._should_patch(("阶段A", ("LLM·k3",), True, False), 1061.0)
    # 阶段变了 → 立即刷
    assert sm._should_patch(("阶段B", ("LLM·k3",), True, False), 1001.0)
    # 活动集合变了（新工具上线）→ 立即刷
    assert sm._should_patch(("阶段A", ("LLM·k3", "MinerU 解析"), True, False), 1001.0)
    # 活动→静默迁移 → 立即刷
    assert sm._should_patch(("阶段A", (), False, False), 1001.0)
    # 卡死警告位翻转 → 立即刷
    assert sm._should_patch(("阶段A", (), False, True), 1001.0)


def test_signature_marks_idle_stuck_warning():
    """静默超 _IDLE_WARN_AFTER：卡死警告位进入签名（实质变化会触发刷新）。"""
    sm = StatusMessenger(FakeSender(), "oc_x", "精读 arXiv:1")
    first = sm._signature([])
    assert first[-1] is False
    sm._idle_since = time.time() - _IDLE_WARN_AFTER - 1
    assert sm._signature([])[-1] is True
    # 活动恢复：警告位归零
    assert sm._signature([("LLM·k3", 1.0)])[-1] is False


def test_card_create_empty_falls_back_to_text():
    """流式卡创建返回空 → 降级文本消息路径（_use_card=False，走 send_text_id/update_text）。"""

    class EmptyCardSender(CardSender):
        def status_card_create(self, chat_id, text):
            return ""  # 创建失败

    sender = EmptyCardSender()
    sm = StatusMessenger(sender, "oc_x", "精读 arXiv:1", interval=999).start()
    assert not sm._use_card and sm._msg_id == "om_status_1"
    sm._patch("帧一")
    assert sender.patches == ["帧一"] and sender.card_updates == []  # 全走文本通道


def test_text_path_ignores_live_window():
    """文本兜底路径没有 10 分钟流式窗口（那是 Card Kit 平台限制）：
    超窗仍照常刷新，不写「状态定格」说明。"""
    import feishu_assistant.modules.status as status_mod
    sender = FakeSender()
    sm = StatusMessenger(sender, "oc_x", "精读 arXiv:1", interval=999).start()
    sm._t0 = time.time() - status_mod._LIVE_WINDOW - 1  # 假装已超窗
    alive = sm._tick()
    assert alive is True and not sm._frozen
    assert not any("状态定格" in p for p in sender.patches)


def test_stop_final_frame_is_last_and_late_frames_dropped():
    """定格帧与刷新线程竞态：stop 后迟到的 _patch 直接丢弃，终帧必是最后一帧。"""
    sender = FakeSender()
    sm = StatusMessenger(sender, "oc_x", "精读 arXiv:1", interval=0.01).start()
    time.sleep(0.05)  # 让刷新线程跑几帧
    sm.stop(ok=True, note="完成")
    assert sender.patches[-1].startswith("✅")
    sm._patch("迟到帧")  # 定格帧已写：迟到帧不得覆盖
    assert sender.patches[-1].startswith("✅")


def test_stop_notifies_sender_to_close_card_seq():
    """stop 时通知 sender 清理流式序号（status_card_close）；无该方法的 sender 静默跳过。"""

    class CloseSender(CardSender):
        def __init__(self):
            super().__init__()
            self.closed: list[str] = []

        def status_card_close(self, handle):
            self.closed.append(handle)

    sender = CloseSender()
    sm = StatusMessenger(sender, "oc_x", "精读 arXiv:1", interval=999).start()
    sm.stop(ok=True)
    assert sender.closed == ["card_1|status_line"]
    # 无 status_card_close 的 sender：不炸
    sm2 = StatusMessenger(CardSender(), "oc_x", "精读 arXiv:1", interval=999).start()
    sm2.stop(ok=True)


# ---- 卡死看门狗：静默超 STUCK_ALERT_AFTER 经 alert_fn 报一次警 ----

class _AlertSink:
    """alert_fn 捕获器（可预设抛异常模拟外发箱故障）。"""

    def __init__(self, raises: bool = False):
        self.texts: list[str] = []
        self.raises = raises

    def __call__(self, text: str) -> None:
        if self.raises:
            raise RuntimeError("外发箱故障")
        self.texts.append(text)


def _make_stuck_sm(alert_fn=None, registry=None):
    """构造已静默 300+s 的 StatusMessenger（可控 registry/时钟，不走 start）。"""
    import feishu_assistant.modules.status as status_mod
    sm = StatusMessenger(
        FakeSender(), "oc_x", "精读 arXiv:1",
        interval=999, registry=registry or ActivityRegistry(), alert_fn=alert_fn,
    )
    sm._stuck_since = time.time() - status_mod.STUCK_ALERT_AFTER - 1  # 假装已静默超阈
    return sm


def test_stuck_alert_fires_once_after_threshold():
    """无在途活动超 STUCK_ALERT_AFTER：alert_fn 报警一次；再次 _tick 不重复报。"""
    sink = _AlertSink()
    sm = _make_stuck_sm(alert_fn=sink)
    sm._tick()
    assert len(sink.texts) == 1
    assert "疑似卡死" in sink.texts[0] and "精读 arXiv:1" in sink.texts[0]
    assert "5 分钟" in sink.texts[0]
    sm._tick()  # 同一轮静默：不重复报警
    assert len(sink.texts) == 1


def test_stuck_alert_resets_on_activity():
    """活动恢复复位：再有静默超阈可重新报警（长任务反复卡顿每次都告知）。"""
    sink = _AlertSink()
    reg = ActivityRegistry()
    sm = _make_stuck_sm(alert_fn=sink, registry=reg)
    sm._tick()
    assert len(sink.texts) == 1 and sm._stuck_alerted
    with reg.track("LLM·k3"):
        sm._tick()  # 有在途活动：复位
        assert not sm._stuck_alerted and sm._stuck_since is None
    import feishu_assistant.modules.status as status_mod
    sm._stuck_since = time.time() - status_mod.STUCK_ALERT_AFTER - 1  # 又卡住
    sm._tick()
    assert len(sink.texts) == 2  # 新一轮静默重新报警


def test_stuck_alert_below_threshold_silent():
    """静默未超阈：不报警，但已开始累计静默起点。"""
    sink = _AlertSink()
    sm = StatusMessenger(
        FakeSender(), "oc_x", "精读 arXiv:1",
        interval=999, registry=ActivityRegistry(), alert_fn=sink,
    )
    sm._tick()
    assert sink.texts == [] and sm._stuck_since is not None and not sm._stuck_alerted


def test_stuck_alert_without_alert_fn_noop():
    """未注入 alert_fn：静默超阈也不炸、不置标志（看门狗静默停用）。"""
    sm = _make_stuck_sm(alert_fn=None)
    sm._tick()
    assert not sm._stuck_alerted


def test_stuck_alert_exception_retried_next_tick():
    """alert_fn 抛异常：_tick 不外溢（不炸状态条线程），不置标志下轮重试。"""
    sink = _AlertSink(raises=True)
    sm = _make_stuck_sm(alert_fn=sink)
    sm._tick()  # 不抛
    assert not sm._stuck_alerted
    sink.raises = False
    sm._tick()  # 下轮重试成功
    assert len(sink.texts) == 1 and sm._stuck_alerted
