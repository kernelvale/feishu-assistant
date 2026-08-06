"""状态条（StatusMessenger）与活动登记（activity.track）测试。"""

from __future__ import annotations

import re
import time

from feishu_assistant.activity import ActivityRegistry
from feishu_assistant.modules.status import StatusMessenger, TaskCard, _IDLE_WARN_AFTER


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
    """带 Card Kit 流式通道的 sender（含重激活与富文本整卡创建）。"""

    def __init__(self):
        super().__init__()
        self.card_updates: list[tuple[str, str]] = []
        self.reactivations = 0
        self.reactivate_ok = True

    def status_card_create(self, chat_id, text):
        self.sent.append(text)
        return "card_1|status_line"

    def rich_card_create(self, chat_id, card_json):
        self.sent.append(card_json)
        return "card_1"

    def status_card_update(self, handle, text):
        assert handle.startswith("card_1|")
        self.card_updates.append((handle, text))
        return ""

    def status_card_reactivate(self, handle):
        assert handle.startswith("card_1")
        self.reactivations += 1
        return self.reactivate_ok

    def status_card_append(self, card_id, content):
        self.appends = getattr(self, "appends", [])
        self.appends.append((card_id, content))
        return getattr(self, "append_code", "")


def test_card_channel_preferred_when_available():
    """sender 有流式卡通道 → 走卡片不走文本消息，PATCH 全流式。"""
    sender = CardSender()
    sm = StatusMessenger(sender, "oc_x", "精读 arXiv:1", interval=999).start()
    assert sm._use_card and sm._msg_id == "card_1|status_line"
    sm._patch("帧一")
    sm._patch("帧二")
    assert [t for _, t in sender.card_updates] == ["帧一", "帧二"]
    assert sender.patches == []  # 文本 PATCH 一次没走


def test_card_channel_never_freezes_on_edit_cap():
    """卡片路径没有编辑上限：即使上游误回 230072 也不定格（按普通错误下帧重试）。"""
    sender = CardSender()
    sm = StatusMessenger(sender, "oc_x", "精读 arXiv:1", interval=999).start()
    sender.status_card_update = lambda h, t: "230072"  # type: ignore[assignment]
    sm._patch("某帧")
    assert not sm._frozen


def test_card_window_reactivates_instead_of_freezing():
    """流式窗口是滑动窗口（最后一次激活 ~10 分钟关闭），不是硬死：
    到重激活周期（_LIVE_WINDOW=480s）调 status_card_reactivate 续命并重置计时，
    超过旧 540s 定格点仍正常 PATCH，绝不写「状态定格」文案。"""
    import feishu_assistant.modules.status as status_mod
    sender = CardSender()
    sm = StatusMessenger(sender, "oc_x", "精读 arXiv:1", interval=999).start()
    sm._window_start = time.time() - status_mod._LIVE_WINDOW - 1  # 到重激活周期
    sm.set_stage("分层精讲")
    alive = sm._tick()
    assert alive is True and not sm._frozen
    assert sender.reactivations == 1  # 触发了重激活
    assert sm._window_start > time.time() - 5  # 窗口计时已重置
    # 超过 540 秒（旧版定格点）仍正常 PATCH，循环不结束
    sm._t0 = time.time() - 541
    sm.set_stage("图表讲解")
    alive = sm._tick()
    assert alive is True and sender.reactivations == 1  # 刚续过命：本周期不再重激活
    assert "图表讲解" in sender.card_updates[-1][1]
    assert not any("状态定格" in u for _, u in sender.card_updates)


def test_card_reactivate_failure_retries_next_tick_no_crash():
    """重激活失败不崩、不定格：只记日志，下帧重试；连续失败退避低频重试。"""
    import feishu_assistant.modules.status as status_mod
    sender = CardSender()
    sender.reactivate_ok = False
    sm = StatusMessenger(sender, "oc_x", "精读 arXiv:1", interval=999).start()
    sm._window_start = time.time() - status_mod._LIVE_WINDOW - 1
    sm._tick()  # 第 1 次失败：不崩、不定格
    assert not sm._frozen and sender.reactivations == 1
    sm._tick()
    assert sender.reactivations == 2  # 下帧重试
    sm._tick()
    assert sender.reactivations == 3  # 第 3 次失败 → 进入退避
    sm._tick()
    assert sender.reactivations == 3  # 退避中：不每帧打飞书
    # 恢复后成功：窗口计时与失败计数重置
    sender.reactivate_ok = True
    sm._next_reactivate_ts = 0.0
    sm._tick()
    assert sender.reactivations == 4 and sm._reactivate_fails == 0
    assert sm._window_start > time.time() - 5


def test_card_update_window_closed_error_triggers_reactivate_and_retry():
    """更新返回「流式窗口关闭」类错误码（非空且非 230072）：重激活后原样重试一次。"""
    sender = CardSender()
    sm = StatusMessenger(sender, "oc_x", "精读 arXiv:1", interval=999).start()
    calls = {"n": 0}
    real_update = sender.status_card_update

    def flaky(handle, text):
        calls["n"] += 1
        return "300309" if calls["n"] == 1 else real_update(handle, text)

    sender.status_card_update = flaky
    sm._patch("新一帧")
    assert sender.reactivations == 1  # 触发了重激活
    assert calls["n"] == 2  # 原样重试了一次
    assert [t for _, t in sender.card_updates] == ["新一帧"] and sm._last_text == "新一帧"


def test_card_update_error_and_reactivate_fails_keeps_going():
    """更新失败 + 重激活也失败：不崩；下一帧照常重发同一文本（_last_text 未变）。"""
    sender = CardSender()
    sender.reactivate_ok = False
    sm = StatusMessenger(sender, "oc_x", "精读 arXiv:1", interval=999).start()
    sender.status_card_update = lambda h, t: "300309"
    sm._patch("某帧")  # 不抛异常
    assert sender.reactivations == 1 and sm._last_text != "某帧"
    sm._patch("某帧")  # 下帧重发同一文本：再次尝试重激活
    assert sender.reactivations == 2


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


# ---- TaskCard：TODO 清单进度卡 ----

PAPER_STAGES = ["解析 PDF", "证据索引", "规划（债务+Plan）", "正文生成",
                "图表讲解", "终审", "发布归档"]


def _make_card(sender=None, **kw):
    kw.setdefault("interval", 999)
    return TaskCard(sender or CardSender(), "oc_x", "精读 arXiv:1", PAPER_STAGES, **kw)


def _el(card_json, element_id):
    """按 element_id 取卡片元素 content（布局加元素时断言不跟着挪）。"""
    return next(e["content"] for e in card_json["body"]["elements"]
                if e.get("element_id") == element_id)


def test_taskcard_render_todo_and_stage_flow():
    """渲染：header 模板色、TODO 计数、阶段切换、图标体系（🌟/🚀/🕐）、current 唯一。"""
    sender = CardSender()
    card = _make_card(sender).start()
    first = sender.sent[0]
    assert first["header"]["template"] == "blue"
    assert first["config"]["streaming_mode"] is True
    todo = _el(first, "todo_body")
    footer = _el(first, "footer_line")
    assert "📋 **TODO (0/7)**" in todo
    assert "🕐" in todo and "解析 PDF" in todo and "发布归档" in todo
    assert "grey" in footer and "⏱ 已用" in footer  # 灰色页脚 + 统计行
    card.set_stage(0)
    todo = _el(card._render(), "todo_body")
    assert "🚀 **解析 PDF**" in todo and todo.count("🚀") == 1
    card.set_stage(1, "🧭 正在建立全文证据索引")
    todo = _el(card._render(), "todo_body")
    assert "🌟 解析 PDF" in todo  # 原 current 打勾
    assert "🚀 **证据索引** · 正在建立全文证据索引" in todo  # detail 随行（emoji 前缀剥掉）
    assert "📋 **TODO (1/7)**" in todo and todo.count("🚀") == 1  # current 唯一


def test_taskcard_detail_truncated_40():
    card = _make_card()
    card.set_stage(0, "x" * 100)
    detail = card._stages[0]["detail"]
    assert len(detail) == 40 and detail.endswith("…")


def test_taskcard_regression_only_updates_current_detail():
    """单调前进：早阶段文本不倒退 TODO，只更新当前阶段 detail。"""
    card = _make_card()
    card.set_stage(5, "终审意见返工 2 处")
    card.set_stage(3, "精讲 层2/3：注意力机制")  # 返工回流的早阶段进度
    assert card._stages[5]["state"] == "current"
    assert card._stages[3]["state"] == "pending"  # 没被打回 current
    assert "层2/3" in card._stages[5]["detail"]


def test_taskcard_set_stage_unknown_text_is_current_detail():
    """瞬时重试文案（paper 直调 set_stage）：不推进阶段，只更新当前 detail。"""
    card = _make_card()
    card.set_stage(3)
    card.set_stage("🌐 网络中断，5 分钟后自动重试（第 1/3 次）")
    assert card._stages[3]["state"] == "current"
    assert "网络中断" in card._stages[3]["detail"]
    assert "🌐" not in card._stages[3]["detail"]


def test_taskcard_render_activity_idle_and_calls():
    """页脚：在途 ⚙️ 短模型名；静默 💤；≥60s 疑似卡死；calls_fn 追加调用数。"""
    reg = ActivityRegistry()

    def footer_of(c):
        return _el(c._render(), "footer_line")

    card = _make_card(registry=reg, calls_fn=lambda: 37)
    with reg.track("LLM·free-tokens/kimi-k3"):
        text = footer_of(card)
        assert "⚙️ kimi-k3" in text and "free-tokens" not in text
        assert "· 调用 37 次" in text
    assert "💤 静默" in footer_of(card)
    card._idle_since = time.time() - 120
    assert "疑似卡死" in footer_of(card)
    # calls_fn 异常：静默省略调用数，不炸渲染
    def bad():
        raise RuntimeError("计数故障")
    card2 = _make_card(calls_fn=bad)
    assert "调用" not in footer_of(card2)


def test_taskcard_finish_success_marks_all_done_and_is_last():
    """定格帧：TODO 终态（成功全部打勾）+ 页脚 ✅ 全程时长 + note；且必是最后一帧。"""
    sender = CardSender()
    card = _make_card(sender).start()
    card.set_stage(0)
    card.mark_done(0, "12 页，5 张图表")
    card.set_stage(1)
    card.finish(ok=True, note="精读完成，文档与详情见下方结果卡")
    todo_final = [t for h, t in sender.card_updates if h.endswith("todo_body")][-1]
    footer_final = [t for h, t in sender.card_updates if h.endswith("footer_line")][-1]
    assert "📋 **TODO (7/7)**" in todo_final
    assert "🌟 解析 PDF · 12 页，5 张图表" in todo_final and "🌟 发布归档" in todo_final
    assert "✅" in footer_final and "全程" in footer_final and "结果卡" in footer_final
    last = sender.card_updates[-1]
    card._patch("迟到帧")  # 定格帧已写：迟到帧不得覆盖
    assert sender.card_updates[-1] == last


def test_taskcard_finish_failure_keeps_terminal_state():
    """失败定格：页脚 ❌ + TODO 终态保留现场（死在哪个阶段一眼可见）。"""
    sender = CardSender()
    card = _make_card(sender).start()
    card.set_stage(0)
    card.mark_done(0)
    card.set_stage(1)
    card.finish(ok=False, note="失败：ReadTimeout")
    todo_final = [t for h, t in sender.card_updates if h.endswith("todo_body")][-1]
    footer_final = [t for h, t in sender.card_updates if h.endswith("footer_line")][-1]
    assert "🌟 解析 PDF" in todo_final
    assert "🚀 **证据索引**" in todo_final and "🕐" in todo_final  # 现场保留
    assert "❌" in footer_final and "ReadTimeout" in footer_final


def test_taskcard_stop_alias_matches_status_messenger_protocol():
    """stop 与 StatusMessenger 同协议（paper._run_guarded 调 stop）。"""
    sender = CardSender()
    card = _make_card(sender).start()
    card.stop(ok=True, note="完成")
    footer_final = [t for h, t in sender.card_updates if h.endswith("footer_line")][-1]
    assert "✅" in footer_final


def test_taskcard_past_540s_still_patches_via_reactivation():
    """钉住本修复核心：超过旧 540s 定格点仍正常 PATCH 且触发了重激活（不再定格）。"""
    import feishu_assistant.modules.status as status_mod
    sender = CardSender()
    card = _make_card(sender).start()
    card._t0 = time.time() - 541
    card._window_start = time.time() - status_mod._LIVE_WINDOW - 1  # 到重激活周期
    card.set_stage(3, "精讲 层1/3：注意力机制")
    card._tick()
    assert sender.reactivations == 1  # 重激活续命
    assert not card._frozen
    assert any("精讲 层1/3" in u for _, u in sender.card_updates)  # 帧正常 PATCH
    assert not any("状态定格" in u for _, u in sender.card_updates)


def test_taskcard_reactivate_failure_retries_next_tick():
    """重激活失败不崩、不打扰用户：下帧重试。"""
    import feishu_assistant.modules.status as status_mod
    sender = CardSender()
    sender.reactivate_ok = False
    card = _make_card(sender).start()
    card._window_start = time.time() - status_mod._LIVE_WINDOW - 1
    card._tick()
    assert sender.reactivations == 1 and not card._frozen
    card._tick()
    assert sender.reactivations == 2  # 下帧重试
    assert not any("状态定格" in u for _, u in sender.card_updates)


def test_taskcard_update_window_error_reactivates_and_retries():
    """更新遇「窗口关闭」类错误码（非 230072）：重激活后原样重试一次。"""
    sender = CardSender()
    card = _make_card(sender).start()
    calls = {"n": 0}
    real_update = sender.status_card_update

    def flaky(handle, text):
        calls["n"] += 1
        return "300309" if calls["n"] == 1 else real_update(handle, text)

    sender.status_card_update = flaky
    card.set_stage(0)
    card._tick()
    assert sender.reactivations == 1 and calls["n"] >= 2
    assert any("🚀 **解析 PDF**" in u for _, u in sender.card_updates)


def test_taskcard_230072_freezes_without_crash():
    """230072 类致命错误：定格不再更新（任务继续）；finish 尽力写终帧不崩。"""
    sender = CardSender()
    card = _make_card(sender).start()
    sender.status_card_update = lambda h, t: "230072"
    card.set_stage(0)
    card._tick()
    assert card._frozen and sender.reactivations == 0  # 230072 不触发重激活
    card._tick()  # 定格后不再 PATCH
    card.finish(ok=True)  # 不崩：尽力写最后一帧（写不进也静默）


def test_taskcard_patches_only_on_substance():
    """实质变化才 PATCH：计时数字与 spinner 帧不驱动刷新。"""
    sender = CardSender()
    card = _make_card(sender).start()
    n0 = len(sender.card_updates)
    card._tick()
    card._tick()
    assert len(sender.card_updates) == n0  # 无实质变化不刷
    card.set_stage(0)
    card._tick()
    assert len(sender.card_updates) == n0 + 1  # 阶段变化 → 刷
    card._tick()
    assert len(sender.card_updates) == n0 + 1  # 同状态不重复刷
    card.set_stage(0, "开始解析 PDF（约 1~2 分钟）")
    card._tick()
    assert len(sender.card_updates) == n0 + 2  # detail 变 → 刷


def test_taskcard_start_failure_silent_degrade():
    """首发失败（创建返回空 / sender 无卡通道）：静默降级，任务继续，不抛不炸。"""

    class EmptyCardSender(CardSender):
        def status_card_create(self, chat_id, text):
            return ""

    card = _make_card(EmptyCardSender(), interval=0.01).start()
    card.set_stage(0)
    card.finish(ok=True)  # 不炸线程、不抛异常
    card2 = _make_card(FakeSender(), interval=0.01).start()  # 无卡通道 sender
    card2.apply_progress("✅ 解析完成（3 页，1 张图表，0 个公式，5s）。")
    card2.finish(ok=False)


# ---- 精读管线进度文本 → TODO 阶段映射 ----

def test_progress_mapping_parse_start_and_done():
    card = _make_card()
    card.apply_progress("✅ arXiv（4.2 MB），开始解析 PDF（约 1~2 分钟）……")
    assert card._stages[0]["state"] == "current"
    card.apply_progress(
        "✅ 解析完成（12 页，5 张图表，3 个公式，45s）。"
        "开始精读生成：骨架、分层精讲、逐图解释完成后会整篇终审，通过后再写入文档。")
    assert card._stages[0]["state"] == "done"
    assert "12 页" in card._stages[0]["detail"] and "5 张图表" in card._stages[0]["detail"]


def test_progress_mapping_evidence_index():
    card = _make_card()
    card.apply_progress("🧭 正在建立全文证据索引（长论文会逐块处理，原文不会截尾）……")
    assert card._stages[1]["state"] == "current"
    card.apply_progress("长论文证据索引 2/9：Introduction")
    assert card._stages[1]["state"] == "current" and "2/9" in card._stages[1]["detail"]


def test_progress_mapping_plan_and_body():
    card = _make_card()
    card.apply_progress("🧠 教学计划生成中（只规划，不在一次调用里硬写整篇）……")
    assert card._stages[2]["state"] == "current"
    card.apply_progress("正文生成 2/5：方法概览")
    assert card._stages[2]["state"] == "done" and card._stages[3]["state"] == "current"
    assert "正文生成 2/5" in card._stages[3]["detail"]
    card.apply_progress("精讲 层1/3：注意力机制")
    assert card._stages[3]["state"] == "current" and "层1/3" in card._stages[3]["detail"]


def test_progress_mapping_plan_done_enters_body_not_regress():
    """「✅ 教学计划完成…开始分层精讲」必须留在正文生成（stage 3），不得退回规划（stage 2）。"""
    card = _make_card()
    card.apply_progress("🧠 教学计划生成中……")
    card.apply_progress("正文生成 1/5：引言")
    assert card._stages[2]["state"] == "done" and card._stages[3]["state"] == "current"
    card.apply_progress("✅ 教学计划完成（识别出 3 个技术层），开始分层精讲")
    assert card._stages[3]["state"] == "current"  # 不倒退
    assert card._stages[2]["state"] == "done"


def test_progress_mapping_figures_review_rework_publish():
    card = _make_card()
    card.apply_progress("图表讲解 2/5（视觉）")
    assert card._stages[4]["state"] == "current"
    card.apply_progress("🔁 终审意见返工 2 处（第 1/2 轮）……")
    assert card._stages[4]["state"] == "done" and card._stages[5]["state"] == "current"
    assert "返工" in card._stages[5]["detail"]
    # 返工期间的早阶段文本不倒退 TODO，只更新当前 detail
    card.apply_progress("精讲 层2/3：注意力机制")
    assert card._stages[5]["state"] == "current" and "层2/3" in card._stages[5]["detail"]
    # 「已通过终审并写入」是发布归档完成（stage 6 done），不是退回终审
    card.apply_progress("📄 精读文档已通过终审并写入：https://feishu.cn/docx/xxx")
    assert card._stages[6]["state"] == "done" and card._stages[5]["state"] == "done"


def test_progress_unmatched_text_updates_current_detail():
    """未命中规则的文本：不推进阶段，作为当前阶段 detail 展示。"""
    card = _make_card()
    card.apply_progress("🧠 教学计划生成中……")
    card.apply_progress("🧾 理解债务编译与准入审核中……")
    assert card._stages[2]["state"] == "current"
    assert "理解债务编译" in card._stages[2]["detail"]


def test_taskcard_done_stage_sanitizes_inprogress_detail():
    """阶段打勾后清掉进行中痕迹：「✅ …生成中……」自相矛盾（用户实证）。"""
    card = _make_card()
    card.set_stage(2, "教学计划生成中……")
    card.set_stage(3, "正文生成 3/16：某节标题")
    assert card._stages[2]["state"] == "done"
    assert card._stages[2]["detail"] == ""  # 进行中措辞清空
    card.set_stage(4, "图表讲解 4/10（视觉读图）")
    assert card._stages[3]["detail"] == ""  # 未完成分数 3/16 清空
    card.set_stage(5, "终审")
    assert card._stages[4]["detail"] == ""  # 4/10 清空


def test_taskcard_done_stage_keeps_result_detail():
    """结果类 detail 打勾后保留：「21 页，10 张图表」「9/9」不是进行中痕迹。"""
    card = _make_card()
    card.apply_progress("开始解析 PDF")
    card.apply_progress("✅ 解析完成（21 页，10 张图表，0 个公式，0s）。开始精读生成")
    assert card._stages[0]["state"] == "done"
    assert "21 页" in card._stages[0]["detail"]
    card.set_stage(1, "长论文证据索引 9/9")
    card.set_stage(2, "教学计划生成中……")
    assert card._stages[1]["detail"] == "长论文证据索引 9/9"  # 9/9 完成分数保留


def test_taskcard_completion_rules_set_final_detail():
    """完成类消息写入终态文案：括号内容优先（「识别出 N 个技术层」）。"""
    card = _make_card()
    card.apply_progress("🧠 教学计划生成中……")
    card.apply_progress("✅ 教学计划完成（识别出 3 个技术层），开始分层精讲")
    assert card._stages[2]["state"] == "done"
    assert card._stages[2]["detail"] == "识别出 3 个技术层"
    card.apply_progress("精讲 层1/3：注意力机制")
    card.apply_progress("✅ 精讲完成（800s，3 层已通过验收），开始图表讲解")
    assert card._stages[3]["state"] == "done"
    assert card._stages[3]["detail"] == "800s，3 层已通过验收"


def test_taskcard_rework_detail_cleared_on_done():
    """返工 detail 属于进行中痕迹：发布归档打勾时终审阶段 detail 清空。"""
    card = _make_card()
    card.set_stage(5, "🔁 终审意见返工 3 处（第 1/2 轮）……")
    card.apply_progress("📄 精读文档已通过终审并写入：https://feishu.cn/docx/xxx")
    assert card._stages[5]["state"] == "done"
    assert card._stages[5]["detail"] == ""
    assert card._stages[6]["state"] == "done"


# ---- 最近动态（关键事件区）与统计行扩展 ----

def test_add_event_append_truncate_and_merge():
    """add_event：追加、50 字截断、连续重复合并 ×N（时间戳取最近一次）。"""
    card = _make_card()
    card.add_event("✎ 补丁修复：相关工作")
    card.add_event("✎ 补丁修复：相关工作")
    card.add_event("✎ 补丁修复：相关工作")
    assert len(card._events) == 1  # 连续相同合并，不刷屏
    assert card._events[0]["text"] == "✎ 补丁修复：相关工作"
    assert card._events[0]["count"] == 3
    card.add_event("⚠ " + "长" * 100)
    assert len(card._events[-1]["text"]) == 50
    assert card._events[-1]["text"].endswith("…")
    card.add_event("✅ 与事件无关的 emoji 前缀会被剥掉")
    assert card._events[-1]["text"] == "与事件无关的 emoji 前缀会被剥掉"


def test_add_event_skips_detail_duplicate_and_periodic():
    """与 TODO 当前 detail 相同跳过；周期性进度文本（已在 detail 实时刷新）不进。"""
    card = _make_card()
    card.set_stage(3, "正文生成 2/5：方法概览")
    card.add_event("正文生成 2/5：方法概览")  # 周期性 + 与 detail 相同（双重复）
    card.add_event("精讲 层1/3：注意力机制")    # 周期性规则词
    card.add_event("图表讲解 2/5（视觉读图）")
    assert card._events == []
    card.set_stage(3, "✎ 补丁修复：相关工作")
    card.add_event("✎ 补丁修复：相关工作")      # 与当前 detail 相同：跳过
    assert card._events == []


def test_add_event_rolling_max_3():
    """上限 3 条滚动：只留最近三条。"""
    card = _make_card()
    for i in range(5):
        card.add_event(f"✎ 补丁修复：第{i}节")
    assert [e["text"] for e in card._events] == [
        "✎ 补丁修复：第2节", "✎ 补丁修复：第3节", "✎ 补丁修复：第4节"]


def test_events_section_render_and_finish_kept():
    """最近动态区：位于 TODO 与页脚之间（recent_events 元素），每行「· HH:MM 文本」；
    不足 3 条有几条显示几条；finish 终态保留最后三条。"""
    sender = CardSender()
    card = _make_card(sender).start()
    ids = [e.get("element_id") for e in sender.sent[0]["body"]["elements"]]
    assert ids == ["todo_body", None, "recent_events", None, "footer_line"]
    assert _el(sender.sent[0], "recent_events") == "📌 **最近动态**"  # 0 条：只有标题
    card.add_event("✎ 补丁修复：相关工作")
    card.add_event("⚠ 方法：审核未过，保留全文加批注")
    content = _el(card._render(), "recent_events")
    lines = content.splitlines()
    assert lines[0] == "📌 **最近动态**" and len(lines) == 3  # 不足 3 条有几条显示几条
    assert re.fullmatch(r"· \d{2}:\d{2} ✎ 补丁修复：相关工作", lines[1])
    assert re.fullmatch(r"· \d{2}:\d{2} ⚠ 方法：审核未过，保留全文加批注", lines[2])
    card.finish(ok=True)
    assert card._last_events == content  # 定格帧事件区原样保留（终态可见）


def test_events_in_signature_drive_patch():
    """事件区纳入签名：事件变化才 PATCH，无变化不刷。"""
    sender = CardSender()
    card = _make_card(sender).start()
    n0 = len(sender.card_updates)
    card._tick()
    assert len(sender.card_updates) == n0  # 无实质变化不刷
    card.add_event("✓ 局审通过：方法（带 2 条待复核意见）")
    card._tick()
    assert len(sender.card_updates) == n0 + 1  # 事件变化 → 刷
    assert any("局审通过" in u for h, u in sender.card_updates
               if h.endswith("recent_events"))


def test_progress_event_texts_route_to_events_not_todo():
    """事件文本（✓/⚠/✎/⤴ 开头）只进事件区；阶段文本仍只进 TODO（互不串区）。"""
    card = _make_card()
    card.apply_progress("正文生成 2/5：方法概览")
    assert card._stages[3]["state"] == "current" and card._events == []
    card.apply_progress("✎ 补丁修复：方法概览")
    card.apply_progress("⚠ 方法概览：审核未过，保留全文加批注")
    card.apply_progress("⤴ 审核发现 Plan 问题，已回写并局部重生成")
    assert [e["text"] for e in card._events] == [
        "✎ 补丁修复：方法概览",
        "⚠ 方法概览：审核未过，保留全文加批注",
        "⤴ 审核发现 Plan 问题，已回写并局部重生成",
    ]
    # detail 不被事件覆盖（仍是阶段进度文本）
    assert card._stages[3]["detail"] == "正文生成 2/5：方法概览"
    # 事件前缀优先于阶段规则：含规则词「终审」的事件文本不触发阶段映射
    card.apply_progress("⚠ 终审服务不可用，按未经终审复核标注交付")
    assert card._stages[5]["state"] == "pending"
    assert card._events[-1]["text"] == "⚠ 终审服务不可用，按未经终审复核标注交付"


def test_stats_line_cached_formatting():
    """缓存格式化三档：0 省略 / 500 原样 / 45200 → 45.2k；cached_fn 异常静默省略。"""
    card = _make_card(calls_fn=lambda: 18, cached_fn=lambda: 0)
    line = card._stats_line()
    assert "调用 18 次" in line and "缓存" not in line  # 缓存 0：省略该项
    card.cached_fn = lambda: 500
    assert "· 缓存 500" in card._stats_line()
    card.cached_fn = lambda: 45200
    assert "· 缓存 45.2k" in card._stats_line()

    def bad():
        raise RuntimeError("计数故障")
    card.cached_fn = bad
    assert "缓存" not in card._stats_line()  # 异常：静默省略，不炸统计行


def test_stats_baseline_delta():
    """基线增量：进程级累计减去任务开始基线，卡片只展示本任务增量。"""
    total = {"calls": 100, "cached": 9000}
    base = dict(total)  # paper 在任务开始时记录的基线
    card = _make_card(
        calls_fn=lambda: total["calls"] - base["calls"],
        cached_fn=lambda: total["cached"] - base["cached"],
    )
    assert "调用 0 次" in card._stats_line() and "缓存" not in card._stats_line()
    total["calls"] += 18
    total["cached"] += 45200
    assert "· 调用 18 次 · 缓存 45.2k" in card._stats_line()


def test_taskcard_finish_with_result_appends_result_element():
    """result_md 非空：结果区追加进同一卡片（元素追加），不再单独发结果卡。"""
    sender = CardSender()
    card = _make_card(sender).start()
    card.set_stage(0)
    card.finish(ok=True, note="精读完成",
                result_md="**📄 精读结果**\n📖 [打开精读文档](http://doc)")
    appends = getattr(sender, "appends", [])
    assert len(appends) == 1
    assert appends[0][0] == "card_1" and "打开精读文档" in appends[0][1]
    footer_final = [t for h, t in sender.card_updates if h.endswith("footer_line")][-1]
    assert "精读完成" in footer_final and "打开精读文档" not in footer_final  # 页脚不混入


def test_taskcard_finish_result_append_fails_falls_back_to_footer():
    """追加失败（含重激活后仍失败）：结果并入页脚，信息不丢。"""
    sender = CardSender()
    sender.append_code = "200850"  # 流式窗口关闭 → 重激活后仍失败
    card = _make_card(sender).start()
    card.set_stage(0)
    card.finish(ok=True, note="精读完成", result_md="**📄 精读结果**\n关键信息")
    footer_final = [t for h, t in sender.card_updates if h.endswith("footer_line")][-1]
    assert "关键信息" in footer_final  # 兜底并入页脚
    assert sender.reactivations >= 1  # 触发过重激活重试


def test_taskcard_done_stage_sanitizes_v9_inprogress_wording():
    """V9 措辞（「正在建立/正在并列执行」）同样属于进行中痕迹，打勾即清。"""
    card = _make_card()
    card.set_stage(1, "V9 正在建立来源导航")
    card.set_stage(2, "V9 作者正在建立轻量叙事简报")
    assert card._stages[1]["detail"] == ""  # 🌟 行下不应残留「正在」
    card.set_stage(3, "单一作者正在写正文")
    assert card._stages[2]["detail"] == ""


def test_taskcard_takeover_resumes_existing_card(tmp_path):
    """原卡接管：有状态文件时不再创建新卡——重激活旧卡并把当前状态刷进去。"""
    state = tmp_path / "card.json"
    sender1 = CardSender()
    card1 = _make_card(sender1, state_path=state).start()
    card1.set_stage(2, "教学计划生成中……")
    card1._tick()
    card_id_1 = card1._msg_id
    # 模拟进程重启：新 sender + 新 TaskCard，同一状态文件
    sender2 = CardSender()
    card2 = _make_card(sender2, state_path=state).start()
    assert card2._msg_id == card_id_1  # 接管旧卡，没建新卡
    assert len(sender2.sent) == 0  # 没有首发新卡片
    assert sender2.reactivations >= 1  # 重激活了
    # 旧卡阶段状态被接管恢复（直跳 stage 2，所以它是 current）
    assert card2._stages[2]["state"] == "current"
    assert "教学计划" in card2._stages[2]["detail"]


def test_taskcard_takeover_failure_creates_new_card(tmp_path):
    """接管失败（旧卡已被删除等）：回落到新建卡。"""
    state = tmp_path / "card.json"
    state.write_text('{"card_id": "card_gone", "stages": []}', encoding="utf-8")
    sender = CardSender()
    sender.reactivate_ok = False  # 重激活失败
    card = _make_card(sender, state_path=state).start()
    assert card._msg_id == "card_1"  # 新建卡（CardSender.rich_card_create 的返回值）
    assert len(sender.sent) == 1


def test_taskcard_finish_discards_state(tmp_path):
    """定格后删除状态文件：下一次同论文启动新卡而非接管已完成卡。"""
    state = tmp_path / "card.json"
    sender = CardSender()
    card = _make_card(sender, state_path=state).start()
    card.set_stage(0)
    card._tick()
    assert state.exists()
    card.finish(ok=True, note="完成")
    assert not state.exists()
