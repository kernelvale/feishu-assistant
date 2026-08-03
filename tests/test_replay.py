# ---- [V7.1 清理] 已停用：对应生产代码 replay 模块（ReplayService/cmd_list/cmd_replay）已停用。经全项目核对生产链路零引用，保留备查 ----
# """死信重放测试：列表、重放流程、非法输入、非文本降级。"""
#
# from __future__ import annotations
#
# from feishu_assistant.modules.paper import MessageContext
# from feishu_assistant.modules.replay import ReplayService
# from feishu_assistant.store import DedupStore
#
# CTX = MessageContext(chat_id="c1", open_id="ou_x", message_id="m1")
#
#
# def make_service(tmp_path, reply="重放回复"):
#     dedup = DedupStore(tmp_path / "t.db")
#     calls = []
#
#     def dispatch(text, ctx):
#         calls.append(text)
#         return reply
#
#     svc = ReplayService(dedup, dispatch)
#     return svc, dedup, calls
#
#
# def test_list_empty(tmp_path):
#     svc, _, _ = make_service(tmp_path)
#     assert "没有待重放" in svc.cmd_list(CTX)
#
#
# def test_list_and_replay_flow(tmp_path):
#     svc, dedup, calls = make_service(tmp_path)
#     dedup.add_dead_letter("m1", "c1", "重复投递（已处理过）", content_text="精读 https://arxiv.org/abs/2603.03329")
#     dedup.add_dead_letter("m2", "c1", "崩溃残留（上次处理未完成，按已处理跳过）", content_text="第二条消息")
#
#     listing = svc.cmd_list(CTX)
#     assert "共 2 条" in listing and "第二条消息" in listing and "精读" in listing
#
#     reply = svc.cmd_replay("1", CTX)  # 序号 1 = 最新的 m2
#     assert "已重放" in reply and calls == ["第二条消息"]
#     # 已标记重放：列表只剩 1 条
#     assert "共 1 条" in svc.cmd_list(CTX)
#     # 再重放同序号 → 现在是 m1
#     svc.cmd_replay("1", CTX)
#     assert calls[-1].startswith("精读 https://")
#     assert "没有待重放" in svc.cmd_list(CTX)
#
#
# def test_replay_invalid_index(tmp_path):
#     svc, dedup, _ = make_service(tmp_path)
#     dedup.add_dead_letter("m1", "c1", "r", content_text="x")
#     assert "没有第 99 条" in svc.cmd_replay("99", CTX)
#     assert "用法" in svc.cmd_replay("abc", CTX)
#
#
# def test_replay_non_text_marks_and_skips(tmp_path):
#     svc, dedup, calls = make_service(tmp_path)
#     dedup.add_dead_letter("m1", "c1", "重复投递", msg_type="file", content_text="论文.pdf")
#     reply = svc.cmd_replay("1", CTX)
#     assert "暂不支持重放" in reply and calls == []
#     assert dedup.list_dead_letters("c1") == []  # 已标记
#
#
# def test_cross_chat_isolation(tmp_path):
#     svc, dedup, _ = make_service(tmp_path)
#     dedup.add_dead_letter("m1", "other_chat", "r", content_text="别人的")
#     assert "没有待重放" in svc.cmd_list(CTX)
