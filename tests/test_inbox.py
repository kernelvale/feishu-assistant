"""modules/inbox.py 测试：命中/不命中规则、分类容错、三库写入、确认回复。全部离线。"""

from __future__ import annotations

import datetime
import time

import pytest

from feishu_assistant.config import Settings
from feishu_assistant.kimi import ChatResult
from feishu_assistant.modules.inbox import (
    CLASSIFY_PROMPT_PATH,
    COLLECT_TABLE,
    IDEA_TABLE,
    MEMO_TABLE,
    InboxService,
    extract_url,
    fallback_classification,
    normalize_classification,
    should_handle,
)
from feishu_assistant.modules.paper import MessageContext

CTX = MessageContext(chat_id="c1", open_id="ou_x", message_id="m1")
NOW = time.mktime(datetime.date(2026, 7, 30).timetuple())


class FakeKimi:
    def __init__(self, reply="{}"):
        self.reply = reply
        self.calls: list[dict] = []

    def chat(self, messages, *, heavy=False, json_mode=False):
        self.calls.append({"messages": messages, "heavy": heavy})
        return self.reply if not callable(self.reply) else self.reply(messages, heavy)

    def chat_full(self, messages, *, heavy=False, json_mode=False):
        return ChatResult(self.chat(messages, heavy=heavy), "", "fake", "fake")


class FakeRecord:
    def __init__(self, record_id, fields):
        self.record_id = record_id
        self.fields = fields


class FakeIO:
    def __init__(self):
        self.records: dict[str, list[FakeRecord]] = {}
        self.dao = self

    def add_record(self, table, fields):
        rec = FakeRecord(f"{table}_r{len(self.records.setdefault(table, [])) + 1}", dict(fields))
        self.records[table].append(rec)
        return rec.record_id

    def fields_of(self, table):
        return [r.fields for r in self.records.get(table, [])]


def make_service(kimi=None, io=None):
    return InboxService(Settings(), kimi or FakeKimi(), io=io if io is not None else FakeIO())


# ---------------------------------------------------------------------------
# 命中规则
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "记一下 Transformer 的 RoPE 设计可以借鉴",
    "备忘：周五前交报销单",
    "提醒我明天下午组会",
    "收藏 https://github.com/xxx/agent-harness",
    "灵感：能不能用 harness 思路做面经评测",
    "这篇文章不错 https://example.com/post/123",
    "看看 https://www.zhihu.com/question/1 这个讨论",
])
def test_should_handle_hit(text):
    assert should_handle(text)


@pytest.mark.parametrize("text", [
    "精读 https://arxiv.org/abs/2603.03329",  # arXiv 链接归精读管线，收件箱不抢
    "https://arxiv.org/pdf/2603.03329",
])
def test_should_handle_arxiv_excluded(text):
    assert not should_handle(text)


# 闲聊锁定：20 句日常对话全部不命中
DAILY_CHATTER = [
    "今天天气怎么样",
    "帮我看看这段代码为什么报错",
    "面试官刚才问我 KV cache，怎么答比较好",
    "晚上吃什么好呢",
    "Transformer 的注意力机制给我讲讲",
    "这篇论文第三章什么意思",
    "我明天有个面试，紧张",
    "你觉得我该投算法岗还是工程岗",
    "刚才那个结论再说一遍",
    "帮我起个会话标题",
    "什么是投机采样",
    "这周读了几篇论文",
    "字节和美团选哪个",
    "怎么准备行为面",
    "LLM 的 context length 为什么重要",
    "给我讲讲 RLHF",
    "我简历该怎么改",
    "组会要讲什么",
    "agent 和普通 workflow 有啥区别",
    "帮我分析下这个 bug 的堆栈",
]


@pytest.mark.parametrize("text", DAILY_CHATTER)
def test_should_handle_daily_chatter_never_hits(text):
    assert not should_handle(text)


def test_extract_url():
    assert extract_url("看 https://example.com/a?b=1 这个") == "https://example.com/a?b=1"
    assert extract_url("没有链接") == ""


def test_extract_url_strips_trailing_punctuation():
    """URL 正则与 paper.py 同款排除：收尾的中文标点/括号不进链接（入库/域名兜底都靠它）。"""
    assert extract_url("看这个 https://example.com/a。") == "https://example.com/a"
    assert extract_url("收藏 https://example.com/x?p=1，很有用") == "https://example.com/x?p=1"
    assert extract_url("链接(https://example.com/y)") == "https://example.com/y"
    assert fallback_classification("看这个 https://example.com/a。")["title"] == "example.com"  # 域名兜底不带标点


def test_should_handle_url_with_chinese_punctuation():
    assert should_handle("这篇不错 https://example.com/post/1。")


# ---------------------------------------------------------------------------
# 分类规整与兜底
# ---------------------------------------------------------------------------

def test_normalize_classification_happy():
    data = {"type": "备忘", "deadline": "2026-08-02", "tags": ["求职", "杂事"], "note": "带电脑"}
    item = normalize_classification(data, "提醒我后天面试", now=NOW)
    assert item["type"] == "备忘" and item["tags"] == ["求职", "杂事"]
    assert item["deadline_ms"] is not None


def test_normalize_classification_invalid_type_falls_back_by_rule():
    item = normalize_classification({"type": "随便"}, "看这个 https://a.com/x", now=NOW)
    assert item["type"] == "收藏"  # 含链接兜底收藏
    item2 = normalize_classification({"type": ""}, "灵光一闪", now=NOW)
    assert item2["type"] == "灵感"  # 无链接兜底灵感


def test_normalize_classification_direction_and_deadline_validation():
    item = normalize_classification(
        {"type": "灵感", "direction": "量子计算", "deadline": "不好说"}, "想法", now=NOW)
    assert item["direction"] == "其他"  # 非法方向归「其他」
    assert item["deadline_ms"] is None  # 解析不了的日期置空


def test_normalize_classification_tags_capped():
    item = normalize_classification({"type": "灵感", "tags": ["a", "b", "c", "d", "e", "f"]}, "x")
    assert item["tags"] == ["a", "b", "c", "d"]


def test_fallback_classification_link_vs_idea():
    item = fallback_classification("看这个 https://github.com/a/b 不错")
    assert item["type"] == "收藏" and item["title"] == "github.com"
    assert fallback_classification("随便一个想法")["type"] == "灵感"


# ---------------------------------------------------------------------------
# try_handle 三库流程
# ---------------------------------------------------------------------------

def test_try_handle_link_to_collect_table():
    kimi = FakeKimi('{"type": "收藏", "title": "Agent 评测框架", "summary": "一个 harness", "tags": ["agent", "评测"]}')
    io = FakeIO()
    svc = make_service(kimi, io)
    reply = svc.try_handle("这个不错 https://example.com/harness", CTX)
    assert "已收藏" in reply and "Agent 评测框架" in reply and "agent、评测" in reply
    rows = io.fields_of(COLLECT_TABLE)
    assert len(rows) == 1
    assert rows[0]["标题"] == "Agent 评测框架"
    assert rows[0]["链接"] == {"text": "Agent 评测框架", "link": "https://example.com/harness"}  # 超链接字段结构
    assert rows[0]["已回顾"] is False and rows[0]["来源"] == "收件箱"
    assert isinstance(rows[0]["收藏时间"], int)
    assert io.records.get(IDEA_TABLE) is None and io.records.get(MEMO_TABLE) is None  # 没写错表
    assert kimi.calls[0]["heavy"] is False  # 分类走 light 链


def test_try_handle_reminder_to_memo_with_deadline():
    kimi = FakeKimi('{"type": "备忘", "deadline": "2026-08-01", "note": "带身份证"}')
    io = FakeIO()
    svc = make_service(kimi, io)
    reply = svc.try_handle("提醒我后天去体检", CTX)
    assert "已记备忘" in reply and "截止 2026-08-01" in reply and "今日提醒" in reply
    rows = io.fields_of(MEMO_TABLE)
    assert rows[0]["状态"] == "待办" and rows[0]["备注"] == "带身份证"
    assert isinstance(rows[0]["截止时间"], int) and isinstance(rows[0]["创建时间"], int)


def test_try_handle_memo_without_deadline_hints():
    kimi = FakeKimi('{"type": "备忘", "deadline": ""}')
    io = FakeIO()
    svc = make_service(kimi, io)
    reply = svc.try_handle("备忘：有空把书架整理了", CTX)
    assert "不会提醒" in reply
    assert "截止时间" not in io.fields_of(MEMO_TABLE)[0]


def test_try_handle_idea_with_direction():
    kimi = FakeKimi('{"type": "灵感", "direction": "Agent Harness", "tags": ["评测"]}')
    io = FakeIO()
    svc = make_service(kimi, io)
    reply = svc.try_handle("记一下：能不能把面经也做成 harness 评测", CTX)
    assert "已记灵感" in reply and "Agent Harness" in reply
    rows = io.fields_of(IDEA_TABLE)
    assert rows[0]["方向"] == "Agent Harness" and rows[0]["标签"] == ["评测"]
    assert "面经" in rows[0]["内容"]


def test_try_handle_model_garbage_fallback():
    io = FakeIO()
    svc = make_service(FakeKimi("模型不想输出 JSON"), io)
    reply = svc.try_handle("看这个 https://example.com/x", CTX)
    assert "已收藏" in reply and "⚠️" in reply  # 兜底提示
    rows = io.fields_of(COLLECT_TABLE)
    assert rows[0]["标题"] == "example.com"  # 域名兜底标题


def test_try_handle_kimi_error_still_saves():
    class BoomKimi(FakeKimi):
        def chat(self, messages, *, heavy=False, json_mode=False):
            from feishu_assistant.kimi import KimiError
            raise KimiError("当前选择调用失败，未自动切换")

    io = FakeIO()
    svc = make_service(BoomKimi(), io)
    reply = svc.try_handle("记一下：测试想法", CTX)
    assert "已记灵感" in reply and "⚠️" in reply
    assert len(io.fields_of(IDEA_TABLE)) == 1


def test_try_handle_non_hit_returns_none():
    io = FakeIO()
    svc = make_service(io=io)
    assert svc.try_handle("今天天气怎么样", CTX) is None
    assert io.records == {}  # 不落任何库


def test_try_handle_without_io():
    svc = InboxService(Settings(), FakeKimi('{"type": "灵感"}'), io=None)
    reply = svc.try_handle("记一下：想法", CTX)
    assert "入库失败" in reply and "未初始化" in reply


def test_real_inbox_template_complete():
    tpl = CLASSIFY_PROMPT_PATH.read_text(encoding="utf-8")
    assert "{{TEXT}}" in tpl and "{{TODAY}}" in tpl
    for marker in ("收藏", "灵感", "备忘", "deadline", "tags"):
        assert marker in tpl
