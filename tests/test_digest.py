"""modules/digest.py 测试：arXiv 解析/排序/TL;DR 容错、今日归集与翻坟、周报聚合、推送重试、调度。

全部离线：fetcher / kimi / io / docx / sender / vector_index 均注入 fake。
"""

from __future__ import annotations

import datetime
import time

import pytest
from apscheduler.schedulers.background import BackgroundScheduler

from feishu_assistant.config import Settings
from feishu_assistant.kimi import ChatResult
from feishu_assistant.modules.digest import (
    ARXIV_PROMPT_PATH,
    COLLECT_TABLE,
    MEMO_TABLE,
    PAPER_TABLE,
    IDEA_TABLE,
    BOARD_TABLE,
    PICK_TABLE,
    WEEKLY_PROMPT_PATH,
    ArxivEntry,
    DigestError,
    DigestService,
    TodayItems,
    WeeklyMaterials,
    build_arxiv_card,
    build_arxiv_query,
    build_arxiv_text,
    build_pick_row,
    build_today_message,
    build_weekly_prompt,
    collect_today_items,
    collect_weekly_materials,
    existing_pick_ids,
    keyword_score,
    parse_arxiv_feed,
    parse_tldr_json,
    pick_resurface,
    rank_entries,
    register_jobs,
    taste_keywords_from_papers,
)

NOW = time.mktime(datetime.date(2026, 7, 30).timetuple())  # 固定"今天"：2026-07-30
NOW_MS = int(NOW * 1000)
DAY = 86_400_000


def _ms(date_str: str) -> int:
    return int(time.mktime(datetime.date.fromisoformat(date_str).timetuple()) * 1000)


def _entry(arxiv_id, title, abstract="", published="2026-07-30"):
    return ArxivEntry(arxiv_id=arxiv_id, title=title, abstract=abstract or f"{title} 的摘要",
                      authors="A, B", published=published, url=f"https://arxiv.org/abs/{arxiv_id}")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeKimi:
    def __init__(self, reply="[]"):
        self.reply = reply
        self.calls: list[dict] = []

    def chat(self, messages, *, heavy=False, json_mode=False):
        self.calls.append({"messages": messages, "heavy": heavy})
        return self.reply(messages, heavy) if callable(self.reply) else self.reply

    def chat_full(self, messages, *, heavy=False, json_mode=False):
        return ChatResult(self.chat(messages, heavy=heavy), "", "fake", "fake")


class FakeRecord:
    def __init__(self, record_id, fields):
        self.record_id = record_id
        self.fields = fields


class FakeIO:
    """CareerIO 协议替身（digest 只读：list_all；归档/升级走 dao 三件套）。"""

    def __init__(self, data=None):
        self.records = {t: [FakeRecord(f"{t}_r{i+1}", dict(f)) for i, f in enumerate(rows)]
                        for t, rows in (data or {}).items()}
        self.dao = self

    def list_all(self, table, max_pages=20):
        return list(self.records.get(table, []))

    def add_record(self, table, fields):
        rec = FakeRecord(f"{table}_r{len(self.records.setdefault(table, [])) + 1}", dict(fields))
        self.records[table].append(rec)
        return rec.record_id

    def find_record_by_field(self, table, field, value):
        for rec in self.records.get(table, []):
            if rec.fields.get(field) == value:
                return rec.record_id
        return None

    def find_record_fields(self, table, field, value):
        for rec in self.records.get(table, []):
            if rec.fields.get(field) == value:
                return dict(rec.fields)
        return None

    def update_record(self, table, record_id, fields):
        for rec in self.records.get(table, []):
            if rec.record_id == record_id:
                rec.fields.update(fields)
                return True
        return False


class FakeDocx:
    def __init__(self):
        self.docs: list[tuple[str, str]] = []

    def create_document(self, title):
        return f"doc_{len(self.docs) + 1}", f"https://feishu.cn/docx/doc_{len(self.docs) + 1}"

    def write_markdown(self, doc_id, md, images=None):
        self.docs.append((doc_id, md))

    def grant_access(self, doc_token, open_id=None):
        return ["tenant_readable"]


class FakeSender:
    """fail_times 次发送失败后恢复成功（测重试）。"""

    def __init__(self, fail_times=0):
        self.texts: list[tuple[str, str]] = []
        self.cards: list[tuple[str, dict]] = []
        self.fail_times = fail_times
        self.attempts = 0

    def _maybe_fail(self) -> bool:
        self.attempts += 1
        return self.attempts <= self.fail_times

    def send_text(self, chat_id, text):
        if self._maybe_fail():
            return False
        self.texts.append((chat_id, text))
        return True

    def send_card(self, chat_id, card):
        if self._maybe_fail():
            return False
        self.cards.append((chat_id, card))
        return True


class FakeVectorIndex:
    def __init__(self, hits=None, size=3):
        self.hits = hits if hits is not None else [("2603.03329", "旧论文", 0.2)]
        self.size = size
        self.search_calls: list[dict] = []  # 记录 exclude_arxiv_id 等参数

    def count(self):
        return self.size

    def search(self, title, tldr="", keywords=None, top_k=3, exclude_arxiv_id=""):
        self.search_calls.append({"title": title, "exclude_arxiv_id": exclude_arxiv_id})
        return list(self.hits)


def make_service(kimi=None, data=None, sender=None, vector_index=None, fetcher=None, docx=None,
                 authority_scorer=None):
    sender = sender if sender is not None else FakeSender()
    svc = DigestService(
        Settings(push_interests=["论文", "agent", "LLM"]), kimi or FakeKimi(),
        sender=sender, io=FakeIO(data), docx=docx if docx is not None else FakeDocx(),
        vector_index=vector_index,
        fetcher=fetcher or (lambda days=2: [_entry("2607.00001", "默认论文")]),
        authority_scorer=authority_scorer or (lambda e: (0.9, ["测试机构"])),  # 离线打桩，默认高权威
        chat_id="oc_push",
    )
    return svc, sender


# ---------------------------------------------------------------------------
# arXiv XML 解析
# ---------------------------------------------------------------------------

SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query</title>
  <entry>
    <id>http://arxiv.org/abs/2607.12345v2</id>
    <updated>2026-07-30T03:00:00Z</updated>
    <published>2026-07-29T17:59:01Z</published>
    <title>  Agent Harness: A Unified\n    Evaluation Framework  </title>
    <summary>  We propose a harness for agents.\n  It improves pass rate by 12%. </summary>
    <author><name>Alice Zhang</name></author>
    <author><name>Bob Li</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2607.99999v1</id>
    <published>2026-07-28T10:00:00Z</published>
    <title>Speculative Decoding for LLM Inference</title>
    <summary>Faster decoding via drafting.</summary>
    <author><name>Carol</name></author>
  </entry>
</feed>
"""


def test_parse_arxiv_feed_happy():
    entries = parse_arxiv_feed(SAMPLE_FEED)
    assert len(entries) == 2
    e = entries[0]
    assert e.arxiv_id == "2607.12345"  # 版本号已剥
    assert e.title == "Agent Harness: A Unified Evaluation Framework"  # 空白已规整
    assert "12%" in e.abstract
    assert e.authors == "Alice Zhang, Bob Li"
    assert e.published == "2026-07-29"
    assert e.url == "https://arxiv.org/abs/2607.12345"


def test_parse_arxiv_feed_invalid_xml():
    with pytest.raises(DigestError, match="XML"):
        parse_arxiv_feed("网关返回了 502：<html>没闭合的错误页")


def test_parse_arxiv_feed_empty():
    assert parse_arxiv_feed('<feed xmlns="http://www.w3.org/2005/Atom"></feed>') == []


def test_build_arxiv_query_parts():
    q = build_arxiv_query(days=2, now=NOW)
    assert "cat:cs.CL" in q and "cat:cs.AI" in q and "cat:cs.LG" in q
    assert "all:agent" in q and "LLM" in q
    assert "submittedDate:[202607280000 TO 202607302359]" in q  # 近 2 天区间


# ---------------------------------------------------------------------------
# 排序打分
# ---------------------------------------------------------------------------

def test_rank_entries_by_score_then_date():
    entries = [_entry("1", "低分新", published="2026-07-30"),
               _entry("2", "高分", published="2026-07-28"),
               _entry("3", "低分旧", published="2026-07-27")]
    scores = {"低分新": 1.0, "高分": 5.0, "低分旧": 1.0}
    ranked = rank_entries(entries, lambda e: scores[e.title])
    assert [e.title for e, _ in ranked] == ["高分", "低分新", "低分旧"]  # 同分按日期新→旧


def test_keyword_score_counts_distinct_hits():
    entry = _entry("1", "Agent reasoning benchmark", "LLM agents with tool use and reasoning")
    assert keyword_score(entry, ["agent", "reasoning", "LLM", "CUDA"]) == 3.0
    assert keyword_score(entry, []) == 0.0


def test_taste_keywords_from_papers():
    records = [
        {"关键词": ["agent", "评测"], "方向": "Agent Harness"},
        {"关键词": ["agent", "推理"]},  # 去重
    ]
    kws = taste_keywords_from_papers(records)
    assert kws == ["agent", "评测", "Agent Harness", "推理"]


# ---------------------------------------------------------------------------
# TL;DR JSON 容错（单篇对象）
# ---------------------------------------------------------------------------

def test_parse_tldr_json_happy():
    parsed = parse_tldr_json('{"tldr": "T1", "reason": "R1"}')
    assert parsed == {"tldr": "T1", "reason": "R1"}


def test_parse_tldr_json_fenced_with_prose():
    parsed = parse_tldr_json('结果：\n```json\n{"tldr": "T", "reason": ""}\n```\n以上')
    assert parsed is not None and parsed["tldr"] == "T"


def test_parse_tldr_json_missing_fields_and_garbage():
    assert parse_tldr_json('{"tldr": "T"}') == {"tldr": "T", "reason": ""}  # 缺省补空
    assert parse_tldr_json("完全不是 JSON") is None
    assert parse_tldr_json('[{"tldr": "T"}]') is None  # 数组不是对象


# ---------------------------------------------------------------------------
# 收藏翻坟与今日归集
# ---------------------------------------------------------------------------

def test_pick_resurface_oldest_unread_first():
    records = [
        {"标题": "已读老", "已回顾": True, "收藏时间": _ms("2026-01-01")},
        {"标题": "未读老", "已回顾": False, "收藏时间": _ms("2026-02-01"),
         "链接": {"text": "t", "link": "https://a.com"}, "摘要": "老摘要"},
        {"标题": "未读新", "已回顾": False, "收藏时间": _ms("2026-07-01")},
        {"标题": "未读中", "收藏时间": _ms("2026-03-01")},
    ]
    picked = pick_resurface(records, n=2)
    assert [p["标题"] for p in picked] == ["未读老", "未读中"]  # 最老优先，已读跳过
    assert picked[0]["链接"] == "https://a.com" and picked[0]["摘要"] == "老摘要"


def test_collect_today_items_windows():
    data = {
        MEMO_TABLE: [
            {"事项": "后天前交表", "状态": "待办", "截止时间": NOW_MS + 2 * DAY},
            {"事项": "已逾期还款", "状态": "待办", "截止时间": NOW_MS - DAY},
            {"事项": "太远的事", "状态": "待办", "截止时间": NOW_MS + 5 * DAY},  # >72h 排除
            {"事项": "已完成的事", "状态": "已完成", "截止时间": NOW_MS + DAY},  # 非待办排除
            {"事项": "没时间的", "状态": "待办"},  # 无截止时间排除
        ],
        BOARD_TABLE: [
            {"公司": "字节", "岗位": "后端", "状态": "已投递", "投递时间": _ms("2026-07-15")},  # 15 天无进展
            {"公司": "美团", "状态": "offer", "下次跟进": _ms("2026-07-01")},  # 终态不跟
        ],
        COLLECT_TABLE: [{"标题": "吃灰收藏", "已回顾": False, "收藏时间": _ms("2026-01-01")}],
    }
    svc, _ = make_service(data=data)
    items = collect_today_items(svc.io, now=NOW)
    assert [m["事项"] for m in items.memos] == ["已逾期还款", "后天前交表"]  # 逾期在前
    assert items.memos[0]["逾期"] == "（已逾期）"
    assert len(items.followups) == 1 and items.followups[0]["公司"] == "字节"
    assert items.resurface[0]["标题"] == "吃灰收藏"


def test_collect_today_items_memo_sorted_by_deadline_ms_across_year():
    """备忘按截止时间毫秒值排序：MM-DD 字符串排序跨年会把 "01-01" 排到 "12-31" 前。"""
    now = time.mktime(datetime.date(2026, 12, 30).timetuple())
    now_ms = int(now * 1000)
    data = {MEMO_TABLE: [
        {"事项": "明年元旦的事", "状态": "待办", "截止时间": now_ms + 2 * DAY},  # 2027-01-01
        {"事项": "今年除夕的事", "状态": "待办", "截止时间": now_ms + DAY},      # 2026-12-31
        {"事项": "昨天逾期的事", "状态": "待办", "截止时间": now_ms - DAY},      # 2026-12-29
    ]}
    svc, _ = make_service(data=data)
    items = collect_today_items(svc.io, now=now)
    assert [m["事项"] for m in items.memos] == ["昨天逾期的事", "今年除夕的事", "明年元旦的事"]
    assert [m["截止"] for m in items.memos] == ["12-29", "12-31", "01-01"]


def test_build_today_message_sections():
    items = TodayItems(
        memos=[{"事项": "交表", "截止": "08-01", "逾期": ""}],
        followups=[{"公司": "字节", "岗位": "后端", "原因": "投递已 9 天无进展"}],
        resurface=[{"标题": "旧文", "链接": "https://a.com", "摘要": "摘"}],
    )
    text = build_today_message(items, now=NOW)
    assert "2026-07-30" in text and "生成于" in text  # 生成时间
    assert "备忘" in text and "交表" in text
    assert "求职跟进" in text and "字节" in text
    assert "翻坟" in text and "旧文" in text and "https://a.com" in text


def test_build_today_message_empty_todo_still_resurface():
    items = TodayItems(resurface=[{"标题": "旧文", "链接": "", "摘要": ""}])
    text = build_today_message(items, now=NOW)
    assert "今日无待办" in text and "旧文" in text  # 无待办也给一条翻坟


def test_build_today_message_all_empty():
    text = build_today_message(TodayItems(), now=NOW)
    assert "今日无待办" in text


def test_build_today_message_evening_variant():
    """晚间回顾档：🌙 标题；默认仍是 🌅 今日提醒。"""
    assert "🌙 晚间回顾" in build_today_message(TodayItems(), now=NOW, evening=True)
    assert "🌅 今日提醒" in build_today_message(TodayItems(), now=NOW)


# ---------------------------------------------------------------------------
# 周报聚合与 prompt
# ---------------------------------------------------------------------------

def test_collect_weekly_materials_7d_window():
    data = {
        PAPER_TABLE: [
            {"标题": "本周论文", "TL;DR": "T", "关键词": ["agent"], "精读时间": NOW_MS - 2 * DAY},
            {"标题": "上月论文", "精读时间": NOW_MS - 20 * DAY},  # 超窗排除
        ],
        COLLECT_TABLE: [
            {"标题": "本周收藏", "摘要": "S", "收藏时间": NOW_MS - DAY},
            {"标题": "老收藏", "收藏时间": NOW_MS - 10 * DAY},
        ],
        IDEA_TABLE: [
            {"内容": "新想法", "标签": ["评测"], "时间": NOW_MS - 3 * DAY},
            {"内容": "老想法", "时间": NOW_MS - 30 * DAY},
        ],
    }
    svc, _ = make_service(data=data)
    materials = collect_weekly_materials(svc.io, now=NOW)
    assert len(materials.papers) == 1 and "本周论文" in materials.papers[0] and "agent" in materials.papers[0]
    assert len(materials.collections) == 1 and "本周收藏" in materials.collections[0]
    assert len(materials.ideas) == 1 and "新想法" in materials.ideas[0] and "评测" in materials.ideas[0]


def test_build_weekly_prompt_fills_all():
    prompt = build_weekly_prompt(WeeklyMaterials(papers=["P"], collections=["C"], ideas=["I"]), now=NOW)
    assert "P" in prompt and "C" in prompt and "I" in prompt and "{{" not in prompt


def test_build_weekly_prompt_empty_placeholders():
    prompt = build_weekly_prompt(WeeklyMaterials(), now=NOW)
    assert "本周无精读" in prompt and "本周无收藏" in prompt and "本周无灵感" in prompt


# ---------------------------------------------------------------------------
# 日报全流程（fake fetcher/kimi/sender）
# ---------------------------------------------------------------------------

TLDR_JSON = '{"tldr": "用 harness 统一评测，pass 率 +12%", "reason": "正中你的方向"}'


def test_arxiv_digest_now_full_flow():
    fetcher = lambda days=2: [_entry("2607.00001", "Agent Harness 框架"), _entry("2607.00002", "推理加速")]
    svc, sender = make_service(kimi=FakeKimi(TLDR_JSON), fetcher=fetcher,
                               vector_index=FakeVectorIndex())
    text = svc.arxiv_digest_now()
    assert "Agent Harness 框架" in text and "生成于" in text
    assert len(sender.cards) == 1
    chat_id, card = sender.cards[0]
    assert chat_id == "oc_push"
    payload = str(card)
    assert "Agent Harness 框架" in payload and "arxiv.org/abs/2607.00001" in payload
    assert "pass 率 +12%" in payload and "正中你的方向" in payload  # TL;DR + 推荐理由
    assert "生成于" in payload and "扫描 2 篇" in payload
    # light 链逐篇做 TL;DR，prompt 含口味与摘要
    assert svc.kimi.calls[0]["heavy"] is False
    assert len(svc.kimi.calls) == 4  # 质量关评审 2 次 + TL;DR 2 次（均逐篇单调用）
    assert "摘要" in svc.kimi.calls[0]["messages"][0]["content"]


def test_arxiv_digest_rank_uses_vector_when_available():
    fetcher = lambda days=2: [_entry("1", "无关论文"), _entry("2", "相关论文")]
    svc, sender = make_service(kimi=FakeKimi(TLDR_JSON), fetcher=fetcher,
                               vector_index=FakeVectorIndex(hits=[("x", "t", 0.1)]))
    # 向量打分两篇同分（同 hits），按日期同天 → 原序稳定；换一个区分场景：
    # 向量索引空库 → 退关键词：口味含「相关」命中第二篇
    svc2, sender2 = make_service(
        kimi=FakeKimi(TLDR_JSON), fetcher=fetcher,
        vector_index=FakeVectorIndex(size=0),  # 库空 → 关键词兜底
        data={PAPER_TABLE: [{"关键词": ["相关"]}]},
    )
    text = svc2.arxiv_digest_now()
    assert text.index("相关论文") < text.index("无关论文")  # 关键词命中的排前


def test_arxiv_digest_tldr_fallback_on_garbage():
    svc, sender = make_service(kimi=FakeKimi("模型瞎说"),
                               fetcher=lambda days=2: [_entry("1", "某论文", abstract="A" * 200)])
    text = svc.arxiv_digest_now()
    assert "某论文" in text and "A" * 80 in text  # 摘要截断兜底


def test_arxiv_digest_empty_feed():
    svc, sender = make_service(fetcher=lambda days=2: [])
    text = svc.arxiv_digest_now()
    assert "没有抓到" in text
    assert sender.texts and "没有抓到" in sender.texts[0][1]


def test_arxiv_digest_fetch_failure_pushes_error():
    def boom(days=2):
        raise DigestError("arXiv API 请求失败：timeout")

    svc, sender = make_service(fetcher=boom)
    text = svc.arxiv_digest_now()
    assert "生成失败" in text and "timeout" in text
    assert sender.texts and "生成失败" in sender.texts[0][1]  # 错误也推送告知


# ---------------------------------------------------------------------------
# 推送历史跨天排除（抓取窗口 2 天：精选库已推 ∪ 论文库已精读不再进候选）
# ---------------------------------------------------------------------------

def test_arxiv_digest_excludes_push_history_cross_day():
    """昨天推过的（精选库）与已精读的（论文库）被跨天排除，只有全新论文进日报。"""
    fetcher = lambda days=2: [_entry("2607.00001", "昨天推过的 agent 论文"),
                              _entry("2607.00002", "已精读的 agent 论文"),
                              _entry("2607.00003", "全新的 agent 论文")]
    data = {
        PICK_TABLE: [{"arXivID": "2607.00001", "日期": NOW_MS - DAY, "来源": "arXiv日报"}],
        PAPER_TABLE: [{"arXivID": "2607.00002", "精读时间": NOW_MS - DAY}],
    }
    svc, sender = make_service(kimi=FakeKimi(TLDR_JSON), fetcher=fetcher, data=data)
    text = svc.arxiv_digest_now()
    assert "全新的 agent 论文" in text
    assert "昨天推过的" not in text and "已精读的" not in text
    assert "扫描 1 篇" in text  # 排除后的候选数
    rows = svc.io.records[PICK_TABLE]
    assert {r.fields["arXivID"] for r in rows} == {"2607.00001", "2607.00003"}  # 只归档新篇


def test_arxiv_digest_all_history_silent(caplog):
    """候选全部命中推送历史：只记日志不推送（与质量闸同级处理）。"""
    fetcher = lambda days=2: [_entry("2607.00001", "agent 论文")]
    data = {PICK_TABLE: [{"arXivID": "2607.00001", "日期": NOW_MS - DAY, "来源": "arXiv日报"}]}
    svc, sender = make_service(fetcher=fetcher, data=data)
    with caplog.at_level("INFO", logger="feishu_assistant.digest"):
        text = svc.arxiv_digest_now()
    assert "均已推送过或已精读" in text and "不打扰" in text
    assert sender.cards == [] and sender.texts == []
    assert any("推送历史" in r.message for r in caplog.records)


def test_arxiv_digest_vector_search_excludes_self():
    """日报向量打分传 exclude_arxiv_id：已入库论文不会自我命中拉满口味分。"""
    vec = FakeVectorIndex()
    svc, _ = make_service(kimi=FakeKimi(TLDR_JSON),
                          fetcher=lambda days=2: [_entry("2607.00009", "agent 论文")],
                          vector_index=vec)
    svc.arxiv_digest_now()
    assert vec.search_calls
    assert all(c["exclude_arxiv_id"] == "2607.00009" for c in vec.search_calls)


# ---------------------------------------------------------------------------
# 精选库归档（日报/周刊 top → 精选库，按 arXivID 全量查重跨天去重）
# ---------------------------------------------------------------------------

def test_build_pick_row_fields():
    row = build_pick_row(_entry("2607.00001", "某论文"), {"tldr": "T", "reason": "R"}, now=NOW)
    assert row["标题"] == "某论文"
    assert row["日期"] == NOW_MS and ms_to_date_local(row["日期"]) == "2026-07-30"
    assert row["链接"] == {"text": "某论文", "link": "https://arxiv.org/abs/2607.00001"}  # 超链接结构
    assert row["arXivID"] == "2607.00001"
    assert row["TL;DR"] == "T" and row["推荐理由"] == "R"
    assert row["状态"] == "未读" and row["来源"] == "arXiv日报"


def ms_to_date_local(ms: int) -> str:
    import time as _time
    return _time.strftime("%Y-%m-%d", _time.localtime(ms / 1000))


def test_existing_pick_ids():
    records = [
        {"日期": NOW_MS, "arXivID": "2607.00001"},
        {"日期": NOW_MS - 86400_000, "arXivID": "2607.00002"},  # 跨天同样进查重集
        {"日期": "非时间戳", "arXivID": [{"text": "段列表ID"}]},  # 段列表兼容
        {"日期": NOW_MS, "arXivID": ""},  # 空 ID 不进集合
    ]
    assert existing_pick_ids(records) == {"2607.00001", "2607.00002", "段列表ID"}


def test_archive_picks_dedup_cross_day(tmp_path):
    entries = [_entry("2607.00001", "论文A"), _entry("2607.00002", "论文B")]
    items = [{"tldr": "t1", "reason": "r1"}, {"tldr": "t2", "reason": "r2"}]
    svc, _ = make_service(data={PICK_TABLE: []})
    # 首次归档 2 篇
    assert svc._archive_picks(entries, items) == 2
    rows = svc.io.records[PICK_TABLE]
    assert len(rows) == 2 and rows[0].fields["状态"] == "未读"
    # 重跑（补发/重试/跨天同一篇再进 top）：按 arXivID 查重跳过，不产生重复行
    assert svc._archive_picks(entries, items) == 0
    assert len(svc.io.records[PICK_TABLE]) == 2


def test_archive_picks_same_batch_timestamp():
    """同一批归档的行共用同一 日期 毫秒值：「最新一批」按 max(日期) 分组的分组键。"""
    entries = [_entry("2607.00001", "论文A"), _entry("2607.00002", "论文B")]
    items = [{"tldr": "t1", "reason": "r1"}, {"tldr": "t2", "reason": "r2"}]
    svc, _ = make_service(data={PICK_TABLE: []})
    assert svc._archive_picks(entries, items) == 2
    dates = {r.fields["日期"] for r in svc.io.records[PICK_TABLE]}
    assert len(dates) == 1


def test_archive_picks_upgrade_uses_why():
    """周刊深评没有 reason 字段：升级推荐理由优先取 why，来源合并，不新增重复行。"""
    svc, _ = make_service(data={PICK_TABLE: [
        {"arXivID": "2601.00001", "日期": NOW_MS - 2 * DAY,
         "来源": "arXiv日报", "推荐理由": "日报一句话"},
    ]})
    entries = [_entry("2601.00001", "论文A")]
    items = [{"why": "周刊深评理由", "tldr": "t1"}]
    archived = svc._archive_picks(entries, items, source="每周精选", upgrade_on_dup=True)
    assert archived == 0  # 升级不新增
    rows = svc.io.records[PICK_TABLE]
    assert len(rows) == 1
    assert rows[0].fields["推荐理由"] == "周刊深评理由"  # 深评覆盖日报一句话
    assert rows[0].fields["来源"] == "arXiv日报+每周精选"


def test_archive_picks_tolerates_missing_table():
    """io 为 None 或表缺失（未补建）：归档静默跳过，不影响推送。"""
    svc = DigestService(Settings(), FakeKimi(), sender=FakeSender(), chat_id="oc")
    assert svc._archive_picks([_entry("1", "t")], [{"tldr": "", "reason": ""}]) == 0


def test_arxiv_digest_now_archives_and_marks_footer():
    fetcher = lambda days=2: [_entry("2607.00001", "Agent Harness 框架")]
    svc, sender = make_service(kimi=FakeKimi(TLDR_JSON), fetcher=fetcher,
                               vector_index=FakeVectorIndex(), data={PICK_TABLE: []})
    text = svc.arxiv_digest_now()
    assert "已归档精选库 1 篇" in text
    rows = svc.io.records[PICK_TABLE]
    assert len(rows) == 1 and rows[0].fields["arXivID"] == "2607.00001"
    assert rows[0].fields["TL;DR"] and rows[0].fields["推荐理由"]
    assert "已归档精选库" in str(sender.cards[0][1])  # 卡片 footer 同样提示


def test_push_retry_once_then_success():
    svc, sender = make_service(sender=FakeSender(fail_times=1))
    svc.arxiv_digest_now()
    assert sender.cards and sender.attempts == 2  # 第一次失败，重试成功


def test_push_retry_exhausted_logs_error(caplog):
    svc, sender = make_service(sender=FakeSender(fail_times=99))  # 一直失败
    with caplog.at_level("ERROR"):
        svc.arxiv_digest_now()
    assert any("重试一次" in r.message for r in caplog.records)
    # 卡片推不出去 → 纯文本兜底也尝试过
    assert sender.attempts >= 3


# ---------------------------------------------------------------------------
# 今日 / 周报服务流程
# ---------------------------------------------------------------------------

def test_today_digest_now_pushes_and_returns():
    data = {MEMO_TABLE: [{"事项": "交表", "状态": "待办", "截止时间": NOW_MS + DAY}]}
    svc, sender = make_service(data=data)
    text = svc.today_digest_now()
    assert "交表" in text and sender.texts and "交表" in sender.texts[0][1]


def test_evening_digest_now_pushes_recap():
    """晚间回顾：同今日提醒素材，🌙 标题，经 push.evening_time 的 cron 触发。"""
    data = {MEMO_TABLE: [{"事项": "交表", "状态": "待办", "截止时间": NOW_MS + DAY}]}
    svc, sender = make_service(data=data)
    text = svc.evening_digest_now()
    assert "🌙 晚间回顾" in text and "交表" in text
    assert sender.texts and "晚间回顾" in sender.texts[0][1]


def test_evening_digest_without_io():
    svc = DigestService(Settings(), FakeKimi(), sender=FakeSender(), chat_id="oc")
    text = svc.evening_digest_now()
    assert "生成失败" in text and "未初始化" in text


def test_today_digest_without_io():
    svc = DigestService(Settings(), FakeKimi(), sender=FakeSender(), chat_id="oc")
    text = svc.today_digest_now()
    assert "生成失败" in text and "未初始化" in text


def test_weekly_report_now_writes_doc_and_pushes(monkeypatch):
    data = {IDEA_TABLE: [{"内容": "做个评测", "时间": NOW_MS - DAY}]}
    docx = FakeDocx()
    svc, sender = make_service(kimi=FakeKimi("## 本周一句话\n围绕评测。"), data=data, docx=docx)
    import feishu_assistant.modules.digest as _digest_mod
    monkeypatch.setattr(_digest_mod.time, "time", lambda: NOW)  # 固定“今天”，防真实日期漂移出窗口
    text = svc.weekly_report_now()
    assert "组会周报已生成" in text and "https://feishu.cn/docx/" in text
    assert docx.docs and "本周一句话" in docx.docs[0][1]
    assert svc.kimi.calls[0]["heavy"] is True  # 周报走 heavy 链
    assert sender.texts and "docx" in sender.texts[0][1]
    # prompt 里带上了灵感材料
    assert "做个评测" in svc.kimi.calls[0]["messages"][0]["content"]


def test_cmd_today_returns_without_push():
    data = {MEMO_TABLE: [{"事项": "交表", "状态": "待办", "截止时间": NOW_MS + DAY}]}
    svc, sender = make_service(data=data)
    reply = svc.cmd_today()
    assert "交表" in reply and sender.texts == []  # 命令回显不重复推送


def test_cmd_arxiv_daily_ack_and_async(monkeypatch):
    svc, _ = make_service()
    ran = []
    svc._start_async = lambda label, job: ran.append((label, job))
    from feishu_assistant.modules.paper import MessageContext
    ack = svc.cmd_arxiv_daily(MessageContext(chat_id="oc_x"))
    assert "收到" in ack and ran and ran[0][0] == "arXiv 日报"


# ---------------------------------------------------------------------------
# 调度注册
# ---------------------------------------------------------------------------

def test_register_jobs_cron_triggers():
    scheduler = BackgroundScheduler()  # 不 start，纯注册
    svc, _ = make_service()
    jobs = register_jobs(scheduler, svc)
    assert jobs == ["arxiv_daily", "today_digest", "today_evening", "weekly_report", "weekly_picks"]
    by_id = {j.id: j for j in scheduler.get_jobs()}
    assert str(by_id["arxiv_daily"].trigger) == "cron[hour='8', minute='30']"
    assert str(by_id["today_digest"].trigger) == "cron[hour='9', minute='0']"
    assert str(by_id["today_evening"].trigger) == "cron[hour='22', minute='0']"  # 晚间回顾
    weekly = str(by_id["weekly_report"].trigger)
    assert "day_of_week='mon'" in weekly and "hour='9'" in weekly and "minute='0'" in weekly
    picks = str(by_id["weekly_picks"].trigger)
    assert "day_of_week='fri'" in picks and "hour='20'" in picks and "minute='0'" in picks
    # 调度器未 start，无需 shutdown（未启动的实例 shutdown 会抛 SchedulerNotRunningError）


def test_register_jobs_custom_time():
    """config.yaml push.daily_time / evening_time 经 main.py 传入后生效。"""
    scheduler = BackgroundScheduler()
    svc, _ = make_service()
    register_jobs(scheduler, svc, arxiv_time="07:45", evening_time="21:30",
                  weekly_day="wed", picks_day="sat", picks_time="21:30")
    by_id = {j.id: j for j in scheduler.get_jobs()}
    assert "hour='7'" in str(by_id["arxiv_daily"].trigger) and "minute='45'" in str(by_id["arxiv_daily"].trigger)
    assert "hour='21'" in str(by_id["today_evening"].trigger) and "minute='30'" in str(by_id["today_evening"].trigger)
    assert "day_of_week='wed'" in str(by_id["weekly_report"].trigger)
    picks = str(by_id["weekly_picks"].trigger)
    assert "day_of_week='sat'" in picks and "hour='21'" in picks and "minute='30'" in picks


# ---------------------------------------------------------------------------
# 真实模板齐全
# ---------------------------------------------------------------------------

def test_real_digest_templates_complete():
    arxiv_tpl = ARXIV_PROMPT_PATH.read_text(encoding="utf-8")
    assert "{{PAPER}}" in arxiv_tpl and "{{TASTE}}" in arxiv_tpl and "tldr" in arxiv_tpl
    weekly_tpl = WEEKLY_PROMPT_PATH.read_text(encoding="utf-8")
    for marker in ("{{DATE_RANGE}}", "{{PAPERS}}", "{{COLLECTIONS}}", "{{IDEAS}}", "价值点"):
        assert marker in weekly_tpl


class TestArxivRss:
    SAMPLE = """<?xml version='1.0' encoding='UTF-8'?>
<rss xmlns:dc="http://purl.org/dc/elements/1.1/" version="2.0">
  <channel>
    <title>cs.CL updates on arXiv.org</title>
    <pubDate>Thu, 30 Jul 2026 00:00:00 -0400</pubDate>
    <item>
      <title>Test Paper One</title>
      <link>https://arxiv.org/abs/2607.26060</link>
      <description>arXiv:2607.26060v1 Announce Type: new
Abstract: We study post-training of LLMs with GRPO.</description>
    </item>
    <item>
      <title>Test Paper Two</title>
      <link>https://arxiv.org/abs/2607.26061</link>
      <description>arXiv:2607.26061v2 Announce Type: replace
Abstract: Agent benchmark.</description>
    </item>
  </channel>
</rss>"""

    def test_parse_items(self):
        from feishu_assistant.modules.digest import parse_arxiv_rss
        entries = parse_arxiv_rss(self.SAMPLE)
        assert len(entries) == 2
        assert entries[0].arxiv_id == "2607.26060"
        assert entries[0].title == "Test Paper One"
        assert "GRPO" in entries[0].abstract
        assert entries[0].url == "https://arxiv.org/abs/2607.26060"
        assert entries[0].published == "2026-07-30"

    def test_parse_bad_xml(self):
        import pytest
        from feishu_assistant.modules.digest import parse_arxiv_rss, DigestError
        with pytest.raises(DigestError):
            parse_arxiv_rss("not xml at all")


class TestTasteKnobs:
    def test_recency_weight(self):
        from feishu_assistant.modules.digest import recency_weight
        now = 1_800_000_000.0
        assert recency_weight(now - 10 * 86400, now) == 1.0
        assert recency_weight(now - 60 * 86400, now) == 0.6
        assert recency_weight(now - 300 * 86400, now) == 0.3
        assert recency_weight(0, now) == 0.6

    def test_keyword_weights_decay(self):
        from feishu_assistant.modules.digest import taste_keyword_weights, keyword_score_weighted, ArxivEntry
        now = 1_800_000_000.0
        recs = [
            {"fields": {"关键词": ["老主题"], "精读时间": int((now - 200 * 86400) * 1000)}},
            {"fields": {"关键词": ["新主题"], "精读时间": int((now - 5 * 86400) * 1000)}},
        ]
        w = taste_keyword_weights(recs, now)
        assert w["新主题"] == 1.0 and w["老主题"] == 0.3
        e = ArxivEntry("1", "新主题 paper", "x", "a", "2026-07-31", "u")
        assert keyword_score_weighted(e, w) == 1.0

    def test_blocklist_lifecycle(self, tmp_path):
        from feishu_assistant.modules.digest import add_blocklist, remove_blocklist, is_blocked, ArxivEntry
        p = tmp_path / "bl.json"
        terms = add_blocklist("speech", p)
        assert terms == ["speech"]
        e = ArxivEntry("1", "Speech recognition survey", "x", "a", "2026", "u")
        assert is_blocked(e, terms)
        terms2, removed = remove_blocklist("speech", p)
        assert removed and terms2 == [] and not is_blocked(e, terms2)

    def test_classify_and_quota(self):
        from feishu_assistant.modules.digest import classify_direction, select_top_with_quota, ArxivEntry
        pt = ArxivEntry("1", "GRPO for LLM post-training", "rlhf", "a", "2026-07-31", "u")
        ag = ArxivEntry("2", "A new agent benchmark", "agents", "a", "2026-07-31", "u")
        ot = ArxivEntry("3", "Theory of mind", "x", "a", "2026-07-30", "u")
        assert classify_direction(pt) == "后训练"
        assert classify_direction(ag) == "Agent"
        assert classify_direction(ot) == "其他"
        ranked = (
            [(ArxivEntry(f"a{i}", f"agent harness benchmark {i}", "agent", "a", "2026-07-31", "u"), 10.0 - i) for i in range(4)]
            + [(ArxivEntry("p1", "GRPO post-training", "rlhf", "a", "2026-07-31", "u"), 9.5),
               (ArxivEntry("p2", "DPO alignment study", "dpo", "a", "2026-07-31", "u"), 6.5)]
        )
        top = select_top_with_quota(ranked, n=5, cap=3)
        dirs = [classify_direction(e) for e, _ in top]
        assert dirs.count("Agent") <= 3 and dirs.count("后训练") >= 2

    def test_quota_relax_when_short(self):
        from feishu_assistant.modules.digest import select_top_with_quota, ArxivEntry
        ranked = [(ArxivEntry(str(i), f"agent benchmark {i}", "agent", "a", "2026-07-31", "u"), 10.0 - i) for i in range(6)]
        top = select_top_with_quota(ranked, n=5, cap=3)
        assert len(top) == 5  # 单一方向也补齐到 n


class TestQualityGate:
    def test_parse_quality_json(self):
        from feishu_assistant.modules.digest import parse_quality_json
        assert parse_quality_json('{"score": 4, "risk": "单一基准"}') == {"score": 4, "risk": "单一基准"}
        assert parse_quality_json('{"score": 9, "risk": ""}')["score"] == 5  # 截断到 1-5
        assert parse_quality_json("not json") is None
        assert parse_quality_json('{"risk": "x"}') is None  # 缺 score

    def test_quality_gate_rescores(self):
        from feishu_assistant.modules.digest import DigestService, ArxivEntry, QUALITY_BONUS
        svc = DigestService.__new__(DigestService)
        reviews = {"a1": 5, "a2": 1, "a3": 3}
        svc._quality_review = lambda e: {"score": reviews[e.arxiv_id], "risk": "r" if e.arxiv_id == "a2" else ""}
        ranked = [
            (ArxivEntry("a1", "t1", "x", "au", "2026-07-31", "u"), 5.0),
            (ArxivEntry("a2", "t2", "x", "au", "2026-07-31", "u"), 9.0),
            (ArxivEntry("a3", "t3", "x", "au", "2026-07-30", "u"), 7.0),
        ]
        gated, risks = svc._quality_gate(ranked)
        # a1: 5+1*BONUS, a2: 9+0.2*BONUS, a3: 7+0.6*BONUS → a2 仍领先但 a1 追近；风险点只含 a2
        scores = {e.arxiv_id: s for e, s in gated}
        assert scores["a1"] == 5.0 + QUALITY_BONUS
        assert scores["a2"] == 9.0 + 0.2 * QUALITY_BONUS
        assert risks == {"a2": "r"}
        # 质量 1 分的 a2 与质量 5 分的 a1 分差被压缩
        assert (scores["a2"] - scores["a1"]) < (9.0 - 5.0)


# ---------------------------------------------------------------------------
# 权威分合并（口味批内归一 ×0.6 + 权威 ×0.4）
# ---------------------------------------------------------------------------

class TestMergeAuthority:
    def test_weights_normalization_and_reasons(self):
        from feishu_assistant.modules.digest import merge_authority
        e1, e2 = _entry("1", "高分"), _entry("2", "低分", published="2026-07-29")
        merged, reasons = merge_authority([(e1, 4.0), (e2, 2.0)], lambda e: (0.5, ["OpenAI"]))
        scores = {e.arxiv_id: s for e, s in merged}
        assert scores["1"] == pytest.approx(0.6 * 1.0 + 0.4 * 0.5)  # 口味归一 1.0 → 0.8
        assert scores["2"] == pytest.approx(0.6 * 0.5 + 0.4 * 0.5)  # 口味归一 0.5 → 0.5
        assert reasons == {"1": ["OpenAI"], "2": ["OpenAI"]}  # 原因全量带回供卡片标注
        assert [e.arxiv_id for e, _ in merged] == ["1", "2"]  # 按合并分降序

    def test_zero_max_taste_gives_zero_norm(self):
        from feishu_assistant.modules.digest import merge_authority
        merged, _ = merge_authority([(_entry("1", "x"), 0.0)], lambda e: (1.0, []))
        assert merged[0][1] == pytest.approx(0.4)  # 口味全 0 不归一，纯权威分

    def test_empty(self):
        from feishu_assistant.modules.digest import merge_authority
        assert merge_authority([], lambda e: (0.0, [])) == ([], {})


# ---------------------------------------------------------------------------
# 日报质量闸（合并分 ≥ QUALITY_THRESHOLD 才推；0 篇过线静默）
# ---------------------------------------------------------------------------

class TestDailyQualityThreshold:
    def test_gate_keeps_only_passing(self):
        # "agent 论文" 命中两个兴趣词（2.0），"量子计算" 零命中（0.0，权威 0.9 → 0.36 < 0.55）
        fetcher = lambda days=2: [_entry("1", "agent 论文"), _entry("2", "量子计算")]
        svc, sender = make_service(kimi=FakeKimi(TLDR_JSON), fetcher=fetcher)
        text = svc.arxiv_digest_now()
        assert "agent 论文" in text and "量子计算" not in text  # 低分被闸掉
        assert "测试机构" in text  # 过线论文带权威标注
        rows = svc.io.records[PICK_TABLE]
        assert len(rows) == 1 and rows[0].fields["arXivID"] == "1"  # 只归档被推送的

    def test_gate_silent_when_nothing_passes(self, caplog):
        fetcher = lambda days=2: [_entry("1", "量子计算"), _entry("2", "凝聚态物理")]
        svc, sender = make_service(fetcher=fetcher, authority_scorer=lambda e: (0.0, []))
        with caplog.at_level("INFO", logger="feishu_assistant.digest"):
            text = svc.arxiv_digest_now()
        assert "质量线" in text and "不打扰" in text
        assert sender.cards == [] and sender.texts == []  # 只记日志不推送
        assert any("静默不推送" in r.message for r in caplog.records)

    def test_gate_pushes_notice_when_silent_disabled(self, monkeypatch):
        from feishu_assistant.modules import digest
        monkeypatch.setattr(digest, "SILENT_WHEN_EMPTY", False)
        svc, sender = make_service(fetcher=lambda days=2: [_entry("1", "量子计算")],
                                   authority_scorer=lambda e: (0.0, []))
        text = svc.arxiv_digest_now()
        assert "质量线" in text
        assert sender.texts and "不打扰" in sender.texts[0][1]  # 非静默模式推说明

    def test_daily_top_n_is_3(self):
        fetcher = lambda days=2: [_entry(str(i), f"agent 论文 {i}") for i in range(6)]
        svc, sender = make_service(kimi=FakeKimi(TLDR_JSON), fetcher=fetcher)
        svc.arxiv_digest_now()
        card = sender.cards[0][1]
        assert "Top 3" in card["header"]["title"]["content"]  # 6 篇过线只取前 3


# ---------------------------------------------------------------------------
# 每周精选（7 天窗口严选 top 5，逐篇深评，来源=每周精选）
# ---------------------------------------------------------------------------

PICK_JSON = ('{"tldr": "统一评测框架，pass 率 +12%", "why": "覆盖你关注的 harness 场景，方法可复用",'
             ' "relation": "正中 Agent Harness 方向", "risk": "只在单一基准验证"}')


def test_parse_pick_review_json():
    from feishu_assistant.modules.digest import parse_pick_review_json
    parsed = parse_pick_review_json('{"tldr": "T", "why": "W", "relation": "R", "risk": "K"}')
    assert parsed == {"tldr": "T", "why": "W", "relation": "R", "risk": "K"}
    assert parse_pick_review_json('{"tldr": "T"}') == {"tldr": "T", "why": "", "relation": "", "risk": ""}
    assert parse_pick_review_json("完全不是 JSON") is None


def test_build_weekly_picks_card_structure():
    from feishu_assistant.modules.digest import build_weekly_picks_card
    items = [{"title": "某论文", "url": "https://arxiv.org/abs/1", "arxiv_id": "1",
              "tldr": "T", "why": "W", "relation": "R", "risk": "K",
              "authority": "Google DeepMind"}]
    card = build_weekly_picks_card(items, scanned=42, days=7, now=NOW, archived=1)
    assert card["header"]["title"]["content"] == "📌 本周精选（7.30 · 周末阅读）"
    payload = str(card)
    for piece in ("某论文", "arxiv.org/abs/1", "T", "📖 W", "🎯 R",
                  "🏛️ Google DeepMind", "⚠️ 风险：K",
                  "read_paper", "mark_read",  # 精读/已读按钮保留
                  "扫描 42 篇", "严选 1 篇", "已归档精选库 1 篇"):
        assert piece in payload


def test_weekly_picks_full_flow():
    fetch_calls = []

    def fetcher(days):
        fetch_calls.append(days)
        return [_entry("2607.00001", "Agent Harness 综述"), _entry("2607.00002", "RLHF 论文新算法")]

    svc, sender = make_service(kimi=FakeKimi(PICK_JSON), fetcher=fetcher, data={PICK_TABLE: []})
    text = svc.weekly_picks_now()
    assert fetch_calls == [7]  # 7 天窗口（fetcher 收到 days 参数）
    assert "本周精选" in text and "周末阅读" in text and "Agent Harness 综述" in text
    # 卡片结构与深评字段
    assert len(sender.cards) == 1
    chat_id, card = sender.cards[0]
    assert chat_id == "oc_push"
    payload = str(card)
    assert "📌 本周精选（" in payload
    assert "覆盖你关注的 harness 场景" in payload  # 为什么值得读
    assert "正中 Agent Harness 方向" in payload    # 与你方向
    assert "只在单一基准验证" in payload           # 风险点
    assert "测试机构" in payload                   # 权威标注
    assert "read_paper" in payload and "mark_read" in payload
    # light 链逐篇深评（每篇 1 次），prompt 注入显式兴趣
    assert len(svc.kimi.calls) == 2
    assert all(c["heavy"] is False for c in svc.kimi.calls)
    assert "论文、agent、LLM" in svc.kimi.calls[0]["messages"][0]["content"]
    # 精选库归档：来源/状态/方向/关键词/推荐理由（why 落入）
    rows = svc.io.records[PICK_TABLE]
    assert len(rows) == 2
    f0, f1 = rows[0].fields, rows[1].fields
    assert f0["来源"] == "每周精选" and f0["状态"] == "未读"
    assert f0["方向"] == "Agent" and f0["关键词"] == ["agent"]
    assert f0["推荐理由"] == "覆盖你关注的 harness 场景，方法可复用"
    assert f1["方向"] == "后训练" and f1["关键词"] == ["论文"]


def test_weekly_picks_recurses_daily_and_upgrades():
    """日报推过的不排除：周刊深评再策展，归档升级原记录（来源合并）而不是新增重复行。"""
    fetcher = lambda days=7: [_entry("2607.00001", "日报推过的 agent 论文"),
                              _entry("2607.00002", "全新的 RLHF 论文")]
    data = {PICK_TABLE: [{"arXivID": "2607.00001", "日期": NOW_MS - 2 * DAY,
                          "来源": "arXiv日报", "推荐理由": "日报一句话"}]}
    svc, sender = make_service(kimi=FakeKimi(PICK_JSON), fetcher=fetcher, data=data)
    text = svc.weekly_picks_now()
    assert "日报推过的 agent 论文" in text and "全新的 RLHF 论文" in text
    rows = svc.io.records[PICK_TABLE]
    assert len(rows) == 2  # 旧行升级 + 新篇入库，无重复行
    old = rows[0].fields
    assert old["推荐理由"] == "覆盖你关注的 harness 场景，方法可复用"  # 深评 why 覆盖日报一句话
    assert old["来源"] == "arXiv日报+每周精选"
    assert rows[1].fields["arXivID"] == "2607.00002" and rows[1].fields["来源"] == "每周精选"


def test_weekly_picks_excludes_curated_and_read():
    """已上过周刊的（含升级合并来源）与已精读的不进周刊候选。"""
    fetcher = lambda days=7: [_entry("2607.00001", "上周策展过的 agent 论文"),
                              _entry("2607.00002", "已精读的 agent 论文"),
                              _entry("2607.00003", "日报推过的 agent 新论文")]
    data = {
        PICK_TABLE: [
            {"arXivID": "2607.00001", "日期": NOW_MS - 6 * DAY, "来源": "arXiv日报+每周精选"},
            {"arXivID": "2607.00003", "日期": NOW_MS - DAY, "来源": "arXiv日报"},
        ],
        PAPER_TABLE: [{"arXivID": "2607.00002", "精读时间": NOW_MS - DAY}],
    }
    svc, sender = make_service(kimi=FakeKimi(PICK_JSON), fetcher=fetcher, data=data)
    text = svc.weekly_picks_now()
    assert "日报推过的 agent 新论文" in text
    assert "上周策展过的" not in text and "已精读的" not in text
    rows = svc.io.records[PICK_TABLE]
    assert len(rows) == 2  # 2607.00003 升级进旧行，无新增
    assert rows[1].fields["来源"] == "arXiv日报+每周精选"


def test_weekly_picks_review_fallback_on_garbage():
    svc, sender = make_service(kimi=FakeKimi("模型瞎说"),
                               fetcher=lambda days=7: [_entry("1", "agent 论文", abstract="B" * 200)])
    text = svc.weekly_picks_now()
    assert "agent 论文" in text and "B" * 80 in text  # 摘要截断兜底，卡片照推


def test_weekly_picks_silent_when_nothing_passes():
    svc, sender = make_service(fetcher=lambda days=7: [_entry("1", "量子计算")],
                               authority_scorer=lambda e: (0.0, []))
    text = svc.weekly_picks_now()
    assert "质量线" in text and "不打扰" in text
    assert sender.cards == [] and sender.texts == []


def test_weekly_picks_empty_feed_pushes_notice():
    svc, sender = make_service(fetcher=lambda days=7: [])
    text = svc.weekly_picks_now()
    assert "没有抓到" in text and sender.texts and "没有抓到" in sender.texts[0][1]


def test_pick_direction_and_interest_hits():
    from feishu_assistant.modules.digest import interest_hits, pick_direction
    assert pick_direction(_entry("1", "GRPO post-training 研究", "rlhf 摘要")) == "后训练"
    assert pick_direction(_entry("2", "multi-agent harness", "agents")) == "Agent"
    assert pick_direction(_entry("3", "量子计算")) == "未分类"  # 其他 → 未分类
    e = _entry("4", "post-training 论文", "rlhf")
    assert interest_hits(e, ["post-training", "agent"]) == ["post-training"]
    assert interest_hits(e, ["agent"]) == []


def test_weekly_picks_template_complete():
    from feishu_assistant.modules.digest import WEEKLY_PICKS_PROMPT_PATH
    tpl = WEEKLY_PICKS_PROMPT_PATH.read_text(encoding="utf-8")
    for marker in ("{{PAPER}}", "{{INTERESTS}}", "why", "relation", "risk"):
        assert marker in tpl


def test_archive_picks_upgrade_on_dup():
    """周刊查重命中时升级推荐理由（跨天命中也生效）；新记录仍按新增计。"""
    svc, _ = make_service(data={PICK_TABLE: [
        {"arXivID": "2601.00001", "日期": NOW_MS - 3 * DAY,  # 3 天前日报归档
         "来源": "arXiv日报", "推荐理由": "旧"},
    ]})
    entries = [_entry("2601.00001", "T"), _entry("2601.00002", "T2")]
    items = [{"reason": "深评理由一", "tldr": "t1"},
             {"reason": "深评理由二", "tldr": "t2"}]
    archived = svc._archive_picks(entries, items, source="每周精选", upgrade_on_dup=True)
    assert archived == 1  # 只有新篇入库
    rows = svc.io.records[PICK_TABLE]
    assert len(rows) == 2
    assert rows[0].fields["推荐理由"] == "深评理由一"  # 跨天命中升级，不产生重复行
    assert rows[0].fields["来源"] == "arXiv日报+每周精选"
    assert rows[0].fields["日期"] > NOW_MS - 3 * DAY  # 升级刷批次键：「精读 N」按 max(日期) 取批不错位
    assert rows[1].fields["arXivID"] == "2601.00002" and rows[1].fields["来源"] == "每周精选"


def test_daily_silent_on_weekend(monkeypatch):
    """周六/周日不推送：返回提示文案，fetcher 与推送都不触发。"""
    import feishu_assistant.modules.digest as dg
    from feishu_assistant.modules.digest import DigestService

    svc = DigestService.__new__(DigestService)
    called = {"fetch": 0, "push": 0}
    svc.fetcher = lambda days: called.__setitem__("fetch", called["fetch"] + 1) or []
    svc._push = lambda *a, **k: called.__setitem__("push", called["push"] + 1) or True
    svc.chat_id = "oc_x"
    monkeypatch.setattr(dg.time, "localtime",
                        lambda *a: dg.time.struct_time((2026, 8, 1, 8, 30, 0, 5, 213, 0)))  # 周六
    text = svc.arxiv_digest_now(respect_weekend=True)
    assert "周末不推送" in text and called == {"fetch": 0, "push": 0}


def test_daily_monday_uses_three_day_window(monkeypatch):
    """周一日报窗口放宽到 3 天（周末存量并入）。"""
    import feishu_assistant.modules.digest as dg
    from feishu_assistant.modules.digest import DigestService

    svc = DigestService.__new__(DigestService)
    seen = {"days": None}
    svc.fetcher = lambda days: seen.__setitem__("days", days) or []
    svc._push = lambda *a, **k: True
    svc.chat_id = "oc_x"
    monkeypatch.setattr(dg.time, "localtime",
                        lambda *a: dg.time.struct_time((2026, 8, 3, 8, 30, 0, 0, 216, 0)))  # 周一
    svc.arxiv_digest_now()
    assert seen["days"] == 3


def test_paper_actions_read_state_button():
    """已读的按钮显示 ✓ 已读；未读显示「已读」待点。"""
    from feishu_assistant.modules.digest import _paper_actions

    item = {"arxiv_id": "x", "title": "T", "status": "已读"}
    btn = _paper_actions(item)["actions"][1]
    assert btn["text"]["content"] == "✓ 已读"
    item["status"] = "未读"
    btn = _paper_actions(item)["actions"][1]
    assert btn["text"]["content"] == "已读"


def test_restyle_card_read():
    """mark_read 回刷：目标论文的已读按钮换成 ✓ 已读样式。"""
    from feishu_assistant.feishu.bot import FeishuBot
    from unittest.mock import MagicMock

    bot = FeishuBot.__new__(FeishuBot)
    card = {"elements": [
        {"tag": "action", "actions": [
            {"tag": "button", "text": {"tag": "plain_text", "content": "📖 精读这篇"},
             "type": "primary", "value": {"act": "read_paper", "arxiv_id": "x"}},
            {"tag": "button", "text": {"tag": "plain_text", "content": "已读"},
             "type": "default", "value": {"act": "mark_read", "arxiv_id": "x"}},
        ]},
    ]}
    resp = MagicMock()
    resp.success.return_value = True
    item = MagicMock()
    import json
    item.body.content = json.dumps(card, ensure_ascii=False)
    resp.data.items = [item]
    bot.client = MagicMock()
    bot.client.im.v1.message.get.return_value = resp
    evt = MagicMock()
    evt.event.context.open_message_id = "om_x"
    out = bot._restyle_card_read(evt, "x")
    btn = out["elements"][0]["actions"][1]
    assert btn["text"]["content"] == "✓ 已读"
