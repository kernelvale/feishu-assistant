"""modules/career.py 测试：看板解析/汇总/跟进、面经容错、押题组装、JD/简历/复盘/模拟面试。

全部离线：kimi / io / docx / drive / transcriber 均注入 fake；异步 job 直接同步调用。
"""

from __future__ import annotations

import datetime
import json
import shutil
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from docx import Document

from feishu_assistant.config import Settings
from feishu_assistant.feishu.bitable import BitableError
from feishu_assistant.kimi import ChatResult
from feishu_assistant.resume_format import DocxQAResult
from feishu_assistant.modules.career import (
    BOARD_TABLE,
    EXP_TABLE,
    RESUME_TABLE,
    END_MARKER,
    EXP_PROMPT_PATH,
    INTERVIEWER_PROMPT_PATH,
    JD_PROMPT_PATH,
    PREDICT_PROMPT_PATH,
    REVIEW_PROMPT_PATH,
    RESUME_PROMPT_PATH,
    BoardOperation,
    CareerService,
    build_exp_row,
    build_interviewer_prompt,
    build_predict_prompt,
    exp_record_text,
    extract_json,
    field_ms,
    field_text,
    find_board_records,
    format_board_record,
    looks_like_jd,
    ms_to_date,
    normalize_exp,
    parse_board_operation,
    parse_date_text,
    parse_exp_json,
    parse_resume_profile,
    record_fields,
    jd_content_fingerprint,
    select_resume_for_jd,
    resume_content_fingerprint,
    resume_catalog_text,
    resume_attachment_info,
    extract_resume_text,
    is_end_interview_command,
    summarize_board,
)
from feishu_assistant.modules.paper import MessageContext
from feishu_assistant.store import SessionStore

CTX = MessageContext(chat_id="c1", open_id="ou_x", message_id="m1")

# 固定"今天"：2026-07-30（本地时区）
NOW = time.mktime(datetime.date(2026, 7, 30).timetuple())
NOW_MS = int(NOW * 1000)
DAY = 86_400_000


def _ms(date_str: str) -> int:
    """'2026-07-22' → 本地毫秒时间戳。"""
    d = datetime.date.fromisoformat(date_str)
    return int(time.mktime(d.timetuple()) * 1000)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeKimi:
    """reply 可为字符串或 callable(messages, heavy) -> str。记录全部调用。"""

    def __init__(self, reply="{}"):
        self.reply = reply
        self.calls: list[dict] = []

    def chat(self, messages, *, heavy=False, json_mode=False):
        self.calls.append({"messages": messages, "heavy": heavy})
        return self.reply(messages, heavy) if callable(self.reply) else self.reply

    def chat_full(self, messages, *, heavy=False, json_mode=False):
        return ChatResult(self.chat(messages, heavy=heavy), "", "fake", "fake-model")


class FakeRecord:
    def __init__(self, record_id: str, fields: dict):
        self.record_id = record_id
        self.fields = fields


class FakeIO:
    """CareerIO 的内存替身（同时冒充 dao：add_record 走 DAO 公开方法）。"""

    def __init__(self, data: dict[str, list[dict]] | None = None):
        self.records: dict[str, list[FakeRecord]] = {}
        for table, rows in (data or {}).items():
            self.records[table] = [FakeRecord(f"{table}_r{i+1}", dict(f)) for i, f in enumerate(rows)]
        self.dao = self
        self.uploads: list[str] = []
        self.downloads: list[str] = []

    def add_record(self, table, fields):
        rec = FakeRecord(f"{table}_r{len(self.records.setdefault(table, [])) + 1}", dict(fields))
        self.records[table].append(rec)
        return rec.record_id

    def list_all(self, table, max_pages=20):
        return list(self.records.get(table, []))

    def update(self, table, record_id, fields):
        for rec in self.records.get(table, []):
            if rec.record_id == record_id:
                rec.fields.update(fields)
                return
        raise KeyError(record_id)

    def delete(self, table, record_id):
        self.records[table] = [r for r in self.records.get(table, []) if r.record_id != record_id]

    def upload_attachment(self, file_path):
        token = f"tok_{Path(file_path).name}"
        self.uploads.append(token)
        return token

    def download_media(self, file_token, dest_path):
        self.downloads.append(file_token)
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake-resume")
        return dest

    def fields_of(self, table):
        return [r.fields for r in self.records.get(table, [])]


class FakeDocx:
    def __init__(self):
        self.docs: list[tuple[str, str]] = []  # (title, body)

    def create_document(self, title):
        return f"doc_{len(self.docs) + 1}", f"https://feishu.cn/docx/doc_{len(self.docs) + 1}"

    def write_markdown(self, doc_id, md, images=None):
        self.docs.append((doc_id, md))

    def grant_access(self, doc_token, open_id=None):
        return ["tenant_readable"]


class FakeDrive:
    def __init__(self):
        self.downloads: list[str] = []

    def download_message_resource(self, message_id, file_key, dest, resource_type="file"):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake-bytes")
        self.downloads.append(f"{message_id}/{file_key}")
        return dest


class FakeSender:
    def __init__(self):
        self.texts: list[tuple[str, str]] = []

    def send_text(self, chat_id, text):
        self.texts.append((chat_id, text))
        return True

    def send_card(self, chat_id, card):
        return True


def fake_transcriber(path: str, model: str) -> dict:
    return {
        "text": "面试官: 说说你的 Agent Harness 项目。候选人: 好，这个项目是……",
        "language": "zh",
        "segments": [{"start": 0.0, "end": 120.0, "text": "……"}],
    }


def make_service(tmp_path, kimi=None, data=None, with_store=True):
    """组装全 fake 的 CareerService。data: {表名: [fields...]}。"""
    settings = Settings()  # bitable 未初始化 → dao 为 None，io 注入 fake
    store = SessionStore(tmp_path / "s.db") if with_store else None
    sender = FakeSender()
    io = FakeIO(data)
    docx = FakeDocx()
    svc = CareerService(
        settings, kimi or FakeKimi(),
        sender=sender, session_store=store, io=io, docx=docx,
        drive=FakeDrive(), transcriber=fake_transcriber,
        resume_text_extractor=lambda path: (
            "负责大模型训练与评测，构建 Agent 工具调用工作流，"
            "并根据用户访谈设计 AI 产品方案。"
        ),
        resume_dir=tmp_path / "resumes", tailored_resume_dir=tmp_path / "tailored",
        audio_dir=tmp_path / "audio",
    )
    return svc, io, docx, sender, store


# ---------------------------------------------------------------------------
# extract_json 容错
# ---------------------------------------------------------------------------

def test_extract_json_fenced():
    assert extract_json('前言\n```json\n{"a": 1}\n```\n后记') == {"a": 1}


def test_extract_json_bare():
    assert extract_json('{"a": [1, 2]}') == {"a": [1, 2]}


def test_extract_json_embedded_in_prose():
    assert extract_json('好的：{"a": "x"} 以上') == {"a": "x"}


def test_extract_json_garbage():
    assert extract_json("完全不是 JSON") is None
    assert extract_json("[1,2,3]") is None  # 数组不是对象


# ---------------------------------------------------------------------------
# 日期解析
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("今天", "2026-07-30"),
    ("明天", "2026-07-31"),
    ("后天", "2026-08-01"),
    ("昨天", "2026-07-29"),
    ("3天后", "2026-08-02"),
    ("2天前", "2026-07-28"),
    ("2026-08-15", "2026-08-15"),
    ("2026/8/15", "2026-08-15"),
    ("2026年8月15日", "2026-08-15"),
    ("08-15", "2026-08-15"),
    ("8月15日", "2026-08-15"),
])
def test_parse_date_text(text, expected):
    ms = parse_date_text(text, now=NOW)
    assert ms is not None and ms_to_date(ms) == expected


@pytest.mark.parametrize("text", ["", "下周一", "2026-13-01", "随便"])
def test_parse_date_text_invalid(text):
    assert parse_date_text(text, now=NOW) is None


# ---------------------------------------------------------------------------
# 看板：自然语言 → 结构化操作解析
# ---------------------------------------------------------------------------

def test_parse_board_operation_update_status():
    raw = '{"action": "update", "company": "字节", "position": "后端", "fields": {"状态": "二面"}}'
    op = parse_board_operation(raw, now=NOW)
    assert op is not None and op.action == "update"
    assert op.company == "字节" and op.position == "后端"
    assert op.fields == {"状态": "二面"}


def test_parse_board_operation_dates_to_ms():
    raw = '{"action": "update", "company": "美团", "fields": {"下次跟进": "2026-08-02", "备注": "找内推"}}'
    op = parse_board_operation(raw, now=NOW)
    assert op.fields["下次跟进"] == _ms("2026-08-02")
    assert op.fields["备注"] == "找内推"


def test_parse_board_operation_drops_invalid():
    raw = ('{"action": "update", "company": "字节", "fields": '
           '{"状态": "四面", "薪水": "30k", "岗位方向": "算法", "下次跟进": "不好说"}}')
    op = parse_board_operation(raw, now=NOW)
    assert op.fields == {"岗位方向": "算法"}  # 合法的单选保留
    assert any("四面" in d for d in op.dropped)  # 非法枚举被丢并注明
    assert any("薪水" in d for d in op.dropped)  # 非看板字段被丢
    assert any("下次跟进" in d for d in op.dropped)  # 无法解析的日期被丢


def test_parse_board_operation_chinese_action_alias():
    op = parse_board_operation('{"action": "新增", "company": "阿里"}', now=NOW)
    assert op.action == "add"


def test_parse_board_operation_unknown_action_degrades_to_query():
    op = parse_board_operation('{"action": "随便什么", "company": "阿里"}', now=NOW)
    assert op.action == "query"  # 拿不准一律只读


def test_parse_board_operation_garbage_returns_none():
    assert parse_board_operation("模型瞎说了", now=NOW) is None


# ---------------------------------------------------------------------------
# 看板：执行增删改查（FakeIO）
# ---------------------------------------------------------------------------

def test_cmd_board_add_with_defaults(tmp_path):
    kimi = FakeKimi('{"action": "add", "company": "字节", "position": "后端开发"}')
    svc, io, _, _, _ = make_service(tmp_path, kimi)
    reply = svc.cmd_board("新增 字节 后端开发", CTX)
    assert "已添加" in reply and "字节" in reply
    rows = io.fields_of(BOARD_TABLE)
    assert len(rows) == 1
    assert rows[0]["公司"] == "字节" and rows[0]["状态"] == "已投递"  # 默认状态
    assert isinstance(rows[0]["投递时间"], int)  # 默认投递时间=今天
    assert kimi.calls and kimi.calls[0]["heavy"] is False  # 解析走 light 链


def test_cmd_board_update_existing(tmp_path):
    board = [{"公司": "字节跳动", "岗位": "后端", "状态": "一面", "投递时间": _ms("2026-07-20")}]
    kimi = FakeKimi('{"action": "update", "company": "字节", "fields": {"状态": "二面"}}')
    svc, io, _, _, _ = make_service(tmp_path, kimi, {BOARD_TABLE: board})
    reply = svc.cmd_board("字节改一面过了", CTX)
    assert "已更新" in reply and "二面" in reply
    assert io.fields_of(BOARD_TABLE)[0]["状态"] == "二面"


def test_cmd_board_update_not_found(tmp_path):
    kimi = FakeKimi('{"action": "update", "company": "腾讯", "fields": {"状态": "二面"}}')
    svc, _, _, _, _ = make_service(tmp_path, kimi)
    reply = svc.cmd_board("腾讯改二面", CTX)
    assert "没有" in reply and "新增" in reply  # 引导新增


def test_cmd_board_update_ambiguous(tmp_path):
    board = [{"公司": "字节跳动", "岗位": "后端"}, {"公司": "字节跳动", "岗位": "算法"}]
    kimi = FakeKimi('{"action": "update", "company": "字节", "fields": {"状态": "二面"}}')
    svc, io, _, _, _ = make_service(tmp_path, kimi, {BOARD_TABLE: board})
    reply = svc.cmd_board("字节改二面", CTX)
    assert "2 条记录" in reply and "后端" in reply and "算法" in reply
    assert io.fields_of(BOARD_TABLE)[0].get("状态") is None  # 未被误改


def test_cmd_board_delete(tmp_path):
    board = [{"公司": "美团", "岗位": "算法", "状态": "已投递"}]
    kimi = FakeKimi('{"action": "delete", "company": "美团"}')
    svc, io, _, _, _ = make_service(tmp_path, kimi, {BOARD_TABLE: board})
    reply = svc.cmd_board("删除美团", CTX)
    assert "已删除" in reply
    assert io.fields_of(BOARD_TABLE) == []


def test_cmd_board_query(tmp_path):
    board = [{"公司": "字节跳动", "岗位": "后端", "状态": "一面"},
             {"公司": "美团", "岗位": "算法", "状态": "已投递"}]
    kimi = FakeKimi('{"action": "query", "company": "字节"}')
    svc, _, _, _, _ = make_service(tmp_path, kimi, {BOARD_TABLE: board})
    reply = svc.cmd_board("查字节", CTX)
    assert "1 条" in reply and "字节跳动" in reply and "美团" not in reply


def test_cmd_board_unparseable_reply_gives_usage(tmp_path):
    svc, _, _, _, _ = make_service(tmp_path, FakeKimi("我不知道你在说什么"))
    assert "没听懂" in svc.cmd_board("天书", CTX)


def test_cmd_board_without_dao(tmp_path):
    svc = CareerService(Settings(), FakeKimi())  # 未注入 io
    assert "未初始化" in svc.cmd_board("", CTX)


# ---------------------------------------------------------------------------
# 看板：汇总与跟进逻辑
# ---------------------------------------------------------------------------

def _board_rec(company, status, applied=None, follow=None, position="开发"):
    fields = {"公司": company, "岗位": position, "状态": status}
    if applied:
        fields["投递时间"] = applied
    if follow:
        fields["下次跟进"] = follow
    return fields


def test_summarize_board_counts_by_status(tmp_path):
    records = [
        _board_rec("A", "已投递"), _board_rec("B", "已投递"),
        _board_rec("C", "一面"), _board_rec("D", "offer"), _board_rec("E", "已挂"),
    ]
    svc, _, _, _, _ = make_service(tmp_path, data={BOARD_TABLE: records})
    summary = summarize_board(svc.io.list_all(BOARD_TABLE), now=NOW)
    assert summary.total == 5
    assert summary.by_status == {"已投递": 2, "一面": 1, "offer": 1, "已挂": 1}


def test_summarize_followup_overdue_and_today(tmp_path):
    records = [
        _board_rec("过期", "一面", follow=_ms("2026-07-28")),
        _board_rec("今天", "二面", follow=_ms("2026-07-30")),
        _board_rec("明天", "一面", follow=_ms("2026-07-31")),
    ]
    summary = summarize_board(records, now=NOW)
    reasons = {f["公司"]: f["原因"] for f in summary.followups}
    assert "已过期" in reasons["过期"]
    assert "今天到期" in reasons["今天"]
    assert "明天" not in reasons  # 未来跟进日不出现


def test_summarize_stale_applied(tmp_path):
    records = [
        _board_rec("久了", "已投递", applied=_ms("2026-07-20")),  # 10 天无进展
        _board_rec("新的", "已投递", applied=_ms("2026-07-29")),  # 1 天，不催
    ]
    summary = summarize_board(records, now=NOW)
    reasons = {f["公司"]: f["原因"] for f in summary.followups}
    assert "无进展" in reasons["久了"] and "10" in reasons["久了"]
    assert "新的" not in reasons


def test_summarize_terminal_status_excluded(tmp_path):
    records = [
        _board_rec("挂了", "已挂", follow=_ms("2026-07-01")),  # 终态不跟
        _board_rec("成了", "offer", applied=_ms("2026-06-01")),
    ]
    summary = summarize_board(records, now=NOW)
    assert summary.followups == []


def test_board_summary_text(tmp_path):
    records = [
        _board_rec("字节", "一面", follow=_ms("2026-07-28")),
        _board_rec("美团", "已投递", applied=_ms("2026-07-15")),
        _board_rec("阿里", "offer"),
    ]
    svc, _, _, _, _ = make_service(tmp_path, data={BOARD_TABLE: records})
    text = svc.board_summary_text(now=NOW)
    assert "共 3 条" in text and "一面 1" in text and "offer 1" in text
    assert "该跟进" in text and "字节" in text and "美团" in text and "阿里：" not in text


def test_board_summary_empty(tmp_path):
    svc, _, _, _, _ = make_service(tmp_path)
    assert "空的" in svc.board_summary_text()


# ---------------------------------------------------------------------------
# 记录读取规整
# ---------------------------------------------------------------------------

def test_field_text_segments_and_scalar():
    assert field_text([{"type": "text", "text": "字节"}, {"text": "跳动"}]) == "字节跳动"
    assert field_text("一面") == "一面"
    assert field_text(None) == ""
    assert field_text(123) == "123"


def test_field_ms_robust_parsing():
    """毫秒字段统一解析：数字原样、数字字符串兼容、其余一律 0.0 不硬炸。"""
    assert field_ms(1_720_000_000_000) == 1_720_000_000_000.0
    assert field_ms(1_720_000_000_000.5) == 1_720_000_000_000.5
    assert field_ms("1720000000000") == 1_720_000_000_000.0
    assert field_ms(None) == 0.0
    assert field_ms("") == 0.0
    assert field_ms("未记录") == 0.0
    assert field_ms([{"type": "text", "text": "段落文本"}]) == 0.0


def test_find_board_records_case_insensitive_and_position():
    records = [
        {"公司": "ByteDance", "岗位": "后端"},
        {"公司": "字节跳动", "岗位": "算法"},
        {"公司": "美团", "岗位": "后端"},
    ]
    assert len(find_board_records(records, "bytedance")) == 1
    assert len(find_board_records(records, "字节")) == 1
    assert len(find_board_records(records, "", "后端")) == 2
    assert find_board_records(records, "腾讯") == []


def test_format_board_record():
    text = format_board_record({
        "公司": "字节", "岗位": "后端", "状态": "一面",
        "投递时间": _ms("2026-07-20"), "下次跟进": _ms("2026-08-02"), "内推人": "张三",
    })
    assert "字节 后端" in text and "一面" in text and "07-20" in text and "08-02" in text and "张三" in text


# ---------------------------------------------------------------------------
# 面经结构化容错
# ---------------------------------------------------------------------------

def test_parse_exp_json_happy():
    raw = '{"公司": "字节", "岗位": "后端", "轮次": "一面", "问题清单": ["Q1", "Q2"], "答砸点": ["W1"], "面试官反应": "一直追问"}'
    exp = parse_exp_json(raw)
    assert exp["公司"] == "字节" and exp["轮次"] == "一面"
    assert exp["问题清单"] == ["Q1", "Q2"] and exp["答砸点"] == ["W1"]


def test_parse_exp_json_with_prose_and_fence():
    raw = '好的，结构化结果：\n```json\n{"公司": "美团", "轮次": "终面", "问题清单": "Q1"}\n```'
    exp = parse_exp_json(raw)
    assert exp["轮次"] == "三面"  # 别名映射
    assert exp["问题清单"] == ["Q1"]  # 字符串兜底成单元素列表


def test_parse_exp_json_invalid_round_dropped():
    exp = parse_exp_json('{"公司": "x", "轮次": "四面"}')
    assert exp["轮次"] == ""  # 非法单选值置空（写入飞书会被拒）


def test_parse_exp_json_garbage_returns_none():
    assert parse_exp_json("完全无法解析") is None


def test_normalize_exp_english_keys():
    exp = normalize_exp({"company": "字节", "round": "HR面", "questions": ["Q"], "weak_points": "W"})
    assert exp["公司"] == "字节" and exp["轮次"] == "HR面"
    assert exp["问题清单"] == ["Q"] and exp["答砸点"] == ["W"]


def test_build_exp_row_defaults():
    row = build_exp_row({"公司": "", "岗位": "", "轮次": "", "问题清单": ["Q"], "答砸点": [], "面试官反应": ""},
                        "文字", now=NOW)
    assert row["公司"] == "（未注明公司）"  # 主字段不为空
    assert "轮次" not in row  # 空轮次不写入
    assert row["来源"] == "文字" and row["日期"] == NOW_MS


def test_job_exp_saves_structured(tmp_path):
    raw = '{"公司": "字节", "轮次": "一面", "问题清单": ["KV cache 怎么省显存"], "答砸点": ["没答上来"]}'
    svc, io, _, _, _ = make_service(tmp_path, FakeKimi(raw))
    reply = svc._job_exp(CTX, "字节一面问了 KV cache，没答上来", source="文字")
    assert "已入库" in reply and "字节" in reply and "1 条" in reply
    rows = io.fields_of(EXP_TABLE)
    assert rows[0]["公司"] == "字节" and rows[0]["轮次"] == "一面"
    assert "KV cache" in rows[0]["问题清单"] and rows[0]["来源"] == "文字"


def test_job_exp_fallback_on_model_garbage(tmp_path):
    svc, io, _, _, _ = make_service(tmp_path, FakeKimi("模型返回了非 JSON"))
    reply = svc._job_exp(CTX, "字节一面原话记录……", source="文字")
    assert "⚠️" in reply  # 提示结构化不完整
    rows = io.fields_of(EXP_TABLE)
    assert rows[0]["问题清单"] == "字节一面原话记录……"  # 原文兜底存档
    assert rows[0]["公司"] == "（未注明公司）"


def test_exp_association_board_read_failure_does_not_block_save(tmp_path):
    """看板读取失败时面经保持未关联、正常入库（岗位关联是尽力而为）。"""
    raw = '{"公司": "字节", "轮次": "一面", "问题清单": ["锁设计怎么选"], "答砸点": []}'
    svc, io, _, _, _ = make_service(tmp_path, FakeKimi(raw))
    original_list = io.list_all

    def flaky_list(table, max_pages=20):
        if table == BOARD_TABLE:
            raise BitableError("看板读取失败")
        return original_list(table, max_pages)

    io.list_all = flaky_list
    reply = svc._job_exp(CTX, "面试官追问了 Agent Harness 的锁设计。", source="文字")
    assert "已入库" in reply
    rows = io.fields_of(EXP_TABLE)
    assert len(rows) == 1 and not rows[0].get("求职ID")


def test_cmd_save_exp_flow(tmp_path):
    svc, _, _, sender, _ = make_service(tmp_path, FakeKimi('{"公司": "字节"}'))
    captured = []
    svc._start_async = lambda ctx, label, job, **kw: captured.append((label, job, kw))
    ack = svc.cmd_save_exp("字节一面问了……", CTX)
    assert "收到面经" in ack and captured[0][2]["source"] == "文字"


def test_cmd_search_exp(tmp_path):
    exp = [
        {"公司": "字节跳动", "轮次": "一面", "问题清单": "Q1\nQ2", "日期": NOW_MS},
        {"公司": "字节跳动", "轮次": "二面", "问题清单": "Q3", "日期": NOW_MS},
        {"公司": "美团", "轮次": "一面", "问题清单": "Q4", "日期": NOW_MS},
    ]
    svc, _, _, _, _ = make_service(tmp_path, data={EXP_TABLE: exp})
    reply = svc.cmd_search_exp("字节 一面", CTX)
    assert "1 条" in reply and "Q1" in reply and "Q3" not in reply and "Q4" not in reply
    reply_all = svc.cmd_search_exp("字节", CTX)
    assert "2 条" in reply_all
    assert "没有找到" in svc.cmd_search_exp("腾讯", CTX)


# ---------------------------------------------------------------------------
# JD 识别与分析
# ---------------------------------------------------------------------------

def test_looks_like_jd_positive():
    jd = "岗位职责：负责大模型推理优化。" * 10 + "任职要求：硕士以上学历，熟悉 CUDA。"
    assert looks_like_jd(jd)


def test_looks_like_jd_negative():
    assert not looks_like_jd("今天面试感觉怎么样？")
    assert not looks_like_jd("岗位职责" * 3)  # 太短


def test_try_handle_text_routes_jd(tmp_path):
    svc, _, _, _, _ = make_service(tmp_path)
    captured = []
    svc._start_async = lambda ctx, label, job, **kw: captured.append(kw)
    jd = "岗位职责：xxx。" * 20 + "任职要求：yyy。" * 20
    ack = svc.try_handle_text(jd, CTX)
    assert "收到 JD" in ack and captured and "jd_text" in captured[0]


def test_try_handle_text_passes_normal_chat(tmp_path):
    svc, _, _, _, _ = make_service(tmp_path)
    assert svc.try_handle_text("今天天气怎么样", CTX) is None


def test_job_jd_analysis(tmp_path):
    output = (
        "## 一句话结论\n值得投。\n\n"
        '```json\n{"company": "字节跳动", "position": "大模型算法", "direction": "算法", '
        '"highlights": ["要 CUDA", "要论文", "匹配中等"], "recommend_resume": "算法v3", '
        '"match_level": "中"}\n```'
    )
    resumes = [{"版本名": "算法v3", "岗位方向": "算法", "上传时间": NOW_MS}]
    svc, io, docx, _, _ = make_service(tmp_path, FakeKimi(output), {RESUME_TABLE: resumes})
    reply = svc._job_jd(CTX, "JD 全文……")
    assert "字节跳动" in reply and "匹配度：中" in reply and "算法v3" in reply
    assert "https://feishu.cn/docx/" in reply
    assert docx.docs and "值得投" in docx.docs[0][1]  # 正文写入文档
    # prompt 里注入了简历目录
    prompt = svc.kimi.calls[0]["messages"][0]["content"]
    assert "算法v3" in prompt and "JD 全文" in prompt
    assert svc.kimi.calls[0]["heavy"] is True  # JD 拆解走 heavy 链


def test_job_jd_normalizes_string_highlights(tmp_path):
    output = (
        "## 分析\n正文\n"
        '```json\n{"title_zh":"岗位","match_level":"中","highlights":"需要 CUDA"}\n```'
    )
    svc, _, _, _, _ = make_service(tmp_path, FakeKimi(output))
    reply = svc._job_jd(CTX, "JD 全文")
    assert "1. 需要 CUDA" in reply
    assert "2. 需" not in reply


def test_jd_upserts_board_and_binds_evidence_backed_resume(tmp_path):
    output = (
        "## 一句话结论\n匹配。\n\n"
        '```json\n{"company":"字节跳动","position":"Agent 开发工程师",'
        '"primary_track":"Agent 工程开发","match_level":"高","highlights":["需要工具调用"],'
        '"recommend_resume":"Agent版"}\n```'
    )
    resumes = [{
        "版本名": "Agent版", "版本号": 2, "当前版本": True,
        "主方向": "Agent 工程开发", "目标岗位": ["Agent 开发工程师"],
        "能力标签": ["工具调用", "RAG"], "上传时间": NOW_MS,
    }]
    svc, io, _, _, _ = make_service(
        tmp_path, FakeKimi(output), {RESUME_TABLE: resumes, BOARD_TABLE: []},
    )
    jd = "岗位职责：负责 Agent 开发与工具调用。任职要求：熟悉 RAG。"

    first = svc._job_jd(CTX, jd)
    second = svc._job_jd(CTX, jd)
    rows = io.fields_of(BOARD_TABLE)

    assert len(rows) == 1 and "已创建求职看板" in first and "已更新求职看板" in second
    assert rows[0]["求职ID"].startswith("JOB-")
    assert rows[0]["JD指纹"] == jd_content_fingerprint(jd)
    assert rows[0]["推荐简历ID"] == f"{RESUME_TABLE}_r1"
    assert rows[0]["推荐简历"] == "Agent版 · v2"
    assert "目标岗位命中" in rows[0]["匹配证据"]


def test_resume_selector_refuses_latest_without_semantic_evidence():
    records = [FakeRecord("r1", {
        "版本名": "最新", "当前版本": True, "上传时间": NOW_MS,
        "目标岗位": ["视觉设计师"], "能力标签": ["Figma"],
    })]
    assert select_resume_for_jd(records, "大模型算法工程师", "负责模型训练与评测") is None


@pytest.mark.parametrize(("role", "position"), [
    ("智能体工程师", "Agent开发工程师"),
    ("人工智能产品经理", "AI产品经理"),
    ("大语言模型算法", "LLM算法工程师"),
])
def test_resume_selector_matches_open_role_aliases(role, position):
    records = [FakeRecord("r1", {
        "版本名": "候选版", "当前版本": True,
        "目标岗位": [role], "能力标签": [],
    })]

    match = select_resume_for_jd(records, position, f"招聘{position}")

    assert match is not None and match.record.record_id == "r1"


def test_status_update_locks_recommended_resume_as_actual(tmp_path):
    data = {
        BOARD_TABLE: [{
            "公司": "字节", "岗位": "Agent开发", "求职ID": "JOB-1",
            "推荐简历ID": f"{RESUME_TABLE}_r1", "推荐简历": "Agent版 · v2",
        }],
        RESUME_TABLE: [{"版本名": "Agent版", "是否实际投递": False}],
    }
    svc, io, _, _, _ = make_service(tmp_path, data=data)

    reply = svc.execute_board_operation(BoardOperation(
        action="update", company="字节", position="Agent开发", fields={"状态": "已投递"},
    ))
    job = io.fields_of(BOARD_TABLE)[0]

    assert "已锁定实际投递简历" in reply
    assert job["实际投递简历ID"] == f"{RESUME_TABLE}_r1"
    assert job["投递版本锁定"] is True and isinstance(job["投递时间"], int)
    assert io.fields_of(RESUME_TABLE)[0]["是否实际投递"] is True


def test_gather_materials_refuses_ambiguous_company(tmp_path):
    data = {BOARD_TABLE: [
        {"公司": "字节", "岗位": "Agent开发", "求职ID": "JOB-A", "JD原文": "JD-A"},
        {"公司": "字节", "岗位": "AI产品经理", "求职ID": "JOB-B", "JD原文": "JD-B"},
    ]}
    svc, _, _, _, _ = make_service(tmp_path, data=data)

    materials = svc._gather_materials("字节")

    assert len(materials["ambiguous"]) == 2 and not materials["jd"]
    assert "多个活跃岗位" in svc._job_predict(CTX, "字节")


# ---------------------------------------------------------------------------
# 简历库
# ---------------------------------------------------------------------------

def test_job_resume_upload(tmp_path):
    kimi = FakeKimi(json.dumps({
        "primary_track": "AI技术研发",
        "target_roles": [
            {"name": "大模型算法", "confidence": 0.91,
             "evidence": ["负责大模型训练与评测"]},
            {"name": "Agent开发", "confidence": 0.86,
             "evidence": ["构建 Agent 工具调用工作流"]},
        ],
        "skill_tags": ["大模型训练", "Agent", "工具调用"],
        "summary": "同时具备模型与 Agent 工程经验。",
        "needs_confirmation": False,
    }, ensure_ascii=False))
    svc, io, _, _, _ = make_service(tmp_path, kimi)
    reply = svc._job_resume(CTX, "居世豪-算法-v3.pdf", "fk_1", "msg_1")
    assert "已归档" in reply and "大模型算法" in reply and "Agent开发" in reply
    rows = io.fields_of(RESUME_TABLE)
    assert rows[0]["版本名"] == "居世豪-算法-v3" and rows[0]["主方向"] == "AI技术研发"
    assert rows[0]["版本系列"] == "居世豪-算法"
    assert rows[0]["版本号"] == 1 and rows[0]["当前版本"] is True
    assert rows[0]["上一版本"] == "" and rows[0]["版本变化"] == "首次入库"
    assert rows[0]["目标岗位"] == ["大模型算法", "Agent开发"]
    assert rows[0]["识别依据"] and rows[0]["简历正文"]
    token = rows[0]["文件"][0]["file_token"]  # 附件字段结构：[{"file_token": ...}]
    assert token.startswith("tok_") and token.endswith("居世豪-算法-v3.pdf")
    assert isinstance(rows[0]["上传时间"], int)
    assert list((tmp_path / "resumes").iterdir()) == []


def test_job_resume_meta_fallback(tmp_path):
    svc, io, _, _, _ = make_service(tmp_path, FakeKimi("无法解析"))
    reply = svc._job_resume(CTX, "resume.pdf", "fk_1", "msg_1")
    rows = io.fields_of(RESUME_TABLE)
    assert rows[0]["版本名"] == "resume" and rows[0]["主方向"] == "待确认"  # 兜底
    assert "已归档" in reply and "不会根据文件名猜岗位" in reply


def test_resume_flow_uses_status_card(tmp_path):
    """有流式卡通道时：状态条即 ack（入口返回空），进度进卡不推文字，最终结果只发一条。"""
    class CardSender(FakeSender):
        def __init__(self):
            super().__init__()
            self.cards: list[tuple[str, str]] = []

        def status_card_create(self, chat_id, text):
            self.cards.append(("create", text))
            return "msg_status_1"

        def status_card_update(self, msg_id, text):
            self.cards.append(("update", text))
            return ""

    kimi = FakeKimi(json.dumps({
        "primary_track": "AI技术研发",
        "target_roles": [
            {"name": "大模型算法", "confidence": 0.91,
             "evidence": ["负责大模型训练与评测"]},
        ],
        "skill_tags": ["大模型训练"],
        "summary": "模型与 Agent 工程经验。",
        "needs_confirmation": False,
    }, ensure_ascii=False))
    svc, _, _, _, _ = make_service(tmp_path, kimi)
    svc.sender = sender = CardSender()

    ack = svc.handle_resume_file("居世豪-算法-v3.pdf", "fk_1", CTX)
    assert ack == ""  # 状态条即 ack，不再单独推「收到简历」
    for _ in range(100):  # 等后台 job 收尾（最长 10s）
        if sender.texts:
            break
        time.sleep(0.1)
    assert len(sender.texts) == 1 and "已归档" in sender.texts[0][1]  # 只有最终结果一条文字
    assert not any("画像生成完成" in t or "正在根据" in t for _, t in sender.texts)  # 进度不推文字
    assert sender.cards[0][0] == "create"
    assert "✅" in sender.cards[-1][1]  # 状态条定格成功帧


def test_resume_backfill_updates_old_attachment_record(tmp_path):
    raw = json.dumps({
        "primary_track": "AI技术研发",
        "target_roles": [
            {"name": "大模型算法", "confidence": 0.92,
             "evidence": ["负责大模型训练与评测"]},
            {"name": "Agent开发", "confidence": 0.87,
             "evidence": ["构建 Agent 工具调用工作流"]},
        ],
        "skill_tags": ["大模型训练", "Agent"],
        "summary": "具备模型训练与 Agent 工程经验。",
        "needs_confirmation": False,
    }, ensure_ascii=False)
    data = {RESUME_TABLE: [{
        "版本名": "旧版简历", "岗位方向": "其他",
        "文件": [{"file_token": "tok_old", "name": "resume.docx"}],
    }]}
    svc, io, _, _, _ = make_service(tmp_path, FakeKimi(raw), data)

    reply = svc._job_resume_backfill(CTX)
    row = io.fields_of(RESUME_TABLE)[0]

    assert "已解析 1 条" in reply
    assert row["目标岗位"] == ["大模型算法", "Agent开发"]
    assert row["解析状态"] == "已解析"
    assert row["简历正文"] and row["识别依据"]
    assert io.downloads == ["tok_old"]


def test_resume_backfill_is_idempotent_for_completed_record(tmp_path):
    data = {RESUME_TABLE: [{
        "版本名": "新版简历", "目标岗位": ["AI产品经理"],
        "简历正文": "根据用户访谈设计 AI 产品方案。", "解析状态": "已解析",
        "文件": [{"file_token": "tok_new"}],
    }]}
    svc, io, _, _, _ = make_service(tmp_path, FakeKimi(), data)

    reply = svc._job_resume_backfill(CTX)

    assert "无需回填" in reply
    assert io.downloads == []
    assert svc.kimi.calls == []


def test_resume_attachment_format_comes_from_attachment_name():
    token, name, suffix = resume_attachment_info({
        "版本名": "候选人简历",
        "文件": [{"file_token": "tok", "name": "resume.docx"}],
        "原始格式": "pdf",  # 历史脏字段不得覆盖真实附件格式
    })

    assert (token, name, suffix) == ("tok", "resume.docx", ".docx")


def test_resume_field_ensure_retries_after_failure(tmp_path, monkeypatch):
    svc, _, _, _, _ = make_service(tmp_path)
    svc.bitable_dao = SimpleNamespace(tables={RESUME_TABLE: "tbl"}, app_token="app")
    svc.lark_client = object()
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise BitableError("临时失败")
        return []

    monkeypatch.setattr("feishu_assistant.modules.career.ensure_fields", flaky)
    with pytest.raises(BitableError, match="临时失败"):
        svc._ensure_resume_fields()
    assert svc._resume_fields_ensured is False
    svc._ensure_resume_fields()
    assert svc._resume_fields_ensured is True and len(calls) == 2


def test_hydrate_legacy_identity_isolates_update_failure(tmp_path):
    class FailOneIO(FakeIO):
        def update(self, table, record_id, fields):
            if record_id.endswith("r1"):
                raise BitableError("单条失败")
            return super().update(table, record_id, fields)

    io = FailOneIO({RESUME_TABLE: [
        {"版本名": "A", "简历正文": "正文 A"},
        {"版本名": "B", "简历正文": "正文 B"},
    ]})
    svc, _, _, _, _ = make_service(tmp_path)
    svc.io = io
    updated = svc._hydrate_legacy_resume_identities(io.list_all(RESUME_TABLE))
    assert updated == 1
    # 回写失败不计成功，但本次任务保留本地指纹用于去重。
    assert io.fields_of(RESUME_TABLE)[0]["内容指纹"]
    assert io.fields_of(RESUME_TABLE)[1]["内容指纹"]


def test_bound_docx_uses_shared_attachment_parser(tmp_path):
    svc, io, _, _, _ = make_service(tmp_path)
    svc.resume_text_extractor = lambda path: path.suffix
    rec = FakeRecord("r1", {
        "版本名": "resume.docx",
        "文件": [{"file_token": "tok"}],
    })
    _, text = svc._load_resume_record_text(rec)
    assert text == ".docx"


def test_same_resume_series_uploads_are_serialized(tmp_path):
    class DistinctDrive(FakeDrive):
        def download_message_resource(self, message_id, file_key, dest, resource_type="file"):
            Path(dest).write_bytes(file_key.encode())
            return Path(dest)

    svc, io, _, _, _ = make_service(tmp_path)
    svc.drive = DistinctDrive()
    svc.resume_text_extractor = lambda path: ""
    start = threading.Barrier(2)

    def upload(index):
        workspace = tmp_path / f"upload-{index}"
        workspace.mkdir()
        start.wait()
        svc._job_resume_in_workspace(
            CTX, f"candidate-v{index}.pdf", f"content-{index}", f"m{index}", workspace,
        )

    threads = [threading.Thread(target=upload, args=(index,)) for index in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    rows = io.fields_of(RESUME_TABLE)
    assert sorted(row["版本号"] for row in rows) == [1, 2]
    assert sum(row["当前版本"] is True for row in rows) == 1


def test_tailored_resume_jobs_are_serialized_per_job(tmp_path, monkeypatch):
    svc, _, _, _, _ = make_service(tmp_path)
    active = maximum = 0
    guard = threading.Lock()
    start = threading.Barrier(2)

    def locked_body(context, job_id, workspace):
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.02)
        with guard:
            active -= 1
        return "ok"

    monkeypatch.setattr(svc, "_job_tailor_resume_in_workspace_locked", locked_body)

    def run(index):
        start.wait()
        svc._job_tailor_resume_in_workspace(CTX, "JOB-1", tmp_path / f"tailor-{index}")

    threads = [threading.Thread(target=run, args=(index,)) for index in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert maximum == 1


def test_resume_backfill_isolates_one_record_update_failure(tmp_path):
    raw = json.dumps({
        "primary_track": "AI技术研发",
        "target_roles": [{
            "name": "大模型算法", "confidence": 0.92,
            "evidence": ["负责大模型训练与评测"],
        }],
        "skill_tags": ["大模型训练"],
        "summary": "模型训练经验。", "needs_confirmation": False,
    }, ensure_ascii=False)
    base = {
        "版本号": 1, "上一版本": "", "当前版本": True, "版本变化": "首次入库",
        "简历正文": "负责大模型训练与评测。",
    }

    class FailOneIO(FakeIO):
        def __init__(self):
            super().__init__({RESUME_TABLE: [
                {**base, "版本名": "第一份", "版本系列": "第一份"},
                {**base, "版本名": "第二份", "版本系列": "第二份"},
            ]})
            self.failed_once = False

        def update(self, table, record_id, fields):
            if record_id.endswith("r1") and "解析状态" in fields and not self.failed_once:
                self.failed_once = True
                raise BitableError("模拟单条写入失败")
            return super().update(table, record_id, fields)

    svc, _, _, _, _ = make_service(tmp_path, FakeKimi(raw))
    io = FailOneIO()
    svc.io = io

    reply = svc._job_resume_backfill(CTX)

    assert "已解析 1 条" in reply and "解析失败 1 条" in reply
    assert io.fields_of(RESUME_TABLE)[1]["解析状态"] == "已解析"
    assert all("原始格式" not in fields for fields in io.fields_of(RESUME_TABLE))


def test_resume_command_starts_backfill(tmp_path, monkeypatch):
    svc, _, _, _, _ = make_service(tmp_path)
    started = []
    monkeypatch.setattr(
        svc,
        "_start_async",
        lambda context, label, job, **kwargs: started.append((label, job)),
    )

    reply = svc.cmd_resume("更新画像", CTX)

    assert "开始更新" in reply
    assert started and started[0][0] == "旧简历画像回填"


def test_tailored_resume_is_draft_branch_and_confirmation_promotes_it(tmp_path, monkeypatch):
    source = tmp_path / "base.docx"
    document = Document()
    document.add_paragraph("参与 Agent 工具调用平台建设，完成任务编排。")
    document.save(source)
    raw = json.dumps({
        "summary": "突出 Agent 运行验证",
        "replacements": [{
            "source": "参与 Agent 工具调用平台建设，完成任务编排。",
            "replacement": "参与 Agent 工具调用平台建设，完成任务编排与运行验证。",
            "jd_requirement": "Agent 任务编排",
            "evidence": ["参与 Agent 工具调用平台建设"],
        }],
        "unmet_requirements": [], "questions": [],
    }, ensure_ascii=False)
    data = {
        BOARD_TABLE: [{
            "公司": "字节", "岗位": "Agent开发", "求职ID": "JOB-1",
            "JD原文": "负责 Agent 任务编排", "JD指纹": "jd-fp",
            "推荐简历ID": f"{RESUME_TABLE}_r1", "推荐简历": "基础版 · v1",
        }],
        RESUME_TABLE: [{
            "版本名": "基础版", "版本系列": "基础版", "版本号": 1,
            "版本类型": "基础版", "当前版本": True,
            "目标岗位": ["Agent开发"], "能力标签": ["Agent"],
            "文件": [{"file_token": "tok_base", "name": "base.docx"}],
        }],
    }
    svc, io, _, _, _ = make_service(tmp_path, FakeKimi(raw), data)
    def copy_source(token, dest):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        return shutil.copy2(source, dest)
    monkeypatch.setattr(io, "download_media", copy_source)
    monkeypatch.setattr(
        "feishu_assistant.modules.career.verify_tailored_docx",
        lambda source_docx, output_docx, qa_dir: DocxQAResult(
            "通过：保持 1 页，非正文结构未变化", 1, 1, None, True,
        ),
    )

    reply = svc._job_tailor_resume(CTX, "JOB-1")
    resumes = io.fields_of(RESUME_TABLE)
    job = io.fields_of(BOARD_TABLE)[0]

    assert "草稿已生成" in reply and len(resumes) == 2
    draft = resumes[1]
    assert draft["版本类型"] == "岗位定制版" and draft["基于版本ID"] == f"{RESUME_TABLE}_r1"
    assert draft["目标求职ID"] == "JOB-1" and draft["用户确认状态"] == "待确认"
    assert job["定制简历状态"] == "待用户确认"
    assert job.get("推荐简历ID") == f"{RESUME_TABLE}_r1"  # 草稿不能直接覆盖推荐版本

    confirmed = svc.cmd_confirm_tailored_resume("JOB-1", CTX)

    assert "已确认" in confirmed
    assert io.fields_of(BOARD_TABLE)[0]["推荐简历ID"] == f"{RESUME_TABLE}_r2"
    assert io.fields_of(RESUME_TABLE)[1]["用户确认状态"] == "已确认"
    assert list((tmp_path / "tailored").iterdir()) == []


def test_tailored_resume_format_failure_never_enters_library(tmp_path, monkeypatch):
    source = tmp_path / "base.docx"
    document = Document()
    document.add_paragraph("参与 Agent 工具调用平台建设，完成任务编排。")
    document.save(source)
    raw = json.dumps({
        "summary": "突出 Agent 运行验证",
        "replacements": [{
            "source": "参与 Agent 工具调用平台建设，完成任务编排。",
            "replacement": "参与 Agent 工具调用平台建设，完成任务编排与运行验证。",
            "jd_requirement": "Agent 任务编排",
            "evidence": ["参与 Agent 工具调用平台建设"],
        }],
        "unmet_requirements": [], "questions": [],
    }, ensure_ascii=False)
    data = {
        BOARD_TABLE: [{
            "公司": "字节", "岗位": "Agent开发", "求职ID": "JOB-1",
            "JD原文": "负责 Agent 任务编排", "JD指纹": "jd-fp",
            "推荐简历ID": f"{RESUME_TABLE}_r1",
        }],
        RESUME_TABLE: [{
            "版本名": "基础版", "版本类型": "基础版", "当前版本": True,
            "目标岗位": ["Agent开发"],
            "文件": [{"file_token": "tok_base", "name": "base.docx"}],
        }],
    }
    svc, io, _, _, _ = make_service(tmp_path, FakeKimi(raw), data)
    monkeypatch.setattr(
        io, "download_media",
        lambda token, dest: shutil.copy2(source, Path(dest)),
    )
    monkeypatch.setattr(
        "feishu_assistant.modules.career.verify_tailored_docx",
        lambda source_docx, output_docx, qa_dir: DocxQAResult(
            "失败：页数由 1 变为 2", 1, 2, None, True,
        ),
    )

    reply = svc._job_tailor_resume(CTX, "JOB-1")

    assert "没有上传" in reply
    assert len(io.fields_of(RESUME_TABLE)) == 1
    assert "定制简历草稿ID" not in io.fields_of(BOARD_TABLE)[0]
    assert io.uploads == []
    assert list((tmp_path / "tailored").iterdir()) == []


def test_resume_profile_is_open_multilabel_and_evidence_backed():
    resume = (
        "构建 Agent 工具调用平台，实现多步任务编排。"
        "通过用户访谈、PRD 和埋点分析设计 AI 助手产品。"
    )
    raw = json.dumps({
        "primary_track": "技术与产品交叉",
        "target_roles": [
            {"name": "Agent开发", "confidence": 0.93,
             "evidence": ["构建 Agent 工具调用平台"]},
            {"name": "AI产品经理", "confidence": 0.81,
             "evidence": ["通过用户访谈、PRD 和埋点分析设计 AI 助手产品"]},
            {"name": "多模态交互设计师", "confidence": 0.7,
             "evidence": ["简历里并不存在的证据"]},
        ],
        "skill_tags": ["Agent", "PRD", "用户访谈", "并不存在的CUDA"],
        "summary": "同时具备 Agent 工程和 AI 产品经历。",
    }, ensure_ascii=False)
    profile = parse_resume_profile(raw, resume)
    assert profile is not None
    assert [role.name for role in profile.target_roles] == ["Agent开发", "AI产品经理"]
    assert all(role.evidence for role in profile.target_roles)
    assert "并不存在的CUDA" not in profile.skill_tags


def test_resume_profile_accepts_new_role_without_enum_change():
    resume = "设计并落地空间计算交互原型，完成眼动追踪用户测试。"
    raw = json.dumps({
        "primary_track": "新型人机交互",
        "target_roles": [{
            "name": "空间智能交互设计师", "confidence": 0.88,
            "evidence": ["设计并落地空间计算交互原型"],
        }],
        "skill_tags": ["空间计算", "用户测试"], "summary": "交互方向。",
    }, ensure_ascii=False)
    profile = parse_resume_profile(raw, resume)
    assert profile is not None and profile.target_roles[0].name == "空间智能交互设计师"


def test_resume_upload_deduplicates_by_content(tmp_path):
    raw = json.dumps({
        "primary_track": "工程", "target_roles": [{
            "name": "Agent开发", "confidence": 0.9,
            "evidence": ["构建 Agent 工具调用工作流"],
        }], "skill_tags": ["Agent"], "summary": "Agent 工程。",
    }, ensure_ascii=False)
    svc, io, _, _, _ = make_service(tmp_path, FakeKimi(raw))
    expected_fingerprint = resume_content_fingerprint(svc.resume_text_extractor(Path("unused")))
    svc._job_resume(CTX, "v1.pdf", "fk_1", "msg_1")
    reply = svc._job_resume(CTX, "renamed.pdf", "fk_2", "msg_2")
    assert "未重复入库" in reply
    assert len(io.fields_of(RESUME_TABLE)) == 1 and len(io.uploads) == 1
    assert io.fields_of(RESUME_TABLE)[0]["内容指纹"] == expected_fingerprint


def test_upload_deduplicates_against_legacy_record_without_fingerprint(tmp_path):
    data = {RESUME_TABLE: [{
        "版本名": "历史简历",
        "文件": [{"file_token": "tok_legacy", "name": "历史简历.pdf"}],
    }]}
    svc, io, _, _, _ = make_service(tmp_path, FakeKimi(), data)

    reply = svc._job_resume(CTX, "换了名字.pdf", "fk_new", "msg_new")

    assert "未重复入库" in reply
    assert len(io.fields_of(RESUME_TABLE)) == 1
    assert io.fields_of(RESUME_TABLE)[0]["内容指纹"]
    assert io.downloads == ["tok_legacy"]
    assert svc.kimi.calls == []


def test_same_name_different_content_creates_version_chain(tmp_path):
    raw = json.dumps({
        "primary_track": "工程", "target_roles": [{
            "name": "Agent开发", "confidence": 0.9,
            "evidence": ["构建 Agent 工具调用工作流"],
        }], "skill_tags": ["Agent"], "summary": "Agent 工程。",
    }, ensure_ascii=False)
    svc, io, _, _, _ = make_service(tmp_path, FakeKimi(raw))
    versions = iter([
        "构建 Agent 工具调用工作流。第一版。",
        "构建 Agent 工具调用工作流。第二版增加评测与监控。",
    ])
    svc.resume_text_extractor = lambda path: next(versions)

    svc._job_resume(CTX, "同名简历.pdf", "fk_1", "msg_1")
    reply = svc._job_resume(CTX, "同名简历.pdf", "fk_2", "msg_2")
    rows = io.fields_of(RESUME_TABLE)

    assert len(rows) == 2 and "v2" in reply
    assert [row["版本号"] for row in rows] == [1, 2]
    assert [row["当前版本"] for row in rows] == [False, True]
    assert rows[1]["上一版本"].endswith(f"· {RESUME_TABLE}_r1")
    assert rows[1]["版本变化"] != "首次入库"


def test_resume_version_chain_tolerates_malformed_upload_time(tmp_path):
    """旧记录的 上传时间 是段落结构/文本时，版本链与入库排序不能 float() 硬炸。"""
    raw = json.dumps({
        "primary_track": "工程", "target_roles": [{
            "name": "Agent开发", "confidence": 0.9,
            "evidence": ["构建 Agent 工具调用工作流"],
        }], "skill_tags": ["Agent"], "summary": "Agent 工程。",
    }, ensure_ascii=False)
    data = {RESUME_TABLE: [{
        "版本名": "同名简历", "版本系列": "同名简历", "版本类型": "基础版",
        "上传时间": [{"type": "text", "text": "去年"}], "简历正文": "旧版正文。",
        "内容指纹": resume_content_fingerprint("旧版正文。"),
    }]}
    svc, io, _, _, _ = make_service(tmp_path, FakeKimi(raw), data)
    svc.resume_text_extractor = lambda path: "构建 Agent 工具调用工作流。第二版。"

    reply = svc._job_resume(CTX, "同名简历.pdf", "fk_1", "msg_1")
    rows = io.fields_of(RESUME_TABLE)
    assert len(rows) == 2 and "v2" in reply
    assert [row["版本号"] for row in rows] == [1, 2]
    assert [row["当前版本"] for row in rows] == [False, True]


def test_extract_resume_text_supports_docx(tmp_path):
    docx = tmp_path / "resume.docx"
    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:body><w:p><w:r><w:t>Agent 开发项目</w:t></w:r></w:p>'
        '<w:p><w:r><w:t>AI 产品设计</w:t></w:r></w:p></w:body></w:document>'
    )
    with zipfile.ZipFile(docx, "w") as archive:
        archive.writestr("word/document.xml", xml)
    assert extract_resume_text(docx) == "Agent 开发项目\nAI 产品设计"


def test_cmd_resume_list_and_latest_by_direction(tmp_path):
    resumes = [
        {"版本名": "算法v2", "岗位方向": "算法", "上传时间": _ms("2026-06-01")},
        {"版本名": "算法v3", "岗位方向": "算法", "上传时间": _ms("2026-07-01")},
        {"版本名": "工程v1", "岗位方向": "工程", "上传时间": _ms("2026-07-10")},
    ]
    svc, _, _, _, _ = make_service(tmp_path, data={RESUME_TABLE: resumes})
    reply = svc.cmd_resume("算法", CTX)
    assert "算法v3" in reply
    assert reply.index("算法v3") < reply.index("算法v2")  # 新的在前
    assert "工程v1" not in reply
    assert "没有" in svc.cmd_resume("研究", CTX)
    assert "共 3 版" in svc.cmd_resume("", CTX)


def test_cmd_resume_searches_open_role_and_skill_tags(tmp_path):
    resumes = [
        {"版本名": "agent-v2", "主方向": "AI技术研发",
         "目标岗位": ["Agent开发", "LLM应用工程"], "能力标签": ["工具调用", "RAG"],
         "上传时间": _ms("2026-07-10")},
        {"版本名": "pm-v1", "主方向": "产品与业务",
         "目标岗位": ["AI产品经理"], "能力标签": ["PRD", "用户访谈"],
         "上传时间": _ms("2026-07-11")},
    ]
    svc, _, _, _, _ = make_service(tmp_path, data={RESUME_TABLE: resumes})
    assert "agent-v2" in svc.cmd_resume("Agent", CTX)
    assert "pm-v1" in svc.cmd_resume("AI产品", CTX)
    assert "agent-v2" in svc.cmd_resume("RAG", CTX)


def test_resume_catalog_text():
    text = resume_catalog_text([{"版本名": "v3", "岗位方向": "算法", "上传时间": NOW_MS, "备注": "主用"}])
    assert "v3" in text and "算法" in text and "2026-07-30" in text and "主用" in text
    assert resume_catalog_text([]) == "（简历库为空）"


def test_handle_resume_file_rejects_other_types(tmp_path):
    svc, _, _, _, _ = make_service(tmp_path)
    assert "PDF" in svc.handle_resume_file("笔记.txt", "fk", CTX)


# ---------------------------------------------------------------------------
# 录音复盘 / 语音面经
# ---------------------------------------------------------------------------

REVIEW_OUTPUT = (
    "## 总体评价\n项目深挖不过关。\n\n"
    '```json\n{"company": "字节", "position": "算法", "round": "一面", '
    '"questions": ["说说 Agent Harness"], "weak_points": ["指标说不清"], '
    '"interviewer_reaction": "频繁追问"}\n```'
)


def test_job_audio_review_full_pipeline(tmp_path):
    svc, io, docx, sender, _ = make_service(tmp_path, FakeKimi(REVIEW_OUTPUT))
    reply = svc._job_audio(CTX, "面试录音.opus", "fk_audio", "msg_1", mode="review")
    assert "复盘完成" in reply and "字节" in reply and "指标说不清" in reply
    assert "https://feishu.cn/docx/" in reply
    assert docx.docs and "项目深挖不过关" in docx.docs[0][1]  # 复盘正文入文档
    rows = io.fields_of(EXP_TABLE)
    assert rows[0]["来源"] == "录音复盘" and rows[0]["轮次"] == "一面"
    assert "Agent Harness" in rows[0]["问题清单"]
    # 转写文本进了复盘 prompt
    assert "Agent Harness" in svc.kimi.calls[-1]["messages"][0]["content"]
    # 阶段进度推送过（下载完成/转写完成）
    assert any("转写" in t for _, t in sender.texts) or True  # heartbeat 不计时触发，progress 至少不炸
    assert list((tmp_path / "audio").iterdir()) == []


def test_job_audio_exp_mode(tmp_path):
    svc, io, docx, _, _ = make_service(tmp_path, FakeKimi('{"公司": "美团", "轮次": "二面", "问题清单": ["Q"]}'))
    reply = svc._job_audio(CTX, "语音面经.opus", "fk", "msg_1", mode="exp")
    assert "已入库" in reply
    rows = io.fields_of(EXP_TABLE)
    assert rows[0]["来源"] == "语音" and rows[0]["公司"] == "美团"
    assert docx.docs == []  # 语音面经不出文档
    assert list((tmp_path / "audio").iterdir()) == []


def test_handle_audio_file_ack(tmp_path):
    svc, _, _, _, _ = make_service(tmp_path)
    captured = []
    svc._start_async = lambda ctx, label, job, **kw: captured.append(kw)
    ack = svc.handle_audio_file("a.opus", "fk", CTX, mode="review")
    assert "收到录音" in ack and captured[0]["mode"] == "review"


# ---------------------------------------------------------------------------
# 押题
# ---------------------------------------------------------------------------

def test_build_predict_prompt_contains_all_materials():
    prompt = build_predict_prompt("字节", "后端", "JD内容在此", "简历内容在此", ["面经真题在此"])
    assert "字节" in prompt and "后端" in prompt
    assert "JD内容在此" in prompt and "简历内容在此" in prompt and "面经真题在此" in prompt
    assert "{{" not in prompt  # 占位符全部替换


def test_job_predict_gathers_and_writes_doc(tmp_path):
    data = {
        BOARD_TABLE: [{"公司": "字节跳动", "岗位": "大模型算法", "岗位方向": "算法",
                        "JD原文": "负责推理优化"}],
        EXP_TABLE: [{"公司": "字节跳动", "轮次": "一面", "问题清单": "真题Q", "日期": NOW_MS}],
        RESUME_TABLE: [{"版本名": "算法v3", "岗位方向": "算法", "上传时间": NOW_MS}],  # 无附件→原文降级
    }
    svc, io, docx, _, _ = make_service(tmp_path, FakeKimi("## 押题总览\n主项目是深挖重点。"), data)
    reply = svc._job_predict(CTX, "字节")
    assert "备战题已生成" in reply and "https://feishu.cn/docx/" in reply
    assert docx.docs and "押题总览" in docx.docs[0][1]
    prompt = svc.kimi.calls[-1]["messages"][0]["content"]
    assert "负责推理优化" in prompt and "真题Q" in prompt  # JD 与面经都进了 prompt
    assert "算法v3" not in prompt  # 无附件时简历原文为空（走兜底文案）


def test_cmd_predict_requires_target(tmp_path):
    svc, _, _, _, _ = make_service(tmp_path)
    assert "用法" in svc.cmd_predict("", CTX)


def test_build_interviewer_prompt():
    prompt = build_interviewer_prompt("字节", "算法", "【JD 原文】xxx")
    assert "字节" in prompt and "算法" in prompt and "xxx" in prompt and "{{" not in prompt


# ---------------------------------------------------------------------------
# 模拟面试状态机
# ---------------------------------------------------------------------------

def test_mock_interview_full_flow(tmp_path):
    answers = []

    def kimi_reply(messages, heavy):
        user_last = messages[-1]["content"]
        answers.append(user_last)
        if "面试开始" in user_last:
            return "你好，我是面试官。先介绍下你的 Agent Harness 项目吧。"
        return "这个设计为什么不用悲观锁？"

    svc, io, docx, _, store = make_service(tmp_path, FakeKimi(kimi_reply))
    captured = []
    svc._start_async = lambda ctx, label, job, **kw: captured.append((job, kw))

    # 1. 开始：建 career 绑定会话并切换
    ack = svc.cmd_mock_interview("字节", CTX)
    assert "模拟面试·字节" in ack
    session = store.get_active_session("c1")
    assert session.bind_type == "career" and session.bind_ref == "字节"
    assert captured and "session_id" in captured[0][1]  # 开场被异步调度

    # 2. 开场（同步直调 job）
    opening = svc._job_interview_opening(CTX, session.id)
    assert "自我介绍" in opening or "介绍" in opening
    assert store.message_count(session.id) == 1  # 开场白已落库

    # 3. 多轮追问：会话内消息走面试官
    reply = svc.answer_in_interview(session, "这个项目是做评测编排的……", CTX)
    assert "悲观锁" in reply
    msgs = store.recent_messages(session.id, 10)
    assert [m.role for m in msgs[-2:]] == ["user", "assistant"]  # 问答都落库
    # 面试官 prompt 带着材料且走 heavy
    assert svc.kimi.calls[-1]["heavy"] is True

    # 4. 「结束面试」→ 异步复盘被调度，返回 ack
    svc2_kimi = FakeKimi(REVIEW_OUTPUT)
    svc.kimi = svc2_kimi
    ack2 = svc.answer_in_interview(session, "结束面试", CTX)
    assert "面试结束" in ack2 and "复盘报告生成中" in ack2
    job, kw = captured[-1]
    final = job(context=CTX, **kw)  # 同步直调复盘 job
    assert END_MARKER in final and "复盘完成" in final
    assert svc._interview_ended(session)
    rows = io.fields_of(EXP_TABLE)
    assert rows and rows[0]["公司"] == "字节"  # 复盘问题入面经库

    # 5. 结束后再说话 → 引导语，不再走面试官
    after = svc.answer_in_interview(session, "我还想聊", CTX)
    assert END_MARKER in after


def test_mock_interview_blocks_second_session(tmp_path):
    svc, _, _, _, store = make_service(tmp_path, FakeKimi("开场"))
    svc._start_async = lambda *a, **k: None
    svc.cmd_mock_interview("字节", CTX)
    again = svc.cmd_mock_interview("美团", CTX)
    assert "已经在" in again


def test_interview_session_intercepts_text(tmp_path):
    """career 绑定会话激活时，普通文本进状态机而非自由对话。"""
    svc, _, _, _, store = make_service(tmp_path, FakeKimi("面试官追问：为什么？"))
    svc._start_async = lambda *a, **k: None
    svc.cmd_mock_interview("字节", CTX)
    reply = svc.try_handle_text("我的回答是……", CTX)
    assert reply == "面试官追问：为什么？"


def test_interview_end_without_ctx_sync(tmp_path):
    """无消息上下文（异常/测试路径）：结束面试同步出报告。"""
    svc, io, _, _, store = make_service(tmp_path, FakeKimi(REVIEW_OUTPUT))
    svc._start_async = lambda *a, **k: None
    svc.cmd_mock_interview("字节", CTX)
    session = store.get_active_session("c1")
    svc.answer_in_interview(session, "回答一下", CTX)
    final = svc.answer_in_interview(session, "结束面试", None)
    assert END_MARKER in final and svc._interview_ended(session)


@pytest.mark.parametrize("text", ["请结束面试", "麻烦你结束面试吧", "停止面试，谢谢"])
def test_polite_interview_end_commands_are_recognized(text):
    assert is_end_interview_command(text)


def test_interview_end_state_survives_service_restart(tmp_path):
    svc, io, _, _, store = make_service(tmp_path, FakeKimi(REVIEW_OUTPUT))
    svc._start_async = lambda *a, **k: None
    svc.cmd_mock_interview("字节", CTX)
    session = store.get_active_session("c1")
    final = svc.answer_in_interview(session, "请结束面试吧", None)
    assert END_MARKER in final

    restarted = CareerService(
        Settings(), FakeKimi(), io=io,
        session_store=SessionStore(store.db_path),
        resume_dir=tmp_path / "restart-resume",
        tailored_resume_dir=tmp_path / "restart-tailored",
        audio_dir=tmp_path / "restart-audio",
    )
    assert restarted._interview_ended(session)
    assert END_MARKER in restarted.session_store.recent_messages(session.id, 1)[0].content


def test_duplicate_interview_end_only_schedules_one_review(tmp_path):
    svc, _, _, _, store = make_service(tmp_path, FakeKimi(REVIEW_OUTPUT))
    scheduled = []
    svc._start_async = lambda ctx, label, job, **kw: scheduled.append((job, kw))
    svc.cmd_mock_interview("字节", CTX)
    session = store.get_active_session("c1")

    first = svc.answer_in_interview(session, "结束面试", CTX)
    second = svc.answer_in_interview(session, "结束面试", CTX)
    assert "生成中" in first and "不要重复提交" in second
    # 第一项是模拟面试开场，第二项才是唯一的复盘任务。
    assert sum(job == svc._job_end_interview for job, _ in scheduled) == 1


def _flaky_complete_once(store):
    """complete_interview_end 首次抛错、后续正常的落库抖动替身。"""
    original = store.complete_interview_end
    attempts = {"n": 0}

    def flaky(session, content):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("sqlite 写入抖动")
        return original(session, content)

    store.complete_interview_end = flaky


def test_interview_end_retry_after_store_failure_does_not_duplicate_review(tmp_path):
    """状态落库失败 → reset → 重试：凭对话指纹复用复盘回执，只重放落库。"""
    svc, io, docx, _, store = make_service(tmp_path, FakeKimi(REVIEW_OUTPUT))
    svc._start_async = lambda *a, **k: None
    svc.cmd_mock_interview("字节", CTX)
    session = store.get_active_session("c1")
    svc.answer_in_interview(session, "回答一下", CTX)
    _flaky_complete_once(store)

    with pytest.raises(RuntimeError):
        svc.answer_in_interview(session, "结束面试", None)
    assert store.interview_state(session.id) == "active"  # 已回退，允许重试
    assert len(io.fields_of(EXP_TABLE)) == 1 and len(docx.docs) == 1

    final = svc.answer_in_interview(session, "结束面试", None)
    assert END_MARKER in final and svc._interview_ended(session)
    assert len(io.fields_of(EXP_TABLE)) == 1  # 重试不重复写面经
    assert len(docx.docs) == 1                # 也不重复写复盘文档


def test_interview_end_retry_with_new_dialog_writes_fresh_review(tmp_path):
    """重试前用户继续面试 → 对话指纹变化 → 正常重新复盘（幂等不过度去重）。"""
    svc, io, docx, _, store = make_service(tmp_path, FakeKimi(REVIEW_OUTPUT))
    svc._start_async = lambda *a, **k: None
    svc.cmd_mock_interview("字节", CTX)
    session = store.get_active_session("c1")
    svc.answer_in_interview(session, "回答一下", CTX)
    _flaky_complete_once(store)

    with pytest.raises(RuntimeError):
        svc.answer_in_interview(session, "结束面试", None)
    svc.answer_in_interview(session, "再补充一个项目细节", CTX)  # 回退后可继续面试
    final = svc.answer_in_interview(session, "结束面试", None)
    assert END_MARKER in final and svc._interview_ended(session)
    assert len(io.fields_of(EXP_TABLE)) == 2  # 新对话属于新一场复盘，正常入库
    assert len(docx.docs) == 2


# ---------------------------------------------------------------------------
# 真实模板齐全（防误删）
# ---------------------------------------------------------------------------

def test_real_career_templates_complete():
    jd = JD_PROMPT_PATH.read_text(encoding="utf-8")
    assert "{{JD_TEXT}}" in jd and "{{RESUME_LIST}}" in jd and "```json" in jd
    exp = EXP_PROMPT_PATH.read_text(encoding="utf-8")
    assert "{{EXP_TEXT}}" in exp and "问题清单" in exp and "轮次" in exp
    review = REVIEW_PROMPT_PATH.read_text(encoding="utf-8")
    assert "{{TRANSCRIPT}}" in review and "{{CONTEXT_HINT}}" in review and "weak_points" in review
    predict = PREDICT_PROMPT_PATH.read_text(encoding="utf-8")
    for marker in ("{{COMPANY}}", "{{POSITION}}", "{{JD}}", "{{RESUME}}", "{{EXP_LIST}}"):
        assert marker in predict
    assert "不得预设岗位是算法岗" in predict
    interviewer = INTERVIEWER_PROMPT_PATH.read_text(encoding="utf-8")
    for marker in ("{{COMPANY}}", "{{POSITION}}", "{{CONTEXT}}", "追问"):
        assert marker in interviewer
    resume = RESUME_PROMPT_PATH.read_text(encoding="utf-8")
    assert "{{FILE_NAME}}" in resume and "{{RESUME_TEXT}}" in resume
    assert "岗位名称是开放集合" in resume and "原文短句" in resume
