"""modules/paper.py 测试：arXiv 解析、截断、模板拼接、元数据、bitable 行、卡片、追问状态机、store、并发幂等、断线/重启恢复。"""

from __future__ import annotations

import socket
import threading
import time

import httpx
import openai
import pytest
import requests

from feishu_assistant.feishu.bitable import BitableError
from feishu_assistant.feishu.drive import DriveError
from feishu_assistant.kimi import KimiError
from feishu_assistant.modules.paper import (
    TRANSIENT_RETRY_BACKOFFS,
    MessageContext,
    PaperPipelineError,
    PaperReadingService,
    _classify_error,
    _is_transient_error,
    build_error_card,
    build_paper_row,
    build_result_card,
    extract_metadata,
    looks_like_followup,
    normalize_paper_url,
    parse_arxiv_id,
    split_sections,
    truncate_markdown,
)
from feishu_assistant.modules.reading import ReadingError
from feishu_assistant.modules.status import StatusMessenger
from feishu_assistant.pdf import PdfParseError
from feishu_assistant.store import PaperStore

# ---- arXiv URL 解析 ----

@pytest.mark.parametrize("text,expected", [
    ("https://arxiv.org/abs/2603.03329", "2603.03329"),
    ("https://arxiv.org/pdf/2603.03329", "2603.03329"),
    ("https://arxiv.org/pdf/2603.03329.pdf", "2603.03329"),
    ("arxiv.org/abs/2603.03329v2", "2603.03329"),
    ("帮我精读 arXiv:2603.03329 这篇", "2603.03329"),
    ("精读 https://arxiv.org/abs/2504.13359 谢谢", "2504.13359"),
    ("http://arxiv.org/pdf/2504.13359v1.pdf", "2504.13359"),
])
def test_parse_arxiv_id_hit(text, expected):
    assert parse_arxiv_id(text) == expected


@pytest.mark.parametrize("text", [
    "今天天气怎么样",
    "https://example.com/paper.pdf",
    "精读这篇",
    "2603.03329",  # 裸数字串不带 arXiv 上下文，不识别（避免误伤）
])
def test_parse_arxiv_id_miss(text):
    assert parse_arxiv_id(text) is None


# ---- 截断 ----

def _make_md(sections: dict[str, int]) -> str:
    """按 {标题: 字符数} 生成测试 markdown。"""
    return "\n\n".join(f"## {title}\n" + "x" * size for title, size in sections.items())


def test_truncate_short_md_unchanged():
    md = _make_md({"Abstract": 100, "Method": 200})
    assert truncate_markdown(md, 60_000) == md


def test_truncate_drops_references_and_appendix():
    md = _make_md({"Abstract": 100, "Method": 500, "References": 9000, "Appendix": 9000})
    out = truncate_markdown(md, max_chars=1000)
    assert "References" not in out and "Appendix" not in out
    assert "Method" in out


def test_truncate_compresses_non_priority_sections():
    md = _make_md({
        "Abstract": 200, "Related Work": 5000, "Method": 3000, "Experiments": 3000,
    })
    out = truncate_markdown(md, max_chars=4000)
    assert "Abstract" in out and "Method" in out and "Experiments" in out
    assert len(out) <= 4000
    # related work 被压缩而非全保留
    assert out.count("x" * 5000) == 0


def test_truncate_hard_limit():
    md = _make_md({"Abstract": 5000, "Method": 20000, "Experiments": 20000, "Conclusion": 5000})
    out = truncate_markdown(md, max_chars=2000)
    assert len(out) <= 2000


def test_split_sections_no_headings():
    assert split_sections("plain text") == [("", "plain text")]


def test_real_templates_complete():
    """教学型精读模板必须齐全且含关键标记。"""
    from feishu_assistant.modules.reading import (
        PLAN_COMPILE_PATH,
        STYLE_PATH,
    )
    style = STYLE_PATH.read_text(encoding="utf-8")
    assert "教学原则" in style and "写作规则" in style
    assert "排版跟着信息关系走" in style
    assert "每句多少字" in style and "单句尽量不超过 50 字" not in style
    for section in ("问题模型", "案例按需使用", "讲法匹配对象", "反事实按需使用"):
        assert section in style

    skeleton = PLAN_COMPILE_PATH.read_text(encoding="utf-8")
    assert "{{SOURCE_CONTENT}}" in skeleton and "{{STYLE}}" in skeleton
    assert "{{GRAPH_CATALOG}}" in skeleton and "{{DEBT_REGISTRY}}" in skeleton
    assert "SIMILAR_PAPERS" not in skeleton
    for section in ("paper_type", "正文结构", "解释债务", "teaching_mode",
                    "primary_assertion_id", "debt_ids", "coverage", "purposes",
                    "running_example", "layers"):
        assert section in skeleton

    layer = (STYLE_PATH.parent / "reading_layer.md").read_text(encoding="utf-8")
    for marker in ("{{SKELETON}}", "{{FIGURE_HINTS}}", "{{CONTINUITY_CONTEXT}}",
                   "{{MUST_EXPLAIN}}", "{{SUCCESS_TEST}}", "{{LAYER_MODE}}"):
        assert marker in layer
    final_review = (STYLE_PATH.parent / "reading_final_review.md").read_text(encoding="utf-8")
    assert "{{READING_CONTENT}}" in final_review and "{{PAPER_CONTENT}}" in final_review
    assert "{{FORMULA_EVIDENCE}}" in final_review
    assert "{{EXPLANATION_CONTRACT}}" in final_review and "闭卷复述" in final_review


# ---- 元数据提取 ----

def test_extract_metadata_happy_path():
    output = '# 精读笔记\n正文内容\n\n```json\n{"title_zh": "自动马具", "year": "2026", "keywords": ["agent", "harness"]}\n```\n'
    body, meta = extract_metadata(output)
    assert body == "# 精读笔记\n正文内容"
    assert meta["title_zh"] == "自动马具" and meta["year"] == "2026"


def test_extract_metadata_missing_block():
    body, meta = extract_metadata("# 只有正文")
    assert meta == {} and "只有正文" in body


def test_extract_metadata_broken_json():
    body, meta = extract_metadata('正文\n```json\n{不是json}\n```')
    assert meta == {} and "正文" in body


# ---- bitable 行映射 ----

def test_build_paper_row_full():
    meta = {
        "title_zh": "自动马具", "authors": "A, B", "year": "2026",
        "keywords": ["agent", "harness", "代码"], "tldr": "一句话",
    }
    row = build_paper_row(meta, "https://feishu.cn/docx/abc", "https://arxiv.org/pdf/2603.03329",
                          "2603.03329", read_at=1_700_000_000)
    assert row["标题"] == "自动马具"
    assert row["作者"] == "A, B" and row["年份"] == "2026"
    assert row["关键词"] == ["agent", "harness", "代码"]  # 多选：字符串列表
    assert row["TL;DR"] == "一句话"
    assert row["arXivID"] == "2603.03329"
    assert row["精读时间"] == 1_700_000_000_000  # 毫秒时间戳
    assert row["文档链接"] == {"text": "精读：自动马具", "link": "https://feishu.cn/docx/abc"}
    assert row["PDF链接"] == {"text": "PDF", "link": "https://arxiv.org/pdf/2603.03329"}


def test_build_paper_row_minimal_meta():
    row = build_paper_row({}, "", "", "attach_1", read_at=1_700_000_000)
    assert row["标题"] == "未命名论文"
    assert "关键词" not in row and "文档链接" not in row and "PDF链接" not in row


# ---- 卡片 ----

def test_result_card_structure():
    meta = {"title_zh": "自动马具", "tldr": "一句话总结",
            "contributions": ["贡献1", "贡献2", "贡献3", "贡献4"]}
    card = build_result_card(meta, "https://feishu.cn/docx/abc", "2603.03329", archived=True)
    assert card["header"]["template"] == "green"
    assert "自动马具" in card["header"]["title"]["content"]
    text = str(card["elements"])
    assert "一句话总结" in text and "贡献3" in text and "贡献4" not in text  # 只取前 3 条
    assert "已归档" in text
    assert card["elements"][-1]["actions"][0]["url"] == "https://feishu.cn/docx/abc"


def test_result_card_not_archived_note():
    card = build_result_card({"title_zh": "t"}, "", "", archived=False)
    assert "归档未成功" in str(card["elements"])


def test_error_card_has_reason_and_suggestion():
    card = build_error_card("《x》", "解析失败", "换 PDF 重试")
    assert card["header"]["template"] == "red"
    assert "解析失败" in str(card["elements"]) and "换 PDF 重试" in str(card["elements"])


def test_reading_quality_error_has_actionable_user_message():
    reason, suggestion = _classify_error(ReadingError("发布前终审未通过：图表只复述数据"))
    assert "图表只复述数据" in reason
    assert "没有放行低质量文档" in suggestion


# ---- 追问状态机 ----

@pytest.mark.parametrize("text", [
    "第三章的 Thompson 采样怎么算的",
    "这张图里 Critic 是干嘛的",
    "作者为什么说老路走不通",
    "刚才那篇论文的实验设置再讲讲",
    "上面对比表第二行什么意思",
])
def test_looks_like_followup_hit(text):
    assert looks_like_followup(text)


@pytest.mark.parametrize("text", ["今天天气怎么样", "帮我写个周报", "你好"])
def test_looks_like_followup_miss(text):
    assert not looks_like_followup(text)


class _FakeKimi:
    def __init__(self):
        self.calls = []

    def chat(self, messages, *, heavy=False, json_mode=False):
        self.calls.append({"messages": messages, "heavy": heavy})
        return "追问回答"


class _FakeOutbox:
    """NotificationOutbox 替身：捕获 enqueue 的 (chat_id, text)。"""

    def __init__(self):
        self.items: list[tuple[str, str]] = []

    def enqueue(self, chat_id, text):
        self.items.append((chat_id, text))
        return len(self.items)


def _make_service(tmp_path, kimi, sender=None, outbox=None) -> PaperReadingService:
    from feishu_assistant.config import Settings
    return PaperReadingService(
        settings=Settings(), kimi=kimi, lark_client=None,  # 追问路径不碰飞书
        store=PaperStore(tmp_path / "test.db"), sender=sender, outbox=outbox,
    )


def test_followup_answers_with_full_context(tmp_path):
    kimi = _FakeKimi()
    svc = _make_service(tmp_path, kimi)
    ctx = MessageContext(chat_id="c1", open_id="ou_x")
    svc.store.save_paper("c1", "2603.03329", "自动马具", "全文 markdown 内容")

    reply = svc.answer_followup("第三章怎么算的", ctx)
    assert reply == "追问回答"
    call = kimi.calls[0]
    assert call["heavy"] is True  # 追问走 heavy 链
    assert "全文 markdown 内容" in call["messages"][1]["content"]
    assert "第三章怎么算的" in call["messages"][1]["content"]


def test_followup_no_paper_returns_none(tmp_path):
    svc = _make_service(tmp_path, _FakeKimi())
    assert svc.answer_followup("第三章怎么算的", MessageContext(chat_id="empty")) is None


def test_followup_unrelated_text_returns_none(tmp_path):
    svc = _make_service(tmp_path, _FakeKimi())
    ctx = MessageContext(chat_id="c1")
    svc.store.save_paper("c1", "x", "t", "md")
    assert svc.answer_followup("今天天气怎么样", ctx) is None


def test_followup_expired_paper_returns_none(tmp_path):
    svc = _make_service(tmp_path, _FakeKimi())
    ctx = MessageContext(chat_id="c1")
    svc.store.save_paper("c1", "x", "t", "md", created_at=time.time() - 48 * 3600)
    assert svc.answer_followup("第三章怎么算的", ctx) is None


def test_followup_cross_chat_isolation(tmp_path):
    svc = _make_service(tmp_path, _FakeKimi())
    svc.store.save_paper("c1", "x", "t", "md")
    assert svc.answer_followup("第三章怎么算的", MessageContext(chat_id="c2")) is None


# ---- PaperStore ----

def test_store_save_and_latest(tmp_path):
    store = PaperStore(tmp_path / "papers.db")
    store.save_paper("c1", "1111.11111", "旧论文", "旧全文", created_at=time.time() - 3600)
    row_id = store.save_paper("c1", "2222.22222", "新论文", "新全文", doc_url="http://d")
    assert row_id == 2
    latest = store.latest_paper("c1")
    assert latest is not None and latest.arxiv_id == "2222.22222"
    assert latest.markdown == "新全文" and latest.doc_url == "http://d"


def test_store_latest_with_time_window(tmp_path):
    store = PaperStore(tmp_path / "papers.db")
    store.save_paper("c1", "x", "t", "md", created_at=time.time() - 25 * 3600)
    assert store.latest_paper("c1", within_hours=24) is None
    assert store.latest_paper("c1", within_hours=48) is not None
    assert store.latest_paper("nobody") is None


# ---- 精读并发幂等（在飞任务锁 / 版本分配 / 链接查重键）----

def test_normalize_paper_url():
    assert normalize_paper_url(" https://openreview.net/forum?id=X. ") == \
        "https://openreview.net/forum?id=X"
    assert normalize_paper_url("https://a.com/b.pdf);") == "https://a.com/b.pdf"


class _FakeSender:
    """TaskCard 流式卡通道 + 结果/错误卡推送的 no-op sender。"""

    def __init__(self, msg_id: str = "om_status"):
        self.msg_id = msg_id
        self.cards: list[dict] = []
        self.texts: list[str] = []

    def send_text(self, chat_id, text):
        self.texts.append(text)
        return True

    def send_card(self, chat_id, card):
        self.cards.append(card)
        return True

    def send_text_id(self, chat_id, text):
        return self.msg_id

    def update_text(self, message_id, text):
        return ""

    def status_card_create(self, chat_id, text):
        return self.msg_id  # 句柄；"" 表示首发失败（静默降级路径用例）

    def rich_card_create(self, chat_id, card_json):
        return self.msg_id  # TaskCard 整卡通道句柄；"" 表示首发失败（静默降级）

    def status_card_update(self, handle, text):
        return ""

    def status_card_reactivate(self, handle):
        return True


class _FakeBitableDAO:
    """最小多维表格 fake：按表名分桶，字段等值查询，记录 add/update 调用。"""

    def __init__(self):
        self.tables: dict[str, list] = {}
        self.app_token = "t"
        self.added: list[tuple[str, dict]] = []
        self.updated: list[tuple[str, str, dict]] = []

    def _bucket(self, table: str) -> list:
        return self.tables.setdefault(table, [])

    def find_record_by_field(self, table, field, value):
        for rid, fields in self._bucket(table):
            if fields.get(field) == value:
                return rid
        return None

    def find_record_fields(self, table, field, value):
        for rid, fields in self._bucket(table):
            if fields.get(field) == value:
                return fields
        return None

    def add_record(self, table, row):
        rid = f"rec{len(self._bucket(table)) + 1}"
        self._bucket(table).append((rid, dict(row)))
        self.added.append((table, row))
        return rid

    def update_record(self, table, record_id, row):
        self.updated.append((table, record_id, row))
        bucket = self._bucket(table)
        for i, (rid, fields) in enumerate(bucket):
            if rid == record_id:
                merged = dict(fields)
                merged.update(row)
                bucket[i] = (rid, merged)


def test_start_async_inflight_blocks_concurrent_duplicate(tmp_path):
    """同一查重键：第二次触发返回 None（防连点）；任务结束释放后可再启动。"""
    svc = _make_service(tmp_path, _FakeKimi(), sender=_FakeSender())
    ctx = MessageContext(chat_id="c1", open_id="ou_x")
    gate = threading.Event()
    entered = threading.Event()

    def slow_job(context, **_):
        entered.set()
        gate.wait(5)
        raise PaperPipelineError("测试收尾")

    try:
        r1 = svc._start_async(ctx, label="arXiv:1", job=slow_job,
                              dedup_key="arxiv:1", arxiv_id="1")
        assert r1 == "om_status"  # 状态条发出去了（状态条即 ack）
        assert entered.wait(2)
        r2 = svc._start_async(ctx, label="arXiv:1", job=slow_job,
                              dedup_key="arxiv:1", arxiv_id="1")
        assert r2 is None  # 同一篇在飞：拦截
        r3 = svc._start_async(ctx, label="arXiv:2", job=slow_job,
                              dedup_key="arxiv:2", arxiv_id="2")
        assert r3 == "om_status"  # 不同篇：放行
    finally:
        gate.set()
    for _ in range(100):  # 等两个任务线程收尾（失败也释放查重键）
        if not svc._inflight:
            break
        time.sleep(0.02)
    assert not svc._inflight
    assert svc.sender.cards  # 两个任务都走了失败兜底推错误卡


def test_start_async_returns_empty_when_status_bar_failed(tmp_path):
    """状态条没发出去 → 返回 ""（调用方回退文字 ack），与「在飞拦截 None」区分。"""
    svc = _make_service(tmp_path, _FakeKimi(), sender=_FakeSender(msg_id=""))
    ctx = MessageContext(chat_id="c1")
    gate = threading.Event()

    def slow_job(context, **_):
        gate.wait(5)
        raise PaperPipelineError("x")

    try:
        r = svc._start_async(ctx, label="L", job=slow_job, dedup_key="k")
        assert r == ""
        assert svc._start_async(ctx, label="L", job=slow_job, dedup_key="k") is None
    finally:
        gate.set()


def test_try_handle_text_duplicate_returns_notice(tmp_path):
    """重复发同一链接/arXiv：回「正在精读中」提示而不是再跑一条管线。"""
    svc = _make_service(tmp_path, _FakeKimi(), sender=_FakeSender())
    ctx = MessageContext(chat_id="c1")
    svc._inflight.add("arxiv:2603.03329")  # 假装在飞
    reply = svc.try_handle_text("精读 https://arxiv.org/abs/2603.03329", ctx)
    assert reply is not None and "正在精读中" in reply
    svc._inflight.add("url:https://openreview.net/forum?id=X")
    reply = svc.try_handle_text("精读 https://openreview.net/forum?id=X", ctx)
    assert reply is not None and "正在精读中" in reply


def test_resolve_version_arxiv_and_url_key(tmp_path):
    """版本分配：arXiv 按 arXivID 字段（兼容老行），链接按「查重键」字段，复读 +1。"""
    svc = _make_service(tmp_path, _FakeKimi())
    dao = _FakeBitableDAO()
    svc.bitable_dao = dao
    assert svc._resolve_version("2603.03329", "") == 1  # 首读
    dao.add_record("论文库", {"arXivID": "2603.03329", "版本": "v2"})
    assert svc._resolve_version("2603.03329", "") == 3  # 复读 +1
    dao.add_record("论文库", {"查重键": "https://x", "版本": "v1"})
    assert svc._resolve_version("", "https://x") == 2  # 链接复读按查重键
    assert svc._resolve_version("", "https://y") == 1
    svc.bitable_dao = None
    assert svc._resolve_version("2603.03329", "") == 1  # 无多维表格


def test_archive_url_paper_dedup_updates_main_record(tmp_path):
    """无 arXiv ID 链接复读：按「查重键」更新主记录（版本+1、旧链接入历史），不重复建行。"""
    svc = _make_service(tmp_path, _FakeKimi())
    dao = _FakeBitableDAO()
    svc.bitable_dao = dao
    key = "https://openreview.net/forum?id=X"
    assert svc._archive({"title_zh": "论文", "_version": 1}, "http://doc1", "", "", dedup_key=key)
    assert len(dao.added) == 1 and dao.added[0][1]["查重键"] == key
    assert svc._archive({"title_zh": "论文", "_version": 2}, "http://doc2", "", "", dedup_key=key)
    assert len(dao.added) == 1  # 复读不重复建行
    assert len(dao.updated) == 1
    row = dao.updated[0][2]
    assert row["版本"] == "v2" and "v1: http://doc1" in row["历史版本"]


def test_archive_arxiv_dedup_unchanged(tmp_path):
    """arXiv 论文归档回归：按 arXivID 查重更新主记录，行内不写「查重键」。"""
    svc = _make_service(tmp_path, _FakeKimi())
    dao = _FakeBitableDAO()
    svc.bitable_dao = dao
    assert svc._archive({"title_zh": "论文", "_version": 1}, "http://doc1", "", "2603.03329")
    assert len(dao.added) == 1 and "查重键" not in dao.added[0][1]
    assert svc._archive({"title_zh": "论文", "_version": 2}, "http://doc2", "", "2603.03329")
    assert len(dao.added) == 1 and len(dao.updated) == 1
    assert "v1: http://doc1" in dao.updated[0][2]["历史版本"]


# ---- 断线/重启恢复：瞬时错误分类、退避重试、任务登记、启动恢复 ----

_OPENAI_REQ = httpx.Request("GET", "https://api.example.com/v1/chat")


@pytest.mark.parametrize("exc", [
    httpx.ConnectTimeout("connect timeout"),        # arXiv/链接下载层
    httpx.ConnectError("connection refused"),
    httpx.ReadTimeout("read timeout"),
    requests.exceptions.ConnectTimeout("t"),        # lark-oapi 传输层（真实事故类型）
    requests.exceptions.ConnectionError("c"),
    requests.exceptions.ReadTimeout("r"),
    socket.gaierror("DNS 解析失败"),
    ConnectionResetError("连接被重置"),
    TimeoutError("超时"),
    openai.APIConnectionError(request=_OPENAI_REQ),  # 大模型 SDK 层
    openai.APITimeoutError(request=_OPENAI_REQ),
])
def test_is_transient_error_bare_network_types(exc):
    assert _is_transient_error(exc)


@pytest.mark.parametrize("make_exc", [
    # 业务包装下面压着网络错误（KimiError/BitableError/DriveError 多层包装链）
    lambda: BitableError("论文库读取失败"),
    lambda: KimiError("模型调用失败"),
    lambda: DriveError("附件下载失败"),
])
def test_is_transient_error_wrapped_chain(make_exc):
    exc = make_exc()
    exc.__cause__ = httpx.ConnectError("connection refused")
    assert _is_transient_error(exc)


def test_is_transient_error_multi_level_chain():
    """两层包装：KimiError → openai 超时 → httpx 超时，沿链找到即瞬时。"""
    inner = openai.APITimeoutError(request=_OPENAI_REQ)
    inner.__cause__ = httpx.ReadTimeout("stream stalled")
    exc = KimiError("当前模型调用失败")
    exc.__cause__ = inner
    assert _is_transient_error(exc)


def test_is_transient_error_via_context_not_cause():
    """__context__（except 块里再抛）同样沿链判定。"""
    exc = BitableError("读取失败")
    exc.__context__ = requests.exceptions.ConnectionError("network down")
    assert _is_transient_error(exc)


@pytest.mark.parametrize("exc", [
    PaperPipelineError("PDF 下载结果异常"),      # 业务校验
    ReadingError("发布前终审未通过"),            # 验收失败
    KimiError("API key 无效"),                   # 凭证错误
    BitableError("code=1254045 无权限"),
    ValueError("参数错误"),
])
def test_is_transient_error_business_errors_never_transient(exc):
    assert not _is_transient_error(exc)


def test_is_transient_error_business_chain_stays_non_transient():
    """业务错误包装业务错误：链上没有网络类型，绝不误判。"""
    exc = PaperPipelineError("解析产物异常")
    exc.__cause__ = PdfParseError("PDF 无法解析")
    assert not _is_transient_error(exc)


class _FakeStatus:
    """记录阶段文案与定格调用的状态条替身（duck-type StatusMessenger）。"""

    def __init__(self):
        self.stages: list[str] = []
        self.stopped: list[tuple[bool, str]] = []

    def set_stage(self, stage: str) -> None:
        self.stages.append(stage)

    def stop(self, ok: bool, note: str = "") -> None:
        self.stopped.append((ok, note))


class _RecordingSender(_FakeSender):
    """在 _FakeSender 之上记录状态条帧写（定格帧断言用；文本/卡片两通道合一）。"""

    def __init__(self, msg_id: str = "om_status"):
        super().__init__(msg_id)
        self.updates: list[str] = []

    def update_text(self, message_id, text):
        self.updates.append(text)
        return ""

    def status_card_update(self, handle, text):
        self.updates.append(text)
        return ""


def _patch_sleep(monkeypatch) -> list[float]:
    """接管 paper 模块的 sleep：记录退避序列并立即返回。"""
    delays: list[float] = []
    monkeypatch.setattr(
        "feishu_assistant.modules.paper.time.sleep", lambda s: delays.append(s)
    )
    return delays


def _wait_task_status(svc: PaperReadingService, dedup_key: str, want: str,
                      timeout: float = 3.0) -> None:
    """轮询任务登记直到进入目标状态（后台线程收尾）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        task = svc.task_store.get(dedup_key)
        if task is not None and task.status == want:
            return
        time.sleep(0.02)
    task = svc.task_store.get(dedup_key)
    raise AssertionError(f"任务 {dedup_key} 未进入 {want}：当前 {task and task.status}")


def test_run_guarded_transient_retry_recovers(tmp_path, monkeypatch):
    """瞬时故障：按 2/5 分钟退避重跑同一 job，第三次成功后走正常完成路径。"""
    delays = _patch_sleep(monkeypatch)
    svc = _make_service(tmp_path, _FakeKimi(), sender=_FakeSender())
    svc.task_store.upsert_start("k1", "arxiv", "1", "L", "c1")
    ctx = MessageContext(chat_id="c1", open_id="ou_x")
    status = _FakeStatus()
    calls = []

    def flaky_job(context, **_):
        calls.append(1)
        if len(calls) <= 2:
            raise httpx.ConnectError("connection refused")
        return {"title_zh": "论文", "tldr": "x"}, "http://doc", True

    svc._run_guarded(ctx, "L", flaky_job, {}, status, "k1")

    assert len(calls) == 3  # 首次 + 两次重试
    assert delays == list(TRANSIENT_RETRY_BACKOFFS[:2])  # 2 分钟、5 分钟
    assert status.stages == [
        "🌐 网络中断，2 分钟后自动重试（第 1/3 次）",
        "🌐 网络中断，5 分钟后自动重试（第 2/3 次）",
    ]
    assert status.stopped == [(True, "精读完成，文档与详情见下方结果卡")]
    assert len(svc.sender.cards) == 1  # 结果卡而非错误卡
    assert svc.sender.cards[0]["header"]["template"] == "green"
    assert svc.task_store.get("k1").status == "done"


def test_run_guarded_transient_retry_exhausted_sends_error_card(tmp_path, monkeypatch):
    """瞬时故障重试耗尽（1+3 次尝试）：走原失败路径——定格 ❌ + 错误卡 + 标 failed。"""
    delays = _patch_sleep(monkeypatch)
    svc = _make_service(tmp_path, _FakeKimi(), sender=_FakeSender())
    svc.task_store.upsert_start("k2", "arxiv", "2", "L", "c1")
    ctx = MessageContext(chat_id="c1")
    status = _FakeStatus()
    calls = []

    def always_down(context, **_):
        calls.append(1)
        raise requests.exceptions.ConnectionError("network down")

    svc._run_guarded(ctx, "L", always_down, {}, status, "k2")

    assert len(calls) == 1 + len(TRANSIENT_RETRY_BACKOFFS)  # 首次 + 3 次重试
    assert delays == list(TRANSIENT_RETRY_BACKOFFS)  # 2/5/10 分钟全部等过
    assert len(status.stages) == 3 and "第 3/3 次" in status.stages[-1]
    assert status.stopped[-1][0] is False  # ❌ 定格
    assert len(svc.sender.cards) == 1
    assert svc.sender.cards[0]["header"]["template"] == "red"  # 错误卡
    assert svc.task_store.get("k2").status == "failed"  # failed：不再自动恢复


def test_run_guarded_non_transient_fails_immediately(tmp_path, monkeypatch):
    """业务错误绝不重试：不等不跑第二次，直接失败路径。"""
    delays = _patch_sleep(monkeypatch)
    svc = _make_service(tmp_path, _FakeKimi(), sender=_FakeSender())
    svc.task_store.upsert_start("k3", "arxiv", "3", "L", "c1")
    status = _FakeStatus()
    calls = []

    def bad_pdf(context, **_):
        calls.append(1)
        raise PaperPipelineError("PDF 下载结果异常")

    svc._run_guarded(MessageContext(chat_id="c1"), "L", bad_pdf, {}, status, "k3")

    assert len(calls) == 1 and delays == [] and status.stages == []
    assert status.stopped[-1][0] is False
    assert svc.sender.cards[0]["header"]["template"] == "red"
    assert svc.task_store.get("k3").status == "failed"


def test_run_guarded_retry_preserves_status_stop_semantics(tmp_path, monkeypatch):
    """真实 StatusMessenger：重试期 set_stage 后，stop 定格帧仍是最后一帧（✅）。"""
    _patch_sleep(monkeypatch)
    sender = _RecordingSender()
    svc = _make_service(tmp_path, _FakeKimi(), sender=sender)
    ctx = MessageContext(chat_id="c1")
    status = StatusMessenger(sender, "c1", "精读 L").start()
    calls = []

    def flaky_job(context, **_):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("x")
        return {"title_zh": "论文"}, "http://doc", True

    svc._run_guarded(ctx, "L", flaky_job, {}, status, "k")

    assert sender.updates, "定格帧应写进状态条"
    assert sender.updates[-1].startswith("✅")  # 定格帧是最后一帧且为成功帧
    assert "❌" not in sender.updates[-1]


# -- 事件通知外发箱：失败/首次重试/卡死事件必达（fake 捕获 enqueue）--

def test_run_guarded_failure_enqueues_notification(tmp_path):
    """失败除错误卡外补一条必达文本（断网时错误卡会丢，外发箱排队补发不丢）。"""
    outbox = _FakeOutbox()
    svc = _make_service(tmp_path, _FakeKimi(), sender=_FakeSender(), outbox=outbox)
    svc.task_store.upsert_start("kn1", "arxiv", "n1", "L", "c1")

    def bad(context, **_):
        raise PaperPipelineError("PDF 下载结果异常，请确认链接有效")

    svc._run_guarded(MessageContext(chat_id="c1"), "L", bad, {}, _FakeStatus(), "kn1")

    assert len(outbox.items) == 1
    chat_id, text = outbox.items[0]
    assert chat_id == "c1"
    assert text.startswith("❌ 精读L失败：") and "PDF 下载结果异常" in text
    assert "可重发链接重试" in text
    assert svc.sender.cards[0]["header"]["template"] == "red"  # 错误卡照发（不替代）


def test_run_guarded_first_transient_retry_enqueues_warning(tmp_path, monkeypatch):
    """首次进入瞬时退避发一条必达「正在自动重试」；多次重试只发一次，成功无 ❌。"""
    _patch_sleep(monkeypatch)
    outbox = _FakeOutbox()
    svc = _make_service(tmp_path, _FakeKimi(), sender=_FakeSender(), outbox=outbox)
    calls = []

    def flaky(context, **_):
        calls.append(1)
        if len(calls) <= 2:
            raise httpx.ConnectError("x")
        return {"title_zh": "论文"}, "http://doc", True

    svc._run_guarded(MessageContext(chat_id="c1"), "arXiv:1", flaky, {}, _FakeStatus(), "kn2")

    warnings = [t for _, t in outbox.items if t.startswith("⚠️")]
    assert warnings == ["⚠️ 精读arXiv:1遇到网络问题，正在自动重试，完成后告诉你"]
    assert not [t for _, t in outbox.items if t.startswith("❌")]


def test_run_guarded_transient_exhausted_enqueues_warning_then_failure(tmp_path, monkeypatch):
    """重试耗尽：先 ⚠️（自动重试中）后 ❌（失败）两条必达——断网事故全程知情。"""
    _patch_sleep(monkeypatch)
    outbox = _FakeOutbox()
    svc = _make_service(tmp_path, _FakeKimi(), sender=_FakeSender(), outbox=outbox)

    def always_down(context, **_):
        raise requests.exceptions.ConnectionError("network down")

    svc._run_guarded(MessageContext(chat_id="c1"), "L", always_down, {}, _FakeStatus(), "kn3")

    texts = [t for _, t in outbox.items]
    assert len(texts) == 2
    assert texts[0].startswith("⚠️ 精读L遇到网络问题")
    assert texts[1].startswith("❌ 精读L失败：") and "可重发链接重试" in texts[1]


def test_run_guarded_without_outbox_notifications_silently_off(tmp_path, monkeypatch):
    """无 outbox 注入：失败/重试路径行为不变（通知静默停用，回归旧行为）。"""
    delays = _patch_sleep(monkeypatch)
    svc = _make_service(tmp_path, _FakeKimi(), sender=_FakeSender())  # outbox=None

    def always_down(context, **_):
        raise requests.exceptions.ConnectionError("x")

    svc._run_guarded(MessageContext(chat_id="c1"), "L", always_down, {}, _FakeStatus(), "kn4")
    assert svc.sender.cards[0]["header"]["template"] == "red"
    assert delays == list(TRANSIENT_RETRY_BACKOFFS)


def test_start_async_wires_stuck_alert_to_outbox(tmp_path):
    """状态条看门狗接线：有 outbox → alert_fn 注入 StatusMessenger，回调即入队。"""
    outbox = _FakeOutbox()
    svc = _make_service(tmp_path, _FakeKimi(), sender=_FakeSender(), outbox=outbox)
    ctx = MessageContext(chat_id="c1")
    gate = threading.Event()

    def slow_ok(context, **_):
        gate.wait(5)
        return {"title_zh": "论文"}, "http://doc", True

    try:
        svc._start_async(ctx, label="arXiv:w1", job=slow_ok,
                         dedup_key="arxiv:w1", arxiv_id="w1")
        assert ctx.status.alert_fn is not None
        ctx.status.alert_fn("⏰ 精读 arXiv:w1 已 5 分钟无进展，疑似卡死")
        assert outbox.items == [("c1", "⏰ 精读 arXiv:w1 已 5 分钟无进展，疑似卡死")]
    finally:
        gate.set()
    _wait_task_status(svc, "arxiv:w1", "done")


def test_start_async_without_outbox_disables_stuck_alert(tmp_path):
    """无 outbox：alert_fn 为 None（看门狗停用，任务照常）。"""
    svc = _make_service(tmp_path, _FakeKimi(), sender=_FakeSender())
    ctx = MessageContext(chat_id="c1")
    gate = threading.Event()

    def slow_ok(context, **_):
        gate.wait(5)
        return {"title_zh": "论文"}, "http://doc", True

    try:
        svc._start_async(ctx, label="arXiv:w2", job=slow_ok,
                         dedup_key="arxiv:w2", arxiv_id="w2")
        assert ctx.status.alert_fn is None
    finally:
        gate.set()
    _wait_task_status(svc, "arxiv:w2", "done")


# -- durable 任务登记：_start_async 登记 → _run_guarded 回写 --

def test_start_async_registers_running_then_marks_done(tmp_path):
    """端到端：启动即登记 running（含上下文），任务成功结束回写 done。"""
    svc = _make_service(tmp_path, _FakeKimi(), sender=_FakeSender())
    ctx = MessageContext(chat_id="c1", open_id="ou_x", message_id="om_1")
    gate = threading.Event()
    entered = threading.Event()

    def slow_ok(context, **_):
        entered.set()
        gate.wait(5)
        return {"title_zh": "论文"}, "http://doc", True

    try:
        r = svc._start_async(ctx, label="arXiv:e1", job=slow_ok,
                             dedup_key="arxiv:e1", arxiv_id="e1")
        assert r == "om_status"
        assert entered.wait(2)
        task = svc.task_store.get("arxiv:e1")
        assert task is not None and task.status == "running"
        assert (task.kind, task.ref, task.chat_id) == ("arxiv", "e1", "c1")
        assert task.open_id == "ou_x" and task.message_id == "om_1"
    finally:
        gate.set()
    _wait_task_status(svc, "arxiv:e1", "done")
    assert svc.task_store.get("arxiv:e1").finished_ts is not None


def test_start_async_marks_failed_on_business_error(tmp_path):
    """端到端：job 业务错误 → 错误卡 + 登记 failed（重启不恢复）。"""
    svc = _make_service(tmp_path, _FakeKimi(), sender=_FakeSender())
    ctx = MessageContext(chat_id="c1")

    def bad(context, **_):
        raise PaperPipelineError("解析失败")

    svc._start_async(ctx, label="arXiv:e2", job=bad, dedup_key="arxiv:e2", arxiv_id="e2")
    _wait_task_status(svc, "arxiv:e2", "failed")
    assert svc.sender.cards and svc.sender.cards[0]["header"]["template"] == "red"


# -- resume_interrupted：启动恢复 --

def _probe_job(calls: list):
    """记录调用参数后立刻以业务错误收尾的 job 替身（让线程快速退出）。"""
    def probe(context, **kwargs):
        calls.append({"context": context, "kwargs": kwargs})
        raise PaperPipelineError("探针收尾")
    return probe


def test_resume_interrupted_restarts_arxiv_task(tmp_path):
    """running 的 arXiv 任务：重建上下文重跑，恢复通知经外发箱必达。"""
    outbox = _FakeOutbox()
    svc = _make_service(tmp_path, _FakeKimi(), sender=_FakeSender(), outbox=outbox)
    svc.task_store.upsert_start(
        "arxiv:r1", "arxiv", "2603.03329", "arXiv:2603.03329", "c1",
        open_id="ou_r", message_id="om_r",
    )
    calls = []
    svc._job_arxiv = _probe_job(calls)

    assert svc.resume_interrupted() == 1
    assert calls and calls[0]["kwargs"] == {"arxiv_id": "2603.03329"}
    ctx = calls[0]["context"]
    assert (ctx.chat_id, ctx.open_id, ctx.message_id) == ("c1", "ou_r", "om_r")
    assert any("上次中断的精读任务已自动恢复：arXiv:2603.03329" in t
               for _, t in outbox.items)
    _wait_task_status(svc, "arxiv:r1", "failed")  # 探针收尾 → 失败路径回写


def test_resume_interrupted_url_and_file_kinds(tmp_path):
    """url / file 两种 kind 同样按登记信息重建（附件解析出 file_name/file_key）。"""
    svc = _make_service(tmp_path, _FakeKimi(), sender=_FakeSender())
    svc.task_store.upsert_start(
        "url:https://x", "url", "https://x", "链接:https://x", "c1")
    svc.task_store.upsert_start(
        "file:fk1", "file", "fk1", "《paper.pdf》", "c1", message_id="om_f")
    url_calls, file_calls = [], []
    svc._job_url = _probe_job(url_calls)
    svc._job_attachment = _probe_job(file_calls)

    assert svc.resume_interrupted() == 2
    assert url_calls[0]["kwargs"] == {"url": "https://x"}
    assert file_calls[0]["kwargs"] == {
        "file_name": "paper.pdf", "file_key": "fk1", "message_id": "om_f",
    }
    _wait_task_status(svc, "url:https://x", "failed")
    _wait_task_status(svc, "file:fk1", "failed")


def test_resume_interrupted_inflight_dedup_skips(tmp_path):
    """重启后用户又手动触发同一篇：在飞锁去重，不重复恢复、不发通知。"""
    outbox = _FakeOutbox()
    svc = _make_service(tmp_path, _FakeKimi(), sender=_FakeSender(), outbox=outbox)
    svc.task_store.upsert_start("arxiv:r2", "arxiv", "2", "arXiv:2", "c1")
    svc._inflight.add("arxiv:r2")  # 重启后新触发的任务已在飞
    calls = []
    svc._job_arxiv = _probe_job(calls)

    assert svc.resume_interrupted() == 0
    assert calls == [] and svc.sender.texts == [] and outbox.items == []
    assert svc.task_store.get("arxiv:r2").status == "running"  # 仍在跑，不动它


def test_resume_interrupted_skips_done_and_failed(tmp_path):
    """语义边界：done/failed 不恢复（failed 已通知过用户，防止恢复死循环）。"""
    svc = _make_service(tmp_path, _FakeKimi(), sender=_FakeSender())
    svc.task_store.upsert_start("arxiv:d", "arxiv", "d", "已完成", "c1")
    svc.task_store.upsert_start("arxiv:f", "arxiv", "f", "已失败", "c1")
    svc.task_store.mark_finished("arxiv:d", ok=True)
    svc.task_store.mark_finished("arxiv:f", ok=False)
    calls = []
    svc._job_arxiv = _probe_job(calls)

    assert svc.resume_interrupted() == 0
    assert calls == [] and svc.sender.texts == []


def test_resume_interrupted_one_failure_does_not_block_others(tmp_path):
    """单条恢复抛异常不影响其他任务，resume 本身也不抛出（不影响启动）。"""
    outbox = _FakeOutbox()
    svc = _make_service(tmp_path, _FakeKimi(), sender=_FakeSender(), outbox=outbox)
    svc.task_store.upsert_start("arxiv:bad", "arxiv", "bad", "坏任务", "c1")
    svc.task_store.upsert_start("arxiv:ok", "arxiv", "ok", "好任务", "c2")
    calls = []
    svc._job_arxiv = _probe_job(calls)
    real_start = svc._start_async

    def flaky_start(context, label, job, *, dedup_key="", **kwargs):
        if dedup_key == "arxiv:bad":
            raise RuntimeError("模拟恢复失败")
        return real_start(context, label, job, dedup_key=dedup_key, **kwargs)

    svc._start_async = flaky_start

    assert svc.resume_interrupted() == 1  # 坏任务跳过，好任务恢复
    assert [c["kwargs"] for c in calls] == [{"arxiv_id": "ok"}]
    assert any("上次中断的精读任务已自动恢复：好任务" in t for _, t in outbox.items)
    _wait_task_status(svc, "arxiv:ok", "failed")  # 探针收尾


# ---- V8 第 8 步收口：发布/落库/向量入库失败不判任务失败 ----

def test_pipeline_store_and_vector_failure_still_delivers(tmp_path, monkeypatch):
    """正文已交付即是成功：sqlite 落库失败、向量入库失败只警告+通知，
    _pipeline 正常返回（文档链接不变），不抛异常。"""
    from feishu_assistant.modules import paper as paper_mod
    from feishu_assistant.modules.reading import ReadingResult

    md_path = tmp_path / "paper.md"
    md_path.write_text("# 全文", encoding="utf-8")
    monkeypatch.setattr(paper_mod, "parse_pdf", lambda src, out: md_path)
    monkeypatch.setattr(paper_mod, "load_figure_assets", lambda p: [])
    monkeypatch.setattr(paper_mod, "load_formula_assets", lambda p: [])
    monkeypatch.setattr(paper_mod, "pdf_page_count", lambda p: 1)
    monkeypatch.setattr(paper_mod, "classify_direction", lambda *a, **k: "其他")

    class _Engine:
        def run(self, *args, **kwargs):
            return ReadingResult(
                body_md="# 已生成正文", metadata={"title_zh": "测试论文"},
                layer_names=[], figure_notes={}, figure_inlines={}, vision_used=False)

    class _Docx:
        def create_document(self, name):
            return "doc-id", "https://feishu.cn/docx/doc-id"

        def grant_access(self, doc_id, open_id=None):
            return True

        def write_markdown(self, doc_id, body, images=None, strict_images=False):
            return None

    class _FailStore:
        db_path = tmp_path / "fail.db"

        def save_paper(self, *args, **kwargs):
            raise RuntimeError("sqlite 宕机")

    class _FailVector:
        def add_paper(self, *args, **kwargs):
            raise RuntimeError("向量索引宕机")

    sender = _FakeSender()
    svc = _make_service(tmp_path, _FakeKimi(), sender=sender)
    svc.reading_engine = _Engine()
    svc.docx = _Docx()
    svc.store = _FailStore()
    svc.vector_index = _FailVector()

    metadata, doc_url, archived = svc._pipeline(
        MessageContext(chat_id="c1"), tmp_path / "p.pdf",
        key="k", pdf_url="https://example.com/p.pdf",
        arxiv_id="2603.03329", meta_hint={})
    assert doc_url == "https://feishu.cn/docx/doc-id"   # 正文照常交付
    assert metadata["store_failed"] and metadata["vector_failed"]
    assert any("落库失败" in t for t in sender.texts)    # 用户被通知
    assert any("向量入库失败" in t for t in sender.texts)


def test_run_guarded_degraded_v9_does_not_claim_final_review_pass(tmp_path):
    svc = _make_service(tmp_path, _FakeKimi(), sender=_FakeSender())
    status = _FakeStatus()

    def degraded_job(context, **_):
        return {
            "title_zh": "论文",
            "release_status": "fact_review_unavailable",
            "final_review_degraded": "事实审查服务不可用",
        }, "http://doc", True

    svc._run_guarded(MessageContext(chat_id="c1"), "L", degraded_job, {}, status, "")

    assert status.stopped == [(True, "精读完成，发布状态：事实审查未完成")]
    assert "通过终审" not in status.stopped[0][1]
    assert svc.sender.cards[0]["header"]["template"] == "orange"
