"""通用论文链接（非 arXiv）精读入口与链接解析测试。"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from feishu_assistant.config import Settings
from feishu_assistant.modules.paper import (
    MessageContext,
    PaperPipelineError,
    PaperReadingService,
    parse_paper_url,
    resolve_paper_url,
)
from feishu_assistant.store import PaperStore


class FakeSender:
    def send_text(self, chat_id, text):
        return True

    def send_card(self, chat_id, card):
        return True

    def send_text_id(self, chat_id, text):
        return ""

    def update_text(self, message_id, text):
        return True


def make_svc():
    return PaperReadingService(Settings(), kimi=None, lark_client=None,
                               store=PaperStore(":memory:"), sender=FakeSender(),
                               vector_index=None)


CTX = MessageContext(chat_id="oc_x", open_id="ou_x", message_id="om_x")

# ---- parse_paper_url：哪些链接自动触发 ----

def test_parse_paper_url_pdf_direct():
    assert parse_paper_url("看看 https://example.com/papers/2501.12345.pdf") == \
        "https://example.com/papers/2501.12345.pdf"


def test_parse_paper_url_known_hosts():
    assert parse_paper_url("https://openreview.net/forum?id=abc123") is not None
    assert parse_paper_url("https://aclanthology.org/2024.acl-long.123/") is not None
    assert parse_paper_url("https://www.biorxiv.org/content/10.1101/2024.01.01.123456v1") is not None


def test_parse_paper_url_ignores_normal_links():
    assert parse_paper_url("https://github.com/user/repo") is None
    assert parse_paper_url("https://feishu.cn/docx/ABC") is None
    assert parse_paper_url("没有链接") is None


# ---- resolve_paper_url：PDF 解析 ----

def test_resolve_pdf_direct():
    pdf, hint = resolve_paper_url("https://example.com/x.pdf?dl=1")
    assert pdf == "https://example.com/x.pdf?dl=1" and hint == {}


def test_resolve_openreview_pattern():
    pdf, _ = resolve_paper_url("https://openreview.net/forum?id=Ys2jXKLM&noteId=zzz")
    assert pdf == "https://openreview.net/pdf?id=Ys2jXKLM"


def test_resolve_ar5iv_pattern():
    pdf, _ = resolve_paper_url("https://ar5iv.labs.arxiv.org/html/2504.13359")
    assert pdf == "https://arxiv.org/pdf/2504.13359"


def test_resolve_aclanthology_pattern():
    pdf, _ = resolve_paper_url("https://aclanthology.org/2024.acl-long.123/")
    assert pdf == "https://aclanthology.org/2024.acl-long.123.pdf"


def _fake_resp(text: str, ctype: str = "text/html") -> httpx.Response:
    return httpx.Response(200, text=text, headers={"content-type": ctype},
                          request=httpx.Request("GET", "https://x"))


def test_resolve_via_citation_pdf_meta(monkeypatch):
    html = """
    <html><head>
      <meta name="citation_title" content="A Great Paper">
      <meta name="citation_author" content="Doe, Jane">
      <meta name="citation_publication_date" content="2025/03/01">
      <meta content="https://publisher.com/x.pdf" name="citation_pdf_url">
    </head><body>landing</body></html>
    """
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _fake_resp(html))
    pdf, hint = resolve_paper_url("https://publisher.com/article/123")
    assert pdf == "https://publisher.com/x.pdf"
    assert hint["title"] == "A Great Paper"
    assert hint["authors"] == "Doe, Jane"
    assert hint["year"] == "2025"


def test_resolve_relative_pdf_url(monkeypatch):
    html = '<meta name="citation_pdf_url" content="/papers/x.pdf">'
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _fake_resp(html))
    pdf, _ = resolve_paper_url("https://conf.org/2025/article/9")
    assert pdf == "https://conf.org/papers/x.pdf"


def test_resolve_no_pdf_raises_chinese(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _fake_resp("<html>新闻页</html>"))
    with pytest.raises(PaperPipelineError, match="没有找到论文 PDF"):
        resolve_paper_url("https://news.site/article/1")


# ---- try_handle_text 路由 ----

def test_handle_text_pdf_link_starts_job(monkeypatch):
    svc = make_svc()
    started = {}
    monkeypatch.setattr(svc, "_start_async",
                        lambda ctx, label, job, **kw: started.update(label=label, kw=kw) or "om_fake")
    reply = svc.try_handle_text("https://example.com/x.pdf", CTX)
    assert reply == ""  # 状态条即 ack，不再单独回消息
    assert started["kw"]["url"] == "https://example.com/x.pdf"


def test_handle_text_openreview_starts_job(monkeypatch):
    svc = make_svc()
    started = {}
    monkeypatch.setattr(svc, "_start_async",
                        lambda ctx, label, job, **kw: started.update(kw=kw) or "om_fake")
    reply = svc.try_handle_text("https://openreview.net/forum?id=abc", CTX)
    assert reply == "" and started["kw"]["url"].startswith("https://openreview.net")


def test_handle_text_unknown_url_needs_keyword(monkeypatch):
    svc = make_svc()
    started = {}
    monkeypatch.setattr(svc, "_start_async",
                        lambda ctx, label, job, **kw: started.update(kw=kw) or "om_fake")
    # 普通链接无「精读」→ 不触发（留给自由对话）
    assert svc.try_handle_text("https://blog.site/post/1", CTX) is None
    assert not started
    # 带「精读」→ 强制尝试
    reply = svc.try_handle_text("精读 https://blog.site/post/1", CTX)
    assert reply == "" and started["kw"]["url"] == "https://blog.site/post/1"


def test_handle_text_arxiv_still_first(monkeypatch):
    svc = make_svc()
    started = {}
    monkeypatch.setattr(svc, "_start_async",
                        lambda ctx, label, job, **kw: started.update(kw=kw) or "om_fake")
    reply = svc.try_handle_text("https://arxiv.org/abs/2504.13359", CTX)
    assert reply == ""
    assert started["kw"].get("arxiv_id") == "2504.13359"  # 走 arXiv 管线而非通用链接


def test_handle_text_fallback_ack_when_status_unsent(monkeypatch):
    """状态条没发出去（返回 ""）→ 回退文字 ack，用户至少有反馈。"""
    svc = make_svc()
    monkeypatch.setattr(svc, "_start_async", lambda ctx, label, job, **kw: "")
    reply = svc.try_handle_text("https://arxiv.org/abs/2504.13359", CTX)
    assert reply and "开始精读" in reply
