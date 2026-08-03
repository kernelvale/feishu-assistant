"""modules/authority.py 测试：S2 信号 / 机构关键词 / 作者团 / 类型加权 / 封顶。

全部离线：Semantic Scholar 查询注入 fake s2_fetcher，不打真实 API。
"""

from __future__ import annotations

import pytest

from feishu_assistant.modules.authority import (
    INSTITUTION_BONUS,
    S2_CITATION_CAP,
    S2_VENUE_BONUS,
    TEAM_BONUS,
    TYPE_BONUS,
    authority_score,
    institution_hits,
)
from feishu_assistant.modules.digest import ArxivEntry


def _entry(arxiv_id="2607.00001", title="某论文", authors="Alice, Bob", abstract=""):
    return ArxivEntry(arxiv_id=arxiv_id, title=title, abstract=abstract or f"{title} 的摘要",
                      authors=authors, published="2026-07-30",
                      url=f"https://arxiv.org/abs/{arxiv_id}")


NO_S2 = lambda aid: None  # S2 无数据/失败（静默按 0 计）


# ---------------------------------------------------------------------------
# Semantic Scholar 信号
# ---------------------------------------------------------------------------

class TestS2Signals:
    def test_venue_and_citations(self):
        fetcher = lambda aid: {"venue": "NeurIPS 2025", "citationCount": 50}
        score, reasons = authority_score(_entry(), s2_fetcher=fetcher)
        # ln(1+50)/10 ≈ 0.39 > 0.2 → 按封顶 0.2 计
        assert score == pytest.approx(S2_VENUE_BONUS + S2_CITATION_CAP)
        assert "NeurIPS 2025" in reasons and "被引 50" in reasons

    def test_citation_log_weight_below_cap(self):
        fetcher = lambda aid: {"venue": "", "citationCount": 2}
        import math
        score, reasons = authority_score(_entry(), s2_fetcher=fetcher)
        assert score == pytest.approx(math.log1p(2) / 10)  # 未触顶，纯对数权重
        assert reasons == ["被引 2"]  # venue 空不计入

    def test_publication_venue_fallback(self):
        fetcher = lambda aid: {"venue": "", "citationCount": 0,
                               "publicationVenue": {"name": "ICLR 2026"}}
        score, reasons = authority_score(_entry(), s2_fetcher=fetcher)
        assert score == pytest.approx(S2_VENUE_BONUS) and reasons == ["ICLR 2026"]

    def test_s2_failure_silent_zero(self):
        score, reasons = authority_score(_entry(), s2_fetcher=NO_S2)
        assert score == 0.0 and reasons == []

    def test_garbage_payload_tolerated(self):
        fetcher = lambda aid: {"venue": 123, "citationCount": "很多"}  # 类型异常
        score, _ = authority_score(_entry(), s2_fetcher=fetcher)
        assert score == pytest.approx(S2_VENUE_BONUS)  # venue 字符串化计入，被引按 0


# ---------------------------------------------------------------------------
# 机构关键词
# ---------------------------------------------------------------------------

class TestInstitutionHits:
    def test_deepmind_absorbs_google(self):
        hits = institution_hits(_entry(authors="Alice, Google DeepMind"))
        assert hits == ["Google DeepMind"]  # deepmind 命中后不再重复标 Google

    def test_multiple_institutions(self):
        hits = institution_hits(_entry(title="A study", authors="OpenAI, Tsinghua University"))
        assert hits == ["OpenAI", "清华大学"]

    def test_bonus_applied_once(self):
        score, reasons = authority_score(
            _entry(authors="OpenAI, Anthropic, DeepSeek"), s2_fetcher=NO_S2)
        assert score == pytest.approx(INSTITUTION_BONUS)  # 多机构也只加一次
        assert reasons == ["OpenAI", "Anthropic", "DeepSeek"]

    def test_word_boundary_no_false_positive(self):
        # Smith 含 mit、limit 含 mit，均不应命中 MIT
        assert institution_hits(_entry(title="Limit theorems", authors="Bob Smith")) == []
        assert institution_hits(_entry(authors="MIT CSAIL")) == ["MIT"]

    def test_chinese_lab_keywords(self):
        hits = institution_hits(_entry(authors="Shanghai AI Lab, SJTU"))
        assert hits == ["上海 AI Lab", "上海交大"]


# ---------------------------------------------------------------------------
# 作者团规模 / 论文类型
# ---------------------------------------------------------------------------

class TestTeamAndType:
    def test_large_team(self):
        authors = ", ".join(f"Author{i}" for i in range(26))
        score, reasons = authority_score(_entry(authors=authors), s2_fetcher=NO_S2)
        assert score == pytest.approx(TEAM_BONUS) and reasons == ["26 作者团队"]

    def test_small_team_no_bonus(self):
        score, _ = authority_score(_entry(authors="A, B, C"), s2_fetcher=NO_S2)
        assert score == 0.0

    def test_empty_authors_no_bonus(self):
        score, _ = authority_score(_entry(authors=""), s2_fetcher=NO_S2)  # RSS 通道无作者
        assert score == 0.0

    def test_type_keywords(self):
        score, reasons = authority_score(_entry(title="A Survey of Agent Benchmarks"),
                                         s2_fetcher=NO_S2)
        assert score == pytest.approx(TYPE_BONUS) and reasons == ["survey/benchmark 类型"]

    def test_type_word_boundary(self):
        score, _ = authority_score(_entry(title="Repositioning the field"), s2_fetcher=NO_S2)
        assert score == 0.0  # "Repositioning" 不算 position 类型


# ---------------------------------------------------------------------------
# 综合封顶
# ---------------------------------------------------------------------------

def test_total_capped_at_1():
    authors = ", ".join(["OpenAI"] + [f"Author{i}" for i in range(11)])
    fetcher = lambda aid: {"venue": "NeurIPS 2025", "citationCount": 100}
    score, reasons = authority_score(
        _entry(title="A Survey of LLM Post-training", authors=authors), s2_fetcher=fetcher)
    # 0.3(venue) + 0.2(被引封顶) + 0.3(机构) + 0.2(团队) + 0.2(类型) = 1.2 → 封顶 1.0
    assert score == 1.0
    assert "NeurIPS 2025" in reasons and "OpenAI" in reasons
    assert "12 作者团队" in reasons and "survey 类型" in reasons


def test_no_signals_zero():
    score, reasons = authority_score(
        _entry(title="Ordinary paper", authors="X, Y"), s2_fetcher=NO_S2)
    assert score == 0.0 and reasons == []


def test_arxiv_org_venue_not_counted_as_published():
    """venue='arXiv.org' 不算正式发表（预印本平台，不得 venue 分）。"""
    from feishu_assistant.modules.authority import authority_score
    from feishu_assistant.modules.digest import ArxivEntry

    entry = ArxivEntry(arxiv_id="2601.00001", title="Ordinary incremental work",
                       abstract="", authors="A. Person", published="2026-01-01", url="")
    score, reasons = authority_score(
        entry, s2_fetcher=lambda aid: {"venue": "arXiv.org", "citationCount": 0})
    assert "arXiv.org" not in reasons
    assert score < 0.3


def test_s2_authors_affiliations_and_first_author_label():
    """S2 作者机构命中 + 一作标注（拿不到作者时按 arXiv 作者串数人头）。"""
    from feishu_assistant.modules.authority import authority_score
    from feishu_assistant.modules.digest import ArxivEntry

    entry = ArxivEntry(arxiv_id="x", title="Some work", abstract="",
                       authors="A, B", published="2026", url="")
    data = {
        "venue": "", "citationCount": 0,
        "authors": [
            {"name": "Kaiyan Zhang", "affiliations": ["Tsinghua University"]},
            {"name": "Yuxin Zuo", "affiliations": []},
        ],
    }
    score, reasons = authority_score(entry, s2_fetcher=lambda aid: data)
    assert "Tsinghua" in " ".join(reasons)
    assert "一作 Kaiyan Zhang（Tsinghua University）" in reasons
    assert score >= 0.3  # 机构命中


def test_s2_authors_team_size_from_s2():
    """作者团规模优先按 S2 authors 计数。"""
    from feishu_assistant.modules.authority import authority_score
    from feishu_assistant.modules.digest import ArxivEntry

    entry = ArxivEntry(arxiv_id="x", title="Team paper", abstract="",
                       authors="A, B", published="2026", url="")
    data = {"venue": "", "citationCount": 0,
            "authors": [{"name": f"P{i}", "affiliations": []} for i in range(12)]}
    score, reasons = authority_score(entry, s2_fetcher=lambda aid: data)
    assert "12 作者团队" in reasons
