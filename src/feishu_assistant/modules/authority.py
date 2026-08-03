"""权威信号打分：论文"值不值得读"的先验估计，供日报质量闸/每周精选排序用。

信号按优先级（单项加分，总分封顶 1.0，每个命中项记入原因列表供卡片标注）：
- Semantic Scholar 免费 API（无需 key，约 1rps 限流，逐篇调用间隔 ≥1s）：
  venue 非空（正式发表）+0.3；citationCount 对数权重 ln(1+c)/10 封顶 +0.2；
  限流/404/网络失败一律静默按 0 计（绝不影响推送主流程）
- 机构/作者关键词（标题+作者串命中，词边界匹配防 Smith→mit 误伤）+0.3
- 作者团规模（≥10 人，大型综述/benchmark 团队特征）+0.2
- 类型加权（标题含 survey|benchmark|position|foundations|principles）+0.2
"""

from __future__ import annotations

import logging
import math
import re
import time
from typing import TYPE_CHECKING, Any, Callable

import httpx

if TYPE_CHECKING:  # 仅类型标注，避免与 digest 循环导入
    from .digest import ArxivEntry

logger = logging.getLogger("feishu_assistant.authority")

# ---- Semantic Scholar 信号 ----
S2_API = "https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
S2_FIELDS = "title,venue,citationCount,publicationVenue,authors.name,authors.affiliations"
S2_TIMEOUT = 10
S2_MIN_INTERVAL = 1.0  # 免费档约 1rps：两次调用的最小间隔（秒）
S2_VENUE_BONUS = 0.3   # venue 非空（正式发表）
S2_CITATION_CAP = 0.2  # 被引对数权重 ln(1+c)/10 的封顶值

# ---- 机构/作者关键词（关键词, 卡片标注名），按优先级序 ----
INSTITUTION_BONUS = 0.3
INSTITUTION_SIGNALS = (
    ("deepmind", "Google DeepMind"),
    ("openai", "OpenAI"),
    ("anthropic", "Anthropic"),
    ("google", "Google"),
    ("meta ai", "Meta AI"),
    ("microsoft research", "Microsoft Research"),
    ("deepseek", "DeepSeek"),
    ("qwen", "Qwen"),
    ("bytedance", "ByteDance"),
    ("tsinghua", "清华大学"),
    ("peking university", "北京大学"),
    ("shanghai ai lab", "上海 AI Lab"),
    ("sjtu", "上海交大"),
    ("mit", "MIT"),
    ("stanford", "Stanford"),
    ("berkeley", "Berkeley"),
    ("cmu", "CMU"),
    ("nus", "NUS"),
    ("cuhk", "CUHK"),
)

# ---- 作者团规模 / 论文类型 ----
TEAM_SIZE_THRESHOLD = 10
TEAM_BONUS = 0.2
TYPE_BONUS = 0.2
TYPE_KEYWORDS = ("survey", "benchmark", "position", "foundations", "principles")

SCORE_CAP = 1.0

_last_s2_call = 0.0  # monotonic 时间戳，限流用


def _throttle() -> None:
    """S2 免费档限速：两次调用至少间隔 S2_MIN_INTERVAL 秒。"""
    global _last_s2_call
    wait = S2_MIN_INTERVAL - (time.monotonic() - _last_s2_call)
    if wait > 0:
        time.sleep(wait)
    _last_s2_call = time.monotonic()


def fetch_s2_paper(arxiv_id: str, timeout: int = S2_TIMEOUT) -> dict[str, Any] | None:
    """查 S2 单篇元数据；限流/404/网络/解析失败一律静默返回 None（按无信号计）。"""
    _throttle()
    try:
        resp = httpx.get(S2_API.format(arxiv_id=arxiv_id),
                         params={"fields": S2_FIELDS}, timeout=timeout)
        if resp.status_code in (404, 429):
            logger.warning("S2 查询 %s 返回 %d（按无信号计）", arxiv_id, resp.status_code)
            return None
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else None
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("S2 查询 %s 失败（按无信号计）：%s", arxiv_id, exc)
        return None


def _word_hit(keyword: str, haystack: str) -> bool:
    """词边界匹配：防 Smith→mit、limit→mit 这类子串误伤。"""
    return re.search(rf"\b{re.escape(keyword)}\b", haystack) is not None


def institution_hits(entry: ArxivEntry) -> list[str]:
    """标题+作者串命中的机构标注名（保序去重；deepmind 命中后吸收 google）。"""
    haystack = f"{entry.title}\n{entry.authors}".lower()
    hits: list[str] = []
    for kw, label in INSTITUTION_SIGNALS:
        if _word_hit(kw, haystack) and not any(kw in h.lower() for h in hits):
            hits.append(label)
    return hits


def authority_score(
    entry: ArxivEntry,
    s2_fetcher: Callable[[str], dict[str, Any] | None] | None = None,
) -> tuple[float, list[str]]:
    """权威分（0~1）+ 命中原因列表。s2_fetcher 可注入打桩，默认真实查询（失败静默）。"""
    fetch = fetch_s2_paper if s2_fetcher is None else s2_fetcher
    score = 0.0
    reasons: list[str] = []

    data = fetch(entry.arxiv_id)
    s2_authors: list[dict[str, Any]] = []
    if data:
        venue = str(data.get("venue") or "").strip()
        if not venue:  # venue 空时退 publicationVenue.name
            pv = data.get("publicationVenue")
            if isinstance(pv, dict):
                venue = str(pv.get("name") or "").strip()
        if venue and venue.lower() not in ("arxiv.org", "arxiv"):  # arXiv 本身不算正式发表
            score += S2_VENUE_BONUS
            reasons.append(venue)
        try:
            citations = int(data.get("citationCount") or 0)
        except (TypeError, ValueError):
            citations = 0
        if citations > 0:
            score += min(S2_CITATION_CAP, math.log1p(citations) / 10)
            reasons.append(f"被引 {citations}")
        s2_authors = [a for a in (data.get("authors") or []) if isinstance(a, dict)]

    # 机构信号：标题/作者串 + S2 作者单位（更准；单位是英文全名，关键词表本就英文）
    hits = institution_hits(entry)
    if not hits and s2_authors:
        affil_text = " ".join(
            aff for a in s2_authors for aff in (a.get("affiliations") or [])).lower()
        hits = [label for kw, label in INSTITUTION_SIGNALS if _word_hit(kw, affil_text)]
    if hits:
        score += INSTITUTION_BONUS
        reasons.extend(hits)

    n_authors = len(s2_authors) or (
        len([a for a in entry.authors.split(",") if a.strip()]) if entry.authors else 0)
    if n_authors >= TEAM_SIZE_THRESHOLD:
        score += TEAM_BONUS
        reasons.append(f"{n_authors} 作者团队")

    # 一作标注（卡片可读性：作者 + 机构，拿不到就只给姓名）
    if s2_authors:
        first = s2_authors[0]
        first_name = str(first.get("name") or "").strip()
        first_affils = (first.get("affiliations") or [])[:1]
        if first_name:
            reasons.append(f"一作 {first_name}" + (f"（{first_affils[0]}）" if first_affils else ""))

    title = entry.title.lower()
    hit_types = [kw for kw in TYPE_KEYWORDS
                 if re.search(rf"\b{re.escape(kw)}s?\b", title)]  # 词边界 + 容忍复数
    if hit_types:
        score += TYPE_BONUS
        reasons.append(f"{'/'.join(hit_types)} 类型")

    return min(SCORE_CAP, score), reasons
