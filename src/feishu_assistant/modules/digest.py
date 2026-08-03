"""P4 自动化推送：arXiv 日报 / 今日提醒 / 每周精选 / 组会周报 + 推送调度。

- arXiv 日报：arXiv API（httpx + 标准库 XML 解析）拉近 1~2 天新论文 →
  推送历史跨天排除（精选库已推送 ∪ 论文库已精读的 arXivID 不进候选；向量检索
  同时 exclude_arxiv_id，已入库论文自我命中不算口味证据）→
  按论文库口味排序（向量相似度优先，无向量/库空退关键词重叠）→
  口味分（批内归一）×0.6 + 权威分（authority.py）×0.4 合并，过质量闸
  （QUALITY_THRESHOLD）才留 → top 3 → light 逐篇质量评审（风险点）+ TL;DR →
  卡片推送（每篇带权威标注）；0 篇过线只记日志不推送；arxiv_digest_now() 供命令立即触发
- 今日提醒：备忘库 72h DDL（含逾期）+ 求职看板该跟进（复用 career.summarize_board）
  + 收藏库翻坟（已回顾=false，最老优先 1~2 条）→ 一条汇总；today_digest_now()
- 晚间回顾：今日提醒的晚间档（同素材，🌙 标题）；evening_digest_now()
- 每周精选：周五 20:00，近 7 天新论文 → 排除已精读与已上过周刊的（日报推过的
  不排除，周刊再策展时归档走 upgrade_on_dup 升级原记录）→ 口味+权威合并排序 →
  过闸取 top 5 → light 逐篇深评（为什么值得读/与你方向/风险点）→ 卡片推送；weekly_picks_now()
- 组会周报：近 7 天 灵感/收藏/精读 → heavy 大纲 → 飞书文档 → 推送链接；weekly_report_now()
- 调度：register_jobs(scheduler, service) 纯函数，cron 日报（默认 8:30，config.yaml
  push.daily_time 覆盖）/ 今日 9:00 / 晚间回顾（默认 22:00，push.evening_time 覆盖）
  / 周一 9:00 周报 / 周五 20:00 每周精选，main.py 一行接入；推送目标 chat_id 由构造注入
  （config.yaml 的 push.chat_id，接线时读）

推送纪律：发送失败重试一次，仍失败记 ERROR 日志；所有推送内容含生成时间。
长任务（日报/周报）走后台线程 ack + 推送，对齐 career.py 的异步模式。
"""

from __future__ import annotations

import logging
import re
import threading
import time
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

from ..config import PROJECT_ROOT, Settings
from ..feishu.bitable import (
    FIELD_TYPE_MULTI_SELECT,
    FIELD_TYPE_SINGLE_SELECT,
    PAPER_DIRECTIONS,
    BitableError,
    FieldSpec,
    ensure_fields,
)
from ..feishu.docx import DocxClient, DocxError
from ..kimi import KimiClient, KimiError
from .authority import authority_score
from .career import (
    extract_json,
    field_text,
    record_fields,
    summarize_board,
)
from .paper import MessageContext, Sender

logger = logging.getLogger("feishu_assistant.digest")

PROMPTS_DIR = PROJECT_ROOT / "prompts"
ARXIV_PROMPT_PATH = PROMPTS_DIR / "digest_arxiv.md"
QUALITY_PROMPT_PATH = PROMPTS_DIR / "digest_quality.md"
QUALITY_K = 15  # 质量关：主题分 top K 进入逐篇评审，再取 top N
QUALITY_BONUS = 3.0  # 质量分满分折算的综合分加成（score/5 * 此值）
WEEKLY_PROMPT_PATH = PROMPTS_DIR / "digest_weekly.md"
WEEKLY_PICKS_PROMPT_PATH = PROMPTS_DIR / "digest_weekly_picks.md"

# ---- 质量闸（日报/每周精选共用）：合并分 = 口味归一分×TASTE_WEIGHT + 权威分×AUTHORITY_WEIGHT ----
QUALITY_THRESHOLD = 0.55  # 合并分过线才推（可调）
TASTE_WEIGHT = 0.6
AUTHORITY_WEIGHT = 0.4
DAILY_TOP_N = 3  # 日报每批最多推送篇数（每周精选每批 ≤5，见 WEEKLY_PICKS_N）
SILENT_WHEN_EMPTY = True  # 0 篇过线时只记日志不推送（False 则推"今日不打扰"说明）

PAPER_TABLE = "论文库"
COLLECT_TABLE = "收藏库"
IDEA_TABLE = "灵感库"
MEMO_TABLE = "备忘库"
BOARD_TABLE = "求职看板"
PICK_TABLE = "精选库"
PICK_SOURCE = "arXiv日报"
PICK_SOURCE_WEEKLY = "每周精选"

# ---- arXiv 抓取参数 ----
ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_RSS = "https://export.arxiv.org/rss/{cat}"  # API 限流时的备用出口（当日新论文全量）
ARXIV_CATEGORIES = ("cs.CL", "cs.AI", "cs.LG")
# 检索池关键词：覆盖 agent/harness 与大模型后训练两个方向
ARXIV_KEYWORDS = (
    '"large language model"', "LLM", "agent", "reasoning", "inference",
    '"post-training"', "RLHF", "GRPO", "DPO", '"reinforcement learning"',
    '"fine-tuning"', "alignment", '"reward model"',
)
ARXIV_MAX_RESULTS = 80
ARXIV_DAYS = 2
WEEKEND_DAYS = 3  # 周一日报窗口（周六/周日不推送，周末存量并入周一）
ARXIV_TIMEOUT = 20
TOP_N = 5

MEMO_WINDOW_HOURS = 72  # 今日提醒：该时间窗内的备忘 DDL
RESURFACE_COUNT = 2  # 收藏翻坟条数（1~2）
WEEKLY_DAYS = 7
WEEKLY_PICKS_DAYS = 7  # 每周精选：arXiv 回溯窗口
WEEKLY_PICKS_N = 5     # 每周精选：过闸后最多推送篇数（严选 3~5）

# ---- 推送调度默认时间（register_jobs 默认值；main.py 启动横幅同源引用）----
DEFAULT_ARXIV_TIME = "08:30"    # arXiv 日报（config.yaml push.daily_time 可覆盖）
DEFAULT_TODAY_TIME = "09:00"    # 今日提醒
DEFAULT_EVENING_TIME = "22:00"  # 晚间回顾（config.yaml push.evening_time 可覆盖）
DEFAULT_WEEKLY_TIME = "09:00"   # 组会周报（周一）
DEFAULT_PICKS_TIME = "20:00"    # 每周精选（周五）


class DigestError(Exception):
    """推送内容生成失败，信息为中文。"""


# ---------------------------------------------------------------------------
# arXiv：查询构造 / 抓取 / XML 解析（纯函数可离线测试）
# ---------------------------------------------------------------------------

@dataclass
class ArxivEntry:
    arxiv_id: str
    title: str
    abstract: str
    authors: str
    published: str  # YYYY-MM-DD
    url: str


def build_arxiv_query(days: int = ARXIV_DAYS, now: float | None = None) -> str:
    """构造 arXiv API 的 search_query：类目 OR + 关键词 OR + submittedDate 区间。"""
    now = now if now is not None else time.time()
    end = time.strftime("%Y%m%d2359", time.localtime(now))
    start = time.strftime("%Y%m%d0000", time.localtime(now - days * 86400))
    cats = " OR ".join(f"cat:{c}" for c in ARXIV_CATEGORIES)
    kws = " OR ".join(f"all:{k}" for k in ARXIV_KEYWORDS)
    return f"({cats}) AND ({kws}) AND submittedDate:[{start} TO {end}]"


def parse_arxiv_feed(xml_text: str) -> list[ArxivEntry]:
    """解析 arXiv Atom feed。非法 XML 抛 DigestError；无条目返回 []。"""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise DigestError(f"arXiv 返回的不是合法 XML：{exc}") from exc
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entries: list[ArxivEntry] = []
    for entry in root.findall("a:entry", ns):
        raw_id = (entry.findtext("a:id", "", ns) or "").strip()
        arxiv_id = re.sub(r"v\d+$", "", raw_id.rsplit("/", 1)[-1])
        title = re.sub(r"\s+", " ", entry.findtext("a:title", "", ns) or "").strip()
        abstract = re.sub(r"\s+", " ", entry.findtext("a:summary", "", ns) or "").strip()
        authors = ", ".join(
            a.findtext("a:name", "", ns) for a in entry.findall("a:author", ns)
        )
        published = (entry.findtext("a:published", "", ns) or "")[:10]
        if not arxiv_id or not title:
            continue
        entries.append(ArxivEntry(
            arxiv_id=arxiv_id, title=title, abstract=abstract,
            authors=authors, published=published,
            url=f"https://arxiv.org/abs/{arxiv_id}",
        ))
    return entries


def parse_arxiv_rss(xml_text: str) -> list[ArxivEntry]:
    """解析 arXiv RSS 2.0（备用出口）。无关键词/日期筛选，当日新论文全量，交给下游排序。"""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise DigestError(f"arXiv RSS 返回的不是合法 XML：{exc}") from exc
    channel_date = (root.findtext("channel/pubDate") or "")[5:16]  # "30 Jul 2026"
    try:
        published = time.strftime("%Y-%m-%d", time.strptime(channel_date, "%d %b %Y")) if channel_date else time.strftime("%Y-%m-%d")
    except ValueError:
        published = time.strftime("%Y-%m-%d")
    entries: list[ArxivEntry] = []
    for item in root.findall("channel/item"):
        title = re.sub(r"\s+", " ", item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        desc = re.sub(r"\s+", " ", item.findtext("description") or "").strip()
        m = re.search(r"arXiv:(\S+?)(v\d+)?\s+Announce Type", desc)
        arxiv_id = re.sub(r"v\d+$", "", m.group(1)) if m else link.rsplit("/", 1)[-1]
        abstract = desc[m.end():].strip() if m else desc
        abstract = re.sub(r"^Abstract:\s*", "", abstract)
        if not arxiv_id or not title:
            continue
        entries.append(ArxivEntry(
            arxiv_id=arxiv_id, title=title, abstract=abstract,
            authors="", published=published,
            url=link or f"https://arxiv.org/abs/{arxiv_id}",
        ))
    return entries


def fetch_arxiv_rss(timeout: int = ARXIV_TIMEOUT) -> list[ArxivEntry]:
    """备用抓取：arXiv RSS 三个类目（API 限流时兜底，当日批次全量）。"""
    entries: dict[str, ArxivEntry] = {}
    for cat in ARXIV_CATEGORIES:
        try:
            resp = httpx.get(ARXIV_RSS.format(cat=cat), timeout=timeout, follow_redirects=True)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("arXiv RSS %s 抓取失败（跳过该类目）：%s", cat, exc)
            continue
        for entry in parse_arxiv_rss(resp.text):
            entries.setdefault(entry.arxiv_id, entry)
    if not entries:
        raise DigestError("arXiv RSS 三个类目全部抓取失败")
    logger.info("arXiv RSS 备用抓取完成：%d 篇（%d 个类目）", len(entries), len(ARXIV_CATEGORIES))
    return list(entries.values())


def fetch_arxiv_entries(
    days: int = ARXIV_DAYS,
    max_results: int = ARXIV_MAX_RESULTS,
    timeout: int = ARXIV_TIMEOUT,
    now: float | None = None,
) -> list[ArxivEntry]:
    """真实抓取 arXiv API（测试注入 fake 替代；另一个网络出口是 authority.py 的 S2 查询）。"""
    params = {
        "search_query": build_arxiv_query(days, now),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "start": "0",
        "max_results": str(max_results),
    }
    try:
        resp = httpx.get(ARXIV_API, params=params, timeout=timeout, follow_redirects=True)
        if resp.status_code == 429:  # arXiv 限流：退避后重试一次
            logger.warning("arXiv 429 限流，30s 后重试一次")
            time.sleep(30)
            resp = httpx.get(ARXIV_API, params=params, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("arXiv API 失败（转 RSS 备用通道）：%s", exc)
        return fetch_arxiv_rss(timeout=timeout)
    entries = parse_arxiv_feed(resp.text)
    logger.info("arXiv 抓取完成：%d 篇（近 %d 天）", len(entries), days)
    return entries


# ---------------------------------------------------------------------------
# 排序：向量相似度优先，关键词重叠兜底
# ---------------------------------------------------------------------------

def _multi_values(value: Any) -> list[str]:
    """bitable 多选字段读值规整为字符串列表。"""
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = field_text(value)
    return [text] if text else []


def taste_keywords_from_papers(paper_records: list[Any]) -> list[str]:
    """从论文库记录提炼口味关键词（关键词多选 + 方向），去重保序。"""
    seen: dict[str, None] = {}
    for rec in paper_records:
        fields = record_fields(rec)
        for kw in _multi_values(fields.get("关键词")) + _multi_values(fields.get("方向")):
            seen.setdefault(kw, None)
    return list(seen)


def keyword_score(entry: ArxivEntry, taste_keywords: list[str]) -> float:
    """标题+摘要中命中的口味关键词个数（大小写不敏感）。"""
    if not taste_keywords:
        return 0.0
    haystack = f"{entry.title}\n{entry.abstract}".lower()
    return float(sum(1 for kw in taste_keywords if kw.lower() in haystack))


# ---------------------------------------------------------------------------
# 口味固化三旋钮：时间衰减 / 负反馈（别推）/ 方向配额
# ---------------------------------------------------------------------------

BLOCKLIST_PATH = PROJECT_ROOT / "data" / "blocklist.json"  # 负推词持久化（「别推 xx」命令写入）
QUOTA_CAP = 3  # 方向配额：top5 中单一方向最多 3 篇


def recency_weight(ts: float, now: float | None = None) -> float:
    """时间衰减：≤30 天 1.0；≤90 天 0.6；更早 0.3；无时间按 0.6。"""
    if not ts:
        return 0.6
    days = ((now if now is not None else time.time()) - ts) / 86400
    if days <= 30:
        return 1.0
    if days <= 90:
        return 0.6
    return 0.3


def _record_ts(fields: dict[str, Any]) -> float:
    """论文库行的精读时间（bitable 毫秒时间戳 → 秒）。"""
    v = fields.get("精读时间")
    if isinstance(v, (int, float)) and v > 0:
        return float(v) / 1000.0
    return 0.0


def taste_keyword_weights(paper_records: list[Any], now: float | None = None) -> dict[str, float]:
    """关键词 → 权重：含该词的论文里时间衰减最大的那条（库内口味衰减版）。"""
    weights: dict[str, float] = {}
    for rec in paper_records:
        fields = record_fields(rec)
        w = recency_weight(_record_ts(fields), now)
        for kw in _multi_values(fields.get("关键词")) + _multi_values(fields.get("方向")):
            if w > weights.get(kw, 0.0):
                weights[kw] = w
    return weights


def keyword_score_weighted(entry: ArxivEntry, weights: dict[str, float]) -> float:
    """加权关键词打分：命中词的权重求和。"""
    haystack = f"{entry.title}\n{entry.abstract}".lower()
    return float(sum(w for kw, w in weights.items() if kw.lower() in haystack))


def load_blocklist(path: Path | str = BLOCKLIST_PATH) -> list[str]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return [str(t) for t in data if str(t).strip()] if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_blocklist(terms: list[str], path: Path | str = BLOCKLIST_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(terms, ensure_ascii=False, indent=1), encoding="utf-8")


def add_blocklist(term: str, path: Path | str = BLOCKLIST_PATH) -> list[str]:
    terms = load_blocklist(path)
    term = term.strip()
    if term and term not in terms:
        terms.append(term)
        save_blocklist(terms, path)
    return terms


def remove_blocklist(term: str, path: Path | str = BLOCKLIST_PATH) -> tuple[list[str], bool]:
    terms = load_blocklist(path)
    term = term.strip()
    removed = term in terms
    if removed:
        terms.remove(term)
        save_blocklist(terms, path)
    return terms, removed


def is_blocked(entry: ArxivEntry, blocklist: list[str]) -> bool:
    haystack = f"{entry.title}\n{entry.abstract}".lower()
    return any(t.lower() in haystack for t in blocklist if t)


POST_TRAINING_KWS = (
    "post-training", "rlhf", "grpo", "dpo", "reinforcement learning",
    "reward model", "alignment", "fine-tuning", "sft", "rlvr",
)
AGENT_KWS = (
    "agent", "harness", "tool use", "tool-use", "function calling",
    "workflow", "multi-agent", "benchmark",
)


def classify_direction(entry: ArxivEntry) -> str:
    """轻量方向分类（配额用）：后训练 / Agent / 其他。"""
    haystack = f"{entry.title}\n{entry.abstract}".lower()
    if any(k in haystack for k in POST_TRAINING_KWS):
        return "后训练"
    if any(k in haystack for k in AGENT_KWS):
        return "Agent"
    return "其他"


def select_top_with_quota(
    ranked: list[tuple[ArxivEntry, float]], n: int = TOP_N, cap: int = QUOTA_CAP
) -> list[tuple[ArxivEntry, float]]:
    """方向配额选取：按分数贪心，单一方向最多 cap 篇；不足 n 篇时放开配额补齐。"""
    picked: list[tuple[ArxivEntry, float]] = []
    counts: dict[str, int] = {}
    for entry, score in ranked:
        direction = classify_direction(entry)
        if counts.get(direction, 0) >= cap:
            continue
        picked.append((entry, score))
        counts[direction] = counts.get(direction, 0) + 1
        if len(picked) >= n:
            return picked
    if len(picked) < n:
        chosen = {id(e) for e, _ in picked}
        for entry, score in ranked:
            if id(entry) in chosen:
                continue
            picked.append((entry, score))
            if len(picked) >= n:
                break
    return picked


# ---- [V7.1 清理] 已停用：生产 _taste_scorer 直接调 vector_index.search。经全项目核对生产链路零引用，保留备查 ----
# def vector_score(entry: ArxivEntry, vector_index: Any) -> float:
#     """与论文库最近一篇的相似度（1 - cosine 距离）；无命中/异常返回 0。
#
#     检索排除论文自身（exclude_arxiv_id）：已入库论文自我命中距离≈0，不算口味证据。"""
#     try:
#         hits = vector_index.search(entry.title, entry.abstract[:300], top_k=1,
#                                    exclude_arxiv_id=entry.arxiv_id)
#     except Exception as exc:
#         logger.warning("向量打分失败（按 0 计）：%s", exc)
#         return 0.0
#     if not hits:
#         return 0.0
#     return max(0.0, 1.0 - float(hits[0][2]))


def rank_entries(
    entries: list[ArxivEntry], scorer: Callable[[ArxivEntry], float]
) -> list[tuple[ArxivEntry, float]]:
    """按 scorer 打分降序；同分按发布日期新→旧。纯函数。"""
    scored = [(entry, float(scorer(entry))) for entry in entries]
    scored.sort(key=lambda t: (t[1], t[0].published), reverse=True)
    return scored


def merge_authority(
    ranked: list[tuple[ArxivEntry, float]],
    authority_fn: Callable[[ArxivEntry], tuple[float, list[str]]],
    taste_weight: float = TASTE_WEIGHT,
    authority_weight: float = AUTHORITY_WEIGHT,
) -> tuple[list[tuple[ArxivEntry, float]], dict[str, list[str]]]:
    """口味分批内归一（最高=1.0，全 0 则全 0）后与权威分加权合并重排。

    返回 (重排列表, arxiv_id→权威命中原因)；权威原因供卡片标注。纯函数。
    """
    if not ranked:
        return [], {}
    max_score = max(score for _, score in ranked)
    reasons: dict[str, list[str]] = {}
    merged: list[tuple[ArxivEntry, float]] = []
    for entry, score in ranked:
        auth, hits = authority_fn(entry)
        reasons[entry.arxiv_id] = hits
        norm = score / max_score if max_score > 0 else 0.0
        merged.append((entry, taste_weight * norm + authority_weight * auth))
    merged.sort(key=lambda t: (t[1], t[0].published), reverse=True)
    return merged, reasons


# ---------------------------------------------------------------------------
# TL;DR 生成与容错解析（逐篇单调用：批处理长 prompt 会触发思考型 light 模型
# 把输出预算耗尽于 reasoning、content 为空——单篇小 prompt 规避，且单篇失败
# 不拖垮整批）
# ---------------------------------------------------------------------------

def parse_tldr_json(raw: str) -> dict[str, str] | None:
    """单篇 TL;DR 输出的容错解析，返回 {"tldr","reason"}；失败 None。"""
    data = extract_json(raw)
    if data is None:
        return None
    return {
        "tldr": str(data.get("tldr") or "").strip(),
        "reason": str(data.get("reason") or "").strip(),
    }


def parse_quality_json(raw: str) -> dict[str, Any] | None:
    """质量评审输出的容错解析，返回 {"score": int, "risk": str}；失败 None。"""
    data = extract_json(raw)
    if data is None:
        return None
    try:
        score = int(data.get("score"))
    except (TypeError, ValueError):
        return None
    return {"score": max(1, min(5, score)), "risk": str(data.get("risk") or "").strip()}


def parse_pick_review_json(raw: str) -> dict[str, str] | None:
    """每周精选深评输出的容错解析，返回 {"tldr","why","relation","risk"}；失败 None。"""
    data = extract_json(raw)
    if data is None:
        return None
    return {
        "tldr": str(data.get("tldr") or "").strip(),
        "why": str(data.get("why") or "").strip(),
        "relation": str(data.get("relation") or "").strip(),
        "risk": str(data.get("risk") or "").strip(),
    }


# ---------------------------------------------------------------------------
# 精选库归档（日报/每周精选 top 沉淀，按 arXivID 全量查重跨天去重）
# ---------------------------------------------------------------------------

def build_pick_row(
    entry: ArxivEntry, item: dict[str, str], now: float | None = None,
    source: str = PICK_SOURCE,
) -> dict[str, Any]:
    """一篇日报/每周精选 → 精选库 bitable 行（字段名/值格式对照 bitable.py schema）。"""
    now_ms = int((now if now is not None else time.time()) * 1000)
    return {
        "标题": entry.title,
        "日期": now_ms,
        "链接": {"text": entry.title, "link": entry.url},  # 超链接字段
        "arXivID": entry.arxiv_id,
        "TL;DR": item.get("tldr", ""),
        "推荐理由": item.get("reason") or item.get("why", ""),  # 每周精选用深评的 why
        "状态": "未读",
        "来源": source,
    }


def pick_direction(entry: ArxivEntry) -> str:
    """精选库方向字段：复用轻量方向分类，"其他"归为未分类。"""
    direction = classify_direction(entry)
    return direction if direction != "其他" else "未分类"


def interest_hits(entry: ArxivEntry, interests: list[str]) -> list[str]:
    """标题+摘要命中的显式兴趣词（精选库关键词字段用，保序）。"""
    haystack = f"{entry.title}\n{entry.abstract}".lower()
    return [kw for kw in interests if kw.lower() in haystack]


def existing_pick_ids(records: list[Any]) -> set[str]:
    """已归档记录 → {arXivID} 查重集合（全量跨天去重，与归档日期无关）。纯函数。"""
    ids: set[str] = set()
    for rec in records:
        aid = field_text(record_fields(rec).get("arXivID")).strip()
        if aid:
            ids.add(aid)
    return ids


@dataclass
class PushHistory:
    """推送历史（跨天排除的数据基础）：三类 arXivID 集合。

    pushed：精选库全部（日报/每周精选推过的，日报不再隔天重复推）；
    curated：其中上过每周精选的（深评策展过的不重复策展；仅日报来源的留给
    周刊再策展，归档走 upgrade_on_dup 升级原记录）；
    read：论文库已精读（读过的任何推送都不再出现，也杜绝向量自我命中过闸）。
    """
    pushed: set[str] = field(default_factory=set)
    curated: set[str] = field(default_factory=set)
    read: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# 今日提醒：数据归集与翻坟（纯函数）
# ---------------------------------------------------------------------------

@dataclass
class TodayItems:
    memos: list[dict[str, str]] = field(default_factory=list)   # {"事项","截止","逾期"}
    followups: list[dict[str, str]] = field(default_factory=list)  # 看板跟进 {"公司","岗位","原因"}
    resurface: list[dict[str, str]] = field(default_factory=list)  # 收藏翻坟 {"标题","链接","摘要"}


def pick_resurface(collect_records: list[Any], n: int = RESURFACE_COUNT) -> list[dict[str, str]]:
    """收藏翻坟：已回顾=false，按收藏时间最老优先取 n 条。纯函数。"""
    candidates = []
    for rec in collect_records:
        fields = record_fields(rec)
        if fields.get("已回顾") is True:
            continue
        collected_at = fields.get("收藏时间")
        candidates.append((
            collected_at if isinstance(collected_at, (int, float)) else float("inf"),
            fields,
        ))
    candidates.sort(key=lambda t: t[0])  # 无时间的沉底，最老优先
    out = []
    for _, fields in candidates[:n]:
        link = fields.get("链接")
        url = link.get("link", "") if isinstance(link, dict) else field_text(link)
        out.append({
            "标题": field_text(fields.get("标题")) or "未命名收藏",
            "链接": url,
            "摘要": field_text(fields.get("摘要")),
        })
    return out


def collect_today_items(io: Any, now: float | None = None) -> TodayItems:
    """归集今日提醒素材：备忘 72h（含逾期）+ 看板跟进 + 收藏翻坟。"""
    now_ms = int((now if now is not None else time.time()) * 1000)
    window_ms = MEMO_WINDOW_HOURS * 3600_000
    items = TodayItems()

    memos: list[tuple[float, dict[str, str]]] = []
    for rec in io.list_all(MEMO_TABLE):
        fields = record_fields(rec)
        if field_text(fields.get("状态")) not in ("", "待办"):
            continue
        deadline = fields.get("截止时间")
        if not isinstance(deadline, (int, float)):
            continue  # 无截止时间的备忘不进每日提醒（收录时已提示）
        if deadline > now_ms + window_ms:
            continue  # 超出 72h 窗口
        memos.append((float(deadline), {
            "事项": field_text(fields.get("事项")),
            "截止": time.strftime("%m-%d", time.localtime(deadline / 1000)),
            "逾期": "（已逾期）" if deadline < now_ms else "",
        }))
    # 按截止时间毫秒值排序：MM-DD 字符串排序跨年/月会错乱（"01-02" 排在 "12-30" 前）
    memos.sort(key=lambda t: t[0])
    items.memos = [m for _, m in memos]

    try:
        summary = summarize_board(io.list_all(BOARD_TABLE), now=now)
        items.followups = summary.followups
    except BitableError as exc:
        logger.warning("今日提醒读取求职看板失败（跳过该板块）：%s", exc)

    items.resurface = pick_resurface(io.list_all(COLLECT_TABLE))
    return items


def build_today_message(items: TodayItems, now: float | None = None, *, evening: bool = False) -> str:
    """今日提醒汇总文本（含生成时间）；evening=True 为晚间回顾档（🌙 标题）。"""
    now = now if now is not None else time.time()
    today = time.strftime("%Y-%m-%d", time.localtime(now))
    gen = time.strftime("%H:%M", time.localtime(now))
    title = "🌙 晚间回顾" if evening else "🌅 今日提醒"
    lines = [f"{title}（{today}，生成于 {gen}）"]

    if not items.memos and not items.followups:
        lines.append("\n✅ 今日无待办事项。")
    else:
        if items.memos:
            lines.append(f"\n📌 备忘（72 小时内截止，{len(items.memos)} 条）")
            for i, m in enumerate(items.memos, 1):
                lines.append(f"{i}. {m['事项']}｜截止 {m['截止']}{m['逾期']}")
        if items.followups:
            lines.append(f"\n💼 求职跟进（{len(items.followups)} 条）")
            for i, f in enumerate(items.followups, 1):
                target = " ".join(p for p in (f["公司"], f["岗位"]) if p)
                lines.append(f"{i}. {target}：{f['原因']}")

    if items.resurface:
        lines.append("\n📚 收藏翻坟（吃灰最久的，捞一条看看）")
        for i, c in enumerate(items.resurface, 1):
            entry = f"{i}. 《{c['标题']}》"
            if c["摘要"]:
                entry += f"——{c['摘要'][:60]}"
            lines.append(entry)
            if c["链接"]:
                lines.append(f"   {c['链接']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 周报：数据聚合与 prompt 组装
# ---------------------------------------------------------------------------

@dataclass
class WeeklyMaterials:
    papers: list[str] = field(default_factory=list)       # 每条：标题+TL;DR+关键词
    collections: list[str] = field(default_factory=list)  # 每条：标题+摘要
    ideas: list[str] = field(default_factory=list)        # 每条：内容+标签


def collect_weekly_materials(io: Any, now: float | None = None) -> WeeklyMaterials:
    """近 7 天的精读论文 / 收藏 / 灵感聚合（周报数据源）。"""
    now_ms = int((now if now is not None else time.time()) * 1000)
    since_ms = now_ms - WEEKLY_DAYS * 86_400_000
    materials = WeeklyMaterials()

    for rec in io.list_all(PAPER_TABLE):
        fields = record_fields(rec)
        read_at = fields.get("精读时间")
        if not isinstance(read_at, (int, float)) or read_at < since_ms:
            continue
        kw = "、".join(_multi_values(fields.get("关键词")))
        line = f"《{field_text(fields.get('标题'))}》"
        tldr = field_text(fields.get("TL;DR"))
        if tldr:
            line += f"：{tldr}"
        if kw:
            line += f"（关键词：{kw}）"
        materials.papers.append(line)

    for rec in io.list_all(COLLECT_TABLE):
        fields = record_fields(rec)
        collected_at = fields.get("收藏时间")
        if not isinstance(collected_at, (int, float)) or collected_at < since_ms:
            continue
        line = field_text(fields.get("标题")) or "未命名收藏"
        summary = field_text(fields.get("摘要"))
        materials.collections.append(f"{line}：{summary}" if summary else line)

    for rec in io.list_all(IDEA_TABLE):
        fields = record_fields(rec)
        created = fields.get("时间")
        if not isinstance(created, (int, float)) or created < since_ms:
            continue
        line = field_text(fields.get("内容"))
        tags = "、".join(_multi_values(fields.get("标签")))
        materials.ideas.append(f"{line}（{tags}）" if tags else line)

    return materials


def build_weekly_prompt(materials: WeeklyMaterials, now: float | None = None) -> str:
    """周报 prompt 组装（模板 + 三料，空板块给明确占位）。"""
    now = now if now is not None else time.time()
    start = time.strftime("%m-%d", time.localtime(now - WEEKLY_DAYS * 86400))
    end = time.strftime("%m-%d", time.localtime(now))
    template = WEEKLY_PROMPT_PATH.read_text(encoding="utf-8")
    return (
        template
        .replace("{{DATE_RANGE}}", f"{start} ~ {end}（近 {WEEKLY_DAYS} 天）")
        .replace("{{PAPERS}}", "\n".join(materials.papers) or "（本周无精读）")
        .replace("{{COLLECTIONS}}", "\n".join(materials.collections) or "（本周无收藏）")
        .replace("{{IDEAS}}", "\n".join(materials.ideas) or "（本周无灵感）")
    )


# ---------------------------------------------------------------------------
# 日报卡片 / 文本
# ---------------------------------------------------------------------------

def _md_element(content: str) -> dict:
    return {"tag": "markdown", "content": content}


def _paper_actions(item: dict[str, str]) -> dict:
    """每篇一组按钮：精读这篇 / 已读（card.action.trigger 回调）。

    已读的按钮直接显示 ✓ 已读（状态来自精选库，未读显示「已读」待点）。"""
    read = str(item.get("status") or "") == "已读"
    return {
        "tag": "action",
        "actions": [
            {"tag": "button",
             "text": {"tag": "plain_text", "content": "📖 精读这篇"},
             "type": "primary",
             "value": {"act": "read_paper", "arxiv_id": item["arxiv_id"],
                       "title": item["title"][:60]}},
            {"tag": "button",
             "text": {"tag": "plain_text", "content": "✓ 已读" if read else "已读"},
             "type": "default",
             "value": {"act": "mark_read", "arxiv_id": item["arxiv_id"],
                       "title": item["title"][:60]}},
        ],
    }


def build_arxiv_card(
    items: list[dict[str, str]], scanned: int, days: int, now: float | None = None,
    archived: int | None = None,
) -> dict:
    """日报卡片：每篇 标题/链接/TL;DR/推荐理由/权威标注/风险点，note 附生成时间与扫描量。"""
    now = now if now is not None else time.time()
    today = time.strftime("%Y-%m-%d", time.localtime(now))
    gen = time.strftime("%H:%M", time.localtime(now))
    elements: list[dict] = []
    for i, item in enumerate(items, 1):
        block = f"**{i}. [{item['title']}]({item['url']})**\n{item['tldr']}"
        if item.get("reason"):
            block += f"\n💡 {item['reason']}"
        if item.get("authority"):
            block += f"\n🏛️ {item['authority']}"
        if item.get("risk"):
            block += f"\n⚠️ 风险：{item['risk']}"
        elements.append(_md_element(block))
        if item.get("arxiv_id"):
            elements.append(_paper_actions(item))
    footer = f"生成于 {today} {gen} · arXiv 近 {days} 天共扫描 {scanned} 篇，按你的论文库口味精选"
    if archived is not None:
        footer += f" · 已归档精选库 {archived} 篇（面板可看状态）"
    elements.append({
        "tag": "note",
        "elements": [{"tag": "plain_text", "content": footer}],
    })
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "blue",
            "title": {"tag": "plain_text", "content": f"📰 arXiv 日报（{today}）Top {len(items)}"},
        },
        "elements": elements,
    }


def build_arxiv_text(items: list[dict[str, str]], scanned: int, days: int,
                     now: float | None = None, archived: int | None = None) -> str:
    """日报纯文本版（命令回显 / 日志 / 推送降级用）。"""
    now = now if now is not None else time.time()
    today = time.strftime("%Y-%m-%d", time.localtime(now))
    gen = time.strftime("%H:%M", time.localtime(now))
    lines = [f"📰 arXiv 日报（{today}）Top {len(items)}，近 {days} 天扫描 {scanned} 篇："]
    for i, item in enumerate(items, 1):
        lines.append(f"{i}. {item['title']}\n   {item['url']}\n   {item['tldr']}")
        if item.get("reason"):
            lines.append(f"   💡 {item['reason']}")
        if item.get("authority"):
            lines.append(f"   🏛️ {item['authority']}")
        if item.get("risk"):
            lines.append(f"   ⚠️ 风险：{item['risk']}")
    footer = f"（生成于 {gen}）"
    if archived is not None:
        footer = f"（生成于 {gen} · 已归档精选库 {archived} 篇）"
    lines.append(footer)
    return "\n".join(lines)


def build_weekly_picks_card(
    items: list[dict[str, str]], scanned: int, days: int, now: float | None = None,
    archived: int | None = None,
) -> dict:
    """每周精选卡片：每篇 标题/TL;DR/为什么值得读/与你方向/权威标注/风险点 + 精读/已读按钮。"""
    now = now if now is not None else time.time()
    lt = time.localtime(now)
    today = time.strftime("%Y-%m-%d", lt)
    gen = time.strftime("%H:%M", lt)
    md = f"{lt.tm_mon}.{lt.tm_mday}"
    elements: list[dict] = []
    for i, item in enumerate(items, 1):
        block = f"**{i}. [{item['title']}]({item['url']})**\n{item['tldr']}"
        if item.get("why"):
            block += f"\n📖 {item['why']}"
        if item.get("relation"):
            block += f"\n🎯 {item['relation']}"
        if item.get("authority"):
            block += f"\n🏛️ {item['authority']}"
        if item.get("risk"):
            block += f"\n⚠️ 风险：{item['risk']}"
        elements.append(_md_element(block))
        if item.get("arxiv_id"):
            elements.append(_paper_actions(item))
    footer = f"生成于 {today} {gen} · arXiv 近 {days} 天共扫描 {scanned} 篇，严选 {len(items)} 篇 · 周末阅读愉快"
    if archived is not None:
        footer += f" · 已归档精选库 {archived} 篇"
    elements.append({
        "tag": "note",
        "elements": [{"tag": "plain_text", "content": footer}],
    })
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "purple",
            "title": {"tag": "plain_text", "content": f"📌 本周精选（{md} · 周末阅读）"},
        },
        "elements": elements,
    }


def build_weekly_picks_text(items: list[dict[str, str]], scanned: int, days: int,
                            now: float | None = None, archived: int | None = None) -> str:
    """每周精选纯文本版（命令回显 / 日志 / 推送降级用）。"""
    now = now if now is not None else time.time()
    lt = time.localtime(now)
    md = f"{lt.tm_mon}.{lt.tm_mday}"
    gen = time.strftime("%H:%M", lt)
    lines = [f"📌 本周精选（{md} · 周末阅读），近 {days} 天扫描 {scanned} 篇，严选 {len(items)} 篇："]
    for i, item in enumerate(items, 1):
        lines.append(f"{i}. {item['title']}\n   {item['url']}\n   {item['tldr']}")
        if item.get("why"):
            lines.append(f"   📖 {item['why']}")
        if item.get("relation"):
            lines.append(f"   🎯 {item['relation']}")
        if item.get("authority"):
            lines.append(f"   🏛️ {item['authority']}")
        if item.get("risk"):
            lines.append(f"   ⚠️ 风险：{item['risk']}")
    footer = f"（生成于 {gen}）"
    if archived is not None:
        footer = f"（生成于 {gen} · 已归档精选库 {archived} 篇）"
    lines.append(footer)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 服务与调度
# ---------------------------------------------------------------------------

def _fmt_now(now: float | None = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(now if now is not None else time.time()))


class DigestService:
    """日报/今日/周报的内容生成与推送。命令触发与定时调度共用 now() 系列。"""

    def __init__(
        self,
        settings: Settings,
        kimi: KimiClient,
        sender: Sender | None = None,
        io: Any = None,
        docx: Any = None,
        vector_index: Any = None,
        fetcher: Callable[[int], list[ArxivEntry]] | None = None,
        authority_scorer: Callable[[ArxivEntry], tuple[float, list[str]]] | None = None,
        chat_id: str = "",
    ) -> None:
        self.settings = settings
        self.kimi = kimi
        self.sender = sender
        self.io = io  # CareerIO 协议（list_all / .dao.add_record）
        self.docx = docx
        self.vector_index = vector_index
        self.fetcher = fetcher or fetch_arxiv_entries  # 参数为回溯天数（日报 2 / 每周精选 7）
        self._authority = authority_scorer or authority_score  # 测试注入打桩，默认真实（S2 失败静默）
        self.chat_id = chat_id  # 定时推送目标（config.yaml push.chat_id，接线注入）
        self._pick_fields_ensured = False

    # ---- 推送（失败重试一次，仍败记 ERROR）----

    def _push(self, chat_id: str, what: str, *, text: str = "", card: dict | None = None) -> bool:
        if self.sender is None or not chat_id:
            logger.warning("推送%s跳过：sender=%s chat_id=%r", what, self.sender is not None, chat_id)
            return False

        def _send() -> bool:
            if card is not None:
                return bool(self.sender.send_card(chat_id, card))
            return bool(self.sender.send_text(chat_id, text))

        if _send():
            return True
        logger.warning("推送%s失败，重试一次：chat=%s", what, chat_id)
        if _send():
            return True
        logger.error("推送%s失败（已重试一次）：chat=%s", what, chat_id)
        return False

    def _run_safely(self, what: str, chat_id: str, job: Callable[[], str]) -> str:
        """内容生成失败的统一兜底：生成错误文本并尽力推送，返回该文本。"""
        try:
            return job()
        except (DigestError, KimiError, BitableError, DocxError) as exc:
            logger.error("%s生成失败：%s", what, exc)
            fallback = f"❌ {what}生成失败：{exc}（{_fmt_now()}）"
        except Exception as exc:  # 未知异常同样不能炸调度器线程
            logger.exception("%s生成出现未知错误", what)
            fallback = f"❌ {what}生成出现未知错误：{type(exc).__name__}（{_fmt_now()}）"
        self._push(chat_id, what, text=fallback)
        return fallback

    def _start_async(self, label: str, job: Callable[[], str]) -> None:
        threading.Thread(
            target=lambda: self._run_safely(label, self.chat_id, job),
            name=f"digest-{int(time.time())}", daemon=True,
        ).start()

    # ---- arXiv 日报 ----

    def _taste_scorer(self) -> Callable[[ArxivEntry], float]:
        """向量索引可用且非空 → 向量打分×时间衰减+兴趣加成；否则加权关键词合并打分。
        负推词（别推）命中直接 -999 淘汰。"""
        interests = list(self.settings.push_interests)
        blocklist = load_blocklist()

        def _finish(fn: Callable[[ArxivEntry], float]) -> Callable[[ArxivEntry], float]:
            return lambda entry: -999.0 if is_blocked(entry, blocklist) else fn(entry)

        if self.vector_index is not None:
            try:
                if self.vector_index.count() > 0:
                    date_by_id = self._paper_dates()
                    # 向量相似度 × 匹配论文的时间衰减 + 兴趣命中加成
                    def _vec(entry: ArxivEntry) -> float:
                        try:
                            # exclude_arxiv_id：已入库论文自我命中（距离≈0）不算口味证据
                            hits = self.vector_index.search(
                                entry.title, entry.abstract[:300], top_k=1,
                                exclude_arxiv_id=entry.arxiv_id)
                        except Exception as exc:
                            logger.warning("向量打分失败（按 0 计）：%s", exc)
                            hits = []
                        base = max(0.0, 1.0 - float(hits[0][2])) if hits else 0.0
                        decay = recency_weight(date_by_id.get(hits[0][0], 0.0)) if hits else 1.0
                        return base * decay + keyword_score(entry, interests)
                    return _finish(_vec)
            except Exception as exc:
                logger.warning("向量索引不可用（退关键词打分）：%s", exc)
        weights: dict[str, float] = {}
        if self.io is not None:
            try:
                weights = taste_keyword_weights(self.io.list_all(PAPER_TABLE))
            except BitableError as exc:
                logger.warning("读取论文库口味失败（仅按显式兴趣打分）：%s", exc)
        for kw in interests:  # 显式兴趣恒定满权重，冷启动期兜底
            weights.setdefault(kw, 1.0)
        return _finish(lambda entry: keyword_score_weighted(entry, weights))

    def _paper_dates(self) -> dict[str, float]:
        """论文库 arXivID → 精读时间（秒），供向量命中的时间衰减查询。"""
        if self.io is None:
            return {}
        try:
            dates: dict[str, float] = {}
            for rec in self.io.list_all(PAPER_TABLE):
                fields = record_fields(rec)
                arxiv_id = str(fields.get("arXivID") or "").strip()
                if arxiv_id:
                    dates[arxiv_id] = _record_ts(fields)
            return dates
        except BitableError as exc:
            logger.warning("读取论文日期失败（时间衰减按满权重）：%s", exc)
            return {}

    def _push_history(self) -> PushHistory:
        """读精选库 + 论文库构建推送历史（跨天排除用）。

        io 缺失或单表读取失败时对应集合按空集（不阻断推送主流程，记 WARNING）。
        """
        history = PushHistory()
        if self.io is None:
            return history
        try:
            for rec in self.io.list_all(PICK_TABLE):
                fields = record_fields(rec)
                aid = field_text(fields.get("arXivID")).strip()
                if not aid:
                    continue
                history.pushed.add(aid)
                if PICK_SOURCE_WEEKLY in field_text(fields.get("来源")):
                    history.curated.add(aid)
        except BitableError as exc:
            logger.warning("读取精选库推送历史失败（本次不做跨天排除）：%s", exc)
            history.pushed.clear()
            history.curated.clear()
        try:
            for rec in self.io.list_all(PAPER_TABLE):
                aid = field_text(record_fields(rec).get("arXivID")).strip()
                if aid:
                    history.read.add(aid)
        except BitableError as exc:
            logger.warning("读取论文库精读历史失败（本次不排除已精读）：%s", exc)
        return history

    def _taste_text(self) -> str:
        base = ""
        if self.io is not None:
            try:
                base = "、".join(taste_keywords_from_papers(self.io.list_all(PAPER_TABLE))[:30])
            except BitableError:
                base = "（论文库读取失败）"
        else:
            base = "（论文库不可用）"
        interests = "、".join(self.settings.push_interests[:10])
        if interests:
            return f"{base}；显式兴趣：{interests}" if base and not base.startswith("（") else f"显式兴趣：{interests}"
        return base or "（论文库为空，按方向通用口味）"

    def _summarize_entries(self, entries: list[ArxivEntry],
                           risks: dict[str, str] | None = None) -> list[dict[str, str]]:
        """top 论文逐篇 light 生成 TL;DR + 推荐理由；单篇失败单篇兜底（摘要截断）。"""
        template = ARXIV_PROMPT_PATH.read_text(encoding="utf-8")
        taste = self._taste_text()
        risks = risks or {}
        items = []
        for entry in entries:
            paper_block = f"{entry.title}\n作者：{entry.authors}\n摘要：{entry.abstract[:800]}"
            prompt = template.replace("{{PAPER}}", paper_block).replace("{{TASTE}}", taste)
            digest: dict[str, str] | None = None
            try:
                raw = self.kimi.chat([{"role": "user", "content": prompt}], heavy=False)
                digest = parse_tldr_json(raw)
            except KimiError as exc:
                logger.warning("日报 TL;DR 单篇模型失败（兜底）：%s", exc)
            if digest is None:
                logger.warning("日报 TL;DR 单篇解析失败（兜底）：%s", entry.title[:40])
                digest = {"tldr": "", "reason": ""}
            tldr = digest["tldr"] or (
                entry.abstract[:80] + "…" if len(entry.abstract) > 80 else entry.abstract
            )
            items.append({"title": entry.title, "url": entry.url, "arxiv_id": entry.arxiv_id,
                          "tldr": tldr, "reason": digest["reason"],
                          "risk": risks.get(entry.arxiv_id, "")})
        return items

    def _quality_review(self, entry: ArxivEntry) -> dict[str, Any]:
        """质量关：单篇 light 快速评审（重要性/新意/实验 1-5 分 + 风险点）；失败按中性 3 分。"""
        try:
            template = QUALITY_PROMPT_PATH.read_text(encoding="utf-8")
        except OSError:
            return {"score": 3, "risk": ""}
        paper_block = f"{entry.title}\n作者：{entry.authors}\n摘要：{entry.abstract[:800]}"
        prompt = template.replace("{{PAPER}}", paper_block)
        try:
            raw = self.kimi.chat([{"role": "user", "content": prompt}], heavy=False)
        except KimiError as exc:
            logger.warning("质量评审单篇模型失败（按 3 分）：%s", exc)
            return {"score": 3, "risk": ""}
        review = parse_quality_json(raw)
        if review is None:
            logger.warning("质量评审单篇解析失败（按 3 分）：%s", entry.title[:40])
            return {"score": 3, "risk": ""}
        return review

    def _quality_gate(self, ranked: list[tuple[ArxivEntry, float]]
                      ) -> tuple[list[tuple[ArxivEntry, float]], dict[str, str]]:
        """主题分 top K → 逐篇质量评审 → 综合分重排。返回 (重排结果, arxiv_id→风险点)。"""
        candidates = ranked[:QUALITY_K]
        risks: dict[str, str] = {}
        rescored: list[tuple[ArxivEntry, float]] = []
        for entry, topic_score in candidates:
            review = self._quality_review(entry)
            if review["risk"]:
                risks[entry.arxiv_id] = review["risk"]
            final = topic_score + review["score"] / 5.0 * QUALITY_BONUS
            rescored.append((entry, final))
            logger.info("质量评审：%s 主题 %.1f + 质量 %d → %.1f", entry.arxiv_id, topic_score, review["score"], final)
        # 未进候选的尾部保持主题分垫底，候选内按综合分重排
        tail = [(e, s) for e, s in ranked[QUALITY_K:]]
        rescored.sort(key=lambda t: (t[1], t[0].published), reverse=True)
        return rescored + tail, risks

    def _ensure_pick_fields(self) -> None:
        """给已有精选库补 schema 演进字段（方向/关键词），进程内只做一次。

        周刊归档会写这两个字段（extras）；表上缺字段时 add_record 被飞书拒绝，
        历史上因此整批归档静默为 0（FakeIO 不校验字段，测试层发现不了）。
        """
        if self._pick_fields_ensured or self.io is None:
            return
        self._pick_fields_ensured = True
        dao = self.io.dao
        tables = getattr(dao, "tables", None)  # 测试替身无此属性时跳过（生产 BitableDAO 必有）
        client = getattr(self.io, "client", None)
        if not tables or client is None:
            return
        table_id = tables.get(PICK_TABLE, "")
        if not table_id:
            return
        try:
            created = ensure_fields(
                client, dao.app_token, table_id,
                [FieldSpec("方向", FIELD_TYPE_SINGLE_SELECT,
                           options=PAPER_DIRECTIONS + ("未分类",)),
                 FieldSpec("关键词", FIELD_TYPE_MULTI_SELECT)],
            )
            if created:
                logger.info("精选库补充字段：%s", "、".join(created))
        except BitableError as exc:
            logger.warning("精选库补字段失败（写入时若缺字段会再报错）：%s", exc)

    def _archive_picks(
        self,
        entries: list[ArxivEntry],
        items: list[dict[str, str]],
        source: str = PICK_SOURCE,
        extras: list[dict[str, Any]] | None = None,
        upgrade_on_dup: bool = False,
    ) -> int:
        """推送 top 逐篇写入精选库（按 arXivID 全量查重，跨天去重）。返回新归档篇数。

        extras：逐篇附加字段（每周精选的方向/关键词）。表缺失/读写失败仅记
        WARNING（归档是沉淀增强，绝不影响推送主流程）。
        upgrade_on_dup：查重命中时，用本次深评升级已有记录的推荐理由
        （每周精选的深评比日报一句话更值，覆盖过去不亏）；查重不看归档日期，
        所以日报推过的篇目上周刊时升级真正生效。
        批次语义：同一批归档的行共用同一 日期 毫秒时间戳——精选库「最新一批」
        应以 max(日期) 取整批再按 来源 区分（日报每批 ≤3、周刊每批 ≤5），
        按时间倒序切固定条数会把相邻两批混为一批。
        """
        if self.io is None:
            return 0
        try:
            existing = existing_pick_ids(self.io.list_all(PICK_TABLE))
            batch_now = time.time()  # 同批同 日期：「最新一批」按 日期 分组的分组键
            self._ensure_pick_fields()  # 周刊 extras 写 方向/关键词，缺字段会被飞书整批拒绝
            archived = 0
            for i, (entry, item) in enumerate(zip(entries, items)):
                if entry.arxiv_id in existing:
                    if upgrade_on_dup:
                        reason = str(
                            item.get("reason") or item.get("why") or item.get("tldr") or ""
                        ).strip()
                        rid = self.io.dao.find_record_by_field(
                            PICK_TABLE, "arXivID", entry.arxiv_id)
                        if rid:
                            # 升级 = 本批再次策展：日期刷到本批批次键，「精读 N」按
                            # max(日期) 取整批时才能指到它（否则序号与卡片错位）。
                            patch: dict[str, Any] = {"日期": int(batch_now * 1000)}
                            if reason:
                                patch["推荐理由"] = reason
                            # 来源合并：旧来源不含本来源则串上（如 arXiv日报+每周精选）
                            prior = self.io.dao.find_record_fields(
                                PICK_TABLE, "arXivID", entry.arxiv_id) or {}
                            prior_src = str(prior.get("来源") or "")
                            if source and source not in prior_src:
                                patch["来源"] = f"{prior_src}+{source}" if prior_src else source
                            if patch:
                                self.io.dao.update_record(PICK_TABLE, rid, patch)
                                logger.info("精选库查重命中，升级记录：%s（%s）",
                                            entry.arxiv_id, "、".join(patch))
                    else:
                        logger.info("精选库查重跳过：%s", entry.arxiv_id)
                    continue
                row = build_pick_row(entry, item, now=batch_now, source=source)
                if extras:
                    row.update(extras[i])
                self.io.dao.add_record(PICK_TABLE, row)
                existing.add(entry.arxiv_id)  # 同批内重复篇目也只入一行
                archived += 1
            if archived:
                logger.info("精选库已归档：%d 篇（来源 %s）", archived, source)
            return archived
        except BitableError as exc:
            logger.warning("精选库归档失败（不影响推送）：%s", exc)
            return 0

    def _pick_status_map(self, arxiv_ids: list[str]) -> dict[str, str]:
        """精选库状态查询（已读/未读/已精读）：卡片按钮初始状态用；失败全未读。"""
        statuses: dict[str, str] = {}
        if self.io is None:
            return statuses
        for aid in arxiv_ids:
            try:
                rid = self.io.dao.find_record_by_field(PICK_TABLE, "arXivID", aid)
                if rid:
                    fields = self.io.dao.find_record_fields(PICK_TABLE, "arXivID", aid) or {}
                    status = fields.get("状态")
                    statuses[aid] = str(status) if isinstance(status, str) and status else "未读"
            except Exception:  # noqa: BLE001（任何 dao 形态差异/网络失败都不许拖垮推送）
                continue
        return statuses

    def arxiv_digest_now(self, chat_id: str | None = None, *,
                         respect_weekend: bool = False) -> str:
        """生成并推送 arXiv 日报，返回纯文本版（命令回显/测试用）。

        跨天去重：抓取窗口 2 天（周一 3 天），先按推送历史排除（精选库已推送 ∪
        论文库已精读的 arXivID）再排序，同一篇不会隔天再推；卡片"扫描 N 篇"
        为排除后的候选数。全部命中历史时与质量闸同级处理（SILENT_WHEN_EMPTY）。
        质量闸：口味分（批内归一）×0.6 + 权威分×0.4 ≥ QUALITY_THRESHOLD 才留，
        0 篇过线时按 SILENT_WHEN_EMPTY 只记日志不推送（用户讨厌被吵）。
        respect_weekend=True（定时任务专用）：周六/周日不推送，周一窗口放宽到
        WEEKEND_DAYS 天（吃周末存量）。手动触发（命令）不受周末限制。
        """
        target = self.chat_id if chat_id is None else chat_id
        weekday = time.localtime().tm_wday  # 0=周一 … 5=周六 6=周日
        if respect_weekend and weekday in (5, 6):
            logger.info("日报周末静默：weekday=%d 不推送", weekday)
            return "周末不推送（周六/周日静默）"
        days = WEEKEND_DAYS if weekday == 0 else ARXIV_DAYS

        def _job() -> str:
            entries = self.fetcher(days)
            if not entries:
                text = (f"📰 arXiv 日报：近 {days} 天没有抓到符合条件的新论文"
                        f"（arXiv 周末/节假日更新少）。（生成于 {_fmt_now()}）")
                self._push(target, "arXiv 日报", text=text)
                return text
            history = self._push_history()
            skip = history.pushed | history.read
            fresh = [e for e in entries if e.arxiv_id not in skip]
            if len(fresh) < len(entries):
                logger.info("日报跨天去重：候选 %d 篇，排除已推送/已精读 %d 篇",
                            len(entries), len(entries) - len(fresh))
            if not fresh:
                text = (f"📰 arXiv 日报：近 {days} 天扫描 {len(entries)} 篇，"
                        f"均已推送过或已精读，今日不打扰。（生成于 {_fmt_now()}）")
                if SILENT_WHEN_EMPTY:
                    logger.info("日报跨天去重：%d 篇候选全部命中推送历史，静默不推送", len(entries))
                else:
                    self._push(target, "arXiv 日报", text=text)
                return text
            entries = fresh
            ranked = rank_entries(entries, self._taste_scorer())
            merged, reasons = merge_authority(ranked, self._authority)
            passed = [(e, s) for e, s in merged if s >= QUALITY_THRESHOLD]
            if not passed:
                text = (f"📰 arXiv 日报：近 {days} 天扫描 {len(entries)} 篇，"
                        f"无一过质量线（{QUALITY_THRESHOLD}），今日不打扰。（生成于 {_fmt_now()}）")
                if SILENT_WHEN_EMPTY:
                    logger.info("日报质量闸：%d 篇候选无一过线（阈值 %.2f），静默不推送",
                                len(entries), QUALITY_THRESHOLD)
                else:
                    self._push(target, "arXiv 日报", text=text)
                return text
            gated, risks = self._quality_gate(passed)
            top = [entry for entry, _ in select_top_with_quota(gated, n=DAILY_TOP_N)]
            items = self._summarize_entries(top, risks)
            statuses = self._pick_status_map([e.arxiv_id for e in top])
            for entry, item in zip(top, items):  # 每篇附权威命中原因（卡片标注）
                item["authority"] = " · ".join(reasons.get(entry.arxiv_id, []))
                item["status"] = statuses.get(entry.arxiv_id, "未读")
            archived = self._archive_picks(top, items)
            card = build_arxiv_card(items, scanned=len(entries), days=days, archived=archived)
            pushed = self._push(target, "arXiv 日报", card=card)
            text = build_arxiv_text(items, scanned=len(entries), days=days, archived=archived)
            if not pushed:  # 卡片推送失败 → 纯文本兜底推一次
                self._push(target, "arXiv 日报（文本版）", text=text)
            return text

        return self._run_safely("arXiv 日报", target, _job)

    # ---- 今日提醒 ----

    def today_digest_now(self, chat_id: str | None = None) -> str:
        """生成并推送今日提醒，返回文本。"""
        target = self.chat_id if chat_id is None else chat_id

        def _job() -> str:
            if self.io is None:
                raise DigestError("多维表格未初始化，无法生成今日提醒。")
            text = build_today_message(collect_today_items(self.io))
            self._push(target, "今日提醒", text=text)
            return text

        return self._run_safely("今日提醒", target, _job)

    # ---- 晚间回顾（今日提醒晚间档：push.evening_time，默认 22:00）----

    def evening_digest_now(self, chat_id: str | None = None) -> str:
        """生成并推送晚间回顾（同今日提醒素材，🌙 标题），返回文本。"""
        target = self.chat_id if chat_id is None else chat_id

        def _job() -> str:
            if self.io is None:
                raise DigestError("多维表格未初始化，无法生成晚间回顾。")
            text = build_today_message(collect_today_items(self.io), evening=True)
            self._push(target, "晚间回顾", text=text)
            return text

        return self._run_safely("晚间回顾", target, _job)

    # ---- 组会周报 ----

    def weekly_report_now(self, chat_id: str | None = None) -> str:
        """生成周报文档并推送链接，返回文本。"""
        target = self.chat_id if chat_id is None else chat_id

        def _job() -> str:
            if self.io is None:
                raise DigestError("多维表格未初始化，无法生成周报。")
            materials = collect_weekly_materials(self.io)
            prompt = build_weekly_prompt(materials)
            body = self.kimi.chat([{"role": "user", "content": prompt}], heavy=True)
            doc_url = ""
            if self.docx is not None:
                title = f"组会周报 {time.strftime('%Y-%m-%d')}"
                doc_id, doc_url = self.docx.create_document(title)
                self.docx.write_markdown(doc_id, body)
                self.docx.grant_access(doc_id)
            counts = (f"本周精读 {len(materials.papers)} 篇 · 收藏 {len(materials.collections)} 条"
                      f" · 灵感 {len(materials.ideas)} 条")
            text = (f"📝 组会周报已生成（{counts}，生成于 {_fmt_now()}）\n"
                    + (f"📄 {doc_url}" if doc_url else "（文档写入失败，正文见日志）"))
            self._push(target, "组会周报", text=text)
            return text

        return self._run_safely("组会周报", target, _job)

    # ---- 每周精选（周五 20:00，7 天窗口严选，周末阅读）----

    def _review_picks(self, entries: list[ArxivEntry],
                      reasons: dict[str, list[str]]) -> list[dict[str, str]]:
        """每周精选：逐篇 light 深评（tldr/为什么值得读/与你方向/风险点）；单篇失败单篇兜底。"""
        try:
            template = WEEKLY_PICKS_PROMPT_PATH.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("每周精选深评模板读取失败（全量兜底）：%s", exc)
            template = ""
        interests = "、".join(self.settings.push_interests) or "（未配置显式兴趣）"
        items = []
        for entry in entries:
            review: dict[str, str] | None = None
            if template:
                paper_block = f"{entry.title}\n作者：{entry.authors}\n摘要：{entry.abstract[:800]}"
                prompt = template.replace("{{PAPER}}", paper_block).replace("{{INTERESTS}}", interests)
                try:
                    raw = self.kimi.chat([{"role": "user", "content": prompt}], heavy=False)
                    review = parse_pick_review_json(raw)
                except KimiError as exc:
                    logger.warning("每周精选深评单篇模型失败（兜底）：%s", exc)
            if review is None:
                logger.warning("每周精选深评单篇解析失败（兜底）：%s", entry.title[:40])
                review = {"tldr": "", "why": "", "relation": "", "risk": ""}
            tldr = review["tldr"] or (
                entry.abstract[:80] + "…" if len(entry.abstract) > 80 else entry.abstract
            )
            items.append({"title": entry.title, "url": entry.url, "arxiv_id": entry.arxiv_id,
                          "tldr": tldr, "why": review["why"], "relation": review["relation"],
                          "risk": review["risk"],
                          "authority": " · ".join(reasons.get(entry.arxiv_id, []))})
        statuses = self._pick_status_map([e.arxiv_id for e in entries])
        for item in items:  # 已读/未读初始状态上卡（按钮文字据此区分）
            item["status"] = statuses.get(item["arxiv_id"], "未读")
        return items

    def weekly_picks_now(self, chat_id: str | None = None) -> str:
        """生成并推送每周精选策展（7 天窗口严选 top 5，逐篇深评），返回纯文本版。

        与日报共用质量闸；0 篇过线时按 SILENT_WHEN_EMPTY 只记日志不推送。
        推送历史：已精读（论文库）与已上过周刊的（精选库来源含每周精选）排除；
        日报推过的不排除——周刊再策展时归档走 upgrade_on_dup 升级原记录
        （深评推荐理由覆盖日报一句话，来源合并为 arXiv日报+每周精选）。
        """
        target = self.chat_id if chat_id is None else chat_id

        def _job() -> str:
            entries = self.fetcher(WEEKLY_PICKS_DAYS)
            if not entries:
                text = (f"📌 本周精选：近 {WEEKLY_PICKS_DAYS} 天没有抓到符合条件的新论文"
                        f"（arXiv 周末/节假日更新少）。（生成于 {_fmt_now()}）")
                self._push(target, "每周精选", text=text)
                return text
            history = self._push_history()
            skip = history.curated | history.read
            fresh = [e for e in entries if e.arxiv_id not in skip]
            if len(fresh) < len(entries):
                logger.info("每周精选历史排除：候选 %d 篇，排除已策展/已精读 %d 篇",
                            len(entries), len(entries) - len(fresh))
            if not fresh:
                text = (f"📌 本周精选：近 {WEEKLY_PICKS_DAYS} 天扫描 {len(entries)} 篇，"
                        f"均已策展过或已精读，本周不打扰。（生成于 {_fmt_now()}）")
                if SILENT_WHEN_EMPTY:
                    logger.info("每周精选历史排除：%d 篇候选全部命中推送历史，静默不推送", len(entries))
                else:
                    self._push(target, "每周精选", text=text)
                return text
            entries = fresh
            ranked = rank_entries(entries, self._taste_scorer())
            merged, reasons = merge_authority(ranked, self._authority)
            top_pairs = [(e, s) for e, s in merged if s >= QUALITY_THRESHOLD][:WEEKLY_PICKS_N]
            if not top_pairs:
                text = (f"📌 本周精选：近 {WEEKLY_PICKS_DAYS} 天扫描 {len(entries)} 篇，"
                        f"无一过质量线（{QUALITY_THRESHOLD}），本周不打扰。（生成于 {_fmt_now()}）")
                if SILENT_WHEN_EMPTY:
                    logger.info("每周精选质量闸：%d 篇候选无一过线（阈值 %.2f），静默不推送",
                                len(entries), QUALITY_THRESHOLD)
                else:
                    self._push(target, "每周精选", text=text)
                return text
            top = [entry for entry, _ in top_pairs]
            items = self._review_picks(top, reasons)
            interests = list(self.settings.push_interests)
            extras = [{"方向": pick_direction(entry),
                       "关键词": interest_hits(entry, interests) or ["未分类"]}
                      for entry in top]
            archived = self._archive_picks(top, items, source=PICK_SOURCE_WEEKLY,
                                           extras=extras, upgrade_on_dup=True)
            card = build_weekly_picks_card(items, scanned=len(entries),
                                           days=WEEKLY_PICKS_DAYS, archived=archived)
            pushed = self._push(target, "每周精选", card=card)
            text = build_weekly_picks_text(items, scanned=len(entries),
                                           days=WEEKLY_PICKS_DAYS, archived=archived)
            if not pushed:  # 卡片推送失败 → 纯文本兜底推一次
                self._push(target, "每周精选（文本版）", text=text)
            return text

        return self._run_safely("每周精选", target, _job)

    # ---- 命令入口（router 接线用）----

    def cmd_arxiv_daily(self, ctx: MessageContext | None = None) -> str:
        """「arXiv日报」：异步生成后推送（本命令只回 ack）。"""
        if ctx is not None and ctx.chat_id:
            self.chat_id = self.chat_id or ctx.chat_id  # 未配置推送目标时落在发起会话
        self._start_async("arXiv 日报", self.arxiv_digest_now)
        return "📰 收到，开始生成 arXiv 日报：抓取 + 排序 + TL;DR 约 1~2 分钟，完成后推送。"

    def cmd_today(self, ctx: MessageContext | None = None) -> str:
        """「今日」：数据归集快，同步返回汇总文本。"""
        return self.today_digest_now(chat_id="")  # 命令回显不重复推送

    def cmd_weekly(self, ctx: MessageContext | None = None) -> str:
        """「组会周报」：heavy 生成较慢，异步推送。"""
        if ctx is not None and ctx.chat_id:
            self.chat_id = self.chat_id or ctx.chat_id
        self._start_async("组会周报", self.weekly_report_now)
        return "📝 收到，开始生成组会周报（heavy 模型，约 2~5 分钟），完成后推送文档链接。"

    # ---- 负推词（别推）命令 ----

    def cmd_block(self, arg: str, ctx: MessageContext | None = None) -> str:
        """「别推 xx」：日报排序中淘汰含该词的论文。"""
        term = (arg or "").strip()
        if not term:
            return "用法：别推 关键词（如：别推 speech）。查看当前列表发「别推列表」。"
        terms = add_blocklist(term)
        return f"🚫 已加入别推：「{term}」。日报将淘汰含该词的论文（当前共 {len(terms)} 条，发「别推列表」查看，「恢复推 {term}」撤销）。"

    def cmd_unblock(self, arg: str, ctx: MessageContext | None = None) -> str:
        """「恢复推 xx」：从负推词列表移除。"""
        term = (arg or "").strip()
        if not term:
            return "用法：恢复推 关键词。查看当前列表发「别推列表」。"
        terms, removed = remove_blocklist(term)
        if removed:
            return f"✅ 已恢复推荐：「{term}」（当前别推 {len(terms)} 条）。"
        return f"「{term}」不在别推列表里（发「别推列表」查看当前 {len(terms)} 条）。"

    def cmd_blocklist(self, ctx: MessageContext | None = None) -> str:
        """「别推列表」：查看负推词。"""
        terms = load_blocklist()
        if not terms:
            return "别推列表为空。发「别推 关键词」添加，日报将淘汰含该词的论文。"
        lines = [f"{i}. {t}" for i, t in enumerate(terms, 1)]
        return "🚫 当前别推列表：\n" + "\n".join(lines) + "\n（「恢复推 关键词」撤销）"


def register_jobs(
    scheduler: Any,
    service: DigestService,
    arxiv_time: str = DEFAULT_ARXIV_TIME,
    today_time: str = DEFAULT_TODAY_TIME,
    evening_time: str = DEFAULT_EVENING_TIME,
    weekly_time: str = DEFAULT_WEEKLY_TIME,
    weekly_day: str = "mon",
    picks_time: str = DEFAULT_PICKS_TIME,
    picks_day: str = "fri",
) -> list[str]:
    """把五个推送任务注册进 APScheduler（cron 触发器），返回 job id 列表。

    main.py 集成：register_jobs(scheduler, digest_service,
        arxiv_time=settings.push_daily_time, evening_time=settings.push_evening_time)。
    时间参数 "HH:MM"，默认值见 DEFAULT_*_TIME 常量（日报/晚间回顾由 config.yaml
    push.daily_time / push.evening_time 覆盖；今日提醒 9:00；组会周报周一 9:00
    （周二组会前）；每周精选周五 20:00（周末阅读））。
    """
    def _hm(t: str) -> tuple[int, int]:
        hour, minute = t.split(":", 1)
        return int(hour), int(minute)

    h, m = _hm(arxiv_time)
    # misfire_grace_time=6h：笔记本合盖睡过推送点后，唤醒时补发（默认 1s 会直接跳过）
    scheduler.add_job(lambda: service.arxiv_digest_now(respect_weekend=True),
                      "cron", hour=h, minute=m,
                      id="arxiv_daily", coalesce=True, misfire_grace_time=6 * 3600)
    h, m = _hm(today_time)
    scheduler.add_job(service.today_digest_now, "cron", hour=h, minute=m,
                      id="today_digest", coalesce=True, misfire_grace_time=6 * 3600)
    h, m = _hm(evening_time)
    scheduler.add_job(service.evening_digest_now, "cron", hour=h, minute=m,
                      id="today_evening", coalesce=True, misfire_grace_time=6 * 3600)
    h, m = _hm(weekly_time)
    scheduler.add_job(service.weekly_report_now, "cron",
                      day_of_week=weekly_day, hour=h, minute=m,
                      id="weekly_report", coalesce=True, misfire_grace_time=6 * 3600)
    h, m = _hm(picks_time)
    scheduler.add_job(service.weekly_picks_now, "cron",
                      day_of_week=picks_day, hour=h, minute=m,
                      id="weekly_picks", coalesce=True, misfire_grace_time=6 * 3600)
    jobs = ["arxiv_daily", "today_digest", "today_evening", "weekly_report", "weekly_picks"]
    logger.info("推送任务已注册：%s（日报 %s / 今日 %s / 晚间回顾 %s / 周报 %s %s / 每周精选 %s %s）",
                "、".join(jobs), arxiv_time, today_time, evening_time, weekly_day, weekly_time,
                picks_day, picks_time)
    return jobs
