"""论文精读核心管线（P2）。

流程：入口识别（arXiv 链接 / PDF 附件 / 精读触发词）→ 下载 PDF → 解析为
markdown（pdf.py）→ 长文证据索引/短文全文规划 → 分节写作与分层精讲 →
飞书文档落地 + 权限授予 → 论文库归档 → sqlite 存全文（支撑追问）→ 卡片推送。

异步 UX：入口立即返回 ack，后台线程跑管线，完成推卡片，失败推中文错误卡片。
追问：近 FOLLOWUP_WINDOW_HOURS 小时内精读过论文的会话，命中追问关键词即带
全文上下文走 heavy 链回答。

并发幂等：同一查重键（arXiv ID / 归一化链接 / 附件 file_key）同时只允许一个在飞
任务（防连点、防重投重复跑）；复读版本号在在飞锁保护区内从论文库主记录分配，
并发复读不撞 vN；无 arXiv ID 的链接论文按「查重键」字段查重，复读更新主记录
而非重复建行。

断线/重启恢复（两层）：
- 瞬时故障自动重试：job 抛出网络/超时/DNS 类错误（沿 __cause__/__context__
  链判定）时按 2/5/10 分钟退避重跑同一 job（检查点让重跑是"续跑"），
  耗尽才走失败路径；业务错误（验收/凭证/参数）绝不重试。
- durable 任务登记：启动时把任务写进 reading_tasks（running），结束写
  done/failed；进程死在中途的 running 任务在下次启动时自动重建上下文重跑，
  已 failed 的不恢复（防恢复死循环）。

事件通知外发箱（NotificationOutbox）：失败、进入首次瞬时重试、疑似卡死
（状态条看门狗，5 分钟无在途活动）、中断自动恢复这四类「用户必须知道」的
事件，在卡片之外补一条必达文本——先 enqueue 落 sqlite，bot 的 flusher
线程 30s 一轮补发。断网期间卡片发送会静默丢失，外发箱不会：这是
「卡片 best-effort、事件必达」的分层。无 outbox 注入时全部退化为不写。
"""

from __future__ import annotations

import json
import logging
import re
import socket
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
import openai
import requests.exceptions
import urllib3.exceptions

from ..config import PROJECT_ROOT, Settings
from ..activity import track
from ..feishu.bitable import (
    PAPER_DIRECTIONS,
    BitableDAO,
    BitableError,
    FieldSpec,
    FIELD_TYPE_SINGLE_SELECT,
    FIELD_TYPE_TEXT,
    ensure_fields,
)
from ..feishu.docx import DocxClient, DocxError, fit_image_size, image_dimensions  # noqa: F401  (fit/image 尺寸助手经本模块 re-export)
from ..feishu.drive import DriveClient, DriveError
from ..kimi import KimiClient, KimiError
from ..lifecycle import LifecycleError, PaperCache, PaperCacheEntry
from ..pdf import PdfParseError, parse_pdf, pdf_page_count
from ..store import (
    FOLLOWUP_WINDOW_HOURS,
    NotificationOutbox,
    PaperStore,
    ReadingTask,
    ReadingTaskStore,
)
from ..vectors import PaperVectorIndex
from .figures import FigureAsset, load_figure_assets
from .formulas import load_formula_assets
from .reading import ReadingEngine, ReadingError, ReadingResult, classify_direction
from .reading_workflow import ReadingWorkspace
from .status import StatusMessenger

logger = logging.getLogger("feishu_assistant.paper")

PAPERS_DIR = PROJECT_ROOT / "data" / "papers"

DEFAULT_MAX_INPUT_CHARS = 60_000  # 追问上下文预算；精读正文不再用这个值截尾
DOWNLOAD_TIMEOUT = 120
METADATA_TIMEOUT = 15

# 瞬时网络故障的自动重试退避（秒）：断网类事故通常分钟级恢复，2/5/10 分钟三档，
# 等待期间状态条明示倒计时；耗尽才走失败路径。重跑安全：PDF 内容寻址缓存 +
# 精读工作区检查点（逐阶段复用）+ 在飞锁串行 + 版本号锁内分配，job 整体幂等，
# 重跑是"续跑"而非"从头再来"。
TRANSIENT_RETRY_BACKOFFS = (120, 300, 600)

# ---------------------------------------------------------------------------
# arXiv 入口识别与下载
# ---------------------------------------------------------------------------

_ARXIV_URL_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/|arXiv:\s*)(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?",
    re.IGNORECASE,
)


def parse_arxiv_id(text: str) -> str | None:
    """从文本里提取 arXiv id（支持 abs/pdf 链接与 arXiv:xxxx.xxxxx 写法），无则 None。"""
    m = _ARXIV_URL_RE.search(text)
    return m.group(1) if m else None


def arxiv_abs_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}"


def arxiv_pdf_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}"


def fetch_arxiv_metadata(arxiv_id: str, timeout: int = METADATA_TIMEOUT) -> dict[str, str]:
    """取标题/作者（export.arxiv.org Atom API）；失败返回空 dict（不阻断管线）。"""
    try:
        resp = httpx.get(
            "https://export.arxiv.org/api/query",
            params={"id_list": arxiv_id}, timeout=timeout, follow_redirects=True,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        entry = root.find("a:entry", ns)
        if entry is None:
            return {}
        title = re.sub(r"\s+", " ", (entry.findtext("a:title", "", ns) or "")).strip()
        authors = ", ".join(
            a.findtext("a:name", "", ns) for a in entry.findall("a:author", ns)
        )
        published = (entry.findtext("a:published", "", ns) or "")[:4]
        return {"title": title, "authors": authors, "year": published}
    except Exception as exc:
        logger.warning("获取 arXiv 元数据失败（%s）：%s", arxiv_id, exc)
        return {}


def download_arxiv_pdf(arxiv_id: str, dest_dir: Path, timeout: int = DOWNLOAD_TIMEOUT) -> Path:
    """下载 arXiv PDF 到 dest_dir/paper.pdf。"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "paper.pdf"
    logger.info("开始下载 arXiv PDF：%s", arxiv_pdf_url(arxiv_id))
    with httpx.stream("GET", arxiv_pdf_url(arxiv_id), timeout=timeout, follow_redirects=True) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=1 << 16):
                f.write(chunk)
    downloaded_size = dest.stat().st_size
    if downloaded_size < 10_000:  # 几 KB 的"PDF"多半是错误页
        dest.unlink(missing_ok=True)
        raise PaperPipelineError(f"arXiv PDF 下载结果异常（{downloaded_size} 字节），请确认 arXiv id 有效：{arxiv_id}")
    logger.info("PDF 下载完成：%s（%.1f KB）", dest, downloaded_size / 1024)
    return dest


# ---------------------------------------------------------------------------
# 通用论文链接（非 arXiv：OpenReview/ACL/bioRxiv/PMLR/CVF/Nature…）
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://[^\s<>\])}，。]+")
_PAPER_HOST_RE = re.compile(
    r"openreview\.net|aclanthology\.org|semanticscholar\.org|biorxiv\.org|medrxiv\.org|"
    r"neurips\.cc|mlr\.press|thecvf\.com|ar5iv\.labs\.arxiv\.org|nature\.com|"
    r"science\.org|ieeexplore\.ieee\.org|dl\.acm\.org|link\.springer\.com|doi\.org",
    re.IGNORECASE,
)


def normalize_paper_url(url: str) -> str:
    """链接归一化（精读查重键用）：去首尾空白与尾随标点，同一链接的不同写法只算一篇。"""
    return url.strip().rstrip(".,;)").strip()


def parse_paper_url(text: str) -> str | None:
    """文本里的论文链接：PDF 直链 / 论文页白名单域名，无则 None。"""
    m = _URL_RE.search(text)
    if not m:
        return None
    url = m.group(0)
    if url.split("?")[0].lower().endswith(".pdf") or _PAPER_HOST_RE.search(url):
        return url
    return None


def _meta_tags(html: str) -> dict[str, str]:
    """提取 <meta name=.. content=..>（两种属性顺序都兼容）。"""
    tags: dict[str, str] = {}
    for m in re.finditer(r"<meta\s+[^>]*>", html, re.IGNORECASE):
        tag = m.group(0)
        name = re.search(r'name=["\']([^"\']+)["\']', tag, re.IGNORECASE)
        content = re.search(r'content=["\']([^"\']*)["\']', tag, re.IGNORECASE)
        if name and content:
            tags[name.group(1)] = content.group(1)
    return tags


def resolve_paper_url(url: str, timeout: int = 30) -> tuple[str, dict[str, str]]:
    """论文链接 → (pdf_url, meta_hint)。

    PDF 直链直接用；OpenReview/ar5iv/ACL 走已知模式省一次抓取；
    其余抓页面找 citation_pdf_url 元标签（出版方标准，Nature/Science/PMLR/IEEE 都带）。
    """
    from urllib.parse import urljoin

    clean = normalize_paper_url(url)
    if clean.split("?")[0].lower().endswith(".pdf"):
        return clean, {}
    m = re.search(r"openreview\.net/forum\?id=([\w-]+)", clean)
    if m:
        return f"https://openreview.net/pdf?id={m.group(1)}", {}
    m = re.search(r"ar5iv\.labs\.arxiv\.org/html/(\d{4}\.\d{4,5})", clean)
    if m:
        return arxiv_pdf_url(m.group(1)), {}
    m = re.search(r"(aclanthology\.org/[\w.-]+)/?$", clean)
    if m:
        return f"https://{m.group(1)}.pdf", {}

    resp = httpx.get(clean, timeout=timeout, follow_redirects=True,
                     headers={"User-Agent": "Mozilla/5.0 (feishu-assistant paper-bot)"})
    resp.raise_for_status()
    if "pdf" in resp.headers.get("content-type", "").lower():
        return clean, {}  # 无 .pdf 后缀但实际是 PDF（query 形式直链）
    tags = _meta_tags(resp.text)
    pdf = tags.get("citation_pdf_url", "").strip()
    if not pdf:
        raise PaperPipelineError(
            f"链接里没有找到论文 PDF（页面无 citation_pdf_url）：{clean}\n"
            "可以把 PDF 文件直接发我，或换出版方页面（OpenReview/ACL/arXiv）链接。"
        )
    hint: dict[str, str] = {}
    if tags.get("citation_title"):
        hint["title"] = re.sub(r"\s+", " ", tags["citation_title"]).strip()
    if tags.get("citation_author"):
        hint["authors"] = tags["citation_author"].strip()
    if tags.get("citation_publication_date"):
        hint["year"] = tags["citation_publication_date"][:4]
    return urljoin(clean, pdf), hint


def download_pdf_url(pdf_url: str, dest_dir: Path, timeout: int = DOWNLOAD_TIMEOUT) -> Path:
    """从任意 URL 下载 PDF 到 dest_dir/paper.pdf（尺寸校验，错误页直接报中文）。"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "paper.pdf"
    logger.info("开始下载 PDF：%s", pdf_url)
    with httpx.stream("GET", pdf_url, timeout=timeout, follow_redirects=True,
                      headers={"User-Agent": "Mozilla/5.0 (feishu-assistant paper-bot)"}) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=1 << 16):
                f.write(chunk)
    downloaded_size = dest.stat().st_size
    if downloaded_size < 10_000:
        dest.unlink(missing_ok=True)
        raise PaperPipelineError(
            f"PDF 下载结果异常（{downloaded_size} 字节），链接可能失效或需登录：{pdf_url}"
        )
    logger.info("PDF 下载完成：%s（%.1f KB）", dest, downloaded_size / 1024)
    return dest


# ---------------------------------------------------------------------------
# 上下文截断：保 abstract/方法/实验/结论，压 related work，删参考文献/附录
# ---------------------------------------------------------------------------

_DROP_SECTION_RE = re.compile(
    r"references|bibliography|appendix|acknowledg|参考文献|附录|致谢", re.IGNORECASE
)
_PRIORITY_SECTION_RE = re.compile(
    r"abstract|introduction|method|approach|experiment|result|conclusion|discussion|"
    r"摘要|引言|方法|实验|结果|结论|讨论",
    re.IGNORECASE,
)


def split_sections(md: str) -> list[tuple[str, str]]:
    """按 markdown 标题切节，返回 [(标题行, 该节全文)]；无标题内容归一节（标题为空）。"""
    sections: list[tuple[str, str]] = []
    current_title = ""
    current: list[str] = []
    for line in md.splitlines():
        if re.match(r"^#{1,3}\s+", line):
            if current:
                sections.append((current_title, "\n".join(current)))
            current_title = line.strip()
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append((current_title, "\n".join(current)))
    return sections


def truncate_markdown(md: str, max_chars: int = DEFAULT_MAX_INPUT_CHARS) -> str:
    """把论文 markdown 压到 max_chars 以内。确定性策略，逐级收紧。"""
    if len(md) <= max_chars:
        return md

    sections = split_sections(md)
    kept = [(t, s) for t, s in sections if not _DROP_SECTION_RE.search(t)]
    dropped = len(sections) - len(kept)
    if dropped:
        logger.info("截断：删除参考文献/附录类章节 %d 节", dropped)

    def join(secs: list[tuple[str, str]]) -> str:
        return "\n\n".join(s for _, s in secs)

    text = join(kept)
    if len(text) <= max_chars:
        return text

    # 非优先章节压到 400 字符
    compressed = [
        (t, s if _PRIORITY_SECTION_RE.search(t) or len(s) <= 400 else s[:400] + "\n…（本节已压缩）")
        for t, s in kept
    ]
    text = join(compressed)
    if len(text) <= max_chars:
        logger.info("截断：非优先章节压缩至 400 字符后达标（%d 字符）", len(text))
        return text

    # 仍超长：非优先章节整节丢弃（保留首尾无标题部分与优先章节）
    essential = [(t, s) for t, s in compressed if _PRIORITY_SECTION_RE.search(t) or not t]
    text = join(essential)
    if len(text) <= max_chars:
        logger.info("截断：丢弃非优先章节后达标（%d 字符）", len(text))
        return text

    # 最后手段：按节等比硬截断
    per = max_chars // max(len(essential), 1)
    text = join([(t, s[:per]) for t, s in essential])
    logger.info("截断：等比硬截断至 %d 字符", len(text))
    return text[:max_chars]


_METADATA_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def extract_metadata(llm_output: str) -> tuple[str, dict[str, Any]]:
    """从模型输出拆出正文与 ```json 元数据块；块缺失/非法时返回默认元数据。"""
    m = _METADATA_RE.search(llm_output)
    if not m:
        logger.warning("模型输出缺少 json 元数据块，使用默认值归档")
        return llm_output.strip(), {}
    body = llm_output[: m.start()].strip()
    try:
        metadata = json.loads(m.group(1))
        if not isinstance(metadata, dict):
            raise ValueError("metadata 不是对象")
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("元数据 json 解析失败：%s", exc)
        return body, {}
    return body, metadata


def build_paper_row(
    metadata: dict[str, Any],
    doc_url: str,
    pdf_url: str,
    arxiv_id: str,
    read_at: float | None = None,
    direction: str = "",
    source: str = "精读",
) -> dict[str, Any]:
    """精读结果 → 论文库 bitable 行（字段名/值格式对照 bitable.py 的 schema）。"""
    read_at = read_at if read_at is not None else time.time()
    row: dict[str, Any] = {
        "标题": metadata.get("title_zh") or metadata.get("title_en") or "未命名论文",
        "作者": str(metadata.get("authors", "")),
        "年份": str(metadata.get("year", "")),
        "TL;DR": str(metadata.get("tldr", "")),
        "arXivID": arxiv_id,
        "精读时间": int(read_at * 1000),  # bitable 日期字段：毫秒时间戳
    }
    keywords = metadata.get("keywords")
    if isinstance(keywords, list) and keywords:
        row["关键词"] = [str(k) for k in keywords][:6]  # 多选：字符串列表，飞书自动建选项
    if doc_url:
        row["文档链接"] = {"text": f"精读：{short_title(metadata)}", "link": doc_url}  # 超链接字段
    if pdf_url:
        row["PDF链接"] = {"text": "PDF", "link": pdf_url}
    if direction:
        row["方向"] = direction  # 单选：值为选项名
    if source:
        row["来源"] = source
    return row


def short_title(metadata: dict[str, Any], max_len: int = 16) -> str:
    """文档/卡片用短标题：优先中文题，去书名号，超长截断。"""
    title = str(metadata.get("title_zh") or metadata.get("title_en") or "未命名论文")
    title = re.sub(r"\s+", " ", title).strip().strip("《》").strip()
    return title[:max_len] + ("…" if len(title) > max_len else "")


def doc_display_name(metadata: dict[str, Any], arxiv_id: str = "",
                     when: float | None = None, version: int = 1) -> str:
    """飞书文档命名：精读：短标题 vN（arXiv ID · MM-DD）。"""
    when = when if when is not None else time.time()
    date = time.strftime("%m-%d", time.localtime(when))
    short = short_title(metadata)
    if arxiv_id:
        return f"精读：{short} v{version}（{arxiv_id} · {date}）"
    return f"精读：{short} v{version}（{date}）"


# ---------------------------------------------------------------------------
# 消息卡片
# ---------------------------------------------------------------------------

def _md_element(content: str) -> dict:
    return {"tag": "markdown", "content": content}


def build_result_card(
    metadata: dict[str, Any],
    doc_url: str,
    arxiv_id: str,
    archived: bool,
    timing_text: str = "",
) -> dict:
    """精读完成卡片：TL;DR + 核心贡献前 3 条 + 文档链接 + 归档提示 + 计时。"""
    title = short_title(metadata, max_len=24)
    tldr = metadata.get("tldr", "")
    contributions = metadata.get("contributions") or []
    contrib_text = "\n".join(f"{i}. {c}" for i, c in enumerate(contributions[:3], 1))
    archive_note = "📚 论文库已归档 · 可以直接追问本文细节（如「第三章 xx 怎么算的」）"
    if not archived:
        archive_note = "⚠️ 论文库归档未成功（精读文档仍可正常使用，可检查配置后重试归档）"

    elements = [
        _md_element(f"**TL;DR**\n{tldr}"),
    ]
    final_degraded = str(metadata.get("final_review_degraded") or "").strip()
    degraded_sections = metadata.get("degraded_sections") or []
    if final_degraded or degraded_sections:
        warn_lines: list[str] = []
        if final_degraded:
            core = bool(metadata.get("final_review_core_failed"))
            warn_lines.append(
                "**终审未通过（含核心链路问题）**" if core else "**终审未全部通过**")
            for i, item in enumerate(
                    [x.strip() for x in final_degraded.split("；") if x.strip()][:5], 1):
                warn_lines.append(f"{i}. {item}")
        if degraded_sections:
            warn_lines.append("**节级批注**（全文保留，详见文档内横幅）：")
            warn_lines.extend(f"- {name}" for name in degraded_sections[:6])
        elements.append(_md_element("⚠️ " + "\n".join(warn_lines)))
    if contrib_text:
        elements.append(_md_element(f"**核心贡献**\n{contrib_text}"))
    if arxiv_id:
        elements.append(_md_element(f"🔗 [arXiv:{arxiv_id}]({arxiv_abs_url(arxiv_id)})"))
    if timing_text:
        elements.append(_md_element(timing_text))
    elements.append({"tag": "note", "elements": [{"tag": "plain_text", "content": archive_note}]})
    if doc_url:
        elements.append({
            "tag": "action",
            "actions": [{
                "tag": "button",
                "text": {"tag": "plain_text", "content": "📖 打开精读文档"},
                "url": doc_url,
                "type": "primary",
            }],
        })
    degraded = bool(metadata.get("final_review_degraded") or metadata.get("degraded_sections"))
    core_failed = bool(metadata.get("final_review_core_failed"))
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "red" if core_failed else ("orange" if degraded else "green"),
            "title": {"tag": "plain_text", "content": (
                f"⚠️ 精读完成（终审未通过）《{title}》" if core_failed
                else f"⚠️ 精读完成（有审核批注）《{title}》" if degraded
                else f"✅ 精读完成《{title}》"
            )},
        },
        "elements": elements,
    }


def build_error_card(paper_label: str, reason: str, suggestion: str) -> dict:
    """精读失败卡片：中文原因 + 可操作建议。"""
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "red",
            "title": {"tag": "plain_text", "content": f"❌ 精读失败{paper_label}"},
        },
        "elements": [
            _md_element(f"**原因**\n{reason}"),
            _md_element(f"**建议**\n{suggestion}"),
        ],
    }


# ---------------------------------------------------------------------------
# 追问状态机
# ---------------------------------------------------------------------------

FOLLOWUP_KEYWORDS = (
    "这篇", "本文", "论文", "作者", "第", "图", "表", "公式", "章节",
    "方法", "实验", "结论", "贡献", "局限", "指标", "数据", "结果", "框架",
    "怎么算", "为什么", "什么意思", "解释", "上面", "刚才", "前文", "原文",
)


def looks_like_followup(text: str) -> bool:
    """是否像针对最近精读论文的追问（明确关键词或引用前文）。"""
    return any(kw in text for kw in FOLLOWUP_KEYWORDS)


_FOLLOWUP_SYSTEM = (
    "你是论文精读助手的追问模式。用户刚精读完一篇论文，现在针对原文追问细节。"
    "回答要求：严格基于给出的论文全文，标注相关章节位置；原文没有的信息明说"
    "「原文未提及」，不要编造；中文回答，简洁直接，先结论后展开。"
)


# ---------------------------------------------------------------------------
# 管线服务
# ---------------------------------------------------------------------------

class PaperPipelineError(Exception):
    """精读管线失败，信息为中文（面向用户的错误卡片用）。"""


class Heartbeat:
    """长任务心跳公共组件（供求职等长任务使用）。"""

    def __init__(self, send: Any, interval: float = 90) -> None:
        self._send = send
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_ts = 0.0

    def start(self) -> None:
        self._start_ts = time.time()
        self._thread = threading.Thread(
            target=self._loop, name="paper-heartbeat", daemon=True
        )
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            mins = max(1, int((time.time() - self._start_ts) // 60))
            self._send(f"⏳ 仍在生成（已 {mins} 分钟）……")

    def stop(self) -> None:
        self._stop.set()


@dataclass
class MessageContext:
    """消息上下文：异步推送与权限授予需要。

    status：本消息触发的精读任务的状态条（_start_async 注入），
    进度阶段行进它——按消息而不是按 chat 绑定，两篇精读并发互不干扰。
    """

    chat_id: str
    open_id: str = ""
    message_id: str = ""
    status: Any = None


class Sender(Protocol):
    """bot 实现的推送接口（paper 模块不反向依赖 bot）。"""

    def send_text(self, chat_id: str, text: str) -> bool: ...
    def send_card(self, chat_id: str, card: dict) -> bool: ...
    def send_text_id(self, chat_id: str, text: str) -> str: ...
    def update_text(self, message_id: str, text: str) -> str: ...


class PaperReadingService:
    def __init__(
        self,
        settings: Settings,
        kimi: KimiClient,
        lark_client: Any,
        store: PaperStore,
        sender: Sender,
        max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
        vector_index: PaperVectorIndex | None = None,
        reading_engine: ReadingEngine | None = None,
        paper_cache: PaperCache | None = None,
        task_store: ReadingTaskStore | None = None,
        outbox: NotificationOutbox | None = None,
    ) -> None:
        self.settings = settings
        self.kimi = kimi
        self.store = store
        self.sender = sender
        self.max_input_chars = max_input_chars
        self.lark_client = lark_client
        self.docx = DocxClient(lark_client)
        self.drive = DriveClient(lark_client)
        self.vector_index = vector_index
        self.reading_engine = reading_engine or ReadingEngine(kimi)
        self.paper_cache = paper_cache or PaperCache(PAPERS_DIR, store=store)
        # 精读任务登记（断线/重启恢复）：默认与 PaperStore 共用同一库文件
        self.task_store = task_store or ReadingTaskStore(store.db_path)
        # 事件通知外发箱（失败/重试/卡死/恢复必达）：None = 事件通知静默停用
        self.outbox = outbox
        self._paper_fields_ensured = False
        # 精读在飞任务（查重键集合）：防连点/重投并发跑同一篇；复读版本号在锁保护区内分配
        self._inflight: set[str] = set()
        self._inflight_lock = threading.Lock()
        # 精读完成后的回调（main.py 注入 session_service.bind_paper_session）：
        # 签名 (context, title, arxiv_id) -> str | None，返回给用户的提示语
        self.on_done: Any = None
        self.bitable_dao: BitableDAO | None = (
            BitableDAO(lark_client, settings.bitable_app_token, settings.bitable_tables)
            if settings.bitable_ready()
            else None
        )

    # ---- 入口：文本（arXiv 链接 / 精读触发词）----

    # ---- 入口：序号引用（「精读 N」→ 精选库最新一批第 N 篇）----

    def read_pick(self, n: int, context: MessageContext) -> str:
        """从精选库最新一批推荐里取第 n 篇触发精读。"""
        if self.bitable_dao is None:
            return "多维表格未初始化，发 /init 初始化后才能用序号精读。"
        records = self.bitable_dao.list_records("精选库", page_size=20)
        if not records:
            return "精选库还是空的——等明早 8:30 的日报，或发「arXiv日报」立即生成一份。"
        # 最新一批：同批归档共用同一时间戳（digest 侧 batch_now），取最大日期毫秒值的
        # 整批；批内保持写入顺序（即推荐序号顺序）。日报每批 ≤3，每周精选每批 ≤5。
        def _ts(rec: Any) -> float:
            v = (rec.fields or {}).get("日期")
            return float(v) if isinstance(v, (int, float)) else 0.0
        latest = max((_ts(rec) for rec in records), default=0.0)
        batch = [rec for rec in records if _ts(rec) == latest]
        if not (1 <= n <= len(batch)):
            return f"最新一批精选共 {len(batch)} 篇，说「精读 1」~「精读 {len(batch)}」。"
        fields = batch[n - 1].fields or {}
        arxiv_id = str(fields.get("arXivID") or "").strip()
        title = str(fields.get("标题") or "").strip()
        if not arxiv_id:
            return f"第 {n} 篇《{title[:30]}》没有 arXiv ID，直接把链接发我吧。"
        r = self._start_async(context, label=f"arXiv:{arxiv_id}", job=self._job_arxiv,
                              dedup_key=f"arxiv:{arxiv_id}", arxiv_id=arxiv_id)
        if r is None:
            return f"📄 第 {n} 篇《{title[:30]}》正在精读中，等它跑完～"
        if r:
            return ""
        return (f"📄 收到，开始精读今日精选第 {n} 篇《{title}》（arXiv:{arxiv_id}）："
                "全文约 12~20 分钟，完成后发你存档文档。")

    def try_handle_text(self, text: str, context: MessageContext | None = None) -> str | None:
        """是精读请求则启动异步管线；状态条发出去了返回 ""（状态条即 ack），
        没发出去回退文字 ack；非精读请求返回 None。"""
        arxiv_id = parse_arxiv_id(text)
        if arxiv_id:
            r = self._start_async(context, label=f"arXiv:{arxiv_id}", job=self._job_arxiv,
                                  dedup_key=f"arxiv:{arxiv_id}", arxiv_id=arxiv_id)
            if r is None:
                return f"📄 arXiv:{arxiv_id} 正在精读中，完成后会推结果卡，不用重复发～"
            if r:
                return ""
            return (
                f"📄 收到，开始精读 arXiv:{arxiv_id}：下载+解析约 1 分钟，全文约 12~20 分钟。"
                "完成后发你存档文档。"
            )
        url = parse_paper_url(text)
        if url or ("精读" in text and _URL_RE.search(text)):
            url = normalize_paper_url(url or _URL_RE.search(text).group(0))  # type: ignore[union-attr]
            r = self._start_async(context, label=f"链接:{url[:30]}", job=self._job_url,
                                  dedup_key=f"url:{url}", url=url)
            if r is None:
                return "📄 这个链接正在精读中，完成后会推结果卡，不用重复发～"
            if r:
                return ""
            return (
                f"📄 收到，开始精读链接论文：{url[:80]}\n"
                "解析链接+下载约 1 分钟，全文约 15 分钟，完成后发你存档文档。"
            )
        if "精读" in text:
            return (
                "想精读论文的话，直接发我论文链接（arXiv / OpenReview / ACL / bioRxiv 等"
                "论文页或 PDF 直链）或把 PDF 文件发给我，我就开始精读～"
            )
        return None

    # ---- 入口：PDF 附件（bot 收到 file 消息时调用）----

    def handle_file(self, file_name: str, file_key: str, context: MessageContext) -> str:
        if not file_name.lower().endswith(".pdf"):
            return "目前只支持 PDF 格式的论文附件哦～也可以直接发论文链接。"
        r = self._start_async(
            context, label=f"《{file_name}》",
            job=self._job_attachment, file_name=file_name, file_key=file_key,
            message_id=context.message_id, dedup_key=f"file:{file_key}",
        )
        if r is None:
            return f"📄 《{file_name}》正在精读中，完成后会推结果卡～"
        if r:
            return ""
        return f"📄 收到，开始精读《{file_name}》：解析约 1 分钟，全文约 12~20 分钟，完成后发你存档文档。"

    # ---- 追问 ----

    def answer_followup(self, text: str, context: MessageContext | None) -> str | None:
        """命中追问状态则带全文上下文回答；否则返回 None（交给自由对话）。"""
        if context is None:
            return None
        paper = self.store.latest_paper(context.chat_id, within_hours=FOLLOWUP_WINDOW_HOURS)
        if paper is None or not looks_like_followup(text):
            return None
        logger.info("命中追问：chat=%s 论文《%s》问题=%s", context.chat_id, paper.title, text[:40])
        context_md = truncate_markdown(paper.markdown, self.max_input_chars)
        messages = [
            {"role": "system", "content": _FOLLOWUP_SYSTEM},
            {"role": "user", "content": (
                f"论文《{paper.title}》全文（markdown，可能经截断）：\n\n{context_md}"
                f"\n\n---\n用户追问：{text}"
            )},
        ]
        try:
            return self.kimi.chat(messages, heavy=True)
        except KimiError as exc:
            return f"调用大模型出错了：{exc}"

    # ---- 异步管线 ----

    def _start_async(self, context: MessageContext | None, label: str, job,
                     *, dedup_key: str = "", **kwargs) -> str | None:
        """启动后台精读。返回值三态：
        - 非空 str：状态条 message_id，状态条即 ack（不再单独回复，聊天框消息最少化）；
        - ""：状态条没发出去，调用方回退文字 ack；
        - None：同一查重键已有在飞任务（防连点/重复触发），调用方回「已在精读中」。

        dedup_key（arXiv ID / 归一化链接 / 附件 file_key）非空时先登记在飞任务，
        任务线程结束才释放——同一篇并发不会重复下载、重复建库行或版本撞号。
        登记同时落到 reading_tasks（running）：进程死在中途，下次启动由
        resume_interrupted 捞起自动重跑；登记写库失败不影响本次运行。
        """
        if context is None or not context.chat_id:
            logger.error("缺少消息上下文，无法异步精读 %s", label)
            return ""
        if dedup_key:
            with self._inflight_lock:
                if dedup_key in self._inflight:
                    logger.info("精读重复触发已拦截：%s（key=%s）", label, dedup_key)
                    return None
                self._inflight.add(dedup_key)
            kind, _, ref = dedup_key.partition(":")
            try:
                self.task_store.upsert_start(
                    dedup_key, kind, ref, label, context.chat_id,
                    open_id=context.open_id,
                    message_id=context.message_id or str(kwargs.get("message_id", "")),
                )
            except Exception:
                logger.exception("精读任务登记失败（不影响本次运行）：%s", dedup_key)
        status = StatusMessenger(
            self.sender, context.chat_id, f"精读 {label}",
            alert_fn=self._stuck_alert_fn(context.chat_id, label),
        ).start()
        context.status = status  # 按消息绑定：两篇精读并发，阶段行各进各的状态条
        thread = threading.Thread(
            target=self._run_guarded, args=(context, label, job, kwargs, status, dedup_key),
            name=f"paper-{int(time.time())}", daemon=True,
        )
        try:
            thread.start()
        except Exception:
            self._release_inflight(dedup_key)  # 线程没起来：当场释放，不永久占用查重键
            raise
        logger.info("精读任务已启动：%s（chat=%s）", label, context.chat_id)
        return status._msg_id

    def _release_inflight(self, dedup_key: str) -> None:
        """任务结束释放在飞查重键（成功/失败都释放，复读才能再触发）。"""
        if dedup_key:
            with self._inflight_lock:
                self._inflight.discard(dedup_key)

    def _mark_task_finished(self, dedup_key: str, ok: bool) -> None:
        """回写任务登记（成功 done / 失败 failed）；写库失败只记日志，不反向影响任务结果。"""
        if not dedup_key:
            return
        try:
            self.task_store.mark_finished(dedup_key, ok)
        except Exception:
            logger.exception("精读任务登记回写失败：%s", dedup_key)

    def _enqueue_notify(self, chat_id: str, text: str) -> None:
        """事件通知落外发箱（flusher 必达补发）；无 outbox 或写库失败只记日志，
        绝不影响任务本身。同文 10 分钟去重由 outbox 保证。"""
        if self.outbox is None:
            return
        try:
            self.outbox.enqueue(chat_id, text)
        except Exception:
            logger.exception("事件通知入队失败（不影响任务）：%s", text[:60])

    def _stuck_alert_fn(self, chat_id: str, label: str):
        """卡死看门狗回调：接外发箱必达；无 outbox 返回 None（看门狗静默停用）。"""
        if self.outbox is None:
            return None

        def _alert(text: str) -> None:
            self._enqueue_notify(chat_id, text)

        return _alert

    def _run_with_transient_retry(self, context: MessageContext, label: str, job,
                                  kwargs: dict, status: StatusMessenger):
        """跑 job；瞬时网络故障按 TRANSIENT_RETRY_BACKOFFS 退避自动重跑，耗尽后原样抛出。

        只有 _is_transient_error 命中的网络/超时/DNS 类错误才重跑——非瞬时错误
        （验收失败、凭证错误、业务错误）立即抛出，绝不重试。
        等待期间只调 set_stage 改阶段文案（「网络中断，X 分钟后自动重试（第 k/3 次）」），
        不触碰 status 的定格/冻结语义；首次进入退避时另经外发箱发一条必达文本
        （断网期状态条/卡片可能都到不了用户）。重跑幂等性见 TRANSIENT_RETRY_BACKOFFS 注释。
        """
        retries = 0
        while True:
            try:
                return job(context=context, **kwargs)
            except Exception as exc:
                if not _is_transient_error(exc) or retries >= len(TRANSIENT_RETRY_BACKOFFS):
                    raise
                delay = TRANSIENT_RETRY_BACKOFFS[retries]
                retries += 1
                if retries == 1:
                    # 首次进入退避：网络可能正断着，错误卡未必发得出，
                    # 必达文本经外发箱排队补发——用户不刷状态条也知道在自动重试
                    self._enqueue_notify(
                        context.chat_id,
                        f"⚠️ 精读{label}遇到网络问题，正在自动重试，完成后告诉你",
                    )
                logger.warning(
                    "精读瞬时网络故障（%s: %s），%d 秒后自动重试（第 %d/%d 次）：%s",
                    type(exc).__name__, exc, delay, retries,
                    len(TRANSIENT_RETRY_BACKOFFS), label,
                )
                status.set_stage(
                    f"🌐 网络中断，{delay // 60} 分钟后自动重试"
                    f"（第 {retries}/{len(TRANSIENT_RETRY_BACKOFFS)} 次）"
                )
                time.sleep(delay)

    def _run_guarded(self, context: MessageContext, label: str, job, kwargs: dict,
                     status: StatusMessenger, dedup_key: str = "") -> None:
        """后台线程本体：瞬时故障重试 → 成功推结果卡 / 失败推错误卡 → 回写任务登记。
        失败除错误卡外另经外发箱补一条必达文本（断网时错误卡会丢，文本补发不丢）。"""
        try:
            metadata, doc_url, archived = self._run_with_transient_retry(
                context, label, job, kwargs, status
            )
        except Exception as exc:  # 统一兜底：失败也要推中文原因
            logger.exception("精读管线失败：%s", label)
            status.stop(ok=False, note=f"失败：{type(exc).__name__}（详见下方错误卡）")
            reason, suggestion = _classify_error(exc)
            self.sender.send_card(context.chat_id, build_error_card(label, reason, suggestion))
            # 错误卡是 best-effort（断网时静默丢失）：补一条必达文本进外发箱，
            # 断网排队、恢复补发——失败事件用户一定会知道（真实事故教训）
            short_reason = reason.strip().splitlines()[0][:80]
            self._enqueue_notify(
                context.chat_id, f"❌ 精读{label}失败：{short_reason}（可重发链接重试）"
            )
            self._mark_task_finished(dedup_key, ok=False)  # failed：不再自动恢复
            return
        finally:
            self._release_inflight(dedup_key)
        status.stop(ok=True, note=f"精读完成，文档与详情见下方结果卡")
        arxiv_id = kwargs.get("arxiv_id", "")
        self.sender.send_card(
            context.chat_id,
            build_result_card(metadata, doc_url, arxiv_id, archived,
                              timing_text=str(metadata.get("_timing", ""))),
        )
        logger.info("精读任务完成：%s → %s", label, doc_url)
        # 完成钩子：切换到绑定会话等（失败不影响主流程）
        if self.on_done is not None:
            try:
                title = metadata.get("title_zh") or metadata.get("title_en") or label
                note = self.on_done(context, title, arxiv_id)
                if note:
                    self.sender.send_text(context.chat_id, note)
            except Exception:
                logger.exception("精读完成钩子执行失败（不影响主流程）")
        # 放在成功路径最后：进程在发卡前死掉，任务保持 running，重启还能恢复补跑
        self._mark_task_finished(dedup_key, ok=True)

    def resume_interrupted(self) -> int:
        """恢复上次进程死在中途的精读任务（reading_tasks 里 status=running 的），
        返回成功重启的任务数。

        语义边界：只恢复 running（进程被杀）；failed（重试耗尽已走失败路径并通知过
        用户）不恢复，防止恢复死循环。逐条重建 MessageContext 调 _start_async：
        在飞锁使「重启后用户又手动触发同一篇」自动去重，job 侧检查点让重跑续跑。
        恢复通知经外发箱必达（重启时网络可能还没好，直发会丢）。
        单条恢复失败只记日志，不影响其他任务与启动流程。
        """
        boot_ts = time.time()
        try:
            tasks = self.task_store.interrupted(older_than=boot_ts)
        except Exception:
            logger.exception("读取中断精读任务失败（不影响启动）")
            return 0
        resumed = 0
        for task in tasks:
            try:
                if self._resume_one(task):
                    resumed += 1
            except Exception:
                logger.exception("恢复中断精读任务失败（跳过，不影响其他）：%s", task.label)
        if tasks:
            logger.info("中断精读任务恢复完成：%d/%d 已重启", resumed, len(tasks))
        return resumed

    def _resume_one(self, task: ReadingTask) -> bool:
        """按登记信息重建单条任务。返回是否真的重启了（在飞去重拦截返回 False）。"""
        if task.kind == "arxiv":
            job, kwargs = self._job_arxiv, {"arxiv_id": task.ref}
        elif task.kind == "url":
            job, kwargs = self._job_url, {"url": task.ref}
        elif task.kind == "file":
            file_name = (
                task.label[1:-1]
                if task.label.startswith("《") and task.label.endswith("》")
                else task.label
            )
            job = self._job_attachment
            kwargs = {"file_name": file_name, "file_key": task.ref,
                      "message_id": task.message_id}
        else:
            logger.warning("未知的精读任务类型，跳过恢复（7 天后自动清理）：%s", task.dedup_key)
            return False
        context = MessageContext(
            chat_id=task.chat_id, open_id=task.open_id, message_id=task.message_id,
        )
        r = self._start_async(context, label=task.label, job=job,
                              dedup_key=task.dedup_key, **kwargs)
        if r is None:
            logger.info("中断任务已在飞（重启后又被触发），不重复恢复：%s", task.label)
            return False
        # 恢复通知走外发箱而非直发：重启时网络可能还没恢复（WS 重连需时间），
        # 直发会静默丢；入队后由 flusher 周期补发，必达。
        self._enqueue_notify(
            task.chat_id, f"🔁 上次中断的精读任务已自动恢复：{task.label}"
        )
        logger.info("中断精读任务已自动恢复：%s（kind=%s）", task.label, task.kind)
        return True

    def answer_in_session(self, session: Any, text: str) -> str:
        """绑定精读会话内的问答：带该会话绑定论文的全文上下文。"""
        paper = self.store.get_paper_by_arxiv(session.bind_ref) if session.bind_ref else None
        if paper is None:
            paper = self.store.latest_paper(session.chat_id)
        if paper is None:
            return "本会话绑定的论文找不到了（可能已被清理），发「切换 N」换个会话聊吧。"
        context_md = truncate_markdown(paper.markdown, self.max_input_chars)
        messages = [
            {"role": "system", "content": _FOLLOWUP_SYSTEM},
            {"role": "user", "content": (
                f"论文《{paper.title}》全文（markdown，可能经截断）：\n\n{context_md}"
                f"\n\n---\n用户提问：{text}"
            )},
        ]
        try:
            return self.kimi.chat(messages, heavy=True)
        except KimiError as exc:
            return f"调用大模型出错了：{exc}"

    # -- 两种 PDF 来源的 job --

    def _job_arxiv(self, context: MessageContext, arxiv_id: str):
        with self.paper_cache.acquire(
            f"arxiv:{arxiv_id}",
            lambda work_dir: download_arxiv_pdf(arxiv_id, work_dir),
        ) as cache_entry:
            pdf_path = cache_entry.pdf_path
            size_mb = pdf_path.stat().st_size / 1024 / 1024
            source = "缓存命中" if cache_entry.cache_hit else "下载完成"
            self._progress(context, f"✅ {source}（{size_mb:.1f} MB），开始解析 PDF（约 1~2 分钟）……")
            meta_hint = fetch_arxiv_metadata(arxiv_id)
            return self._pipeline(
                context, pdf_path, key=arxiv_id,
                pdf_url=arxiv_pdf_url(arxiv_id), arxiv_id=arxiv_id, meta_hint=meta_hint,
                cache_entry=cache_entry,
            )

    def _job_url(self, context: MessageContext, url: str):
        """非 arXiv 论文链接：解析出 PDF（页面 citation_pdf_url）→ 与 arXiv 同一管线。"""
        import hashlib

        with track("解析论文链接"):
            pdf_url, hint = resolve_paper_url(url)
        key = "url_" + hashlib.md5(url.encode()).hexdigest()[:10]
        with self.paper_cache.acquire(
            f"url:{pdf_url}",
            lambda work_dir: download_pdf_url(pdf_url, work_dir),
        ) as cache_entry:
            pdf_path = cache_entry.pdf_path
            size_mb = pdf_path.stat().st_size / 1024 / 1024
            source = "缓存命中" if cache_entry.cache_hit else "下载完成"
            self._progress(context, f"✅ {source}（{size_mb:.1f} MB），开始解析 PDF（约 1~2 分钟）……")
            meta_hint = dict(hint)
            meta_hint.setdefault("title", key)
            meta_hint["source"] = "链接精读"
            return self._pipeline(
                context, pdf_path, key=key, pdf_url=pdf_url, arxiv_id="",
                meta_hint=meta_hint, cache_entry=cache_entry, dedup_key=url,
            )

    def _job_attachment(self, context: MessageContext, file_name: str, file_key: str, message_id: str):
        key = f"attach_{int(time.time())}"

        def download(work_dir: Path) -> Path:
            with track("下载附件"):
                return self.drive.download_message_resource(
                    message_id, file_key, work_dir / "paper.pdf"
                )

        private_scope = context.open_id or context.chat_id or "private-paper"
        with self.paper_cache.acquire(
            f"attachment:{private_scope}:{file_key}", download,
            private_scope=private_scope,
        ) as cache_entry:
            self._progress(context, "✅ 附件已安全缓存，开始解析 PDF（约 1~2 分钟）……")
            return self._pipeline(
                context, cache_entry.pdf_path, key=key, pdf_url="", arxiv_id="",
                meta_hint={"title": file_name.removesuffix(".pdf")},
                cache_entry=cache_entry,
            )

    def _resolve_version(self, arxiv_id: str, dedup_key: str) -> int:
        """从论文库主记录读当前版本：复读 → v{n+1}；首读/无多维表格 → v1。

        查重键：arXiv 论文按 arXivID 字段（兼容改版前的老行），
        无 ID 链接论文按「查重键」字段（归一化链接）。
        调用点在在飞锁保护区内（同一查重键任务串行），并发复读不会撞 vN。
        """
        if self.bitable_dao is None:
            return 1
        prior: dict | None = None
        if arxiv_id:
            prior = self.bitable_dao.find_record_fields("论文库", "arXivID", arxiv_id)
        elif dedup_key:
            prior = self.bitable_dao.find_record_fields("论文库", "查重键", dedup_key)
        if not prior:
            return 1
        try:
            version = int(str(prior.get("版本") or "v1").lstrip("v")) + 1
        except ValueError:
            version = 2
        logger.info("复读：%s → 版本 v%d", arxiv_id or dedup_key, version)
        return version

    # -- 公共精读管线 --

    def _progress(self, context: MessageContext, text: str, *, persist: bool = False) -> None:
        """阶段切换：默认只更新状态条阶段行（不刷屏）；persist=True 才发真实消息
        （仅文档链接这类需要留存的信息）。发送失败仅记日志（不打断管线）。"""
        logger.info("进度：%s", text)
        status = getattr(context, "status", None)
        if status is not None:
            status.set_stage(text)
        if persist:
            try:
                self.sender.send_text(context.chat_id, text)
            except Exception:
                logger.exception("进度推送失败")

    def _pipeline(
        self,
        context: MessageContext,
        pdf_path: Path,
        *,
        key: str,
        pdf_url: str,
        arxiv_id: str,
        meta_hint: dict[str, str],
        cache_entry: PaperCacheEntry | None = None,
        dedup_key: str = "",
    ) -> tuple[dict, str, bool]:
        """解析 → 教学精读 → 整篇终审 → 一次性写文档 → 归档 → 落库 → 向量。
        dedup_key：无 arXiv ID 的链接论文的查重键（归一化链接），版本演进与归档查重用。"""
        t_start = time.time()
        with track("MinerU 解析 PDF"):
            md_path = (
                self.paper_cache.ensure_parsed(
                    cache_entry,
                    lambda source, output: parse_pdf(source, output),
                )
                if cache_entry is not None
                else parse_pdf(pdf_path, pdf_path.parent)
            )
        paper_md = md_path.read_text(encoding="utf-8")
        figures = load_figure_assets(md_path.parent)
        formulas = load_formula_assets(md_path.parent)
        pages = pdf_page_count(pdf_path)
        self._progress(
            context,
            f"✅ 解析完成（{pages or '?'} 页，{len(figures)} 张图表，{len(formulas)} 个公式，"
            f"{time.time() - t_start:.0f}s）。"
            "开始精读生成：骨架、分层精讲、逐图解释完成后会整篇终审，通过后再写入文档。",
        )

        images = {asset.label: asset.img_path for asset in figures}
        # 版本方案 B：从论文库主记录读当前版本，复读 → 新版本文档 v{n}。
        # 此处处于在飞锁保护区（同一查重键同时只有一个任务跑到这里），
        # 版本读-改-写不会并发撞号（并发双 vN、版本记录自相矛盾的根因）。
        version = self._resolve_version(arxiv_id, dedup_key)
        reading_kwargs = {
            "arxiv_id": arxiv_id,
            "meta_hint": meta_hint,
            "progress": lambda s: self._progress(context, s),
        }
        if cache_entry is not None and isinstance(self.reading_engine, ReadingEngine):
            # 检查点和 PDF 共用内容寻址对象目录；进程退出后租约自动释放，复读可续跑。
            workspace = ReadingWorkspace(cache_entry.object_dir)
            with workspace.lease():
                reading = self.reading_engine.run(
                    paper_md, figures, formulas, workspace=workspace, **reading_kwargs,
                )
        else:
            # 测试 fake / 非缓存入口保持原协议，避免向第三方注入实现强塞新参数。
            reading = self.reading_engine.run(
                paper_md, figures, formulas, **reading_kwargs,
            )

        metadata = reading.metadata
        metadata["_version"] = version
        title = metadata.get("title_zh") or metadata.get("title_en") or key
        metadata["direction"] = classify_direction(
            self.kimi, title, str(metadata.get("tldr", ""))
        )
        logger.info("方向分类：%s → %s", title, metadata["direction"])
        images.update(reading.formula_images)

        # 产出优先：发布失败只重试发布，绝不重新生成正文
        doc_id, doc_url = "", ""
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                doc_id, doc_url = self.docx.create_document(
                    doc_display_name(metadata, arxiv_id, version=version)
                )
                granted = self.docx.grant_access(doc_id, open_id=context.open_id or None)
                if not granted:
                    raise DocxError("文档权限授予全部失败，未发布不可访问的文档")
                with track("写入飞书文档"):
                    self.docx.write_markdown(
                        doc_id, reading.body_md, images=images, strict_images=True,
                    )
                last_err = None
                break
            except Exception as exc:  # 发布服务故障：有界重试
                last_err = exc
                logger.warning("飞书发布失败（第 %d 次）：%s", attempt + 1, exc)
                time.sleep(min(30, 5 * (attempt + 1)))
        if last_err is not None:
            # 保住已生成 Markdown：落盘本地 + 通知，不重跑管线
            fallback = Path("data") / f"unpublished-{arxiv_id or key}.md"
            fallback.write_text(reading.body_md, encoding="utf-8")
            metadata["publish_failed"] = str(last_err)
            metadata["unpublished_body"] = str(fallback)
            logger.error("发布三次仍失败，正文已保存到 %s", fallback)
            self._progress(
                context,
                f"⚠️ 精读已生成但飞书发布失败（{last_err}）；正文已保存本地：{fallback}",
                persist=True,
            )
            return metadata, "", False
        t_doc = time.time()

        with track("归档多维表格"):
            archived = self._archive(metadata, doc_url, pdf_url, arxiv_id, dedup_key)
        self.store.save_paper(
            context.chat_id, arxiv_id or key, title,
            paper_md, doc_url=doc_url, pdf_path=str(pdf_path),
        )
        if self.vector_index is not None and arxiv_id:
            self.vector_index.add_paper(
                arxiv_id, title,
                str(metadata.get("tldr", "")), metadata.get("keywords") or [],
            )
        self._progress(context, f"📄 精读文档已通过终审并写入：{doc_url}", persist=True)

        # 计时埋点：写日志并随完成卡展示（回应速度质疑用）
        metadata["_timing"] = self._timing_text(reading, t_start, t_doc, time.time())
        logger.info("精读计时：%s", metadata["_timing"].replace(chr(10), " | "))
        return metadata, doc_url, archived

    @staticmethod
    def _timing_text(reading: ReadingResult, t_start: float, t_doc: float, t_end: float) -> str:
        timer = reading.timer
        parts = []
        if timer is not None:
            parts.append(f"骨架 {timer.elapsed('骨架'):.0f}s")
            layer_marks = [n for n, _ in timer.marks if n.startswith("层")]
            if layer_marks:
                layers_secs = "、".join(f"层{i + 1} {timer.elapsed(n):.0f}s"
                                       for i, n in enumerate(layer_marks))
                parts.append(
                    f"精讲 {timer.duration('精讲开始', '精讲'):.0f}s（{layers_secs}）"
                )
            parts.append(f"图表 {timer.elapsed('图表'):.0f}s")
        parts.append(f"文档 {t_end - t_doc:.0f}s")
        parts.append(f"全程 {t_end - t_start:.0f}s")
        parts.append(f"模型调用 {reading.model_calls} 次")
        return "⏱ " + " · ".join(parts)

    def _archive(self, metadata: dict, doc_url: str, pdf_url: str, arxiv_id: str,
                 dedup_key: str = "") -> bool:
        """写入论文库一行；未初始化多维表格则跳过。返回是否已归档。

        版本方案 B：一行 = 一篇论文。复读时「文档链接」指向最新版、「版本」+1、
        旧链接串入「历史版本」；精读时间取最新一轮。
        查重：arXiv 论文按 arXivID 字段（兼容老行），无 ID 链接论文按「查重键」
        字段（归一化链接）——链接复读同样更新主记录，不再重复建行。
        """
        if self.bitable_dao is None:
            logger.warning("多维表格未初始化，跳过论文库归档")
            return False
        try:
            self._ensure_paper_fields()
            row = build_paper_row(
                metadata, doc_url, pdf_url, arxiv_id,
                direction=str(metadata.get("direction", "")),
                source=str(metadata.get("source", "精读")),
            )
            version = int(metadata.get("_version", 1))
            row["版本"] = f"v{version}"
            if dedup_key:
                row["查重键"] = dedup_key
            # 查重定位主记录：arXiv → arXivID 字段；链接 → 查重键字段；都无 → 必建新行
            id_field = "arXivID" if arxiv_id else "查重键"
            id_value = arxiv_id or dedup_key
            existing_id = (
                self.bitable_dao.find_record_by_field("论文库", id_field, id_value)
                if id_value else None
            )
            if existing_id:
                prior = self.bitable_dao.find_record_fields("论文库", id_field, id_value) or {}
                prev_version = str(prior.get("版本") or "v1")
                old_link = prior.get("文档链接")
                old_url = old_link.get("link", "") if isinstance(old_link, dict) else ""
                history = str(prior.get("历史版本") or "").strip()
                if old_url:
                    history = (history + "\n" if history else "") + f"{prev_version}: {old_url}"
                row["历史版本"] = history
                self.bitable_dao.update_record("论文库", existing_id, row)
                logger.info("论文库主记录更新（%s → v%d，历史版本 %d 条）：record=%s",
                            id_value, version, history.count("http"), existing_id)
                self._backfill_pick_status(arxiv_id)
                return True
            record_id = self.bitable_dao.add_record("论文库", row)
            logger.info("论文库已归档：record=%s", record_id)
            self._backfill_pick_status(arxiv_id)
            return True
        except Exception as exc:
            logger.exception("论文库归档失败（不影响其他产出）：%s", exc)
            return False

    def _backfill_pick_status(self, arxiv_id: str) -> None:
        """精选库状态回填：日报精选被精读后，把对应行状态更新为「已精读」。

        闭环：arXiv 日报写入精选库（未读）→ 用户触发精读 → 论文库归档 → 此回填。
        失败仅记 WARNING（绝不阻断精读主流程）。
        TODO: 「已读」状态暂不自动触发，留待精选库卡片/面板按钮回调
        （需消息卡片交互权限与回调处理）后再接。
        """
        if not arxiv_id or self.bitable_dao is None:
            return
        try:
            record_id = self.bitable_dao.find_record_by_field("精选库", "arXivID", arxiv_id)
            if not record_id:
                return  # 非日报精选来源，无需回填
            self.bitable_dao.update_record("精选库", record_id, {"状态": "已精读"})
            logger.info("精选库状态回填：%s → 已精读（record=%s）", arxiv_id, record_id)
        except BitableError as exc:
            logger.warning("精选库状态回填失败（不影响归档）：%s", exc)

    def _ensure_paper_fields(self) -> None:
        """给已有论文库补 schema 演进字段（方向/来源/版本/历史版本/查重键），进程内只做一次。"""
        if self._paper_fields_ensured or self.bitable_dao is None:
            return
        self._paper_fields_ensured = True
        table_id = self.bitable_dao.tables.get("论文库", "")
        if not table_id:
            return
        try:
            created = ensure_fields(
                self.lark_client, self.bitable_dao.app_token, table_id,
                [FieldSpec("方向", FIELD_TYPE_SINGLE_SELECT, options=PAPER_DIRECTIONS),
                 FieldSpec("来源", FIELD_TYPE_TEXT),
                 FieldSpec("版本", FIELD_TYPE_TEXT),
                 FieldSpec("历史版本", FIELD_TYPE_TEXT),
                 FieldSpec("查重键", FIELD_TYPE_TEXT)],
            )
            if created:
                logger.info("论文库补充字段：%s", "、".join(created))
        except BitableError as exc:
            logger.warning("补充字段失败（写入时若缺字段会再报错）：%s", exc)


# 瞬时网络故障类型：连接/超时/DNS 类——断网恢复后重跑同一 job 才有意义。
# 覆盖四层来源：内置（飞书 SDK/标准库）、httpx（arXiv/链接下载）、
# openai（大模型 SDK）、requests/urllib3（lark-oapi 传输层）。
_TRANSIENT_ERROR_TYPES: tuple[type[BaseException], ...] = (
    ConnectionError,          # 内置：含 ConnectionResetError/AbortedError/RefusedError/BrokenPipeError
    TimeoutError,             # 内置（socket.timeout 自 3.10 起是其别名）
    socket.gaierror,          # DNS 解析失败
    httpx.TimeoutException,   # ConnectTimeout/ReadTimeout/WriteTimeout/PoolTimeout
    httpx.NetworkError,       # ConnectError/ReadError/WriteError/CloseError
    openai.APIConnectionError,  # 含其子类 APITimeoutError
    requests.exceptions.ConnectionError,  # 含 ConnectTimeout（同时是 Timeout 子类）
    requests.exceptions.Timeout,          # ReadTimeout 等
    urllib3.exceptions.NameResolutionError,  # DNS
    urllib3.exceptions.ConnectTimeoutError,
    urllib3.exceptions.NewConnectionError,
)


def _is_transient_error(exc: BaseException) -> bool:
    """异常（含 __cause__/__context__ 包装链）是否瞬时网络故障：断网/超时/DNS 类。

    业务包装（KimiError/BitableError/DriveError/DocxError）下面可能压着
    httpx/requests/urllib3/openai 的网络错误，沿链逐层判定。
    非网络错误（验收失败、凭证错误、参数/业务校验）永远返回 False——
    这类错误重跑只会延迟用户看到真实原因，绝不重试。
    """
    seen: set[int] = set()
    stack: list[BaseException | None] = [exc]
    while stack:
        cur = stack.pop()
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))
        if isinstance(cur, _TRANSIENT_ERROR_TYPES):
            return True
        stack.append(cur.__cause__)
        stack.append(cur.__context__)
    return False


def _classify_error(exc: Exception) -> tuple[str, str]:
    """把异常翻译成给用户看的中文原因 + 建议。"""
    if isinstance(exc, PaperPipelineError):
        return str(exc), "检查 arXiv 链接是否正确后重试；也可以换成直接发 PDF 文件。"
    if isinstance(exc, LifecycleError):
        return str(exc), "本地缓存正在清理或磁盘空间不足；稍后重试，系统不会强删运行中的任务。"
    if isinstance(exc, PdfParseError):
        return str(exc), "可以换个清晰的 PDF 重试；若是扫描件，暂建议先发 arXiv 链接。"
    if isinstance(exc, KimiError):
        return str(exc), "当前选择的 API/模型不可用；系统没有自动切换，请检查 key、额度或服务状态后重试，也可在「模型」面板手动选择。"
    if isinstance(exc, ReadingError):
        return str(exc), "自动质检没有放行低质量文档；可稍后重试，图表缺失时请换清晰 PDF 或论文原始链接。"
    if isinstance(exc, DocxError):
        return str(exc), "检查应用是否开通 docx:document 权限并已发布新版本。"
    if isinstance(exc, DriveError):
        return str(exc), "检查应用是否开通 im:resource 权限并已发布新版本。"
    if isinstance(exc, BitableError):
        return str(exc), "检查多维表格权限（bitable:app）或重新运行 feishu-assistant init。"
    if isinstance(exc, httpx.HTTPError):
        return f"网络请求失败：{exc}", "检查网络后重试；arXiv 偶尔抽风，换个时间再发。"
    return f"{type(exc).__name__}：{exc}", "把这条错误发给开发者排查（日志里有完整 traceback）。"
