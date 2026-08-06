"""P4 万能收件箱：命中规则的自然语言 → light 分类 → 收藏库/灵感库/备忘库。

- 命中规则（保守，闲聊绝不接管）：
  (a) 含 http(s) 链接（arXiv 链接除外——那是精读管线的地盘）
  (b) 以「记一下/备忘/提醒我/收藏/灵感」开头
- 分类：light 模型按 prompts/inbox_classify.md 出 JSON（类型/标题/摘要/标签/
  方向/截止时间），容错解析 + 规则兜底（含链接→收藏，否则→灵感）
- 截止时间：复用 career.parse_date_text（今天/明天/N天后/YYYY-MM-DD…）
- 确认回复：入哪类 / 标签 / 提醒时间；备忘无时间时提示如何补提醒

注入点：kimi（fake 出 JSON）、io（CareerIO 协议：.dao.add_record），全离线可测。
"""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT, Settings
from ..feishu.bitable import (
    BitableError,
    FIELD_TYPE_TEXT,
    FIELD_TYPE_URL,
    FieldSpec,
    ensure_fields,
)
from ..kimi import KimiClient, KimiError
from .career import extract_json, ms_to_date, parse_date_text
from .paper import MessageContext, parse_arxiv_id

logger = logging.getLogger("feishu_assistant.inbox")

PROMPTS_DIR = PROJECT_ROOT / "prompts"
CLASSIFY_PROMPT_PATH = PROMPTS_DIR / "inbox_classify.md"

COLLECT_TABLE = "收藏库"
IDEA_TABLE = "灵感库"
MEMO_TABLE = "备忘库"

INBOX_PREFIXES = ("记一下", "备忘", "提醒我", "收藏", "灵感")
IDEA_DIRECTIONS = ("Agent Harness", "大模型开发", "其他")

# 与 paper.py 的 _URL_RE 同款（排除收尾的 <>])} 与中文标点「，。」「。」）：
# 该正则是 paper.py 模块私有，两处保持同步，改一边要同步另一边。
_URL_RE = re.compile(r"https?://[^\s<>\])}，。]+")


def extract_url(text: str) -> str:
    """第一条 http(s) 链接（不含收尾的中文标点/括号）；没有返回 ""。"""
    m = _URL_RE.search(text)
    return m.group(0) if m else ""


# 归档意图（动作词+目标库名）：「个人随笔：…写进灵感库」「把这段话记录进灵感」
_ARCHIVE_ACTION = r"(?:写进|写入|记录进|记录到|记到|存入|存到|存下|收录到?|放到?|保存到?|归档到?)"
_ARCHIVE_TARGET = r"(?:灵感库?|收藏库?|备忘库?)"
_ARCHIVE_INTENT_RE = re.compile(
    rf"({_ARCHIVE_ACTION}).{{0,8}}({_ARCHIVE_TARGET})")
# 引导兜底：提了库名又像是想写但没写对（排除「灵感库里有几条记录？」这类纯提问）
_TABLE_NAME_RE = re.compile(r"(灵感库|收藏库|备忘库)")
_QUESTION_HINT_RE = re.compile(r"(吗|呢|？|\?)$|怎么|什么|^(哪|是|能)")

_ARCHIVE_DEICTIC = (
    r"(?:这段话|这段内容|这段文字|这个|这个链接|以上内容|以下内容|"
    r"上面这段|刚才这段|刚才那段|我的随笔|我的想法)"
)
_ARCHIVE_COMMAND_PREFIX_RE = re.compile(
    rf"^\s*(?:请|麻烦|帮我)?\s*(?:把\s*)?(?:{_ARCHIVE_DEICTIC}\s*)?"
    rf"{_ARCHIVE_ACTION}\s*.{{0,8}}?{_ARCHIVE_TARGET}\s*[：:,，]?\s*"
)
_ARCHIVE_COMMAND_SUFFIX_RE = re.compile(
    rf"\s*(?:[,，；;]\s*)?(?:请|麻烦|帮我)?\s*(?:把\s*)?"
    rf"(?:{_ARCHIVE_DEICTIC}\s*)?{_ARCHIVE_ACTION}\s*.{{0,8}}?"
    rf"{_ARCHIVE_TARGET}\s*(?:谢谢|麻烦了)?\s*$"
)
_ARCHIVE_SIMPLE_PREFIX_RE = re.compile(
    r"^\s*(?:记一下|备忘|提醒我|收藏|灵感)\s*[：:,，]?\s*"
)
_TAG_DIRECTIVE_RE = re.compile(
    r"(?:[,，；;]\s*)?(?:标签|tags?)\s*[：:]?\s*(?P<tags>[^。；;\n]+)\s*$",
    re.IGNORECASE,
)

IDEA_ASSOCIATION_FIELDS = (
    FieldSpec("关联论文", FIELD_TYPE_TEXT),
    FieldSpec("论文ID", FIELD_TYPE_TEXT),
    FieldSpec("论文链接", FIELD_TYPE_URL),
)


def explicit_archive_kind(text: str) -> str:
    """用户明确点名目标库时返回对应类型；没有明确目标返回空串。"""
    stripped = text.strip()
    match = _ARCHIVE_INTENT_RE.search(stripped)
    if match:
        target = match.group(2)
        if target.startswith("灵感"):
            return "灵感"
        if target.startswith("收藏"):
            return "收藏"
        return "备忘"
    if stripped.startswith("灵感") and not stripped.startswith("灵感库"):
        return "灵感"
    if stripped.startswith("收藏") and not stripped.startswith("收藏库"):
        return "收藏"
    if stripped.startswith(("备忘", "提醒我")) and not stripped.startswith("备忘库"):
        return "备忘"
    return ""


def has_explicit_archive_intent(text: str) -> bool:
    """是否应在论文追问/绑定会话前优先交给确定性写库路径。"""
    return bool(explicit_archive_kind(text))


def explicit_tags(text: str) -> list[str]:
    """提取末尾显式标签指令；未写分隔符时整段视为一个标签。"""
    match = _TAG_DIRECTIVE_RE.search(text.strip())
    if match is None:
        return []
    raw = match.group("tags").strip()
    values = re.split(r"[、,，/|]+", raw)
    return [value.strip() for value in values if value.strip()][:4]


def archive_content(text: str, previous_text: str = "") -> str:
    """剥离归档动作、目标库和标签指令，只返回真正要存的正文。

    「把这段话记录进灵感」这类纯指代命令会回退到 previous_text；若当前消息
    同时包含正文和尾随命令，只删命令，不碰正文。
    """
    content = text.strip()
    content = _TAG_DIRECTIVE_RE.sub("", content).strip()
    content = _ARCHIVE_COMMAND_PREFIX_RE.sub("", content, count=1).strip()
    content = _ARCHIVE_COMMAND_SUFFIX_RE.sub("", content, count=1).strip()
    content = _ARCHIVE_SIMPLE_PREFIX_RE.sub("", content, count=1).strip()
    content = content.strip(" \t\r\n，,；;：:")
    compact = re.sub(r"\s+", "", content)
    references = {
        "", "这段话", "这段内容", "这段文字", "这个", "以上内容", "以下内容",
        "上面这段", "刚才这段", "刚才那段", "我的随笔", "我的想法",
    }
    if compact in references:
        return previous_text.strip()
    return content


def should_handle(text: str) -> bool:
    """收件箱命中规则：含非 arXiv 链接，或以归档前缀开头，或显式归档意图。闲聊一律 False。"""
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith(INBOX_PREFIXES) \
            and not stripped.startswith(("灵感库", "收藏库", "备忘库")):
        # 「灵感」是归档前缀，但「灵感库怎么用/灵感库里有什么」是在问库，不是归档
        return True
    if _ARCHIVE_INTENT_RE.search(stripped):
        return True  # 「写进/存入/收录到 + 目标库」的自然说法（血泪：此前漏判致模型编造已收录）
    url = extract_url(stripped)
    if url and parse_arxiv_id(stripped) is None:
        return True  # 有链接且不是 arXiv（arXiv 归精读管线）
    return False


def archive_guidance(text: str) -> str | None:
    """有归档意图但说法没命中规则时的确定性引导（不走模型，防止模型编造「已收录」）。
    纯提问（「灵感库怎么用」「备忘库有几条记录？」）不引导，返回 None 交给正常路径。"""
    stripped = text.strip()
    if not _TABLE_NAME_RE.search(stripped):
        return None
    if _QUESTION_HINT_RE.search(stripped):
        return None
    return (
        "这条我没法直接帮你存进库里（写库要走固定入口，我代写不了）。"
        "正确说法举例：\n"
        "· 「记一下：内容」→ 存备忘（要提醒就带个时间，如「记一下 明天交表」）\n"
        "· 「写进灵感库：内容」→ 存灵感（可再补一句「标签 xxx」）\n"
        "· 「收藏：链接或内容」→ 存收藏\n"
        "你再发一次，我就按规则入库。"
    )


def _to_tags(value: Any, limit: int = 4) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()][:limit]


def normalize_classification(data: dict[str, Any], text: str, now: float | None = None) -> dict[str, Any]:
    """把模型 JSON 规整为入库项：类型校验 + 规则兜底 + 日期转毫秒。"""
    kind = str(data.get("type") or "").strip()
    url = extract_url(text)
    if kind not in ("收藏", "灵感", "备忘"):
        kind = "收藏" if url else "灵感"
    direction = str(data.get("direction") or "").strip()
    if direction not in IDEA_DIRECTIONS:
        direction = "其他"
    deadline_ms = None
    deadline_raw = str(data.get("deadline") or "").strip()
    if deadline_raw:
        deadline_ms = parse_date_text(deadline_raw, now)
    return {
        "type": kind,
        "title": str(data.get("title") or "").strip(),
        "summary": str(data.get("summary") or "").strip(),
        "tags": _to_tags(data.get("tags")),
        "direction": direction,
        "deadline_ms": deadline_ms,
        "note": str(data.get("note") or "").strip(),
        "url": url,
        "raw": text,
    }


def fallback_classification(text: str) -> dict[str, Any]:
    """模型失败时的规则兜底：含链接→收藏（标题用域名），否则→灵感。"""
    url = extract_url(text)
    domain = re.sub(r"^https?://", "", url).split("/", 1)[0] if url else ""
    return {
        "type": "收藏" if url else "灵感",
        "title": domain or "未命名收藏",
        "summary": text[:100],
        "tags": [],
        "direction": "其他",
        "deadline_ms": None,
        "note": "",
        "url": url,
        "raw": text,
    }


class InboxService:
    """收件箱总入口：try_handle 命中即接管（router 的 inbox_fn 接线）。"""

    def __init__(
        self,
        settings: Settings,
        kimi: KimiClient,
        io: Any = None,
        *,
        session_store: Any = None,
        paper_store: Any = None,
    ) -> None:
        self.settings = settings
        self.kimi = kimi
        self.io = io  # CareerIO 协议（写表走 .dao.add_record）
        self.session_store = session_store
        self.paper_store = paper_store
        self._idea_fields_ensured = False
        self._idea_fields_lock = threading.Lock()

    def _ensure_idea_fields(self) -> None:
        """给既有灵感库幂等补论文关联字段；fake IO/未初始化时跳过。"""
        if self._idea_fields_ensured or self.io is None:
            return
        dao = getattr(self.io, "dao", None)
        client = getattr(dao, "client", None)
        app_token = str(getattr(dao, "app_token", "") or "")
        tables = getattr(dao, "tables", {}) or {}
        table_id = str(tables.get(IDEA_TABLE, "") or "")
        if client is None or not app_token or not table_id:
            return
        with self._idea_fields_lock:
            if self._idea_fields_ensured:
                return
            created = ensure_fields(
                client, app_token, table_id, list(IDEA_ASSOCIATION_FIELDS)
            )
            self._idea_fields_ensured = True
            if created:
                logger.info("灵感库补充关联字段：%s", "、".join(created))

    def _previous_user_content(self, ctx: MessageContext | None) -> str:
        """取活跃会话里最近一条有效用户正文，供「把这段话记下」解析指代。"""
        if ctx is None or not ctx.chat_id or self.session_store is None:
            return ""
        try:
            session = self.session_store.get_active_session(ctx.chat_id)
            if session is None:
                return ""
            messages = self.session_store.recent_messages(session.id, limit=20)
        except Exception as exc:
            logger.warning("读取归档指代上文失败（按无上文处理）：%s", exc)
            return ""
        for message in reversed(messages):
            if getattr(message, "role", "") != "user":
                continue
            candidate = archive_content(str(getattr(message, "content", "") or ""))
            if candidate:
                return candidate
        return ""

    @staticmethod
    def _bound_title(session: Any) -> str:
        title = str(getattr(session, "title", "") or "").strip()
        match = re.fullmatch(r"《(.+)》精读", title)
        return match.group(1).strip() if match else title

    def _active_paper(self, ctx: MessageContext | None) -> dict[str, str]:
        """当前论文绑定会话对应的确定性关联信息；无绑定则不猜。"""
        if ctx is None or not ctx.chat_id or self.session_store is None:
            return {}
        try:
            session = self.session_store.get_active_session(ctx.chat_id)
        except Exception as exc:
            logger.warning("读取灵感关联会话失败（按无关联处理）：%s", exc)
            return {}
        if session is None or getattr(session, "bind_type", "") != "paper":
            return {}

        paper = None
        bind_ref = str(getattr(session, "bind_ref", "") or "").strip()
        if self.paper_store is not None:
            try:
                if bind_ref:
                    paper = self.paper_store.get_paper_by_arxiv(
                        bind_ref, chat_id=ctx.chat_id
                    )
                if paper is None:
                    recent = self.paper_store.recent_papers(ctx.chat_id, limit=8)
                    bound_title = self._bound_title(session)
                    paper = next(
                        (p for p in recent if p.title == bound_title
                         or p.title in bound_title or bound_title in p.title),
                        recent[0] if recent else None,
                    )
            except Exception as exc:
                logger.warning("读取灵感关联论文失败（退化为会话信息）：%s", exc)

        title = str(getattr(paper, "title", "") or "").strip() \
            or self._bound_title(session)
        paper_id = str(getattr(paper, "arxiv_id", "") or "").strip() or bind_ref
        doc_url = str(getattr(paper, "doc_url", "") or "").strip()
        if not title:
            return {}
        return {"title": title, "id": paper_id, "url": doc_url}

    def _classify(self, text: str) -> tuple[dict[str, Any], bool]:
        """light 分类，返回 (入库项, 是否走了兜底)。"""
        template = CLASSIFY_PROMPT_PATH.read_text(encoding="utf-8")
        today = ms_to_date(int(time.time() * 1000))
        prompt = template.replace("{{TEXT}}", text).replace("{{TODAY}}", today)
        try:
            raw = self.kimi.chat([{"role": "user", "content": prompt}], heavy=False)
        except KimiError as exc:
            logger.warning("收件箱分类模型失败（规则兜底）：%s", exc)
            return fallback_classification(text), True
        data = extract_json(raw)
        if data is None:
            logger.warning("收件箱分类 JSON 解析失败（规则兜底）：%s", raw[:120])
            return fallback_classification(text), True
        return normalize_classification(data, text), False

    def _save(self, item: dict[str, Any]) -> None:
        """写入对应表。io 缺失时抛 BitableError（调用方给中文提示）。"""
        if self.io is None:
            raise BitableError("多维表格未初始化，收件箱写不了。先发 /init 初始化。")
        now_ms = int(time.time() * 1000)
        kind = item["type"]
        if kind == "收藏":
            title = item["title"] or "未命名收藏"
            row: dict[str, Any] = {
                "标题": title,
                "摘要": item["summary"] or item["raw"][:100],
                "收藏时间": now_ms,
                "已回顾": False,
                "来源": "收件箱",
            }
            if item["url"]:
                row["链接"] = {"text": title, "link": item["url"]}  # 超链接字段
            if item["tags"]:
                row["标签"] = item["tags"]  # 多选：飞书自动建选项
            self.io.dao.add_record(COLLECT_TABLE, row)
        elif kind == "灵感":
            self._ensure_idea_fields()
            row = {
                "内容": item["raw"][:5_000],
                "方向": item["direction"],
                "时间": now_ms,
            }
            if item["tags"]:
                row["标签"] = item["tags"]
            paper = item.get("paper") or {}
            if paper.get("title"):
                row["关联论文"] = paper["title"]
            if paper.get("id"):
                row["论文ID"] = paper["id"]
            if paper.get("url"):
                row["论文链接"] = {
                    "text": paper.get("title") or "精读文档",
                    "link": paper["url"],
                }
            self.io.dao.add_record(IDEA_TABLE, row)
        else:  # 备忘
            row = {
                "事项": item["raw"][:200],
                "状态": "待办",
                "创建时间": now_ms,
            }
            if item["deadline_ms"]:
                row["截止时间"] = item["deadline_ms"]
            if item["note"]:
                row["备注"] = item["note"]
            self.io.dao.add_record(MEMO_TABLE, row)
        logger.info("收件箱入库：%s（标签 %s）", kind, "、".join(item["tags"]) or "无")

    def _confirm(self, item: dict[str, Any], fallback: bool) -> str:
        kind = item["type"]
        tag_text = f"｜标签：{'、'.join(item['tags'])}" if item["tags"] else ""
        if kind == "收藏":
            reply = f"✅ 已收藏：《{item['title'] or '未命名收藏'}》{tag_text}"
        elif kind == "灵感":
            reply = f"💡 已记灵感（{item['direction']}）{tag_text}：{item['raw'][:40]}"
            paper = item.get("paper") or {}
            if paper.get("title"):
                reply += f"\n📄 关联论文：《{paper['title']}》"
        else:
            reply = f"📌 已记备忘：{item['raw'][:40]}"
            if item["deadline_ms"]:
                reply += f"\n⏰ 截止 {ms_to_date(item['deadline_ms'])}，今日提醒会带上它。"
            else:
                reply += "\n（没提到时间，不会提醒；要提醒就说「提醒我 xx 明天」）"
        if fallback:
            reply += "\n⚠️ 分类模型开小差，已按规则兜底归类，可在表里手动调整。"
        return reply

    def try_handle(self, text: str, ctx: MessageContext | None = None) -> str | None:
        """命中规则则分类入库并返回确认；意图近似但没写对则给确定性引导；
        都不沾边返回 None（交给自由对话）。"""
        if not should_handle(text):
            return archive_guidance(text)
        prior = self._previous_user_content(ctx)
        content = archive_content(text, previous_text=prior)
        if not content:
            return (
                "❌ 没找到要记录的正文。请把内容和指令一起发，例如："
                "「写进灵感库：你的想法」；或先发正文，再说「把这段话记录进灵感」。"
            )
        item, fallback = self._classify(content)
        requested_tags = explicit_tags(text)
        if requested_tags:
            item["tags"] = list(dict.fromkeys([*requested_tags, *item["tags"]]))[:4]
        # 模型只负责摘要/标签等字段；用户明确点名的目标库是确定性约束，禁止改判。
        forced_kind = explicit_archive_kind(text)
        if forced_kind:
            item["type"] = forced_kind
        item["raw"] = content
        if item["type"] == "灵感":
            item["paper"] = self._active_paper(ctx)
        try:
            self._save(item)
        except BitableError as exc:
            return f"❌ 收件箱入库失败：{exc}"
        return self._confirm(item, fallback)
