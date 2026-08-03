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
import time
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT, Settings
from ..feishu.bitable import BitableError
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


def should_handle(text: str) -> bool:
    """收件箱命中规则：含非 arXiv 链接，或以归档前缀开头。闲聊一律 False。"""
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith(INBOX_PREFIXES):
        return True
    url = extract_url(stripped)
    if url and parse_arxiv_id(stripped) is None:
        return True  # 有链接且不是 arXiv（arXiv 归精读管线）
    return False


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

    def __init__(self, settings: Settings, kimi: KimiClient, io: Any = None) -> None:
        self.settings = settings
        self.kimi = kimi
        self.io = io  # CareerIO 协议（写表走 .dao.add_record）

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
            row = {
                "内容": item["raw"][:500],
                "方向": item["direction"],
                "时间": now_ms,
            }
            if item["tags"]:
                row["标签"] = item["tags"]
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
        """命中规则则分类入库并返回确认；不命中返回 None（交给自由对话）。"""
        if not should_handle(text):
            return None
        item, fallback = self._classify(text.strip())
        try:
            self._save(item)
        except BitableError as exc:
            return f"❌ 收件箱入库失败：{exc}"
        return self._confirm(item, fallback)
