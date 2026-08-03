"""P3 求职面试模块：看板 / JD 分析 / 简历库 / 面经库 / 录音复盘 / 押题 / 模拟面试。

架构（对齐 paper.py 的成熟模式）：
- 命令入口 cmd_*：由 router 注册表接线（见仓库 P3 接线清单），同步返回 ack
- 重活（heavy 生成 / 转写）全部后台线程跑，StatusMessenger 状态条即 ack：
  阶段行进同一条流式卡（10 分钟窗口内自更新，不逐条推消息），完成定格 + 结果单独一条
- 数据层：BitableDAO（add）+ feishu/career_io.CareerIO（翻页读/更新/删除/附件）
- 纯函数（解析/汇总/prompt 组装）与 IO 分离，全部可离线测试：
  parse_board_operation / parse_exp_json / summarize_board / looks_like_jd 等
- 模拟面试：绑定 career 会话（store.py 原生支持 bind_type=career），
  会话内消息经 try_handle_text 拦截进状态机，「结束面试」出复盘报告

注入点：io（CareerIO）、docx、drive、transcriber、session_store、sender、outbox
均可传 fake，测试无需真实飞书/模型。
"""

from __future__ import annotations

import json
import hashlib
import logging
import re
import shutil
import threading
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from xml.etree import ElementTree

from ..asr import ASRError, transcribe
from ..config import PROJECT_ROOT, Settings
from ..feishu.bitable import (
    INTERVIEW_ROUNDS,
    JOB_DIRECTIONS,
    JOB_STATUS,
    BitableDAO,
    BitableError,
    FIELD_TYPE_CHECKBOX,
    FIELD_TYPE_MULTI_SELECT,
    FIELD_TYPE_NUMBER,
    FIELD_TYPE_TEXT,
    FieldSpec,
    ensure_fields,
)
from ..feishu.career_io import CareerIO
from ..feishu.docx import DocxClient, DocxError
from ..feishu.drive import DriveClient, DriveError
from ..kimi import KimiClient, KimiError
from ..lifecycle import sensitive_workspace
from ..resume_format import parse_tailor_plan, patch_docx_text, verify_tailored_docx
from ..store import NotificationOutbox, Session, SessionStore
from .paper import Heartbeat, MessageContext, Sender, extract_metadata
from .status import StatusMessenger

logger = logging.getLogger("feishu_assistant.career")

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

BOARD_TABLE = "求职看板"
EXP_TABLE = "面经库"
RESUME_TABLE = "简历库"

PROMPTS_DIR = PROJECT_ROOT / "prompts"
JD_PROMPT_PATH = PROMPTS_DIR / "career_jd.md"
EXP_PROMPT_PATH = PROMPTS_DIR / "career_exp.md"
REVIEW_PROMPT_PATH = PROMPTS_DIR / "career_review.md"
PREDICT_PROMPT_PATH = PROMPTS_DIR / "career_predict.md"
INTERVIEWER_PROMPT_PATH = PROMPTS_DIR / "career_interviewer.md"
RESUME_PROMPT_PATH = PROMPTS_DIR / "career_resume.md"
RESUME_TAILOR_PROMPT_PATH = PROMPTS_DIR / "career_resume_tailor.md"

CAREER_DIR = PROJECT_ROOT / "data" / "career"
AUDIO_DIR = CAREER_DIR / "audio"
RESUME_DIR = CAREER_DIR / "resumes"
TAILORED_RESUME_DIR = CAREER_DIR / "tailored"

TERMINAL_STATUS = ("已拒", "已挂", "offer")  # 终态：不再跟进
FOLLOWUP_STALE_DAYS = 7  # 「已投递」超过该天数无进展 → 该跟进

# 看板字段（与 bitable.py schema 对齐）
BOARD_FIELDS = ("公司", "岗位", "岗位方向", "JD原文", "状态", "投递时间", "下次跟进", "内推人", "备注")
BOARD_DATE_FIELDS = ("投递时间", "下次跟进")
BOARD_SELECT_FIELDS = {"岗位方向": JOB_DIRECTIONS, "状态": JOB_STATUS}

# 输入预算（字符）：喂给 heavy 模型的材料上限
MAX_JD_CHARS = 20_000
MAX_TRANSCRIPT_CHARS = 40_000
MAX_RESUME_CHARS = 6_000
MAX_RESUME_STORE_CHARS = 20_000
MAX_EXP_CHARS = 8_000
MAX_DIALOG_CHARS = 30_000

DAO_MISSING = "多维表格未初始化，求职功能用不了。先发 /init 初始化 7 张表。"

END_MARKER = "本场模拟面试已结束"

RESUME_FILE_EXTENSIONS = (".pdf", ".docx")

RESUME_PROFILE_FIELDS = (
    FieldSpec("版本系列", FIELD_TYPE_TEXT),
    FieldSpec("版本号", FIELD_TYPE_NUMBER),
    FieldSpec("上一版本", FIELD_TYPE_TEXT),
    FieldSpec("当前版本", FIELD_TYPE_CHECKBOX),
    FieldSpec("版本变化", FIELD_TYPE_TEXT),
    FieldSpec("版本类型", FIELD_TYPE_TEXT),
    FieldSpec("基于版本ID", FIELD_TYPE_TEXT),
    FieldSpec("分支版本号", FIELD_TYPE_NUMBER),
    FieldSpec("分支当前版本", FIELD_TYPE_CHECKBOX),
    FieldSpec("目标求职ID", FIELD_TYPE_TEXT),
    FieldSpec("目标公司", FIELD_TYPE_TEXT),
    FieldSpec("定制目标岗位", FIELD_TYPE_TEXT),
    FieldSpec("JD指纹", FIELD_TYPE_TEXT),
    FieldSpec("事实校验状态", FIELD_TYPE_TEXT),
    FieldSpec("用户确认状态", FIELD_TYPE_TEXT),
    FieldSpec("格式继承状态", FIELD_TYPE_TEXT),
    FieldSpec("版面校验状态", FIELD_TYPE_TEXT),
    FieldSpec("修改对照", FIELD_TYPE_TEXT),
    FieldSpec("待确认项", FIELD_TYPE_TEXT),
    FieldSpec("是否实际投递", FIELD_TYPE_CHECKBOX),
    FieldSpec("主方向", FIELD_TYPE_TEXT),
    FieldSpec("目标岗位", FIELD_TYPE_MULTI_SELECT),
    FieldSpec("能力标签", FIELD_TYPE_MULTI_SELECT),
    FieldSpec("简历摘要", FIELD_TYPE_TEXT),
    FieldSpec("识别依据", FIELD_TYPE_TEXT),
    FieldSpec("解析状态", FIELD_TYPE_TEXT),
    FieldSpec("内容指纹", FIELD_TYPE_TEXT),
    FieldSpec("简历正文", FIELD_TYPE_TEXT),
)


class CareerError(Exception):
    """求职模块失败，信息为中文（面向用户推送用）。"""


# ---------------------------------------------------------------------------
# 纯函数：JSON 容错提取
# ---------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any] | None:
    """从模型输出里容错提取第一个 JSON 对象；失败返回 None。

    依次尝试：```json 围栏 → 整段 json.loads → 首个 { 到末个 } 截取。
    """
    m = _JSON_FENCE_RE.search(text)
    if m:
        try:
            data = json.loads(m.group(1))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass
    try:
        data = json.loads(text.strip())
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None
    return None


# ---------------------------------------------------------------------------
# 纯函数：日期解析与格式化
# ---------------------------------------------------------------------------

_RELATIVE_DAY_RE = re.compile(r"^(\d+)\s*天(后|前)$")
_FULL_DATE_RE = re.compile(r"^(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})日?$")
_MD_DATE_RE = re.compile(r"^(\d{1,2})[-/月](\d{1,2})日?$")


def _date_ms(year: int, month: int, day: int) -> int | None:
    try:
        import datetime as _dt

        d = _dt.date(year, month, day)
    except ValueError:
        return None
    return int(time.mktime(d.timetuple()) * 1000)


def parse_date_text(text: str, now: float | None = None) -> int | None:
    """把日期描述解析成毫秒时间戳（当天 00:00 本地时间）；解析不了返回 None。

    支持：今天/明天/后天/昨天、N天后/N天前、2026-07-30、2026/7/30、
    2026年7月30日、07-30、7月30日（缺年份取今年）。
    """
    now_ms = int((now if now is not None else time.time()) * 1000)
    today = ms_to_date(now_ms)
    year = int(today[:4])
    s = (text or "").strip()
    if not s:
        return None
    relative = {"今天": 0, "明天": 1, "后天": 2, "昨天": -1, "前天": -2}
    if s in relative:
        return now_ms + relative[s] * 86_400_000
    m = _RELATIVE_DAY_RE.match(s)
    if m:
        delta = int(m.group(1)) * (1 if m.group(2) == "后" else -1)
        return now_ms + delta * 86_400_000
    m = _FULL_DATE_RE.match(s)
    if m:
        return _date_ms(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = _MD_DATE_RE.match(s)
    if m:
        return _date_ms(year, int(m.group(1)), int(m.group(2)))
    return None


def ms_to_date(ms: int | float) -> str:
    """毫秒时间戳 → YYYY-MM-DD（本地时区）。"""
    return time.strftime("%Y-%m-%d", time.localtime(ms / 1000))


# ---------------------------------------------------------------------------
# 纯函数：bitable 记录读取规整
# ---------------------------------------------------------------------------

def record_fields(rec: Any) -> dict[str, Any]:
    """SDK 记录对象或裸 dict → fields dict。"""
    if isinstance(rec, dict):
        return rec.get("fields", rec)
    return getattr(rec, "fields", None) or {}


def record_id_of(rec: Any) -> str:
    if isinstance(rec, dict):
        return str(rec.get("record_id", ""))
    return str(getattr(rec, "record_id", "") or "")


def field_text(value: Any) -> str:
    """bitable 读出的字段值 → 纯文本。

    文本字段读出来是段落段列表 [{"type":"text","text":"..."}]，需拼接；
    单选是字符串；数字/日期原样 str。
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for seg in value:
            if isinstance(seg, dict):
                parts.append(str(seg.get("text", "")))
            else:
                parts.append(str(seg))
        return "".join(parts)
    return str(value)


def field_list(value: Any) -> list[str]:
    """把 bitable 多选/文本字段规整为去重字符串列表。"""
    if value is None:
        return []
    if isinstance(value, list):
        raw = [seg.get("text", "") if isinstance(seg, dict) else seg for seg in value]
    else:
        raw = re.split(r"[\n,，、;；|]+", str(value))
    out: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


def field_ms(value: Any) -> float:
    """bitable 毫秒时间字段 → 浮点毫秒；空值、文本段列表、非数字一律 0.0。

    飞书日期字段正常读出 int/float 毫秒；旧数据或人工改表可能出现数字
    字符串（兼容解析）或段落结构（拼接后不是数字 → 0.0，不能 float() 硬炸）。
    """
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(field_text(value).strip() or "0")
    except ValueError:
        return 0.0


def is_end_interview_command(text: str) -> bool:
    """识别简短、明确的结束意图，同时避免把普通讨论误判为命令。"""
    normalized = re.sub(r"[。！!?？~～，,\s]+$", "", (text or "").strip())
    return bool(re.fullmatch(
        r"(?:请|麻烦你?|请你|帮我|可以)?(?:结束面试|停止面试|结束|不面了)"
        r"(?:[，,\s]*(?:吧|了|谢谢))?",
        normalized,
    ))


def resume_attachment_info(fields: dict[str, Any]) -> tuple[str, str, str]:
    """从附件本身取得 token、文件名和格式，不依赖已废弃的冗余格式字段。"""
    attach = fields.get("文件")
    attachment = (
        attach[0]
        if isinstance(attach, list) and attach and isinstance(attach[0], dict)
        else {}
    )
    token = str(attachment.get("file_token") or "")
    name = str(attachment.get("name") or field_text(fields.get("版本名")) or "resume.pdf")
    suffix = Path(name).suffix.lower()
    if suffix not in RESUME_FILE_EXTENSIONS:
        note = field_text(fields.get("备注"))
        inferred = re.search(r"\.(pdf|docx)\b", note, re.IGNORECASE)
        suffix = "." + inferred.group(1).lower() if inferred else ".pdf"
    return token, name, suffix


def find_board_records(
    records: list[Any], company: str, position: str = ""
) -> list[Any]:
    """按公司（大小写不敏感的包含匹配）+ 可选岗位筛选看板记录。"""
    company = company.strip().lower()
    position = position.strip().lower()
    out = []
    for rec in records:
        fields = record_fields(rec)
        if company and company not in field_text(fields.get("公司")).lower():
            continue
        if position and position not in field_text(fields.get("岗位")).lower():
            continue
        out.append(rec)
    return out


def jd_content_fingerprint(jd_text: str) -> str:
    """对 JD 做空白与大小写归一化，供幂等更新和材料过期判断。"""
    normalized = re.sub(r"\s+", "", jd_text or "").casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""


def make_job_id(company: str, position: str, now_ms: int | None = None) -> str:
    """生成短而稳定可读的求职机会 ID；同岗位重复 JD 由 upsert 复用旧 ID。"""
    stamp = time.strftime("%Y%m%d", time.localtime((now_ms or int(time.time() * 1000)) / 1000))
    seed = f"{company.strip().casefold()}|{position.strip().casefold()}|{time.time_ns()}"
    return f"JOB-{stamp}-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:8].upper()}"


def find_job_record(
    records: list[Any], *, job_id: str = "", company: str = "", position: str = "",
    jd_fingerprint: str = "",
) -> Any | None:
    """优先按求职ID/JD指纹精确找岗位；公司岗位仅在唯一时自动承接。"""
    if job_id:
        exact = [r for r in records if field_text(record_fields(r).get("求职ID")) == job_id]
        return exact[0] if len(exact) == 1 else None
    if jd_fingerprint:
        exact = [r for r in records if field_text(record_fields(r).get("JD指纹")) == jd_fingerprint]
        if len(exact) == 1:
            return exact[0]
    matches = find_board_records(records, company, position)
    active = [r for r in matches if field_text(record_fields(r).get("状态")) not in TERMINAL_STATUS]
    candidates = active or matches
    return candidates[0] if len(candidates) == 1 else None


@dataclass(frozen=True)
class ResumeMatch:
    record: Any
    score: int
    level: str
    evidence: tuple[str, ...]


def select_resume_for_jd(
    records: list[Any], position: str, jd_text: str, jd_fingerprint: str = "",
) -> ResumeMatch | None:
    """从证据化画像选择简历；语义无命中时返回 None，不以“最新”冒充匹配。"""
    target = _semantic_label(f"{position}\n{jd_text}")
    ranked: list[tuple[int, float, Any, tuple[str, ...]]] = []
    for rec in records:
        fields = record_fields(rec)
        version_type = field_text(fields.get("版本类型")) or "基础版"
        if version_type == "岗位定制版" and field_text(fields.get("JD指纹")) != jd_fingerprint:
            continue
        if version_type == "岗位定制版" and field_text(fields.get("用户确认状态")) != "已确认":
            continue
        semantic_score = 0
        evidence: list[str] = []
        for role in field_list(fields.get("目标岗位")):
            normalized = _semantic_label(role)
            if _semantic_match(normalized, target, minimum_overlap=0.55):
                semantic_score += 12
                evidence.append(f"目标岗位命中：{role}")
        for skill in field_list(fields.get("能力标签")):
            normalized = _semantic_label(skill)
            if len(normalized) >= 2 and _semantic_match(normalized, target, minimum_overlap=0.75):
                semantic_score += 4
                evidence.append(f"能力标签命中：{skill}")
        track = field_text(fields.get("主方向"))
        normalized_track = _semantic_label(track)
        if normalized_track and _semantic_match(normalized_track, target, minimum_overlap=0.75):
            semantic_score += 6
            evidence.append(f"主方向命中：{track}")
        if version_type == "岗位定制版" and jd_fingerprint:
            semantic_score += 20
            evidence.append("已有该 JD 的岗位定制版")
        if semantic_score <= 0:
            continue
        score = semantic_score
        if fields.get("当前版本") is True or fields.get("分支当前版本") is True:
            score += 2
        uploaded_score = field_ms(fields.get("上传时间"))
        ranked.append((score, uploaded_score, rec, tuple(evidence[:8])))
    if not ranked:
        return None
    score, _, record, evidence = max(ranked, key=lambda item: (item[0], item[1]))
    level = "高" if score >= 18 else "中" if score >= 10 else "低"
    return ResumeMatch(record=record, score=score, level=level, evidence=evidence)


_SEMANTIC_ALIASES = (
    (re.compile(r"人工智能", re.IGNORECASE), "ai"),
    (re.compile(r"智能体", re.IGNORECASE), "agent"),
    (re.compile(r"大语言模型|大型语言模型|大模型", re.IGNORECASE), "llm"),
    (re.compile(r"\bllms?\b", re.IGNORECASE), "llm"),
    (re.compile(r"产品经(?:理)?", re.IGNORECASE), "产品经理"),
)


def _semantic_label(value: str) -> str:
    """用于开放岗位名称匹配的轻量归一化，不把岗位限制在固定枚举。"""
    text = str(value or "").casefold()
    for pattern, replacement in _SEMANTIC_ALIASES:
        text = pattern.sub(replacement, text)
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def _semantic_units(value: str) -> set[str]:
    """英文按词、中文按双字片段形成开放词面特征。"""
    units = set(re.findall(r"[a-z][a-z0-9+#.-]*", value))
    for chunk in re.findall(r"[\u4e00-\u9fff]+", value):
        units.update(chunk[i:i + 2] for i in range(max(1, len(chunk) - 1)))
    return {unit for unit in units if unit}


def _semantic_match(label: str, target: str, minimum_overlap: float) -> bool:
    if not label or not target:
        return False
    if label in target:
        return True
    label_units = _semantic_units(label)
    if not label_units:
        return False
    return len(label_units & _semantic_units(target)) / len(label_units) >= minimum_overlap


# ---------------------------------------------------------------------------
# 纯函数：看板操作解析（自然语言 → 结构化操作，light 模型输出的后置处理）
# ---------------------------------------------------------------------------

@dataclass
class BoardOperation:
    """一次看板操作。fields 已校验（非法字段/枚举/日期进 dropped 并注明原因）。"""

    action: str  # add | update | delete | query
    company: str = ""
    position: str = ""
    fields: dict[str, Any] = field(default_factory=dict)
    dropped: list[str] = field(default_factory=list)


_ACTION_ALIASES = {
    "add": "add", "新增": "add", "添加": "add", "投递": "add", "记录": "add",
    "update": "update", "更新": "update", "修改": "update", "改": "update",
    "delete": "delete", "删除": "delete", "移除": "delete",
    "query": "query", "查询": "query", "查看": "query", "查": "query",
}


def parse_board_operation(raw: str, now: float | None = None) -> BoardOperation | None:
    """解析 light 模型的 JSON 输出为 BoardOperation；JSON 彻底无效返回 None。"""
    data = extract_json(raw)
    if data is None:
        return None
    action_raw = str(data.get("action", "")).strip().lower()
    action = _ACTION_ALIASES.get(action_raw, "query")  # 拿不准一律降级为只读查询
    op = BoardOperation(
        action=action,
        company=str(data.get("company") or "").strip(),
        position=str(data.get("position") or "").strip(),
    )
    fields_in = data.get("fields")
    if isinstance(fields_in, dict):
        for key, value in fields_in.items():
            if key in ("公司", "岗位") or key not in BOARD_FIELDS:
                op.dropped.append(f"{key}（不是看板字段）")
                continue
            if key in BOARD_SELECT_FIELDS:
                options = BOARD_SELECT_FIELDS[key]
                value = str(value).strip()
                if value in options:
                    op.fields[key] = value
                else:
                    op.dropped.append(f"{key}={value}（可选值：{'/'.join(options)}）")
            elif key in BOARD_DATE_FIELDS:
                ms = parse_date_text(str(value), now)
                if ms is None:
                    op.dropped.append(f"{key}={value}（日期无法解析）")
                else:
                    op.fields[key] = ms
            else:
                op.fields[key] = str(value).strip()
    return op


# ---------------------------------------------------------------------------
# 纯函数：看板汇总与跟进
# ---------------------------------------------------------------------------

@dataclass
class BoardSummary:
    total: int
    by_status: dict[str, int]
    followups: list[dict[str, str]]  # {"公司","岗位","原因"}


def summarize_board(records: list[Any], now: float | None = None) -> BoardSummary:
    """分状态计数 + 跟进清单（纯函数，records 为 SDK 记录或裸 dict 均可）。

    跟进规则（终态 已拒/已挂/offer 不再跟进）：
    - 「下次跟进」日期 ≤ 今天
    - 「状态」仍为已投递且「投递时间」距今超过 FOLLOWUP_STALE_DAYS 天
    """
    now_ms = int((now if now is not None else time.time()) * 1000)
    today = ms_to_date(now_ms)
    stale_ms = FOLLOWUP_STALE_DAYS * 86_400_000

    by_status: dict[str, int] = {}
    followups: list[dict[str, str]] = []
    for rec in records:
        fields = record_fields(rec)
        status = field_text(fields.get("状态")) or "待评估"
        by_status[status] = by_status.get(status, 0) + 1
        if status in TERMINAL_STATUS:
            continue
        company = field_text(fields.get("公司"))
        position = field_text(fields.get("岗位"))
        follow_ms = fields.get("下次跟进")
        if isinstance(follow_ms, (int, float)):
            follow_date = ms_to_date(follow_ms)
            if follow_date <= today:
                reason = "下次跟进今天到期" if follow_date == today else f"下次跟进 {follow_date[5:]} 已过期"
                followups.append({"公司": company, "岗位": position, "原因": reason})
                continue
        applied_ms = fields.get("投递时间")
        if status == "已投递" and isinstance(applied_ms, (int, float)):
            days = (now_ms - applied_ms) // 86_400_000
            if now_ms - applied_ms > stale_ms:
                followups.append({"公司": company, "岗位": position, "原因": f"投递已 {days} 天无进展"})
    return BoardSummary(total=len(records), by_status=by_status, followups=followups)


def format_board_record(fields: dict[str, Any]) -> str:
    """一条看板记录 → 单行摘要。"""
    parts = [field_text(fields.get("公司")), field_text(fields.get("岗位"))]
    head = " ".join(p for p in parts if p)
    details = []
    for key in ("岗位方向", "状态"):
        v = field_text(fields.get(key))
        if v:
            details.append(v)
    for key in ("投递时间", "下次跟进"):
        v = fields.get(key)
        if isinstance(v, (int, float)):
            details.append(f"{key} {ms_to_date(v)[5:]}")
    for key in ("内推人", "备注"):
        v = field_text(fields.get(key))
        if v:
            details.append(f"{key}：{v}")
    return f"{head}（{'，'.join(details)}）" if details else head


# ---------------------------------------------------------------------------
# 纯函数：面经结构化容错
# ---------------------------------------------------------------------------

_ROUND_ALIASES = {
    "初面": "一面", "首轮": "一面", "一面": "一面",
    "二面": "二面", "三面": "三面", "终面": "三面", "leader面": "三面", "主管面": "三面",
    "hr面": "HR面", "hr": "HR面", "HR": "HR面", "HR面": "HR面",
    "笔试": "笔试",
}


def _to_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def normalize_exp(data: dict[str, Any]) -> dict[str, Any]:
    """把面经 JSON（中文键或复盘英文键）规整为面经库字段 dict。

    输入兼容两种键风格：career_exp 模板的 公司/轮次/问题清单…，
    career_review 元数据的 company/round/questions…。
    轮次非法一律置空（单选字段写入非法值会被飞书拒绝）。
    """
    company = data.get("公司") if "公司" in data else data.get("company")
    position = data.get("岗位") if "岗位" in data else data.get("position")
    round_raw = data.get("轮次") if "轮次" in data else data.get("round")
    questions = data.get("问题清单") if "问题清单" in data else data.get("questions")
    weak = data.get("答砸点") if "答砸点" in data else data.get("weak_points")
    reaction = data.get("面试官反应") if "面试官反应" in data else data.get("interviewer_reaction")

    round_text = str(round_raw or "").strip()
    round_norm = _ROUND_ALIASES.get(round_text) or _ROUND_ALIASES.get(round_text.lower(), "")
    return {
        "公司": str(company or "").strip(),
        "岗位": str(position or "").strip(),
        "轮次": round_norm,
        "问题清单": _to_str_list(questions),
        "答砸点": _to_str_list(weak),
        "面试官反应": str(reaction or "").strip(),
    }


def parse_exp_json(raw: str) -> dict[str, Any] | None:
    """面经结构化输出的容错解析；JSON 彻底无效返回 None（调用方走兜底）。"""
    data = extract_json(raw)
    if data is None:
        return None
    return normalize_exp(data)


def build_exp_row(exp: dict[str, Any], source: str, now: float | None = None) -> dict[str, Any]:
    """规整后的面经 → 面经库 bitable 行。来源取值：文字 / 语音 / 录音复盘。"""
    now_ms = int((now if now is not None else time.time()) * 1000)
    row: dict[str, Any] = {
        "公司": exp["公司"] or "（未注明公司）",
        "岗位": exp["岗位"],
        "问题清单": "\n".join(exp["问题清单"]),
        "答砸点": "\n".join(exp["答砸点"]),
        "面试官反应": exp["面试官反应"],
        "日期": now_ms,
        "来源": source,
    }
    if exp["轮次"]:
        row["轮次"] = exp["轮次"]
    return row


# ---------------------------------------------------------------------------
# 纯函数：JD 识别 / prompt 组装 / 简历目录
# ---------------------------------------------------------------------------

_JD_KEYWORDS = (
    "岗位职责", "职位描述", "工作职责", "任职要求", "职位要求", "岗位要求", "加分项",
)


def looks_like_jd(text: str) -> bool:
    """长文本且含 ≥2 个 JD 标志词 → 判定为粘贴的 JD 全文。"""
    if len(text) < 150:
        return False
    hits = sum(1 for kw in _JD_KEYWORDS if kw in text)
    return hits >= 2


def _clip(text: str, max_chars: int) -> str:
    text = text or ""
    return text if len(text) <= max_chars else text[:max_chars] + "\n…（已截断）"


def render_prompt(template_path: Path, **kwargs: str) -> str:
    """读模板并替换 {{KEY}} 占位符（同 paper_reading.md 的占位风格）。"""
    template = template_path.read_text(encoding="utf-8")
    for key, value in kwargs.items():
        template = template.replace("{{" + key + "}}", value)
    return template


def resume_catalog_text(records: list[Any]) -> str:
    """简历库记录 → 带正文证据的画像目录（JD 匹配用）。"""
    if not records:
        return "（简历库为空）"
    lines = []
    for rec in records:
        fields = record_fields(rec)
        name = field_text(fields.get("版本名")) or "未命名"
        series = field_text(fields.get("版本系列")) or resume_series_name(name)
        version = resume_version_number(fields)
        version_type = field_text(fields.get("版本类型")) or "基础版"
        branch_version = _positive_int(fields.get("分支版本号"))
        current = (
            fields.get("分支当前版本") is True
            if version_type == "岗位定制版" else fields.get("当前版本") is True
        )
        direction = field_text(fields.get("岗位方向"))
        track = field_text(fields.get("主方向"))
        roles = field_list(fields.get("目标岗位"))
        skills = field_list(fields.get("能力标签"))
        summary = field_text(fields.get("简历摘要"))
        evidence = field_text(fields.get("识别依据"))
        uploaded_ms = field_ms(fields.get("上传时间"))
        date = ms_to_date(uploaded_ms) if uploaded_ms > 0 else "?"
        note = field_text(fields.get("备注"))
        version_label = (
            f"{series}｜岗位定制分支 v{branch_version}"
            if version_type == "岗位定制版" else f"{series} v{version}" if version else series
        )
        head = (
            f"- {name}｜{version_label}{'（当前）' if current else ''}｜"
            f"{track or direction or '未标注'}｜上传于 {date}"
        )
        details = []
        if roles:
            details.append("目标岗位：" + "、".join(roles[:6]))
        if skills:
            details.append("能力标签：" + "、".join(skills[:10]))
        if summary:
            details.append("摘要：" + summary[:400])
        if evidence:
            details.append("识别依据：" + evidence[:800])
        if not details and note:
            details.append("备注：" + note)
        lines.append(head + (("\n  " + "\n  ".join(details)) if details else ""))
    return "\n".join(lines)


@dataclass(frozen=True)
class ResumeRole:
    name: str
    confidence: float
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class ResumeProfile:
    primary_track: str
    target_roles: tuple[ResumeRole, ...]
    skill_tags: tuple[str, ...]
    summary: str
    needs_confirmation: bool = False

    @property
    def confidence(self) -> float:
        return max((role.confidence for role in self.target_roles), default=0.0)


def resume_content_fingerprint(text: str) -> str:
    """对规范化简历正文做稳定指纹，避免同一文件换名重复入库。"""
    normalized = re.sub(r"\s+", "", text or "").casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""


def resume_file_fingerprint(path: Path, resume_text: str = "") -> str:
    """优先按正文去重；正文抽取失败时退化为文件字节指纹。"""
    if text_fingerprint := resume_content_fingerprint(resume_text):
        return text_fingerprint
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "file:" + digest.hexdigest()


_RESUME_VERSION_SUFFIX_RE = re.compile(
    r"(?:[\s._·-]*(?:v(?:ersion)?|版本)\s*\d+(?:\.\d+)*)$", re.IGNORECASE,
)


def resume_series_name(file_name: str) -> str:
    """从文件名得到稳定的简历系列名；仅剥离明确版本后缀，不猜岗位。"""
    stem = Path(file_name or "未命名简历").stem.strip()
    series = _RESUME_VERSION_SUFFIX_RE.sub("", stem).strip(" ._·-")
    return series or stem or "未命名简历"


def resume_series_key(value: str) -> str:
    """系列匹配键：忽略大小写及常见分隔符，但保留实际展示名称。"""
    return re.sub(r"[\s._·-]+", "", resume_series_name(value)).casefold()


def resume_version_number(fields: dict[str, Any]) -> int:
    """读取版本号；旧记录没有专用字段时尝试从版本名末尾推断。"""
    raw = fields.get("版本号")
    if isinstance(raw, (int, float)) and raw > 0:
        return int(raw)
    raw_text = field_text(raw)
    try:
        if raw_text and float(raw_text) > 0:
            return int(float(raw_text))
    except ValueError:
        pass
    name = field_text(fields.get("版本名"))
    match = re.search(r"(?:v(?:ersion)?|版本)\s*(\d+)(?:\.\d+)*$", name, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def _positive_int(value: Any) -> int:
    """兼容飞书 number 字段返回 int/float/数字字符串。"""
    try:
        number = int(float(field_text(value)))
        return number if number > 0 else 0
    except (TypeError, ValueError):
        return 0


def resume_version_change_summary(
    previous_fields: dict[str, Any] | None,
    new_profile_fields: dict[str, Any],
    new_text: str,
) -> str:
    """本地生成可核对的版本变化，不为此额外调用模型。"""
    if not previous_fields:
        return "首次入库"
    parts: list[str] = []
    old_roles = set(field_list(previous_fields.get("目标岗位")))
    new_roles = set(field_list(new_profile_fields.get("目标岗位")))
    if added := sorted(new_roles - old_roles):
        parts.append("新增目标岗位：" + "、".join(added[:5]))
    if removed := sorted(old_roles - new_roles):
        parts.append("移除目标岗位：" + "、".join(removed[:5]))
    old_skills = set(field_list(previous_fields.get("能力标签")))
    new_skills = set(field_list(new_profile_fields.get("能力标签")))
    if added := sorted(new_skills - old_skills):
        parts.append("新增能力标签：" + "、".join(added[:8]))
    if removed := sorted(old_skills - new_skills):
        parts.append("移除能力标签：" + "、".join(removed[:8]))
    old_text = field_text(previous_fields.get("简历正文"))
    if old_text:
        delta = len(new_text) - len(old_text)
        if delta:
            parts.append(f"正文长度变化：{delta:+d} 字")
    elif new_text:
        parts.append(f"新增可检索正文：{len(new_text)} 字")
    return "；".join(parts) or "内容已变化，岗位画像与上一版一致"


def _evidence_is_in_resume(evidence: str, resume_text: str) -> bool:
    needle = re.sub(r"\s+", "", evidence).casefold()
    haystack = re.sub(r"\s+", "", resume_text).casefold()
    return len(needle) >= 3 and needle in haystack


def _clean_open_label(value: Any, max_chars: int = 40) -> str:
    """开放岗位/能力标签只做文本清洗，不用固定枚举压缩新角色。"""
    return re.sub(r"[\r\n|]+", " ", str(value or "")).strip()[:max_chars]


def parse_resume_profile(raw: str, resume_text: str) -> ResumeProfile | None:
    """验收模型简历画像：岗位开放，但每个岗位必须有原文证据。"""
    data = extract_json(raw)
    if not data:
        return None
    roles: list[ResumeRole] = []
    for item in data.get("target_roles") or []:
        if not isinstance(item, dict):
            continue
        name = _clean_open_label(item.get("name"))
        try:
            confidence = float(item.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence > 1:
            confidence /= 100
        evidence = tuple(
            e for e in (_clean_open_label(v, 160) for v in (item.get("evidence") or []))
            if e and _evidence_is_in_resume(e, resume_text)
        )[:3]
        if name and evidence:
            roles.append(ResumeRole(name, min(1.0, max(0.0, confidence)), evidence))
    if not roles:
        return None
    roles.sort(key=lambda role: role.confidence, reverse=True)
    skills: list[str] = []
    for value in data.get("skill_tags") or []:
        label = _clean_open_label(value, 30)
        # 能力标签同样必须能在正文中定位，避免一个幻觉标签污染后续 JD 匹配。
        if label and _evidence_is_in_resume(label, resume_text) and label not in skills:
            skills.append(label)
    return ResumeProfile(
        primary_track=_clean_open_label(data.get("primary_track"), 30) or roles[0].name,
        target_roles=tuple(roles[:8]),
        skill_tags=tuple(skills[:20]),
        summary=_clean_open_label(data.get("summary"), 500),
        needs_confirmation=bool(data.get("needs_confirmation")),
    )


def resume_profile_fields(profile: ResumeProfile | None, resume_text: str,
                          fingerprint: str) -> dict[str, Any]:
    """把验收后的画像转成简历库字段；无画像时明确标记待确认。"""
    if profile is None:
        return {
            "主方向": "待确认", "目标岗位": ["待确认"],
            "能力标签": [], "简历摘要": "", "识别依据": "",
            "解析状态": "画像待确认" if resume_text else "正文解析失败",
            "内容指纹": fingerprint,
            "简历正文": _clip(resume_text, MAX_RESUME_STORE_CHARS),
        }
    evidence_lines = [
        f"{role.name}（{role.confidence:.0%}）：" + "；".join(role.evidence)
        for role in profile.target_roles
    ]
    return {
        "主方向": profile.primary_track,
        "目标岗位": [role.name for role in profile.target_roles],
        "能力标签": list(profile.skill_tags),
        "简历摘要": profile.summary,
        "识别依据": "\n".join(evidence_lines),
        "解析状态": "需用户确认" if profile.needs_confirmation else "已解析",
        "内容指纹": fingerprint,
        "简历正文": _clip(resume_text, MAX_RESUME_STORE_CHARS),
    }


def build_predict_prompt(
    company: str, position: str, jd: str, resume: str, exps: list[str]
) -> str:
    """押题 prompt 组装（测试断言三料俱全的接缝）。"""
    exp_text = "\n\n".join(f"【面经 {i}】\n{e}" for i, e in enumerate(exps, 1)) or "（库内无相关面经）"
    return render_prompt(
        PREDICT_PROMPT_PATH,
        COMPANY=company or "（未指定）",
        POSITION=position or "（未指定）",
        JD=_clip(jd, MAX_JD_CHARS) or "（看板无 JD 原文，按岗位名与面经押题）",
        RESUME=_clip(resume, MAX_RESUME_CHARS) or "（无可用简历正文；只能生成明确标注为通用岗位题的内容）",
        EXP_LIST=_clip(exp_text, MAX_EXP_CHARS),
    )


def build_interviewer_prompt(company: str, position: str, context: str) -> str:
    return render_prompt(
        INTERVIEWER_PROMPT_PATH,
        COMPANY=company or "目标公司",
        POSITION=position or "目标岗位",
        CONTEXT=context or "（无看板/简历/面经材料，按岗位通用情况面试）",
    )


def exp_record_text(rec: Any) -> str:
    """一条面经记录 → 可塞进 prompt / 回话的文本块。"""
    fields = record_fields(rec)
    head = " ".join(
        p for p in (field_text(fields.get("公司")), field_text(fields.get("岗位")),
                    field_text(fields.get("轮次"))) if p
    )
    date = fields.get("日期")
    if isinstance(date, (int, float)):
        head += f"（{ms_to_date(date)[5:]}）"
    body_parts = []
    questions = field_text(fields.get("问题清单"))
    if questions:
        body_parts.append(f"问题：{questions}")
    weak = field_text(fields.get("答砸点"))
    if weak:
        body_parts.append(f"答砸点：{weak}")
    reaction = field_text(fields.get("面试官反应"))
    if reaction:
        body_parts.append(f"面试官反应：{reaction}")
    return head + "\n" + "\n".join(body_parts)


# ---------------------------------------------------------------------------
# 内联小 prompt（light 模型结构化解析，不走 prompts/ 目录）
# ---------------------------------------------------------------------------

_BOARD_OP_PROMPT = """\
你是求职看板的操作解析器。今天是 __TODAY__。把用户的一句自然语言解析成对「求职看板」表的结构化操作，只输出 JSON。

表字段：公司、岗位、岗位方向（算法/研究/工程/其他）、JD原文、状态（已投递/笔试/一面/二面/三面/HR面/offer/已拒/已挂）、投递时间、下次跟进、内推人、备注。

输出格式（只输出 JSON，不要任何其他文字）：
{"action": "add|update|delete|query", "company": "公司名", "position": "岗位名（没提填空字符串）", "fields": {"字段名": "值"}}

规则：
- add=新增投递记录；update=更新已有记录（如"字节改一面过了"）；delete=删除；query=查询。
- 状态表示当前所处环节。"X面过了/通过"→推进到下一环节（如"一面过了"→"二面"）；"挂了/被拒"→"已挂"；"拿到offer"→"offer"。
- 日期一律输出 YYYY-MM-DD（今天是 __TODAY__），"明天""3天后"等换算成具体日期。
- fields 只放要写入/更新的字段（不要放公司、岗位，它们走 company/position）；用户没提的字段不要输出。

用户的话：__TEXT__"""


# ---------------------------------------------------------------------------
# 服务
# ---------------------------------------------------------------------------

class CareerService:
    """求职模块总入口。router 命令 handler 与 bot 文件/语音入口都落在这里。"""

    def __init__(
        self,
        settings: Settings,
        kimi: KimiClient,
        lark_client: Any = None,
        sender: Sender | None = None,
        session_store: SessionStore | None = None,
        io: CareerIO | None = None,
        docx: Any = None,
        drive: Any = None,
        transcriber: Any = None,
        resume_text_extractor: Callable[[Path], str] | None = None,
        resume_dir: Path | None = None,
        tailored_resume_dir: Path | None = None,
        audio_dir: Path | None = None,
        outbox: NotificationOutbox | None = None,
    ) -> None:
        self.settings = settings
        self.kimi = kimi
        self.sender = sender
        self.session_store = session_store
        self.lark_client = lark_client
        # 事件通知外发箱（任务失败必达）：None = 失败通知只靠直发
        self.outbox = outbox
        self.bitable_dao: BitableDAO | None = (
            BitableDAO(lark_client, settings.bitable_app_token, settings.bitable_tables)
            if lark_client is not None and settings.bitable_ready()
            else None
        )
        self.io: CareerIO | None = io if io is not None else (
            CareerIO(self.bitable_dao) if self.bitable_dao is not None else None
        )
        self.docx = docx if docx is not None else (
            DocxClient(lark_client) if lark_client is not None else None
        )
        self.drive = drive if drive is not None else (
            DriveClient(lark_client) if lark_client is not None else None
        )
        self.transcriber = transcriber  # asr.Transcriber；None → mlx-whisper 惰性导入
        self.resume_text_extractor = resume_text_extractor or extract_resume_text
        self.resume_dir = resume_dir or RESUME_DIR
        self.tailored_resume_dir = tailored_resume_dir or TAILORED_RESUME_DIR
        self.audio_dir = audio_dir or AUDIO_DIR
        self._resume_fields_ensured = False
        self._resume_fields_lock = threading.Lock()
        self._operation_locks_guard = threading.Lock()
        self._operation_locks: dict[tuple[str, str], threading.RLock] = {}
        # 模拟面试状态：session_id → 渲染好的面试官 system prompt；已结束集合
        self._interview_prompts: dict[int, str] = {}
        self._ended_sessions: set[int] = set()
        # 复盘回执：session_id → (对话指纹, 复盘回复)。状态落库失败后的重试
        # 凭指纹复用首份回执，只重放 complete_interview_end，不重复写文档与面经。
        self._review_receipts: dict[int, tuple[str, str]] = {}

    # ------------------------------------------------------------------
    # 通用：IO 检查 / 进度推送 / 异步管线
    # ------------------------------------------------------------------

    def _require_io(self) -> CareerIO | None:
        return self.io

    @contextmanager
    def _operation_lock(self, scope: str, key: str):
        """单进程内按业务键串行化读-算-写链；多实例部署需替换为分布式锁。"""
        lock_key = (scope, key.casefold())
        with self._operation_locks_guard:
            lock = self._operation_locks.setdefault(lock_key, threading.RLock())
        with lock:
            yield

    def _ensure_resume_fields(self) -> None:
        """给既有简历库幂等补画像字段；无真实飞书客户端时跳过。"""
        if self._resume_fields_ensured or self.bitable_dao is None or self.lark_client is None:
            return
        with self._resume_fields_lock:
            if self._resume_fields_ensured:
                return
            table_id = self.bitable_dao.tables.get(RESUME_TABLE, "")
            if not table_id:
                return
            try:
                created = ensure_fields(
                    self.lark_client, self.bitable_dao.app_token, table_id,
                    list(RESUME_PROFILE_FIELDS),
                )
            except BitableError as exc:
                logger.warning("简历库画像字段补建失败（后续操作将重试）：%s", exc)
                raise
            self._resume_fields_ensured = True
            if created:
                logger.info("简历库补充画像字段：%s", "、".join(created))

    def _classify_resume_profile(
        self, file_name: str, resume_text: str
    ) -> ResumeProfile | None:
        """生成并验收简历画像；首次证据不合格时自动纠正一次。"""
        prompt = render_prompt(
            RESUME_PROMPT_PATH,
            FILE_NAME=file_name,
            RESUME_TEXT=_clip(resume_text, MAX_RESUME_STORE_CHARS),
        )
        try:
            raw = self.kimi.chat(
                [{"role": "user", "content": prompt}], heavy=True, json_mode=True,
            )
            profile = parse_resume_profile(raw, resume_text)
            if profile is not None:
                return profile
            logger.warning("简历画像未通过原文证据验收：%s", file_name)
            retry_prompt = (
                prompt
                + "\n\n## 本次必须修正\n"
                + "上一次结果没有为任何岗位提供能在简历正文中逐字找到的证据。"
                "请重新输出完整 JSON；每条 evidence 必须直接复制简历原文短句，"
                "没有证据的岗位不要输出。"
            )
            retry_raw = self.kimi.chat(
                [{"role": "user", "content": retry_prompt}],
                heavy=True, json_mode=True,
            )
            return parse_resume_profile(retry_raw, resume_text)
        except KimiError as exc:
            logger.warning("简历画像识别失败（待确认入库）：%s", exc)
            return None

    def _repair_resume_version_chains(self, records: list[Any]) -> int:
        """幂等补齐所有简历系列的版本号、前驱和当前版本标记。"""
        if self.io is None:
            return 0
        groups: dict[str, list[Any]] = {}
        for rec in records:
            fields = record_fields(rec)
            if field_text(fields.get("版本类型")) == "岗位定制版":
                continue
            name = field_text(fields.get("版本系列")) or field_text(fields.get("版本名"))
            groups.setdefault(resume_series_key(name), []).append(rec)

        updated = 0
        for grouped in groups.values():
            grouped.sort(key=lambda rec: (
                field_ms(record_fields(rec).get("上传时间")),
                record_id_of(rec),
            ))
            series = field_text(record_fields(grouped[0]).get("版本系列")) or resume_series_name(
                field_text(record_fields(grouped[0]).get("版本名"))
            )
            previous: Any | None = None
            for index, rec in enumerate(grouped, 1):
                fields = record_fields(rec)
                previous_ref = ""
                if previous is not None:
                    previous_fields = record_fields(previous)
                    previous_name = field_text(previous_fields.get("版本名")) or series
                    previous_ref = f"{previous_name} · v{index - 1} · {record_id_of(previous)}"
                desired = {
                    "版本系列": series,
                    "版本号": index,
                    "上一版本": previous_ref,
                    "当前版本": index == len(grouped),
                }
                changes: dict[str, Any] = {}
                if field_text(fields.get("版本系列")) != series:
                    changes["版本系列"] = series
                if resume_version_number(fields) != index:
                    changes["版本号"] = index
                if field_text(fields.get("上一版本")) != previous_ref:
                    changes["上一版本"] = previous_ref
                if fields.get("当前版本") is not (index == len(grouped)):
                    changes["当前版本"] = index == len(grouped)
                if not field_text(fields.get("版本变化")):
                    changes["版本变化"] = "历史版本链初始化" if len(grouped) > 1 else "首次入库"
                if changes and record_id_of(rec):
                    try:
                        self.io.update(RESUME_TABLE, record_id_of(rec), changes)
                    except BitableError as exc:
                        logger.warning(
                            "简历版本链单条更新失败，继续处理其他记录（%s）：%s",
                            record_id_of(rec), exc,
                        )
                    else:
                        fields.update(changes)
                        updated += 1
                previous = rec
        return updated

    def _hydrate_legacy_resume_identities(self, records: list[Any]) -> int:
        """为旧记录本地补正文指纹，使新旧数据遵守同一去重规则；不调用模型。"""
        if self.io is None:
            return 0
        updated = 0
        for rec in records:
            fields = record_fields(rec)
            if field_text(fields.get("内容指纹")):
                continue
            record_id = record_id_of(rec)
            text = field_text(fields.get("简历正文"))
            fingerprint = resume_content_fingerprint(text)
            if not fingerprint:
                token, _, suffix = resume_attachment_info(fields)
                if not token:
                    continue
                with sensitive_workspace(self.resume_dir, f"identity-{record_id}") as workspace:
                    dest = workspace / f"resume{suffix}"
                    try:
                        self.io.download_media(token, dest)
                        text = self.resume_text_extractor(dest)
                        fingerprint = resume_file_fingerprint(dest, text)
                    except (DriveError, OSError, ValueError) as exc:
                        logger.warning("旧简历身份指纹补全失败：%s", exc)
                        continue
            changes: dict[str, Any] = {"内容指纹": fingerprint}
            if text and not field_text(fields.get("简历正文")):
                changes["简历正文"] = _clip(text, MAX_RESUME_STORE_CHARS)
            try:
                self.io.update(RESUME_TABLE, record_id, changes)
            except BitableError as exc:
                logger.warning("旧简历身份指纹单条更新失败，继续入库（%s）：%s", record_id, exc)
                # 当前任务仍使用已算出的本地指纹做去重，避免因回写失败重复归档。
                fields.update(changes)
                continue
            fields.update(changes)
            updated += 1
        return updated

    def _progress(self, context: MessageContext, text: str) -> None:
        """阶段行进状态条（10 分钟流式窗口内自我更新，不每条进度推新消息）；
        无状态条时回退文字推送（测试/无 sender 路径）。"""
        logger.info("进度：%s", text)
        status = getattr(context, "status", None)
        if status is not None:
            status.set_stage(text)
            return
        if self.sender is None:
            return
        try:
            self.sender.send_text(context.chat_id, text)
        except Exception:
            logger.exception("进度推送失败")

    def _start_async(self, context: MessageContext, label: str, job: Callable, **kwargs) -> str:
        """启动后台求职任务；返回状态条 message_id（"" = 状态条没发出去，调用方回退文字 ack）。

        状态条即 ack：与精读同一套 StatusMessenger，短任务全程在一条卡里自更新。
        """
        status: StatusMessenger | None = None
        if self.sender is not None and context.chat_id:
            status = StatusMessenger(self.sender, context.chat_id, label).start()
            context.status = status  # 按消息绑定：并发任务阶段行各进各的状态条
        thread = threading.Thread(
            target=self._run_guarded, args=(context, label, job, kwargs, status),
            name=f"career-{int(time.time())}", daemon=True,
        )
        thread.start()
        logger.info("求职任务已启动：%s（chat=%s）", label, context.chat_id)
        return status._msg_id if status is not None else ""

    def _enqueue_notify(self, chat_id: str, text: str) -> None:
        """事件通知落外发箱（flusher 必达补发）；无 outbox 或写库失败只记日志。"""
        if self.outbox is None:
            return
        try:
            self.outbox.enqueue(chat_id, text)
        except Exception:
            logger.exception("事件通知入队失败（不影响任务）：%s", text[:60])

    def _run_guarded(self, context: MessageContext, label: str, job: Callable, kwargs: dict,
                     status: "StatusMessenger | None" = None) -> None:
        """job 返回最终回复文本；异常统一翻译成中文原因推给用户。
        失败回复若直发未送达（断网），落外发箱由 flusher 必达补发——
        成功回复维持直发（时效优先），失败事件必须让用户知道。"""
        failed = False
        try:
            reply = job(context=context, **kwargs)
        except Exception as exc:
            failed = True
            logger.exception("求职任务失败：%s", label)
            reply = f"❌ {label}失败：{_classify_error(exc)}"
            if status is not None:
                status.stop(ok=False, note="失败，原因见下方")
        else:
            if status is not None:
                status.stop(ok=True, note="完成，结果见下方")
        if self.sender is not None:
            delivered = self.sender.send_text(context.chat_id, reply)
            if failed and not delivered:
                # 直发丢了：同一文案入外发箱（同文 10 分钟去重，不会重复打扰）
                self._enqueue_notify(context.chat_id, reply)
        else:
            logger.warning("无 sender，任务结果仅记录：%s", reply[:100])

    # ------------------------------------------------------------------
    # 看板：汇总 / 自然语言增删改查
    # ------------------------------------------------------------------

    def cmd_board(self, arg: str, ctx: MessageContext | None = None) -> str:
        """「看板」命令：无参→汇总；有参→light 解析成操作后执行。"""
        if self._require_io() is None:
            return DAO_MISSING
        arg = (arg or "").strip()
        if not arg:
            return self.board_summary_text()
        prompt = _BOARD_OP_PROMPT.replace("__TODAY__", ms_to_date(int(time.time() * 1000))).replace("__TEXT__", arg)
        try:
            raw = self.kimi.chat([{"role": "user", "content": prompt}], heavy=False)
        except KimiError as exc:
            return f"调用大模型出错了：{exc}"
        op = parse_board_operation(raw)
        if op is None:
            return (
                "没听懂这句看板操作。可以这样说：\n"
                "「看板 新增 字节 后端岗」「看板 字节改一面过了」\n"
                "「看板 字节下次跟进明天」「看板 查字节」「看板 删除字节」"
            )
        return self.execute_board_operation(op)

    def board_summary_text(self, now: float | None = None) -> str:
        """「进展如何」：分状态汇总 + 该跟进清单。"""
        io = self._require_io()
        if io is None:
            return DAO_MISSING
        try:
            records = io.list_all(BOARD_TABLE)
        except BitableError as exc:
            return f"读取求职看板失败：{exc}"
        summary = summarize_board(records, now)
        if summary.total == 0:
            return "看板还是空的。发「看板 新增 字节 后端岗」记一笔投递吧。"
        counts = " · ".join(
            f"{s} {summary.by_status[s]}" for s in JOB_STATUS if s in summary.by_status
        )
        extra = " · ".join(
            f"{s} {n}" for s, n in summary.by_status.items() if s not in JOB_STATUS
        )
        lines = [f"📊 求职看板（共 {summary.total} 条）", counts + (f" · {extra}" if extra else "")]
        if summary.followups:
            lines.append(f"\n⏰ 该跟进了（{len(summary.followups)}）")
            for i, f in enumerate(summary.followups, 1):
                target = " ".join(p for p in (f["公司"], f["岗位"]) if p)
                lines.append(f"{i}. {target}：{f['原因']}")
        else:
            lines.append("\n✅ 暂无需要跟进的项。")
        return "\n".join(lines)

    def execute_board_operation(self, op: BoardOperation) -> str:
        """执行解析好的看板操作，返回给用户的中文确认。"""
        io = self._require_io()
        if io is None:
            return DAO_MISSING
        handler = {
            "add": self._op_add, "update": self._op_update,
            "delete": self._op_delete, "query": self._op_query,
        }[op.action]
        try:
            reply = handler(io, op)
        except BitableError as exc:
            return f"看板操作失败：{exc}"
        if op.dropped:
            reply += "\n⚠️ 部分内容未写入：" + "；".join(op.dropped)
        return reply

    def _op_add(self, io: CareerIO, op: BoardOperation) -> str:
        if not op.company:
            return "新增投递至少要说公司名，如「看板 新增 字节 后端岗」。"
        row: dict[str, Any] = {
            "公司": op.company,
            "求职ID": make_job_id(op.company, op.position),
        }
        if op.position:
            row["岗位"] = op.position
        row.setdefault("状态", "已投递")
        row.setdefault("投递时间", int(time.time() * 1000))
        row.update(op.fields)  # fields 优先级高于默认值
        io.dao.add_record(BOARD_TABLE, row)
        logger.info("看板新增：%s %s", op.company, op.position)
        return f"✅ 已添加：{format_board_record(row)}"

    def _find_or_complain(self, io: CareerIO, op: BoardOperation) -> tuple[list[Any], str | None]:
        """按 company/position 找记录；找不到/有多条时给出中文提示。"""
        if not op.company:
            return [], "要先说公司名，如「看板 字节改一面过了」。"
        records = io.list_all(BOARD_TABLE)
        matches = find_board_records(records, op.company, op.position)
        if not matches and op.position:  # 岗位没匹配上时退化为只按公司找，给用户更准的提示
            matches = find_board_records(records, op.company)
        if not matches:
            return [], f"看板里没有「{op.company}」的记录。要新增的话发「看板 新增 {op.company} 岗位名」。"
        if len(matches) > 1:
            lines = [f"「{op.company}」有 {len(matches)} 条记录，我没法确定改哪条："]
            for i, rec in enumerate(matches, 1):
                lines.append(f"{i}. {format_board_record(record_fields(rec))}")
            lines.append("请带上岗位名再说一次，如「看板 字节 后端 改二面」。")
            return [], "\n".join(lines)
        return matches, None

    def _op_update(self, io: CareerIO, op: BoardOperation) -> str:
        if not op.fields:
            return "没说改什么。如「看板 字节改一面过了」「看板 字节下次跟进明天」。"
        matches, error = self._find_or_complain(io, op)
        if error:
            return error
        rec = matches[0]
        current = record_fields(rec)
        changes = dict(op.fields)
        status = field_text(changes.get("状态"))
        lock_note = ""
        if status in JOB_STATUS and status != "":
            if status == "已投递" and not isinstance(current.get("投递时间"), (int, float)):
                changes.setdefault("投递时间", int(time.time() * 1000))
            if not field_text(current.get("实际投递简历ID")):
                recommended_id = field_text(current.get("推荐简历ID"))
                recommended_label = field_text(current.get("推荐简历"))
                if recommended_id:
                    changes.update({
                        "实际投递简历ID": recommended_id,
                        "实际投递简历": recommended_label,
                        "投递版本锁定": True,
                    })
                    try:
                        io.update(RESUME_TABLE, recommended_id, {"是否实际投递": True})
                    except BitableError as exc:
                        logger.warning("简历实际投递标记失败：%s", exc)
                    lock_note = f"\n🔒 已锁定实际投递简历：{recommended_label}"
                else:
                    lock_note = "\n⚠️ 该岗位没有证据充分的推荐简历，尚未锁定投递版本。"
        io.update(BOARD_TABLE, record_id_of(rec), changes)
        merged = {**current, **changes}
        logger.info("看板更新：%s ← %s", record_id_of(rec), "、".join(changes))
        return f"✅ 已更新：{format_board_record(merged)}{lock_note}"

    def _op_delete(self, io: CareerIO, op: BoardOperation) -> str:
        matches, error = self._find_or_complain(io, op)
        if error:
            return error
        rec = matches[0]
        io.delete(BOARD_TABLE, record_id_of(rec))
        logger.info("看板删除：%s", record_id_of(rec))
        return f"🗑 已删除：{format_board_record(record_fields(rec))}"

    def _op_query(self, io: CareerIO, op: BoardOperation) -> str:
        records = io.list_all(BOARD_TABLE)
        matches = find_board_records(records, op.company, op.position) if op.company else records
        if not matches:
            who = f"「{op.company}」" if op.company else "看板"
            return f"{who}没有找到记录。"
        lines = [f"🔎 查到 {len(matches)} 条记录："]
        for i, rec in enumerate(matches[:10], 1):
            lines.append(f"{i}. {format_board_record(record_fields(rec))}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # JD 分析（异步 heavy）
    # ------------------------------------------------------------------

    def analyze_jd(self, jd_text: str, ctx: MessageContext) -> str:
        """JD 全文入口：立即 ack，后台跑拆解。"""
        msg_id = self._start_async(ctx, "JD 分析", self._job_jd, jd_text=jd_text)
        if msg_id:
            return ""
        return "📋 收到 JD，开始拆解（硬性要求/加分项/潜台词/简历匹配），约 1~2 分钟，完成后发你文档。"

    def _job_jd(self, context: MessageContext, jd_text: str) -> str:
        resumes_text = "（简历库不可用）"
        if self.io is not None:
            try:
                resumes_text = resume_catalog_text(self.io.list_all(RESUME_TABLE))
            except BitableError as exc:
                logger.warning("读取简历库失败（按无简历分析）：%s", exc)
        prompt = render_prompt(
            JD_PROMPT_PATH, JD_TEXT=_clip(jd_text, MAX_JD_CHARS), RESUME_LIST=resumes_text
        )
        self._progress(context, "🤖 JD 拆解生成中（heavy 模型，约 1~2 分钟）……")
        output = self._heavy_chat([{"role": "user", "content": prompt}], context)
        body, meta = extract_metadata(output)

        company = str(meta.get("company") or "").strip()
        position = str(meta.get("position") or "").strip()
        title = " ".join(p for p in (company, position) if p) or "未识别岗位"
        doc_url = self._write_doc(context, f"JD分析：{title}", body)

        job_id = ""
        board_action = ""
        bound_resume = ""
        if self.io is not None and company and position:
            try:
                job_id, board_action, bound_resume = self._upsert_job_from_jd(
                    jd_text, meta, doc_url,
                )
            except BitableError as exc:
                logger.error("JD 自动承接到看板失败：%s", exc)

        lines = [f"✅ JD 拆解完成：{title}"]
        if meta.get("match_level"):
            lines.append(f"简历匹配度：{meta['match_level']}")
        raw_highlights = meta.get("highlights")
        highlights = (
            [str(h).strip() for h in raw_highlights if str(h).strip()]
            if isinstance(raw_highlights, list)
            else field_list(raw_highlights)
        )[:3]
        lines += [f"{i}. {h}" for i, h in enumerate(highlights, 1)]
        if meta.get("recommend_resume") and not bound_resume:
            lines.append(f"💡 推荐简历：{meta['recommend_resume']}")
        if bound_resume:
            lines.append(f"🔗 看板绑定简历：{bound_resume}")
        if job_id:
            lines.append(f"📌 已{board_action}求职看板：{job_id}")
        if doc_url:
            lines.append(f"📄 完整分析：{doc_url}")
        return "\n".join(lines)

    def _upsert_job_from_jd(
        self, jd_text: str, meta: dict[str, Any], doc_url: str,
    ) -> tuple[str, str, str]:
        """把 JD 分析结果幂等写入看板，并绑定证据最充分的简历版本。"""
        if self.io is None:
            return "", "", ""
        company = str(meta.get("company") or "").strip()
        position = str(meta.get("position") or "").strip()
        if not company or not position:
            return "", "", ""
        fingerprint = jd_content_fingerprint(jd_text)
        board_records = self.io.list_all(BOARD_TABLE)
        existing = find_job_record(
            board_records, company=company, position=position, jd_fingerprint=fingerprint,
        )
        existing_fields = record_fields(existing) if existing is not None else {}
        job_id = field_text(existing_fields.get("求职ID")) or make_job_id(company, position)

        resume_records = self.io.list_all(RESUME_TABLE)
        match = select_resume_for_jd(resume_records, position, jd_text, fingerprint)
        now_ms = int(time.time() * 1000)
        row: dict[str, Any] = {
            "公司": company,
            "岗位": position,
            "求职ID": job_id,
            "JD原文": _clip(jd_text, MAX_JD_CHARS),
            "JD指纹": fingerprint,
            "材料更新时间": now_ms,
        }
        if doc_url:
            row["JD分析文档"] = {"text": f"JD分析：{company} {position}", "link": doc_url}
        track = str(meta.get("primary_track") or "").strip().lower()
        if re.search(r"研究|research|scientist", track):
            row["岗位方向"] = "研究"
        elif re.search(r"算法|algorithm|模型训练|模型优化", track):
            row["岗位方向"] = "算法"
        elif re.search(r"工程|开发|engineer|agent|平台|后端", track):
            row["岗位方向"] = "工程"
        else:
            row["岗位方向"] = "其他"

        bound_resume = ""
        if match is not None:
            resume_fields = record_fields(match.record)
            resume_name = field_text(resume_fields.get("版本名")) or "未命名简历"
            version = resume_version_number(resume_fields)
            bound_resume = f"{resume_name}{f' · v{version}' if version else ''}"
            row.update({
                "推荐简历ID": record_id_of(match.record),
                "推荐简历": bound_resume,
                "简历匹配结论": f"{match.level}（证据评分 {match.score}）",
                "匹配证据": "\n".join(match.evidence),
            })
        else:
            row.update({
                "推荐简历ID": "",
                "推荐简历": "",
                "简历匹配结论": "证据不足",
                "匹配证据": "简历画像中的目标岗位、主方向和能力标签均未命中该 JD。",
            })

        if existing is None:
            self.io.dao.add_record(BOARD_TABLE, row)
            action = "创建"
        else:
            # 推荐版本可随材料更新；实际投递版本和锁定状态绝不自动覆盖。
            self.io.update(BOARD_TABLE, record_id_of(existing), row)
            action = "更新"
        logger.info("JD 已%s看板：%s %s（%s）", action, company, position, job_id)
        return job_id, action, bound_resume

    def _refresh_job_resume_recommendations(self) -> int:
        """新简历入库后重算所有活跃 JD 的推荐版本，不改已锁定的实际投递版本。"""
        if self.io is None:
            return 0
        resumes = self.io.list_all(RESUME_TABLE)
        updated = 0
        for job in self.io.list_all(BOARD_TABLE):
            fields = record_fields(job)
            if field_text(fields.get("状态")) in TERMINAL_STATUS:
                continue
            jd_text = field_text(fields.get("JD原文"))
            position = field_text(fields.get("岗位"))
            if not jd_text or not position:
                continue
            fingerprint = field_text(fields.get("JD指纹")) or jd_content_fingerprint(jd_text)
            match = select_resume_for_jd(resumes, position, jd_text, fingerprint)
            changes: dict[str, Any]
            if match is None:
                changes = {
                    "推荐简历ID": "", "推荐简历": "", "简历匹配结论": "证据不足",
                    "匹配证据": "简历画像中的目标岗位、主方向和能力标签均未命中该 JD。",
                }
            else:
                resume_fields = record_fields(match.record)
                name = field_text(resume_fields.get("版本名")) or "未命名简历"
                version = resume_version_number(resume_fields)
                label = f"{name}{f' · v{version}' if version else ''}"
                changes = {
                    "推荐简历ID": record_id_of(match.record),
                    "推荐简历": label,
                    "简历匹配结论": f"{match.level}（证据评分 {match.score}）",
                    "匹配证据": "\n".join(match.evidence),
                }
            if all(
                field_text(fields.get(key)) == str(value)
                for key, value in changes.items()
            ):
                continue
            changes["材料更新时间"] = int(time.time() * 1000)
            self.io.update(BOARD_TABLE, record_id_of(job), changes)
            updated += 1
        return updated

    def _repair_career_links(self) -> dict[str, int]:
        """幂等迁移旧看板、简历和面经的中心 ID；歧义记录不自动关联。"""
        if self.io is None:
            return {"boards": 0, "resumes": 0, "experiences": 0}
        counts = {"boards": 0, "resumes": 0, "experiences": 0}
        boards = self.io.list_all(BOARD_TABLE)
        for rec in boards:
            fields = record_fields(rec)
            changes: dict[str, Any] = {}
            company = field_text(fields.get("公司"))
            position = field_text(fields.get("岗位"))
            if not field_text(fields.get("求职ID")):
                changes["求职ID"] = make_job_id(company, position)
            jd_text = field_text(fields.get("JD原文"))
            if jd_text and not field_text(fields.get("JD指纹")):
                changes["JD指纹"] = jd_content_fingerprint(jd_text)
            if changes:
                self.io.update(BOARD_TABLE, record_id_of(rec), changes)
                fields.update(changes)
                counts["boards"] += 1

        resumes = self.io.list_all(RESUME_TABLE)
        for rec in resumes:
            fields = record_fields(rec)
            if field_text(fields.get("版本类型")):
                continue
            self.io.update(RESUME_TABLE, record_id_of(rec), {"版本类型": "基础版"})
            fields["版本类型"] = "基础版"
            counts["resumes"] += 1

        for rec in self.io.list_all(EXP_TABLE):
            fields = record_fields(rec)
            if field_text(fields.get("求职ID")):
                continue
            job = find_job_record(
                boards,
                company=field_text(fields.get("公司")),
                position=field_text(fields.get("岗位")),
            )
            if job is None:
                continue
            job_fields = record_fields(job)
            changes = {
                "求职ID": field_text(job_fields.get("求职ID")),
                "关联简历ID": (
                    field_text(job_fields.get("实际投递简历ID"))
                    or field_text(job_fields.get("推荐简历ID"))
                ),
                "关联简历": (
                    field_text(job_fields.get("实际投递简历"))
                    or field_text(job_fields.get("推荐简历"))
                ),
            }
            self.io.update(EXP_TABLE, record_id_of(rec), changes)
            counts["experiences"] += 1
        return counts

    # ------------------------------------------------------------------
    # 简历库：上传（异步）/ 查询（同步）
    # ------------------------------------------------------------------

    def handle_resume_file(self, file_name: str, file_key: str, context: MessageContext) -> str:
        """简历附件入口（bot 文件消息路由过来）。"""
        if self._require_io() is None:
            return DAO_MISSING
        if not file_name.lower().endswith(RESUME_FILE_EXTENSIONS):
            return "简历支持 PDF / docx 格式哦～把简历文件直接发给我就行。"
        msg_id = self._start_async(
            context, f"简历《{file_name}》入库", self._job_resume,
            file_name=file_name, file_key=file_key, message_id=context.message_id,
        )
        if msg_id:  # 状态条即 ack：进度与结果都进同一条卡
            return ""
        return f"📄 收到简历《{file_name}》，开始入库（下载 → 上传简历库），约半分钟。"

    def _job_resume(self, context: MessageContext, file_name: str, file_key: str, message_id: str) -> str:
        with sensitive_workspace(self.resume_dir, "resume-upload") as workspace:
            return self._job_resume_in_workspace(
                context, file_name, file_key, message_id, workspace,
            )

    def _job_resume_in_workspace(
        self, context: MessageContext, file_name: str, file_key: str,
        message_id: str, workspace: Path,
    ) -> str:
        series = resume_series_name(file_name)
        with self._operation_lock("resume-series", resume_series_key(series)):
            return self._job_resume_in_workspace_locked(
                context, file_name, file_key, message_id, workspace,
            )

    def _job_resume_in_workspace_locked(
        self, context: MessageContext, file_name: str, file_key: str,
        message_id: str, workspace: Path,
    ) -> str:
        if self.drive is None:
            raise CareerError("飞书客户端未配置，无法下载附件。")
        self._ensure_resume_fields()
        dest = workspace / Path(file_name).name
        self.drive.download_message_resource(message_id, file_key, dest)
        resume_text = self.resume_text_extractor(dest)
        fingerprint = resume_file_fingerprint(dest, resume_text)
        records = self.io.list_all(RESUME_TABLE) if self.io is not None else []
        self._hydrate_legacy_resume_identities(records)
        if fingerprint and self.io is not None:
            for rec in records:
                fields = record_fields(rec)
                if field_text(fields.get("内容指纹")) == fingerprint:
                    existing = field_text(fields.get("版本名")) or "未命名简历"
                    return f"♻️ 这份简历与已归档的「{existing}」内容相同，未重复入库。"

        self._repair_resume_version_chains(records)
        name = Path(file_name).stem
        series = resume_series_name(file_name)
        same_series = [
            rec for rec in records
            if field_text(record_fields(rec).get("版本类型")) != "岗位定制版"
            if resume_series_key(
                field_text(record_fields(rec).get("版本系列"))
                or field_text(record_fields(rec).get("版本名"))
            ) == resume_series_key(series)
        ]
        same_series.sort(key=lambda rec: (
            resume_version_number(record_fields(rec)),
            field_ms(record_fields(rec).get("上传时间")),
        ))
        previous = same_series[-1] if same_series else None
        version = len(same_series) + 1

        profile: ResumeProfile | None = None
        if resume_text:
            self._progress(context, "🧠 正文已解析，正在根据项目与经历证据生成多标签岗位画像……")
            profile = self._classify_resume_profile(file_name, resume_text)

        profile_fields = resume_profile_fields(profile, resume_text, fingerprint)
        previous_fields = record_fields(previous) if previous is not None else None
        version_change = resume_version_change_summary(previous_fields, profile_fields, resume_text)
        self._progress(context, "⬆️ 画像生成完成，上传到简历库……")
        token = self.io.upload_attachment(dest)
        now_ms = int(time.time() * 1000)
        row = {
            "版本名": name,
            "版本系列": series,
            "版本号": version,
            "上一版本": (
                f"{field_text(previous_fields.get('版本名')) or series} · "
                f"v{resume_version_number(previous_fields) or version - 1} · {record_id_of(previous)}"
                if previous is not None and previous_fields is not None else ""
            ),
            "当前版本": True,
            "版本变化": version_change,
            "版本类型": "基础版",
            "事实校验状态": "已通过原文证据校验" if profile is not None else "待确认",
            "用户确认状态": "待确认",
            "是否实际投递": False,
            "文件": [{"file_token": token}],
            "上传时间": now_ms,
            **profile_fields,
        }
        new_record_id = self.io.dao.add_record(RESUME_TABLE, row)
        if previous is not None:
            try:
                self.io.update(RESUME_TABLE, record_id_of(previous), {"当前版本": False})
            except Exception:
                try:
                    self.io.delete(RESUME_TABLE, new_record_id)
                except Exception:
                    logger.exception(
                        "简历版本切换失败且补偿删除失败；下次入库将执行版本链修复：%s",
                        new_record_id,
                    )
                raise
        refreshed_jobs = self._refresh_job_resume_recommendations()
        logger.info("简历已入库：%s（%s）token=%s", name, row["解析状态"], token)
        if profile is None:
            reason = "无法从文件抽取正文" if not resume_text else "岗位判断缺少可核对的原文证据"
            return (
                f"⚠️ 简历已归档：{name}，但画像标记为「{row['解析状态']}」。\n"
                f"原因：{reason}；系统不会根据文件名猜岗位。"
            )
        role_text = "、".join(
            f"{role.name} {role.confidence:.0%}" for role in profile.target_roles[:4]
        )
        return (
            f"✅ 简历已归档：{name}\n"
            f"版本：{series} v{version}（当前版本）\n"
            f"变化：{version_change}\n"
            f"主方向：{profile.primary_track}\n目标岗位：{role_text}\n"
            f"能力标签：{'、'.join(profile.skill_tags[:8]) or '未提取'}\n"
            "岗位判断已连同简历原文证据入库，JD 分析和押题会使用这份画像。"
            + (f"\n已自动更新 {refreshed_jobs} 个活跃岗位的推荐简历。" if refreshed_jobs else "")
        )

    def cmd_resume(self, arg: str, ctx: MessageContext | None = None) -> str:
        """「简历」命令：无参列全部；任意岗位/能力/版本关键词均可检索。"""
        io = self._require_io()
        if io is None:
            return DAO_MISSING
        arg = (arg or "").strip()
        if arg in ("更新画像", "补全画像", "更新简历库", "重新识别"):
            if ctx is None:
                return "更新简历画像需要消息上下文。"
            msg_id = self._start_async(ctx, "旧简历画像回填", self._job_resume_backfill)
            if msg_id:
                return ""
            return "🔄 开始更新旧简历画像：逐份解析附件并回写原记录，完成后发你结果。"
        try:
            records = io.list_all(RESUME_TABLE)
        except BitableError as exc:
            return f"读取简历库失败：{exc}"
        if not records:
            return "简历库还是空的。把简历 PDF/docx 文件直接发给我，我就会入库。"

        matches = records
        query = "" if arg in ("", "列表", "全部") else arg.casefold()
        if query:
            def searchable(rec: Any) -> str:
                fields = record_fields(rec)
                values = [field_text(fields.get(key)) for key in (
                    "版本名", "版本系列", "版本变化", "岗位方向", "主方向", "简历摘要", "备注",
                )]
                values += field_list(fields.get("目标岗位"))
                values += field_list(fields.get("能力标签"))
                return " ".join(values).casefold()
            matches = [rec for rec in records if query in searchable(rec)]
            if not matches:
                return f"简历库里没有匹配「{arg}」的版本。\n" + resume_catalog_text(records)

        def _uploaded(rec: Any) -> float:
            return field_ms(record_fields(rec).get("上传时间"))

        matches = sorted(matches, key=_uploaded, reverse=True)
        lines = [f"🗂 简历库（{('匹配「' + arg + '」，') if query else ''}共 {len(matches)} 版）："]
        for i, rec in enumerate(matches[:8], 1):
            fields = record_fields(rec)
            name = field_text(fields.get("版本名")) or "未命名"
            series = field_text(fields.get("版本系列")) or resume_series_name(name)
            version = resume_version_number(fields)
            version_type = field_text(fields.get("版本类型")) or "基础版"
            branch_version = _positive_int(fields.get("分支版本号"))
            roles = field_list(fields.get("目标岗位"))
            d = "、".join(roles[:3]) or field_text(fields.get("主方向")) or field_text(fields.get("岗位方向"))
            date = ms_to_date(_uploaded(rec)) if _uploaded(rec) else "?"
            current = (
                fields.get("分支当前版本") is True
                if version_type == "岗位定制版" else fields.get("当前版本") is True
            )
            mark = " 👈当前" if current else ""
            version_label = (
                f"{series}｜定制分支v{branch_version}"
                if version_type == "岗位定制版" else f"{series} v{version}" if version else series
            )
            lines.append(f"{i}. {name}｜{version_label}｜{d or '未标注'}｜{date}{mark}")
        lines.append("文件在「简历库」多维表格里可直接下载；JD 分析/押题会自动引用最新版。")
        return "\n".join(lines)

    def _job_resume_backfill(
        self, context: MessageContext | None = None, force: bool = False
    ) -> str:
        """给旧简历回填正文与证据化画像；默认跳过已经完成的记录。"""
        io = self._require_io()
        if io is None:
            return DAO_MISSING
        self._ensure_resume_fields()
        records = io.list_all(RESUME_TABLE)
        chain_updates = self._repair_resume_version_chains(records)
        pending = []
        for rec in records:
            fields = record_fields(rec)
            complete = (
                bool(field_text(fields.get("简历正文")))
                and bool(field_list(fields.get("目标岗位")))
                and field_text(fields.get("解析状态")) in ("已解析", "需用户确认")
            )
            if force or not complete:
                pending.append(rec)
        if not pending:
            chain_text = f"；已补齐 {chain_updates} 条版本链" if chain_updates else ""
            return f"✅ 简历库共 {len(records)} 条，画像都已是最新，无需回填{chain_text}。"

        success = pending_confirmation = failed = 0
        for index, rec in enumerate(pending, 1):
            label = f"未命名简历-{index}"
            try:
                fields = record_fields(rec)
                record_id = record_id_of(rec)
                label = field_text(fields.get("版本名")) or label
                stored_text = field_text(fields.get("简历正文"))
                token, original_name, suffix = resume_attachment_info(fields)

                self._progress(context, f"🔄 更新简历画像 {index}/{len(pending)}：{label}")
                resume_text = stored_text
                if not resume_text and token:
                    with sensitive_workspace(self.resume_dir, f"backfill-{record_id}") as workspace:
                        dest = workspace / f"resume{suffix}"
                        io.download_media(token, dest)
                        resume_text = self.resume_text_extractor(dest)

                if not resume_text:
                    io.update(
                        RESUME_TABLE,
                        record_id,
                        {"解析状态": "附件缺失" if not token else "正文解析失败"},
                    )
                    failed += 1
                    continue

                profile = self._classify_resume_profile(original_name, resume_text)
                profile_fields = resume_profile_fields(
                    profile,
                    resume_text,
                    resume_content_fingerprint(resume_text),
                )
                io.update(RESUME_TABLE, record_id, profile_fields)
                if profile is None:
                    pending_confirmation += 1
                else:
                    success += 1
            except Exception as exc:
                # 回填是可恢复批任务：单条附件、模型或表格异常必须隔离并留日志。
                logger.exception("旧简历「%s」回填失败，继续处理下一条：%s", label, exc)
                failed += 1

        lines = [
            f"✅ 旧简历画像更新完成：处理 {len(pending)}/{len(records)} 条",
            f"已解析 {success} 条 · 待确认 {pending_confirmation} 条 · 解析失败 {failed} 条",
        ]
        if chain_updates:
            lines.append(f"版本链已补齐 {chain_updates} 条记录。")
        if pending_confirmation or failed:
            lines.append(
                "未成功记录已计入结果；能写回的记录会标明解析状态，且不会根据文件名猜岗位；"
                "可修正附件后再次发「简历 更新画像」。"
            )
        else:
            lines.append("目标岗位、能力标签、摘要、识别依据和正文均已回写原记录。")
        return "\n".join(lines)

    def cmd_tailor_resume(self, arg: str, ctx: MessageContext | None = None) -> str:
        """「定制简历 求职ID/公司岗位」：生成待确认的原格式岗位分支。"""
        if self._require_io() is None:
            return DAO_MISSING
        target = (arg or "").strip()
        if not target:
            return "用法：定制简历 求职ID，或“定制简历 公司 岗位”。"
        if ctx is None:
            return "定制简历需要消息上下文。"
        job, choices = self._resolve_job_target(target)
        if choices:
            lines = ["同一公司有多个岗位，请指定要定制哪一个："]
            lines += [f"{i}. {choice}" for i, choice in enumerate(choices, 1)]
            return "\n".join(lines)
        if job is None:
            return "求职看板里没有找到这个岗位。请先粘贴 JD，让系统建立求职记录。"
        job_id = field_text(record_fields(job).get("求职ID"))
        msg_id = self._start_async(
            ctx, f"定制简历·{job_id}", self._job_tailor_resume, job_id=job_id,
        )
        if msg_id:
            return ""
        return "🧩 开始按原简历格式生成岗位定制草稿；完成后需要你确认，系统不会直接锁定投递。"

    def _download_resume_attachment(
        self, rec: Any, prefix: str, destination_dir: Path | None = None,
    ) -> Path:
        fields = record_fields(rec)
        token, _, suffix = resume_attachment_info(fields)
        if not token:
            raise CareerError("基础简历没有可下载的原始附件。")
        safe_prefix = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", prefix)[:100]
        dest = (destination_dir or self.tailored_resume_dir) / f"{safe_prefix}_source{suffix}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.io.download_media(token, dest)
        return dest

    def _job_tailor_resume(self, context: MessageContext, job_id: str) -> str:
        """JD + 已推荐基础简历 → 原 DOCX 定点替换 → 渲染验收 → 草稿入库。"""
        with sensitive_workspace(self.tailored_resume_dir, f"tailor-{job_id}") as workspace:
            return self._job_tailor_resume_in_workspace(context, job_id, workspace)

    def _job_tailor_resume_in_workspace(
        self, context: MessageContext, job_id: str, workspace: Path,
    ) -> str:
        with self._operation_lock("tailored-resume", job_id):
            return self._job_tailor_resume_in_workspace_locked(context, job_id, workspace)

    def _job_tailor_resume_in_workspace_locked(
        self, context: MessageContext, job_id: str, workspace: Path,
    ) -> str:
        jobs = self.io.list_all(BOARD_TABLE)
        job = find_job_record(jobs, job_id=job_id)
        if job is None:
            return f"找不到求职记录 {job_id}，可能已被删除。"
        job_fields = record_fields(job)
        jd_text = field_text(job_fields.get("JD原文"))
        if not jd_text:
            return "这个岗位没有 JD 原文，无法生成有证据的定制简历。"
        resumes = self.io.list_all(RESUME_TABLE)
        base_id = (
            field_text(job_fields.get("实际投递简历ID"))
            or field_text(job_fields.get("推荐简历ID"))
        )
        base = next((rec for rec in resumes if record_id_of(rec) == base_id), None)
        if base is None:
            match = select_resume_for_jd(
                resumes,
                field_text(job_fields.get("岗位")),
                jd_text,
                field_text(job_fields.get("JD指纹")),
            )
            base = match.record if match is not None else None
        if base is None:
            return "没有找到证据充分的基础简历，不能为了生成定制版而硬选最新简历。"
        base_fields = record_fields(base)
        if field_text(base_fields.get("版本类型")) == "岗位定制版":
            parent_id = field_text(base_fields.get("基于版本ID"))
            base = next((rec for rec in resumes if record_id_of(rec) == parent_id), base)
            base_fields = record_fields(base)

        self._progress(context, "📥 下载基础简历并提取原格式……")
        source_path = self._download_resume_attachment(base, job_id, workspace)
        if source_path.suffix.lower() != ".docx":
            return (
                "原简历只有 PDF，无法保证原格式可编辑继承。请上传对应 DOCX 后重新执行；"
                "系统不会用统一模板冒充原格式。"
            )
        resume_text = extract_resume_text(source_path)
        if not resume_text:
            return "基础 DOCX 无法提取正文，已停止生成，避免产生不可核对内容。"

        prompt = render_prompt(
            RESUME_TAILOR_PROMPT_PATH,
            RESUME_TEXT=_clip(resume_text, MAX_RESUME_STORE_CHARS),
            JD_TEXT=_clip(jd_text, MAX_JD_CHARS),
        )
        self._progress(context, "🧠 生成有原文证据的最小替换计划……")
        raw = self.kimi.chat(
            [{"role": "user", "content": prompt}], heavy=True, json_mode=True,
        )
        plan = parse_tailor_plan(raw, resume_text)
        if plan is None:
            return "模型没有生成通过事实验收的修改项，本次未创建简历草稿。"

        company = field_text(job_fields.get("公司"))
        position = field_text(job_fields.get("岗位"))
        base_name = field_text(base_fields.get("版本名")) or "基础简历"
        branch_records = [
            rec for rec in resumes
            if field_text(record_fields(rec).get("版本类型")) == "岗位定制版"
            and field_text(record_fields(rec).get("目标求职ID")) == job_id
        ]
        branch_version = max(
            (_positive_int(record_fields(rec).get("分支版本号")) for rec in branch_records),
            default=0,
        ) + 1
        safe_name = re.sub(
            r"[^\w\u4e00-\u9fff.-]+", "_",
            f"{base_name}_{company}_{position}_定制_v{branch_version}",
        )[:140]
        output_docx = workspace / f"{safe_name}.docx"
        self._progress(context, "🧱 在原 DOCX 中定点替换，保留样式、图片和页眉页脚……")
        try:
            patch_docx_text(source_path, output_docx, plan)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            return f"原格式定点替换失败：{exc}。未生成降级模板。"

        qa_dir = workspace / f"qa_{job_id}_{branch_version}"
        self._progress(context, "🔎 渲染并检查页数与非正文结构……")
        qa = verify_tailored_docx(source_path, output_docx, qa_dir)
        if not qa.passed:
            return (
                f"定制稿未通过格式门禁：{qa.status}。本次没有上传，也没有写入简历库；"
                "原简历和现有版本链不受影响。"
            )
        output_pdf: Path | None = None
        if qa.output_pdf is not None:
            output_pdf = workspace / f"{safe_name}.pdf"
            shutil.copy2(qa.output_pdf, output_pdf)
        optimized_text = extract_resume_text(output_docx)
        if not optimized_text:
            return "生成后的 DOCX 无法提取正文，草稿未入库。"

        optimized_fingerprint = resume_content_fingerprint(optimized_text)
        jd_fingerprint = field_text(job_fields.get("JD指纹")) or jd_content_fingerprint(jd_text)
        existing_draft = next((
            rec for rec in branch_records
            if field_text(record_fields(rec).get("基于版本ID")) == record_id_of(base)
            and field_text(record_fields(rec).get("JD指纹")) == jd_fingerprint
            and field_text(record_fields(rec).get("内容指纹")) == optimized_fingerprint
        ), None)
        if existing_draft is not None:
            existing_fields = record_fields(existing_draft)
            existing_id = record_id_of(existing_draft)
            existing_name = field_text(existing_fields.get("版本名")) or safe_name
            self.io.update(BOARD_TABLE, record_id_of(job), {
                "定制简历草稿ID": existing_id,
                "定制简历草稿": f"{existing_name} · 分支v{_positive_int(existing_fields.get('分支版本号'))}",
                "定制简历状态": (
                    "已确认" if field_text(existing_fields.get("用户确认状态")) == "已确认"
                    else "待用户确认"
                ),
                "材料更新时间": int(time.time() * 1000),
            })
            return f"♻️ 相同 JD 和基础版本已经生成过同内容定制稿，已复用：{existing_name}。"

        tokens = [self.io.upload_attachment(output_docx)]
        if output_pdf is not None:
            tokens.append(self.io.upload_attachment(output_pdf))
        previous_branch = max(
            branch_records,
            key=lambda rec: _positive_int(record_fields(rec).get("分支版本号")),
            default=None,
        )
        parent = previous_branch or base
        parent_fields = record_fields(parent)
        parent_version = (
            _positive_int(parent_fields.get("分支版本号"))
            if previous_branch is not None else resume_version_number(parent_fields)
        )
        comparison = "\n\n".join(
            f"原：{change.source}\n改：{change.replacement}\nJD：{change.jd_requirement}\n证据：{'；'.join(change.evidence)}"
            for change in plan.replacements
        )
        questions = [*(f"待补充：{q}" for q in plan.questions)]
        questions += [*(f"未覆盖：{q}" for q in plan.unmet_requirements)]
        row = {
            "版本名": safe_name,
            "版本系列": field_text(base_fields.get("版本系列")) or resume_series_name(base_name),
            "版本号": resume_version_number(base_fields),
            "上一版本": f"{field_text(parent_fields.get('版本名'))} · v{parent_version} · {record_id_of(parent)}",
            "当前版本": False,
            "版本变化": plan.summary,
            "版本类型": "岗位定制版",
            "基于版本ID": record_id_of(base),
            "分支版本号": branch_version,
            "分支当前版本": True,
            "目标求职ID": job_id,
            "目标公司": company,
            "定制目标岗位": position,
            "JD指纹": jd_fingerprint,
            "事实校验状态": "已通过自动原文证据校验",
            "用户确认状态": "待确认",
            "格式继承状态": "原DOCX正文定点替换；非正文包结构保持",
            "版面校验状态": qa.status,
            "修改对照": _clip(comparison, MAX_RESUME_STORE_CHARS),
            "待确认项": "\n".join(questions),
            "是否实际投递": False,
            "主方向": field_text(base_fields.get("主方向")),
            "目标岗位": field_list(base_fields.get("目标岗位")),
            "能力标签": field_list(base_fields.get("能力标签")),
            "简历摘要": field_text(base_fields.get("简历摘要")),
            "识别依据": field_text(base_fields.get("识别依据")),
            "解析状态": "已解析",
            "内容指纹": optimized_fingerprint,
            "简历正文": _clip(optimized_text, MAX_RESUME_STORE_CHARS),
            "文件": [{"file_token": token} for token in tokens],
            "上传时间": int(time.time() * 1000),
        }
        draft_id = self.io.dao.add_record(RESUME_TABLE, row)
        if previous_branch is not None:
            try:
                self.io.update(
                    RESUME_TABLE, record_id_of(previous_branch), {"分支当前版本": False},
                )
            except Exception:
                try:
                    self.io.delete(RESUME_TABLE, draft_id)
                except Exception:
                    logger.exception(
                        "定制分支切换失败且补偿删除失败；需执行分支版本对账：%s", draft_id,
                    )
                raise
        self.io.update(BOARD_TABLE, record_id_of(job), {
            "定制简历草稿ID": draft_id,
            "定制简历草稿": f"{safe_name} · 分支v{branch_version}",
            "定制简历状态": "待用户确认",
            "材料更新时间": int(time.time() * 1000),
        })
        lines = [
            f"✅ 岗位定制简历草稿已生成：{safe_name}",
            f"基于：{base_name}｜求职ID：{job_id}",
            f"修改 {len(plan.replacements)} 处｜事实校验：通过",
            f"格式/版面：{qa.status}",
            "文件已归档到简历库，确认内容后发“确认定制简历 " + job_id + "”。",
        ]
        if questions:
            lines.append(f"⚠️ 仍有 {len(questions)} 项待确认，详见简历库“待确认项”。")
        return "\n".join(lines)

    def cmd_confirm_tailored_resume(
        self, arg: str, ctx: MessageContext | None = None,
    ) -> str:
        """确认草稿后提升为岗位推荐版本；已锁定的实际投递版本保持不动。"""
        if self._require_io() is None:
            return DAO_MISSING
        target = (arg or "").strip()
        if not target:
            return "用法：确认定制简历 求职ID。"
        job, choices = self._resolve_job_target(target)
        if choices:
            return "存在多个岗位，请使用求职ID确认。"
        if job is None:
            return "没有找到对应求职记录。"
        job_fields = record_fields(job)
        draft_id = field_text(job_fields.get("定制简历草稿ID"))
        if not draft_id:
            return "这个岗位没有待确认的定制简历草稿。"
        draft = next(
            (rec for rec in self.io.list_all(RESUME_TABLE) if record_id_of(rec) == draft_id),
            None,
        )
        if draft is None:
            return "定制简历草稿记录不存在，可能已被删除。"
        draft_fields = record_fields(draft)
        label = field_text(job_fields.get("定制简历草稿")) or field_text(draft_fields.get("版本名"))
        if field_text(draft_fields.get("用户确认状态")) == "已确认":
            return f"ℹ️ 这份定制简历已经确认过：{label}。"
        if not field_text(draft_fields.get("事实校验状态")).startswith("已通过"):
            return "这份草稿没有通过事实校验，不能提升为推荐简历。"
        if not field_text(draft_fields.get("版面校验状态")).startswith("通过："):
            return "这份草稿没有通过原格式版面校验，不能提升为推荐简历。"
        self.io.update(RESUME_TABLE, draft_id, {"用户确认状态": "已确认"})
        self.io.update(BOARD_TABLE, record_id_of(job), {
            "推荐简历ID": draft_id,
            "推荐简历": label,
            "定制简历状态": "已确认",
            "材料更新时间": int(time.time() * 1000),
        })
        if job_fields.get("投递版本锁定") is True:
            return f"✅ 已确认定制简历：{label}\n原实际投递版本已锁定，本次没有覆盖它。"
        return f"✅ 已确认定制简历：{label}\n已设为该岗位推荐版本；状态进入投递时才会锁定为实际投递版本。"

    # ------------------------------------------------------------------
    # 面经库：文字/语音入库（异步）/ 检索（同步）
    # ------------------------------------------------------------------

    def cmd_save_exp(self, arg: str, ctx: MessageContext | None = None) -> str:
        """「记面经」命令：文字面经 → heavy 结构化 → 面经库。"""
        if self._require_io() is None:
            return DAO_MISSING
        text = (arg or "").strip()
        if not text:
            return "用法：记面经 <面经全文>，如「记面经 字节一面，问了 KV cache、投机采样……」"
        if ctx is None:
            return "记面经需要消息上下文。"
        msg_id = self._start_async(ctx, "面经入库", self._job_exp, text=text, source="文字")
        if msg_id:
            return ""
        return "📝 收到面经，开始结构化（公司/轮次/问题/答砸点），约 1 分钟。"

    def _job_exp(self, context: MessageContext, text: str, source: str) -> str:
        prompt = render_prompt(EXP_PROMPT_PATH, EXP_TEXT=_clip(text, MAX_EXP_CHARS))
        fallback = False
        try:
            raw = self.kimi.chat([{"role": "user", "content": prompt}], heavy=True)
        except KimiError as exc:
            logger.warning("面经结构化模型失败（走兜底）：%s", exc)
            raw, fallback = "", True
        exp = parse_exp_json(raw) if raw else None
        if exp is None:
            fallback = True
            exp = {"公司": "", "岗位": "", "轮次": "",
                   "问题清单": [text[:500]], "答砸点": [], "面试官反应": ""}
        row = build_exp_row(exp, source)
        self._associate_exp_row(row, exp)
        self.io.dao.add_record(EXP_TABLE, row)
        logger.info("面经已入库：%s %s（来源 %s）", row["公司"], exp["轮次"] or "?", source)

        head = f"✅ 面经已入库：{row['公司']}" + (f" {exp['轮次']}" if exp["轮次"] else "")
        lines = [head, f"问题 {len(exp['问题清单'])} 条" +
                 (f"，答砸点 {len(exp['答砸点'])} 条" if exp["答砸点"] else "")]
        if fallback:
            lines.append("⚠️ 结构化不完整，已按原文存档；可在面经库表里手动补充公司/轮次。")
        return "\n".join(lines)

    def cmd_search_exp(self, arg: str, ctx: MessageContext | None = None) -> str:
        """「面经」命令：按公司/轮次检索面经库。"""
        io = self._require_io()
        if io is None:
            return DAO_MISSING
        arg = (arg or "").strip()
        round_kw = ""
        m = re.search(r"(笔试|一面|二面|三面|HR面|hr面|HR)", arg)
        if m:
            round_kw = _ROUND_ALIASES.get(m.group(1), m.group(1))
            company = (arg[: m.start()] + arg[m.end():]).strip(" 的的")
        else:
            company = arg
        try:
            records = io.list_all(EXP_TABLE)
        except BitableError as exc:
            return f"读取面经库失败：{exc}"
        matches = []
        for rec in records:
            fields = record_fields(rec)
            if company and company.lower() not in field_text(fields.get("公司")).lower():
                continue
            if round_kw and field_text(fields.get("轮次")) != round_kw:
                continue
            matches.append(rec)
        if not matches:
            who = f"「{arg}」" if arg else "面经库"
            return f"{who}没有找到面经。发「记面经 <内容>」记一条，或发面试录音自动复盘入库。"
        lines = [f"📒 找到 {len(matches)} 条面经："]
        for rec in matches[:5]:
            lines.append("———")
            lines.append(exp_record_text(rec))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 录音复盘 / 语音面经（异步：下载 → ASR → heavy）
    # ------------------------------------------------------------------

    def handle_audio_file(
        self, file_name: str, file_key: str, context: MessageContext, mode: str = "review"
    ) -> str:
        """音频入口。mode=review 录音复盘（出文档+入面经库）；mode=exp 语音面经（仅入库）。"""
        if self._require_io() is None:
            return DAO_MISSING
        label = "录音复盘" if mode == "review" else "语音面经"
        msg_id = self._start_async(
            context, label, self._job_audio,
            file_name=file_name or "audio.opus", file_key=file_key,
            message_id=context.message_id, mode=mode,
        )
        if msg_id:
            return ""
        if mode == "review":
            return "🎙 收到录音，开始复盘：转写约 1~3 分钟，复盘报告 2~5 分钟，过程会持续汇报。"
        return "🎙 收到语音面经，开始转写与结构化，约 1~3 分钟。"

    def _job_audio(self, context: MessageContext, file_name: str, file_key: str,
                   message_id: str, mode: str) -> str:
        with sensitive_workspace(self.audio_dir, "audio") as workspace:
            return self._job_audio_in_workspace(
                context, file_name, file_key, message_id, mode, workspace,
            )

    def _job_audio_in_workspace(
        self, context: MessageContext, file_name: str, file_key: str,
        message_id: str, mode: str, workspace: Path,
    ) -> str:
        if self.drive is None:
            raise CareerError("飞书客户端未配置，无法下载音频。")
        dest = workspace / Path(file_name).name
        self.drive.download_message_resource(message_id, file_key, dest)
        size_mb = dest.stat().st_size / 1024 / 1024
        self._progress(context, f"✅ 音频已下载（{size_mb:.1f} MB），开始转写（mlx-whisper）……")

        heartbeat = Heartbeat(lambda s: self._progress(context, s), interval=60)
        heartbeat.start()
        try:
            result = transcribe(dest, transcriber=self.transcriber)
        finally:
            heartbeat.stop()
        self._progress(
            context,
            f"✅ 转写完成（{len(result.text)} 字，约 {result.duration / 60:.0f} 分钟音频），"
            + ("生成复盘报告……" if mode == "review" else "结构化面经……"),
        )

        if mode == "exp":
            return self._job_exp(context, result.text, source="语音")
        return self._finish_review(context, result.text, hint=f"来源录音文件：{file_name}")

    def _associate_exp_row(
        self, row: dict[str, Any], exp: dict[str, Any], job_ref: str = "",
    ) -> None:
        """仅在求职机会唯一时把面经关联过去；有歧义则保持未关联。

        尽力而为的富化：看板读取失败只记日志，绝不在文档已落地后炸掉
        复盘/面经管线（否则重试会重复写文档）。
        """
        if self.io is None:
            return
        try:
            jobs = self.io.list_all(BOARD_TABLE)
        except BitableError as exc:
            logger.warning("读取看板失败，面经保持未关联：%s", exc)
            return
        job = find_job_record(
            jobs,
            job_id=job_ref if job_ref.startswith("JOB-") else "",
            company=str(exp.get("公司") or ""),
            position=str(exp.get("岗位") or ""),
        )
        if job is None:
            return
        fields = record_fields(job)
        resume_id = (
            field_text(fields.get("实际投递简历ID"))
            or field_text(fields.get("推荐简历ID"))
        )
        resume_label = (
            field_text(fields.get("实际投递简历"))
            or field_text(fields.get("推荐简历"))
        )
        row.update({
            "求职ID": field_text(fields.get("求职ID")),
            "关联简历ID": resume_id,
            "关联简历": resume_label,
        })

    def _finish_review(
        self, context: MessageContext, transcript: str, hint: str, job_ref: str = "",
    ) -> str:
        """转写文本 → heavy 复盘报告 → 飞书文档 + 面经库。录音复盘与模拟面试共用。"""
        prompt = render_prompt(
            REVIEW_PROMPT_PATH,
            TRANSCRIPT=_clip(transcript, MAX_TRANSCRIPT_CHARS),
            CONTEXT_HINT=hint or "（无）",
        )
        output = self._heavy_chat([{"role": "user", "content": prompt}], context)
        body, meta = extract_metadata(output)
        exp = normalize_exp(meta)

        company = exp["公司"]
        title = " ".join(p for p in (company, exp["轮次"]) if p) or "未注明场次"
        doc_url = self._write_doc(context, f"面试复盘：{title} {ms_to_date(int(time.time()*1000))}", body)

        row = build_exp_row(exp, "录音复盘")
        self._associate_exp_row(row, exp, job_ref)
        if self.io is not None:
            try:
                self.io.dao.add_record(EXP_TABLE, row)
            except BitableError as exc:
                logger.error("面经库写入失败（不影响复盘文档）：%s", exc)

        lines = [f"✅ 复盘完成：{title}"]
        for i, w in enumerate(exp["答砸点"][:3], 1):
            lines.append(f"{i}. {w}")
        if doc_url:
            lines.append(f"📄 复盘报告：{doc_url}")
        lines.append("📒 问题清单已同步入面经库。")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 押题（异步 heavy）
    # ------------------------------------------------------------------

    def cmd_predict(self, arg: str, ctx: MessageContext | None = None) -> str:
        """「押题」命令：押题 [公司]。"""
        if self._require_io() is None:
            return DAO_MISSING
        target = (arg or "").strip()
        if not target:
            return "用法：押题 <公司名>，如「押题 字节」。会综合看板 JD、简历库与面经库出题。"
        if ctx is None:
            return "押题需要消息上下文。"
        msg_id = self._start_async(ctx, f"押题·{target}", self._job_predict, target=target)
        if msg_id:
            return ""
        return (
            f"🎯 收到，开始为「{target}」押题：汇总 JD + 简历 + 库内面经，"
            "heavy 生成约 2~5 分钟，完成后发你押题文档。"
        )

    def _resolve_job_target(self, target: str) -> tuple[Any | None, list[str]]:
        """把求职ID或“公司 岗位”解析成唯一看板记录；歧义时不猜。"""
        if self.io is None:
            return None, []
        records = self.io.list_all(BOARD_TABLE)
        normalized = re.sub(r"\s+", "", target).casefold()
        exact_id = [
            rec for rec in records
            if field_text(record_fields(rec).get("求职ID")).casefold() == target.casefold()
        ]
        if len(exact_id) == 1:
            return exact_id[0], []
        matches = []
        for rec in records:
            fields = record_fields(rec)
            company = re.sub(r"\s+", "", field_text(fields.get("公司"))).casefold()
            position = re.sub(r"\s+", "", field_text(fields.get("岗位"))).casefold()
            if company and (company in normalized or normalized in company):
                matches.append(rec)
                if position and position in normalized:
                    return rec, []
        if len(matches) == 1:
            return matches[0], []
        choices = [
            " ".join(filter(None, (
                field_text(record_fields(rec).get("公司")),
                field_text(record_fields(rec).get("岗位")),
                field_text(record_fields(rec).get("求职ID")),
            )))
            for rec in matches
        ]
        return None, choices

    def _load_resume_record_text(self, rec: Any) -> tuple[str, str]:
        """按已绑定记录读取简历，保证投递后不被“最新版本”替换。"""
        fields = record_fields(rec)
        label = field_text(fields.get("版本名")) or "未命名简历"
        version = resume_version_number(fields)
        if version:
            label += f" · v{version}"
        stored_text = field_text(fields.get("简历正文"))
        if stored_text:
            return label, stored_text
        token, _, suffix = resume_attachment_info(fields)
        if not token:
            return label, ""
        with sensitive_workspace(self.resume_dir, "bound-resume") as workspace:
            dest = workspace / f"resume{suffix}"
            try:
                self.io.download_media(token, dest)
                return label, self.resume_text_extractor(dest)
            except (DriveError, OSError, ValueError) as exc:
                logger.warning("绑定简历读取失败：%s", exc)
                return label, ""

    def _gather_materials(self, target: str) -> dict[str, Any]:
        """按唯一求职机会汇总 JD、锁定简历和面经；同公司多岗位时拒绝猜测。"""
        materials: dict[str, Any] = {
            "company": target, "position": "", "job_id": "", "jd": "",
            "resume": "", "resume_label": "", "exps": [], "ambiguous": [],
        }
        if self.io is None:
            return materials
        job, choices = self._resolve_job_target(target)
        if choices:
            materials["ambiguous"] = choices
            return materials

        direction = ""
        job_fields = record_fields(job) if job is not None else {}
        if job_fields:
            materials["company"] = field_text(job_fields.get("公司")) or target
            materials["position"] = field_text(job_fields.get("岗位"))
            materials["job_id"] = field_text(job_fields.get("求职ID"))
            materials["jd"] = field_text(job_fields.get("JD原文"))
            direction = field_text(job_fields.get("岗位方向"))

        try:
            resume_records = self.io.list_all(RESUME_TABLE)
            bound_id = (
                field_text(job_fields.get("实际投递简历ID"))
                or field_text(job_fields.get("推荐简历ID"))
            )
            bound = next((r for r in resume_records if record_id_of(r) == bound_id), None)
            if bound is not None:
                label, text = self._load_resume_record_text(bound)
            else:
                label, text = self._load_resume_text(direction, materials["position"])
            materials["resume_label"] = label
            materials["resume"] = text
        except Exception as exc:
            logger.warning("简历读取失败（按无简历继续）：%s", exc)

        try:
            exp_records = self.io.list_all(EXP_TABLE)
            if materials["job_id"]:
                same = [
                    r for r in exp_records
                    if field_text(record_fields(r).get("求职ID")) == materials["job_id"]
                ]
                # 旧面经尚无求职ID时，仅在岗位已唯一解析的前提下按公司补充。
                same += [
                    r for r in exp_records
                    if not field_text(record_fields(r).get("求职ID"))
                    and materials["company"].lower()
                    in field_text(record_fields(r).get("公司")).lower()
                ]
            else:
                same = [
                    r for r in exp_records
                    if target.lower() in field_text(record_fields(r).get("公司")).lower()
                ]
            materials["exps"] = [exp_record_text(r) for r in same[:5]]
        except BitableError as exc:
            logger.warning("读取面经库失败：%s", exc)
        return materials

    def _load_resume_text(self, direction: str = "", position: str = "") -> tuple[str, str]:
        """取简历原文：优先匹配具体岗位画像，再考虑旧方向与上传时间。

        新记录直接使用入库时抽取的正文；旧记录才下载附件兜底。
        """
        records = self.io.list_all(RESUME_TABLE)
        if not records:
            return "", ""

        def _uploaded(rec: Any) -> float:
            return field_ms(record_fields(rec).get("上传时间"))

        position_terms = {
            term.casefold() for term in re.findall(r"[A-Za-z][A-Za-z0-9+.#_-]{1,}|[一-鿿]{2,8}", position)
        }
        normalized_position = re.sub(r"\s+", "", position).casefold()

        def _match_score(rec: Any) -> tuple[int, int, float]:
            fields = record_fields(rec)
            profile_text = " ".join([
                field_text(fields.get("主方向")),
                *field_list(fields.get("目标岗位")),
                *field_list(fields.get("能力标签")),
                field_text(fields.get("简历摘要")),
            ]).casefold()
            score = sum(3 for term in position_terms if term in profile_text)
            for role in field_list(fields.get("目标岗位")):
                normalized_role = re.sub(r"\s+", "", role).casefold()
                if normalized_role and normalized_position and (
                    normalized_role in normalized_position or normalized_position in normalized_role
                ):
                    score += 8
            if direction and field_text(fields.get("岗位方向")) == direction:
                score += 2
            if field_text(fields.get("解析状态")) == "已解析":
                score += 1
            return score, int(fields.get("当前版本") is True), _uploaded(rec)

        rec = max(records, key=_match_score)
        fields = record_fields(rec)
        label = field_text(fields.get("版本名")) or "未命名简历"
        version = resume_version_number(fields)
        if version:
            label += f" · v{version}"
        stored_text = field_text(fields.get("简历正文"))
        if stored_text:
            return label, stored_text
        token, _, suffix = resume_attachment_info(fields)
        if not token:
            logger.warning("简历「%s」无附件 file_token，仅有元信息", label)
            return label, ""
        with sensitive_workspace(self.resume_dir, "latest-resume") as workspace:
            dest = workspace / f"resume{suffix}"
            try:
                self.io.download_media(token, dest)
            except DriveError as exc:
                logger.warning("简历附件下载失败：%s", exc)
                return label, ""
            return label, self.resume_text_extractor(dest)

    def _job_predict(self, context: MessageContext, target: str) -> str:
        self._progress(context, "🔍 汇总材料中：看板 JD + 简历库 + 面经库……")
        materials = self._gather_materials(target)
        if materials["ambiguous"]:
            choices = "\n".join(f"{i}. {item}" for i, item in enumerate(materials["ambiguous"], 1))
            return (
                "同一公司有多个活跃岗位，我没有自动猜：\n" + choices
                + "\n请用“押题 求职ID”或“押题 公司 岗位”重新指定。"
            )
        prompt = build_predict_prompt(
            materials["company"], materials["position"], materials["jd"],
            materials["resume"], materials["exps"],
        )
        self._progress(context, "🤖 押题生成中（heavy 模型，约 2~5 分钟）……")
        output = self._heavy_chat([{"role": "user", "content": prompt}], context)

        position = materials["position"]
        title = " ".join(p for p in (target, position) if p)
        doc_url = self._write_doc(context, f"押题：{title}", output)

        notes = []
        if not materials["jd"]:
            notes.append("看板无该公司 JD 原文（可把 JD 发我做分析后自动关联）")
        if not materials["resume"]:
            notes.append("简历库无可用简历原文")
        if not materials["exps"]:
            notes.append("库内无该公司面经")
        lines = [f"✅ 备战题已生成：{title}（已按岗位能力、经历深挖和实战形式组织）"]
        if doc_url:
            lines.append(f"📄 押题文档：{doc_url}")
        if notes:
            lines.append("⚠️ " + "；".join(notes) + "，已按通用情况出题。")
        lines.append("💬 想实战演练可发「模拟面试 " + target + "」。")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 文字模拟面试（career 绑定会话状态机）
    # ------------------------------------------------------------------

    def cmd_mock_interview(self, arg: str, ctx: MessageContext | None = None) -> str:
        """「模拟面试」命令：建 career 绑定会话并切换，面试官异步开场。"""
        if self._require_io() is None:
            return DAO_MISSING
        if self.session_store is None or ctx is None:
            return "模拟面试需要会话功能与消息上下文。"
        active = self.session_store.get_active_session(ctx.chat_id)
        if active is not None and active.bind_type == "career" and not self._interview_ended(active):
            return "你已经在一场模拟面试中了。发「结束面试」收尾，或发「切换 N」先离开。"
        target = (arg or "").strip() or "目标公司"
        job, choices = self._resolve_job_target(target)
        if choices:
            lines = ["同一公司有多个活跃岗位，请指定后再开始模拟面试："]
            lines += [f"{i}. {choice}" for i, choice in enumerate(choices, 1)]
            lines.append("用法：模拟面试 求职ID，或“模拟面试 公司 岗位”。")
            return "\n".join(lines)
        job_fields = record_fields(job) if job is not None else {}
        company = field_text(job_fields.get("公司")) or target
        position = field_text(job_fields.get("岗位"))
        bind_ref = field_text(job_fields.get("求职ID")) or target
        title_target = " ".join(filter(None, (company, position)))
        session = self.session_store.create_session(
            ctx.chat_id, f"模拟面试·{title_target}", bind_type="career", bind_ref=bind_ref
        )
        self.session_store.ensure_interview_state(session.id)
        self.session_store.set_active_session(ctx.chat_id, session.id)
        logger.info("模拟面试开始：chat=%s session=%s 求职机会=%s", ctx.chat_id, session.code, bind_ref)
        self._start_async(ctx, "模拟面试开场", self._job_interview_opening, session_id=session.id)
        return (
            f"💼 已切换到会话「{session.title}」，面试官准备中……\n"
            "之后直接回答即可（多轮追问）；发「结束面试」出复盘报告。"
        )

    def _job_interview_opening(self, context: MessageContext, session_id: int) -> str:
        session = self.session_store.get_session(session_id)
        if session is None:
            raise CareerError("模拟面试会话找不到了。")
        system_prompt = self._get_interviewer_prompt(session)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "（面试开始，请按开场要求开始）"},
        ]
        reply = self._heavy_chat(messages, context)
        self.session_store.add_message(session_id, session.chat_id, "assistant", reply)
        return reply

    def _get_interviewer_prompt(self, session: Session) -> str:
        """装配（并缓存）面试官 system prompt：模板 + 看板/简历/面经素材。"""
        cached = self._interview_prompts.get(session.id)
        if cached:
            return cached
        materials = self._gather_materials(session.bind_ref)
        context_parts = []
        if materials["jd"]:
            context_parts.append(f"【JD 原文】\n{_clip(materials['jd'], MAX_JD_CHARS)}")
        if materials["resume"]:
            context_parts.append(
                f"【候选人简历（{materials['resume_label']}）】\n{_clip(materials['resume'], MAX_RESUME_CHARS)}"
            )
        if materials["exps"]:
            context_parts.append(f"【库内面经】\n{_clip(chr(10).join(materials['exps']), MAX_EXP_CHARS)}")
        prompt = build_interviewer_prompt(
            materials["company"], materials["position"], "\n\n".join(context_parts)
        )
        self._interview_prompts[session.id] = prompt
        return prompt

    def _interview_ended(self, session: Session) -> bool:
        if session.id in self._ended_sessions:
            return True
        if self.session_store is not None:
            if self.session_store.interview_state(session.id) == "ended":
                return True
            # 兼容升级前已经结束、但尚无 interview_states 记录的旧会话。
            recent = self.session_store.recent_messages(session.id, 1)
            if recent and recent[-1].role == "assistant" and END_MARKER in recent[-1].content:
                return True
        return False

    def answer_in_interview(
        self, session: Session, text: str, ctx: MessageContext | None = None
    ) -> str:
        """career 绑定会话内的消息：面试官多轮追问 / 「结束面试」出复盘。"""
        if self.session_store is None:
            return "会话功能未配置。"
        if self._interview_ended(session):
            return f"（{END_MARKER}）发「切换 N」回到其他会话，或重新发「模拟面试 公司」再来一场。"
        if self.session_store.interview_state(session.id) == "ending":
            return "🏁 面试已结束，复盘报告正在生成，请不要重复提交。"

        if is_end_interview_command(text):
            if not self.session_store.begin_interview_end(session.id):
                state = self.session_store.interview_state(session.id)
                return (
                    f"（{END_MARKER}）发「切换 N」回到其他会话。"
                    if state == "ended"
                    else "🏁 面试已结束，复盘报告正在生成，请不要重复提交。"
                )
            try:
                self.session_store.add_message(session.id, session.chat_id, "user", text)
            except Exception:
                self.session_store.reset_interview_end(session.id)
                raise
            if ctx is not None:
                try:
                    msg_id = self._start_async(
                        ctx, "模拟面试复盘", self._job_end_interview, session_id=session.id)
                except Exception:
                    self.session_store.reset_interview_end(session.id)
                    raise
                if msg_id:
                    return ""
                return "🏁 收到，面试结束。复盘报告生成中（约 1~2 分钟）……"
            return self._end_interview_with_recovery(session)  # 无上下文（测试路径）同步出报告

        self.session_store.add_message(session.id, session.chat_id, "user", text)
        system_prompt = self._get_interviewer_prompt(session)
        history = self.session_store.recent_messages(session.id, 30)
        messages = [{"role": "system", "content": system_prompt}]
        dialog_chars = 0
        trimmed: list[dict[str, str]] = []
        for msg in reversed(history):  # 从最新往回装，控制总长
            if dialog_chars + len(msg.content) > MAX_DIALOG_CHARS and trimmed:
                break
            trimmed.append({"role": msg.role, "content": msg.content})
            dialog_chars += len(msg.content)
        messages += list(reversed(trimmed))
        try:
            result = self.kimi.chat_full(messages, heavy=True)
            reply = result.content
        except KimiError as exc:
            reply = f"调用大模型出错了：{exc}"
        self.session_store.add_message(session.id, session.chat_id, "assistant", reply)
        return reply

    def _job_end_interview(self, context: MessageContext, session_id: int) -> str:
        session = self.session_store.get_session(session_id)
        if session is None:
            raise CareerError("模拟面试会话找不到了。")
        return self._end_interview_with_recovery(session, context)

    def _end_interview_with_recovery(
        self, session: Session, context: MessageContext | None = None,
    ) -> str:
        try:
            return self._end_interview_now(session, context)
        except Exception:
            self.session_store.reset_interview_end(session.id)
            raise

    def _end_interview_now(self, session: Session, context: MessageContext | None = None) -> str:
        """生成复盘报告（heavy + 文档 + 面经库），标记会话结束。

        副作用按对话指纹幂等：complete_interview_end 落库失败会 reset 允许重试，
        重试凭指纹复用首份回执，只重放状态落库；用户继续面试后对话变化，
        指纹不同则正常重新复盘。（进程重启丢回执是已知残余风险，与
        _ended_sessions 的内存语义一致。）
        """
        history = self.session_store.all_messages(session.id)
        dialog = "\n".join(
            f"{'候选人' if m.role == 'user' else '面试官'}: {m.content}" for m in history
        )
        # 结束命令本身会被重复落库（每次重试各一条），不带面试内容，指纹须剔除。
        fingerprint = hashlib.sha256("\n".join(
            f"{m.role}: {m.content}" for m in history
            if not (m.role == "user" and is_end_interview_command(m.content))
        ).encode("utf-8")).hexdigest()
        receipt = self._review_receipts.get(session.id)
        if receipt is not None and receipt[0] == fingerprint:
            reply = receipt[1]
        else:
            ctx = context or MessageContext(chat_id=session.chat_id)
            reply = self._finish_review(
                ctx, dialog, hint=f"模拟面试练习：{session.bind_ref}", job_ref=session.bind_ref,
            )
            self._review_receipts[session.id] = (fingerprint, reply)
        final = f"🏁 {END_MARKER}。\n{reply}\n发「切换 N」可回到其他会话。"
        self.session_store.complete_interview_end(session, final)
        self._review_receipts.pop(session.id, None)
        self._ended_sessions.add(session.id)
        return final

    # ------------------------------------------------------------------
    # 自由文本钩子（router 的 career_fn 接线）：面试会话 > JD 粘贴
    # ------------------------------------------------------------------

    def try_handle_text(self, text: str, ctx: MessageContext | None = None) -> str | None:
        """非命令文本的求职入口；不归我管返回 None。"""
        if ctx is not None and self.session_store is not None:
            session = self.session_store.get_active_session(ctx.chat_id)
            if session is not None and session.bind_type == "career":
                return self.answer_in_interview(session, text, ctx)
        if looks_like_jd(text):
            if ctx is None:
                return None
            return self.analyze_jd(text, ctx)
        return None

    # ------------------------------------------------------------------
    # 内部：heavy 调用 / 文档落地
    # ------------------------------------------------------------------

    def _heavy_chat(self, messages: list[dict[str, str]], context: MessageContext) -> str:
        """heavy 链调用，带心跳；失败抛 KimiError 由 _run_guarded 翻译。"""
        heartbeat = Heartbeat(lambda s: self._progress(context, s))
        heartbeat.start()
        try:
            return self.kimi.chat(messages, heavy=True)
        finally:
            heartbeat.stop()

    def _write_doc(self, context: MessageContext, title: str, body_md: str) -> str:
        """复盘/分析/押题 markdown → 飞书文档 + 授权，返回 URL；失败返回 ""（不炸管线）。"""
        if self.docx is None:
            logger.warning("docx 客户端未配置，跳过文档落地：%s", title)
            return ""
        try:
            doc_id, doc_url = self.docx.create_document(title)
            self.docx.write_markdown(doc_id, body_md)
            self.docx.grant_access(doc_id, open_id=context.open_id or None)
            logger.info("文档已落地：%s → %s", title, doc_url)
            return doc_url
        except DocxError as exc:
            logger.error("飞书文档写入失败（正文改走消息推送）：%s", exc)
            return ""


def extract_resume_text(path: Path) -> str:
    """从 PDF/docx 抽取简历正文；失败返回空字符串且不猜岗位。"""
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            import pymupdf4llm  # 惰性导入（重依赖）

            return pymupdf4llm.to_markdown(str(path)).strip()
        if suffix == ".docx":
            # docx 是 ZIP + WordprocessingML，用标准库即可稳定取段落与表格文本。
            with zipfile.ZipFile(path) as archive:
                xml = archive.read("word/document.xml")
            root = ElementTree.fromstring(xml)
            namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
            lines: list[str] = []
            for paragraph in root.iter(namespace + "p"):
                text = "".join(node.text or "" for node in paragraph.iter(namespace + "t")).strip()
                if text:
                    lines.append(text)
            return "\n".join(lines)
        return ""
    except Exception as exc:
        logger.warning("简历文本抽取失败（%s）：%s", path.name, exc)
        return ""


def _classify_error(exc: Exception) -> str:
    """把异常翻译成给用户看的中文原因。"""
    if isinstance(exc, CareerError):
        return str(exc)
    if isinstance(exc, ASRError):
        return str(exc)
    if isinstance(exc, KimiError):
        return f"{exc}（系统没有自动切换；请检查当前 API/模型，或在「模型」面板手动选择）"
    if isinstance(exc, DocxError):
        return f"{exc}（检查 docx:document 权限与版本发布）"
    if isinstance(exc, DriveError):
        return f"{exc}（检查 im:resource / drive:drive 权限与版本发布）"
    if isinstance(exc, BitableError):
        return f"{exc}（检查 bitable:app 权限，或重跑 feishu-assistant init）"
    return f"{type(exc).__name__}：{exc}（日志里有完整 traceback，可发给开发者排查）"
