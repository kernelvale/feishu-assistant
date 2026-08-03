"""岗位定制简历的事实验收、原 DOCX 最小替换与渲染校验。

设计目标：原文件是格式权威。只修改 word/document.xml 中被明确选中的文本节点，
其余 package part 原样复制；不从空白模板重建简历。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lxml import etree

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"


@dataclass(frozen=True)
class TailorReplacement:
    source: str
    replacement: str
    jd_requirement: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class TailorPlan:
    summary: str
    replacements: tuple[TailorReplacement, ...]
    unmet_requirements: tuple[str, ...]
    questions: tuple[str, ...]


@dataclass(frozen=True)
class DocxQAResult:
    status: str
    source_pages: int | None
    output_pages: int | None
    output_pdf: Path | None
    preserved_parts: bool

    @property
    def passed(self) -> bool:
        """只有完成结构与渲染验收的结果才允许进入简历库。"""
        return self.preserved_parts and self.status.startswith("通过：")


def _extract_json(raw: str) -> dict[str, Any] | None:
    text = (raw or "").strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            try:
                value = json.loads(text[start:end + 1])
                return value if isinstance(value, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def _clean(value: Any, limit: int) -> str:
    return re.sub(r"[\r\n]+", " ", str(value or "")).strip()[:limit]


def _in_resume(snippet: str, resume_text: str) -> bool:
    needle = re.sub(r"\s+", "", snippet).casefold()
    haystack = re.sub(r"\s+", "", resume_text).casefold()
    return len(needle) >= 3 and needle in haystack


def _number_tokens(text: str) -> set[str]:
    return set(re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?%?", text or ""))


def parse_tailor_plan(raw: str, resume_text: str) -> TailorPlan | None:
    """验收模型替换计划：原文/证据可追溯，且禁止凭空增加数字。"""
    data = _extract_json(raw)
    if not data:
        return None
    resume_numbers = _number_tokens(resume_text)
    replacements: list[TailorReplacement] = []
    used_sources: set[str] = set()
    for item in data.get("replacements") or []:
        if not isinstance(item, dict):
            continue
        source = _clean(item.get("source"), 2000)
        replacement = _clean(item.get("replacement"), 2400)
        if not source or not replacement or source == replacement or source in used_sources:
            continue
        if resume_text.count(source) != 1:
            continue
        if len(replacement) > max(len(source) * 1.5, len(source) + 24):
            continue
        if _number_tokens(replacement) - resume_numbers:
            continue
        evidence = tuple(
            value for value in (
                _clean(v, 300) for v in (item.get("evidence") or [])
            ) if value and _in_resume(value, resume_text)
        )[:3]
        if not evidence:
            continue
        replacements.append(TailorReplacement(
            source=source,
            replacement=replacement,
            jd_requirement=_clean(item.get("jd_requirement"), 300),
            evidence=evidence,
        ))
        used_sources.add(source)
    if not replacements:
        return None
    return TailorPlan(
        summary=_clean(data.get("summary"), 300),
        replacements=tuple(replacements[:20]),
        unmet_requirements=tuple(_clean(v, 300) for v in (data.get("unmet_requirements") or []) if _clean(v, 300))[:20],
        questions=tuple(_clean(v, 300) for v in (data.get("questions") or []) if _clean(v, 300))[:20],
    )


def _set_text(node: etree._Element, value: str) -> None:
    node.text = value
    if value.startswith(" ") or value.endswith(" "):
        node.set(XML_SPACE, "preserve")
    else:
        node.attrib.pop(XML_SPACE, None)


def _replace_in_paragraph(paragraph: etree._Element, source: str, replacement: str) -> bool:
    nodes = paragraph.xpath(".//w:t", namespaces=NS)
    values = [node.text or "" for node in nodes]
    combined = "".join(values)
    start = combined.find(source)
    if start < 0 or combined.find(source, start + 1) >= 0:
        return False
    end = start + len(source)
    spans: list[tuple[int, int]] = []
    cursor = 0
    for value in values:
        spans.append((cursor, cursor + len(value)))
        cursor += len(value)
    affected = [i for i, (left, right) in enumerate(spans) if right > start and left < end]
    if not affected:
        return False
    first, last = affected[0], affected[-1]
    first_left, _ = spans[first]
    _, last_right = spans[last]
    prefix = values[first][:max(0, start - first_left)]
    suffix = values[last][len(values[last]) - max(0, last_right - end):] if last_right > end else ""
    _set_text(nodes[first], prefix + replacement + (suffix if first == last else ""))
    for index in affected[1:]:
        _set_text(nodes[index], suffix if index == last else "")
    return True


def patch_docx_text(source_docx: Path, output_docx: Path, plan: TailorPlan) -> list[str]:
    """复制原 DOCX，仅定点修改正文 XML；找不到或不唯一的 source 直接失败。"""
    output_docx.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_docx, "r") as archive:
        document_xml = archive.read("word/document.xml")
        root = etree.fromstring(document_xml)
        paragraphs = root.xpath(".//w:p", namespaces=NS)
        applied: list[str] = []
        for change in plan.replacements:
            hits = [
                paragraph for paragraph in paragraphs
                if change.source in "".join(paragraph.xpath(".//w:t/text()", namespaces=NS))
            ]
            if len(hits) != 1 or not _replace_in_paragraph(hits[0], change.source, change.replacement):
                raise ValueError(f"原格式定位失败或不唯一：{change.source[:80]}")
            applied.append(change.source)
        patched_xml = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
        with zipfile.ZipFile(output_docx, "w") as target:
            for info in archive.infolist():
                payload = patched_xml if info.filename == "word/document.xml" else archive.read(info.filename)
                target.writestr(info, payload)
    return applied


def _part_hashes(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        return {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in archive.namelist() if name != "word/document.xml"
        }


def preserved_docx_parts(source_docx: Path, output_docx: Path) -> bool:
    """除正文 XML 外，样式、图片、页眉页脚、关系等必须逐字节保持。"""
    return _part_hashes(source_docx) == _part_hashes(output_docx)


def _render_pdf(docx_path: Path, output_dir: Path) -> Path | None:
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        mac_soffice = Path("/Applications/LibreOffice.app/Contents/MacOS/soffice")
        soffice = str(mac_soffice) if mac_soffice.exists() else None
    if not soffice:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = Path(tempfile.mkdtemp(prefix="resume-lo-", dir=output_dir))
    env = dict(os.environ)
    env.update({"HOME": str(profile), "TMPDIR": str(profile)})
    command = [
        soffice, "--headless", f"-env:UserInstallation=file://{profile}",
        "--convert-to", "pdf", "--outdir", str(output_dir), str(docx_path),
    ]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
    pdf = output_dir / f"{docx_path.stem}.pdf"
    if result.returncode != 0 or not pdf.exists() or pdf.stat().st_size == 0:
        return None
    return pdf


def _pdf_pages(path: Path | None) -> int | None:
    if path is None:
        return None
    try:
        import fitz
        with fitz.open(path) as document:
            return document.page_count
    except Exception:
        return None


def _pdf_layout_signature(path: Path | None) -> tuple[tuple[float, float, int], ...] | None:
    """提取每页尺寸和可见文本量，同时拒绝落在页面边界外的文本块。"""
    if path is None:
        return None
    try:
        import fitz

        signature: list[tuple[float, float, int]] = []
        with fitz.open(path) as document:
            for page in document:
                rect = page.rect
                visible_chars = 0
                for block in page.get_text("blocks"):
                    if len(block) < 5 or not str(block[4]).strip():
                        continue
                    x0, y0, x1, y1 = (float(block[i]) for i in range(4))
                    # LibreOffice/PDF 浮点坐标允许 2pt 误差；明显越界通常意味着溢出或裁切。
                    if x0 < rect.x0 - 2 or y0 < rect.y0 - 2 or x1 > rect.x1 + 2 or y1 > rect.y1 + 2:
                        return None
                    visible_chars += len(re.sub(r"\s+", "", str(block[4])))
                signature.append((round(rect.width, 1), round(rect.height, 1), visible_chars))
        return tuple(signature)
    except Exception:
        return None


def verify_tailored_docx(
    source_docx: Path, output_docx: Path, qa_dir: Path,
) -> DocxQAResult:
    """结构保持 + DOCX→PDF 渲染 + 页数门禁。"""
    preserved = preserved_docx_parts(source_docx, output_docx)
    source_pdf = _render_pdf(source_docx, qa_dir / "source")
    output_pdf = _render_pdf(output_docx, qa_dir / "output")
    source_pages, output_pages = _pdf_pages(source_pdf), _pdf_pages(output_pdf)
    source_layout = _pdf_layout_signature(source_pdf)
    output_layout = _pdf_layout_signature(output_pdf)
    if not preserved:
        status = "失败：原文档非正文结构发生变化"
    elif source_pdf is None or output_pdf is None:
        status = "失败：当前环境无法完成原稿与定制稿的渲染校验"
    elif source_layout is None or output_layout is None:
        status = "失败：渲染结果存在页面越界、裁切或无法读取的版面"
    elif source_pages is None or output_pages is None:
        status = "失败：无法读取原稿或定制稿页数"
    elif source_pages != output_pages:
        status = f"失败：页数由 {source_pages} 变为 {output_pages}"
    elif any(
        abs(source_layout[i][0] - output_layout[i][0]) > 1
        or abs(source_layout[i][1] - output_layout[i][1]) > 1
        for i in range(len(source_layout))
    ):
        status = "失败：定制稿页面尺寸与原稿不一致"
    elif sum(page[2] for page in output_layout) == 0 < sum(page[2] for page in source_layout):
        status = "失败：定制稿渲染后正文不可见"
    else:
        status = f"通过：保持 {output_pages} 页，非正文结构未变化"
    return DocxQAResult(status, source_pages, output_pages, output_pdf, preserved)
