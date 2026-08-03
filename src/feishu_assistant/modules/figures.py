"""图表抽取与配对：MinerU content_list → 图片文件 ↔ 原始图注。

MinerU 输出目录的 *_content_list.json 里，image / chart / table 三类条目带：
- img_path：区域截图（图表的完整图像内容）
- image_caption / chart_caption / table_caption：原始图注（Figure N / Table N caption）
- page_idx、bbox、在 content_list 中的顺序（即原文顺序）

配对原则：图注里优先选以 Figure/Table + 编号开头的条目；完全没有图注的
跳过并记日志（宁缺毋滥）。跨页续表（caption 含 "continued"）全部保留并标注。
"""

from __future__ import annotations

import json
import logging
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("feishu_assistant.figures")

_LABEL_RE = re.compile(r"^(Figure|Table|Fig\.?|表|图)\s*(\d+)", re.IGNORECASE)
_CONTINUED_RE = re.compile(r"continued|续表|续图", re.IGNORECASE)
_CAPTION_KEYS = ("image_caption", "chart_caption", "table_caption")
_LATEX_SYMBOLS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "epsilon": "ε", "rho": "ρ", "sigma": "σ", "theta": "θ",
    "lambda": "λ", "mu": "μ", "pi": "π", "tau": "τ",
    "times": "×", "propto": "∝", "leq": "≤", "geq": "≥",
}


def _normalize_caption(caption: str) -> str:
    """清掉 MinerU 混入图注开头的 HTML/上标残片，并保留正式编号后的原文。"""
    text = re.sub(r"<[^>]+>", "", str(caption or ""))
    match = re.search(r"(?:Figure|Table|Fig\.?|表|图)\s*\d+", text, re.IGNORECASE)
    if match and match.start() <= 80:
        text = text[match.start():]
    return text.strip()


@dataclass
class FigureAsset:
    """一张配对成功的图/表。"""

    label: str          # 原论文正式图号；跨页续表保留“续”标记
    caption: str        # 完整原始图注
    img_path: Path      # 图片文件绝对路径
    kind: str           # image | chart | table
    page_idx: int       # 0 基页码（PDF 页序）
    order: int          # 在原文中的顺序
    section: str = ""   # 章节归属（最近的上级标题文本）

    @property
    def is_table(self) -> bool:
        return self.label.lower().startswith(("table", "表"))


def _pick_caption(item: dict) -> str:
    """从 caption 列表里选正式图注：优先 Figure/Table 编号开头；退而拼接全部。"""
    caps: list[str] = []
    for key in _CAPTION_KEYS:
        caps.extend(item.get(key) or [])
    caps = [_normalize_caption(c) for c in caps if c and str(c).strip()]
    if not caps:
        return ""
    for c in caps:
        if _LABEL_RE.match(c):
            return c
    # MinerU 有时把图内文字也收进 caption，正式图注通常是含 Figure N 的那条
    for c in caps:
        if re.search(r"(Figure|Table)\s*\d+", c, re.IGNORECASE):
            return c
    return caps[-1]


def _captions(item: dict) -> list[str]:
    captions: list[str] = []
    for key in _CAPTION_KEYS:
        captions.extend(item.get(key) or [])
    return [_normalize_caption(caption) for caption in captions if str(caption).strip()]


def _formal_captions(item: dict) -> list[str]:
    """返回条目中所有带正式 Figure/Table 编号的图注，而非只取第一条。"""
    return [caption for caption in _captions(item) if _LABEL_RE.match(caption)]


def _is_panel_caption(caption: str) -> bool:
    """识别 (a)/(b)/（1）等子图标题；它们不是可独立发布的正式图号。"""
    return bool(re.match(r"^\s*[（(][A-Za-z0-9一二三四五六七八九]+[)）]", caption))


def _compose_panels(parse_dir: Path, items: list[dict], start: int, end: int) -> Path | None:
    """按 PDF bbox 位置把同一正式图的连续子图恢复成一张合成图。"""
    try:
        from PIL import Image

        images = [Image.open(parse_dir / str(item["img_path"])).convert("RGB") for item in items]
        boxes = [item.get("bbox") for item in items]
        if all(isinstance(box, list) and len(box) == 4 for box in boxes):
            valid_boxes = [box for box in boxes if isinstance(box, list)]
            scale_x = statistics.median(
                image.width / max(1, float(box[2]) - float(box[0]))
                for image, box in zip(images, valid_boxes)
            )
            scale_y = statistics.median(
                image.height / max(1, float(box[3]) - float(box[1]))
                for image, box in zip(images, valid_boxes)
            )
            left = min(float(box[0]) for box in valid_boxes)
            top = min(float(box[1]) for box in valid_boxes)
            right = max(float(box[2]) for box in valid_boxes)
            bottom = max(float(box[3]) for box in valid_boxes)
            canvas = Image.new(
                "RGB",
                (max(1, round((right - left) * scale_x)),
                 max(1, round((bottom - top) * scale_y))),
                "white",
            )
            for image, box in zip(images, valid_boxes):
                width = max(1, round((float(box[2]) - float(box[0])) * scale_x))
                height = max(1, round((float(box[3]) - float(box[1])) * scale_y))
                resized = image.resize((width, height))
                x = round((float(box[0]) - left) * scale_x)
                y = round((float(box[1]) - top) * scale_y)
                canvas.paste(resized, (x, y))
        else:
            height = max(image.height for image in images)
            resized = [
                image.resize((max(1, round(image.width * height / image.height)), height))
                for image in images
            ]
            canvas = Image.new("RGB", (sum(image.width for image in resized), height), "white")
            x = 0
            for image in resized:
                canvas.paste(image, (x, 0))
                x += image.width
        output_dir = parse_dir / "images" / "_composites"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"panels-{start}-{end}.png"
        canvas.save(output, format="PNG")
        return output
    except (OSError, KeyError, ValueError) as exc:
        logger.warning("复合图恢复失败（条目 %d-%d）：%s", start, end, exc)
        return None


def _make_label(caption: str, kind: str, order: int) -> str:
    m = _LABEL_RE.match(caption)
    if m:
        raw = m.group(1).lower()
        word = "Figure" if raw.startswith(("figure", "fig")) or raw == "图" else "Table"
        label = f"{word} {m.group(2)}"
    else:
        label = f"{kind}-{order}"
    if _CONTINUED_RE.search(caption):
        label += "（续）"
    return label


def plain_figure_caption(caption: str) -> str:
    """把图注中的轻量 LaTeX 降为可读纯文本，避免原始命令泄漏到文档。"""
    text = _normalize_caption(caption)
    for command, symbol in _LATEX_SYMBOLS.items():
        text = re.sub(rf"\\{command}\b", symbol, text)
    text = re.sub(r"\\(?:scriptstyle|displaystyle|textstyle)\b", "", text)
    text = re.sub(r"\\(?:mathrm|mathbf|mathit|text)\s*\{([^{}]*)\}", r"\1", text)
    text = text.replace("$", "")
    text = re.sub(r"\\([{}_%&#])", r"\1", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\\[A-Za-z]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_figure_assets(parse_dir: Path | str) -> list[FigureAsset]:
    """从 MinerU 输出目录加载图表配对，按原文顺序返回。解析失败/无文件返回 []。"""
    parse_dir = Path(parse_dir)
    candidates = sorted(parse_dir.glob("*_content_list.json"))
    if not candidates:
        logger.warning("找不到 content_list.json：%s", parse_dir)
        return []
    try:
        items = json.loads(candidates[0].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("content_list.json 解析失败：%s", exc)
        return []

    visual_indices = [
        index for index, item in enumerate(items)
        if item.get("type") in ("image", "chart", "table")
    ]

    # MinerU 偶尔把同页相邻两张表的两个正式图注都挂到后一张图上。
    # 仅在“图注数 == 连续视觉条目数”时按页面顺序一一恢复，避免语义猜测。
    caption_overrides: dict[int, str] = {}
    for position, index in enumerate(visual_indices):
        formal = _formal_captions(items[index])
        if len(formal) <= 1:
            continue
        group = [index]
        cursor = position - 1
        while cursor >= 0 and len(group) < len(formal):
            previous = visual_indices[cursor]
            prev_item = items[previous]
            if (prev_item.get("page_idx") != items[index].get("page_idx")
                    or prev_item.get("type") != items[index].get("type")
                    or _formal_captions(prev_item)):
                break
            group.insert(0, previous)
            cursor -= 1
        if len(group) == len(formal):
            caption_overrides.update(dict(zip(group, formal)))
            logger.info("恢复同页多图注配对：%s", "、".join(formal))

    # 连续的 (a)/(b)/(c) 子图若最后一块携带正式 Figure 图注，则合并为父图；
    # 子图不再以 chart-N 之类内部名称单独发布。
    panel_groups: dict[int, list[int]] = {}
    panel_members: set[int] = set()
    for position, index in enumerate(visual_indices):
        item = items[index]
        formal = _formal_captions(item)
        captions = _captions(item)
        if len(formal) != 1 or not any(_is_panel_caption(c) for c in captions):
            continue
        group = [index]
        cursor = position - 1
        while cursor >= 0:
            previous = visual_indices[cursor]
            prev_item = items[previous]
            prev_captions = _captions(prev_item)
            if (prev_item.get("page_idx") != item.get("page_idx")
                    or prev_item.get("type") != item.get("type")
                    or not prev_captions
                    or _formal_captions(prev_item)
                    or not all(_is_panel_caption(c) for c in prev_captions)):
                break
            group.insert(0, previous)
            cursor -= 1
        if len(group) > 1:
            panel_groups[index] = group
            panel_members.update(group[:-1])

    sections: dict[int, str] = {}
    current_section = ""
    for order, item in enumerate(items):
        if item.get("type") == "text" and item.get("text_level"):
            current_section = (item.get("text") or "").strip()[:60]
        sections[order] = current_section

    assets: list[FigureAsset] = []
    label_counts: dict[str, int] = {}
    for order, item in enumerate(items):
        itype = item.get("type")
        if itype not in ("image", "chart", "table"):
            continue
        if order in panel_members:
            continue
        img_rel = item.get("img_path") or ""
        img_path = parse_dir / img_rel
        if not img_rel or not img_path.exists():
            logger.warning("图表条目缺图片文件，跳过：%s", img_rel or item)
            continue
        caption = caption_overrides.get(order) or _pick_caption(item)
        if order in panel_groups:
            group = panel_groups[order]
            composite = _compose_panels(
                parse_dir, [items[index] for index in group], group[0], group[-1],
            )
            if composite is None:
                logger.warning("正式复合图无法恢复，跳过并交由完整性闸门处理：%s", caption)
                continue
            img_path = composite
        if not caption:
            logger.warning("图表无图注，宁缺毋滥跳过：%s", img_rel)
            continue
        if not _LABEL_RE.match(caption):
            logger.info("跳过无正式编号的图表碎片：%s（%s）", img_rel, caption[:80])
            continue
        base_label = _make_label(caption, itype, order)
        label_counts[base_label] = label_counts.get(base_label, 0) + 1
        label = (base_label if label_counts[base_label] == 1
                 else f"{base_label}（片段{label_counts[base_label]}）")
        assets.append(FigureAsset(
            label=label,
            caption=caption,
            img_path=img_path,
            kind=itype,
            page_idx=int(item.get("page_idx", 0)),
            order=panel_groups.get(order, [order])[0],
            section=sections.get(order, ""),
        ))
    logger.info("图表配对完成：%d 张（%s）", len(assets),
                "、".join(a.label for a in assets) or "无")
    return assets


def figure_inventory_text(assets: list[FigureAsset]) -> str:
    """给 prompt 用的图表清单（无图时给明确说明）。"""
    if not assets:
        return "（本文未抽取到图表，正文中不要引用任何图）"
    lines = [f"共 {len(assets)} 张图表，写作时严格使用清单中的原始编号："]
    for a in assets:
        section = f"，章节：{a.section}" if a.section else ""
        lines.append(f"- {a.label}（第 {a.page_idx + 1} 页{section}）：{a.caption}")
    return "\n".join(lines)
