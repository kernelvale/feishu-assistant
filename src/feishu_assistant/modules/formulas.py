"""复杂公式资产：从 MinerU content_list 读取原 PDF 公式截图与 LaTeX。

公式图片来自论文原文，不由模型重绘。加载时只做等比放大并转成 PNG，保证飞书中
可读；LaTeX 仅进入模型来源材料，不直接泄漏到最终文档。
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("feishu_assistant.formulas")

_TAG_RE = re.compile(r"\\tag\s*\{([^{}]+)\}")
_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass
class FormulaAsset:
    label: str
    latex: str
    img_path: Path
    page_idx: int
    order: int
    section: str = ""


def normalize_formula_label(value: str) -> str:
    """把 Equation 3 / Eq.3 / 式 3 / 公式3 归一成同一匹配键。"""
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = re.sub(r"片段\s*(\d+)", r"fragment\1", text)
    text = re.sub(r"^(?:equation|eq\.?|公式|式)\s*", "", text)
    return re.sub(r"[^a-z0-9.-]+", "", text)


def _formula_label(latex: str, page_idx: int, page_order: int) -> str:
    match = _TAG_RE.search(latex)
    if match:
        tag = re.sub(r"\s+", "", match.group(1)).strip()
        if tag:
            return f"Equation {tag}"
    return f"Equation p{page_idx + 1}-{page_order}"


def _prepare_formula_image(source: Path, destination: Path) -> None:
    """把原公式截图转成适合文档展示的高清 PNG；不改变公式内容。"""
    from PIL import Image

    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = image.convert("RGB")
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("公式图片尺寸无效")
        # MinerU 公式裁图通常只有 200~500px 宽；放大后让飞书按页面宽度展示。
        scale = min(4, max(1, (1200 + width - 1) // width, (180 + height - 1) // height))
        if scale > 1:
            image = image.resize((width * scale, height * scale), Image.Resampling.LANCZOS)
        image.save(destination, format="PNG", optimize=True)


def load_formula_assets(parse_dir: Path | str) -> list[FormulaAsset]:
    """读取公式资产，按原文顺序返回；损坏条目跳过并记录原因。"""
    parse_dir = Path(parse_dir)
    candidates = sorted(parse_dir.glob("*_content_list.json"))
    if not candidates:
        return []
    try:
        items = json.loads(candidates[0].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("公式 content_list 解析失败：%s", exc)
        return []

    assets: list[FormulaAsset] = []
    label_counts: dict[str, int] = {}
    page_counts: dict[int, int] = {}
    current_section = ""
    output_dir = parse_dir / "formula_images"
    for order, item in enumerate(items):
        if item.get("type") == "text" and item.get("text_level"):
            current_section = str(item.get("text") or "").strip()[:60]
            continue
        if item.get("type") != "equation":
            continue
        latex = str(item.get("text") or "").strip()
        img_rel = str(item.get("img_path") or "")
        source = parse_dir / img_rel
        if not latex or not img_rel or not source.exists():
            logger.warning("公式缺 LaTeX 或原图，跳过：order=%d path=%s", order, img_rel)
            continue
        page_idx = int(item.get("page_idx", 0))
        page_counts[page_idx] = page_counts.get(page_idx, 0) + 1
        base_label = _formula_label(latex, page_idx, page_counts[page_idx])
        label_counts[base_label] = label_counts.get(base_label, 0) + 1
        label = (base_label if label_counts[base_label] == 1
                 else f"{base_label}（片段{label_counts[base_label]}）")
        safe_name = _SAFE_RE.sub("_", label).strip("_") or f"formula_{order}"
        destination = output_dir / f"{safe_name}_{source.stem[:12]}.png"
        try:
            _prepare_formula_image(source, destination)
        except (OSError, ValueError) as exc:
            logger.warning("公式图片处理失败，跳过 %s：%s", label, exc)
            continue
        assets.append(FormulaAsset(
            label=label,
            latex=latex,
            img_path=destination,
            page_idx=page_idx,
            order=order,
            section=current_section,
        ))
    logger.info("公式资产加载完成：%d 个", len(assets))
    return assets


def formula_inventory_text(assets: list[FormulaAsset], *, empty_scope: str = "paper") -> str:
    if not assets:
        if empty_scope == "layer":
            return "（本层未选择复杂公式；不要输出 MATH 标记。）"
        return "（本文未抽取到可用的复杂公式图片；不要输出 MATH 标记。）"
    lines = ["可用复杂公式如下。只有确实帮助理解核心机制时才选用："]
    for asset in assets:
        section = f"，章节：{asset.section}" if asset.section else ""
        lines.append(
            f"- {asset.label}（第 {asset.page_idx + 1} 页{section}）\n"
            f"  原式 LaTeX：{asset.latex}"
        )
    return "\n".join(lines)
