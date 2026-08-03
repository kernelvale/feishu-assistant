"""复杂公式资产测试：原 PDF 公式截图、标签、放大 PNG 与清单。"""

from __future__ import annotations

import json

from PIL import Image

from feishu_assistant.modules.formulas import (
    formula_inventory_text,
    load_formula_assets,
    normalize_formula_label,
)


def _formula_dir(tmp_path):
    parse_dir = tmp_path / "auto"
    images = parse_dir / "images"
    images.mkdir(parents=True)
    for name in ("eq1.jpg", "eq2.jpg", "eq3.jpg"):
        Image.new("RGB", (240, 40), "white").save(images / name, format="JPEG")
    items = [
        {"type": "text", "text": "3 Method", "text_level": 1},
        {"type": "equation", "img_path": "images/eq1.jpg",
         "text": r"$$x = \frac{a}{b}.\tag{1}$$", "page_idx": 2},
        {"type": "equation", "img_path": "images/eq2.jpg",
         "text": r"$$y = \sum_i z_i$$", "page_idx": 2},
        {"type": "equation", "img_path": "images/eq3.jpg",
         "text": r"$$x' = x + 1.\tag{1}$$", "page_idx": 3},
        {"type": "equation", "img_path": "images/missing.jpg",
         "text": r"$$bad$$", "page_idx": 4},
    ]
    (parse_dir / "paper_content_list.json").write_text(
        json.dumps(items), encoding="utf-8",
    )
    return parse_dir


def test_load_formula_assets_uses_original_images_and_unique_labels(tmp_path):
    assets = load_formula_assets(_formula_dir(tmp_path))
    assert [asset.label for asset in assets] == [
        "Equation 1", "Equation p3-2", "Equation 1（片段2）",
    ]
    assert all(asset.img_path.suffix == ".png" and asset.img_path.exists() for asset in assets)
    assert all(asset.section == "3 Method" for asset in assets)
    with Image.open(assets[0].img_path) as image:
        assert image.width >= 960 and image.height >= 160
    assert r"\frac{a}{b}" in assets[0].latex


def test_formula_inventory_contains_source_and_empty_contract(tmp_path):
    assets = load_formula_assets(_formula_dir(tmp_path))
    inventory = formula_inventory_text(assets)
    assert "Equation 1" in inventory and "第 3 页" in inventory
    assert "原式 LaTeX" in inventory
    assert "不要输出 MATH 标记" in formula_inventory_text([])
    assert "本层未选择复杂公式" in formula_inventory_text([], empty_scope="layer")


def test_formula_label_normalization_handles_model_spelling_drift():
    values = ["Equation 3", "Eq.3", "Eq 3", "式 3", "公式３"]
    assert {normalize_formula_label(value) for value in values} == {"3"}
    assert normalize_formula_label("Equation 1（片段2）") == "1fragment2"


def test_load_formula_assets_without_content_list_is_empty(tmp_path):
    assert load_formula_assets(tmp_path) == []
