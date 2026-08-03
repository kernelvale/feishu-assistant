"""figures.py 图注配对测试：content_list 解析、caption 选择、续表、缺图/缺注跳过。"""

from __future__ import annotations

import json

from PIL import Image

from feishu_assistant.modules.figures import (
    figure_inventory_text,
    load_figure_assets,
    plain_figure_caption,
)


def _make_parse_dir(tmp_path):
    auto = tmp_path / "auto"
    (auto / "images").mkdir(parents=True)
    for name in ("a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"):
        (auto / "images" / name).write_bytes(b"\xff\xd8\xff")
    content_list = [
        {"type": "text", "text": "1 Introduction", "text_level": 1},
        {"type": "image", "img_path": "images/a.jpg",
         "image_caption": ["Figure 1: A framework."], "page_idx": 2},
        {"type": "text", "text": "3 Method", "text_level": 1},
        {"type": "chart", "img_path": "images/b.jpg",
         "chart_caption": ["env_id", "Figure 2: Curve."], "page_idx": 3},
        {"type": "table", "img_path": "images/c.jpg",
         "table_caption": ["Table 1: Games."], "page_idx": 5},
        {"type": "table", "img_path": "images/d.jpg",
         "table_caption": ["Table 1 – continued from previous page"], "page_idx": 6},
        {"type": "image", "img_path": "images/missing.jpg",
         "image_caption": ["Figure 9: gone."], "page_idx": 7},   # 图片文件不存在 → 跳过
        {"type": "image", "img_path": "images/e.jpg", "page_idx": 8},  # 无图注 → 跳过
    ]
    (auto / "paper_content_list.json").write_text(
        json.dumps(content_list, ensure_ascii=False), encoding="utf-8")
    return auto


def test_load_figure_assets_pairs_and_labels(tmp_path):
    parse_dir = _make_parse_dir(tmp_path)
    assets = load_figure_assets(parse_dir)
    assert [a.label for a in assets] == ["Figure 1", "Figure 2", "Table 1", "Table 1（续）"]
    assert assets[0].caption == "Figure 1: A framework."
    assert assets[1].caption == "Figure 2: Curve."  # 滤掉了 env_id 噪声条目
    assert assets[0].section == "1 Introduction"
    assert assets[1].section == "3 Method"  # 章节归属跟随最近标题
    assert assets[2].is_table and not assets[0].is_table
    assert assets[0].img_path.exists()


def test_load_figure_assets_missing_file_or_caption_skipped(tmp_path):
    assets = load_figure_assets(_make_parse_dir(tmp_path))
    labels = [a.label for a in assets]
    assert "Figure 9" not in labels  # 图片缺失
    assert all(a.caption for a in assets)  # 无图注的 e.jpg 被跳过


def test_load_figure_assets_no_content_list(tmp_path):
    assert load_figure_assets(tmp_path) == []


def test_figure_inventory_text(tmp_path):
    assets = load_figure_assets(_make_parse_dir(tmp_path))
    text = figure_inventory_text(assets)
    assert "共 4 张" in text and "Figure 1" in text and "第 3 页" in text
    assert figure_inventory_text([]) .startswith("（本文未抽取到图表")


def test_plain_figure_caption_removes_latex_commands_without_paper_specific_rules():
    caption = r"Table X: $\beta=0.02$, $w \propto d$, and \mathrm{score}."
    cleaned = plain_figure_caption(caption)
    assert cleaned == "Table X: β=0.02, w ∝ d, and score."
    assert "$" not in cleaned and "\\" not in cleaned


def test_caption_html_or_superscript_prefix_does_not_hide_formal_table_label(tmp_path):
    parse_dir = tmp_path / "caption-prefix"
    (parse_dir / "images").mkdir(parents=True)
    Image.new("RGB", (120, 60), "white").save(parse_dir / "images" / "table.png")
    items = [{
        "type": "table", "img_path": "images/table.png", "page_idx": 2,
        "table_caption": [
            "<sup>weights</sup>Table 3: Classification of feedback types."
        ],
    }]
    (parse_dir / "paper_content_list.json").write_text(json.dumps(items), encoding="utf-8")

    assets = load_figure_assets(parse_dir)

    assert [asset.label for asset in assets] == ["Table 3"]
    assert assets[0].caption.startswith("Table 3:")


def test_duplicate_figure_labels_remain_unique(tmp_path):
    parse_dir = tmp_path / "dup"
    (parse_dir / "images").mkdir(parents=True)
    for name in ("a.jpg", "b.jpg"):
        (parse_dir / "images" / name).write_bytes(b"\xff\xd8\xff")
    items = [
        {"type": "image", "img_path": "images/a.jpg",
         "image_caption": ["Figure 1: left panel."], "page_idx": 1},
        {"type": "image", "img_path": "images/b.jpg",
         "image_caption": ["Figure 1: right panel."], "page_idx": 2},
    ]
    (parse_dir / "paper_content_list.json").write_text(
        json.dumps(items), encoding="utf-8"
    )
    assets = load_figure_assets(parse_dir)
    assert [asset.label for asset in assets] == ["Figure 1", "Figure 1（片段2）"]
    assert len({asset.label for asset in assets}) == 2


def test_consecutive_subpanels_are_merged_into_formal_parent_figure(tmp_path):
    parse_dir = tmp_path / "panels"
    (parse_dir / "images").mkdir(parents=True)
    for name, color in (("a.png", "red"), ("b.png", "green"), ("c.png", "blue")):
        Image.new("RGB", (100, 80), color).save(parse_dir / "images" / name)
    items = [
        {"type": "chart", "img_path": "images/a.png", "page_idx": 1,
         "bbox": [0, 0, 100, 80], "chart_caption": ["(a) First view"]},
        {"type": "chart", "img_path": "images/b.png", "page_idx": 1,
         "bbox": [100, 0, 200, 80], "chart_caption": ["(b) Second view"]},
        {"type": "chart", "img_path": "images/c.png", "page_idx": 1,
         "bbox": [200, 0, 300, 80],
         "chart_caption": ["(c) Third view", "Figure 4: Three complementary views."]},
    ]
    (parse_dir / "paper_content_list.json").write_text(json.dumps(items), encoding="utf-8")
    assets = load_figure_assets(parse_dir)
    assert [asset.label for asset in assets] == ["Figure 4"]
    assert assets[0].img_path.name == "panels-0-2.png"
    with Image.open(assets[0].img_path) as merged:
        assert merged.width == 300 and merged.height == 80


def test_multiple_formal_captions_are_reassigned_to_same_page_tables(tmp_path):
    parse_dir = tmp_path / "tables"
    (parse_dir / "images").mkdir(parents=True)
    for name in ("upper.png", "lower.png"):
        Image.new("RGB", (120, 60), "white").save(parse_dir / "images" / name)
    items = [
        {"type": "table", "img_path": "images/upper.png", "page_idx": 3,
         "bbox": [0, 0, 120, 60]},
        {"type": "table", "img_path": "images/lower.png", "page_idx": 3,
         "bbox": [0, 80, 120, 140],
         "table_caption": ["Table 4: Main metrics.", "Table 5: Per-task metrics."]},
    ]
    (parse_dir / "paper_content_list.json").write_text(json.dumps(items), encoding="utf-8")
    assets = load_figure_assets(parse_dir)
    assert [asset.label for asset in assets] == ["Table 4", "Table 5"]
    assert assets[0].img_path.name == "upper.png"
    assert assets[1].img_path.name == "lower.png"
