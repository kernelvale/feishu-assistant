"""docx V2 增补测试：长单元格表格保护、[[FIG:label]] 图片标记、图片块构建。"""

from __future__ import annotations

from feishu_assistant.feishu.docx import (
    BLOCK_TYPE_BULLET,
    TABLE_CELL_PROTECT_CHARS,
    _table_to_bullets,
    markdown_to_units,
)

LONG_CELL = "这是一段非常长的单元格内容，" * 20  # >150 字


def test_long_cell_table_becomes_bullets():
    md = f"| 方案 | 缺点 |\n|---|---|\n| 老路A | {LONG_CELL} |"
    units = markdown_to_units(md)
    kinds = [k for k, _ in units]
    assert "table" not in kinds  # 不生成表格
    blocks = units[0][1]
    assert blocks[0]["block_type"] == BLOCK_TYPE_BULLET
    text = str(blocks[0])
    assert "老路A" in text and "缺点" in text


def test_short_table_stays_table():
    md = "| 指标 | 值 |\n|---|---|\n| 合法率 | 100% |"
    units = markdown_to_units(md)
    assert units[0][0] == "table"


def test_protection_threshold():
    md = "| a | b |\n|---|---|\n| x | " + "y" * (TABLE_CELL_PROTECT_CHARS + 1) + " |"
    assert markdown_to_units(md)[0][0] == "blocks"
    md2 = "| a | b |\n|---|---|\n| x | " + "y" * 10 + " |"
    assert markdown_to_units(md2)[0][0] == "table"


def test_table_to_bullets_format():
    rows = [["方案", "缺点"], ["老路A", "太依赖人工"], ["老路B", "泛化差"]]
    blocks = _table_to_bullets(rows)
    assert len(blocks) == 2
    first = str(blocks[0])
    assert "老路A" in first and "缺点：太依赖人工" in first


def test_fig_marker_becomes_image_unit():
    md = "前文\n\n[[FIG:Figure 1]]\n\n后文"
    units = markdown_to_units(md)
    kinds = [k for k, _ in units]
    assert kinds == ["blocks", "image", "blocks"]
    assert units[1][1] == "Figure 1"


def test_math_marker_becomes_independent_image_unit():
    units = markdown_to_units("前文\n\n[[MATH:Equation 3]]\n\n公式解释")
    assert [kind for kind, _ in units] == ["blocks", "image", "blocks"]
    assert units[1][1] == "Equation 3"


def test_image_block_dict_compatible_with_sdk():
    from lark_oapi.api.docx.v1 import Block
    block = Block({"block_type": 27, "image": {"token": "tok", "width": 640, "height": 400, "align": 1}})
    assert block.block_type == 27 and block.image.token == "tok"
