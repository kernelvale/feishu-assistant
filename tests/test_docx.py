"""feishu/docx.py 测试：markdown→block 转换、表格 descendant 结构、行内元素解析。全部离线。"""

from __future__ import annotations

import pytest

from feishu_assistant.feishu.docx import (
    BLOCK_TYPE_BULLET,
    BLOCK_TYPE_ORDERED,
    BLOCK_TYPE_TABLE,
    BLOCK_TYPE_TABLE_CELL,
    BLOCK_TYPE_TEXT,
    DocxClient,
    DocxError,
    build_table_descendants,
    make_text_elements,
    markdown_to_units,
    split_table_row,
)

# ---- 行内元素 ----

def test_text_elements_plain():
    els = make_text_elements("普通文本")
    assert els == [{"text_run": {"content": "普通文本"}}]


def test_text_elements_bold():
    els = make_text_elements("前面**加粗内容**后面")
    assert els[0]["text_run"]["content"] == "前面"
    assert els[1]["text_run"]["content"] == "加粗内容"
    assert els[1]["text_run"]["text_element_style"]["bold"] is True
    assert els[2]["text_run"]["content"] == "后面"


def test_text_elements_inline_code_and_link():
    els = make_text_elements("`code` 和 [标题](https://arxiv.org/abs/1234.56789)")
    assert els[0]["text_run"]["text_element_style"]["inline_code"] is True
    link_el = els[2]["text_run"]
    assert link_el["content"] == "标题"
    assert "arxiv.org" in link_el["text_element_style"]["link"]["url"]


def test_text_elements_empty():
    assert make_text_elements("") == [{"text_run": {"content": ""}}]


# ---- 表格行解析 ----

def test_split_table_row():
    assert split_table_row("| 指标 | 本文 | baseline |") == ["指标", "本文", "baseline"]
    assert split_table_row("|a|b|") == ["a", "b"]


# ---- markdown_to_units ----

def test_headings_and_paragraph():
    units = markdown_to_units("# 一级\n\n正文段落\n\n## 二级")
    blocks = units[0][1]
    assert blocks[0]["block_type"] == 3 and "heading1" in blocks[0]
    assert blocks[1]["block_type"] == BLOCK_TYPE_TEXT
    assert blocks[2]["block_type"] == 4 and "heading2" in blocks[2]


def test_bullet_and_ordered():
    units = markdown_to_units("- 项目一\n- 项目二\n1. 第一\n2. 第二")
    blocks = units[0][1]
    assert blocks[0]["block_type"] == BLOCK_TYPE_BULLET
    assert blocks[0]["bullet"]["elements"][0]["text_run"]["content"] == "项目一"
    assert blocks[2]["block_type"] == BLOCK_TYPE_ORDERED
    assert blocks[3]["ordered"]["elements"][0]["text_run"]["content"] == "第二"


def test_nested_lists_preserve_indentation_level():
    units = markdown_to_units(
        "- 阶段一\n  - 输入\n    - 约束\n  1. 执行动作\n- 阶段二"
    )
    blocks = units[0][1]

    assert "style" not in blocks[0]["bullet"]
    assert blocks[1]["bullet"]["style"]["indentation_level"] == "1"
    assert blocks[2]["bullet"]["style"]["indentation_level"] == "2"
    assert blocks[3]["ordered"]["style"]["indentation_level"] == "1"
    assert "style" not in blocks[4]["bullet"]


def test_table_becomes_separate_unit():
    md = "前文\n\n| 指标 | 本文 |\n|---|---|\n| 合法率 | 100% |\n\n后文"
    units = markdown_to_units(md)
    kinds = [k for k, _ in units]
    assert kinds == ["blocks", "table", "blocks"]
    rows = units[1][1]
    assert rows == [["指标", "本文"], ["合法率", "100%"]]  # 分隔行已丢弃


def test_table_pads_short_rows():
    md = "| a | b | c |\n|---|---|---|\n| 1 | 2 |"
    units = markdown_to_units(md)
    rows = units[0][1]
    assert rows[1] == ["1", "2", ""]


def test_code_fence_becomes_code_block():
    units = markdown_to_units("```python\ndef f():\n    return 1\n```\n正文")
    code_unit, text_unit = units[0], units[1]
    block = code_unit[1][0]
    assert block["block_type"] == 14  # 真正的代码块
    assert block["code"]["style"]["language"] == 49  # python
    assert block["code"]["elements"][0]["text_run"]["content"] == "def f():\n    return 1"
    assert text_unit[1][0]["block_type"] == BLOCK_TYPE_TEXT


def test_code_fence_unknown_lang_falls_back_plaintext():
    units = markdown_to_units("```weirdlang\nx=1\n```")
    assert units[0][1][0]["code"]["style"]["language"] == 1


def test_strict_images_rejects_missing_mapping():
    client = DocxClient(client=None)
    with pytest.raises(DocxError, match="缺少本地文件"):
        client.write_markdown("doc", "[[FIG:Figure 1]]", strict_images=True)


def test_strict_images_rejects_upload_failure(tmp_path, monkeypatch):
    image = tmp_path / "f.jpg"
    image.write_bytes(b"not-a-real-image")
    client = DocxClient(client=None)

    def fail_upload(*args, **kwargs):
        raise DocxError("upload failed")

    monkeypatch.setattr(client, "add_image", fail_upload)
    with pytest.raises(DocxError, match="Figure 1.*上传失败"):
        client.write_markdown(
            "doc", "[[FIG:Figure 1]]", images={"Figure 1": image}, strict_images=True,
        )


def test_standalone_bold_stays_bold_paragraph_and_out_of_toc():
    units = markdown_to_units("**痛点**\n正文带 **加粗** 不升级")
    blocks = units[0][1]
    assert blocks[0]["block_type"] == BLOCK_TYPE_TEXT and "text" in blocks[0]
    first_run = blocks[0]["text"]["elements"][0]["text_run"]
    assert first_run["content"] == "痛点"
    assert first_run["text_element_style"]["bold"] is True
    assert blocks[1]["block_type"] == BLOCK_TYPE_TEXT  # 行内加粗保持段落


def test_quote_and_divider():
    units = markdown_to_units("> 引述一句\n\n---\n\n后文")
    blocks = units[0][1]
    assert blocks[0]["block_type"] == 15
    assert blocks[0]["quote"]["elements"][0]["text_run"]["content"] == "引述一句"
    assert blocks[1]["block_type"] == 22 and "divider" in blocks[1]
    assert blocks[2]["block_type"] == BLOCK_TYPE_TEXT


def test_empty_markdown():
    assert markdown_to_units("") == []
    assert markdown_to_units("\n\n  \n") == []


# ---- 表格 descendant 结构 ----

def test_build_table_descendants_structure():
    rows = [["指标", "本文", "baseline"], ["合法率", "100%", "62%"]]
    children_id, descendants = build_table_descendants(rows, id_prefix="t0")
    assert children_id == ["t0"]

    table = descendants[0]
    assert table["block_id"] == "t0"
    assert table["block_type"] == BLOCK_TYPE_TABLE
    assert table["table"]["property"]["row_size"] == 2
    assert table["table"]["property"]["column_size"] == 3
    assert table["table"]["property"]["header_row"] is True
    assert len(table["children"]) == 6  # 2 行 × 3 列

    # 每个 cell 后面紧跟它的文本块，且 cell 内容正确
    cells = {d["block_id"]: d for d in descendants[1:]}
    first_cell = cells["t0_c0"]
    assert first_cell["block_type"] == BLOCK_TYPE_TABLE_CELL
    text_block = cells[first_cell["children"][0]]
    assert text_block["text"]["elements"][0]["text_run"]["content"] == "指标"
    # 第二行第一列 = 合法率
    cell3 = cells["t0_c3"]
    assert cells[cell3["children"][0]]["text"]["elements"][0]["text_run"]["content"] == "合法率"


def test_table_descendants_ids_unique_across_tables():
    _, d1 = build_table_descendants([["a"]], id_prefix="tbl0")
    _, d2 = build_table_descendants([["b"]], id_prefix="tbl40")
    ids1 = {d["block_id"] for d in d1}
    ids2 = {d["block_id"] for d in d2}
    assert ids1.isdisjoint(ids2)  # 同一文档多次写入不冲突


# ---- dict-init 兼容性（SDK Block 能接受转换结果）----

def test_blocks_compatible_with_sdk_model():
    from lark_oapi.api.docx.v1 import Block
    units = markdown_to_units("# 标题\n正文**加粗**\n\n| a |\n|---|\n| b |")
    for kind, payload in units:
        if kind == "blocks":
            for b in payload:
                block = Block(b)
                assert block.block_type is not None
        else:
            children_id, descendants = build_table_descendants(payload)
            for d in descendants:
                Block(d)  # 不抛异常即可
