"""bitable.py 测试：7 张表的 schema 定义与 build_field 转换。全部离线运行。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from feishu_assistant.feishu.bitable import (
    FIELD_TYPE_ATTACHMENT,
    FIELD_TYPE_CHECKBOX,
    FIELD_TYPE_DATETIME,
    FIELD_TYPE_MULTI_SELECT,
    FIELD_TYPE_SINGLE_SELECT,
    FIELD_TYPE_TEXT,
    FIELD_TYPE_URL,
    TABLE_SCHEMAS,
    BitableDAO,
    BitableError,
    FieldSpec,
    build_field,
)

EXPECTED_TABLES = {"论文库", "收藏库", "灵感库", "求职看板", "面经库", "简历库", "备忘库", "精选库"}

# 每张表的字段名与类型，严格对照需求定义
EXPECTED_FIELDS: dict[str, list[tuple[str, int]]] = {
    "论文库": [
        ("标题", 1), ("作者", 1), ("年份", 1), ("关键词", 4), ("TL;DR", 1),
        ("文档链接", 15), ("PDF链接", 15), ("arXivID", 1), ("精读时间", 5),
        ("方向", 3), ("来源", 1),
    ],
    "收藏库": [
        ("标题", 1), ("链接", 15), ("摘要", 1), ("标签", 4), ("来源", 1),
        ("收藏时间", 5), ("已回顾", 7),
    ],
    "灵感库": [("内容", 1), ("标签", 4), ("方向", 3), ("时间", 5)],
    "求职看板": [
        ("公司", 1), ("岗位", 1), ("求职ID", 1), ("岗位方向", 3), ("JD原文", 1),
        ("JD指纹", 1), ("JD分析文档", 15), ("推荐简历ID", 1), ("推荐简历", 1),
        ("定制简历草稿ID", 1), ("定制简历草稿", 1), ("定制简历状态", 1),
        ("实际投递简历ID", 1), ("实际投递简历", 1), ("简历匹配结论", 1),
        ("匹配证据", 1), ("投递版本锁定", 7), ("材料更新时间", 5), ("状态", 3),
        ("投递时间", 5), ("下次跟进", 5), ("内推人", 1), ("备注", 1),
    ],
    "面经库": [
        ("公司", 1), ("岗位", 1), ("求职ID", 1), ("关联简历ID", 1), ("关联简历", 1),
        ("轮次", 3), ("问题清单", 1), ("答砸点", 1),
        ("面试官反应", 1), ("日期", 5), ("来源", 3),
    ],
    "简历库": [
        ("版本名", 1), ("版本系列", 1), ("版本号", 2), ("上一版本", 1),
        ("当前版本", 7), ("版本变化", 1), ("版本类型", 1), ("基于版本ID", 1),
        ("分支版本号", 2), ("分支当前版本", 7), ("目标求职ID", 1), ("目标公司", 1), ("定制目标岗位", 1),
        ("JD指纹", 1), ("事实校验状态", 1), ("用户确认状态", 1),
        ("格式继承状态", 1), ("版面校验状态", 1), ("修改对照", 1), ("待确认项", 1),
        ("是否实际投递", 7), ("主方向", 1), ("目标岗位", 4),
        ("能力标签", 4), ("简历摘要", 1), ("识别依据", 1),
        ("解析状态", 1), ("内容指纹", 1), ("简历正文", 1),
        ("文件", 17), ("上传时间", 5),
    ],
    "备忘库": [("事项", 1), ("截止时间", 5), ("状态", 3), ("备注", 1), ("创建时间", 5)],
    # 精选库（P5）：标题置首（主字段必须文本），日期紧随其后
    "精选库": [
        ("标题", 1), ("日期", 5), ("链接", 15), ("arXivID", 1), ("TL;DR", 1),
        ("推荐理由", 1), ("状态", 3), ("来源", 1), ("方向", 3), ("关键词", 4),
    ],
}

EXPECTED_OPTIONS = {
    ("灵感库", "方向"): ("Agent Harness", "大模型开发", "其他"),
    ("求职看板", "岗位方向"): ("算法", "研究", "工程", "其他"),
    ("求职看板", "状态"): ("已投递", "笔试", "一面", "二面", "三面", "HR面", "offer", "已拒", "已挂"),
    ("面经库", "轮次"): ("笔试", "一面", "二面", "三面", "HR面"),
    ("面经库", "来源"): ("文字", "语音", "录音复盘"),
    ("备忘库", "状态"): ("待办", "已完成", "已取消"),
    ("论文库", "方向"): ("Agent Harness", "大模型开发", "Agent 系统", "其他"),
    ("精选库", "状态"): ("未读", "已读", "已精读"),
}

VALID_TYPE_CODES = {1, 2, 3, 4, 5, 7, 15, 17}


def test_seven_tables_with_expected_names():
    assert set(TABLE_SCHEMAS) == EXPECTED_TABLES


@pytest.mark.parametrize("table", sorted(EXPECTED_FIELDS))
def test_table_fields_match_spec(table):
    actual = [(f.name, f.type) for f in TABLE_SCHEMAS[table]]
    assert actual == EXPECTED_FIELDS[table]


def test_every_table_first_field_is_text():
    """主字段（第一个字段）必须是文本类型，否则建表流程不成立。"""
    for table, fields in TABLE_SCHEMAS.items():
        assert fields[0].type == FIELD_TYPE_TEXT, f"{table} 首字段不是文本"


# ---- schema 演进：幂等补建（精选库 P5）----

def test_missing_schemas_pure_logic():
    from feishu_assistant.feishu.bitable import missing_schemas

    # 已有 7 张 → 只缺精选库
    seven = {name: f"tbl{i}" for i, name in enumerate(
        ("论文库", "收藏库", "灵感库", "求职看板", "面经库", "简历库", "备忘库"))}
    missing = missing_schemas(seven)
    assert list(missing) == ["精选库"]
    # 8 张齐全 → 无缺失（幂等，重复执行不重建）
    assert missing_schemas({**seven, "精选库": "tblX"}) == {}
    # 空 app → 全缺
    assert len(missing_schemas({})) == len(TABLE_SCHEMAS)


def test_ensure_tables_migrates_fields_of_existing_tables(monkeypatch):
    """表已存在时也必须补字段，不能因为“不缺表”就静默跳过。"""
    import feishu_assistant.feishu.bitable as bitable

    schemas = {
        "简历库": [FieldSpec("版本名", FIELD_TYPE_TEXT), FieldSpec("目标岗位", FIELD_TYPE_MULTI_SELECT)],
        "精选库": [FieldSpec("标题", FIELD_TYPE_TEXT)],
    }
    monkeypatch.setattr(bitable, "list_tables", lambda client, token: {"简历库": "tbl_resume"})
    monkeypatch.setattr(bitable, "create_table", lambda client, token, name, fields: "tbl_curated")

    calls = []

    def fake_ensure_fields(client, token, table_id, fields):
        calls.append((table_id, [field.name for field in fields]))
        return ["目标岗位"]

    monkeypatch.setattr(bitable, "ensure_fields", fake_ensure_fields)

    tables, created_tables, created_fields = bitable.ensure_tables(object(), "app", schemas)

    assert tables == {"简历库": "tbl_resume", "精选库": "tbl_curated"}
    assert created_tables == ["精选库"]
    assert created_fields == {"简历库": ["目标岗位"]}
    assert calls == [("tbl_resume", ["版本名", "目标岗位"])]


def test_ensure_tables_is_idempotent_when_schema_is_current(monkeypatch):
    import feishu_assistant.feishu.bitable as bitable

    schemas = {"简历库": [FieldSpec("版本名", FIELD_TYPE_TEXT)]}
    monkeypatch.setattr(bitable, "list_tables", lambda client, token: {"简历库": "tbl_resume"})
    monkeypatch.setattr(bitable, "ensure_fields", lambda client, token, table_id, fields: [])

    tables, created_tables, created_fields = bitable.ensure_tables(object(), "app", schemas)

    assert tables == {"简历库": "tbl_resume"}
    assert created_tables == []
    assert created_fields == {}


def test_all_type_codes_valid():
    for table, fields in TABLE_SCHEMAS.items():
        for f in fields:
            assert f.type in VALID_TYPE_CODES, f"{table}.{f.name} 类型码非法: {f.type}"


@pytest.mark.parametrize(("table", "field_name"), sorted(EXPECTED_OPTIONS))
def test_select_field_options(table, field_name):
    spec = next(f for f in TABLE_SCHEMAS[table] if f.name == field_name)
    assert spec.type == FIELD_TYPE_SINGLE_SELECT
    assert spec.options == EXPECTED_OPTIONS[(table, field_name)]


def test_multi_select_fields_have_no_preset_options():
    """关键词/标签等多选字段不预置选项，由写入时自动创建。"""
    for table, fields in TABLE_SCHEMAS.items():
        for f in fields:
            if f.type == FIELD_TYPE_MULTI_SELECT:
                assert f.options == (), f"{table}.{f.name} 不应预置选项"


def test_date_fields_have_formatter():
    for table, fields in TABLE_SCHEMAS.items():
        for f in fields:
            if f.type == FIELD_TYPE_DATETIME:
                assert f.date_formatter, f"{table}.{f.name} 缺 date_formatter"


# ---- build_field 转换 ----

def test_build_field_text():
    field = build_field(FieldSpec("标题", FIELD_TYPE_TEXT))
    assert field.field_name == "标题"
    assert field.type == 1
    assert field.property is None


def test_build_field_single_select_with_options():
    field = build_field(FieldSpec("方向", FIELD_TYPE_SINGLE_SELECT, options=("A", "B")))
    assert field.type == 3
    assert [o.name for o in field.property.options] == ["A", "B"]


def test_build_field_date_with_formatter():
    field = build_field(FieldSpec("精读时间", FIELD_TYPE_DATETIME, date_formatter="yyyy/MM/dd"))
    assert field.type == 5
    assert field.property.date_formatter == "yyyy/MM/dd"


def test_build_field_special_types():
    assert build_field(FieldSpec("已回顾", FIELD_TYPE_CHECKBOX)).type == 7
    assert build_field(FieldSpec("链接", FIELD_TYPE_URL)).type == 15
    assert build_field(FieldSpec("文件", FIELD_TYPE_ATTACHMENT)).type == 17


class _ListResponse:
    def __init__(self, data):
        self.data = data
        self.code = 0
        self.msg = ""

    def success(self):
        return True

    def get_log_id(self):
        return "test"


class _PagedRecordEndpoint:
    def __init__(self, pages):
        self.pages = pages
        self.tokens = []

    def list(self, request):
        token = request.page_token or ""
        self.tokens.append(token)
        return _ListResponse(self.pages[token])


def _paged_dao(pages):
    endpoint = _PagedRecordEndpoint(pages)
    client = SimpleNamespace(
        bitable=SimpleNamespace(
            v1=SimpleNamespace(app_table_record=endpoint),
        ),
    )
    return BitableDAO(client, "app", {"论文库": "tbl"}), endpoint


def test_record_lookup_continues_to_later_pages():
    first = SimpleNamespace(record_id="r1", fields={"arXivID": "old"})
    target = SimpleNamespace(record_id="r2", fields={"arXivID": "target"})
    dao, endpoint = _paged_dao({
        "": SimpleNamespace(items=[first], has_more=True, page_token="page-2"),
        "page-2": SimpleNamespace(items=[target], has_more=False, page_token=None),
    })

    assert dao.find_record_by_field("论文库", "arXivID", "target") == "r2"
    assert endpoint.tokens == ["", "page-2"]


def test_record_pagination_rejects_invalid_continuation_token():
    dao, _ = _paged_dao({
        "": SimpleNamespace(items=[], has_more=True, page_token=""),
    })

    with pytest.raises(BitableError, match="page_token 无效"):
        dao.list_records("论文库")
