"""飞书多维表格（Bitable）数据层。

P1 能力：
- TABLE_SCHEMAS：8 张业务表的字段定义（纯数据，可无凭证测试）
- create_app / create_tables / init_bitable：自动建 app 与 8 张表
- list_tables / ensure_tables / missing_schemas：schema 演进幂等迁移（补缺表与缺字段）
- BitableDAO：记录增查的薄封装（后续模块的数据入口）

字段类型码（lark-oapi / 开放平台文档）：文本=1、数字=2、单选=3、多选=4、
日期=5、复选框=7、超链接=15、附件=17。

注意：lark-oapi 1.7.x 的 AppTable 模型不支持建表时内联 fields，
因此建表流程为：建表（仅名称）→ 列出默认主字段 → 重命名主字段为
schema 第一个字段 → 逐个创建其余字段。各表第一个字段均为文本类型，
与主字段类型限制（必须文本）兼容。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator

import lark_oapi as lark
from lark_oapi.api.bitable.v1 import (
    AppTable,
    AppTableField,
    AppTableFieldProperty,
    AppTableFieldPropertyOption,
    AppTableRecord,
    CreateAppRequest,
    CreateAppTableFieldRequest,
    CreateAppTableRecordRequest,
    CreateAppTableRequest,
    CreateAppTableRequestBody,
    ListAppTableFieldRequest,
    ListAppTableRecordRequest,
    ReqApp,
    UpdateAppTableFieldRequest,
)

logger = logging.getLogger("feishu_assistant.bitable")

# ---- 字段类型码 ----
FIELD_TYPE_TEXT = 1
FIELD_TYPE_NUMBER = 2
FIELD_TYPE_SINGLE_SELECT = 3
FIELD_TYPE_MULTI_SELECT = 4
FIELD_TYPE_DATETIME = 5
FIELD_TYPE_CHECKBOX = 7
FIELD_TYPE_URL = 15
FIELD_TYPE_ATTACHMENT = 17

DEFAULT_DATE_FORMATTER = "yyyy/MM/dd"
DEFAULT_APP_NAME = "飞书个人助理"

JOB_DIRECTIONS = ("算法", "研究", "工程", "其他")
JOB_STATUS = ("已投递", "笔试", "一面", "二面", "三面", "HR面", "offer", "已拒", "已挂")
INTERVIEW_ROUNDS = ("笔试", "一面", "二面", "三面", "HR面")
PAPER_DIRECTIONS = ("Agent Harness", "大模型开发", "Agent 系统", "其他")


class BitableError(Exception):
    """多维表格操作失败，信息为中文（含飞书返回的 code/msg/log_id）。"""


@dataclass(frozen=True)
class FieldSpec:
    """一个字段的定义。options 仅单选/多选用；date_formatter 仅日期用。"""

    name: str
    type: int
    options: tuple[str, ...] = ()
    date_formatter: str | None = None


def _text(name: str) -> FieldSpec:
    return FieldSpec(name, FIELD_TYPE_TEXT)


def _url(name: str) -> FieldSpec:
    return FieldSpec(name, FIELD_TYPE_URL)


def _date(name: str) -> FieldSpec:
    return FieldSpec(name, FIELD_TYPE_DATETIME, date_formatter=DEFAULT_DATE_FORMATTER)


def _single(name: str, options: tuple[str, ...]) -> FieldSpec:
    return FieldSpec(name, FIELD_TYPE_SINGLE_SELECT, options=options)


def _multi(name: str) -> FieldSpec:
    # 多选不预置选项，写入记录时由飞书自动创建
    return FieldSpec(name, FIELD_TYPE_MULTI_SELECT)


# ---- 7 张业务表 schema（表名即 config.yaml 中 tables 的键）----
TABLE_SCHEMAS: dict[str, list[FieldSpec]] = {
    "论文库": [
        _text("标题"),
        _text("作者"),
        _text("年份"),
        _multi("关键词"),
        _text("TL;DR"),
        _url("文档链接"),
        _url("PDF链接"),
        _text("arXivID"),
        _date("精读时间"),
        _single("方向", PAPER_DIRECTIONS),
        _text("来源"),
    ],
    "收藏库": [
        _text("标题"),
        _url("链接"),
        _text("摘要"),
        _multi("标签"),
        _text("来源"),
        _date("收藏时间"),
        FieldSpec("已回顾", FIELD_TYPE_CHECKBOX),
    ],
    "灵感库": [
        _text("内容"),
        _multi("标签"),
        _single("方向", ("Agent Harness", "大模型开发", "其他")),
        _text("关联论文"),
        _text("论文ID"),
        _url("论文链接"),
        _date("时间"),
    ],
    "求职看板": [
        _text("公司"),
        _text("岗位"),
        _text("求职ID"),
        _single("岗位方向", JOB_DIRECTIONS),
        _text("JD原文"),
        _text("JD指纹"),
        _url("JD分析文档"),
        _text("推荐简历ID"),
        _text("推荐简历"),
        _text("定制简历草稿ID"),
        _text("定制简历草稿"),
        _text("定制简历状态"),
        _text("实际投递简历ID"),
        _text("实际投递简历"),
        _text("简历匹配结论"),
        _text("匹配证据"),
        FieldSpec("投递版本锁定", FIELD_TYPE_CHECKBOX),
        _date("材料更新时间"),
        _single("状态", JOB_STATUS),
        _date("投递时间"),
        _date("下次跟进"),
        _text("内推人"),
        _text("备注"),
    ],
    "面经库": [
        _text("公司"),
        _text("岗位"),
        _text("求职ID"),
        _text("关联简历ID"),
        _text("关联简历"),
        _single("轮次", INTERVIEW_ROUNDS),
        _text("问题清单"),
        _text("答砸点"),
        _text("面试官反应"),
        _date("日期"),
        _single("来源", ("文字", "语音", "录音复盘")),
    ],
    "简历库": [
        _text("版本名"),
        _text("版本系列"),
        FieldSpec("版本号", FIELD_TYPE_NUMBER),
        _text("上一版本"),
        FieldSpec("当前版本", FIELD_TYPE_CHECKBOX),
        _text("版本变化"),
        _text("版本类型"),
        _text("基于版本ID"),
        FieldSpec("分支版本号", FIELD_TYPE_NUMBER),
        FieldSpec("分支当前版本", FIELD_TYPE_CHECKBOX),
        _text("目标求职ID"),
        _text("目标公司"),
        _text("定制目标岗位"),
        _text("JD指纹"),
        _text("事实校验状态"),
        _text("用户确认状态"),
        _text("格式继承状态"),
        _text("版面校验状态"),
        _text("修改对照"),
        _text("待确认项"),
        FieldSpec("是否实际投递", FIELD_TYPE_CHECKBOX),
        _text("主方向"),
        _multi("目标岗位"),
        _multi("能力标签"),
        _text("简历摘要"),
        _text("识别依据"),
        _text("解析状态"),
        _text("内容指纹"),
        _text("简历正文"),
        FieldSpec("文件", FIELD_TYPE_ATTACHMENT),
        _date("上传时间"),
    ],
    "备忘库": [
        _text("事项"),
        _date("截止时间"),
        _single("状态", ("待办", "已完成", "已取消")),
        _text("备注"),
        _date("创建时间"),
    ],
    # 精选库（P5）：arXiv 日报每日精选沉淀，与论文库形成状态闭环
    # （未读 → 已读 → 已精读）。标题置首：主字段必须是文本类型。
    "精选库": [
        _text("标题"),
        _date("日期"),
        _url("链接"),
        _text("arXivID"),
        _text("TL;DR"),
        _text("推荐理由"),
        _single("状态", ("未读", "已读", "已精读")),
        _text("来源"),
        _single("方向", PAPER_DIRECTIONS + ("未分类",)),
        _multi("关键词"),
    ],
}


def build_field(spec: FieldSpec) -> AppTableField:
    """把 FieldSpec 转成 SDK 的 AppTableField（纯函数，可离线测试）。"""
    builder = AppTableField.builder().field_name(spec.name).type(spec.type)
    prop_builder = AppTableFieldProperty.builder()
    has_prop = False
    if spec.options:
        prop_builder.options(
            [AppTableFieldPropertyOption.builder().name(o).build() for o in spec.options]
        )
        has_prop = True
    if spec.date_formatter:
        prop_builder.date_formatter(spec.date_formatter)
        has_prop = True
    if has_prop:
        builder.property(prop_builder.build())
    return builder.build()


def _check(resp: Any, action: str) -> Any:
    """统一检查飞书响应，失败抛 BitableError（中文 + log_id 方便排查）。"""
    if not resp.success():
        raise BitableError(
            f"{action}失败：code={resp.code}, msg={resp.msg}, "
            f"log_id={resp.get_log_id()}（可拿 log_id 到开放平台排查）"
        )
    return resp.data


def create_app(client: lark.Client, name: str = DEFAULT_APP_NAME) -> str:
    """创建一个 bitable app，返回 app_token。"""
    req = CreateAppRequest.builder().request_body(ReqApp.builder().name(name).build()).build()
    data = _check(client.bitable.v1.app.create(req), f"创建多维表格「{name}」")
    return data.app.app_token


def create_table(client: lark.Client, app_token: str, name: str, fields: list[FieldSpec]) -> str:
    """在 app 下建一张表并按 schema 建字段，返回 table_id。"""
    if not fields:
        raise BitableError(f"表「{name}」的字段定义为空。")
    if fields[0].type != FIELD_TYPE_TEXT:
        raise BitableError(f"表「{name}」的第一个字段必须是文本类型（主字段限制）。")

    # 1. 建表（SDK 1.7.x 不支持内联 fields）
    req = (
        CreateAppTableRequest.builder()
        .app_token(app_token)
        .request_body(
            CreateAppTableRequestBody.builder().table(AppTable.builder().name(name).build()).build()
        )
        .build()
    )
    data = _check(client.bitable.v1.app_table.create(req), f"创建表「{name}」")
    table_id = data.table_id

    # 2. 找到自动生成的默认主字段
    list_req = (
        ListAppTableFieldRequest.builder()
        .app_token(app_token)
        .table_id(table_id)
        .page_size(100)
        .build()
    )
    list_data = _check(client.bitable.v1.app_table_field.list(list_req), f"读取表「{name}」字段")
    items = list_data.items or []
    primary = next((f for f in items if f.is_primary), items[0] if items else None)
    if primary is None:
        raise BitableError(f"表「{name}」创建后没有任何字段，异常。")

    # 3. 主字段重命名为 schema 第一个字段（主字段只能改名字，类型固定文本）
    update_req = (
        UpdateAppTableFieldRequest.builder()
        .app_token(app_token)
        .table_id(table_id)
        .field_id(primary.field_id)
        .request_body(build_field(fields[0]))
        .build()
    )
    _check(client.bitable.v1.app_table_field.update(update_req), f"重命名表「{name}」主字段")

    # 4. 逐个创建其余字段
    for spec in fields[1:]:
        field_req = (
            CreateAppTableFieldRequest.builder()
            .app_token(app_token)
            .table_id(table_id)
            .request_body(build_field(spec))
            .build()
        )
        _check(
            client.bitable.v1.app_table_field.create(field_req),
            f"表「{name}」创建字段「{spec.name}」",
        )
    return table_id


def ensure_fields(
    client: lark.Client, app_token: str, table_id: str, specs: list[FieldSpec]
) -> list[str]:
    """给已有表补字段（schema 演进用）：列出现有字段，缺的才建。返回新建字段名列表。"""
    list_req = (
        ListAppTableFieldRequest.builder()
        .app_token(app_token)
        .table_id(table_id)
        .page_size(100)
        .build()
    )
    data = _check(client.bitable.v1.app_table_field.list(list_req), "读取字段列表")
    existing = {f.field_name for f in (data.items or [])}
    created: list[str] = []
    for spec in specs:
        if spec.name in existing:
            continue
        field_req = (
            CreateAppTableFieldRequest.builder()
            .app_token(app_token)
            .table_id(table_id)
            .request_body(build_field(spec))
            .build()
        )
        _check(client.bitable.v1.app_table_field.create(field_req), f"补充字段「{spec.name}」")
        created.append(spec.name)
    return created


def create_tables(
    client: lark.Client, app_token: str, schemas: dict[str, list[FieldSpec]] | None = None
) -> dict[str, str]:
    """按 schema 批量建表，返回 {表名: table_id}。任一表失败即抛错（已建的表保留）。"""
    schemas = schemas or TABLE_SCHEMAS
    tables: dict[str, str] = {}
    for name, fields in schemas.items():
        tables[name] = create_table(client, app_token, name, fields)
    return tables


def init_bitable(client: lark.Client, app_name: str = DEFAULT_APP_NAME) -> tuple[str, dict[str, str]]:
    """一键初始化：建 app + 建 8 张表，返回 (app_token, {表名: table_id})。"""
    app_token = create_app(client, app_name)
    tables = create_tables(client, app_token)
    return app_token, tables


def missing_schemas(
    existing: dict[str, str], schemas: dict[str, list[FieldSpec]] | None = None
) -> dict[str, list[FieldSpec]]:
    """已有表名 → 还缺的 schema（纯函数，init 幂等补建的决策部分）。"""
    schemas = schemas or TABLE_SCHEMAS
    return {name: fields for name, fields in schemas.items() if name not in existing}


def list_tables(client: lark.Client, app_token: str, max_pages: int = 10) -> dict[str, str]:
    """列出 app 下全部表，返回 {表名: table_id}。"""
    from lark_oapi.api.bitable.v1 import ListAppTableRequest

    tables: dict[str, str] = {}
    page_token = ""
    for _ in range(max_pages):
        builder = ListAppTableRequest.builder().app_token(app_token).page_size(100)
        if page_token:
            builder.page_token(page_token)
        data = _check(client.bitable.v1.app_table.list(builder.build()), "列出多维表格")
        for t in data.items or []:
            tables[t.name] = t.table_id
        if not data.has_more or not data.page_token:
            break
        page_token = data.page_token
    return tables


def ensure_tables(
    client: lark.Client,
    app_token: str,
    schemas: dict[str, list[FieldSpec]] | None = None,
) -> tuple[dict[str, str], list[str], dict[str, list[str]]]:
    """幂等迁移业务表 schema：补缺表，也给已有表补缺字段。

    返回 ``(全量 {表名: table_id}, 新建表名列表, {表名: 新建字段列表})``。
    现有字段和值不会被修改；任何一步失败即抛错（已经完成的创建会保留）。
    """
    schemas = schemas or TABLE_SCHEMAS
    existing = list_tables(client, app_token)
    created_tables: list[str] = []
    created_fields: dict[str, list[str]] = {}
    for name, fields in missing_schemas(existing, schemas).items():
        existing[name] = create_table(client, app_token, name, fields)
        created_tables.append(name)
        logger.info("补建表「%s」：%s", name, existing[name])

    # 对迁移前已经存在的表逐一核对字段。刚创建的表已包含完整 schema，
    # 不再重复请求字段列表，避免无意义的网络调用。
    for name, fields in schemas.items():
        if name in created_tables or name not in existing:
            continue
        added = ensure_fields(client, app_token, existing[name], fields)
        if added:
            created_fields[name] = added
            logger.info("为表「%s」补建字段：%s", name, "、".join(added))

    # 只回传 schema 内的表（app 里可能还有「数据表」等无关表，不能写回 config）
    return (
        {name: existing[name] for name in schemas if name in existing},
        created_tables,
        created_fields,
    )


def _field_text(value: Any) -> str:
    """bitable 文本字段值兼容：字符串或段列表（[{text: ...}, ...]）。"""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "".join(str(seg.get("text", "")) for seg in value if isinstance(seg, dict)).strip()
    return ""


class BitableDAO:
    """记录级读写的薄封装。table_name 用中文表名（如「论文库」）。"""

    def __init__(self, client: lark.Client, app_token: str, tables: dict[str, str]) -> None:
        if not app_token:
            raise BitableError("缺少 bitable app_token，请先运行 feishu-assistant init。")
        self.client = client
        self.app_token = app_token
        self.tables = tables

    def _table_id(self, table_name: str) -> str:
        table_id = self.tables.get(table_name, "")
        if not table_id:
            raise BitableError(f"找不到表「{table_name}」的 table_id，请先运行 feishu-assistant init。")
        return table_id

    def add_record(self, table_name: str, fields: dict[str, Any]) -> str:
        """插入一条记录，返回 record_id。fields 的键为中文段名，值需符合字段类型。"""
        req = (
            CreateAppTableRecordRequest.builder()
            .app_token(self.app_token)
            .table_id(self._table_id(table_name))
            .request_body(AppTableRecord.builder().fields(fields).build())
            .build()
        )
        data = _check(self.client.bitable.v1.app_table_record.create(req), f"表「{table_name}」写入记录")
        return data.record.record_id

    def find_record_by_field(self, table_name: str, field_name: str, value: str) -> str | None:
        """按文本字段值查全部记录（如 arXivID 查重），返回 record_id 或 None。"""
        for record in self.iter_records(table_name, page_size=100):
            field_value = (record.fields or {}).get(field_name)
            if _field_text(field_value) == value:
                return record.record_id
        return None

    def find_record_fields(self, table_name: str, field_name: str, value: str) -> dict | None:
        """按文本字段值查记录，返回整条记录的 fields（版本演进/历史串接要读旧值）。"""
        for record in self.iter_records(table_name, page_size=100):
            field_value = (record.fields or {}).get(field_name)
            if _field_text(field_value) == value:
                return record.fields or {}
        return None

    def update_record(self, table_name: str, record_id: str, fields: dict[str, Any]) -> None:
        """更新一条记录的部分字段（如复读时刷新精读时间）。"""
        from lark_oapi.api.bitable.v1 import UpdateAppTableRecordRequest

        req = (
            UpdateAppTableRecordRequest.builder()
            .app_token(self.app_token)
            .table_id(self._table_id(table_name))
            .record_id(record_id)
            .request_body(AppTableRecord.builder().fields(fields).build())
            .build()
        )
        _check(
            self.client.bitable.v1.app_table_record.update(req),
            f"表「{table_name}」更新记录 {record_id}",
        )

    def iter_records(
        self, table_name: str, page_size: int = 100,
    ) -> Iterator[AppTableRecord]:
        """逐页遍历整张表；异常分页状态显式失败，禁止静默返回残缺数据。"""
        if not 1 <= page_size <= 500:
            raise ValueError("page_size 必须在 1..500 之间")
        table_id = self._table_id(table_name)
        page_token = ""
        seen_tokens: set[str] = set()
        while True:
            builder = (
                ListAppTableRecordRequest.builder()
                .app_token(self.app_token)
                .table_id(table_id)
                .page_size(page_size)
            )
            if page_token:
                builder.page_token(page_token)
            data = _check(
                self.client.bitable.v1.app_table_record.list(builder.build()),
                f"表「{table_name}」读取记录",
            )
            yield from data.items or []
            if not data.has_more:
                return
            next_token = str(data.page_token or "")
            if not next_token or next_token in seen_tokens:
                raise BitableError(
                    f"表「{table_name}」分页状态异常：has_more=true 但 page_token 无效"
                )
            seen_tokens.add(next_token)
            page_token = next_token

    def list_records(self, table_name: str, page_size: int = 100) -> list[AppTableRecord]:
        """读取整张表；page_size 仅控制单次请求批量，不再代表总记录上限。"""
        return list(self.iter_records(table_name, page_size=page_size))
