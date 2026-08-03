#!/usr/bin/env python3
"""论文库面板视图初始化（幂等，可重复执行）。

- 确保「方向」「来源」字段存在（schema 演进）
- 创建三个 grid 视图（已存在则跳过）：
  1. 📚 按方向分组
  2. 🕐 按精读时间倒序
  3. 📅 按年份分组

注意（实测结论）：飞书开放 API 的视图接口**不支持设置分组与排序**
（hierarchy_config 是给"层级视图"用的，要求自关联字段，对单选字段 patch 会
报 1254001）。所以脚本只负责把视图建好，分组/排序需要你在多维表格里对这三个
视图各点一次：📚/📅 设「分组」为 方向/年份，🕐 设「排序」为 精读时间倒序。
设置一次后永久生效。

用法：uv run python scripts/init_views.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lark_oapi.api.bitable.v1 import (  # noqa: E402
    CreateAppTableViewRequest,
    ListAppTableViewRequest,
    ReqView,
)

from feishu_assistant.config import load_settings  # noqa: E402
from feishu_assistant.feishu import build_client  # noqa: E402
from feishu_assistant.feishu.bitable import (  # noqa: E402
    FIELD_TYPE_SINGLE_SELECT,
    FIELD_TYPE_TEXT,
    PAPER_DIRECTIONS,
    FieldSpec,
    ensure_fields,
)

VIEWS = [
    ("📚 按方向分组", "在视图里点「分组」→ 选「方向」"),
    ("🕐 按精读时间倒序", "在视图里点「排序」→ 选「精读时间」倒序"),
    ("📅 按年份分组", "在视图里点「分组」→ 选「年份」"),
]


def check(resp, action):
    if not resp.success():
        print(f"❌ {action}：code={resp.code}, msg={resp.msg}, log_id={resp.get_log_id()}")
        sys.exit(1)
    return resp.data


def main() -> int:
    settings = load_settings()
    if not settings.bitable_ready():
        print("❌ 多维表格未初始化（config.yaml 缺 app_token/table_id），先 feishu-assistant init")
        return 1
    client = build_client(settings)
    app_token = settings.bitable_app_token
    table_id = settings.bitable_tables["论文库"]

    print("1/3 确保「方向」「来源」字段存在……")
    created = ensure_fields(client, app_token, table_id, [
        FieldSpec("方向", FIELD_TYPE_SINGLE_SELECT, options=PAPER_DIRECTIONS),
        FieldSpec("来源", FIELD_TYPE_TEXT),
    ])
    print(f"    新建字段：{created or '无（已存在）'}")

    print("2/3 读取现有视图……")
    views_data = check(client.bitable.v1.app_table_view.list(
        ListAppTableViewRequest.builder().app_token(app_token).table_id(table_id)
        .page_size(100).build()), "读取视图")
    existing = {v.view_name for v in (views_data.items or [])}
    print(f"    现有视图：{existing or '无'}")

    print("3/3 创建缺失视图……")
    for name, manual_hint in VIEWS:
        if name in existing:
            print(f"    跳过（已存在）：{name}")
            continue
        data = check(client.bitable.v1.app_table_view.create(
            CreateAppTableViewRequest.builder()
            .app_token(app_token).table_id(table_id)
            .request_body(ReqView.builder().view_name(name).view_type("grid").build())
            .build()), f"创建视图「{name}」")
        print(f"    已创建：{name}（{data.view.view_id}）")

    print("\n✅ 视图初始化完成。开放 API 不支持设置分组/排序，请对视图各做一次手动设置：")
    for name, manual_hint in VIEWS:
        print(f"  • {name}：{manual_hint}")
    print(f"\n面板入口：https://feishu.cn/base/{app_token}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
