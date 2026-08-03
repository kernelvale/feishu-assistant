#!/usr/bin/env python3
"""精选库补建脚本（真实凭证，幂等）：只补建缺失的「精选库」表并写回 config.yaml。

- 已有 app（app_token 在 config.yaml）上执行：列出全部表 → 缺精选库才建 →
  table_id 写回 bitable.tables → 再列出验证表确实在 app 里
- 顺手建两个 grid 视图占位（「按日期倒序」「按状态分组」）：
  SDK 1.7 的视图属性只支持 filter/hidden_fields，排序与分组无法在 API 侧设置，
  需在多维表格 UI 里各点一下（脚本末尾打印手动步骤）

用法：
    uv run python scripts/init_picks_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from feishu_assistant.config import (  # noqa: E402
    ConfigError,
    load_settings,
    write_bitable_config,
)
from feishu_assistant.feishu import build_client  # noqa: E402
from feishu_assistant.feishu.bitable import (  # noqa: E402
    BitableError,
    ensure_tables,
    list_tables,
)

PICK_TABLE = "精选库"
VIEWS = ("按日期倒序", "按状态分组")


def create_views(client, app_token: str, table_id: str) -> None:
    """建 grid 视图占位；已存在/失败仅打印提示（视图非关键路径）。"""
    from lark_oapi.api.bitable.v1 import (
        CreateAppTableViewRequest,
        ListAppTableViewRequest,
        ReqView,
    )

    try:
        req = (
            ListAppTableViewRequest.builder()
            .app_token(app_token).table_id(table_id).page_size(100).build()
        )
        resp = client.bitable.v1.app_table_view.list(req)
        existing = {v.view_name for v in (resp.data.items or [])} if resp.success() else set()
        for name in VIEWS:
            if name in existing:
                print(f"    视图「{name}」已存在，跳过")
                continue
            create_req = (
                CreateAppTableViewRequest.builder()
                .app_token(app_token).table_id(table_id)
                .request_body(ReqView.builder().view_name(name).view_type("grid").build())
                .build()
            )
            create_resp = client.bitable.v1.app_table_view.create(create_req)
            if create_resp.success():
                print(f"    视图「{name}」已创建（grid）✓")
            else:
                print(f"    ⚠️  视图「{name}」创建失败：code={create_resp.code}, msg={create_resp.msg}")
    except Exception as exc:
        print(f"    ⚠️  视图创建异常（不影响表使用）：{exc}")


def main() -> int:
    settings = load_settings()
    if not settings.feishu_app_id or not settings.feishu_app_secret:
        print("❌ .env 缺少 FEISHU_APP_ID / FEISHU_APP_SECRET")
        return 1
    if not settings.bitable_app_token:
        print("❌ config.yaml 缺少 bitable.app_token，先运行 feishu-assistant init")
        return 1

    client = build_client(settings)
    app_token = settings.bitable_app_token

    print("1/4 列出 app 现有表……")
    try:
        before = list_tables(client, app_token)
    except BitableError as exc:
        print(f"❌ 列表失败：{exc}")
        return 1
    print(f"    现有 {len(before)} 张：{'、'.join(before)}")

    print("2/4 幂等补建缺失表……")
    try:
        tables, created = ensure_tables(client, app_token)
    except BitableError as exc:
        print(f"❌ 补建失败：{exc}")
        return 1
    if not created:
        print(f"    无缺失表（{PICK_TABLE}已存在：{tables.get(PICK_TABLE)}）")
    else:
        print(f"    新建：{'、'.join(created)} → {tables[created[0]]}")

    print("3/4 table_id 写回 config.yaml……")
    merged = {**settings.bitable_tables, **tables}
    if merged != settings.bitable_tables:
        try:
            write_bitable_config(app_token, merged)
        except ConfigError as exc:
            print(f"❌ 写回失败：{exc}\n    请手动把 {PICK_TABLE}: \"{tables[PICK_TABLE]}\" 写入 bitable.tables")
            return 1
        print("    已写回 ✓")
    else:
        print("    无变化，跳过")

    print("4/4 验证精选库在 app 中 + 建视图……")
    after = list_tables(client, app_token)
    pick_id = after.get(PICK_TABLE)
    if not pick_id:
        print("❌ 验证失败：精选库未出现在 app 里")
        return 1
    print(f"    ✅ 精选库已在 app 中：{pick_id}")
    create_views(client, app_token, pick_id)

    print("\n✅ 精选库补建完成。")
    print("手动一步（API 不支持设置排序/分组）：打开多维表格「精选库」→")
    print("  · 视图「按日期倒序」：排序 → 日期 → 降序")
    print("  · 视图「按状态分组」：分组 → 状态")
    return 0


if __name__ == "__main__":
    sys.exit(main())
