#!/usr/bin/env python3
"""P3 求职模块真实 API 冒烟脚本（用 .env 里的真实飞书凭证 + 真实大模型）。

验证路径：看板汇总读取 → light 模型看板操作解析 →（可选）写入一条测试记录
→ 更新 → 删除。默认只读，--write 才做写入/更新/删除全流程。

用法：
    uv run python scripts/career_smoke.py            # 只读冒烟（安全）
    uv run python scripts/career_smoke.py --write    # 含写入闭环（会建/删一条「冒烟测试公司」记录）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from feishu_assistant.config import load_settings  # noqa: E402
from feishu_assistant.feishu import build_client  # noqa: E402
from feishu_assistant.feishu.bitable import BitableDAO  # noqa: E402
from feishu_assistant.feishu.career_io import CareerIO  # noqa: E402
from feishu_assistant.kimi import KimiClient  # noqa: E402
from feishu_assistant.modules.career import (  # noqa: E402
    BOARD_TABLE,
    CareerService,
    find_board_records,
    parse_board_operation,
)

SMOKE_COMPANY = "冒烟测试公司"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="执行写入/更新/删除闭环")
    args = parser.parse_args()

    settings = load_settings()
    if not settings.feishu_app_id or not settings.feishu_app_secret:
        print("❌ .env 缺少 FEISHU_APP_ID / FEISHU_APP_SECRET")
        return 1
    if not settings.bitable_ready():
        print("❌ 多维表格未初始化，先运行 feishu-assistant init")
        return 1

    client = build_client(settings)
    dao = BitableDAO(client, settings.bitable_app_token, settings.bitable_tables)
    io = CareerIO(dao)
    kimi = KimiClient(settings)
    svc = CareerService(settings, kimi, lark_client=client, io=io)

    print("1/4 读取求职看板（翻页全量）……")
    records = io.list_all(BOARD_TABLE)
    print(f"    看板现有 {len(records)} 条记录 ✓")
    print("    汇总预览：")
    for line in svc.board_summary_text().splitlines()[:6]:
        print(f"      {line}")

    print("2/4 light 模型解析看板操作（真实调用，不写表）……")
    raw = kimi.chat(
        [{"role": "user", "content":
          "你是求职看板的操作解析器。今天是 2026-07-30。把「字节后端岗改一面过了」解析成 JSON："
          '{"action": "add|update|delete|query", "company": "", "position": "", "fields": {}}。只输出 JSON。'}],
        heavy=False,
    )
    op = parse_board_operation(raw)
    if op is None:
        print(f"    ❌ 解析失败，模型返回：{raw[:200]}")
        return 1
    print(f"    解析结果：action={op.action} company={op.company} fields={op.fields} ✓")

    if not args.write:
        print("3/4 跳过（--write 才做写入闭环）")
        print("4/4 跳过")
        print("\n✅ 只读冒烟通过：看板读取 + light 解析均正常")
        return 0

    print("3/4 写入一条冒烟记录并更新……")
    from feishu_assistant.modules.career import parse_date_text

    record_id = dao.add_record(BOARD_TABLE, {
        "公司": SMOKE_COMPANY, "岗位": "测试岗", "状态": "已投递",
        "投递时间": parse_date_text("今天"),
    })
    print(f"    已写入：{record_id} ✓")
    io.update(BOARD_TABLE, record_id, {"状态": "一面", "备注": "冒烟更新"})
    found = find_board_records(io.list_all(BOARD_TABLE), SMOKE_COMPANY)
    assert found and found[0].fields.get("状态") == "一面", "更新未生效"
    print("    已更新为「一面」✓")

    print("4/4 删除冒烟记录……")
    io.delete(BOARD_TABLE, record_id)
    assert not find_board_records(io.list_all(BOARD_TABLE), SMOKE_COMPANY), "删除未生效"
    print("    已删除 ✓")

    print("\n✅ career 冒烟全部通过：读取 → 解析 → 写入 → 更新 → 删除")
    return 0


if __name__ == "__main__":
    sys.exit(main())
