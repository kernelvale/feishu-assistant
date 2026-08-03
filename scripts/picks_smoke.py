#!/usr/bin/env python3
"""精选库闭环真实冒烟（真实凭证）：写入 → 查重 → 状态回填 → 清理。

模拟"日报归档 → 精读回填"全链路（用 TEST 开头的假 arXivID，绝不碰真实数据）：
1. build_pick_row 写入一条 TEST 精选（状态=未读）
2. 同日同 arXivID 再归档一次 → existing_pick_keys 查重命中，不产生重复行
3. 模拟精读回填：find_record_by_field(arXivID) → update_record(状态=已精读)
4. 清理：删除测试行，验证表恢复干净

用法：
    uv run python scripts/picks_smoke.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from feishu_assistant.config import load_settings  # noqa: E402
from feishu_assistant.feishu import build_client  # noqa: E402
from feishu_assistant.feishu.bitable import BitableDAO  # noqa: E402
from feishu_assistant.feishu.career_io import CareerIO  # noqa: E402
from feishu_assistant.modules.digest import (  # noqa: E402
    PICK_TABLE,
    ArxivEntry,
    build_pick_row,
    existing_pick_keys,
)

TEST_ID = "TEST.00001"


def main() -> int:
    settings = load_settings()
    if not settings.bitable_ready() or PICK_TABLE not in settings.bitable_tables:
        print("❌ 精选库未补建，先运行 uv run python scripts/init_picks_table.py")
        return 1

    client = build_client(settings)
    dao = BitableDAO(client, settings.bitable_app_token, settings.bitable_tables)
    io = CareerIO(dao)

    entry = ArxivEntry(
        arxiv_id=TEST_ID, title="【冒烟】精选库闭环测试论文", abstract="fake",
        authors="Smoke", published="2026-07-31", url=f"https://arxiv.org/abs/{TEST_ID}",
    )
    item = {"tldr": "冒烟 TL;DR", "reason": "冒烟推荐理由"}

    created_ids: list[str] = []
    try:
        print("1/4 写入一条 TEST 精选（状态=未读）……")
        row = build_pick_row(entry, item)
        assert row["状态"] == "未读" and row["来源"] == "arXiv日报"
        rid = dao.add_record(PICK_TABLE, row)
        created_ids.append(rid)
        print(f"    已写入：{rid} ✓")

        print("2/4 同日同 arXivID 再归档 → 查重应命中……")
        keys = existing_pick_keys(io.list_all(PICK_TABLE))
        today = time.strftime("%Y-%m-%d")
        if (today, TEST_ID) in keys:
            print(f"    查重命中 ({today}, {TEST_ID})，跳过重复写入 ✓")
        else:
            print("    ❌ 查重未命中（键集合里没有今天的测试行）")
            return 1

        print("3/4 模拟精读回填：按 arXivID 找到并更新为「已精读」……")
        found = dao.find_record_by_field(PICK_TABLE, "arXivID", TEST_ID)
        if not found:
            print("    ❌ find_record_by_field 没找到测试行")
            return 1
        dao.update_record(PICK_TABLE, found, {"状态": "已精读"})
        # 读回验证
        still = [r for r in io.list_all(PICK_TABLE)
                 if r.record_id == found and r.fields.get("状态") == "已精读"]
        if still:
            print(f"    状态已回填为「已精读」✓（record={found}）")
        else:
            print("    ❌ 读回验证失败：状态不是已精读")
            return 1

        print("4/4 清理测试行……")
        for rid in created_ids:
            io.delete(PICK_TABLE, rid)
        remain = dao.find_record_by_field(PICK_TABLE, "arXivID", TEST_ID)
        if remain:
            print("    ❌ 清理失败：测试行仍在")
            return 1
        print("    测试行已删除，表已恢复 ✓")

        print("\n✅ 精选库闭环冒烟通过：写入 → 查重 → 回填 → 清理")
        return 0
    except Exception as exc:
        # 尽力清理，不把测试数据留在表里
        for rid in created_ids:
            try:
                io.delete(PICK_TABLE, rid)
            except Exception:
                pass
        print(f"\n❌ 冒烟失败（已尽力清理）：{type(exc).__name__}：{exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
