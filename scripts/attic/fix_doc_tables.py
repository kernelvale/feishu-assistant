#!/usr/bin/env python3
"""存量修复：把指定飞书文档里所有表格的列宽修成「首列窄+其余均分」。

用法：uv run python scripts/fix_doc_tables.py <document_id>
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://open.feishu.cn/open-apis"


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def column_widths(col_size: int) -> list[int]:
    page = 700
    if col_size <= 1:
        return [page]
    first = 130 if col_size == 2 else (150 if col_size == 3 else 110)
    rest = max(80, (page - first) // (col_size - 1))
    return [first] + [rest] * (col_size - 1)


def main() -> int:
    if len(sys.argv) < 2:
        print("用法：fix_doc_tables.py <document_id>")
        return 1
    doc_id = sys.argv[1]
    env = load_env()
    token = httpx.post(
        f"{BASE}/auth/v3/tenant_access_token/internal",
        json={"app_id": env["FEISHU_APP_ID"], "app_secret": env["FEISHU_APP_SECRET"]},
        timeout=15,
    ).json()["tenant_access_token"]
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # 拉顶层块，找表格
    items: list[dict] = []
    page_token = ""
    while True:
        params = {"page_size": 500}
        if page_token:
            params["page_token"] = page_token
        r = httpx.get(f"{BASE}/docx/v1/documents/{doc_id}/blocks/{doc_id}/children",
                      headers=headers, params=params, timeout=20).json()
        if r.get("code") != 0:
            print(f"列块失败: {r}")
            return 1
        data = r["data"]
        items.extend(data.get("items", []))
        if not data.get("has_more"):
            break
        page_token = data.get("page_token", "")

    tables = [b for b in items if b.get("block_type") == 31]
    print(f"顶层块 {len(items)} 个，其中表格 {len(tables)} 张")

    for b in tables:
        block_id = b["block_id"]
        col_size = int((b.get("table") or {}).get("property", {}).get("column_size", 0))
        if col_size <= 0:
            continue
        widths = column_widths(col_size)
        for idx, w in enumerate(widths):
            resp = httpx.patch(
                f"{BASE}/docx/v1/documents/{doc_id}/blocks/{block_id}",
                headers=headers,
                json={"update_table_property": {"column_index": idx, "column_width": w}},
                timeout=20,
            ).json()
            if resp.get("code") != 0:
                print(f"  ✗ 表 {block_id} 列{idx} 失败: {resp.get('code')} {resp.get('msg')}")
                break
        else:
            print(f"  ✓ 表 {block_id}：{col_size} 列 → {widths}")
    print("完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
