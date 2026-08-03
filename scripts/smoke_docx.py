#!/usr/bin/env python3
"""docx / drive 真实 API 冒烟脚本（用 .env 里的真实飞书凭证）。

验证路径：创建文档 → 写入 markdown（标题/加粗/列表/表格/链接）→ 权限授予
→ 打印 URL → 询问前自动删除文档（drive 删除接口）。

用法：
    uv run python scripts/smoke_docx.py           # 建文档→验证→删除
    uv run python scripts/smoke_docx.py --keep    # 保留文档（手动打开 URL 验证权限）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from feishu_assistant.config import load_settings  # noqa: E402
from feishu_assistant.feishu import build_client  # noqa: E402
from feishu_assistant.feishu.docx import DocxClient  # noqa: E402

SMOKE_MD = """\
# 冒烟测试文档

这是一段**加粗文本**和 `行内代码`，以及 [arXiv 链接](https://arxiv.org/abs/2603.03329)。

## 无序列表
- 第一项
- 第二项

## 有序列表
1. 第一条
2. 第二条

## 表格
| 指标 | 本文 | baseline |
|---|---|---|
| 合法移动率 | 100% | 62% |
| 覆盖游戏数 | 145 | — |
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true", help="不删除测试文档")
    args = parser.parse_args()

    settings = load_settings()
    if not settings.feishu_app_id or not settings.feishu_app_secret:
        print("❌ .env 缺少 FEISHU_APP_ID / FEISHU_APP_SECRET")
        return 1

    client = build_client(settings)
    docx = DocxClient(client)

    print("1/4 创建文档……")
    doc_id, url = docx.create_document("feishu-assistant 冒烟测试（可删除）")
    print(f"    document_id={doc_id}\n    url={url}")

    print("2/4 写入 markdown（标题/加粗/列表/表格/链接）……")
    docx.write_markdown(doc_id, SMOKE_MD)
    print("    写入完成")

    print("3/4 权限授予（组织内可读）……")
    granted = docx.grant_access(doc_id)
    print(f"    已生效：{granted}")

    if args.keep:
        print(f"4/4 已保留文档，请手动打开验证：{url}")
        return 0

    print("4/4 删除测试文档……")
    from lark_oapi.api.drive.v1 import DeleteFileRequest

    req = DeleteFileRequest.builder().file_token(doc_id).type("docx").build()
    resp = client.drive.v1.file.delete(req)
    if resp.success():
        print("    已删除 ✓")
    else:
        print(f"    删除失败（请手动删除 {url}）：code={resp.code}, msg={resp.msg}")
        return 1

    print("\n✅ docx 冒烟全部通过：创建 → 写入 → 授权 → 删除")
    return 0


if __name__ == "__main__":
    sys.exit(main())
