#!/usr/bin/env python3
"""向用户飞书私聊推送通知（完成通知/验收清单用）。

用法：uv run python scripts/notify_user.py "要推送的文本"
凭证走 .env；chat_id 从 logs/bot.log 里最近一次用户消息自动提取。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def find_chat_id() -> str:
    log = (ROOT / "logs" / "bot.log").read_text(errors="ignore")
    matches = re.findall(r"chat=(oc_[0-9a-f]{32})", log)
    if not matches:
        raise SystemExit("未能在日志中找到用户 chat_id")
    return matches[-1]


def main() -> int:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print("用法：notify_user.py '消息文本'")
        return 1
    text = sys.argv[1]
    env = load_env()
    token = httpx.post(
        "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
        json={"app_id": env["FEISHU_APP_ID"], "app_secret": env["FEISHU_APP_SECRET"]},
        timeout=15,
    ).json()["tenant_access_token"]
    chat_id = find_chat_id()
    r = httpx.post(
        "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "receive_id": chat_id,
            "msg_type": "text",
            "content": json.dumps({"text": text}, ensure_ascii=False),
        },
        timeout=15,
    ).json()
    if r.get("code") != 0:
        print(f"发送失败: {r}")
        return 1
    print(f"已推送到 chat={chat_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
