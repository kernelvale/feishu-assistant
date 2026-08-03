"""飞书相关能力：消息收发（bot）、多维表格（bitable）、云文档（docx）、云盘（drive）。"""

from __future__ import annotations

import lark_oapi as lark

from ..config import Settings


def build_client(settings: Settings) -> lark.Client:
    """构造飞书 API 客户端（REST 调用用；WS 长连接由 bot.py 单独建立）。"""
    return (
        lark.Client.builder()
        .app_id(settings.feishu_app_id)
        .app_secret(settings.feishu_app_secret)
        .log_level(lark.LogLevel.INFO)
        .build()
    )
