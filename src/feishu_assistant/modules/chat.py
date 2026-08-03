"""自由对话模块：Kimi + prompts/assistant_system.md。

prompt 模板放在项目根目录 prompts/ 下（方便用户直接编辑），
通过包内相对项目根的路径加载。
"""

from __future__ import annotations

from pathlib import Path

from ..kimi import KimiClient, KimiError, is_heavy_task

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"
ASSISTANT_SYSTEM_PATH = PROMPTS_DIR / "assistant_system.md"

FALLBACK_SYSTEM_PROMPT = "你是一名个人助理，用简洁清晰的中文回答用户问题。"


def load_system_prompt(path: Path = ASSISTANT_SYSTEM_PATH) -> str:
    """读取系统提示词；文件缺失时退回内置兜底文案并打印警告。"""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return FALLBACK_SYSTEM_PROMPT
    return text or FALLBACK_SYSTEM_PROMPT


class ChatModule:
    """单轮自由对话（P1 不维护多轮上下文，P2 再按 chat_id 做会话记忆）。"""

    def __init__(self, kimi: KimiClient, system_prompt: str | None = None) -> None:
        self.kimi = kimi
        self.system_prompt = system_prompt if system_prompt is not None else load_system_prompt()

    def reply(self, user_text: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text},
        ]
        try:
            return self.kimi.chat(messages, heavy=is_heavy_task(user_text))
        except KimiError as exc:
            return f"调用大模型出错了：{exc}"
