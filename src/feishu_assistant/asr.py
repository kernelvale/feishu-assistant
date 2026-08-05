"""语音转写（ASR）：mlx-whisper large-v3-turbo，惰性加载。

- mlx-whisper 是重型依赖（带模型权重下载），只在首次转写时导入，
  本模块导入本身轻量，可离线测试（transcriber 参数注入 fake）
- 模型默认 mlx-community/whisper-large-v3-turbo：Apple Silicon 上
  精度接近 large-v3、速度快数倍，适合本地面试录音
- 产出：TranscriptResult（text + language + duration + segments），
  调用方（career 复盘/语音面经）只依赖这个数据类，不碰 mlx 原生结构

依赖安装留待集成（uv add mlx-whisper）；未安装时抛 ASRError 并给安装指引。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("feishu_assistant.asr")

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


class ASRError(Exception):
    """语音转写失败，信息为中文（面向用户的提示用）。"""


@dataclass
class TranscriptResult:
    """一次转写的结果。segments 为带时间戳的分句（可能为空列表）。"""

    text: str
    language: str = ""
    duration: float = 0.0
    segments: list[dict[str, Any]] = field(default_factory=list)


# transcriber 签名：(audio_path, model) -> dict（mlx_whisper.transcribe 的返回结构）
Transcriber = Callable[[str, str], dict[str, Any]]


def _mlx_transcribe(audio_path: str, model: str) -> dict[str, Any]:
    """默认转写器：惰性导入 mlx_whisper（未安装时给中文安装指引）。"""
    try:
        import mlx_whisper  # 惰性导入，参考 pdf.py 对 mineru/pymupdf 的惰性模式
    except ImportError as exc:
        raise ASRError(
            "语音转写依赖 mlx-whisper 未安装。请在项目根目录运行：uv add mlx-whisper"
        ) from exc
    logger.info("mlx-whisper 转写开始：%s（model=%s）", audio_path, model)
    return mlx_whisper.transcribe(audio_path, path_or_hf_repo=model)


def _result_from_raw(raw: dict[str, Any]) -> TranscriptResult:
    """把 mlx_whisper.transcribe 的返回 dict 规整为 TranscriptResult。"""
    text = str(raw.get("text") or "").strip()
    segments = [dict(s) for s in (raw.get("segments") or []) if isinstance(s, dict)]
    duration = 0.0
    if segments:
        duration = max(float(s.get("end") or 0.0) for s in segments)
    return TranscriptResult(
        text=text,
        language=str(raw.get("language") or ""),
        duration=duration,
        segments=segments,
    )


def transcribe(
    audio_path: Path | str,
    *,
    model: str = DEFAULT_MODEL,
    transcriber: Transcriber | None = None,
) -> TranscriptResult:
    """转写音频文件，返回 TranscriptResult。

    transcriber 可注入（测试传 fake）；默认走 mlx-whisper 惰性导入。
    失败抛 ASRError（中文信息）。
    """
    path = Path(audio_path)
    if not path.exists():
        raise ASRError(f"音频文件不存在：{path}")

    fn = transcriber or _mlx_transcribe
    try:
        raw = fn(str(path), model)
    except ASRError:
        raise
    except Exception as exc:  # 模型下载失败/格式不支持等统一翻译
        raise ASRError(f"语音转写失败：{type(exc).__name__}：{exc}") from exc

    result = _result_from_raw(raw)
    if not result.text:
        raise ASRError("转写结果为空：可能是静音、音量过小或格式不受支持。")
    logger.info(
        "转写完成：%s → %d 字符，%.0f 秒，语言 %s",
        path.name, len(result.text), result.duration, result.language or "?",
    )
    return result
