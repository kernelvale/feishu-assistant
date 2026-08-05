"""asr.py 测试：转写契约（fake transcriber）、错误路径、音频识别。全部离线。"""

from __future__ import annotations

import sys

import pytest

from feishu_assistant.asr import ASRError, transcribe


def fake_transcriber(path: str, model: str) -> dict:
    return {
        "text": " 你好，我是面试官。先做个自我介绍吧。 ",
        "language": "zh",
        "segments": [
            {"start": 0.0, "end": 12.5, "text": "你好，我是面试官。"},
            {"start": 12.5, "end": 30.0, "text": "先做个自我介绍吧。"},
        ],
    }


def test_transcribe_happy_path(tmp_path):
    audio = tmp_path / "a.opus"
    audio.write_bytes(b"fake")
    result = transcribe(audio, transcriber=fake_transcriber)
    assert result.text == "你好，我是面试官。先做个自我介绍吧。"  # 首尾空白已规整
    assert result.language == "zh"
    assert result.duration == 30.0  # 取 segments 最大 end
    assert len(result.segments) == 2


def test_transcribe_no_segments_duration_zero(tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"fake")
    result = transcribe(audio, transcriber=lambda p, m: {"text": "内容", "language": "zh"})
    assert result.text == "内容" and result.duration == 0.0 and result.segments == []


def test_transcribe_file_missing(tmp_path):
    with pytest.raises(ASRError, match="不存在"):
        transcribe(tmp_path / "nope.opus", transcriber=fake_transcriber)


def test_transcribe_empty_text_raises(tmp_path):
    audio = tmp_path / "a.opus"
    audio.write_bytes(b"fake")
    with pytest.raises(ASRError, match="为空"):
        transcribe(audio, transcriber=lambda p, m: {"text": "  ", "segments": []})


def test_transcribe_transcriber_error_translated(tmp_path):
    audio = tmp_path / "a.opus"
    audio.write_bytes(b"fake")

    def boom(path: str, model: str) -> dict:
        raise RuntimeError("model download failed")

    with pytest.raises(ASRError, match="转写失败"):
        transcribe(audio, transcriber=boom)


def test_default_transcriber_without_mlx_gives_install_hint(tmp_path, monkeypatch):
    """未安装 mlx-whisper 时：ASRError 且带 uv add 指引（不真实导入重依赖）。"""
    audio = tmp_path / "a.opus"
    audio.write_bytes(b"fake")
    monkeypatch.setitem(sys.modules, "mlx_whisper", None)  # import 即 ImportError
    with pytest.raises(ASRError, match="uv add mlx-whisper"):
        transcribe(audio)  # 不传 transcriber → 走默认惰性导入路径
