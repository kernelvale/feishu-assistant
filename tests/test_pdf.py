"""pdf.py 测试：MinerU 输出发现、降级链逻辑。全部离线（子进程与 pymupdf4llm 均 mock）。"""

from __future__ import annotations

import subprocess

import pytest

from feishu_assistant.pdf import PdfParseError, discover_markdown, parse_pdf

# ---- discover_markdown ----

def test_discover_markdown_by_stem(tmp_path):
    nested = tmp_path / "paper" / "auto"
    nested.mkdir(parents=True)
    md = nested / "paper.md"
    md.write_text("# 论文", encoding="utf-8")
    (nested / "other.md").write_text("x", encoding="utf-8")
    assert discover_markdown(tmp_path, "paper") == md


def test_discover_markdown_fallback_any_md(tmp_path):
    (tmp_path / "deep").mkdir()
    md = tmp_path / "deep" / "result.md"
    md.write_text("x", encoding="utf-8")
    assert discover_markdown(tmp_path, "nonexistent") == md


def test_discover_markdown_none(tmp_path):
    assert discover_markdown(tmp_path, "paper") is None


# ---- parse_pdf 降级链 ----

@pytest.fixture
def fake_pdf(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.7 fake content for testing" * 10)
    return pdf


def test_mineru_mps_success_no_fallback(fake_pdf, tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    md = out_dir / "paper" / "auto" / "paper.md"

    def fake_run(cmd, **kwargs):
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text("# MinerU 解析结果", encoding="utf-8")
        assert "mps" in kwargs["env"]["MINERU_DEVICE_MODE"]  # 默认先试 mps
        assert kwargs["env"]["MINERU_MODEL_SOURCE"] == "modelscope"
        return subprocess.CompletedProcess(cmd, 0, stdout="ok")

    monkeypatch.setattr("feishu_assistant.pdf.subprocess.run", fake_run)
    result = parse_pdf(fake_pdf, out_dir)
    assert result == md


def test_mps_fails_then_cpu_success(fake_pdf, tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    devices: list[str] = []

    def fake_run(cmd, **kwargs):
        device = kwargs["env"]["MINERU_DEVICE_MODE"]
        devices.append(device)
        if device == "mps":
            return subprocess.CompletedProcess(cmd, 1, stdout="mps boom")
        md = out_dir / "paper.md"
        md.write_text("# cpu 结果", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok")

    monkeypatch.setattr("feishu_assistant.pdf.subprocess.run", fake_run)
    result = parse_pdf(fake_pdf, out_dir)
    assert result.name == "paper.md"
    assert devices == ["mps", "cpu"]  # mps 失败降级 cpu


def test_mineru_both_fail_then_pymupdf(fake_pdf, tmp_path, monkeypatch):
    out_dir = tmp_path / "out"

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="boom")

    monkeypatch.setattr("feishu_assistant.pdf.subprocess.run", fake_run)
    monkeypatch.setattr(
        "feishu_assistant.pdf._parse_with_pymupdf",
        lambda pdf, out: out / "fallback.md",
    )
    result = parse_pdf(fake_pdf, out_dir)
    assert result.name == "fallback.md"


def test_all_backends_fail_raises_chinese(fake_pdf, tmp_path, monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="mineru boom")

    monkeypatch.setattr("feishu_assistant.pdf.subprocess.run", fake_run)

    def bad_pymupdf(pdf, out):
        raise PdfParseError("pymupdf boom")

    monkeypatch.setattr("feishu_assistant.pdf._parse_with_pymupdf", bad_pymupdf)
    with pytest.raises(PdfParseError, match="MinerU 与 PyMuPDF4LLM 均不可用"):
        parse_pdf(fake_pdf, tmp_path / "out")


def test_mineru_timeout_treated_as_failure(fake_pdf, tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_run(cmd, **kwargs):
        calls["n"] += 1
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr("feishu_assistant.pdf.subprocess.run", fake_run)
    monkeypatch.setattr("feishu_assistant.pdf._parse_with_pymupdf",
                        lambda pdf, out: out / "fallback.md")
    assert parse_pdf(fake_pdf, tmp_path / "out", timeout=1).name == "fallback.md"
    assert calls["n"] == 2  # mps、cpu 各试一次


def test_missing_pdf_raises(tmp_path):
    with pytest.raises(PdfParseError, match="不存在"):
        parse_pdf(tmp_path / "nope.pdf", tmp_path / "out")
