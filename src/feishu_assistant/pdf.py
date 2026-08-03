"""PDF 解析：MinerU 为主，PyMuPDF4LLM 兜底。

- MinerU（CLI 子进程）：模型权重走 modelscope 镜像（MINERU_MODEL_SOURCE），
  macOS arm64 设备模式优先 mps（MINERU_DEVICE_MODE），报错自动降级 cpu 重试
- PyMuPDF4LLM：MinerU 失败/超时后的纯文本兜底，保证管线不中断
- 产出：结构化 markdown 写到 data/papers/{arxiv_id}/ 下，返回路径

mineru / pymupdf4llm 均为惰性导入/子进程调用，本模块导入本身轻量，可离线测试。
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("feishu_assistant.pdf")

DEFAULT_PARSE_TIMEOUT = 900  # 单篇解析超时（秒）；首次运行含模型下载会更久
MINERU_BACKEND = "pipeline"  # 经典管线后端，macOS 上最稳（无需 VLM 权重）


class PdfParseError(Exception):
    """PDF 解析彻底失败（主备都挂了），信息为中文。"""


def discover_markdown(output_dir: Path, stem: str) -> Path | None:
    """在 MinerU 输出目录里找生成的 markdown（兼容不同版本的目录布局）。"""
    if not output_dir.exists():
        return None
    candidates = sorted(output_dir.rglob(f"{stem}.md"))
    if candidates:
        return candidates[0]
    all_md = sorted(output_dir.rglob("*.md"))
    return all_md[0] if all_md else None


def _mineru_bin() -> Path:
    """venv 里的 mineru CLI 路径。"""
    return Path(sys.executable).parent / "mineru"


def _run_mineru(pdf_path: Path, output_dir: Path, device: str, timeout: int) -> Path:
    """以指定设备模式跑一次 MinerU，成功返回 markdown 路径，失败抛异常。"""
    env = os.environ.copy()
    env["MINERU_MODEL_SOURCE"] = "modelscope"  # 权重下载优先 modelscope 镜像（HF 慢）
    env["MINERU_DEVICE_MODE"] = device
    cmd = [
        str(_mineru_bin()),
        "-p", str(pdf_path),
        "-o", str(output_dir),
        "-b", MINERU_BACKEND,
    ]
    logger.info("MinerU 解析开始：%s（device=%s, backend=%s）", pdf_path.name, device, MINERU_BACKEND)
    proc = subprocess.run(
        cmd, env=env, timeout=timeout,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if proc.returncode != 0:
        tail = (proc.stdout or "")[-800:]
        raise PdfParseError(f"MinerU 退出码 {proc.returncode}（device={device}）：{tail}")
    md = discover_markdown(output_dir, pdf_path.stem)
    if md is None:
        raise PdfParseError(f"MinerU 运行成功但找不到输出的 markdown（{output_dir}）")
    logger.info("MinerU 解析完成：%s", md)
    return md


def pdf_page_count(pdf_path: Path | str) -> int:
    """PDF 页数（进度推送用）；读取失败返回 0。"""
    try:
        import pymupdf  # 惰性导入

        with pymupdf.open(str(pdf_path)) as doc:
            return doc.page_count
    except Exception:
        return 0


def _parse_with_pymupdf(pdf_path: Path, output_dir: Path) -> Path:
    """PyMuPDF4LLM 兜底：纯文本抽取，无版面分析。"""
    import pymupdf4llm  # 惰性导入

    md_text = pymupdf4llm.to_markdown(str(pdf_path))
    if not md_text.strip():
        raise PdfParseError("PyMuPDF4LLM 抽取结果为空（可能是扫描件且无 OCR）")
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"{pdf_path.stem}.md"
    md_path.write_text(md_text, encoding="utf-8")
    logger.info("PyMuPDF4LLM 解析完成：%s", md_path)
    return md_path


def parse_pdf(pdf_path: Path | str, output_dir: Path | str, timeout: int = DEFAULT_PARSE_TIMEOUT) -> Path:
    """解析 PDF 为 markdown，返回 markdown 路径。

    顺序：MinerU(mps) → MinerU(cpu) → PyMuPDF4LLM。每次降级都记 WARNING 日志。
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    if not pdf_path.exists():
        raise PdfParseError(f"PDF 文件不存在：{pdf_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    for device in ("mps", "cpu"):
        try:
            return _run_mineru(pdf_path, output_dir, device, timeout)
        except (PdfParseError, subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("MinerU（device=%s）失败，%s：%s",
                           device, "降级设备重试" if device == "mps" else "降级 PyMuPDF4LLM", exc)
            errors.append(f"MinerU({device}): {exc}")

    try:
        return _parse_with_pymupdf(pdf_path, output_dir)
    except Exception as exc:  # 兜底也挂了才抛
        errors.append(f"PyMuPDF4LLM: {exc}")
        raise PdfParseError("PDF 解析失败（MinerU 与 PyMuPDF4LLM 均不可用）：" + "；".join(errors)) from exc
