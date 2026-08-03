"""import_history.py 测试：arXiv 提取、扫描计划、dry-run。脚本按路径加载，离线运行。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "attic" / "import_history.py"
spec = importlib.util.spec_from_file_location("import_history", SCRIPT)
import_history = importlib.util.module_from_spec(spec)
spec.loader.exec_module(import_history)


def _mk_pdf(path: Path):
    path.write_bytes(b"%PDF-1.4 fake" * 10)


# ---- arXiv id 提取 ----

def test_extract_arxiv_from_filename():
    assert import_history.extract_arxiv_id(Path("2504.13359v2.pdf")) == "2504.13359"
    assert import_history.extract_arxiv_id(Path("2603.03329.pdf")) == "2603.03329"


def test_extract_arxiv_from_first_page_text():
    assert import_history.extract_arxiv_id(Path("随机名字.pdf"), "arXiv:2603.03329v1") == "2603.03329"


def test_extract_arxiv_none():
    assert import_history.extract_arxiv_id(Path("random.pdf"), "没有编号") is None


# ---- 扫描计划 ----

def test_collect_plan_and_limit(tmp_path):
    for i in range(25):
        _mk_pdf(tmp_path / f"paper{i:02d}.pdf")
    plan = import_history.collect_plan(tmp_path, limit=20)
    assert len(plan) == 20  # 上限生效
    named = tmp_path / "2504.13359v2.pdf"
    _mk_pdf(named)
    plan = import_history.collect_plan(tmp_path, limit=20)
    arxiv_items = [p for p in plan if p["arxiv_id"] == "2504.13359"]
    assert arxiv_items and arxiv_items[0]["meta_mode"] == "arxiv-api"
    no_id = [p for p in plan if p["arxiv_id"] is None]
    assert all(p["meta_mode"] == "light-model" for p in no_id)


# ---- dry-run ----

def test_main_dry_run_prints_plan(tmp_path, capsys, monkeypatch):
    _mk_pdf(tmp_path / "2504.13359v2.pdf")
    monkeypatch.setattr("sys.argv", ["import_history.py", "--dir", str(tmp_path)])
    assert import_history.main() == 0
    out = capsys.readouterr().out
    assert "dry-run" in out and "2504.13359" in out and "arXiv" in out


def test_main_missing_dir(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["import_history.py", "--dir", "/nonexistent-dir-xyz"])
    assert import_history.main() == 1
    assert "不存在" in capsys.readouterr().out
