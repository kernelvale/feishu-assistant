#!/usr/bin/env python3
"""本地实跑已有解析产物，不创建飞书文档、不更新归档。

用法：uv run python scripts/reading_smoke.py <arxiv_id> [output.md]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feishu_assistant.config import load_settings
from feishu_assistant.kimi import KimiClient
from feishu_assistant.modules.figures import load_figure_assets
from feishu_assistant.modules.formulas import load_formula_assets
from feishu_assistant.modules.reading import ReadingEngine


def main() -> int:
    if len(sys.argv) < 2:
        print("用法：reading_smoke.py <arxiv_id> [output.md]")
        return 1
    arxiv_id = sys.argv[1].strip()
    parse_dir = ROOT / "data" / "papers" / arxiv_id / "paper" / "auto"
    paper_path = parse_dir / "paper.md"
    if not paper_path.exists():
        print(f"找不到已有解析产物：{paper_path}")
        return 1
    output = (
        Path(sys.argv[2]).expanduser().resolve()
        if len(sys.argv) >= 3
        else Path("/private/tmp") / f"reading-smoke-{arxiv_id}.md"
    )
    engine = ReadingEngine(KimiClient(load_settings()))
    result = engine.run(
        paper_path.read_text(encoding="utf-8"),
        load_figure_assets(parse_dir),
        load_formula_assets(parse_dir),
        arxiv_id=arxiv_id,
        progress=lambda message: print(message, flush=True),
    )
    output.write_text(result.body_md, encoding="utf-8")
    print(f"本地产物：{output}")
    print(f"模型调用：{result.model_calls}，视觉读图：{result.vision_used}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
