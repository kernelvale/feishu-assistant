#!/usr/bin/env python3
"""历史 PDF 批量导入论文库。

扫指定目录（默认 ~/Downloads）顶层 PDF：
- 元数据：文件名/正文里的 arXiv id 优先（命中则走 arXiv API 取标题作者），
  否则用首页文本走 light 模型提取
- 写入论文库（来源="历史导入"，不生成精读文档，不设精读时间）
- 默认 dry-run 只打印计划；--apply 才真正解析与写入；每次上限 --limit（默认 20）
- --deep <pdf路径>：对单篇跑教学型精读（生成飞书文档+归档，同 bot 流程）

用法：
    uv run python scripts/import_history.py                 # dry-run
    uv run python scripts/import_history.py --apply         # 实际导入
    uv run python scripts/import_history.py --deep ~/Downloads/xxx.pdf
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from feishu_assistant.config import Settings, load_settings  # noqa: E402
from feishu_assistant.modules.paper import (  # noqa: E402
    MessageContext,
    PaperReadingService,
    arxiv_pdf_url,
    build_paper_row,
    fetch_arxiv_metadata,
)
from feishu_assistant.modules.reading import classify_direction  # noqa: E402

ARXIV_IN_TEXT_RE = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)
ARXIV_IN_NAME_RE = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")
IMPORT_SOURCE = "历史导入"


def extract_arxiv_id(pdf_path: Path, first_page_text: str = "") -> str | None:
    """文件名 → 首页文本，找 arXiv id。"""
    m = ARXIV_IN_NAME_RE.search(pdf_path.stem)
    if m:
        return m.group(1)
    m = ARXIV_IN_TEXT_RE.search(first_page_text)
    return m.group(1) if m else None


def read_first_page_text(pdf_path: Path, max_chars: int = 3000) -> str:
    """pymupdf 抽首页文本（轻量，不走 MinerU）。"""
    try:
        import pymupdf

        with pymupdf.open(str(pdf_path)) as doc:
            return doc[0].get_text()[:max_chars]
    except Exception:
        return ""


def metadata_via_light(kimi, first_page_text: str) -> dict:
    """首页文本 → light 模型提取元数据（JSON）。失败返回空 dict。"""
    prompt = (
        "从论文首页文本提取元数据，只输出 JSON（不要注释）：\n"
        '{"title_zh":"中文标题（可直译）","title_en":"英文原标题","authors":"逗号分隔",'
        '"affiliation":"机构","venue":"","year":"年份","keywords":["3~5个"],"tldr":"100字内一句话"}\n\n'
        f"首页文本：\n{first_page_text[:2500]}"
    )
    try:
        out = kimi.chat([{"role": "user", "content": prompt}], heavy=False)
        m = re.search(r"\{.*\}", out, re.DOTALL)
        return json.loads(m.group(0)) if m else {}
    except Exception as exc:
        print(f"    ⚠️  light 元数据提取失败：{exc}")
        return {}


def collect_plan(downloads: Path, limit: int) -> list[dict]:
    """扫描并生成导入计划（dry-run 与 apply 共用）。

    文件名含 简历/resume/cv 的跳过（不是论文，走求职简历库流程）。
    """
    import re
    resume_pat = re.compile(r"简历|(?:^|[^a-z0-9])resume(?:[^a-z0-9]|$)|(?:^|[^a-z0-9])cv(?:[^a-z0-9]|$)", re.IGNORECASE)
    pdfs = [p for p in sorted(downloads.glob("*.pdf")) if not resume_pat.search(p.name)][:limit]
    plan = []
    for pdf in pdfs:
        first_page = read_first_page_text(pdf)
        arxiv_id = extract_arxiv_id(pdf, first_page)
        plan.append({
            "path": pdf, "arxiv_id": arxiv_id,
            "meta_mode": "arxiv-api" if arxiv_id else "light-model",
        })
    return plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="实际写入（默认 dry-run）")
    parser.add_argument("--deep", metavar="PDF", help="对单篇 PDF 跑教学型精读")
    parser.add_argument("--dir", default=str(Path.home() / "Downloads"), help="扫描目录")
    parser.add_argument("--limit", type=int, default=20, help="本次上限（默认 20）")
    args = parser.parse_args()

    settings = load_settings()

    if args.deep:
        return run_deep(settings, Path(args.deep))

    downloads = Path(args.dir).expanduser()
    if not downloads.exists():
        print(f"❌ 目录不存在：{downloads}")
        return 1

    plan = collect_plan(downloads, args.limit)
    print(f"扫描 {downloads}：{len(plan)} 个 PDF（上限 {args.limit}）\n")
    for i, item in enumerate(plan, 1):
        arxiv_text = f"arXiv:{item['arxiv_id']}" if item["arxiv_id"] else "无 arXiv id（走 light 提取）"
        print(f"  {i}. {item['path'].name} → {arxiv_text}")

    if not args.apply:
        print("\n（dry-run）以上不会写入。加 --apply 实际导入；--deep <pdf> 单篇教学型精读。")
        return 0

    if not settings.bitable_ready():
        print("❌ 多维表格未初始化，无法写入论文库")
        return 1

    from feishu_assistant.feishu import build_client
    from feishu_assistant.feishu.bitable import BitableDAO
    from feishu_assistant.kimi import KimiClient

    client = build_client(settings)
    kimi = KimiClient(settings)
    dao = BitableDAO(client, settings.bitable_app_token, settings.bitable_tables)
    existing_ids, existing_titles = load_existing_keys(dao)

    ok = skipped = failed = 0
    for item in plan:
        pdf, arxiv_id = item["path"], item["arxiv_id"]
        print(f"\n▶ {pdf.name}")
        if arxiv_id and arxiv_id in existing_ids:
            print("    跳过：论文库已存在该 arXiv id")
            skipped += 1
            continue
        if not arxiv_id and pdf.stem in existing_titles:
            print("    跳过：论文库已存在同名记录（无 arXiv id 按标题去重）")
            skipped += 1
            continue
        try:
            metadata = fetch_arxiv_metadata(arxiv_id) if arxiv_id else {}
            if not metadata.get("title"):
                metadata = metadata_via_light(kimi, read_first_page_text(pdf))
            if not metadata:
                print("    ❌ 元数据提取失败，跳过")
                failed += 1
                continue
            title = metadata.get("title_zh") or metadata.get("title_en") or pdf.stem
            direction = classify_direction(kimi, title, str(metadata.get("tldr", "")))
            row = build_paper_row(
                {**metadata, "title_zh": title}, "", arxiv_pdf_url(arxiv_id) if arxiv_id else "",
                arxiv_id or "", direction=direction, source=IMPORT_SOURCE,
            )
            row.pop("精读时间", None)  # 历史导入不算精读
            dao.add_record("论文库", row)
            print(f"    ✅ 已写入：《{title}》（方向：{direction}）")
            ok += 1
        except Exception as exc:
            print(f"    ❌ 失败：{exc}")
            failed += 1

    print(f"\n完成：写入 {ok}，跳过 {skipped}，失败 {failed}")
    return 0 if failed == 0 else 1


def load_existing_keys(dao) -> tuple[set[str], set[str]]:
    """读论文库现有 arXivID 与标题（双键去重：无 arXiv id 的按标题挡重复导入）。"""
    try:
        records = dao.list_records("论文库", page_size=100)
        ids, titles = set(), set()
        for r in records:
            value = (r.fields or {}).get("arXivID")
            if isinstance(value, str) and value:
                ids.add(value)
            elif isinstance(value, list):
                ids.add("".join(str(seg.get("text", "")) for seg in value if isinstance(seg, dict)))
            t = (r.fields or {}).get("标题")
            if isinstance(t, str) and t.strip():
                titles.add(t.strip())
        return ids, titles
    except Exception as exc:
        print(f"⚠️  读取现有记录失败（按无重复处理）：{exc}")
        return set(), set()


def run_deep(settings: Settings, pdf_path: Path) -> int:
    """单篇教学型精读（与 bot 同管线：解析→生成→终审→飞书文档→归档）。"""
    if not pdf_path.exists():
        print(f"❌ 文件不存在：{pdf_path}")
        return 1
    from feishu_assistant.feishu import build_client
    from feishu_assistant.kimi import KimiClient
    from feishu_assistant.store import PaperStore
    from feishu_assistant.vectors import PaperVectorIndex

    class ConsoleSender:
        def send_text(self, chat_id, text):
            print(f"  {text}")
            return True

        def send_card(self, chat_id, card):
            print(f"  [卡片] {card['header']['title']['content']}")
            return True

    kimi = KimiClient(settings)
    service = PaperReadingService(
        settings, kimi, build_client(settings), PaperStore(),
        sender=ConsoleSender(), vector_index=PaperVectorIndex(),
    )
    ctx = MessageContext(chat_id="import", open_id="", message_id="")
    first_page = read_first_page_text(pdf_path)
    arxiv_id = extract_arxiv_id(pdf_path, first_page) or ""
    meta_hint = fetch_arxiv_metadata(arxiv_id) if arxiv_id else {}
    if not meta_hint.get("title"):
        meta_hint = {"title": pdf_path.stem}
    try:
        metadata, doc_url, archived = service._pipeline(
            ctx, pdf_path, key=arxiv_id or f"import_{int(__import__('time').time())}",
            pdf_url=arxiv_pdf_url(arxiv_id) if arxiv_id else "",
            arxiv_id=arxiv_id, meta_hint=meta_hint,
        )
    except Exception as exc:
        print(f"❌ 精读失败：{exc}")
        return 1
    print(f"\n✅ 完成：{doc_url}（归档={'是' if archived else '否'}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
