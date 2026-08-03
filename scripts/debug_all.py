#!/usr/bin/env python3
"""feishu-assistant 全身体检（只读，不改任何数据）。

用法：uv run python scripts/debug_all.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from feishu_assistant.config import load_settings, missing_credentials, usable_candidates  # noqa: E402
from feishu_assistant.modules.paper import doc_display_name  # noqa: E402

ok_count = warn_count = fail_count = 0


def report(ok: bool | None, label: str, detail: str = "") -> None:
    global ok_count, warn_count, fail_count
    if ok is True:
        ok_count += 1
        print(f"  ✅ {label}" + (f"：{detail}" if detail else ""))
    elif ok is False:
        fail_count += 1
        print(f"  ❌ {label}" + (f"：{detail}" if detail else ""))
    else:
        warn_count += 1
        print(f"  ⚠️  {label}" + (f"：{detail}" if detail else ""))


def main() -> int:
    print("🩺 feishu-assistant 体检\n")

    print("[1] 配置与凭证")
    try:
        settings = load_settings()
        missing = missing_credentials(settings)
        report(not missing, "凭证齐备", "；".join(missing) if missing else "飞书+模型均就位")
        for kind in ("heavy", "light"):
            usable = usable_candidates(settings, heavy=(kind == "heavy"))
            report(bool(usable), f"{kind} 链可用候选", f"{len(usable)} 个")
    except Exception as exc:
        report(False, "配置加载", str(exc))
        return 1

    print("\n[2] 多维表格")
    if settings.bitable_ready():
        try:
            from feishu_assistant.feishu import build_client
            from feishu_assistant.feishu.bitable import BitableDAO

            dao = BitableDAO(build_client(settings), settings.bitable_app_token, settings.bitable_tables)
            records = dao.list_records("论文库", page_size=20)
            report(True, "论文库连通", f"近期记录 {len(records)} 条")
            # 检查文档链接显示文本是否符合命名规则
            bad = []
            for r in records:
                link = (r.fields or {}).get("文档链接")
                text = ""
                if isinstance(link, dict):
                    text = str(link.get("text", ""))
                elif isinstance(link, str):
                    text = link
                if text and not re.match(r"^精读：.+", text):
                    bad.append(text)
            report(not bad, "文档链接命名规则（精读：短标题）", "；".join(bad[:3]) if bad else f"抽查 {len(records)} 条合规")
        except Exception as exc:
            report(False, "论文库连通", str(exc))
    else:
        report(None, "多维表格", "未初始化")

    print("\n[3] 本地 sqlite")
    try:
        import sqlite3

        db = Path("data/papers.db")
        if not db.exists():
            report(None, "papers.db", "尚未创建（无精读/会话记录）")
        else:
            conn = sqlite3.connect(db)
            for table, label in (("papers", "精读论文"), ("sessions", "会话"),
                                 ("messages", "会话消息"), ("seen_messages", "去重记录"),
                                 ("dead_letters", "死信")):
                try:
                    n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    report(True, f"{label}（{table}）", f"{n} 条")
                except sqlite3.OperationalError:
                    report(None, f"{label}（{table}）", "表不存在（对应功能未用过）")
            try:
                n = conn.execute("SELECT COUNT(*) FROM paper_vecs").fetchone()[0]
                report(True, "向量索引（paper_vecs）", f"{n} 条")
            except sqlite3.OperationalError:
                # vec0 虚表需要加载 sqlite-vec 扩展才能查询；换用真实索引类探测
                try:
                    import sys
                    sys.path.insert(0, str(Path("src").resolve()))
                    from feishu_assistant.vectors import PaperVectorIndex
                    n = PaperVectorIndex(db).count()
                    report(True, "向量索引（paper_vecs）", f"{n} 条")
                except Exception as exc2:
                    report(None, "向量索引", f"探测失败：{exc2}")
            conn.close()
    except Exception as exc:
        report(False, "sqlite 检查", str(exc))

    print("\n[4] 图表产物")
    papers_dir = Path("data/papers")
    if papers_dir.exists():
        figure_dirs = [d for d in papers_dir.glob("*/mineru_out/**/images") if d.is_dir()]
        total_imgs = sum(len(list(d.glob("*.jpg"))) for d in figure_dirs)
        report(total_imgs > 0, "MinerU 图表截图", f"{total_imgs} 张（{len(figure_dirs)} 个目录）")
        # 命名规则示例
        demo = doc_display_name({"title_zh": "AutoHarness"}, "2603.03329")
        report(bool(re.match(r"^精读：.+ v\d+（\d{4}\.\d{4,5} · \d{2}-\d{2}）$", demo)),
               "文档命名规则函数", demo)
    else:
        report(None, "data/papers", "无解析产物")

    print(f"\n结果：✅ {ok_count}  ⚠️ {warn_count}  ❌ {fail_count}")
    return 1 if fail_count else 0


if __name__ == "__main__":
    sys.exit(main())
