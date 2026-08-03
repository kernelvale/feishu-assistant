#!/usr/bin/env python3
"""扫描或执行本地缓存生命周期策略；默认只报告，--apply 才删除。"""

from __future__ import annotations

import argparse

from feishu_assistant.lifecycle import PaperCache, StoragePolicy, cleanup_sensitive_roots
from feishu_assistant.modules.career import AUDIO_DIR, RESUME_DIR, TAILORED_RESUME_DIR
from feishu_assistant.modules.paper import PAPERS_DIR
from feishu_assistant.store import PaperStore


def _mb(value: int) -> str:
    return f"{value / 1024 / 1024:.1f} MB"


def main() -> int:
    parser = argparse.ArgumentParser(description="安全扫描/清理论文缓存和敏感临时文件")
    parser.add_argument("--apply", action="store_true", help="实际执行；默认 dry-run")
    args = parser.parse_args()

    policy = StoragePolicy.from_env()
    store = PaperStore()
    cache = PaperCache(PAPERS_DIR, policy=policy, store=store)
    papers = cache.cleanup(dry_run=not args.apply, include_legacy=True)
    sensitive = cleanup_sensitive_roots(
        [RESUME_DIR, TAILORED_RESUME_DIR, AUDIO_DIR],
        older_than_seconds=policy.sensitive_ttl_seconds,
        dry_run=not args.apply,
    )
    mode = "已执行" if args.apply else "仅扫描"
    print(
        f"{mode}：论文缓存扫描 {papers.scanned_entries} 项（历史 {papers.legacy_entries}），"
        f"符合清理条件 {papers.eligible_entries} 项/{_mb(papers.eligible_bytes)}，"
        f"删除 {papers.removed_entries} 项，释放 {_mb(papers.freed_bytes)}，"
        f"剩余 {_mb(papers.remaining_bytes)}，活跃跳过 {papers.active_skipped}。"
    )
    print(
        f"{mode}：敏感临时文件扫描 {sensitive.scanned_entries} 项，"
        f"符合清理条件 {sensitive.eligible_entries} 项/{_mb(sensitive.eligible_bytes)}，"
        f"删除 {sensitive.removed_entries} 项，释放 {_mb(sensitive.freed_bytes)}，"
        f"剩余 {_mb(sensitive.remaining_bytes)}，不安全路径跳过 {sensitive.unsafe_skipped}。"
    )
    if papers.over_capacity:
        print("警告：论文缓存仍超过容量或条目上限；通常表示对象正在使用。")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
