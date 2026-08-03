from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from feishu_assistant.lifecycle import (
    LifecycleError,
    PaperCache,
    StoragePolicy,
    cleanup_sensitive_roots,
    sensitive_workspace,
)
from feishu_assistant.store import PaperStore


PDF_BYTES = b"%PDF-1.7\n" + b"paper-content\n" * 1000


def _policy(**overrides) -> StoragePolicy:
    values = {
        "paper_ttl_seconds": 3600,
        "paper_max_bytes": 50 * 1024 * 1024,
        "paper_max_entries": 20,
        "paper_max_item_bytes": 10 * 1024 * 1024,
        "min_free_bytes": 1,
        "incomplete_ttl_seconds": 60,
        "sensitive_ttl_seconds": 60,
    }
    values.update(overrides)
    return StoragePolicy(**values)


def _downloader(counter: list[int], payload: bytes = PDF_BYTES):
    def download(directory: Path) -> Path:
        counter.append(1)
        destination = directory / "paper.pdf"
        destination.write_bytes(payload)
        return destination
    return download


def test_content_cache_reuses_download_and_parser(tmp_path):
    downloads: list[int] = []
    parses: list[int] = []
    cache = PaperCache(tmp_path / "papers", policy=_policy())

    def parser(_pdf: Path, output: Path) -> Path:
        parses.append(1)
        markdown = output / "nested" / "paper.md"
        markdown.parent.mkdir(parents=True)
        markdown.write_text("正文", encoding="utf-8")
        return markdown

    with cache.acquire("arxiv:1", _downloader(downloads)) as first:
        first_md = cache.ensure_parsed(first, parser)
        first_id = first.object_id
        assert first.cache_hit is False and first_md.read_text() == "正文"
    with cache.acquire("arxiv:1", _downloader(downloads)) as second:
        second_md = cache.ensure_parsed(second, parser)
        assert second.cache_hit is True and second.object_id == first_id
        assert second_md.read_text() == "正文"

    assert len(downloads) == 1
    assert len(parses) == 1


def test_private_attachments_do_not_share_object_identity(tmp_path):
    cache = PaperCache(tmp_path / "papers", policy=_policy())
    first_calls: list[int] = []
    second_calls: list[int] = []
    with cache.acquire("attachment:a", _downloader(first_calls), private_scope="user-a") as a:
        a_id = a.object_id
    with cache.acquire("attachment:b", _downloader(second_calls), private_scope="user-b") as b:
        b_id = b.object_id
    assert a_id != b_id
    assert first_calls == [1] and second_calls == [1]


def test_downloader_cannot_escape_incoming_directory(tmp_path):
    cache = PaperCache(tmp_path / "papers", policy=_policy())
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(PDF_BYTES)
    with pytest.raises(LifecycleError, match="临时目录之外"):
        with cache.acquire("bad", lambda _: outside):
            pass
    assert outside.exists()


def test_oversized_pdf_is_rejected_without_leaving_incoming_file(tmp_path):
    cache = PaperCache(
        tmp_path / "papers",
        policy=_policy(paper_max_item_bytes=len(PDF_BYTES) - 1),
    )
    with pytest.raises(LifecycleError, match="单文件缓存上限"):
        with cache.acquire("too-large", _downloader([])):
            pass
    assert list(cache.incoming_dir.iterdir()) == []
    assert list(cache.objects_dir.iterdir()) == []


def test_cleanup_skips_active_then_removes_expired_and_reconciles_db(tmp_path):
    store = PaperStore(tmp_path / "papers.db")
    cache = PaperCache(tmp_path / "papers", policy=_policy(paper_ttl_seconds=1), store=store)
    calls: list[int] = []
    context = cache.acquire("arxiv:locked", _downloader(calls))
    entry = context.__enter__()
    store.save_paper("chat", "id", "title", "md", pdf_path=str(entry.pdf_path))
    (entry.object_dir / ".last_used.json").write_text(
        json.dumps({"at": time.time() - 100}), encoding="utf-8",
    )
    try:
        active_report = cache.cleanup()
        assert active_report.active_skipped == 1
        assert entry.pdf_path.exists()
    finally:
        context.__exit__(None, None, None)
    (entry.object_dir / ".last_used.json").write_text(
        json.dumps({"at": time.time() - 100}), encoding="utf-8",
    )
    report = cache.cleanup()
    assert report.removed_entries == 1
    assert not entry.object_dir.exists()
    assert store.latest_paper("chat").pdf_path == ""


def test_concurrent_same_source_downloads_once(tmp_path):
    cache = PaperCache(tmp_path / "papers", policy=_policy())
    calls: list[int] = []
    calls_lock = threading.Lock()

    def download(directory: Path) -> Path:
        with calls_lock:
            calls.append(1)
        time.sleep(0.05)
        path = directory / "paper.pdf"
        path.write_bytes(PDF_BYTES)
        return path

    def use_cache() -> str:
        with cache.acquire("arxiv:same", download) as entry:
            return entry.content_hash

    with ThreadPoolExecutor(max_workers=2) as pool:
        hashes = list(pool.map(lambda _: use_cache(), range(2)))
    assert hashes[0] == hashes[1]
    assert len(calls) == 1


def test_same_cached_paper_allows_concurrent_readers(tmp_path):
    cache = PaperCache(tmp_path / "papers", policy=_policy())
    calls: list[int] = []
    with cache.acquire("arxiv:shared", _downloader(calls)):
        pass
    barrier = threading.Barrier(2, timeout=2)

    def reader() -> bool:
        with cache.acquire("arxiv:shared", _downloader(calls)):
            barrier.wait()
            return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda _: reader(), range(2))) == [True, True]
    assert len(calls) == 1


def test_incomplete_directory_respects_lock_and_is_recovered(tmp_path):
    cache = PaperCache(tmp_path / "papers", policy=_policy(incomplete_ttl_seconds=1))
    incoming = cache.incoming_dir / "orphan"
    incoming.mkdir()
    (incoming / "partial.pdf").write_bytes(b"partial")
    old = time.time() - 100
    os.utime(incoming, (old, old))
    report = cache.cleanup()
    assert report.removed_entries == 1
    assert not incoming.exists()


def test_entry_limit_rejects_new_download_when_all_slots_are_active(tmp_path):
    cache = PaperCache(
        tmp_path / "papers",
        policy=_policy(paper_max_entries=1),
    )
    first_context = cache.acquire("arxiv:first", _downloader([]))
    first_context.__enter__()
    try:
        with pytest.raises(LifecycleError, match="都在使用"):
            with cache.acquire("arxiv:second", _downloader([])):
                pass
    finally:
        first_context.__exit__(None, None, None)


def test_sensitive_workspace_cleans_on_success_and_exception(tmp_path):
    root = tmp_path / "resumes"
    with sensitive_workspace(root, "upload") as workspace:
        (workspace / "resume.pdf").write_bytes(b"private")
        saved = workspace
    assert not saved.exists()

    with pytest.raises(RuntimeError):
        with sensitive_workspace(root, "upload") as workspace:
            (workspace / "resume.pdf").write_bytes(b"private")
            failed = workspace
            raise RuntimeError("boom")
    assert not failed.exists()


def test_sensitive_sweeper_removes_old_files_but_never_symlink_target(tmp_path):
    root = tmp_path / "audio"
    root.mkdir()
    old_file = root / "old.opus"
    old_file.write_bytes(b"voice")
    old = time.time() - 100
    os.utime(old_file, (old, old))
    outside = tmp_path / "keep.txt"
    outside.write_text("keep", encoding="utf-8")
    (root / "unsafe-link").symlink_to(outside)

    report = cleanup_sensitive_roots([root], older_than_seconds=1)
    assert report.removed_entries == 1 and report.unsafe_skipped == 1
    assert not old_file.exists() and outside.read_text() == "keep"


def test_sensitive_sweeper_skips_active_workspace(tmp_path):
    root = tmp_path / "resumes"
    context = sensitive_workspace(root, "long-job")
    workspace = context.__enter__()
    (workspace / "resume.pdf").write_bytes(b"private")
    old = time.time() - 100
    os.utime(workspace, (old, old))
    try:
        report = cleanup_sensitive_roots([root], older_than_seconds=1)
        assert report.active_skipped == 1
        assert workspace.exists()
    finally:
        context.__exit__(None, None, None)
    assert not workspace.exists()
