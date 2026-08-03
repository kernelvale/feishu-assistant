"""本地下载与解析产物的安全生命周期管理。

论文使用内容寻址缓存：来源清单只负责在 TTL 内复用，PDF 内容哈希决定真实缓存对象，
解析器版本决定解析子目录。对象锁贯穿下载后的整条发布管线，清理器无法删除活跃任务。

简历、音频等敏感材料不进入共享缓存；调用方应使用 ``sensitive_workspace`` 并在任务结束
时立即清理。定时扫描只负责清除进程崩溃后遗留的过期文件。
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

logger = logging.getLogger("feishu_assistant.lifecycle")

PARSER_CACHE_VERSION = "pdf-parser-v1"
_RESERVED_DIRS = {"objects", "sources", "locks", "incoming", "trash"}


class LifecycleError(RuntimeError):
    """缓存损坏、磁盘压力或不安全路径。"""


@dataclass(frozen=True)
class StoragePolicy:
    paper_ttl_seconds: int = 7 * 24 * 3600
    paper_max_bytes: int = 2 * 1024 * 1024 * 1024
    paper_max_entries: int = 20
    paper_max_item_bytes: int = 250 * 1024 * 1024
    min_free_bytes: int = 512 * 1024 * 1024
    incomplete_ttl_seconds: int = 24 * 3600
    sensitive_ttl_seconds: int = 3600

    @classmethod
    def from_env(cls) -> "StoragePolicy":
        def positive_int(name: str, default: int) -> int:
            raw = os.getenv(name, "").strip()
            if not raw:
                return default
            try:
                value = int(raw)
            except ValueError:
                logger.warning("忽略非法生命周期配置 %s=%r", name, raw)
                return default
            return value if value > 0 else default

        return cls(
            paper_ttl_seconds=positive_int("PAPER_CACHE_TTL_HOURS", 168) * 3600,
            paper_max_bytes=positive_int("PAPER_CACHE_MAX_MB", 2048) * 1024 * 1024,
            paper_max_entries=positive_int("PAPER_CACHE_MAX_ENTRIES", 20),
            paper_max_item_bytes=positive_int("PAPER_CACHE_MAX_ITEM_MB", 250) * 1024 * 1024,
            min_free_bytes=positive_int("STORAGE_MIN_FREE_MB", 512) * 1024 * 1024,
            incomplete_ttl_seconds=positive_int("INCOMPLETE_FILE_TTL_HOURS", 24) * 3600,
            sensitive_ttl_seconds=positive_int("SENSITIVE_FILE_TTL_MINUTES", 60) * 60,
        )


@dataclass
class CleanupReport:
    scanned_entries: int = 0
    removed_entries: int = 0
    removed_files: int = 0
    freed_bytes: int = 0
    active_skipped: int = 0
    unsafe_skipped: int = 0
    remaining_bytes: int = 0
    over_capacity: bool = False
    legacy_entries: int = 0
    eligible_entries: int = 0
    eligible_bytes: int = 0


@dataclass(frozen=True)
class PaperCacheEntry:
    object_id: str
    content_hash: str
    object_dir: Path
    pdf_path: Path
    parse_dir: Path
    cache_hit: bool


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_size(path: Path) -> int:
    if path.is_symlink():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    try:
        children = list(path.iterdir())
    except OSError:
        return 0
    for child in children:
        total += _tree_size(child)
    return total


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=True))
        return True
    except (OSError, ValueError):
        return False


def _safe_rmtree(path: Path, root: Path) -> bool:
    """只删除指定根目录下的普通目录；软链接和根目录本身永不递归删除。"""
    if path == root or path.is_symlink() or not _is_within(path, root):
        logger.error("拒绝清理不安全路径：%s（root=%s）", path, root)
        return False
    try:
        shutil.rmtree(path)
        return True
    except FileNotFoundError:
        return True
    except OSError:
        logger.exception("清理目录失败：%s", path)
        return False


class FileLease:
    """基于 flock 的跨线程/跨进程租约；进程崩溃后内核自动释放。"""

    def __init__(self, path: Path, *, shared: bool = False) -> None:
        self.path = path
        self.shared = shared
        self._handle: Any = None

    def acquire(self, *, blocking: bool = True) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        handle = open(self.path, "a+", encoding="utf-8")
        mode = fcntl.LOCK_SH if self.shared else fcntl.LOCK_EX
        flags = mode | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(handle.fileno(), flags)
        except BlockingIOError:
            handle.close()
            return False
        if not self.shared:
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps({"pid": os.getpid(), "acquired_at": time.time()}))
            handle.flush()
        self._handle = handle
        return True

    def downgrade_to_shared(self) -> None:
        if self._handle is None:
            raise LifecycleError("尚未取得租约，不能降级为共享锁")
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_SH)
        self.shared = True

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "FileLease":
        if not self.acquire():
            raise LifecycleError(f"无法取得缓存锁：{self.path}")
        return self

    def __exit__(self, *_: Any) -> None:
        self.release()


class PaperCache:
    def __init__(
        self,
        root: Path,
        *,
        policy: StoragePolicy | None = None,
        store: Any = None,
    ) -> None:
        self.root = Path(root)
        self.policy = policy or StoragePolicy.from_env()
        self.store = store
        self.objects_dir = self.root / "objects"
        self.sources_dir = self.root / "sources"
        self.locks_dir = self.root / "locks"
        self.incoming_dir = self.root / "incoming"
        self.trash_dir = self.root / "trash"
        for path in (
            self.root, self.objects_dir, self.sources_dir, self.locks_dir,
            self.incoming_dir, self.trash_dir,
        ):
            path.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _source_id(self, source_key: str) -> str:
        return hashlib.sha256(source_key.encode("utf-8")).hexdigest()

    def _source_manifest(self, source_key: str) -> Path:
        return self.sources_dir / f"{self._source_id(source_key)}.json"

    def _object_id(self, content_hash: str, private_scope: str) -> str:
        if not private_scope:
            return content_hash
        return hashlib.sha256(f"private:{private_scope}:{content_hash}".encode()).hexdigest()

    def _entry(self, object_id: str, content_hash: str, cache_hit: bool) -> PaperCacheEntry:
        object_dir = self.objects_dir / object_id
        return PaperCacheEntry(
            object_id=object_id,
            content_hash=content_hash,
            object_dir=object_dir,
            pdf_path=object_dir / "paper.pdf",
            parse_dir=object_dir / PARSER_CACHE_VERSION,
            cache_hit=cache_hit,
        )

    def _raw_valid(self, entry: PaperCacheEntry) -> bool:
        marker = _read_json(entry.object_dir / ".raw_complete.json")
        try:
            size = entry.pdf_path.stat().st_size
        except OSError:
            return False
        return (
            not entry.pdf_path.is_symlink()
            and marker.get("content_hash") == entry.content_hash
            and marker.get("size") == size
            and size >= 10_000
            and _hash_file(entry.pdf_path) == entry.content_hash
        )

    def _touch(self, entry: PaperCacheEntry) -> None:
        _atomic_json(entry.object_dir / ".last_used.json", {"at": time.time()})

    @contextlib.contextmanager
    def acquire(
        self,
        source_key: str,
        downloader: Callable[[Path], Path],
        *,
        private_scope: str = "",
    ) -> Iterator[PaperCacheEntry]:
        """取得一个不可被清理的论文缓存对象；downloader 接收隔离的临时目录。"""
        preflight = self.cleanup(include_legacy=True)
        if preflight.over_capacity:
            raise LifecycleError("论文缓存已超过容量上限，且当前没有可安全释放的缓存；请稍后重试。")

        source_lock = FileLease(self.locks_dir / f"source-{self._source_id(source_key)}.lock")
        object_lease: FileLease | None = None
        entry: PaperCacheEntry | None = None
        incoming: Path | None = None
        source_lock.acquire()
        try:
            manifest_path = self._source_manifest(source_key)
            manifest = _read_json(manifest_path)
            checked_at = float(manifest.get("checked_at") or 0)
            fresh = time.time() - checked_at <= self.policy.paper_ttl_seconds
            content_hash = str(manifest.get("content_hash") or "")
            object_id = str(manifest.get("object_id") or "")
            if fresh and re.fullmatch(r"[0-9a-f]{64}", content_hash) and re.fullmatch(
                r"[0-9a-f]{64}", object_id
            ):
                candidate = self._entry(object_id, content_hash, True)
                lease = FileLease(candidate.object_dir / ".active.lock", shared=True)
                if candidate.object_dir.exists() and lease.acquire() and self._raw_valid(candidate):
                    entry, object_lease = candidate, lease
                else:
                    lease.release()

            if entry is None:
                reservation = self.cleanup(include_legacy=True, reserve_entries=1)
                if reservation.over_capacity:
                    raise LifecycleError(
                        "论文缓存条目已达上限，且现有对象都在使用；已拒绝新下载而不是强制删除"
                    )
                if shutil.disk_usage(self.root).free < self.policy.min_free_bytes:
                    raise LifecycleError("本地磁盘剩余空间低于安全水位，已停止接收新的论文下载")
                incoming = Path(tempfile.mkdtemp(prefix="paper-", dir=self.incoming_dir))
                os.chmod(incoming, 0o700)
                with FileLease(incoming / ".active.lock"):
                    downloaded = Path(downloader(incoming))
                    if downloaded.is_symlink() or not _is_within(downloaded, incoming):
                        raise LifecycleError("下载器返回了临时目录之外或软链接形式的 PDF")
                    with open(downloaded, "rb") as handle:
                        head = handle.read(1024)
                    if b"%PDF" not in head:
                        raise LifecycleError("下载结果不是有效 PDF")
                    if downloaded.stat().st_size > self.policy.paper_max_item_bytes:
                        raise LifecycleError("论文 PDF 超过单文件缓存上限，已停止处理")
                    content_hash = _hash_file(downloaded)
                    object_id = self._object_id(content_hash, private_scope)
                    candidate = self._entry(object_id, content_hash, False)
                    candidate.object_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
                    lease = FileLease(candidate.object_dir / ".active.lock")
                    lease.acquire()
                    object_lease = lease
                    entry = candidate
                    if not self._raw_valid(candidate):
                        temp_pdf = candidate.object_dir / f".paper.{uuid.uuid4().hex}.tmp"
                        shutil.copy2(downloaded, temp_pdf)
                        os.replace(temp_pdf, candidate.pdf_path)
                        _atomic_json(candidate.object_dir / ".raw_complete.json", {
                            "content_hash": content_hash,
                            "size": candidate.pdf_path.stat().st_size,
                            "created_at": time.time(),
                        })
                    lease.downgrade_to_shared()
                    _atomic_json(manifest_path, {
                        "source_id": self._source_id(source_key),
                        "content_hash": content_hash,
                        "object_id": object_id,
                        "checked_at": time.time(),
                        "private": bool(private_scope),
                    })
        except Exception:
            if object_lease is not None:
                object_lease.release()
            raise
        finally:
            source_lock.release()
            if incoming is not None:
                _safe_rmtree(incoming, self.root)

        if entry is None or object_lease is None:
            raise LifecycleError("无法建立论文缓存对象")
        try:
            self._touch(entry)
            yield entry
        finally:
            self._touch(entry)
            object_lease.release()
            try:
                self.cleanup(include_legacy=True)
            except Exception:
                logger.exception("论文任务结束后的容量整理失败")

    def ensure_parsed(
        self,
        entry: PaperCacheEntry,
        parser: Callable[[Path, Path], Path],
    ) -> Path:
        with FileLease(entry.object_dir / ".parse.lock"):
            marker_path = entry.parse_dir / ".parse_complete.json"
            marker = _read_json(marker_path)
            relative = str(marker.get("markdown") or "")
            if marker.get("parser_version") == PARSER_CACHE_VERSION and relative:
                candidate = entry.parse_dir / relative
                if candidate.is_file() and not candidate.is_symlink() and _is_within(candidate, entry.parse_dir):
                    return candidate

            if entry.parse_dir.exists() and not _safe_rmtree(entry.parse_dir, entry.object_dir):
                raise LifecycleError("无法清理不完整的论文解析缓存")
            temp_parse = entry.object_dir / f".{PARSER_CACHE_VERSION}.{uuid.uuid4().hex}.tmp"
            temp_parse.mkdir(parents=True, mode=0o700)
            try:
                markdown = Path(parser(entry.pdf_path, temp_parse))
                if markdown.is_symlink() or not markdown.is_file() or not _is_within(markdown, temp_parse):
                    raise LifecycleError("解析器返回了缓存目录之外的 Markdown")
                relative = str(markdown.relative_to(temp_parse))
                _atomic_json(temp_parse / ".parse_complete.json", {
                    "parser_version": PARSER_CACHE_VERSION,
                    "markdown": relative,
                    "completed_at": time.time(),
                })
                os.replace(temp_parse, entry.parse_dir)
                return entry.parse_dir / relative
            except Exception:
                _safe_rmtree(temp_parse, entry.object_dir)
                raise

    def _last_used(self, path: Path) -> float:
        marker = _read_json(path / ".last_used.json")
        try:
            return float(marker.get("at") or path.stat().st_mtime)
        except OSError:
            return 0.0

    def _candidate_dirs(self, include_legacy: bool) -> list[tuple[Path, bool]]:
        candidates: list[tuple[Path, bool]] = []
        try:
            candidates.extend((path, False) for path in self.objects_dir.iterdir() if path.is_dir())
            if include_legacy:
                candidates.extend(
                    (path, True) for path in self.root.iterdir()
                    if path.is_dir() and path.name not in _RESERVED_DIRS and not path.is_symlink()
                )
        except OSError:
            logger.exception("扫描论文缓存失败：%s", self.root)
        return candidates

    def _clear_db_prefix(self, original: Path) -> None:
        if self.store is None:
            return
        try:
            self.store.clear_pdf_paths_under(original)
        except Exception:
            logger.exception("清理缓存后同步论文数据库路径失败：%s", original)

    def cleanup(
        self, *, dry_run: bool = False, include_legacy: bool = False,
        reserve_entries: int = 0,
    ) -> CleanupReport:
        """按 TTL、容量与条目上限执行 LRU 清理；活跃对象、软链接和越界路径跳过。"""
        report = CleanupReport()
        now = time.time()

        # 崩溃可能留下 incoming；只有超过恢复窗口且能取得租约时才处理。
        try:
            incoming_children = list(self.incoming_dir.iterdir())
        except OSError:
            incoming_children = []
        for path in incoming_children:
            if path.is_symlink() or not path.is_dir() or not _is_within(path, self.root):
                report.unsafe_skipped += 1
                continue
            try:
                stale = now - path.stat().st_mtime > self.policy.incomplete_ttl_seconds
            except OSError:
                continue
            if not stale:
                continue
            report.eligible_entries += 1
            report.eligible_bytes += _tree_size(path)
            lease = FileLease(path / ".active.lock")
            if not lease.acquire(blocking=False):
                report.active_skipped += 1
                continue
            try:
                size = _tree_size(path)
                if not dry_run and _safe_rmtree(path, self.root):
                    report.removed_entries += 1
                    report.removed_files += 1
                    report.freed_bytes += size
            finally:
                lease.release()

        candidates: list[dict[str, Any]] = []
        for path, legacy in self._candidate_dirs(include_legacy):
            report.scanned_entries += 1
            report.legacy_entries += int(legacy)
            if path.is_symlink() or not _is_within(path, self.root):
                report.unsafe_skipped += 1
                continue
            size = _tree_size(path)
            used = self._last_used(path)
            candidates.append({"path": path, "legacy": legacy, "size": size, "used": used})

        total = sum(item["size"] for item in candidates)
        count = len(candidates)
        target_entries = max(0, self.policy.paper_max_entries - max(0, reserve_entries))
        expired = {
            str(item["path"]) for item in candidates
            if now - item["used"] > self.policy.paper_ttl_seconds
        }
        ordered = sorted(candidates, key=lambda item: item["used"])
        selected: list[dict[str, Any]] = []
        for item in ordered:
            must_remove = str(item["path"]) in expired
            over_limit = total > self.policy.paper_max_bytes or count > target_entries
            if not must_remove and not over_limit:
                continue
            selected.append(item)
            total -= item["size"]
            count -= 1

        for item in selected:
            path = item["path"]
            report.eligible_entries += 1
            report.eligible_bytes += item["size"]
            lease = FileLease(path / ".active.lock")
            if not lease.acquire(blocking=False):
                report.active_skipped += 1
                total += item["size"]
                count += 1
                continue
            try:
                if dry_run:
                    continue
                trash = self.trash_dir / f"{path.name}-{uuid.uuid4().hex}"
                try:
                    os.replace(path, trash)
                except OSError:
                    logger.exception("缓存移入回收区失败：%s", path)
                    continue
                self._clear_db_prefix(path)
                if _safe_rmtree(trash, self.root):
                    report.removed_entries += 1
                    report.removed_files += 1
                    report.freed_bytes += item["size"]
            finally:
                lease.release()

        # 上次若崩溃在 rename 之后，回收区中的对象已不可能被任务引用，可继续删除。
        if not dry_run:
            for trash in list(self.trash_dir.iterdir()):
                if trash.is_dir() and not trash.is_symlink():
                    _safe_rmtree(trash, self.root)
            for manifest in list(self.sources_dir.glob("*.json")):
                data = _read_json(manifest)
                object_id = str(data.get("object_id") or "")
                if object_id and not (self.objects_dir / object_id).exists():
                    manifest.unlink(missing_ok=True)
            if self.store is not None:
                try:
                    self.store.clear_missing_pdf_paths()
                except Exception:
                    logger.exception("论文数据库失效路径对账失败")

        remaining_candidates = self._candidate_dirs(include_legacy)
        report.remaining_bytes = _tree_size(self.root)
        report.over_capacity = (
            report.remaining_bytes > self.policy.paper_max_bytes
            or len(remaining_candidates) > target_entries
        )
        return report


@contextlib.contextmanager
def sensitive_workspace(root: Path, prefix: str) -> Iterator[Path]:
    """创建权限收紧的敏感任务目录，并在成功、失败或 return 时一律删除。"""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    safe_prefix = re.sub(r"[^A-Za-z0-9_-]+", "-", prefix).strip("-") or "task"
    path = Path(tempfile.mkdtemp(prefix=f".{safe_prefix}-", dir=root))
    os.chmod(path, 0o700)
    lease = FileLease(path / ".active.lock")
    lease.acquire()
    try:
        yield path
    finally:
        _safe_rmtree(path, root)
        lease.release()


def cleanup_sensitive_roots(
    roots: list[Path],
    *,
    older_than_seconds: int,
    dry_run: bool = False,
) -> CleanupReport:
    """清理敏感目录中的崩溃遗留；只接受根目录下的直接子项。"""
    report = CleanupReport()
    cutoff = time.time() - older_than_seconds
    for root in roots:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            children = list(root.iterdir())
        except OSError:
            logger.exception("扫描敏感目录失败：%s", root)
            continue
        for child in children:
            report.scanned_entries += 1
            if child.is_symlink() or not _is_within(child, root):
                report.unsafe_skipped += 1
                continue
            try:
                modified = child.stat().st_mtime
            except OSError:
                continue
            if modified > cutoff:
                continue
            size = _tree_size(child)
            report.eligible_entries += 1
            report.eligible_bytes += size
            lease: FileLease | None = None
            if child.is_dir():
                lease = FileLease(child / ".active.lock")
                if not lease.acquire(blocking=False):
                    report.active_skipped += 1
                    continue
            if dry_run:
                if lease is not None:
                    lease.release()
                continue
            removed = _safe_rmtree(child, root) if child.is_dir() else False
            if child.is_file():
                try:
                    child.unlink()
                    removed = True
                except OSError:
                    logger.exception("清理敏感文件失败：%s", child)
            if lease is not None:
                lease.release()
            if removed:
                report.removed_entries += 1
                report.removed_files += 1
                report.freed_bytes += size
    report.remaining_bytes = sum(_tree_size(root) for root in roots)
    return report
