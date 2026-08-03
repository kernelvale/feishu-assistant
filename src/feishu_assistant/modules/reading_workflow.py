"""精读生成工作区：原子保存阶段产物，并用文件租约支持崩溃后继续。

工作区位于论文内容寻址缓存内部。PDF/解析缓存负责保住原始材料；本模块只保住
证据索引、教学计划、章节正文和审核结果。进程崩溃后 flock 自动释放，下一次
同源精读会复用已经验收通过的阶段产物。
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ..lifecycle import FileLease


PIPELINE_VERSION = "reading-v3"


class ReadingWorkspace:
    """一个论文、一版生成协议对应的持久化工作区。"""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root) / PIPELINE_VERSION
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    @contextmanager
    def lease(self) -> Iterator["ReadingWorkspace"]:
        """同一论文同一协议只允许一个生成者，避免并发覆盖阶段产物。"""
        with FileLease(self.root / ".generation.lock"):
            yield self

    def _path(self, key: str, suffix: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in key)
        return self.root / f"{safe}.{suffix}"

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass

    def get_text(self, key: str) -> str | None:
        try:
            return self._path(key, "md").read_text(encoding="utf-8")
        except OSError:
            return None

    def put_text(self, key: str, value: str) -> None:
        self._atomic_write(self._path(key, "md"), value)

    def get_json(self, key: str) -> dict[str, Any] | list[Any] | None:
        try:
            return json.loads(self._path(key, "json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def put_json(self, key: str, value: Any) -> None:
        self._atomic_write(
            self._path(key, "json"),
            json.dumps(value, ensure_ascii=False, indent=2),
        )

    def ensure_signature(self, signature: str) -> None:
        """Prompt/协议变化时原子淘汰旧阶段产物，禁止静默复用旧语义缓存。"""
        manifest = self.get_json("workflow-signature")
        if isinstance(manifest, dict) and manifest.get("signature") == signature:
            return
        for path in self.root.iterdir():
            if not path.is_file() or path.is_symlink():
                continue
            if path.name in {".generation.lock"} or path.name.startswith(".workflow-signature"):
                continue
            if path.suffix in {".md", ".json"}:
                path.unlink(missing_ok=True)
        self.put_json("workflow-signature", {"signature": signature})

    def invalidate_from(self, prefixes: tuple[str, ...]) -> None:
        """协议或上游产物变化时，只删除指定下游阶段；文件删除范围受工作区限制。"""
        for path in self.root.iterdir():
            if not path.is_file() or path.is_symlink():
                continue
            if any(path.name.startswith(prefix) for prefix in prefixes):
                path.unlink(missing_ok=True)
