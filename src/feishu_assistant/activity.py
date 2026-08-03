"""全局活动登记：实时记录正在运行的底层操作（LLM 调用 / 子进程 / 云文档写入）。

状态条（StatusMessenger）每帧读取 snapshot() 渲染「活动：LLM·k3（45s）」；
为空表示没有任务在跑（等待/卡死），转圈停住——这就是"活着"探针。

用法：
    from feishu_assistant.activity import track
    with track("MinerU 解析"):
        parse_pdf(...)
"""

from __future__ import annotations

import contextlib
import threading
import time


class ActivityRegistry:
    """线程安全的在途操作登记表。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[int, tuple[str, float]] = {}
        self._seq = 0

    @contextlib.contextmanager
    def track(self, label: str):
        with self._lock:
            self._seq += 1
            ident = self._seq
            self._items[ident] = (label, time.time())
        try:
            yield
        finally:
            with self._lock:
                self._items.pop(ident, None)

    def snapshot(self) -> list[tuple[str, float]]:
        """[(label, 已运行秒数)]，按启动顺序。"""
        now = time.time()
        with self._lock:
            return [(label, now - t0) for label, t0 in self._items.values()]


REGISTRY = ActivityRegistry()
track = REGISTRY.track
