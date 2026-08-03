"""论文向量索引：fastembed(bge-large-en-v1.5, onnx CPU) + sqlite-vec。

- 精读完成时把 标题+TL;DR+关键词 向量化入库（papers.db 同库 vec0 虚表）
- 日报与精选排序时检索库内 top-k 相似论文（cosine 距离，超阈值/库空则跳过）
- embedder 可注入（测试用假向量，无需下载模型）；真实模型惰性加载
  （首次从 HF/modelscope 下载权重，可用 FASTEMBED_CACHE_PATH / HF_ENDPOINT 控制）
"""

from __future__ import annotations

import logging
import sqlite3
import struct
import time
from pathlib import Path
from typing import Any, Callable

from .config import PROJECT_ROOT

logger = logging.getLogger("feishu_assistant.vectors")

DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "papers.db"
DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"  # fastembed 本版无 bge-m3，en-v1.5 对英文论文更稳
EMBEDDING_DIM = 1024
DEFAULT_TOP_K = 3
DEFAULT_MAX_DISTANCE = 0.45  # cosine 距离阈值（越小越严格）；超过即视为"不相似，不硬凑"

Embedder = Callable[[list[str]], list[list[float]]]


def _default_embedder() -> Embedder:
    """惰性加载 fastembed bge-m3（首次下载权重，onnx CPU 推理）。"""
    from fastembed import TextEmbedding  # 惰性导入

    logger.info("加载向量模型 %s（首次需下载权重）……", DEFAULT_MODEL)
    model = TextEmbedding(DEFAULT_MODEL)

    def embed(texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in model.embed(texts)]

    return embed


def _serialize_f32(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


class PaperVectorIndex:
    """paper_vecs 虚表：arxiv_id/标题 + 向量。实例方法全部线程安全（每次新连接）。"""

    def __init__(
        self,
        db_path: Path | str = DEFAULT_DB_PATH,
        embedder: Embedder | None = None,
        max_distance: float = DEFAULT_MAX_DISTANCE,
    ) -> None:
        self.db_path = str(db_path)
        self._embedder = embedder  # None 时首次使用才加载真实模型
        self.max_distance = max_distance
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS paper_vecs USING vec0("
                f"  arxiv_id TEXT PRIMARY KEY,"
                f"  title TEXT,"
                f"  embedding FLOAT[{EMBEDDING_DIM}] distance_metric=cosine"
                f")"
            )

    def _conn(self) -> sqlite3.Connection:
        import sqlite_vec  # 惰性导入

        conn = sqlite3.connect(self.db_path, timeout=5)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = _default_embedder()
        return self._embedder

    @staticmethod
    def _doc_text(title: str, tldr: str, keywords: list[str] | None) -> str:
        kw = "、".join(keywords or [])
        return f"{title}\n{tldr}\n关键词：{kw}".strip()

    def add_paper(
        self, arxiv_id: str, title: str, tldr: str, keywords: list[str] | None = None
    ) -> bool:
        """向量化并入库（同 arxiv_id 覆盖）。失败记日志返回 False（不阻断主流程）。"""
        if not arxiv_id:
            return False
        try:
            vec = self.embedder([self._doc_text(title, tldr, keywords)])[0]
            with self._conn() as conn:
                conn.execute("DELETE FROM paper_vecs WHERE arxiv_id = ?", (arxiv_id,))
                conn.execute(
                    "INSERT INTO paper_vecs (arxiv_id, title, embedding) VALUES (?, ?, ?)",
                    (arxiv_id, title, _serialize_f32(vec)),
                )
            logger.info("向量入库：%s《%s》", arxiv_id, title)
            return True
        except Exception as exc:
            logger.warning("向量入库失败（跳过，不影响精读）：%s", exc)
            return False

    def search(
        self, title: str, tldr: str = "", keywords: list[str] | None = None,
        top_k: int = DEFAULT_TOP_K, exclude_arxiv_id: str = "",
    ) -> list[tuple[str, str, float]]:
        """库内相似检索，返回 [(arxiv_id, title, distance)]，按距离升序、阈值过滤。"""
        try:
            vec = self.embedder([self._doc_text(title, tldr, keywords)])[0]
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT arxiv_id, title, distance FROM paper_vecs"
                    " WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                    (_serialize_f32(vec), top_k + (1 if exclude_arxiv_id else 0)),
                ).fetchall()
        except Exception as exc:
            logger.warning("向量检索失败（按无相似处理）：%s", exc)
            return []
        hits = [
            (r[0], r[1], float(r[2])) for r in rows
            if r[0] != exclude_arxiv_id and float(r[2]) <= self.max_distance
        ]
        return hits[:top_k]

    def count(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) FROM paper_vecs").fetchone()
        return int(row[0])
