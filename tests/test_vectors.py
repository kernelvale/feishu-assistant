"""vectors.py 测试：假 embedder + 真 sqlite-vec。入库/检索/阈值/排除/upsert/空库。"""

from __future__ import annotations

from feishu_assistant.vectors import PaperVectorIndex


def fake_embedder(texts: list[str]) -> list[list[float]]:
    """按关键词给确定性单位向量（1024 维）。"""
    vecs = []
    for t in texts:
        v = [0.0] * 1024
        if "agent" in t.lower():
            v[0] = 1.0
        elif "chess" in t.lower():
            v[1] = 1.0
        else:
            v[2] = 1.0
        vecs.append(v)
    return vecs


def make_index(tmp_path, max_distance=0.45) -> PaperVectorIndex:
    return PaperVectorIndex(tmp_path / "v.db", embedder=fake_embedder, max_distance=max_distance)


def test_add_and_search_nearest(tmp_path):
    idx = make_index(tmp_path)
    idx.add_paper("2603.03329", "AutoHarness agent harness", "用代码 harness 包裹 agent", ["agent"])
    idx.add_paper("1111.11111", "Chess engine", "国际象棋引擎", ["chess"])
    hits = idx.search("agent 系统", "agent harness 方法")
    assert hits and hits[0][0] == "2603.03329"  # agent 向量最近
    assert all(h[0] != "1111.11111" or h[2] <= 0.45 for h in hits)


def test_search_excludes_self(tmp_path):
    idx = make_index(tmp_path)
    idx.add_paper("2603.03329", "AutoHarness agent", "tldr", ["agent"])
    hits = idx.search("AutoHarness agent", "tldr", exclude_arxiv_id="2603.03329")
    assert hits == []


def test_threshold_filters_dissimilar(tmp_path):
    idx = make_index(tmp_path, max_distance=0.01)
    idx.add_paper("1111.11111", "Chess", "chess engine", ["chess"])
    assert idx.search("agent 系统", "agent") == []  # 距离超阈值，不硬凑


def test_upsert_same_arxiv_id(tmp_path):
    idx = make_index(tmp_path)
    idx.add_paper("x", "标题一", "a", [])
    idx.add_paper("x", "标题二", "b", [])
    assert idx.count() == 1
    assert idx.search("标题二", "b")[0][1] == "标题二"


def test_empty_index_returns_empty(tmp_path):
    idx = make_index(tmp_path)
    assert idx.search("anything", "") == []
    assert idx.count() == 0


def test_add_without_arxiv_id_skipped(tmp_path):
    idx = make_index(tmp_path)
    assert idx.add_paper("", "t", "s", []) is False
