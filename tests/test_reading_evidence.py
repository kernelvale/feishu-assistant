from __future__ import annotations

import json

import pytest

from feishu_assistant.config import PROJECT_ROOT
from feishu_assistant.modules.reading_evidence import (
    SOURCE_CHUNK_CHARS,
    _chunk_base_title,
    _json_object,
    EvidenceBundle,
    EvidenceCard,
    EvidenceError,
    EvidenceIndexer,
    SourceChunk,
    split_source,
)
from feishu_assistant.modules.reading_workflow import ReadingWorkspace


class EvidenceKimi:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        prompt = str(messages[-1]["content"])
        chunk_id = prompt.split("chunk id：", 1)[-1].splitlines()[0].strip()
        return json.dumps({
            "chunk_id": chunk_id,
            "orientation": "本块说明研究问题与方法关系",
            "definitions": ["术语A：原文定义"],
            "claims": ["主张A"],
            "mechanisms": ["机制A"],
            "evidence": ["实验A"],
            "limitations": [],
            "open_questions": [],
            "relations": [],
            "figures": [],
            "formulas": [],
        }, ensure_ascii=False)


def test_split_source_preserves_tail_without_truncation():
    source = "# 开头\n\n" + "甲" * 55_000 + "\n\n## 附录\n\n" + "乙" * 31_000 + "尾部证据"
    chunks = split_source(source, chunk_chars=10_000)
    assert "".join(chunk.text for chunk in chunks) == source
    assert chunks[-1].text.endswith("尾部证据")


def test_exact_source_refs_fail_instead_of_silently_dropping_later_chunks():
    chunks = [
        SourceChunk("chunk-0001", "A", "甲" * 40, 1),
        SourceChunk("chunk-0002", "B", "乙" * 40, 2),
    ]
    bundle = EvidenceBundle(chunks, [EvidenceCard("chunk-0001", "A")], True)
    with pytest.raises(EvidenceError, match="不能静默截掉"):
        bundle.source_packet(["chunk-0001", "chunk-0002"], max_chars=60)


def test_duplicate_exact_refs_are_deduplicated_without_false_missing_error():
    chunk = SourceChunk("chunk-0001", "A", "原文", 1)
    bundle = EvidenceBundle([chunk], [], True)
    packet = bundle.source_packet(["chunk-0001", "chunk-0001"])
    assert packet.count("SOURCE chunk-0001") == 1


def test_hint_recall_scans_the_whole_chunk_not_only_first_8000_chars():
    bundle = EvidenceBundle([
        SourceChunk("chunk-0001", "无关", "FIRST", 1),
        SourceChunk("chunk-0002", "目标章节", "x" * 9000 + "late-needle", 2),
    ], [], True)
    packet = bundle.source_packet([], ["late-needle"])
    assert "SOURCE chunk-0002" in packet and "late-needle" in packet
    assert "SOURCE chunk-0001" not in packet


def test_code_fence_headings_do_not_split_source():
    source = "# Real\n```python\n## Fake\nprint(1)\n```\n## End\n正文"
    chunks = split_source(source)
    # 相邻小节归并成一个包（总大小远低于 24k），标题带「等 N 节」；围栏内伪标题只进正文
    assert [chunk.title for chunk in chunks] == ["Real 等2节"]
    assert "## Fake" in chunks[0].text


def test_long_evidence_index_is_checkpointed_and_reused(tmp_path):
    source = "# 综述\n\n" + "研究内容。" * 25_000 + "\n\n## 结论\n\n尾部结论"
    workspace = ReadingWorkspace(tmp_path)
    first_client = EvidenceKimi()
    bundle, calls = EvidenceIndexer(first_client, PROJECT_ROOT / "prompts").build(
        source, workspace=workspace,
    )
    assert bundle.long_mode is True
    assert calls == len(bundle.chunks) == first_client.calls
    assert any(chunk.text.endswith("尾部结论") for chunk in bundle.chunks)

    second_client = EvidenceKimi()
    resumed, resumed_calls = EvidenceIndexer(
        second_client, PROJECT_ROOT / "prompts",
    ).build(source, workspace=workspace)
    assert resumed_calls == 0
    assert second_client.calls == 0
    assert [card.chunk_id for card in resumed.cards] == [card.chunk_id for card in bundle.cards]


def test_workspace_round_trip_and_prefix_invalidation(tmp_path):
    workspace = ReadingWorkspace(tmp_path)
    workspace.put_text("section-001", "正文")
    workspace.put_json("section-001-review", {"pass": True})
    workspace.put_text("layer-001", "精讲")
    assert workspace.get_text("section-001") == "正文"
    assert workspace.get_json("section-001-review") == {"pass": True}

    workspace.invalidate_from(("section-",))
    assert workspace.get_text("section-001") is None
    assert workspace.get_json("section-001-review") is None
    assert workspace.get_text("layer-001") == "精讲"


def test_workspace_signature_invalidates_old_prompt_artifacts(tmp_path):
    workspace = ReadingWorkspace(tmp_path)
    workspace.ensure_signature("prompt-v1")
    workspace.put_text("section-001", "旧 prompt 产物")
    workspace.put_json("section-001-review", {"pass": True})

    workspace.ensure_signature("prompt-v2")
    assert workspace.get_text("section-001") is None
    assert workspace.get_json("section-001-review") is None
    assert workspace.get_json("workflow-signature") == {"signature": "prompt-v2"}


def test_nested_same_char_fences_do_not_close_outer():
    """同字符围栏嵌套：```` 外层内的 ``` 是内容不是闭合（CommonMark 长度规则）。"""
    md = "# Real\n````markdown\n```\n## Fake\n```\n````\n## End\n正文"
    chunks = split_source(md)
    assert [chunk.title for chunk in chunks] == ["Real 等2节"]
    assert "## Fake" in chunks[0].text


def test_longer_fence_closes_shorter_outer():
    """收尾 fence 长于开头也合规闭合（``` 开头、```` 收尾）。"""
    md = "# Real\n```\n## Fake\n````\n## End\n正文"
    chunks = split_source(md)
    assert [chunk.title for chunk in chunks] == ["Real 等2节"]
    assert "## Fake" in chunks[0].text


def test_split_source_merges_small_sections_into_packs():
    """小节密集论文：相邻小块归并到 ≤24k 的包，块数不爆炸，索引不超规划上限。"""
    sections = "\n".join(f"## 第{i}节\n" + "正文。" * 1000 + "\n" for i in range(1, 67))
    chunks = split_source(sections)
    assert len(chunks) < 20  # 66 个小节归并成十几个包，而不是 66 块
    assert all(len(c.text) <= SOURCE_CHUNK_CHARS for c in chunks)
    total = sum(len(c.text) for c in chunks)
    assert total == len(sections)  # 不丢一个字符
    assert "等" in chunks[0].title and "节" in chunks[0].title


def test_split_source_merge_preserves_big_section_splitting():
    """归并不破坏既有行为：超长章节仍按段落切片，小章节单独成块。"""
    big = "## 大章\n" + "长段落。" * 8000 + "\n"
    small = "## 小章\n" + "短。" * 100 + "\n"
    chunks = split_source(big + small)
    assert len(chunks) >= 2
    assert all(len(c.text) <= SOURCE_CHUNK_CHARS for c in chunks)


def test_source_packet_auto_includes_sibling_fragments():
    """计划只引用「大章（片段 1）」时，同基名的片段 2/3 自动并入，不拿半截论述。"""
    chunks = [
        SourceChunk("chunk-0001", "大章（片段 1）", "上半段。", 1),
        SourceChunk("chunk-0002", "大章（片段 2）", "下半段。", 2),
        SourceChunk("chunk-0003", "大章（片段 3）", "收尾段。", 3),
        SourceChunk("chunk-0004", "别的章", "无关。", 4),
    ]
    bundle = EvidenceBundle(chunks=chunks, cards=[], long_mode=True)
    packet = bundle.source_packet(["chunk-0001"])
    assert "上半段" in packet and "下半段" in packet and "收尾段" in packet
    assert "无关" not in packet
    assert packet.index("上半段") < packet.index("下半段") < packet.index("收尾段")  # 原文顺序


def test_source_packet_sibling_match_tolerates_pack_suffix():
    """片段被归并进「等 N 节」的包时，基名仍能正确剥离并匹配亲属。"""
    assert _chunk_base_title("大章（片段 3） 等3节") == "大章"
    assert _chunk_base_title("大章（片段 1）") == "大章"
    assert _chunk_base_title("普通章节") == "普通章节"


def test_planner_material_accepts_large_index_within_cap():
    """45 图 111 页级综述（~85k 索引）在放宽后的上限内放行；超 160k 仍拒绝。"""
    fat = "x" * 4_000
    cards = [EvidenceCard(chunk_id=f"chunk-{i:04d}", section=f"s{i}", claims=[fat])
             for i in range(1, 22)]  # 21 张 × 4k ≈ 85k
    bundle = EvidenceBundle(chunks=[], cards=cards, long_mode=True)
    assert len(bundle.planner_material("")) > 80_000  # 超旧上限但不再报错

    too_many = [EvidenceCard(chunk_id=f"chunk-{i:04d}", section=f"s{i}", claims=[fat])
                for i in range(1, 45)]  # 44 张 ≈ 180k
    bundle2 = EvidenceBundle(chunks=[], cards=too_many, long_mode=True)
    with pytest.raises(EvidenceError, match="需要增加分层归并"):
        bundle2.planner_material("")


def test_json_object_extracts_prose_wrapped_json():
    """编码向模型把 JSON 裹在解释文字里：首尾大括号跨度提取，不再判非法。"""
    wrapped = '好的，以下是证据卡：\n{"orientation": "概览", "claims": ["c1"]}\n以上。'
    assert _json_object(wrapped)["orientation"] == "概览"
    with pytest.raises(EvidenceError, match="非法 JSON"):
        _json_object("完全不是 JSON")


def test_card_build_retries_once_on_invalid_json(tmp_path):
    """证据卡首次返回非法 JSON 时带修正重试一次；重试成功正常建卡。"""
    source = "# 综述\n\n" + "研究内容。" * 25_000 + "\n\n## 结论\n\n尾部结论"
    workspace = ReadingWorkspace(tmp_path)

    class FlakyKimi(EvidenceKimi):
        def chat(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return "这不是 JSON，是解释文字。"
            return super().chat(messages, **kwargs)

    client = FlakyKimi()
    bundle, calls = EvidenceIndexer(client, PROJECT_ROOT / "prompts").build(
        source, workspace=workspace)
    assert bundle.long_mode is True and bundle.cards
    assert calls > len(bundle.chunks)  # 首次失败的块多了一次修正调用


def test_card_build_falls_back_to_source_navigation_after_two_non_json_outputs(tmp_path):
    """API 成功但连续返回普通文本时，不应把格式偏差升级成整篇论文失败。"""
    source = "# 综述\n\n" + "这是可核对的原文证据，解释了研究问题与方法。" * 8_000
    workspace = ReadingWorkspace(tmp_path)

    class PlainTextKimi:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages, **kwargs):
            self.calls += 1
            return "这里是正常的文字回答，但不是 JSON。"

    client = PlainTextKimi()
    bundle, calls = EvidenceIndexer(client, PROJECT_ROOT / "prompts").build(
        source, workspace=workspace,
    )

    assert bundle.long_mode is True
    assert calls == client.calls == 2 * len(bundle.chunks)
    assert all(card.orientation.startswith("原文导航摘录：") for card in bundle.cards)
    assert all(not card.claims for card in bundle.cards)  # 兜底不猜测论文结论
