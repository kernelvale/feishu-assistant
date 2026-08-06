from __future__ import annotations

import json
import threading
import time

import pytest

from feishu_assistant.config import PROJECT_ROOT
from feishu_assistant.kimi import FrozenKimiClient, KimiError
from feishu_assistant.modules.reading_evidence import (
    SHORT_PAPER_CHARS,
    SOURCE_CHUNK_CHARS,
    _chunk_base_title,
    _json_object,
    EvidenceBundle,
    EvidenceCard,
    EvidenceError,
    EvidenceIndexer,
    paper_budget_profile,
    SourceChunk,
    paper_size_tier,
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


def test_exact_source_refs_over_budget_packed_transparently_not_killed():
    """第 8 步收口：strict refs 超预算不再判死——按块预算装包（首块必装），
    装不下的块在包尾透明标注（不静默截掉）。"""
    chunks = [
        SourceChunk("chunk-0001", "A", "甲" * 40, 1),
        SourceChunk("chunk-0002", "B", "乙" * 40, 2),
    ]
    bundle = EvidenceBundle(chunks, [EvidenceCard("chunk-0001", "A")], True)
    packet = bundle.source_packet(["chunk-0001", "chunk-0002"], max_chars=60)
    assert "SOURCE chunk-0001" in packet          # 首块必装（按预算截断内容）
    assert "SOURCE chunk-0002" not in packet      # 装不下的块让位
    assert "chunk-0002" in packet                 # 但未装入本身透明标注
    assert "单包容量限制未装入" in packet


def test_missing_source_refs_still_fail_closed():
    chunks = [SourceChunk("chunk-0001", "A", "甲" * 10, 1)]
    bundle = EvidenceBundle(chunks, [EvidenceCard("chunk-0001", "A")], True)
    with pytest.raises(EvidenceError, match="不存在的原文块"):
        bundle.source_packet(["chunk-0001", "chunk-9999"])


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


def test_source_packet_info_marks_fallback_and_budget_truncation():
    bundle = EvidenceBundle([
        SourceChunk("chunk-0001", "目标", "甲" * 80, 1),
        SourceChunk("chunk-0002", "后续", "乙" * 80, 2),
    ], [], True)

    fallback = bundle.source_packet_info([], ["目标"], max_chars=200)
    assert fallback.used_fallback is True
    assert fallback.complete is False

    exact = bundle.source_packet_info(["chunk-0001", "chunk-0002"], max_chars=60)
    assert exact.omitted_ids or exact.truncated_ids
    assert exact.complete is False


def test_code_fence_headings_do_not_split_source():
    source = "# Real\n```python\n## Fake\nprint(1)\n```\n## End\n正文"
    chunks = split_source(source)
    # 相邻小节归并成一个包（总大小远低于 24k），标题带「等 N 节」；围栏内伪标题只进正文
    assert [chunk.title for chunk in chunks] == ["Real 等2节"]
    assert "## Fake" in chunks[0].text


def test_long_evidence_index_is_checkpointed_and_reused(tmp_path):
    source = "# 综述\n\n" + "研究内容。" * 45_000 + "\n\n## 结论\n\n尾部结论"
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


def test_planner_material_uses_layered_merge_for_oversized_index():
    """超出 160k 的证据索引分层归并，不丢掉尾部 chunk。"""
    fat = "x" * 4_000
    cards = [EvidenceCard(chunk_id=f"chunk-{i:04d}", section=f"s{i}", claims=[fat])
             for i in range(1, 22)]  # 21 张 × 4k ≈ 85k
    bundle = EvidenceBundle(chunks=[], cards=cards, long_mode=True)
    assert len(bundle.planner_material("")) > 80_000  # 超旧上限但不再报错

    too_many = [EvidenceCard(chunk_id=f"chunk-{i:04d}", section=f"s{i}", claims=[fat])
                for i in range(1, 45)]  # 44 张 ≈ 180k
    bundle2 = EvidenceBundle(chunks=[], cards=too_many, long_mode=True)
    material = bundle2.planner_material("")
    assert len(material) <= 160_000
    assert "分层归并" in material
    for index in range(1, 45):
        assert f"chunk-{index:04d}" in material


def test_json_object_extracts_prose_wrapped_json():
    """编码向模型把 JSON 裹在解释文字里：首尾大括号跨度提取，不再判非法。"""
    wrapped = '好的，以下是证据卡：\n{"orientation": "概览", "claims": ["c1"]}\n以上。'
    assert _json_object(wrapped)["orientation"] == "概览"
    with pytest.raises(EvidenceError, match="非法 JSON"):
        _json_object("完全不是 JSON")


def test_card_build_retries_once_on_invalid_json(tmp_path):
    """证据卡首次返回非法 JSON 时带修正重试一次；重试成功正常建卡。"""
    source = "# 综述\n\n" + "研究内容。" * 45_000 + "\n\n## 结论\n\n尾部结论"
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
    source = "# 综述\n\n" + "这是可核对的原文证据，解释了研究问题与方法。" * 10_000
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


def test_planner_material_second_layer_keeps_every_chunk_reference():
    """卡片级压缩仍过长时进入分组层，所有 source ref 继续可见。"""
    fat = "论据" * 300
    cards = [
        EvidenceCard(
            chunk_id=f"chunk-{index:04d}",
            section=f"章节 {index}",
            orientation=f"方向 {index} " + fat,
            claims=[fat, fat, fat],
            mechanisms=[fat, fat, fat],
            evidence=[fat, fat, fat],
            limitations=[fat, fat, fat],
        )
        for index in range(1, 501)
    ]
    material = EvidenceBundle(chunks=[], cards=cards, long_mode=True).planner_material("")

    assert len(material) <= 160_000
    assert "evidence-group-" in material
    for index in range(1, 501):
        assert f"chunk-{index:04d}" in material


def test_single_reasoning_exhausted_chunk_degrades_only_that_chunk(tmp_path):
    """一个 chunk 双败只生成该块导航卡，其余真实证据卡和检查点保留。"""
    source = (
        "# 第一章\n\n" + "甲段内容。" * 15_000
        + "\n\n## 第二章\n\n" + "乙段内容。" * 15_000
        + "\n\n## 第三章\n\n" + "丙段内容。" * 15_000
    )

    class OneChunkFails:
        def __init__(self) -> None:
            self.calls: dict[str, int] = {}

        def chat(self, messages, **kwargs):
            prompt = str(messages[-1]["content"])
            chunk_id = prompt.split("## 原文块 ", 1)[1].split()[0].strip()
            self.calls[chunk_id] = self.calls.get(chunk_id, 0) + 1
            if chunk_id == "chunk-0002":
                raise KimiError("reasoning-exhaustion: 正文为空")
            return json.dumps({
                "chunk_id": chunk_id,
                "orientation": f"{chunk_id} 的真实证据卡",
                "definitions": [],
                "claims": ["可核对主张"],
                "mechanisms": [],
                "evidence": [],
                "limitations": [],
                "open_questions": [],
                "relations": [],
                "figures": [],
                "formulas": [],
            }, ensure_ascii=False)

    client = OneChunkFails()
    workspace = ReadingWorkspace(tmp_path)
    bundle, calls = EvidenceIndexer(client, PROJECT_ROOT / "prompts").build(
        source, workspace=workspace,
    )

    assert calls == sum(client.calls.values())
    assert client.calls["chunk-0002"] == 2
    assert list(bundle.degraded_chunks) == ["chunk-0002"]
    assert bundle.cards[1].orientation.startswith("原文导航摘录：")
    assert all(
        card.orientation.endswith("的真实证据卡")
        for index, card in enumerate(bundle.cards)
        if index != 1
    )
    assert workspace.get_json("evidence-chunk-0002")["_degraded_reason"]
    assert bundle.index_metrics["degraded_chunks"] == 1


def test_real_client_path_runs_evidence_calls_concurrently():
    """线程池本身能达到多路 inflight；若生产日志仍串行，瓶颈位于网关侧。"""
    source = "\n\n".join(
        f"## 章节 {index}\n" + (f"内容 {index}。" * 12_000)
        for index in range(1, 5)
    )

    class BlockingRecorder:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.active = 0
            self.max_active = 0

        def chat(self, messages, **kwargs):
            prompt = str(messages[-1]["content"])
            chunk_id = prompt.split("## 原文块 ", 1)[1].split()[0].strip()
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(0.04)
                return json.dumps({
                    "chunk_id": chunk_id,
                    "orientation": f"{chunk_id} 索引",
                    "definitions": [], "claims": ["主张"], "mechanisms": [],
                    "evidence": [], "limitations": [], "open_questions": [],
                    "relations": [], "figures": [], "formulas": [],
                }, ensure_ascii=False)
            finally:
                with self.lock:
                    self.active -= 1

    # V9 生产路径传入的是 FrozenKimiClient；旧实现只识别 KimiClient，
    # 导致 workers 永远为 1。这里钉住真实包装类型也必须进入并发路径。
    recorder = BlockingRecorder()
    client = object.__new__(FrozenKimiClient)
    client.chat = recorder.chat
    bundle, _ = EvidenceIndexer(client, PROJECT_ROOT / "prompts").build(source)

    assert bundle.index_metrics["workers"] > 1
    assert bundle.index_metrics["max_inflight_calls"] > 1
    assert recorder.max_active > 1


# ---- 预算按论文尺寸分路：统一分档 + 长文分层来源包 ----

def test_paper_size_tier_boundaries():
    """统一分档阈值 200k（估算宁大勿小，用户裁定）：≤ 阈值全文直通，超出走长文。"""
    assert SHORT_PAPER_CHARS == 200_000
    assert paper_size_tier(199_999) == "short"
    assert paper_size_tier(200_000) == "short"
    assert paper_size_tier(200_001) == "long"
    assert paper_size_tier(0) == "short"


def test_budget_profile_scales_input_output_and_draft_floor_with_paper_size():
    compact = paper_budget_profile(10_000)
    standard = paper_budget_profile(150_000)
    long = paper_budget_profile(417_000)
    extra = paper_budget_profile(700_000)

    assert [item.tier for item in (compact, standard, long, extra)] == [
        "compact", "standard", "long", "extra_long",
    ]
    assert compact.author_packet_chars < standard.author_packet_chars < long.author_packet_chars
    assert compact.audit_output_tokens < standard.audit_output_tokens < long.audit_output_tokens
    assert long.recommended_draft_chars >= 50_000
    assert long.minimum_draft_chars >= 32_000
    assert extra.author_parts == 4


def test_evidence_retry_uses_low_reasoning_only_after_high_quality_first_attempt():
    source = "# 综述\n\n" + "研究内容。" * 45_000

    class FirstInvalidThenValid(EvidenceKimi):
        def __init__(self) -> None:
            super().__init__()
            self.kwargs_seen: list[dict] = []

        def chat(self, messages, **kwargs):
            self.kwargs_seen.append(dict(kwargs))
            if len(self.kwargs_seen) == 1:
                return "not-json"
            return super().chat(messages, **kwargs)

    client = FirstInvalidThenValid()
    bundle, _ = EvidenceIndexer(client, PROJECT_ROOT / "prompts").build(source)

    assert bundle.long_mode is True
    assert client.kwargs_seen[0]["reasoning_effort"] == "high"
    assert client.kwargs_seen[1]["reasoning_effort"] == "low"


_SURVEY_WORDS = [
    "归纳偏置", "奖励整形", "课程学习", "策略蒸馏", "轨迹压缩", "约束传播",
    "反向信用", "探索噪声", "价值分解", "离线微调", "对抗鲁棒", "符号接地",
    "工具调用", "记忆回流", "任务分解", "自我批判", "多轮规划", "安全护栏",
]


def _survey_bundle() -> tuple[EvidenceBundle, list[SourceChunk]]:
    """2509 规模假综述：18 章 × ~18k ≈ 325k 字符，每块配一张真实证据卡。"""
    md = "\n\n".join(
        f"## 第 {i} 章\n" + (f"本章讨论{_SURVEY_WORDS[i - 1]}。" * 2_000)
        for i in range(1, 19)
    )
    chunks = split_source(md)
    cards = [
        EvidenceCard(
            chunk_id=chunk.chunk_id,
            section=chunk.title,
            orientation=f"{chunk.title} 概览",
            claims=[f"{chunk.title} 的核心主张"],
        )
        for chunk in chunks
    ]
    return EvidenceBundle(chunks=chunks, cards=cards, long_mode=True), chunks


def test_layered_source_packet_covers_every_chunk_within_budget():
    """长文分层包：命中块全文 + 未命中块证据卡摘要，不超预算、不截尾、不降级。"""
    bundle, chunks = _survey_bundle()
    refs = [chunks[1].chunk_id, chunks[7].chunk_id, chunks[16].chunk_id]
    packet = bundle.layered_source_packet_info(refs, hints=[], max_chars=250_000)

    assert len(packet.text) <= 250_000
    assert packet.used_fallback is False          # 不再报「退回关键词召回」降级
    assert not packet.omitted_ids and not packet.truncated_ids
    assert packet.complete is True
    for chunk in chunks:
        assert chunk.chunk_id in packet.text      # 任何块都不截尾丢弃
    for chunk in (chunks[1], chunks[7], chunks[16]):
        assert f"SOURCE {chunk.chunk_id}" in packet.text
        assert chunk.text in packet.text          # 命中块全文装入
    assert f"SUMMARY {chunks[0].chunk_id}" in packet.text
    assert f"SUMMARY {chunks[-1].chunk_id}" in packet.text  # 尾部块以摘要保留
    assert "的核心主张" in packet.text             # 未命中块用证据卡 statements


def test_layered_source_packet_demotes_overflow_fulltext_to_summary():
    """全文候选超预算时，多出的候选自动降为摘要块：不省略、不截断、不超预算。"""
    bundle, chunks = _survey_bundle()
    refs = [chunk.chunk_id for chunk in chunks[:6]]
    packet = bundle.layered_source_packet_info(refs, hints=[], max_chars=100_000)

    assert len(packet.text) <= 100_000
    assert packet.used_fallback is False
    assert not packet.omitted_ids and not packet.truncated_ids
    full = [chunk for chunk in chunks if f"SOURCE {chunk.chunk_id}" in packet.text]
    assert 0 < len(full) < 6                      # 装不下的候选降级为摘要
    for chunk in chunks:
        assert chunk.chunk_id in packet.text      # 仍没有任何块被丢弃


def test_layered_source_packet_promotes_hint_matched_chunks_without_fallback():
    """无精确引用时提示词只决定哪些块提升为全文，不再构成召回降级。"""
    bundle, chunks = _survey_bundle()
    packet = bundle.layered_source_packet_info([], hints=["奖励整形 设计"], max_chars=250_000)

    assert packet.used_fallback is False
    assert f"SOURCE {chunks[1].chunk_id}" in packet.text   # 提示词命中的块给全文
    assert f"SUMMARY {chunks[0].chunk_id}" in packet.text  # 其余块摘要兜底


def test_evidence_card_first_attempt_preserves_reasoning_quality():
    """证据卡首轮用 high；只有格式/空输出纠错才允许降为 low。"""
    source = "# 综述\n\n" + "研究内容。" * 45_000 + "\n\n## 结论\n\n尾部结论"

    class KwargRecorder(EvidenceKimi):
        def __init__(self) -> None:
            super().__init__()
            self.kwargs_seen: list[dict] = []

        def chat(self, messages, **kwargs):
            self.kwargs_seen.append(dict(kwargs))
            return super().chat(messages, **kwargs)

    client = KwargRecorder()
    bundle, calls = EvidenceIndexer(client, PROJECT_ROOT / "prompts").build(source)
    assert bundle.long_mode is True and calls > 0
    assert client.kwargs_seen
    assert all(kwargs.get("reasoning_effort") == "high" for kwargs in client.kwargs_seen)
    assert all("low_reasoning" not in kwargs for kwargs in client.kwargs_seen)
