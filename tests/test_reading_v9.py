"""V9 叙事优先精读管线的端到端行为测试。

这些测试刻意只使用一个按 prompt 路由的 fake client，不复用 V7/V8 的测试桩。
目标不是证明模型文风，而是锁住 V9 最重要的发布语义：两次专项审查、精确小补丁、
验证失败回滚、审查不可用不伪造通过、20% 改动上限，以及 reading-v5 检查点续跑。
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest

from feishu_assistant.kimi import KimiError
from feishu_assistant.modules.reading import ReadingError
from feishu_assistant.modules.reading_evidence import (
    EvidenceBundle,
    EvidenceCard,
    SourceChunk,
    split_source,
)
from feishu_assistant.modules.reading_v9 import (
    AUTHOR_PACKET_CHARS_LONG,
    FACT_PACKET_CHARS_LONG,
    V9ReadingEngine,
    _brief_takeaway_checklist,
    _digest_payload,
    _parse_audit,
)
from feishu_assistant.modules.reading_v9_models import (
    AuditReport,
    EditCard,
    NarrativeBrief,
    RevisionEdit,
    V9ValidationError,
    VerificationResult,
)
from feishu_assistant.modules.reading_v9_quality import (
    apply_edit_script,
    duplicate_candidates,
    mechanical_quality_report,
    parse_revision_edits,
    select_edit_cards,
)
from feishu_assistant.modules.reading_workflow import (
    ReadingWorkspace,
    V9_PIPELINE_VERSION,
)


PAPER_MD = dedent(
    """
    # A Small Harness Paper

    ## Abstract

    本文研究一个执行 harness 如何通过记录失败轨迹、修改工具约束并重新运行任务，
    改善代理在长程任务中的稳定性。论文把 harness 视为策略行为的一部分，而不是
    与策略无关的外壳。

    ## Problem

    单次生成经常把早期局部错误带到后续步骤。只增加模型采样次数不能保证失败信息
    被利用，也不能保证工具调用满足环境约束。

    ## Method

    方法先执行候选策略，再把失败轨迹压缩为结构化反馈，随后只修改与失败有关的
    harness 规则并重新执行。核心循环由执行、诊断、定点修改和再次执行四步组成。

    ## Results

    完整方法在主要任务上优于只重采样的基线。消融实验显示，移除结构化失败反馈后，
    改善明显缩小。论文没有声称该结果可以直接外推到所有代理任务。

    ## Limitations

    额外执行会增加成本；如果诊断错误，后续修改也可能放大偏差。
    """
).strip()


DUPLICATE_SPAN_A = (
    "这里第一次完整解释核心循环：系统先运行候选策略，记录失败轨迹，再把轨迹压缩成"
    "针对工具约束的反馈，最后只修改相关规则并重新执行。四步之间存在明确因果关系，"
    "因此读者可以看见失败信息怎样真正进入下一次行为。"
)
DUPLICATE_SPAN_B = (
    "换一种说法，核心循环仍然是先执行候选策略，再收集失败轨迹，把轨迹整理成工具"
    "约束反馈，随后修改对应规则并重新执行。这个段落没有加入新的证据、比较或边界，"
    "只是把前面的四步流程再次完整讲了一遍。"
)


CLEAN_AUTHOR_BODY = dedent(
    f"""
    ## 问题不只是模型会不会回答

    长程任务里的失败通常不是某一句答案错了，而是早期工具调用留下的局部偏差一路传到
    后面。普通重采样只会再次生成候选，并不会保证系统理解上一次为什么失败。于是，
    即使模型本身能力足够，执行环境也可能持续把它推回同一种错误路径。

    ## 论文真正改变了什么

    论文的关键转向，是把 harness 看成策略行为的一部分。它记录哪些动作失败、环境拒绝
    了什么输入，并决定下一轮允许模型看到哪些反馈。这意味着执行外壳不再只是被动承载
    模型输出，而会改变后续搜索方向。

    ## 核心循环怎样工作

    {DUPLICATE_SPAN_A}

    一个具体例子是：代理第一次提交了格式错误的工具参数。系统不是笼统要求“再试一次”，
    而是保留失败位置，只收紧该参数的格式约束。下一轮仍沿用已经正确的步骤，因此修正
    发生在错误点附近，不需要把整条轨迹推倒重来。

    ## 实验为什么能支持这个解释

    完整方法优于只重采样的基线，而移除结构化失败反馈后，改善明显缩小。这组比较支持
    “失败信息进入下一轮规则”是有效组成部分，但它没有证明同一机制在所有代理任务中都
    必然有效。证据的价值在于区分了定点反馈与单纯增加尝试次数。

    ## 成本和边界

    每轮诊断与再次执行都会增加调用成本。如果诊断本身错误，系统还可能把错误解释写进
    新约束。因此，这套方法适合失败可以定位、规则可以局部修改的任务；对于反馈极其含糊
    的开放任务，收益和风险仍需单独验证。

    ## 应该怎样记住这篇论文

    最值得保留的认识不是“多运行几次会更好”，而是执行环境能够保存失败信息并改变下一轮
    行为。论文由此重新划定了模型策略与外部 harness 的边界，也给出了可以通过消融实验
    检查这一主张的方式。
    """
).strip()


DUPLICATED_AUTHOR_BODY = CLEAN_AUTHOR_BODY.replace(
    "## 实验为什么能支持这个解释",
    f"## 一段没有新增作用的重复解释\n\n{DUPLICATE_SPAN_B}\n\n## 实验为什么能支持这个解释",
)

assert len(CLEAN_AUTHOR_BODY) >= 600
assert len(DUPLICATED_AUTHOR_BODY) >= 600
assert DUPLICATED_AUTHOR_BODY.count(DUPLICATE_SPAN_A) == 1
assert DUPLICATED_AUTHOR_BODY.count(DUPLICATE_SPAN_B) == 1


_STAGE_MARKERS = {
    "brief": "# 任务：为论文精读建立叙事简报（Narrative Brief）",
    "author": "# 任务：论文精读 V9 · 单一作者生成连续初稿",
    "fact": "# 任务：论文精读 V9 · 事实与证据审查",
    "reader": "# 任务：论文精读 V9 · 真实读者审查",
    "revision": "# 任务：论文精读 V9 · 同一作者的有限编辑",
    "verify": "# 任务：论文精读 V9 · 改动点验证",
}


class FakeV9Client:
    """按 V9 prompt 标题分发响应，并保留完整调用轨迹。"""

    def __init__(self, responses: Mapping[str, Any]) -> None:
        self.responses = dict(responses)
        self.calls: list[dict[str, Any]] = []

    @staticmethod
    def _stage(prompt: str) -> str:
        for stage, marker in _STAGE_MARKERS.items():
            if marker in prompt:
                return stage
        raise AssertionError(f"fake client 收到未知 V9 prompt：{prompt[:160]!r}")

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        prompt = str(messages[-1]["content"])
        stage = self._stage(prompt)
        self.calls.append({"stage": stage, "prompt": prompt, "kwargs": dict(kwargs)})
        if stage not in self.responses:
            raise AssertionError(f"V9 不应调用 {stage!r} 阶段；实际调用轨迹：{self.stages}")
        response = self.responses[stage]
        if isinstance(response, BaseException):
            raise response
        if callable(response):
            response = response(prompt)
        if isinstance(response, str):
            return response
        return json.dumps(response, ensure_ascii=False)

    @property
    def stages(self) -> list[str]:
        return [str(item["stage"]) for item in self.calls]

    def prompt_for(self, stage: str) -> str:
        matches = [str(item["prompt"]) for item in self.calls if item["stage"] == stage]
        assert len(matches) == 1, f"期望 {stage!r} 恰好调用一次，实际 {len(matches)} 次"
        return matches[0]


def _budget_sized_author_part(prompt: str, *, first_body: str = "") -> str:
    """按生产 prompt 声明的本段下限生成不触发补写的确定性测试正文。"""
    match = re.search(r"本部分建议约 [\d,]+ 字符，硬下限 ([\d,]+) 字符", prompt)
    minimum = int(match.group(1).replace(",", "")) if match else 600
    part_match = re.search(r"第 (\d+)/(\d+) 个连续写作范围", prompt)
    index = int(part_match.group(1)) if part_match else 1
    prefix = first_body if index == 1 and first_body else f"## 第 {index} 部分的独特讲解\n\n"
    sentence = f"第 {index} 部分继续解释本组机制、证据与适用边界，并保持与其他部分不同。"
    repeats = max(1, (minimum + 800 - len(prefix)) // len(sentence) + 1)
    return prefix + sentence * repeats


def _brief() -> dict[str, Any]:
    beats = [
        (
            "beat-1", "失败从哪里来", "为什么普通重采样不够？", "先建立真实问题",
            "读者应先知道失败为何会累积，而不是只看到结果变差。", "short",
        ),
        (
            "beat-2", "关键转向", "为什么 harness 会改变策略？", "建立全文主张",
            "读者应能说出关键转向是把失败反馈纳入下一轮行为，而不只是增加采样。", "medium",
        ),
        (
            "beat-3", "核心循环", "失败信息怎样进入下一轮？", "用具体过程完成深讲",
            "读者应能复述执行、诊断、定点修改和再次执行如何首尾相接。", "deep",
        ),
        (
            "beat-4", "实验辨析", "实验如何区分反馈与重采样？", "把机制连接到证据",
            "读者应能分清实验支持了反馈机制的作用，也知道它没有证明什么。", "medium",
        ),
        (
            "beat-5", "边界与结论", "什么时候不该外推？", "收束适用范围",
            "读者应带走方法成立的范围和额外成本，而不是一个无条件的成功结论。", "short",
        ),
    ]
    return {
        "title_zh": "Harness 如何利用失败轨迹",
        "title_en": "A Small Harness Paper",
        "paper_type": "system",
        "reader_question": "失败轨迹怎样通过 harness 改变代理的下一轮行为？",
        "paper_turn": "把 harness 从被动执行外壳改看成策略的一部分。",
        "tldr": "系统把失败轨迹转成局部规则修改，再用消融实验检验这条机制。",
        "contributions": ["提出定点反馈循环", "用消融区分反馈与重采样"],
        "beats": [
            {
                "id": beat_id,
                "title": title,
                "question": question,
                "purpose": purpose,
                "reader_takeaway": takeaway,
                "source_refs": [],
                "figure_refs": [],
                "length": length,
            }
            for beat_id, title, question, purpose, takeaway, length in beats
        ],
        "deep_dive_ids": ["beat-3"],
        "concept_homes": [
            {"concept": "失败反馈循环", "beat_id": "beat-3", "full_explanation": True}
        ],
        "figure_homes": [],
        "exclusions": ["不逐项复述所有实验数字"],
        "target": {"h2_max": 7, "total_heading_max": 18, "target_chars": 6_000},
    }


def _fact_ok() -> dict[str, Any]:
    return {
        "status": "complete",
        "issues": [],
        "checked_claims": ["核心循环", "消融结论", "外推边界"],
        "reason": "",
    }


def _reader_ok() -> dict[str, Any]:
    return {"status": "complete", "issues": [], "reason": ""}


def _reader_duplication() -> dict[str, Any]:
    return {
        "status": "complete",
        "issues": [
            {
                "issue_id": "reader-1",
                "severity": "P1",
                "kind": "duplication",
                "span_a": DUPLICATE_SPAN_A,
                "span_b": DUPLICATE_SPAN_B,
                "keep": "a",
                "operation": "delete",
                "reader_harm": "后一段没有新证据或新用途，迫使读者重新读取同一四步流程。",
                "required_outcome": "保留第一次完整解释，删除后一次无新增作用的复述。",
                "max_scope": "paragraph",
            }
        ],
        "reason": "",
    }


def _delete_duplicate_script() -> dict[str, Any]:
    return {
        "edits": [
            {
                "edit_id": "edit-1",
                "operation": "delete",
                "old_text": DUPLICATE_SPAN_B,
                "new_text": "",
                "reason": "落实 reader-1：删除后一次无新增作用的完整复述。",
                "preserve": ["失败信息怎样真正进入下一次行为"],
            }
        ]
    }


def _verification_ok() -> dict[str, Any]:
    return {
        "status": "complete",
        "results": [
            {
                "edit_id": "edit-1",
                "target_fixed": True,
                "new_fact_error": False,
                "protected_content_lost": False,
                "broken_reference": False,
                "decision": "accept",
                "reason": "重复段已删除，首次机制解释和后续证据链仍完整。",
            }
        ],
        "reason": "",
    }


def _base_responses(author_body: str = CLEAN_AUTHOR_BODY) -> dict[str, Any]:
    return {
        "brief": _brief(),
        "author": author_body,
        "fact": _fact_ok(),
        "reader": _reader_ok(),
    }


def _expected_author_document(author_body: str) -> str:
    return (
        "# 《Harness 如何利用失败轨迹》精读笔记\n\n"
        "> **一句话理解**：系统把失败轨迹转成局部规则修改，再用消融实验检验这条机制。\n\n"
        f"{author_body}\n"
    )


def _all_mapping_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(_all_mapping_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys.update(_all_mapping_keys(item))
    return keys


def _v9_workspace(tmp_path: Path) -> ReadingWorkspace:
    return ReadingWorkspace(tmp_path, pipeline_version=V9_PIPELINE_VERSION)


def test_v9_two_audits_with_no_issues_publish_untouched_author_draft() -> None:
    client = FakeV9Client(_base_responses())

    result = V9ReadingEngine(client).run(PAPER_MD)

    assert client.stages == ["brief", "author", "fact", "reader"]
    assert result.body_md == _expected_author_document(CLEAN_AUTHOR_BODY)
    assert result.metadata["pipeline_version"] == "reading-v5"
    assert result.metadata["release_status"] == "verified_final"
    assert result.metadata["fact_audit_status"] == "complete"
    assert result.metadata["reader_audit_status"] == "complete"
    assert result.metadata["revision_status"] == "not_needed"
    assert result.metadata["accepted_edits"] == []
    assert "pass" not in _all_mapping_keys(result.metadata)
    assert "score" not in _all_mapping_keys(result.metadata)


def test_v9_duplication_becomes_exact_patch_and_is_adopted_after_verification() -> None:
    responses = _base_responses(DUPLICATED_AUTHOR_BODY)
    responses.update(
        {
            "reader": _reader_duplication(),
            "revision": _delete_duplicate_script(),
            "verify": _verification_ok(),
        }
    )
    client = FakeV9Client(responses)

    result = V9ReadingEngine(client).run(PAPER_MD)

    assert client.stages == ["brief", "author", "fact", "reader", "revision", "verify"]
    assert DUPLICATE_SPAN_A in result.body_md
    assert DUPLICATE_SPAN_B not in result.body_md
    assert result.metadata["release_status"] == "verified_final"
    assert result.metadata["revision_status"] == "verified"
    assert result.metadata["verification_status"] == "complete"
    assert [item["edit_id"] for item in result.metadata["accepted_edits"]] == ["edit-1"]
    assert result.metadata["accepted_edits"][0]["old_text"] == DUPLICATE_SPAN_B
    assert result.metadata["accepted_edits"][0]["new_text"] == ""

    reader_prompt = client.prompt_for("reader")
    assert "语义落点清单" in reader_prompt
    assert "读者应能复述执行、诊断、定点修改和再次执行如何首尾相接。" in reader_prompt
    assert "不要因为没有出现原句就报告" in reader_prompt
    revision_prompt = client.prompt_for("revision")
    assert DUPLICATE_SPAN_B in revision_prompt
    assert "reader-1" in revision_prompt
    verify_prompt = client.prompt_for("verify")
    assert DUPLICATE_SPAN_B in verify_prompt
    assert '"edit_id": "edit-1"' in verify_prompt


def test_v9_verification_unavailable_rolls_back_to_immutable_author_draft(
    tmp_path: Path,
) -> None:
    responses = _base_responses(DUPLICATED_AUTHOR_BODY)
    responses.update(
        {
            "reader": _reader_duplication(),
            "revision": _delete_duplicate_script(),
            "verify": KimiError("verification service unavailable"),
        }
    )
    client = FakeV9Client(responses)
    workspace = _v9_workspace(tmp_path)

    result = V9ReadingEngine(client).run(PAPER_MD, workspace=workspace)

    author_draft = workspace.get_text("author-draft")
    edited_candidate = workspace.get_text("edited-candidate")
    assert author_draft == _expected_author_document(DUPLICATED_AUTHOR_BODY)
    assert edited_candidate is not None and DUPLICATE_SPAN_B not in edited_candidate
    assert result.body_md.endswith(author_draft)
    assert DUPLICATE_SPAN_B in result.body_md
    assert result.metadata["release_status"] == "revision_rolled_back"
    assert result.metadata["revision_status"] == "verification_unavailable"
    assert result.metadata["verification_status"] == "unavailable"
    assert result.metadata["accepted_edits"] == []
    assert client.stages[-1] == "verify"


def test_v9_fact_audit_unavailable_only_blocks_fact_cards_without_fake_pass() -> None:
    responses = _base_responses(DUPLICATED_AUTHOR_BODY)
    responses["fact"] = KimiError("fact reviewer unavailable")
    responses["reader"] = _reader_duplication()
    responses["revision"] = _delete_duplicate_script()
    responses["verify"] = _verification_ok()
    client = FakeV9Client(responses)

    result = V9ReadingEngine(client).run(PAPER_MD)

    assert client.stages == ["brief", "author", "fact", "reader", "revision", "verify"]
    assert result.metadata["release_status"] == "fact_review_unavailable"
    assert result.metadata["fact_audit_status"] == "unavailable"
    assert result.metadata["revision_status"] == "verified"
    assert len(result.metadata["accepted_edits"]) == 1
    assert DUPLICATE_SPAN_A in result.body_md
    assert DUPLICATE_SPAN_B not in result.body_md
    assert "事实审查未完成" in result.body_md
    keys = _all_mapping_keys(result.metadata)
    assert "pass" not in keys
    assert "score" not in keys
    serialized = json.dumps(result.metadata, ensure_ascii=False)
    assert '"pass"' not in serialized
    assert '"score"' not in serialized


def test_v9_edit_script_over_twenty_percent_is_rejected_as_a_whole() -> None:
    large_target = "需要保留初稿基线的长段落。" * 24
    body = ("稳定前文。" * 28) + large_target + ("稳定后文。" * 28)
    payload = {
        "edits": [
            {
                "edit_id": "edit-large",
                "operation": "delete",
                "old_text": large_target,
                "new_text": "",
                "reason": "模拟错误的整段大范围删除。",
                "preserve": [],
            }
        ]
    }

    edits = parse_revision_edits(payload, body=body)
    applied = apply_edit_script(body, edits, max_change_ratio=0.20)

    assert applied.body == body
    assert applied.accepted == ()
    assert applied.change_ratio > 0.20
    assert [item.code for item in applied.rejected] == ["change_ratio_exceeded"]
    assert "整批回滚" in applied.rejected[0].reason


def test_v9_workspace_is_reading_v5_and_second_run_reuses_all_stage_caches(
    tmp_path: Path,
) -> None:
    workspace = _v9_workspace(tmp_path)
    first_client = FakeV9Client(_base_responses())
    first_engine = V9ReadingEngine(first_client)

    first = first_engine.run(PAPER_MD, workspace=workspace)

    assert first_engine.workspace_version == V9_PIPELINE_VERSION == "reading-v5"
    assert workspace.pipeline_version == "reading-v5"
    assert workspace.root == tmp_path / "reading-v5"
    assert not (tmp_path / "reading-v4").exists()
    assert first_client.stages == ["brief", "author", "fact", "reader"]
    assert workspace.get_json("narrative-brief") is not None
    assert workspace.get_text("author-part-01") == CLEAN_AUTHOR_BODY
    assert workspace.get_text("author-draft") == first.body_md
    assert workspace.get_json("fact-audit") is not None
    assert workspace.get_json("reader-audit") is not None
    assert workspace.get_text("final-body") == first.body_md

    # 第二次运行使用一个没有任何响应的客户端；只要发生模型调用，fake 就会立刻失败。
    second_client = FakeV9Client({})
    second = V9ReadingEngine(second_client).run(PAPER_MD, workspace=workspace)

    assert second_client.stages == []
    assert second.model_calls == 0
    assert second.body_md == first.body_md
    assert second.metadata["brief_status"] == "complete"


def test_v9_cached_fallback_brief_is_regenerated_instead_of_reused(
    tmp_path: Path,
) -> None:
    workspace = _v9_workspace(tmp_path)
    fallback = NarrativeBrief.fallback("缓存中的兜底简报").to_dict()
    workspace.put_json("narrative-brief", fallback)
    workspace.put_json("narrative-brief-status", {"status": "complete", "detail": "complete"})
    workspace.put_json("narrative-brief-manifest", {
        "brief_sha256": "old-digest",
        "status": "complete",
    })
    workspace.put_text("author-draft", "必须保留的已有作者初稿")
    bundle = EvidenceBundle(
        chunks=[SourceChunk("chunk-0001", "正文", PAPER_MD, 1)],
        cards=[],
        long_mode=False,
    )
    client = FakeV9Client({"brief": _brief()})
    engine = V9ReadingEngine(client)

    brief, calls, status = engine._brief(
        client, PAPER_MD, [], [], {}, bundle, workspace,
    )

    assert calls == 1
    assert client.stages == ["brief"]
    assert status == "complete"
    assert brief.tldr == _brief()["tldr"]
    assert brief.concept_homes
    assert brief.figure_homes == ()
    assert workspace.get_json("narrative-brief-status") == {
        "status": "complete",
        "detail": "complete",
    }
    manifest = workspace.get_json("narrative-brief-manifest")
    assert isinstance(manifest, dict)
    assert manifest["status"] == "complete"
    assert manifest["brief_sha256"] == _digest_payload(_brief())
    assert "quality_reason" not in manifest
    assert workspace.get_text("author-draft") is None


def test_v9_brief_digest_mismatch_invalidates_only_stale_downstream_checkpoint(
    tmp_path: Path,
) -> None:
    workspace = _v9_workspace(tmp_path)
    fallback = NarrativeBrief.fallback("需要校验摘要").to_dict()
    workspace.put_json("narrative-brief", fallback)
    workspace.put_json("narrative-brief-status", {"status": "degraded", "detail": "old"})
    workspace.put_json("narrative-brief-manifest", {
        "brief_sha256": "stale-brief-digest",
        "status": "degraded",
    })
    workspace.put_json("author-manifest", {
        "brief_sha256": "another-brief-digest",
        "source_sha256": "source",
    })
    workspace.put_text("author-draft", "绑定了另一版 Brief 的正文")
    bundle = EvidenceBundle(
        chunks=[SourceChunk("chunk-0001", "正文", PAPER_MD, 1)],
        cards=[],
        long_mode=False,
    )

    client = FakeV9Client({"brief": _brief()})
    V9ReadingEngine(client)._brief(
        client, PAPER_MD, [], [], {}, bundle, workspace,
    )

    manifest = workspace.get_json("narrative-brief-manifest")
    assert isinstance(manifest, dict)
    assert manifest["brief_sha256"] == _digest_payload(_brief())
    assert manifest["status"] == "complete"
    assert workspace.get_text("author-draft") is None


def test_v9_rejects_legacy_reading_v4_workspace_before_any_model_call(
    tmp_path: Path,
) -> None:
    client = FakeV9Client({})
    legacy_workspace = ReadingWorkspace(tmp_path)  # V7/V8 默认 reading-v4

    with pytest.raises(ReadingError, match="reading-v5"):
        V9ReadingEngine(client).run(PAPER_MD, workspace=legacy_workspace)

    assert client.stages == []


def test_v9_schemas_reject_legacy_pass_score_and_unknown_contract_fields() -> None:
    with pytest.raises(V9ValidationError, match="未知字段"):
        AuditReport.from_dict(
            {
                "status": "complete",
                "issues": [],
                "checked_claims": [],
                "reason": "",
                "pass": True,
                "score": 8,
            },
            kind="fact",
        )

    invalid_brief = _brief()
    invalid_brief["must_explain"] = ["禁止进入 V9 Brief 的旧合同字段"]
    with pytest.raises(V9ValidationError, match="未知字段"):
        NarrativeBrief.from_dict(invalid_brief)


def test_v9_duplicate_detector_only_returns_paired_candidates() -> None:
    body = (
        "## 第一次解释\n\n" + DUPLICATE_SPAN_A +
        "\n\n## 新证据\n\n这里加入一个真正不同的实验结论。" * 8 +
        "\n\n## 重复解释\n\n" + DUPLICATE_SPAN_A
    )

    candidates = duplicate_candidates(body, min_chars=60)

    assert candidates
    exact = candidates[0]
    assert exact.kind == "exact"
    assert exact.span_a == DUPLICATE_SPAN_A
    assert exact.span_b == DUPLICATE_SPAN_A
    assert exact.first_start < exact.second_start


def test_v9_mechanical_guard_reports_jargon_placeholder_and_figure_overuse() -> None:
    body = dedent(
        """
        # 标题

        ## 方法

        本章先偿还 must_explain，并给出有限判断。TODO：稍后补充。

        [[FIG:Figure 1]]

        ## 证据

        [[FIG:Figure 1]]

        ### 回指

        [[FIG:Figure 1]]
        """
    ).strip()

    report = mechanical_quality_report(body, figures=["Figure 1"])
    codes = [item.code for item in report.findings]

    assert report.planning_jargon_count >= 2
    assert report.placeholder_count == 1
    assert dict(report.figure_references)["Figure 1"] == 3
    assert "planning_jargon" in codes
    assert "placeholder" in codes
    assert "figure_reference_overuse" in codes


def test_v9_mechanical_guard_finds_english_but_exempts_code_and_names() -> None:
    body = dedent(
        """
        ## hopefully 只在标题里不检查

        这一句残留 hopefully，需要被机械检查定位。

        > 引用里的 untranslated 仍属于文章内容。

        调用 parseMove() 后读取 `rawEnglish`；AutoHarness 是白名单专有名词。

        ```python
        hopefully = parseMove(rawEnglish)
        ```
        """
    ).strip()

    report = mechanical_quality_report(body, english_allowlist={"AutoHarness"})
    findings = [item for item in report.findings if item.code == "untranslated_english"]

    assert report.untranslated_english_count == 2
    assert report.untranslated_english_words == ("hopefully", "untranslated")
    assert len(findings) == 2
    assert findings[0].line == 3


def test_v9_edit_card_selection_rejects_move_and_ambiguous_targets() -> None:
    repeated = "这句话在正文里出现两次。"
    body = f"## A\n\n{repeated}\n\n## B\n\n{repeated}"
    fact = AuditReport.from_dict(
        {"status": "complete", "issues": [], "checked_claims": [], "reason": ""},
        kind="fact",
    )
    reader = AuditReport.from_dict(
        {
            "status": "complete",
            "issues": [
                {
                    "issue_id": "reader-ambiguous",
                    "severity": "P1",
                    "kind": "mechanical_tone",
                    "span_a": repeated,
                    "span_b": "",
                    "keep": "none",
                    "operation": "shorten",
                    "reader_harm": "目标不唯一，不能安全自动修改。",
                    "required_outcome": "只保留自然表达。",
                    "max_scope": "sentence",
                },
                {
                    "issue_id": "reader-move",
                    "severity": "P1",
                    "kind": "missing_bridge",
                    "span_a": "## A",
                    "span_b": "",
                    "keep": "none",
                    "operation": "move",
                    "reader_harm": "顺序不自然。",
                    "required_outcome": "调整位置。",
                    "max_scope": "sentence",
                },
            ],
            "reason": "",
        },
        kind="reader",
    )

    selection = select_edit_cards(body, fact, reader)

    assert selection.cards == ()
    assert {item.code for item in selection.rejected} == {
        "target_ambiguous", "move_not_atomic",
    }


def test_v9_fact_quote_with_ellipsis_recovers_only_one_exact_paragraph() -> None:
    paragraph = (
        "高违规率游戏的提升最明显。Secretary 的合法率已经饱和，但奖励仍然上升。"
        "因此不能写成哪里没有违规、哪里就必然打平。"
    )
    body = f"## 证据\n\n{paragraph}\n\n## 边界\n\n其他结论。"
    issue = {
        "issue_id": "fact-ellipsis",
        "severity": "P1",
        "kind": "number",
        "draft_quote": (
            "Secretary 的合法率已经饱和，但奖励仍然上升。……"
            "因此不能写成哪里没有违规、哪里就必然打平。"
        ),
        "source_location": "Results",
        "source_evidence": "逐游戏数据给出反例。",
        "problem": "审核引文省略了同段中间文字。",
        "required_outcome": "保留反例并收窄全称判断。",
        "operation": "correct",
        "max_scope": "paragraph",
    }
    fact = AuditReport.from_dict(
        {"status": "complete", "issues": [issue], "checked_claims": [], "reason": ""},
        kind="fact",
    )
    reader = AuditReport.from_dict(_reader_ok(), kind="reader")

    selection = select_edit_cards(body, fact, reader)

    assert len(selection.cards) == 1
    assert selection.cards[0].target_quote == paragraph
    assert not any(item.code == "target_missing" for item in selection.rejected)

    ambiguous = select_edit_cards(f"{paragraph}\n\n{paragraph}", fact, reader)
    assert ambiguous.cards == ()
    assert [item.code for item in ambiguous.rejected] == ["target_missing"]

    missing_anchor_issue = dict(issue)
    missing_anchor_issue["issue_id"] = "fact-missing-anchor"
    missing_anchor_issue["draft_quote"] = (
        "Secretary 的合法率已经饱和，但奖励仍然上升。……"
        "正文中不存在的第二个精确句子需要同样被唯一定位。"
    )
    missing_fact = AuditReport.from_dict(
        {
            "status": "complete",
            "issues": [missing_anchor_issue],
            "checked_claims": [],
            "reason": "",
        },
        kind="fact",
    )
    missing = select_edit_cards(body, missing_fact, reader)
    assert missing.cards == ()
    assert [item.code for item in missing.rejected] == ["target_missing"]


def test_v9_verification_accept_requires_all_four_safety_conditions() -> None:
    with pytest.raises(V9ValidationError, match="全部安全"):
        VerificationResult.from_dict(
            {
                "edit_id": "edit-1",
                "target_fixed": True,
                "new_fact_error": True,
                "protected_content_lost": False,
                "broken_reference": False,
                "decision": "accept",
                "reason": "错误地试图放行。",
            }
        )


def test_v9_monotonic_gate_rejects_reviewer_jargon_growth() -> None:
    target = "这里先给出有限判断。"
    author_body = CLEAN_AUTHOR_BODY.replace(
        "论文的关键转向，是把 harness 看成策略行为的一部分。",
        target + "\n\n论文的关键转向，是把 harness 看成策略行为的一部分。",
    )
    reader = {
        "status": "complete",
        "issues": [
            {
                "issue_id": "reader-tone",
                "severity": "P1",
                "kind": "mechanical_tone",
                "span_a": target,
                "span_b": "",
                "keep": "none",
                "operation": "shorten",
                "reader_harm": "这句像审核报告。",
                "required_outcome": "改成自然的读者表达。",
                "max_scope": "sentence",
            }
        ],
        "reason": "",
    }
    revision = {
        "edits": [
            {
                "edit_id": "edit-jargon",
                "operation": "replace",
                "old_text": target,
                "new_text": "有限判断有限判断",
                "reason": "reader-tone",
                "preserve": [],
            }
        ]
    }
    responses = _base_responses(author_body)
    responses.update({"reader": reader, "revision": revision})
    client = FakeV9Client(responses)

    result = V9ReadingEngine(client).run(PAPER_MD)

    assert client.stages == ["brief", "author", "fact", "reader", "revision"]
    assert "verify" not in client.stages
    assert result.metadata["release_status"] == "revision_rolled_back"
    assert result.metadata["revision_status"].startswith("rejected:planning_jargon_growth")
    assert result.body_md.endswith(_expected_author_document(author_body))
    assert result.body_md.count("有限判断") == 1


def test_v9_multi_part_authoring_passes_real_prior_prose_to_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        V9ReadingEngine,
        "_part_count",
        staticmethod(lambda brief, bundle, paper_len: 2),
    )
    part_one = dedent(
        """
        ## 失败从哪里来

        长程任务把早期局部错误带到后续步骤。普通重采样没有保存失败位置，也没有解释
        工具为何拒绝输入，因此下一轮很容易重新走回同一条错误路径。这一现象说明，问题
        不只是模型能否生成答案，还包括执行环境能否保留可用于下一次行为的信息。

        ## 关键转向

        论文把 harness 看成策略的一部分。它记录环境拒绝了什么、哪些步骤已经正确，并
        决定下一轮只修改哪里。这个视角把被动外壳变成了会改变搜索方向的机制。

        ## 核心循环

        系统先运行候选策略，再记录失败轨迹，把轨迹压缩成局部反馈，最后只修改相关规则
        并重新执行。具体而言，正确步骤被保留，错误参数的约束被收紧，下一轮从失败附近
        继续，而不是整条轨迹从头生成。这段过程给出了后文分析实验时唯一需要的机制基础。
        """
    ).strip() + ("\n\n机制补充说明。" * 35)
    part_two = dedent(
        """
        ## 实验为什么能支持这个解释

        基于前面的循环，实验只需要比较定点反馈与普通重采样，而不必重新定义 harness。
        消融结果显示移除结构化反馈后改善缩小，因此证据对应的是失败信息进入下一轮规则。

        ## 成本与边界

        额外诊断和执行会增加成本，错误诊断也可能放大偏差。结论应限制在失败可以定位、
        规则能够局部修改的任务中，不能直接外推到所有开放式代理场景。
        """
    ).strip() + ("\n\n边界补充说明。" * 55)
    author_calls: list[str] = []

    def author_response(prompt: str) -> str:
        author_calls.append(prompt)
        return part_one if len(author_calls) == 1 else part_two

    responses = _base_responses()
    responses["author"] = author_response
    client = FakeV9Client(responses)

    result = V9ReadingEngine(client).run(PAPER_MD)

    assert client.stages == ["brief", "author", "author", "fact", "reader"]
    assert len(author_calls) == 2
    assert part_one in author_calls[1]
    assert "已有正文是正式前文，直接承接，不重新开场" in author_calls[1]
    assert part_one in result.body_md and part_two in result.body_md
    assert result.metadata["release_status"] == "verified_final"


def test_v9_unavailable_audit_is_not_cached_as_a_permanent_terminal_state(
    tmp_path: Path,
) -> None:
    workspace = _v9_workspace(tmp_path)
    first = FakeV9Client(_base_responses())
    first.responses["fact"] = KimiError("temporary fact outage")
    first_result = V9ReadingEngine(first).run(PAPER_MD, workspace=workspace)
    assert first_result.metadata["release_status"] == "fact_review_unavailable"
    assert first.stages == ["brief", "author", "fact", "reader"]

    second = FakeV9Client({"fact": _fact_ok()})
    second_result = V9ReadingEngine(second).run(PAPER_MD, workspace=workspace)

    assert second.stages == ["fact"]
    assert second_result.metadata["fact_audit_status"] == "complete"
    assert second_result.metadata["release_status"] == "verified_final"


def test_v9_revision_rejects_a_patch_not_bound_to_any_edit_card() -> None:
    unrelated = "一个具体例子是：代理第一次提交了格式错误的工具参数。"
    responses = _base_responses(DUPLICATED_AUTHOR_BODY)
    responses.update(
        {
            "reader": _reader_duplication(),
            "revision": {
                "edits": [
                    {
                        "edit_id": "edit-unbound",
                        "operation": "delete",
                        "old_text": unrelated,
                        "new_text": "",
                        "reason": "reader-1",
                        "preserve": [],
                    }
                ]
            },
        }
    )
    client = FakeV9Client(responses)

    result = V9ReadingEngine(client).run(PAPER_MD)

    assert client.stages == ["brief", "author", "fact", "reader", "revision"]
    assert "verify" not in client.stages
    assert result.metadata["release_status"] == "revision_rolled_back"
    assert result.metadata["revision_status"].startswith("rejected:")
    assert unrelated in result.body_md
    assert DUPLICATE_SPAN_B in result.body_md


def test_v9_opposite_duplicate_cards_cannot_both_be_selected() -> None:
    fact = AuditReport.from_dict(
        {"status": "complete", "issues": [], "checked_claims": [], "reason": ""},
        kind="fact",
    )
    reader = AuditReport.from_dict(
        {
            "status": "complete",
            "issues": [
                {
                    "issue_id": "dup-keep-a",
                    "severity": "P1",
                    "kind": "duplication",
                    "span_a": DUPLICATE_SPAN_A,
                    "span_b": DUPLICATE_SPAN_B,
                    "keep": "a",
                    "operation": "delete",
                    "reader_harm": "保留 A，删除 B。",
                    "required_outcome": "保留 A 的机制解释。",
                    "max_scope": "paragraph",
                },
                {
                    "issue_id": "dup-keep-b",
                    "severity": "P1",
                    "kind": "duplication",
                    "span_a": DUPLICATE_SPAN_A,
                    "span_b": DUPLICATE_SPAN_B,
                    "keep": "b",
                    "operation": "delete",
                    "reader_harm": "保留 B，删除 A。",
                    "required_outcome": "保留 B 的机制解释。",
                    "max_scope": "paragraph",
                },
            ],
            "reason": "",
        },
        kind="reader",
    )

    selection = select_edit_cards(DUPLICATED_AUTHOR_BODY, fact, reader)

    assert len(selection.cards) == 1
    assert any(item.code == "overlap" for item in selection.rejected)


def test_v9_workspace_signature_changes_when_input_or_route_changes(tmp_path: Path) -> None:
    workspace = _v9_workspace(tmp_path)
    first = FakeV9Client(_base_responses())
    V9ReadingEngine(first).run(PAPER_MD, workspace=workspace)

    class AlternateRouteFake(FakeV9Client):
        pass

    second = AlternateRouteFake(_base_responses())
    changed_paper = PAPER_MD + "\n\n新增输入版本。"
    V9ReadingEngine(second).run(changed_paper, workspace=workspace)

    assert second.stages == ["brief", "author", "fact", "reader"]
    manifest = workspace.get_json("input-manifest")
    assert isinstance(manifest, dict)
    assert manifest["paper_sha256"]


def test_v9_reader_review_unavailable_still_runs_fact_only_revision() -> None:
    old_text = "论文的关键转向，是把 harness 看成策略行为的一部分。"
    new_text = "论文的关键转向，是把 harness 视为会影响后续搜索的策略组成部分。"
    fact_issue = {
        "issue_id": "same-id",
        "severity": "P1",
        "kind": "fact",
        "draft_quote": old_text,
        "source_location": "Method",
        "source_evidence": "原文明确把 harness 视为策略行为的一部分。",
        "problem": "需要确认关键转向的表述边界。",
        "required_outcome": "保留准确的关键转向，不扩大因果结论。",
        "operation": "correct",
        "max_scope": "sentence",
    }
    client = FakeV9Client({
        **_base_responses(),
        "fact": {"status": "complete", "issues": [fact_issue], "checked_claims": ["turn"], "reason": ""},
        "reader": KimiError("reader service unavailable"),
        "revision": {
            "edits": [
                {
                    "edit_id": "edit-fact",
                    "operation": "replace",
                    "old_text": old_text,
                    "new_text": new_text,
                    "reason": "落实 same-id",
                    "preserve": [],
                }
            ]
        },
        "verify": {
            "status": "complete",
            "results": [
                {
                    "edit_id": "edit-fact",
                    "target_fixed": True,
                    "new_fact_error": False,
                    "protected_content_lost": False,
                    "broken_reference": False,
                    "decision": "accept",
                    "reason": "事实表述已按来源收窄，局部上下文完整。",
                }
            ],
            "reason": "",
        },
    })

    result = V9ReadingEngine(client).run(PAPER_MD)

    assert client.stages == ["brief", "author", "fact", "reader", "revision", "verify"]
    expected_body = CLEAN_AUTHOR_BODY.replace(old_text, new_text)
    assert result.body_md.endswith(_expected_author_document(expected_body))
    assert old_text not in result.body_md
    assert new_text in result.body_md
    assert "真实读者审查未完成" in result.body_md
    assert result.metadata["reader_audit_status"] == "unavailable"
    assert result.metadata["release_status"] == "reader_review_unavailable"
    assert result.metadata["revision_status"] == "verified"
    assert result.metadata["open_fact_issues"] == []
    assert len(result.metadata["accepted_edits"]) == 1


def test_v9_resolved_issue_keys_keep_fact_and_reader_namespaces_separate() -> None:
    fact_card = EditCard(
        card_id="card-fact",
        issue_ids=("same-id",),
        audit_kind="fact",
        severity="P1",
        operation="correct",
        target_quote="事实句子",
        required_result="修正事实句子",
        max_scope="sentence",
        evidence="原文证据",
    )
    accepted = RevisionEdit(
        edit_id="edit-fact",
        operation="replace",
        old_text="事实句子",
        new_text="修正后的事实句子",
        reason="落实 card-fact",
        preserve=(),
    )

    keys = V9ReadingEngine._resolved_issue_ids([fact_card], [accepted])

    assert ("fact", "same-id") in keys
    assert ("reader", "same-id") not in keys


def test_v9_edit_card_budget_and_unpaired_merge_are_fail_closed() -> None:
    def fact_issue(index: int) -> dict[str, Any]:
        quote = f"事实句子 {index}"
        return {
            "issue_id": f"fact-{index}",
            "severity": "P1",
            "kind": "fact",
            "draft_quote": quote,
            "source_location": "Results",
            "source_evidence": "原文证据",
            "problem": "事实需要修正。",
            "required_outcome": "修正该事实。",
            "operation": "correct",
            "max_scope": "sentence",
        }

    body = "\n\n".join(f"事实句子 {i}" for i in range(1, 5)) + "\n\n桥接句"
    fact = AuditReport.from_dict(
        {
            "status": "complete",
            "issues": [fact_issue(i) for i in range(1, 5)],
            "checked_claims": [],
            "reason": "",
        },
        kind="fact",
    )
    reader = AuditReport.from_dict(
        {
            "status": "complete",
            "issues": [
                {
                    "issue_id": "reader-merge",
                    "severity": "P1",
                    "kind": "missing_bridge",
                    "span_a": "桥接句",
                    "span_b": "",
                    "keep": "none",
                    "operation": "merge",
                    "reader_harm": "缺少承接。",
                    "required_outcome": "补上自然承接。",
                    "max_scope": "sentence",
                }
            ],
            "reason": "",
        },
        kind="reader",
    )

    selection = select_edit_cards(body, fact, reader)

    assert len(selection.cards) == 3
    assert {item.code for item in selection.rejected} >= {"fact_budget", "paired_target_missing"}


def test_v9_revision_binding_rejects_heading_in_sentence_patch() -> None:
    card = EditCard(
        card_id="card-heading",
        issue_ids=("reader-1",),
        audit_kind="reader",
        severity="P1",
        operation="shorten",
        target_quote="原句",
        required_result="保持一句话表达",
        max_scope="sentence",
    )
    edit = RevisionEdit(
        edit_id="edit-heading",
        operation="replace",
        old_text="原句",
        new_text="改后的句子\n\n## 偷加的小节\n\n新的解释",
        reason="落实 card-heading",
        preserve=(),
    )

    bound, errors = V9ReadingEngine._bind_revision_edits([edit], [card], "## 原文\n\n原句")

    assert bound == []
    assert any("new_text 超出编辑卡范围" in error for error in errors)


def test_v9_long_source_fallback_cannot_become_verified_and_survives_cache(
    tmp_path: Path,
) -> None:
    chunks = [
        SourceChunk("chunk-0001", "问题", "问题来源。" * 12_000, 1),
        SourceChunk("chunk-0002", "方法", "方法证据。" * 12_000, 2),
    ]
    bundle = EvidenceBundle(chunks=chunks, cards=[], long_mode=True)

    class LongBundleEngine(V9ReadingEngine):
        def _evidence(self, client, paper_md, workspace, progress):
            return bundle, "确定性来源索引", 0, "complete"

    def author_part(prompt: str) -> str:
        return _budget_sized_author_part(prompt, first_body=CLEAN_AUTHOR_BODY)

    brief = _brief()
    fact_issue = {
        "issue_id": "fact-long-1",
        "severity": "P1",
        "kind": "fact",
        "draft_quote": "论文的关键转向，是把 harness 看成策略行为的一部分。",
        "source_location": "Method",
        "source_evidence": "原文支持该表述。",
        "problem": "关键转向表述需要核对。",
        "required_outcome": "只保留来源支持的关键转向。",
        "operation": "correct",
        "max_scope": "sentence",
    }
    first_client = FakeV9Client({
        "brief": brief,
        "author": author_part,
        "fact": {"status": "complete", "issues": [fact_issue], "checked_claims": [], "reason": ""},
        "reader": _reader_ok(),
    })
    workspace = _v9_workspace(tmp_path)
    paper = PAPER_MD + "\n\n" + "补充原文。" * 30_000

    first = LongBundleEngine(first_client).run(paper, workspace=workspace)

    assert first.metadata["release_status"] == "author_draft"
    assert first.metadata["source_coverage_complete"] is False
    assert any("关键词召回" in item for item in first.metadata["source_coverage_issues"])
    assert "revision" not in first_client.stages
    assert "verify" not in first_client.stages

    second_client = FakeV9Client({})
    second = LongBundleEngine(second_client).run(paper, workspace=workspace)

    assert second_client.stages == []
    assert second.metadata["release_status"] == "author_draft"
    assert second.metadata["source_coverage_complete"] is False


def test_v9_degraded_rerun_removes_stale_verified_final(tmp_path: Path) -> None:
    workspace = _v9_workspace(tmp_path)
    first = V9ReadingEngine(FakeV9Client(_base_responses())).run(
        PAPER_MD, workspace=workspace,
    )
    assert first.metadata["release_status"] == "verified_final"
    assert workspace.get_text("verified-final") == first.body_md

    workspace.put_json("reader-audit", {
        "status": "unavailable",
        "issues": [],
        "reason": "force retry",
    })
    second_client = FakeV9Client({"reader": KimiError("reader retry unavailable")})
    second = V9ReadingEngine(second_client).run(PAPER_MD, workspace=workspace)

    assert second_client.stages == ["reader"]
    assert second.metadata["release_status"] == "reader_review_unavailable"
    assert workspace.get_text("verified-final") is None
    assert workspace.get_text("final-body") == second.body_md


def test_v9_paragraph_patch_cannot_expand_into_a_mini_explanation() -> None:
    card = EditCard(
        card_id="card-paragraph",
        issue_ids=("reader-para",),
        audit_kind="reader",
        severity="P1",
        operation="shorten",
        target_quote="短段",
        required_result="保持局部修正",
        max_scope="paragraph",
    )
    edit = RevisionEdit(
        edit_id="edit-paragraph",
        operation="replace",
        old_text="短段",
        new_text="扩写" * 1_500,
        reason="落实 card-paragraph",
        preserve=(),
    )

    bound, errors = V9ReadingEngine._bind_revision_edits([edit], [card], "## 原文\n\n短段")

    assert bound == []
    assert any("new_text 超出编辑卡范围" in error for error in errors)


def test_v9_reader_audit_normalizes_keep_and_span_b_for_non_duplication() -> None:
    payload = {
        "status": "complete",
        "issues": [
            {
                "issue_id": "reader-tone",
                "severity": "P1",
                "kind": "mechanical_tone",
                "span_a": "一段机械表达。",
                "span_b": "模型误填的第二片段",
                "keep": "a",
                "operation": "shorten",
                "reader_harm": "语气像审查报告。",
                "required_outcome": "改成自然叙述。",
                "max_scope": "sentence",
            }
        ],
        "reason": "",
    }

    report = _parse_audit(payload, "reader", "一段机械表达。")

    assert report.status == "complete"
    assert report.issues[0].keep == "none"
    assert report.issues[0].span_b == ""


def test_v9_audit_parse_failure_gets_one_correction_retry(tmp_path: Path) -> None:
    attempts = 0

    def reader_response(prompt: str) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return {
                "status": "complete",
                "issues": [
                    {
                        "issue_id": "reader-dup",
                        "severity": "P1",
                        "kind": "duplication",
                        "span_a": DUPLICATE_SPAN_A,
                        "span_b": DUPLICATE_SPAN_B,
                        "keep": "none",
                        "operation": "delete",
                        "reader_harm": "重复。",
                        "required_outcome": "保留一处。",
                        "max_scope": "paragraph",
                    }
                ],
                "reason": "",
            }
        return _reader_ok()

    client = FakeV9Client({**_base_responses(), "reader": reader_response})
    workspace = _v9_workspace(tmp_path)
    result = V9ReadingEngine(client).run(PAPER_MD, workspace=workspace)

    assert attempts == 2
    assert client.stages == ["brief", "author", "fact", "reader", "reader"]
    assert result.metadata["reader_audit_status"] == "complete"
    assert "程序校验原因" in client.calls[-1]["prompt"]
    audit_attempts = workspace.get_json("reader-audit-attempts")
    assert isinstance(audit_attempts, list) and len(audit_attempts) == 2
    assert audit_attempts[0]["error"]
    assert workspace.get_text("reader-audit-raw-01")
    assert workspace.get_json("reader-audit-decision")["status"] == "complete"


def test_v9_p2_fact_enters_limited_revision_cards_and_is_transparent_if_unresolved() -> None:
    issue = {
        "issue_id": "fact-p2",
        "severity": "P2",
        "kind": "number",
        "draft_quote": "合计成本约 640 美元。",
        "source_location": "Results",
        "source_evidence": "原文明确为合计约 640 美元。",
        "problem": "成稿把合计成本写成每组各 640 美元。",
        "required_outcome": "保留合计约 640 美元，不得写成每组各 640 美元。",
        "operation": "correct",
        "max_scope": "sentence",
    }
    body = CLEAN_AUTHOR_BODY + "\n\n" + issue["draft_quote"]
    fact = AuditReport.from_dict(
        {"status": "complete", "issues": [issue], "checked_claims": [], "reason": ""},
        kind="fact",
    )
    reader = AuditReport.from_dict(_reader_ok(), kind="reader")

    selection = select_edit_cards(body, fact, reader)

    assert len(selection.cards) == 1
    assert selection.cards[0].severity == "P2"
    assert not any(item.code == "deferred_p2" for item in selection.rejected)

    class UnresolvedP2Engine(V9ReadingEngine):
        def _audit(self, reviewer, kind, body, source, figures, formulas, candidates, workspace):
            return (fact if kind == "fact" else reader), 0

    result = UnresolvedP2Engine(FakeV9Client(_base_responses())).run(PAPER_MD)
    assert result.metadata["release_status"] == "verified_final"
    assert [item["issue_id"] for item in result.metadata["unresolved_p2_fact_issues"]] == ["fact-p2"]


def test_v9_brief_takeaway_is_general_semantic_target_and_serialized() -> None:
    brief = NarrativeBrief.from_dict(_brief())

    checklist = _brief_takeaway_checklist(brief)
    payload = brief.to_dict()

    assert len(checklist) == 5
    assert checklist[2] == {
        "beat_id": "beat-3",
        "title": "核心循环",
        "length": "deep",
        "reader_takeaway": "读者应能复述执行、诊断、定点修改和再次执行如何首尾相接。",
    }
    assert payload["beats"][0]["reader_takeaway"]
    assert brief.schema_version == "reading-v9-brief-v2"


def test_v9_missing_takeaway_reader_issue_becomes_bridge_card() -> None:
    body = "## 核心机制\n\n这段只罗列了步骤，没有说明它们为什么首尾相接。"
    fact = AuditReport.from_dict(
        {"status": "complete", "issues": [], "checked_claims": [], "reason": ""},
        kind="fact",
    )
    reader = AuditReport.from_dict(
        {
            "status": "complete",
            "issues": [
                {
                    "issue_id": "reader-takeaway",
                    "severity": "P1",
                    "kind": "missing_takeaway",
                    "span_a": "这段只罗列了步骤，没有说明它们为什么首尾相接。",
                    "span_b": "",
                    "keep": "none",
                    "operation": "bridge",
                    "reader_harm": "读者看到了流程，但没有得到机制判断。",
                    "required_outcome": "让读者理解每一步如何共同形成机制。",
                    "max_scope": "paragraph",
                }
            ],
            "reason": "",
        },
        kind="reader",
    )

    selection = select_edit_cards(body, fact, reader)

    assert len(selection.cards) == 1
    assert selection.cards[0].operation == "bridge"
    assert selection.cards[0].target_quote == "这段只罗列了步骤，没有说明它们为什么首尾相接。"


def test_v9_tldr_over_limit_is_truncated_without_rejecting_brief() -> None:
    payload = _brief()
    payload["tldr"] = "这是一条有真实论文内容支撑的长摘要。" * 40

    brief = NarrativeBrief.from_dict(payload)

    assert len(brief.tldr) == 200
    assert brief.tldr.endswith("…")
    assert brief.concept_homes
    assert len(brief.beats) == 5


def test_v9_brief_parse_failure_gets_shared_correction_retry() -> None:
    attempts = 0

    def brief_response(prompt: str) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            invalid = _brief()
            invalid.pop("concept_homes")
            return invalid
        return _brief()

    client = FakeV9Client({**_base_responses(), "brief": brief_response})
    result = V9ReadingEngine(client).run(PAPER_MD)

    assert attempts == 2
    assert client.stages[:2] == ["brief", "brief"]
    assert result.metadata["brief_status"] == "complete"
    assert "程序校验原因" in client.calls[1]["prompt"]
    assert "NarrativeBrief 缺少字段" in client.calls[1]["prompt"]


def test_v9_fact_audit_parse_failure_uses_same_correction_retry() -> None:
    attempts = 0

    def fact_response(prompt: str) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return {"status": "complete", "issues": [], "reason": ""}
        return _fact_ok()

    client = FakeV9Client({**_base_responses(), "fact": fact_response})
    result = V9ReadingEngine(client).run(PAPER_MD)

    assert attempts == 2
    assert client.stages == ["brief", "author", "fact", "fact", "reader"]
    assert result.metadata["fact_audit_status"] == "complete"
    assert "程序校验原因" in client.calls[3]["prompt"]


def test_v9_fallback_engineering_tldr_never_leaks_to_body_or_metadata() -> None:
    attempts = 0

    def invalid_brief(_: str) -> str:
        nonlocal attempts
        attempts += 1
        return "这不是 JSON"

    client = FakeV9Client({**_base_responses(), "brief": invalid_brief})
    result = V9ReadingEngine(client).run(PAPER_MD)

    assert attempts == 2
    assert client.stages[:2] == ["brief", "brief"]
    assert result.metadata["brief_status"] == "degraded"
    assert result.metadata["tldr"] == ""
    assert "叙事简报不可用" not in result.body_md
    assert "一句话理解" not in result.body_md


def test_v9_reader_fuzzy_anchor_is_normalized_to_unique_body_quote() -> None:
    actual = (
        "系统先执行候选策略，记录失败轨迹，再把轨迹压缩成针对工具约束的反馈，"
        "最后只修改相关规则并重新执行。"
    )
    approximate = actual.replace("只修改相关规则", "仅修改相关规则")
    body = f"## 核心机制\n\n{actual}\n"
    fact = AuditReport.from_dict(_fact_ok(), kind="fact")
    reader = AuditReport.from_dict({
        "status": "complete",
        "issues": [{
            "issue_id": "reader-fuzzy",
            "severity": "P1",
            "kind": "mechanical_tone",
            "span_a": approximate,
            "span_b": "",
            "keep": "none",
            "operation": "shorten",
            "reader_harm": "表达过于机械。",
            "required_outcome": "压缩为自然表达。",
            "max_scope": "paragraph",
        }],
        "reason": "",
    }, kind="reader")

    selection = select_edit_cards(body, fact, reader)

    assert not selection.rejected
    assert len(selection.cards) == 1
    assert selection.cards[0].target_quote == actual


def test_v9_empty_revision_retries_once_and_can_reach_verified_final(
    tmp_path: Path,
) -> None:
    attempts = 0

    def revision_response(prompt: str) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return {"edits": []}
        return _delete_duplicate_script()

    responses = _base_responses(DUPLICATED_AUTHOR_BODY)
    responses.update({
        "reader": _reader_duplication(),
        "revision": revision_response,
        "verify": _verification_ok(),
    })
    client = FakeV9Client(responses)
    workspace = _v9_workspace(tmp_path)

    result = V9ReadingEngine(client).run(PAPER_MD, workspace=workspace)

    assert attempts == 2
    assert client.stages.count("revision") == 2
    assert result.metadata["release_status"] == "verified_final"
    assert result.metadata["revision_status"] == "verified"
    attempts_payload = workspace.get_json("revision-attempts")
    assert isinstance(attempts_payload, list) and len(attempts_payload) == 2
    assert attempts_payload[0]["payload"] == {"edits": []}
    assert workspace.get_json("revision-script") == _delete_duplicate_script()
    assert workspace.get_json("revision-decision")["status"] == "candidate"
    assert workspace.get_json("verification-decision")["status"] == "complete"
    assert workspace.get_text("verification-raw")


def test_v9_empty_revision_after_retry_preserves_script_and_decision(
    tmp_path: Path,
) -> None:
    responses = _base_responses(DUPLICATED_AUTHOR_BODY)
    responses.update({
        "reader": _reader_duplication(),
        "revision": {"edits": []},
    })
    client = FakeV9Client(responses)
    workspace = _v9_workspace(tmp_path)

    result = V9ReadingEngine(client).run(PAPER_MD, workspace=workspace)

    assert client.stages.count("revision") == 2
    assert result.metadata["release_status"] == "revision_rolled_back"
    assert result.metadata["revision_status"] == "no_safe_edits:empty_edits"
    assert workspace.get_json("revision-script") == {"edits": []}
    decision = workspace.get_json("revision-decision")
    assert decision["status"] == "no_safe_edits"
    assert decision["reason"] == "empty_edits"
    assert decision["raw_attempt_count"] == 2
    assert workspace.get_json("revision-attempts")


def test_v9_source_degradation_still_allows_reader_only_revision(
    tmp_path: Path,
) -> None:
    class DegradedEvidenceEngine(V9ReadingEngine):
        def _evidence(self, client, paper_md, workspace, progress):
            bundle = EvidenceBundle(
                chunks=[SourceChunk("chunk-0001", "正文", paper_md, 1)],
                cards=[],
                long_mode=False,
                degraded_chunks={"chunk-0001": "reasoning-exhaustion"},
            )
            return bundle, "导航材料", 0, "degraded:1 个证据块降级"

    responses = _base_responses(DUPLICATED_AUTHOR_BODY)
    responses.update({
        "reader": _reader_duplication(),
        "revision": _delete_duplicate_script(),
        "verify": _verification_ok(),
    })
    client = FakeV9Client(responses)
    workspace = _v9_workspace(tmp_path)

    result = DegradedEvidenceEngine(client).run(PAPER_MD, workspace=workspace)

    assert "revision" in client.stages and "verify" in client.stages
    assert DUPLICATE_SPAN_B not in result.body_md
    assert result.metadata["revision_status"] == "verified"
    assert result.metadata["release_status"] == "author_draft"
    assert result.metadata["source_coverage_complete"] is False
    assert result.metadata["degraded_evidence_chunks"] == {
        "chunk-0001": "reasoning-exhaustion"
    }


# ---- 预算按论文尺寸分路：长文分层来源包 ----

_SURVEY_WORDS = [
    "归纳偏置", "奖励整形", "课程学习", "策略蒸馏", "轨迹压缩", "约束传播",
    "反向信用", "探索噪声", "价值分解", "离线微调", "对抗鲁棒", "符号接地",
    "工具调用", "记忆回流", "任务分解", "自我批判", "多轮规划", "安全护栏",
]


def _survey_paper() -> str:
    """2509 规模假综述：18 章 × ~18k ≈ 325k 字符（> SHORT_PAPER_CHARS，进 long 档）。"""
    return "\n\n".join(
        f"## 第 {i} 章\n" + (f"本章讨论{_SURVEY_WORDS[i - 1]}。" * 2_000)
        for i in range(1, 19)
    )


def _survey_bundle(paper: str) -> EvidenceBundle:
    chunks = split_source(paper)
    cards = [
        EvidenceCard(
            chunk_id=chunk.chunk_id,
            section=chunk.title,
            orientation=f"{chunk.title} 概览",
            claims=[f"{chunk.title} 的核心主张"],
        )
        for chunk in chunks
    ]
    return EvidenceBundle(chunks=chunks, cards=cards, long_mode=True)


def test_v9_long_paper_packets_layer_fulltext_and_summary_within_budget() -> None:
    """长文装包：命中块全文 + 未命中块摘要，不超 25 万预算，不再产生覆盖降级。"""
    paper = _survey_paper()
    bundle = _survey_bundle(paper)
    chunks = bundle.chunks
    engine = V9ReadingEngine(FakeV9Client({}))
    group = [{
        "id": "beat-1",
        "title": "奖励整形怎么设计",
        "question": "奖励整形怎么设计？",
        "purpose": "讲清设计",
        "source_refs": [chunks[1].chunk_id, "不存在的章节标题"],  # 未解析引用被摘要层吸收
    }]

    author_packet = engine._source_for_group(paper, bundle, group)
    assert len(author_packet) <= AUTHOR_PACKET_CHARS_LONG
    assert f"SOURCE {chunks[1].chunk_id}" in author_packet   # 命中块全文
    assert f"SUMMARY {chunks[0].chunk_id}" in author_packet  # 未命中块证据卡摘要
    assert chunks[-1].chunk_id in author_packet              # 绝不截尾丢弃
    assert "的核心主张" in author_packet
    assert engine._source_coverage_issues == []              # 不再报「退回关键词召回」

    brief = NarrativeBrief.from_dict(_brief())
    fact_packet = engine._fact_source(paper, bundle, brief)
    assert len(fact_packet) <= FACT_PACKET_CHARS_LONG
    assert "chunk-" in fact_packet
    assert engine._source_coverage_issues == []


def test_v9_long_paper_full_run_no_longer_sticks_at_author_draft(tmp_path: Path) -> None:
    """2509 实证场景端到端：证据卡齐全时，417k 级长文来源覆盖完整，可发布终稿。"""
    paper = _survey_paper()
    bundle = _survey_bundle(paper)
    chunks = bundle.chunks

    class LongBundleEngine(V9ReadingEngine):
        def _evidence(self, client, paper_md, workspace, progress):
            return bundle, "确定性来源索引", 0, "complete"

    brief_payload = _brief()
    brief_payload["beats"][0]["source_refs"] = [chunks[2].chunk_id]

    def author_part(prompt: str) -> str:
        # 各部分按尺寸预算下限生成且互不相同，避免测试桩本身触发深度护栏。
        return _budget_sized_author_part(prompt, first_body=CLEAN_AUTHOR_BODY)

    client = FakeV9Client({
        "brief": brief_payload,
        "author": author_part,
        "fact": _fact_ok(),
        "reader": _reader_ok(),
    })

    result = LongBundleEngine(client).run(paper, workspace=_v9_workspace(tmp_path))

    assert result.metadata["release_status"] == "verified_final"
    assert result.metadata["source_coverage_complete"] is True
    assert result.metadata["budget_profile"]["tier"] == "long"
    assert result.metadata["author_draft_chars"] >= result.metadata["budget_profile"]["minimum_draft_chars"]
    assert not any(
        "关键词召回" in item for item in result.metadata["source_coverage_issues"]
    )
    author_prompts = [
        str(call["prompt"]) for call in client.calls if call["stage"] == "author"
    ]
    assert author_prompts  # 长文分多段连续写
    assert any(f"SOURCE {chunks[2].chunk_id}" in prompt for prompt in author_prompts)
    fact_prompt = client.prompt_for("fact")
    assert f"SOURCE {chunks[2].chunk_id}" in fact_prompt
    assert "SUMMARY chunk-" in fact_prompt  # 未命中块以证据卡摘要进入事实审查


def test_v9_long_paper_under_depth_gets_one_continuation_then_stops(tmp_path: Path) -> None:
    """max_tokens 只是上限；模型连续短答时不能把长论文摘要当成精读发布。"""
    paper = _survey_paper()
    bundle = _survey_bundle(paper)

    class LongBundleEngine(V9ReadingEngine):
        def _evidence(self, client, paper_md, workspace, progress):
            return bundle, "确定性来源索引", 0, "complete"

    client = FakeV9Client({
        "brief": _brief(),
        "author": "## 过短正文\n\n" + "只有概括，没有展开。" * 35,
    })

    with pytest.raises(ReadingError, match="低于.*下限|部分过短"):
        LongBundleEngine(client).run(paper, workspace=_v9_workspace(tmp_path))

    author_calls = [call for call in client.calls if call["stage"] == "author"]
    assert len(author_calls) == 2  # 首写 + 一次同范围补写，预算有界
    assert "同一部分的补写" in author_calls[1]["prompt"]
