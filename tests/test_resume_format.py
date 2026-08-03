from __future__ import annotations

import json

from docx import Document

from feishu_assistant.resume_format import (
    DocxQAResult,
    parse_tailor_plan,
    patch_docx_text,
    preserved_docx_parts,
)


def test_docx_qa_only_passes_completed_render_gate():
    assert DocxQAResult("通过：保持 1 页，非正文结构未变化", 1, 1, None, True).passed
    assert not DocxQAResult("失败：页数由 1 变为 2", 1, 2, None, True).passed
    assert not DocxQAResult("通过：保持 1 页，非正文结构未变化", 1, 1, None, False).passed


def test_tailor_plan_requires_exact_evidence_and_rejects_invented_numbers():
    resume = "参与 Agent 工具调用平台建设，完成任务编排。项目上线后稳定运行。"
    raw = json.dumps({
        "summary": "突出 Agent 相关事实",
        "replacements": [
            {
                "source": "参与 Agent 工具调用平台建设，完成任务编排。",
                "replacement": "参与 Agent 工具调用平台建设，完成任务编排与运行验证。",
                "jd_requirement": "Agent 工具调用",
                "evidence": ["参与 Agent 工具调用平台建设"],
            },
            {
                "source": "项目上线后稳定运行。",
                "replacement": "项目上线后稳定性提升 30%。",
                "jd_requirement": "稳定性",
                "evidence": ["项目上线后稳定运行"],
            },
        ],
        "unmet_requirements": ["大模型训练"],
        "questions": ["是否有量化指标"],
    }, ensure_ascii=False)

    plan = parse_tailor_plan(raw, resume)

    assert plan is not None and len(plan.replacements) == 1
    assert "30%" not in plan.replacements[0].replacement
    assert plan.questions == ("是否有量化指标",)


def test_patch_docx_changes_only_document_xml_and_preserves_format(tmp_path):
    source = tmp_path / "source.docx"
    output = tmp_path / "output.docx"
    document = Document()
    paragraph = document.add_paragraph()
    first = paragraph.add_run("参与 Agent ")
    first.bold = True
    paragraph.add_run("工具调用平台建设，完成任务编排。")
    document.sections[0].header.paragraphs[0].text = "姓名｜联系方式"
    document.save(source)
    resume = "参与 Agent 工具调用平台建设，完成任务编排。"
    plan = parse_tailor_plan(json.dumps({
        "summary": "突出岗位相关内容",
        "replacements": [{
            "source": resume,
            "replacement": "参与 Agent 工具调用平台建设，完成任务编排与运行验证。",
            "jd_requirement": "任务编排",
            "evidence": ["参与 Agent 工具调用平台建设"],
        }],
        "unmet_requirements": [], "questions": [],
    }, ensure_ascii=False), resume)
    assert plan is not None

    applied = patch_docx_text(source, output, plan)
    rendered = Document(output)

    assert applied == [resume]
    assert "运行验证" in rendered.paragraphs[0].text
    assert rendered.paragraphs[0].runs[0].bold is True
    assert rendered.sections[0].header.paragraphs[0].text == "姓名｜联系方式"
    assert preserved_docx_parts(source, output) is True
