"""精读发布测试：整篇终审后单次写入、阶段计时、复读与命名。"""

from __future__ import annotations

from pathlib import Path

import pytest

from feishu_assistant.config import Settings
from feishu_assistant.modules.paper import (
    MessageContext,
    PaperReadingService,
    fit_image_size,
)
from feishu_assistant.modules.reading import ReadingResult, StageTimer
from feishu_assistant.store import PaperStore

CTX = MessageContext(chat_id="c1", open_id="ou_x", message_id="m1")


# ---- 图片尺寸 ----

def test_fit_image_size():
    assert fit_image_size(1280, 800, 640) == (640, 400)
    assert fit_image_size(300, 200, 640) == (300, 200)
    assert fit_image_size(0, 0) == (640, 384)


# ---- 精读发布管线 ----

class FakeSender:
    def __init__(self):
        self.texts: list[str] = []
        self.cards: list[dict] = []
        self.md_cards: list[tuple[str, str]] = []  # (title, markdown)

    def send_text(self, chat_id, text):
        self.texts.append(text)
        return True

    def send_card(self, chat_id, card):
        self.cards.append(card)
        return True

    def send_md_card(self, chat_id, title, markdown):
        self.md_cards.append((title, markdown))
        return True


class FakeDocx:
    def __init__(self):
        self.writes = []
        self.titles = []
        self.strict_flags = []

    def create_document(self, title):
        self.titles.append(title)
        return "doc1", "https://feishu.cn/docx/doc1"

    def write_markdown(self, doc_id, md, images=None, start_index=0, strict_images=False):
        self.writes.append((doc_id, md, images, start_index))
        self.strict_flags.append(strict_images)

    def grant_access(self, doc_id, open_id=None):
        return ["tenant_readable"]


class FakeReadingEngine:
    def run(self, paper_md, figures, formulas=None, arxiv_id="", meta_hint=None,
            progress=lambda s: None):
        progress("🧠 骨架生成中（light 链，约 1~2 分钟）……")
        timer = StageTimer()
        timer.mark("骨架")
        progress("✅ 骨架完成，开始分层精讲")
        timer.mark("层1")
        timer.mark("精讲")
        timer.mark("图表")
        return ReadingResult(
            body_md="# 《T》精读\n[[FIG:Figure 1]]" + (
                "\n" + "\n".join(f"[[MATH:{asset.label}]]" for asset in (formulas or []))
                if formulas else ""
            ),
            metadata={"title_zh": "T", "tldr": "一句话", "keywords": ["k"]},
            layer_names=["层A"], figure_notes={}, figure_inlines={},
            vision_used=True, model_calls=3, timer=timer,
            formula_images={asset.label: asset.img_path for asset in (formulas or [])},
        )


def _make_service(tmp_path, monkeypatch):
    from feishu_assistant.lifecycle import PaperCache
    sender = FakeSender()
    store = PaperStore(tmp_path / "p.db")
    svc = PaperReadingService(
        Settings(), kimi=None, lark_client=None,
        store=store, sender=sender,
        vector_index=None, reading_engine=FakeReadingEngine(),
        paper_cache=PaperCache(tmp_path / "cache", store=store),
    )
    svc.docx = FakeDocx()
    def fake_parse(pdf, out):
        path = Path(out) / "paper.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# 论文全文", encoding="utf-8")
        return path
    monkeypatch.setattr("feishu_assistant.modules.paper.parse_pdf", fake_parse)
    monkeypatch.setattr("feishu_assistant.modules.paper.load_figure_assets", lambda d: [])
    monkeypatch.setattr("feishu_assistant.modules.paper.load_formula_assets", lambda d: [])
    monkeypatch.setattr("feishu_assistant.modules.paper.pdf_page_count", lambda p: 9)
    monkeypatch.setattr("feishu_assistant.modules.paper.classify_direction",
                        lambda kimi, t, s: "Agent Harness")
    return svc, sender


def test_pipeline_writes_only_canonical_body_after_generation(tmp_path, monkeypatch):
    svc, sender = _make_service(tmp_path, monkeypatch)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF fake")
    metadata, doc_url, archived = svc._pipeline(
        CTX, pdf, key="2603.03329", pdf_url="", arxiv_id="2603.03329", meta_hint={},
    )
    assert doc_url.endswith("doc1") and archived is False
    assert metadata["direction"] == "Agent Harness"

    # 静默 UX：阶段提示全部进状态条（本用例未注册状态条 → 不落地），
    # 聊天框真实消息只有文档链接一条（persist=True）
    texts = sender.texts
    assert len(texts) == 1
    assert texts[0].startswith("📄") and "https://feishu.cn/docx/doc1" in texts[0]
    # 内容不进聊天框；最终 canonical body 只写一次，不发布中间半成品。
    assert sender.md_cards == []
    assert len(svc.docx.writes) == 1
    assert svc.docx.writes[0][1] == "# 《T》精读\n[[FIG:Figure 1]]"
    assert svc.docx.writes[0][3] == 0
    assert svc.docx.strict_flags == [True]

    # 计时埋点进入 metadata（完成卡用）
    timing = metadata["_timing"]
    assert "骨架" in timing and "精讲" in timing and "图表" in timing
    assert "模型调用 3 次" in timing and "全程" in timing
    # sqlite 落库
    assert svc.store.latest_paper("c1") is not None


def test_pipeline_doc_naming_rule(tmp_path, monkeypatch):
    svc, _ = _make_service(tmp_path, monkeypatch)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF fake")
    svc._pipeline(CTX, pdf, key="k", pdf_url="", arxiv_id="2603.03329",
                  meta_hint={"title": "AutoHarness"})
    import re
    # 文档在终审后创建，使用最终 metadata 标题而不是解析前的临时标题。
    assert re.fullmatch(r"精读：T v1（2603\.03329 · \d{2}-\d{2}）", svc.docx.titles[0])


def test_pipeline_reread_creates_versioned_doc(tmp_path, monkeypatch):
    """版本方案 B：复读生成 v2 新文档；论文库主记录指向最新版，旧链接入历史版本。"""

    class FakeDAO:
        app_token = "app"
        tables = {"论文库": "tbl1"}

        def __init__(self):
            self.updates = []

        def find_record_fields(self, table, field, value):
            return {"版本": "v1", "文档链接": {"text": "旧", "link": "https://feishu.cn/docx/v1doc"},
                    "历史版本": ""}

        def find_record_by_field(self, table, field, value):
            return "rec1"

        def update_record(self, table, record_id, fields):
            self.updates.append((record_id, fields))

    svc, _ = _make_service(tmp_path, monkeypatch)
    dao = FakeDAO()
    svc.bitable_dao = dao
    monkeypatch.setattr(svc, "_ensure_paper_fields", lambda: None)
    monkeypatch.setattr(svc, "_backfill_pick_status", lambda arxiv_id: None)
    svc.store.save_paper("c1", "2603.03329", "T", "旧全文",
                         doc_url="https://feishu.cn/docx/v1doc")
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF fake")
    _, doc_url, archived = svc._pipeline(
        CTX, pdf, key="k", pdf_url="", arxiv_id="2603.03329", meta_hint={"title": "AutoHarness"},
    )
    # 新建 v2 文档（不续写旧文档）
    assert len(svc.docx.titles) == 1 and " v2（2603.03329 · " in svc.docx.titles[0]
    assert doc_url.endswith("doc1") and archived is True
    # 主记录更新：文档链接→最新版、版本→v2、历史版本串起 v1
    record_id, fields = dao.updates[0]
    assert record_id == "rec1"
    assert fields["版本"] == "v2"
    assert "v1: https://feishu.cn/docx/v1doc" in fields["历史版本"]
    assert fields["文档链接"]["link"].endswith("doc1")


def test_pipeline_first_read_no_prior(tmp_path, monkeypatch):
    """无 bitable 先记录（dao 为空）：正常新建文档，归档跳过不炸。"""
    svc, _ = _make_service(tmp_path, monkeypatch)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF fake")
    _, doc_url, archived = svc._pipeline(
        CTX, pdf, key="k", pdf_url="", arxiv_id="2603.03329", meta_hint={"title": "X"},
    )
    assert len(svc.docx.titles) == 1 and " v1（2603.03329 · " in svc.docx.titles[0]
    assert archived is False  # dao None → 跳过归档


def test_pipeline_passes_figure_paths_to_doc(tmp_path, monkeypatch):
    svc, _ = _make_service(tmp_path, monkeypatch)
    from feishu_assistant.modules.figures import FigureAsset
    asset = FigureAsset("Figure 1", "cap", tmp_path / "f.jpg", "image", 1, 0)
    (tmp_path / "f.jpg").write_bytes(b"\xff\xd8\xff")
    monkeypatch.setattr("feishu_assistant.modules.paper.load_figure_assets", lambda d: [asset])
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF fake")
    svc._pipeline(CTX, pdf, key="k", pdf_url="", arxiv_id="x", meta_hint={})
    _, _, images, _ = svc.docx.writes[0]
    assert images == {"Figure 1": tmp_path / "f.jpg"}


def test_pipeline_combines_formula_images_with_figure_images(tmp_path, monkeypatch):
    svc, _ = _make_service(tmp_path, monkeypatch)
    from feishu_assistant.modules.figures import FigureAsset
    from feishu_assistant.modules.formulas import FormulaAsset

    figure_path = tmp_path / "f.jpg"
    formula_path = tmp_path / "eq.png"
    figure_path.write_bytes(b"\xff\xd8\xff")
    formula_path.write_bytes(b"png")
    figure = FigureAsset("Figure 1", "cap", figure_path, "image", 1, 0)
    formula = FormulaAsset("Equation 2", r"x=1", formula_path, 1, 1, "Method")
    monkeypatch.setattr("feishu_assistant.modules.paper.load_figure_assets", lambda d: [figure])
    monkeypatch.setattr("feishu_assistant.modules.paper.load_formula_assets", lambda d: [formula])
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF fake")
    svc._pipeline(CTX, pdf, key="k", pdf_url="", arxiv_id="x", meta_hint={})
    images = svc.docx.writes[0][2]
    assert images == {"Figure 1": figure_path, "Equation 2": formula_path}


def test_pipeline_write_failure_saves_body_and_skips_archive(tmp_path, monkeypatch):
    """发布写入失败：有界重试后保住正文（本地落盘+通知），不归档不重跑。"""
    monkeypatch.setattr("time.sleep", lambda *_: None)
    svc, _ = _make_service(tmp_path, monkeypatch)

    def fail_write(*args, **kwargs):
        raise RuntimeError("write failed")

    svc.docx.write_markdown = fail_write
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF fake")
    metadata, doc_url, archived = svc._pipeline(
        CTX, pdf, key="k", pdf_url="", arxiv_id="x", meta_hint={})
    assert doc_url == "" and archived is False
    assert metadata["publish_failed"]
    assert Path(metadata["unpublished_body"]).read_text(encoding="utf-8")
    assert svc.store.latest_paper("c1") is None


def test_pipeline_permission_failure_saves_body_without_publishing(tmp_path, monkeypatch):
    """权限授予失败：重试后保住正文，不写文档不归档。"""
    monkeypatch.setattr("time.sleep", lambda *_: None)
    svc, sender = _make_service(tmp_path, monkeypatch)
    svc.docx.grant_access = lambda *args, **kwargs: []
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF fake")
    metadata, doc_url, archived = svc._pipeline(
        CTX, pdf, key="k", pdf_url="", arxiv_id="x", meta_hint={})
    assert doc_url == "" and archived is False
    assert metadata["publish_failed"]
    assert svc.docx.writes == []
    assert svc.store.latest_paper("c1") is None




def test_job_arxiv_progress_download(tmp_path, monkeypatch):
    svc, sender = _make_service(tmp_path, monkeypatch)
    def fake_download(_arxiv_id, directory):
        pdf = Path(directory) / "paper.pdf"
        pdf.write_bytes(b"%PDF fake" * 2000)
        return pdf
    monkeypatch.setattr("feishu_assistant.modules.paper.download_arxiv_pdf", fake_download)
    monkeypatch.setattr("feishu_assistant.modules.paper.fetch_arxiv_metadata", lambda a: {"title": "T"})
    svc._job_arxiv(CTX, "2603.03329")
    # 静默 UX：「✅ 下载完成」只进状态条阶段行，聊天框真实消息仅文档链接
    assert sender.texts == [t for t in sender.texts if t.startswith("📄")]
    assert not any(t.startswith("✅ 下载完成") for t in sender.texts)


def test_progress_updates_status_bar_stage(tmp_path, monkeypatch):
    """消息上下文带状态条：_progress 只更新阶段行；persist=True 才发真实消息。
    状态条按 MessageContext 绑定——两篇精读并发各进各的条，互不串行。"""
    svc, sender = _make_service(tmp_path, monkeypatch)

    class FakeStatus:
        def __init__(self):
            self.stages: list[str] = []

        def set_stage(self, s):
            self.stages.append(s)

    s1, s2 = FakeStatus(), FakeStatus()
    ctx1 = MessageContext(chat_id=CTX.chat_id, status=s1)
    ctx2 = MessageContext(chat_id=CTX.chat_id, status=s2)
    svc._progress(ctx1, "🧠 骨架生成中")
    svc._progress(ctx2, "🧠 另一篇骨架生成中")
    svc._progress(ctx1, "📄 文档已建：http://x", persist=True)
    assert s1.stages == ["🧠 骨架生成中", "📄 文档已建：http://x"]
    assert s2.stages == ["🧠 另一篇骨架生成中"]  # 并发不串
    assert sender.texts == ["📄 文档已建：http://x"]
