"""P3/P4 集成测试：命令注册终表、钩子链顺序、命令委托、bot 文件分流、config 解析。

全部离线：fake 服务 + 直接构造 Router/FeishuBot，不碰网络与真实凭证。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from feishu_assistant.config import Settings, load_settings
from feishu_assistant.feishu.bot import FeishuBot, extract_audio, extract_file
from feishu_assistant.modules.paper import MessageContext
from feishu_assistant.modules.router import COMMANDS, Router

CTX = MessageContext(chat_id="c1", open_id="ou_x", message_id="m1")


# ---------------------------------------------------------------------------
# 命令注册终表
# ---------------------------------------------------------------------------

def _registry() -> dict[str, tuple]:
    """口令 → (handler, help) 映射（含 arg_pattern 命令的主口令）。"""
    return {name: (cmd.handler, cmd.help) for cmd in COMMANDS for name in cmd.names}


def test_registry_contains_all_p3_p4_commands():
    reg = _registry()
    expected = {
        # P4
        "arXiv日报": "_cmd_arxiv_daily", "arxiv日报": "_cmd_arxiv_daily",
        "组会周报": "_cmd_weekly", "周报": "_cmd_weekly",
        "今日": "_cmd_today", "今日提醒": "_cmd_today",
        # P3
        "看板": "_cmd_career_board", "求职看板": "_cmd_career_board",
        "进展": "_cmd_career_progress", "进展如何": "_cmd_career_progress",
        "押题": "_cmd_career_predict", "记面经": "_cmd_career_save_exp",
        "面经": "_cmd_career_search_exp", "面经查询": "_cmd_career_search_exp",
        "简历": "_cmd_career_resume", "简历列表": "_cmd_career_resume",
        "模拟面试": "_cmd_career_mock",
        "面试复盘": "_cmd_career_review_hint", "复盘": "_cmd_career_review_hint",
    }
    for name, handler in expected.items():
        assert name in reg, f"注册表缺口令「{name}」"
        assert reg[name][0] == handler, f"「{name}」handler 应为 {handler}"


def test_registry_no_duplicate_names_or_handlers():
    names = [name for cmd in COMMANDS for name in cmd.names]
    assert len(names) == len(set(names)), "口令重复"
    handlers = [cmd.handler for cmd in COMMANDS]
    assert len(handlers) == len(set(handlers)), "handler 重复"


def test_placeholders_removed():
    handlers = {cmd.handler for cmd in COMMANDS}
    assert not any(h.startswith("_cmd_coming") for h in handlers), "P3 占位命令未清干净"
    for cmd in COMMANDS:
        assert "即将上线" not in cmd.help and "P3 上线" not in cmd.help and "P4 上线" not in cmd.help
    # 每个 handler 都在 Router 上有实现
    router = Router()
    for cmd in COMMANDS:
        assert callable(getattr(router, cmd.handler, None)), cmd.handler


# ---------------------------------------------------------------------------
# 钩子链顺序：followup → career → inbox → chat
# ---------------------------------------------------------------------------

def make_chain_router():
    calls: list[str] = []
    router = Router(
        chat_fn=lambda t, c: (calls.append("chat"), "CHAT")[1],
        paper_fn=lambda t, c: None,
        followup_fn=lambda t, c: (calls.append("followup"), None)[1],
        career_fn=lambda t, c: (calls.append("career"), "CAREER" if t == "career命中" else None)[1],
        inbox_fn=lambda t, c: (calls.append("inbox"), "INBOX" if t == "inbox命中" else None)[1],
    )
    return router, calls


def test_hook_chain_career_before_inbox_before_chat():
    router, calls = make_chain_router()
    assert router.handle("career命中") == "CAREER"
    assert calls == ["followup", "career"]  # career 接管后不触达 inbox/chat

    calls.clear()
    assert router.handle("inbox命中") == "INBOX"
    assert calls == ["followup", "career", "inbox"]

    calls.clear()
    assert router.handle("普通闲聊") == "CHAT"
    assert calls == ["followup", "career", "inbox", "chat"]  # 顺序全链


# ---------------------------------------------------------------------------
# 命令委托（fake 服务记录调用）
# ---------------------------------------------------------------------------

class FakeCareerSvc:
    def __init__(self):
        self.calls: list[tuple] = []

    def cmd_board(self, arg, ctx):
        self.calls.append(("board", arg))
        return f"看板:{arg}"

    def cmd_predict(self, arg, ctx):
        self.calls.append(("predict", arg))
        return f"押题:{arg}"

    def cmd_save_exp(self, arg, ctx):
        self.calls.append(("save_exp", arg))
        return f"记面经:{arg}"

    def cmd_search_exp(self, arg, ctx):
        self.calls.append(("search_exp", arg))
        return f"面经:{arg}"

    def cmd_resume(self, arg, ctx):
        self.calls.append(("resume", arg))
        return f"简历:{arg}"

    def cmd_mock_interview(self, arg, ctx):
        self.calls.append(("mock", arg))
        return f"模拟面试:{arg}"


class FakeDigestSvc:
    def __init__(self):
        self.calls: list[str] = []

    def cmd_today(self, ctx):
        self.calls.append("today")
        return "今日汇总"

    def cmd_arxiv_daily(self, ctx):
        self.calls.append("arxiv")
        return "日报 ack"

    def cmd_weekly(self, ctx):
        self.calls.append("weekly")
        return "周报 ack"


def test_command_delegation_to_services():
    career, digest = FakeCareerSvc(), FakeDigestSvc()
    router = Router(chat_fn=lambda t, c: "CHAT", career_svc=career, digest_svc=digest)

    assert router.handle("看板", CTX) == "看板:"
    assert router.handle("看板 字节改一面过了", CTX) == "看板:字节改一面过了"
    assert router.handle("求职看板 新增 美团 算法", CTX) == "看板:新增 美团 算法"
    assert router.handle("进展", CTX) == "看板:"  # 进展 = 空参看板汇总
    assert router.handle("押题 字节", CTX) == "押题:字节"
    assert router.handle("记面经 一面问了 KV cache", CTX) == "记面经:一面问了 KV cache"
    assert router.handle("面经 字节 一面", CTX) == "面经:字节 一面"
    assert router.handle("面经查询 字节", CTX) == "面经:字节"
    assert router.handle("简历 算法", CTX) == "简历:算法"
    assert router.handle("简历列表", CTX) == "简历:"
    assert router.handle("模拟面试 字节", CTX) == "模拟面试:字节"

    assert router.handle("今日", CTX) == "今日汇总"
    assert router.handle("arXiv日报", CTX) == "日报 ack"
    assert router.handle("组会周报", CTX) == "周报 ack"
    assert digest.calls == ["today", "arxiv", "weekly"]


def test_review_hint_command():
    router = Router(chat_fn=lambda t, c: "CHAT")
    reply = router.handle("复盘", CTX)
    assert "录音" in reply and "记面经" in reply
    assert router.handle("面试复盘", CTX) == reply


# ---------------------------------------------------------------------------
# bot 文件分流
# ---------------------------------------------------------------------------

class FakePaperSvc:
    def __init__(self):
        self.files: list[tuple] = []

    def handle_file(self, file_name, file_key, context):
        self.files.append((file_name, file_key))
        return f"精读:{file_name}"


class FakeCareerFileSvc:
    def __init__(self):
        self.resumes: list[tuple] = []
        self.audios: list[tuple] = []

    def handle_resume_file(self, file_name, file_key, context):
        self.resumes.append((file_name, file_key))
        return f"简历:{file_name}"

    def handle_audio_file(self, file_name, file_key, context, mode="review"):
        self.audios.append((file_name, file_key, mode))
        return f"音频:{file_name}"


def make_bot():
    settings = Settings(feishu_app_id="cli_x", feishu_app_secret="secret_x")
    bot = FeishuBot(settings, Router())
    paper, career = FakePaperSvc(), FakeCareerFileSvc()
    bot.paper_service = paper
    bot.career_service = career
    return bot, paper, career


def test_bot_route_file_pdf_to_paper():
    bot, paper, career = make_bot()
    assert bot._route_file("attention.pdf", "fk", CTX) == "精读:attention.pdf"
    assert paper.files == [("attention.pdf", "fk")] and career.resumes == []


def test_bot_route_file_resume_pdf_and_docx_to_career():
    bot, paper, career = make_bot()
    assert bot._route_file("居世豪-简历-v3.pdf", "fk1", CTX) == "简历:居世豪-简历-v3.pdf"
    assert bot._route_file("resume_2026.pdf", "fk2", CTX) == "简历:resume_2026.pdf"
    assert bot._route_file("求职简历.docx", "fk3", CTX) == "简历:求职简历.docx"
    assert len(career.resumes) == 3 and paper.files == []


def test_bot_route_file_audio_to_career():
    bot, paper, career = make_bot()
    assert bot._route_file("面试录音.opus", "fk", CTX) == "音频:面试录音.opus"
    assert bot._route_file("interview.m4a", "fk2", CTX) == "音频:interview.m4a"
    assert career.audios == [
        ("面试录音.opus", "fk", "review"),
        ("interview.m4a", "fk2", "review"),
    ]
    assert paper.files == []


def test_bot_route_named_voice_experience_to_exp_mode():
    bot, _, career = make_bot()
    assert bot._route_file("字节二面面经.opus", "fk", CTX) == "音频:字节二面面经.opus"
    assert career.audios == [("字节二面面经.opus", "fk", "exp")]


def test_bot_route_file_career_missing_falls_back():
    bot, paper, _ = make_bot()
    bot.career_service = None
    assert bot._route_file("简历.pdf", "fk", CTX) == "精读:简历.pdf"  # career 缺位时 PDF 兜底精读
    assert "未就绪" in bot._route_file("简历.docx", "fk", CTX)  # docx 精读接不住，给提示
    assert "未就绪" in bot._route_file("a.opus", "fk", CTX)


def _fake_event(msg_type: str, content: dict):
    return SimpleNamespace(
        event=SimpleNamespace(
            message=SimpleNamespace(message_type=msg_type, content=json.dumps(content)),
        )
    )


def test_extract_audio_and_file():
    audio = _fake_event("audio", {"file_key": "fk_voice", "duration": 3200})
    assert extract_audio(audio) == "fk_voice"
    assert extract_file(audio) is None  # audio 不走文件通道
    file_msg = _fake_event("file", {"file_name": "a.pdf", "file_key": "fk_file"})
    assert extract_file(file_msg) == ("a.pdf", "fk_file")
    assert extract_audio(file_msg) is None
    text_msg = _fake_event("text", {"text": "你好"})
    assert extract_audio(text_msg) is None and extract_file(text_msg) is None


# ---------------------------------------------------------------------------
# config push_chat_id 解析
# ---------------------------------------------------------------------------

_MINIMAL_CONFIG = """\
providers:
  free-tokens:
    base_url: "https://free-tokens.msh.team/v1"
    api_key_env: "FREE_TOKENS_API_KEY"
    enabled: true
routing:
  heavy:
    - provider: "free-tokens"
      model: "kimi-k3"
  light:
    - provider: "free-tokens"
      model: "kimi-k3-highspeed"
push:
  daily_time: "08:30"
  chat_id: "oc_test_push_123"
"""


def test_config_push_chat_id_parsed(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_MINIMAL_CONFIG, encoding="utf-8")
    env = tmp_path / ".env"
    env.write_text("", encoding="utf-8")
    settings = load_settings(env_path=env, config_path=cfg)
    assert settings.push_chat_id == "oc_test_push_123"
    assert settings.push_daily_time == "08:30"


def test_config_push_chat_id_default_empty(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_MINIMAL_CONFIG.replace('  chat_id: "oc_test_push_123"\n', ""), encoding="utf-8")
    env = tmp_path / ".env"
    env.write_text("", encoding="utf-8")
    settings = load_settings(env_path=env, config_path=cfg)
    assert settings.push_chat_id == ""


# ---------------------------------------------------------------------------
# 精选库（P5）：config 写回追加 + 精读回填状态
# ---------------------------------------------------------------------------

_CONFIG_WITH_7_TABLES = """\
providers:
  free-tokens:
    base_url: "https://free-tokens.msh.team/v1"
    api_key_env: "FREE_TOKENS_API_KEY"
    enabled: true
routing:
  heavy:
    - provider: "free-tokens"
      model: "kimi-k3"
  light:
    - provider: "free-tokens"
      model: "kimi-k3-highspeed"
bitable:
  app_token: ""
  tables:
    论文库: ""
    收藏库: ""
    灵感库: ""
    求职看板: ""
    面经库: ""
    简历库: ""
    备忘库: ""
"""


def test_write_bitable_config_appends_missing_table(tmp_path):
    """schema 新增的表（精选库）在 config.yaml 没有行：追加而非报错。"""
    from feishu_assistant.config import load_settings, write_bitable_config

    cfg = tmp_path / "config.yaml"
    cfg.write_text(_CONFIG_WITH_7_TABLES, encoding="utf-8")
    tables = {name: f"tbl_{name}" for name in
              ("论文库", "收藏库", "灵感库", "求职看板", "面经库", "简历库", "备忘库", "精选库")}
    write_bitable_config("app_tok_1", tables, config_path=cfg)

    # 写回后可被 load_settings 正常解析（YAML 合法且键齐全）
    env = tmp_path / ".env"
    env.write_text("", encoding="utf-8")
    settings = load_settings(env_path=env, config_path=cfg)
    assert settings.bitable_app_token == "app_tok_1"
    assert settings.bitable_tables["精选库"] == "tbl_精选库"  # 追加行生效
    assert settings.bitable_tables["论文库"] == "tbl_论文库"  # 原有行被替换
    assert settings.bitable_ready()  # 8 张表全部有值 → ready


class FakeDaoForBackfill:
    """BitableDAO 替身：精选库查字段 + 更新。"""

    def __init__(self, found="rec_pick_1"):
        self.found = found
        self.updated: list[tuple] = []

    def find_record_by_field(self, table, field, value):
        assert table == "精选库" and field == "arXivID"
        return self.found if value == "2607.00001" else None

    def update_record(self, table, record_id, fields):
        self.updated.append((table, record_id, fields))


def test_paper_backfill_pick_status(tmp_path):
    from feishu_assistant.modules.paper import PaperReadingService
    from feishu_assistant.store import PaperStore

    svc = PaperReadingService(
        settings=Settings(), kimi=None, lark_client=None,
        store=PaperStore(tmp_path / "p.db"), sender=None,
    )
    dao = FakeDaoForBackfill()
    svc.bitable_dao = dao

    svc._backfill_pick_status("2607.00001")  # 命中 → 更新为已精读
    assert dao.updated == [("精选库", "rec_pick_1", {"状态": "已精读"})]

    dao.updated.clear()
    svc._backfill_pick_status("2607.99999")  # 非日报精选 → 不动
    assert dao.updated == []

    dao.updated.clear()
    svc._backfill_pick_status("")  # 空 ID（附件精读）→ 直接跳过
    assert dao.updated == []

    svc.bitable_dao = None  # 多维表格未初始化 → 静默
    svc._backfill_pick_status("2607.00001")


class TestMdForCard:
    def test_headers_and_tables(self):
        from feishu_assistant.feishu.bot import md_for_card
        md = ("## 标题一\n\n| 字段 | 内容 |\n|---|---|\n| 题目 | X |\n| 年份 | 2026 |\n\n"
              "### 三列\n| 指标 | 本文 | 对照 |\n|---|---|---|\n| 合法率 | 100% | 67.97% |\n\n"
              "**加粗** 和 [链接](https://arxiv.org)\n")
        out = md_for_card(md)
        assert "**标题一**" in out and "##" not in out.replace("#{1,", "")
        assert "• **题目**：X" in out
        assert "**指标 / 本文 / 对照**" in out and "• 合法率 / 100% / 67.97%" in out
        assert "|" not in out  # 管道表格全部转写
        assert "[链接](https://arxiv.org)" in out

    def test_non_table_pipes_kept(self):
        from feishu_assistant.feishu.bot import md_for_card
        out = md_for_card("普通 | 竖线文本 | 不是表格")
        assert "普通 | 竖线文本 | 不是表格" in out


class TestCardActionCallback:
    def _bot_with_svc(self, dao):
        from feishu_assistant.feishu.bot import FeishuBot
        from feishu_assistant.config import Settings
        from unittest.mock import MagicMock
        bot = FeishuBot.__new__(FeishuBot)
        svc = MagicMock()
        svc.bitable_dao = dao
        bot.paper_service = svc
        return bot

    def _evt(self, value):
        from unittest.mock import MagicMock
        d = MagicMock()
        d.event.action.value = value
        return d

    def test_mark_read_flips_status(self):
        from unittest.mock import MagicMock
        dao = MagicMock()
        dao.find_record_by_field.return_value = "rec1"
        bot = self._bot_with_svc(dao)
        resp = bot._on_card_action(self._evt({"act": "mark_read", "arxiv_id": "2603.03329", "title": "T"}))
        assert resp.toast.type == "success"
        dao.update_record.assert_called_once_with("精选库", "rec1", {"状态": "已读"})

    def test_mark_read_not_found(self):
        from unittest.mock import MagicMock
        dao = MagicMock()
        dao.find_record_by_field.return_value = None
        bot = self._bot_with_svc(dao)
        resp = bot._on_card_action(self._evt({"act": "mark_read", "arxiv_id": "x", "title": "T"}))
        assert resp.toast.type == "warning"
        dao.update_record.assert_not_called()

    def test_unknown_act(self):
        from unittest.mock import MagicMock
        dao = MagicMock()
        bot = self._bot_with_svc(dao)
        resp = bot._on_card_action(self._evt({"act": "zzz", "arxiv_id": "x", "title": "T"}))
        assert resp.toast.type == "warning"

    def test_toast_attrs(self):
        # SDK 版本差异回归：toast/response 必须用属性赋值（构造 kwarg 会 TypeError）
        from unittest.mock import MagicMock
        dao = MagicMock()
        bot = self._bot_with_svc(dao)
        resp = bot._on_card_action(self._evt({"act": "zzz"}))
        assert isinstance(resp.toast.content, str) and resp.toast.content


class TestModelService:
    def _svc(self, tmp_path, monkeypatch):
        from feishu_assistant.modules import models
        monkeypatch.setattr(models, "OVERRIDE_PATH", tmp_path / "ov.json")
        from feishu_assistant.modules.models import ModelService
        return models, ModelService(self._settings())

    def _settings(self):
        from feishu_assistant.config import Settings, ProviderConfig, ModelCandidate
        return Settings(
            providers={
                "free-tokens": ProviderConfig("free-tokens", "https://x/v1", "K", True, "key1"),
                "moonshot": ProviderConfig("moonshot", "https://y/v1", "K2", True, "key2"),
                "kimi-coding": ProviderConfig("kimi-coding", "https://z/v1", "K3", False, ""),
            },
            routing_heavy=[ModelCandidate("free-tokens", "kimi-k3"), ModelCandidate("moonshot", "kimi-k2-thinking")],
            routing_light=[ModelCandidate("free-tokens", "kimi-k3-highspeed"), ModelCandidate("moonshot", "kimi-k2-0905")],
        )

    def test_switch_moves_to_front_and_persists(self, tmp_path, monkeypatch):
        models, svc = self._svc(tmp_path, monkeypatch)
        msg = svc.switch("heavy", "moonshot/kimi-k2-thinking")
        assert "已切换" in msg
        assert svc.settings.routing_heavy[0].label() == "moonshot/kimi-k2-thinking"
        assert svc.settings.routing_light[0].label() == "moonshot/kimi-k2-thinking"
        # 重启语义：重新加载 override 后依然首选
        applied = models.apply_override(self._settings(), models.load_override(tmp_path / "ov.json"))
        assert applied == ["active=moonshot/kimi-k2-thinking"]

    def test_switch_disabled_or_no_key_rejected(self, tmp_path, monkeypatch):
        models, svc = self._svc(tmp_path, monkeypatch)
        assert "disabled" in svc.switch("heavy", "kimi-coding/k3")
        assert "没配 key" not in svc.switch("heavy", "kimi-coding/k3")

    def test_switch_unknown_provider(self, tmp_path, monkeypatch):
        models, svc = self._svc(tmp_path, monkeypatch)
        assert "没有 provider" in svc.switch("heavy", "openai/gpt-5")

    def test_override_preserved_when_provider_unusable(self, tmp_path, monkeypatch):
        models = __import__("feishu_assistant.modules.models", fromlist=["apply_override"])
        s = self._settings()
        s.providers["moonshot"].api_key = ""
        applied = models.apply_override(s, {"heavy": "moonshot/kimi-k2-thinking"})
        assert applied == ["active=moonshot/kimi-k2-thinking（由旧 heavy/light 选择迁移）"]
        assert s.routing_heavy[0].label() == "moonshot/kimi-k2-thinking"
        assert s.routing_light[0].label() == "moonshot/kimi-k2-thinking"

    def test_status_text_contains_chains_and_providers(self, tmp_path, monkeypatch):
        models, svc = self._svc(tmp_path, monkeypatch)
        text = svc.status_text()
        assert "全局模型" in text and "free-tokens" in text and "kimi-coding" in text

    # ---- 交互卡片 ----

    def _svc_static_menu(self, tmp_path, monkeypatch):
        """_provider_models 打桩成静态菜单，避免 free-tokens 走真实 /v1/models。"""
        models, svc = self._svc(tmp_path, monkeypatch)
        menus = {"free-tokens": ["kimi-k3", "kimi-k3-highspeed", "gpt-5.5"],
                 "moonshot": ["kimi-k2-thinking"]}
        monkeypatch.setattr(svc, "_provider_models", lambda name: menus.get(name, []))
        return models, svc

    def test_build_card_selects_and_options(self, tmp_path, monkeypatch):
        models, svc = self._svc_static_menu(tmp_path, monkeypatch)
        card = svc.build_card()
        selects = [a for el in card["elements"] if el["tag"] == "action"
                   for a in el["actions"] if a["tag"] == "select_static"]
        assert len(selects) == 1
        global_sel = selects[0]
        assert global_sel["value"] == {"act": "model_switch", "chain": "all"}
        assert global_sel["initial_option"] == "free-tokens/kimi-k3"
        vals = [o["value"] for o in global_sel["options"]]
        # 只列可用 provider（kimi-coding disabled 不进选项）
        assert "free-tokens/gpt-5.5" in vals and "moonshot/kimi-k2-thinking" in vals
        assert not any(v.startswith("kimi-coding/") for v in vals)
        # 恢复默认按钮 + 添加 provider 翻页按钮（表单已拆到独立卡片）
        buttons = [a for el in card["elements"] if el["tag"] == "action"
                   for a in el["actions"] if a["tag"] == "button"]
        acts = [b["value"]["act"] for b in buttons]
        assert "model_reset" in acts and "model_add_form" in acts
        assert not any(el["tag"] == "form" for el in card["elements"])  # 面板卡不含表单

    def test_add_form_card_and_navigation(self, tmp_path, monkeypatch):
        models, svc = self._svc_static_menu(tmp_path, monkeypatch)
        toast, ok, form_card = svc.handle_card_action("model_add_form")
        assert ok
        forms = [el for el in form_card["elements"] if el["tag"] == "form"]
        assert len(forms) == 1
        names = [e.get("name") for e in forms[0]["elements"]]
        assert names[:5] == ["pname", "base_url", "api_key", "models", "protocol"]
        submit = forms[0]["elements"][-1]
        assert submit["action_type"] == "form_submit" and submit["name"] == "submit_add"
        # 返回按钮翻回面板
        back = next(a for el in form_card["elements"] if el["tag"] == "action"
                    for a in el["actions"] if a["value"]["act"] == "model_panel")
        toast2, ok2, panel = svc.handle_card_action("model_panel")
        assert ok2 and not any(el["tag"] == "form" for el in panel["elements"])

    def test_card_action_switch_returns_fresh_card(self, tmp_path, monkeypatch):
        models, svc = self._svc_static_menu(tmp_path, monkeypatch)
        toast, ok, card = svc.handle_card_action("model_switch", chain="heavy",
                                                 option="free-tokens/gpt-5.5")
        assert ok and "gpt-5.5" in toast
        assert svc.settings.routing_heavy[0].label() == "free-tokens/gpt-5.5"
        assert svc.settings.routing_light[0].label() == "free-tokens/gpt-5.5"
        global_sel = next(a for el in card["elements"] if el["tag"] == "action"
                          for a in el["actions"] if a["tag"] == "select_static")
        assert global_sel["initial_option"] == "free-tokens/gpt-5.5"

    def test_card_action_reject_shows_reason(self, tmp_path, monkeypatch):
        models, svc = self._svc_static_menu(tmp_path, monkeypatch)
        toast, ok, _ = svc.handle_card_action("model_switch", chain="heavy",
                                              option="kimi-coding/k3")
        assert not ok and "disabled" in toast

    def test_card_action_reset_restores_defaults(self, tmp_path, monkeypatch):
        models, svc = self._svc_static_menu(tmp_path, monkeypatch)
        svc.switch("heavy", "moonshot/kimi-k2-thinking")
        assert (tmp_path / "ov.json").exists()
        defaults = self._settings()  # 全新默认链
        import feishu_assistant.config as config_mod
        monkeypatch.setattr(config_mod, "load_settings", lambda: defaults)
        toast, ok, card = svc.handle_card_action("model_reset")
        assert ok and "默认" in toast
        assert not (tmp_path / "ov.json").exists()
        assert svc.settings.routing_heavy[0].label() == "free-tokens/kimi-k3"

    # ---- 卡片添加 provider ----

    def _svc_add(self, tmp_path, monkeypatch):
        """custom/env/pending 路径全部打到 tmp，避免碰真实文件；还原真实 _provider_models。"""
        models, svc = self._svc_static_menu(tmp_path, monkeypatch)
        monkeypatch.setattr(models, "CUSTOM_PATH", tmp_path / "custom.json")
        monkeypatch.setattr(models, "ENV_FILE", tmp_path / ".env")
        monkeypatch.setattr(models, "PENDING_PATH", tmp_path / "pending.json")
        monkeypatch.setattr(svc, "_provider_models",
                            lambda name: models.ModelService._provider_models(svc, name))
        return models, svc

    def test_add_provider_persists_and_hot_applies(self, tmp_path, monkeypatch):
        models, svc = self._svc_add(tmp_path, monkeypatch)
        msg, ok = svc.add_provider("openrouter", "https://openrouter.ai/api/v1/",
                                   "sk-or-x", "gpt-5.5, claude-4")
        assert ok and "openrouter" in msg
        # 持久化：custom json + .env
        saved = models.load_custom(tmp_path / "custom.json")
        assert saved["openrouter"]["base_url"] == "https://openrouter.ai/api/v1"  # 尾斜杠被去掉
        assert saved["openrouter"]["api_key_env"] == "OPENROUTER_API_KEY"
        assert "OPENROUTER_API_KEY=sk-or-x" in (tmp_path / ".env").read_text()
        # 热生效：provider 可用 + 两模型进 heavy/light 链尾（不是首选）
        assert svc.settings.providers["openrouter"].usable()
        assert svc.settings.routing_heavy[0].label() == "free-tokens/kimi-k3"
        assert svc.settings.routing_heavy[-1].label() == "openrouter/claude-4"
        assert svc.settings.routing_light[-1].label() == "openrouter/claude-4"
        # 下拉里能选到新模型
        assert svc._provider_models("openrouter") == ["gpt-5.5", "claude-4"]

    def test_add_provider_validation(self, tmp_path, monkeypatch):
        models, svc = self._svc_add(tmp_path, monkeypatch)
        assert not svc.add_provider("Bad Name", "https://x/v1", "k", "m")[1]      # 大写/空格
        assert not svc.add_provider("free-tokens", "https://x/v1", "k", "m")[1]  # 已存在
        assert not svc.add_provider("p2", "x/v1", "k", "m")[1]                   # 非 http
        assert not svc.add_provider("p2", "https://x/v1", "k", " ,")[1]          # 空模型
        assert not (tmp_path / "custom.json").exists()

    def test_add_provider_via_form_action(self, tmp_path, monkeypatch):
        models, svc = self._svc_add(tmp_path, monkeypatch)
        toast, ok, card = svc.handle_card_action(
            "model_add_provider",
            form_value={"pname": "groq", "base_url": "https://api.groq.com/openai/v1",
                        "api_key": "gsk_x", "models": "llama-4"})
        assert ok and "groq" in toast
        # 成功后面板卡回刷，全局下拉含新 provider 的模型
        global_sel = next(a for el in card["elements"] if el["tag"] == "action"
                          for a in el["actions"] if a["tag"] == "select_static")
        assert "groq/llama-4" in [o["value"] for o in global_sel["options"]]

    def test_add_provider_failure_stays_on_form(self, tmp_path, monkeypatch):
        models, svc = self._svc_add(tmp_path, monkeypatch)
        toast, ok, card = svc.handle_card_action(
            "model_add_provider",
            form_value={"pname": "Bad Name", "base_url": "https://x/v1",
                        "api_key": "k", "models": "m"})
        assert not ok
        assert card is None  # 校验失败不刷新卡片：表单已填内容保留

    # ---- 模型留空 → 探测 /models → 挑选卡 ----

    def _submit_no_models(self, svc, monkeypatch, fetched=None, probe_exc=None):
        if probe_exc is not None:
            monkeypatch.setattr(svc, "_probe_models",
                                lambda *a: (_ for _ in ()).throw(probe_exc))
        else:
            monkeypatch.setattr(svc, "_probe_models", lambda *a: fetched)
        return svc.handle_card_action(
            "model_add_provider",
            form_value={"pname": "openrouter", "base_url": "https://openrouter.ai/api/v1",
                        "api_key": "sk-or-x", "models": "", "protocol": ""})

    def test_probe_success_shows_pick_card_and_stashes_pending(self, tmp_path, monkeypatch):
        models, svc = self._svc_add(tmp_path, monkeypatch)
        toast, ok, card = self._submit_no_models(svc, monkeypatch,
                                                 fetched=["gpt-5.5", "claude-4"])
        assert ok and "2 个模型" in toast
        # 挑选卡：列出模型 + 提交按钮 act=model_confirm_provider
        assert any(el["tag"] == "form" for el in card["elements"])
        md = next(el["content"] for el in card["elements"] if el["tag"] == "markdown")
        assert "gpt-5.5" in md and "claude-4" in md
        # 凭证暂存服务端（含 key），卡片 JSON 里不出现 key
        pending = models.load_pending(tmp_path / "pending.json")
        assert pending["api_key"] == "sk-or-x" and pending["fetched"] == ["gpt-5.5", "claude-4"]
        assert "sk-or-x" not in str(card)

    def test_probe_failure_keeps_form_values(self, tmp_path, monkeypatch):
        models, svc = self._svc_add(tmp_path, monkeypatch)
        toast, ok, card = self._submit_no_models(svc, monkeypatch,
                                                 probe_exc=RuntimeError("boom"))
        assert not ok and "手动填写" in toast
        assert card is None  # 不刷卡，已填内容保留
        assert not (tmp_path / "pending.json").exists()

    def test_confirm_all_models(self, tmp_path, monkeypatch):
        models, svc = self._svc_add(tmp_path, monkeypatch)
        self._submit_no_models(svc, monkeypatch, fetched=["gpt-5.5", "claude-4"])
        toast, ok, card = svc.handle_card_action("model_confirm_provider",
                                                 form_value={"models_pick": ""})
        assert ok and "openrouter" in toast
        assert svc._provider_models("openrouter") == ["gpt-5.5", "claude-4"]  # 留空=全部
        assert not (tmp_path / "pending.json").exists()  # 确认后清理暂存
        assert svc.settings.routing_heavy[-1].label() == "openrouter/claude-4"

    def test_confirm_subset_and_invalid_name(self, tmp_path, monkeypatch):
        models, svc = self._svc_add(tmp_path, monkeypatch)
        self._submit_no_models(svc, monkeypatch, fetched=["gpt-5.5", "claude-4"])
        # 填错模型名 → 留在挑选卡，不入库
        toast, ok, card = svc.handle_card_action("model_confirm_provider",
                                                 form_value={"models_pick": "gpt5.5"})
        assert not ok and "不在探测列表" in toast
        assert any(el["tag"] == "form" for el in card["elements"])
        assert "openrouter" not in svc.settings.providers
        # 填对部分 → 只加部分
        toast, ok, card = svc.handle_card_action("model_confirm_provider",
                                                 form_value={"models_pick": "claude-4"})
        assert ok
        assert svc._provider_models("openrouter") == ["claude-4"]

    def test_confirm_without_pending(self, tmp_path, monkeypatch):
        models, svc = self._svc_add(tmp_path, monkeypatch)
        toast, ok, card = svc.handle_card_action("model_confirm_provider",
                                                 form_value={"models_pick": ""})
        assert not ok and "过期" in toast

    # ---- 已存在 provider → 更新（补 key 启用）----

    def test_update_existing_provider_with_models(self, tmp_path, monkeypatch):
        """填已存在的 kimi-coding + 模型 → 更新 key 启用，写它自己的 env 变量。"""
        models, svc = self._svc_add(tmp_path, monkeypatch)
        toast, ok, card = svc.handle_card_action(
            "model_add_provider",
            form_value={"pname": "kimi-coding", "base_url": "https://api.kimi.com/coding",
                        "api_key": "sk-kimi-new", "models": "k3", "protocol": "anthropic"})
        assert ok and "已更新" in toast
        p = svc.settings.providers["kimi-coding"]
        assert p.enabled and p.api_key == "sk-kimi-new" and p.protocol == "anthropic"
        assert p.base_url == "https://api.kimi.com/coding"
        # key 写进它自己配置的 env 变量（K3 是 _settings 里的 api_key_env）
        env_text = (tmp_path / ".env").read_text()
        assert "K3=sk-kimi-new" in env_text and "KIMI_CODING_API_KEY" not in env_text
        # 模型入链尾；custom 里是 override 条目
        assert svc.settings.routing_heavy[-1].label() == "kimi-coding/k3"
        saved = models.load_custom(tmp_path / "custom.json")
        assert saved["kimi-coding"]["override"] is True

    def test_update_existing_via_probe_confirm(self, tmp_path, monkeypatch):
        """已存在 provider 模型留空 → 探测 → 确认 → 走更新而非新建。"""
        models, svc = self._svc_add(tmp_path, monkeypatch)
        monkeypatch.setattr(svc, "_probe_models", lambda *a: ["k3", "k3-256k"])
        toast, ok, card = svc.handle_card_action(
            "model_add_provider",
            form_value={"pname": "kimi-coding", "base_url": "https://api.kimi.com/coding",
                        "api_key": "sk-kimi-new", "models": "", "protocol": "anthropic"})
        assert ok and "探测成功" in toast
        toast, ok, card = svc.handle_card_action("model_confirm_provider",
                                                 form_value={"models_pick": ""})
        assert ok and "已更新" in toast
        p = svc.settings.providers["kimi-coding"]
        assert p.enabled and p.api_key == "sk-kimi-new"
        assert svc.settings.routing_heavy[-1].label() == "kimi-coding/k3-256k"

    def test_update_env_replaces_existing_line(self, tmp_path, monkeypatch):
        """.env 已有同名变量（含空值）→ 替换而不是追加重复行。"""
        models, svc = self._svc_add(tmp_path, monkeypatch)
        (tmp_path / ".env").write_text("K3=\nFREE_TOKENS_API_KEY=ft\n")
        msg, ok = svc.update_provider("kimi-coding", "https://api.kimi.com/coding",
                                      "sk-replaced", "k3", protocol="anthropic")
        assert ok
        env_text = (tmp_path / ".env").read_text()
        assert env_text.count("K3=") == 1 and "K3=sk-replaced" in env_text
        assert "FREE_TOKENS_API_KEY=ft" in env_text  # 其他行不动

    def test_merge_override_enables_yaml_provider(self, tmp_path, monkeypatch):
        """重启语义：custom 的 override 条目启用并覆盖 yaml 同名 provider。"""
        import json as _json
        from feishu_assistant import config as config_mod
        (tmp_path / "custom_providers.json").write_text(_json.dumps(
            {"kimi-coding": {"override": True, "base_url": "https://api.kimi.com/coding",
                             "protocol": "anthropic", "models": ["k3"]}}))
        monkeypatch.setenv("K3", "sk-merged")
        providers = dict(self._settings().providers)
        heavy = list(self._settings().routing_heavy)
        light = list(self._settings().routing_light)
        warnings: list[str] = []
        config_mod._merge_custom_providers(providers, heavy, light, warnings,
                                           path=tmp_path / "custom_providers.json")
        p = providers["kimi-coding"]
        assert p.enabled and p.protocol == "anthropic"
        assert p.base_url == "https://api.kimi.com/coding" and p.api_key == "sk-merged"
        assert heavy[-1].label() == "kimi-coding/k3"

    def test_custom_providers_merge_on_load(self, tmp_path, monkeypatch):
        """load_settings 合并 custom_providers.json：provider 出现 + 模型在链尾 + key 从 env 解析。"""
        import json as _json
        from feishu_assistant import config as config_mod
        custom_file = tmp_path / "custom_providers.json"
        custom_file.write_text(_json.dumps(
            {"openrouter": {"base_url": "https://openrouter.ai/api/v1",
                            "api_key_env": "OPENROUTER_API_KEY", "models": ["gpt-5.5"]}}))
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-y")
        providers = dict(self._settings().providers)
        heavy = list(self._settings().routing_heavy)
        light = list(self._settings().routing_light)
        warnings: list[str] = []
        config_mod._merge_custom_providers(providers, heavy, light, warnings, path=custom_file)
        assert providers["openrouter"].usable()
        assert heavy[-1].label() == "openrouter/gpt-5.5"
        assert light[-1].label() == "openrouter/gpt-5.5"
        # 再合并一次不重复
        config_mod._merge_custom_providers(providers, heavy, light, warnings, path=custom_file)
        assert [c.label() for c in heavy].count("openrouter/gpt-5.5") == 1


class TestModelCardActionViaBot:
    """bot 回调入口：model_switch 走 model_service，返回 toast + 回刷卡片。"""

    def test_bot_model_switch(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock
        from feishu_assistant.feishu.bot import FeishuBot
        from feishu_assistant.modules import models
        from feishu_assistant.modules.models import ModelService
        from feishu_assistant.config import Settings, ProviderConfig, ModelCandidate

        monkeypatch.setattr(models, "OVERRIDE_PATH", tmp_path / "ov.json")
        svc = ModelService(Settings(
            providers={"free-tokens": ProviderConfig("free-tokens", "https://x/v1", "K", True, "key1")},
            routing_heavy=[ModelCandidate("free-tokens", "kimi-k3")],
            routing_light=[ModelCandidate("free-tokens", "kimi-k3-highspeed")],
        ))
        monkeypatch.setattr(svc, "_provider_models", lambda name: ["kimi-k3", "gpt-5.5"])
        bot = FeishuBot.__new__(FeishuBot)
        bot.model_service = svc

        evt = MagicMock()
        evt.event.action.value = {"act": "model_switch", "chain": "heavy"}
        evt.event.action.option = "free-tokens/gpt-5.5"
        resp = bot._on_card_action(evt)
        assert resp.toast.type == "success"
        assert resp.card is not None and resp.card.type == "raw"
        assert svc.settings.routing_heavy[0].label() == "free-tokens/gpt-5.5"


class TestModelSwitchUndo:
    def _svc(self, tmp_path, monkeypatch):
        from feishu_assistant.modules import models
        monkeypatch.setattr(models, "OVERRIDE_PATH", tmp_path / "ov.json")
        monkeypatch.setattr(models, "UNDO_PATH", tmp_path / "undo.json")
        from feishu_assistant.modules.models import ModelService
        return models, ModelService(self._settings())

    def _settings(self):
        from feishu_assistant.config import Settings, ProviderConfig, ModelCandidate
        return Settings(
            providers={
                "free-tokens": ProviderConfig("free-tokens", "https://x/v1", "K", True, "key1"),
                "moonshot": ProviderConfig("moonshot", "https://y/v1", "K2", True, "key2"),
            },
            routing_heavy=[ModelCandidate("free-tokens", "kimi-k3"), ModelCandidate("moonshot", "kimi-k2-thinking")],
            routing_light=[ModelCandidate("free-tokens", "kimi-k3-highspeed")],
        )

    def test_switch_receipt_carries_undo_hint(self, tmp_path, monkeypatch):
        _, svc = self._svc(tmp_path, monkeypatch)
        msg = svc.switch("heavy", "moonshot/kimi-k2-thinking")
        assert "已切换" in msg and "撤销切换" in msg

    def test_undo_restores_default_when_no_prev_override(self, tmp_path, monkeypatch):
        models, svc = self._svc(tmp_path, monkeypatch)
        monkeypatch.setattr("feishu_assistant.config.load_settings", lambda: self._settings())
        svc.switch("heavy", "moonshot/kimi-k2-thinking")
        msg = svc.undo_last()
        assert "已撤销" in msg and "free-tokens/kimi-k3" in msg
        assert svc.settings.routing_heavy[0].label() == "free-tokens/kimi-k3"
        assert models.load_override(tmp_path / "ov.json")["active"] == "free-tokens/kimi-k3"
        assert svc.undo_last() == "没有可撤销的模型切换。"

    def test_undo_rolls_back_to_previous_override(self, tmp_path, monkeypatch):
        models, svc = self._svc(tmp_path, monkeypatch)
        menus = {"free-tokens": ["kimi-k3", "kimi-k3-highspeed", "gpt-5.5"],
                 "moonshot": ["kimi-k2-thinking"]}
        monkeypatch.setattr(svc, "_provider_models", lambda name: menus.get(name, []))
        svc.switch("heavy", "moonshot/kimi-k2-thinking")
        svc.switch("heavy", "free-tokens/gpt-5.5")
        msg = svc.undo_last()
        assert "已撤销" in msg and "moonshot/kimi-k2-thinking" in msg
        assert svc.settings.routing_heavy[0].label() == "moonshot/kimi-k2-thinking"
        assert models.load_override(tmp_path / "ov.json")["active"] == "moonshot/kimi-k2-thinking"

    def test_undo_without_history(self, tmp_path, monkeypatch):
        _, svc = self._svc(tmp_path, monkeypatch)
        assert svc.undo_last() == "没有可撤销的模型切换。"
