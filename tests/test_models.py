"""models.py 测试：pending 暂存生命周期（放弃/过期即清）、派生 .env 变量撞名拒绝。"""

from __future__ import annotations

import json
import os
import time

import pytest

from feishu_assistant.config import ModelCandidate, ProviderConfig, Settings
from feishu_assistant.modules import models
from feishu_assistant.modules.models import ModelService


def make_settings() -> Settings:
    return Settings(
        providers={
            "free-tokens": ProviderConfig(
                "free-tokens", "https://x/v1", "FREE_TOKENS_API_KEY", True, "key1"),
            "moonshot": ProviderConfig(
                "moonshot", "https://y/v1", "MOONSHOT_API_KEY", True, "key2"),
        },
        routing_heavy=[ModelCandidate("free-tokens", "kimi-k3")],
        routing_light=[ModelCandidate("free-tokens", "kimi-k3")],
    )


@pytest.fixture
def svc(tmp_path, monkeypatch):
    """custom/env/pending 路径全部打到 tmp，避免碰真实文件。"""
    monkeypatch.setattr(models, "CUSTOM_PATH", tmp_path / "custom.json")
    monkeypatch.setattr(models, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr(models, "PENDING_PATH", tmp_path / "pending.json")
    return ModelService(make_settings())


def submit_probe(svc, monkeypatch, pname="openrouter", fetched=("gpt-5.5",)):
    """模型留空提交表单 → 探测成功 → 暂存 pending 并展示挑选卡。"""
    monkeypatch.setattr(svc, "_probe_models", lambda *a: list(fetched))
    return svc.handle_card_action(
        "model_add_provider",
        form_value={"pname": pname, "base_url": "https://openrouter.ai/api/v1",
                    "api_key": "sk-or-x", "models": "", "protocol": ""})


# ---- pending 暂存生命周期：探测 → 落盘 → 确认/放弃/过期都要清 ----

def test_pending_cleared_on_abandon(tmp_path, svc, monkeypatch):
    """挑选卡点「返回模型面板」= 放弃添加：含 key 的暂存必须删掉。"""
    toast, ok, _ = submit_probe(svc, monkeypatch)
    assert ok and (tmp_path / "pending.json").exists()
    toast, ok, card = svc.handle_card_action("model_panel")
    assert ok and not (tmp_path / "pending.json").exists()


def test_pending_expired_loads_empty_and_deletes(tmp_path, svc):
    """超过 TTL 的暂存：读取即视为过期，返回空并删文件。"""
    models.save_pending({"pname": "openrouter", "api_key": "sk-or-x"})
    path = tmp_path / "pending.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["ts"] = time.time() - (models.PENDING_TTL_SECONDS + 60)  # 手动老化
    path.write_text(json.dumps(data), encoding="utf-8")
    assert models.load_pending() == {}
    assert not path.exists()


def test_pending_legacy_without_ts_treated_as_expired(tmp_path, svc):
    """旧版残留（无时间戳、明文 key）：按过期处理，首次读取即清理。"""
    path = tmp_path / "pending.json"
    path.write_text(json.dumps({"pname": "old", "api_key": "sk-leak"}), encoding="utf-8")
    assert models.load_pending() == {}
    assert not path.exists()


def test_pending_fresh_loads_ok(tmp_path, svc):
    models.save_pending({"pname": "openrouter", "api_key": "sk-or-x", "fetched": ["m1"]})
    pending = models.load_pending()
    assert pending["api_key"] == "sk-or-x" and pending["fetched"] == ["m1"]


def test_confirm_after_expiry_rejected(tmp_path, svc, monkeypatch):
    """暂存过期后再点确认：提示已过期，不入库。"""
    submit_probe(svc, monkeypatch)
    path = tmp_path / "pending.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["ts"] = time.time() - (models.PENDING_TTL_SECONDS + 60)
    path.write_text(json.dumps(data), encoding="utf-8")
    toast, ok, card = svc.handle_card_action("model_confirm_provider",
                                             form_value={"models_pick": ""})
    assert not ok and "过期" in toast
    assert "openrouter" not in svc.settings.providers
    assert not path.exists()


def test_direct_add_success_clears_stale_pending(tmp_path, svc, monkeypatch):
    """探测暂存还在时直接按表单（填了模型）添加成功：残留暂存一并清掉。"""
    models.save_pending({"pname": "stale", "api_key": "sk-stale", "fetched": ["m"]})
    toast, ok, card = svc.handle_card_action(
        "model_add_provider",
        form_value={"pname": "openrouter", "base_url": "https://openrouter.ai/api/v1",
                    "api_key": "sk-or-x", "models": "gpt-5.5", "protocol": ""})
    assert ok and not (tmp_path / "pending.json").exists()


# ---- 派生 .env 变量撞名：拒绝而不是覆盖 ----

def test_add_provider_rejects_legacy_kimi_api_key(tmp_path, svc, monkeypatch):
    """引爆链：.env 有旧版 KIMI_API_KEY，加名为 kimi 的 provider 不得覆盖它。"""
    monkeypatch.setenv("KIMI_API_KEY", "sk-legacy")
    (tmp_path / ".env").write_text("KIMI_API_KEY=sk-legacy\n", encoding="utf-8")
    msg, ok = svc.add_provider("kimi", "https://new.example/v1", "sk-new", "m1")
    assert not ok and "KIMI_API_KEY" in msg
    # 全线不动：.env、custom、进程内 providers、os.environ
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "KIMI_API_KEY=sk-legacy\n"
    assert not (tmp_path / "custom.json").exists()
    assert "kimi" not in svc.settings.providers
    assert os.environ["KIMI_API_KEY"] == "sk-legacy"


def test_add_provider_rejects_other_providers_env(tmp_path, svc):
    """派生名撞上别的 provider 已配置的 api_key_env：拒绝。"""
    svc.settings.providers["foo"] = ProviderConfig(
        "foo", "https://foo/v1", "BAR_API_KEY", True, "key-foo")
    msg, ok = svc.add_provider("bar", "https://bar.example/v1", "sk-bar", "m1")
    assert not ok and "BAR_API_KEY" in msg and "foo" in msg
    assert "bar" not in svc.settings.providers


def test_add_provider_rejects_nonempty_env_orphan(tmp_path, svc):
    """.env 里已有非空同名变量（无 provider 引用）：也是别人的 key，拒绝。"""
    (tmp_path / ".env").write_text("ACME_API_KEY=sk-orphan\n", encoding="utf-8")
    msg, ok = svc.add_provider("acme", "https://acme.example/v1", "sk-new", "m1")
    assert not ok and "ACME_API_KEY" in msg
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "ACME_API_KEY=sk-orphan\n"


def test_add_provider_allows_empty_env_line(tmp_path, svc):
    """.env 里同名变量是空值占位：覆盖无害，允许添加并替换该行。"""
    (tmp_path / ".env").write_text("ACME_API_KEY=\n", encoding="utf-8")
    msg, ok = svc.add_provider("acme", "https://acme.example/v1", "sk-new", "m1")
    assert ok
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert env_text.count("ACME_API_KEY=") == 1 and "ACME_API_KEY=sk-new" in env_text


def test_form_submit_rejects_collision_early(tmp_path, svc, monkeypatch):
    """表单提交（含探测路径）在校验阶段就拒绝撞名：不落 pending、不探测。"""
    (tmp_path / ".env").write_text("KIMI_API_KEY=sk-legacy\n", encoding="utf-8")
    probed = {"called": False}

    def fake_probe(*a):
        probed["called"] = True
        return ["m1"]

    monkeypatch.setattr(svc, "_probe_models", fake_probe)
    toast, ok, card = svc.handle_card_action(
        "model_add_provider",
        form_value={"pname": "kimi", "base_url": "https://new.example/v1",
                    "api_key": "sk-new", "models": "", "protocol": ""})
    assert not ok and "KIMI_API_KEY" in toast
    assert card is None  # 校验失败不刷卡：已填内容保留
    assert not probed["called"]
    assert not (tmp_path / "pending.json").exists()
