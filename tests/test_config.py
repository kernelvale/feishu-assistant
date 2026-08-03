"""config.py 测试：providers/routing 解析、旧 key 兼容、警告收集、写回。全部离线运行。"""

from __future__ import annotations

import pytest

from feishu_assistant.config import (
    ConfigError,
    ModelCandidate,
    ProviderConfig,
    Settings,
    format_missing_guide,
    load_settings,
    missing_credentials,
    usable_candidates,
    write_bitable_config,
)

ENV_KEYS = (
    "FEISHU_APP_ID", "FEISHU_APP_SECRET",
    "FREE_TOKENS_API_KEY", "KIMI_CODING_API_KEY", "MOONSHOT_API_KEY",
    "KIMI_API_KEY",  # 旧变量
)

SAMPLE_YAML = """\
# 注释：这一行必须被保留
providers:
  free-tokens:
    base_url: "https://free-tokens.msh.team/v1"
    api_key_env: "FREE_TOKENS_API_KEY"
    enabled: true
  kimi-coding:
    base_url: "https://api.kimi.com/coding/v1"
    api_key_env: "KIMI_CODING_API_KEY"
    enabled: false
  moonshot:
    base_url: "https://api.moonshot.cn/v1"
    api_key_env: "MOONSHOT_API_KEY"
    enabled: false
routing:
  heavy:
    - provider: "free-tokens"
      model: "kimi-k3"
    - provider: "moonshot"
      model: "kimi-k2-thinking"
  light:
    - provider: "free-tokens"
      model: "kimi-k3-highspeed"
    - provider: "moonshot"
      model: "kimi-k2-0905-preview"
kimi:
  timeout: 60
  max_retries: 2
push:
  daily_time: "08:30"
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


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """保证测试不受真实环境变量影响。"""
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def _load(tmp_path, yaml_text=SAMPLE_YAML, env_text=""):
    return load_settings(_write(tmp_path, ".env", env_text), _write(tmp_path, "config.yaml", yaml_text),
                         tmp_path / "no_custom.json")  # 隔绝真实 custom_providers.json


# ---- providers / routing 解析 ----

def test_load_full_new_schema(tmp_path, monkeypatch):
    monkeypatch.setenv("FEISHU_APP_ID", "cli_xxx")
    monkeypatch.setenv("FEISHU_APP_SECRET", "sec_yyy")
    monkeypatch.setenv("FREE_TOKENS_API_KEY", "sk-ft")
    monkeypatch.setattr("feishu_assistant.modules.models.load_override", lambda path=None: {})
    s = _load(tmp_path)

    assert s.feishu_app_id == "cli_xxx"
    assert set(s.providers) == {"free-tokens", "kimi-coding", "moonshot"}

    ft = s.providers["free-tokens"]
    assert ft.base_url == "https://free-tokens.msh.team/v1"
    assert ft.api_key_env == "FREE_TOKENS_API_KEY"
    assert ft.enabled and ft.api_key == "sk-ft" and ft.usable()

    kc = s.providers["kimi-coding"]
    assert not kc.enabled and not kc.usable()

    assert [(c.provider, c.model) for c in s.routing_heavy] == [
        ("free-tokens", "kimi-k3"), ("moonshot", "kimi-k2-thinking"),
    ]
    assert [(c.provider, c.model) for c in s.routing_light] == [
        ("free-tokens", "kimi-k3-highspeed"), ("moonshot", "kimi-k2-0905-preview"),
    ]
    assert s.kimi_timeout == 60 and s.kimi_max_retries == 2
    assert s.push_daily_time == "08:30"
    assert not s.bitable_ready()


def test_enabled_provider_missing_key_warns_and_skips(tmp_path):
    yaml_text = SAMPLE_YAML.replace("enabled: false", "enabled: true")  # 三个 provider 全开
    s = _load(tmp_path, yaml_text)  # .env 全空 → 全部缺 key
    assert not any(p.usable() for p in s.providers.values())
    joined = "\n".join(s.warnings)
    assert "free-tokens" in joined and "FREE_TOKENS_API_KEY" in joined
    assert "moonshot" in joined and "MOONSHOT_API_KEY" in joined
    # 当前选择不可用（即使后面有可用项也不能自动跳过去）
    assert usable_candidates(s, heavy=True) == []
    assert usable_candidates(s, heavy=False) == []


def test_unusable_selected_model_is_not_skipped(tmp_path, monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-ms")
    yaml_text = SAMPLE_YAML.replace(
        'api_key_env: "MOONSHOT_API_KEY"\n    enabled: false',
        'api_key_env: "MOONSHOT_API_KEY"\n    enabled: true',
    )
    s = _load(tmp_path, yaml_text)
    # 首项 free-tokens 缺 key；即使 moonshot 可用也不能静默切过去。
    assert usable_candidates(s, heavy=True) == []


def test_disabled_provider_not_usable_even_with_key(tmp_path, monkeypatch):
    monkeypatch.setenv("KIMI_CODING_API_KEY", "sk-kc")
    s = _load(tmp_path)
    assert s.providers["kimi-coding"].api_key == "sk-kc"
    assert not s.providers["kimi-coding"].usable()  # enabled: false


# ---- 旧 KIMI_API_KEY 兼容 ----

def test_legacy_kimi_api_key_maps_to_moonshot(tmp_path, monkeypatch):
    monkeypatch.setenv("KIMI_API_KEY", "sk-legacy")
    s = _load(tmp_path)
    assert s.providers["moonshot"].api_key == "sk-legacy"
    assert any("KIMI_API_KEY" in w and "MOONSHOT_API_KEY" in w for w in s.warnings)


def test_moonshot_api_key_takes_precedence_over_legacy(tmp_path, monkeypatch):
    monkeypatch.setenv("KIMI_API_KEY", "sk-legacy")
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-new")
    s = _load(tmp_path)
    assert s.providers["moonshot"].api_key == "sk-new"
    assert not any("KIMI_API_KEY" in w for w in s.warnings)


# ---- 默认值与校验 ----

def test_missing_providers_and_routing_fall_back_to_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr("feishu_assistant.modules.models.load_override", lambda path=None: {})
    s = _load(tmp_path, yaml_text="push:\n  daily_time: '07:00'\n")
    assert "free-tokens" in s.providers
    assert s.routing_heavy[0].label() == "free-tokens/kimi-k3"
    assert any("providers" in w for w in s.warnings)


def test_routing_referencing_unknown_provider_raises(tmp_path):
    bad = SAMPLE_YAML.replace('provider: "moonshot"\n      model: "kimi-k2-thinking"',
                              'provider: "not-exist"\n      model: "kimi-k2-thinking"')
    with pytest.raises(ConfigError, match="not-exist"):
        _load(tmp_path, bad)


def test_routing_entry_missing_model_raises(tmp_path):
    bad = SAMPLE_YAML.replace('model: "kimi-k3"', '')
    with pytest.raises(ConfigError, match="provider 和 model"):
        _load(tmp_path, bad)


# ---- missing_credentials ----

def test_missing_credentials_reports_feishu_and_model_key(tmp_path):
    s = _load(tmp_path)
    missing = missing_credentials(s)
    joined = "\n".join(missing)
    assert "FEISHU_APP_ID" in joined and "FEISHU_APP_SECRET" in joined
    assert "大模型 API Key" in joined and "FREE_TOKENS_API_KEY" in joined
    guide = format_missing_guide(missing)
    assert ".env" in guide and "docs/p0-setup.md" in guide


def test_missing_credentials_ok_with_feishu_and_one_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("FEISHU_APP_ID", "a")
    monkeypatch.setenv("FEISHU_APP_SECRET", "b")
    monkeypatch.setenv("FREE_TOKENS_API_KEY", "sk-ft")
    monkeypatch.setattr("feishu_assistant.modules.models.load_override", lambda path=None: {})
    s = _load(tmp_path)
    assert missing_credentials(s) == []


# ---- 其它（文件错误 / bitable 写回）----

def test_config_file_missing_raises_chinese_error(tmp_path):
    with pytest.raises(ConfigError, match="找不到配置文件"):
        load_settings(tmp_path / ".env", tmp_path / "config.yaml")


def test_config_yaml_bad_syntax_raises_chinese_error(tmp_path):
    cfg = _write(tmp_path, "config.yaml", "providers:\n  free-tokens: [unclosed\n")
    with pytest.raises(ConfigError, match="格式错误"):
        load_settings(_write(tmp_path, ".env", ""), cfg)


def test_write_bitable_config_roundtrip_preserves_comments(tmp_path, monkeypatch):
    cfg = _write(tmp_path, "config.yaml", SAMPLE_YAML)
    tables = {
        "论文库": "tblA", "收藏库": "tblB", "灵感库": "tblC", "求职看板": "tblD",
        "面经库": "tblE", "简历库": "tblF", "备忘库": "tblG",
    }
    write_bitable_config("bascnTOKEN", tables, cfg)

    text = cfg.read_text(encoding="utf-8")
    assert "# 注释：这一行必须被保留" in text  # 注释未丢
    assert 'app_token: "bascnTOKEN"' in text
    assert '论文库: "tblA"' in text
    assert '备忘库: "tblG"' in text

    monkeypatch.setenv("FEISHU_APP_ID", "a")
    monkeypatch.setenv("FEISHU_APP_SECRET", "b")
    s = load_settings(_write(tmp_path, ".env", ""), cfg)
    assert s.bitable_app_token == "bascnTOKEN"
    assert s.bitable_tables == tables
    assert s.bitable_ready()


def test_write_bitable_config_is_idempotent_with_inline_comments(tmp_path):
    cfg = _write(
        tmp_path,
        "config.yaml",
        "bitable:\n"
        "  app_token: \"old\"  # token 注释\n"
        "  tables:\n"
        "    精选库: \"old_table\"  # init 补建追加\n",
    )
    write_bitable_config("new", {"精选库": "tbl_new"}, cfg)
    once = cfg.read_text(encoding="utf-8")
    write_bitable_config("new", {"精选库": "tbl_new"}, cfg)
    twice = cfg.read_text(encoding="utf-8")
    assert twice == once
    assert twice.count("精选库:") == 1
    assert "# token 注释" in twice and "# init 补建追加" in twice


def test_write_bitable_config_missing_section_raises(tmp_path):
    cfg = _write(tmp_path, "config.yaml", "kimi:\n  timeout: 60\n")
    with pytest.raises(ConfigError, match="app_token"):
        write_bitable_config("tok", {}, cfg)


# ---- 数据类小行为 ----

def test_candidate_label_and_provider_usable():
    assert ModelCandidate("free-tokens", "kimi-k3").label() == "free-tokens/kimi-k3"
    p = ProviderConfig("x", "https://x/v1", "X_KEY", enabled=True, api_key="k")
    assert p.usable()
    p.api_key = ""
    assert not p.usable()


# ---- 模型路由 override ----

def test_apply_override_syncs_global_selection_to_both_modes():
    """一次全局选择同步到 heavy/light；链外模型也不能静默丢弃。"""
    from feishu_assistant.modules.models import apply_override

    s = Settings(
        providers={
            "free-tokens": ProviderConfig("free-tokens", "https://x/v1", "K", enabled=True, api_key="k"),
        },
        routing_heavy=[ModelCandidate("free-tokens", "kimi-k3")],
        routing_light=[ModelCandidate("free-tokens", "a"), ModelCandidate("free-tokens", "b")],
    )
    applied = apply_override(s, {"active": "free-tokens/b"})
    assert s.routing_heavy[0].label() == "free-tokens/b"
    assert s.routing_light[0].label() == "free-tokens/b"
    assert applied == ["active=free-tokens/b"]


def test_apply_override_preserves_unusable_user_selection():
    from feishu_assistant.modules.models import apply_override

    s = Settings(
        providers={"dead": ProviderConfig("dead", "https://x/v1", "K", enabled=True, api_key="")},
        routing_heavy=[ModelCandidate("dead", "a")],
    )
    assert apply_override(s, {"heavy": "dead/b"}) == ["active=dead/b（由旧 heavy/light 选择迁移）"]
    assert [c.label() for c in s.routing_heavy] == ["dead/b", "dead/a"]
    assert s.routing_light[0].label() == "dead/b"
    assert usable_candidates(s, heavy=True) == []
