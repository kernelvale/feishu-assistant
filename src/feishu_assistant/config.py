"""配置加载与校验。

配置分两层：
- .env：密钥（飞书两个 + per-provider 的大模型 key），不进 git
- config.yaml：业务配置（providers、全局模型选择、推送时间、多维表格映射）

schema 说明：
- providers: 映射，每个 provider 有 base_url / api_key_env（.env 变量名）/ enabled
- routing: 为兼容旧配置保留 heavy / light；全局选择取首项，故障时不自动切换

设计原则：缺凭证不抛 traceback，而是收集缺失项与警告，由调用方打印中文指引。
向后兼容：.env 里的旧变量 KIMI_API_KEY 自动映射为 MOONSHOT_API_KEY。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

# src/feishu_assistant/config.py → parents[2] 即项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# 飞书密钥及其中文说明（用于缺失时的指引）
REQUIRED_ENV: dict[str, str] = {
    "FEISHU_APP_ID": "飞书自建应用 App ID（open.feishu.cn → 开发者后台 → 凭证与基础信息）",
    "FEISHU_APP_SECRET": "飞书自建应用 App Secret（同上页面）",
}
MODEL_KEY_HINT = (
    "大模型 API Key —— 请为当前选择的 provider 配上 key"
    "（默认主力是 free-tokens，对应 .env 变量 FREE_TOKENS_API_KEY）"
)

# 旧变量 → 新变量 的兼容映射（.env 层面）
LEGACY_ENV_MAP = {"KIMI_API_KEY": "MOONSHOT_API_KEY"}

# providers / routing 段缺失时的默认值（与 config.yaml 样例一致）
DEFAULT_PROVIDERS: dict[str, dict] = {
    "free-tokens": {
        "base_url": "https://free-tokens.msh.team/v1",
        "api_key_env": "FREE_TOKENS_API_KEY",
        "enabled": True,
    },
    "kimi-coding": {
        "base_url": "https://api.kimi.com/coding",
        "api_key_env": "KIMI_CODING_API_KEY",
        "enabled": False,
        "protocol": "anthropic",  # Claude Code 风格端点；base_url 不带 /v1，客户端后补
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "api_key_env": "MOONSHOT_API_KEY",
        "enabled": False,
    },
}
DEFAULT_ROUTING: dict[str, list[tuple[str, str]]] = {
    "heavy": [("free-tokens", "kimi-k3")],
    "light": [("free-tokens", "kimi-k3")],
}
class ConfigError(Exception):
    """配置文件本身存在问题（不是缺凭证，而是文件缺失/格式错误）。"""


@dataclass
class ProviderConfig:
    """一个大模型供给方。api_key 在加载时从 .env 解析填入。

    protocol："openai"（默认，/chat/completions）或 "anthropic"（/messages，
    Claude Code 风格端点，如 api.kimi.com/coding 的 Anthropic 协议面）。
    """

    name: str
    base_url: str
    api_key_env: str
    enabled: bool = True
    api_key: str = ""
    protocol: str = "openai"

    def usable(self) -> bool:
        """enabled 且 key 就位，才会参与调用。"""
        return self.enabled and bool(self.api_key)


@dataclass
class ModelCandidate:
    """routing 链上的一个候选：provider + model。"""

    provider: str
    model: str

    def label(self) -> str:
        return f"{self.provider}/{self.model}"


@dataclass
class Settings:
    """全部运行时配置。密钥允许为空字符串，由 missing_credentials() 报告。"""

    # 飞书密钥（来自 .env）
    feishu_app_id: str = ""
    feishu_app_secret: str = ""
    # 大模型：provider 映射 + 兼容旧格式的 heavy/light 存储（全局只认首项）
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    routing_heavy: list[ModelCandidate] = field(default_factory=list)
    routing_light: list[ModelCandidate] = field(default_factory=list)
    # 大模型调用参数
    kimi_timeout: int = 60
    kimi_max_retries: int = 2
    max_input_chars: int = 60_000  # 追问上下文预算；V3 精读使用全文证据索引，不按此值截尾
    # 会话系统上下文预算（session 段，键见 modules/session.py ContextBudget）
    session_budget: dict[str, int] = field(default_factory=dict)
    # 加载过程中收集的中文警告（key 缺失、旧变量兼容等）
    warnings: list[str] = field(default_factory=list)
    # 定时推送（P2+ 使用）
    push_daily_time: str = "08:30"
    push_evening_time: str = "22:00"
    push_chat_id: str = ""  # 定时推送目标会话（P4：日报/今日/周报），空则只响应命令触发
    push_interests: list[str] = field(default_factory=list)  # 显式兴趣声明：与论文库口味合并参与日报排序
    # 多维表格（init 命令自动写回 config.yaml 后生效）
    bitable_app_token: str = ""
    bitable_tables: dict[str, str] = field(default_factory=dict)

    def bitable_ready(self) -> bool:
        """多维表格是否已完成 init（app_token 与 7 张表 table_id 均已写回）。"""
        return bool(self.bitable_app_token) and all(self.bitable_tables.values())


def usable_candidates(settings: Settings, heavy: bool) -> list[ModelCandidate]:
    """返回全局选择且可用的唯一模型；heavy 仅保留调用参数兼容。"""
    chain = settings.routing_heavy
    if not chain:
        return []
    selected = chain[0]
    provider = settings.providers.get(selected.provider)
    return [selected] if provider is not None and provider.usable() else []


def load_yaml(config_path: Path = CONFIG_PATH) -> dict:
    """读取 config.yaml；文件缺失或格式错误时抛 ConfigError（中文信息）。"""
    if not config_path.exists():
        raise ConfigError(
            f"找不到配置文件 {config_path}\n"
            "请在项目根目录创建 config.yaml（仓库里有带注释的样例可直接用）。"
        )
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"config.yaml 格式错误，请检查 YAML 语法：\n{exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError("config.yaml 顶层必须是键值对（mapping）结构。")
    return data


def _read_env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _resolve_legacy_keys(warnings: list[str]) -> None:
    """旧 .env 变量映射到新变量（新变量为空时才生效，不覆盖）。"""
    for old, new in LEGACY_ENV_MAP.items():
        if _read_env(old) and not _read_env(new):
            os.environ[new] = _read_env(old)
            warnings.append(
                f"检测到旧变量 {old}，已按 {new} 使用；建议在 .env 中改名为 {new}。"
            )


def _parse_providers(cfg: dict, warnings: list[str]) -> dict[str, ProviderConfig]:
    """解析 providers 段；段缺失时用默认值。enabled 但 key 缺失的收集中文警告。"""
    raw = cfg.get("providers")
    if raw is None:
        raw = DEFAULT_PROVIDERS
        warnings.append("config.yaml 缺少 providers 段，已按默认供给方配置（free-tokens 主力）。")
    if not isinstance(raw, dict) or not raw:
        raise ConfigError("config.yaml 的 providers 段必须是非空映射。")

    providers: dict[str, ProviderConfig] = {}
    for name, pcfg in raw.items():
        name = str(name)
        pcfg = pcfg or {}
        if not isinstance(pcfg, dict):
            raise ConfigError(f"providers.{name} 必须是映射（base_url/api_key_env/enabled）。")
        default = DEFAULT_PROVIDERS.get(name, {})
        base_url = str(pcfg.get("base_url") or default.get("base_url") or "")
        api_key_env = str(pcfg.get("api_key_env") or default.get("api_key_env") or "")
        if not base_url or not api_key_env:
            raise ConfigError(f"providers.{name} 必须配置 base_url 和 api_key_env。")
        enabled = bool(pcfg.get("enabled", default.get("enabled", True)))
        api_key = _read_env(api_key_env)
        if enabled and not api_key:
            warnings.append(
                f"provider「{name}」已 enabled 但 .env 缺少 {api_key_env}；"
                "若当前选中它，调用会直接报错，不会自动换 provider。"
            )
        providers[name] = ProviderConfig(
            name=name, base_url=base_url, api_key_env=api_key_env,
            enabled=enabled, api_key=api_key,
            protocol=str(pcfg.get("protocol") or default.get("protocol") or "openai"),
        )
    return providers


def _parse_routing(cfg: dict, providers: dict[str, ProviderConfig]) -> tuple[list[ModelCandidate], list[ModelCandidate]]:
    """解析 routing 段；候选引用的 provider 必须已定义。"""
    raw = cfg.get("routing")
    if raw is None:
        raw = {k: [{"provider": p, "model": m} for p, m in v] for k, v in DEFAULT_ROUTING.items()}
    if not isinstance(raw, dict):
        raise ConfigError("config.yaml 的 routing 段必须是映射（heavy/light 各一个列表）。")

    chains: dict[str, list[ModelCandidate]] = {}
    for kind in ("heavy", "light"):
        entries = raw.get(kind)
        if not isinstance(entries, list) or not entries:
            raise ConfigError(f"routing.{kind} 必须是非空列表（元素为 provider + model）。")
        chain = []
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("provider") or not entry.get("model"):
                raise ConfigError(f"routing.{kind} 的每个候选都要写 provider 和 model：{entry!r}")
            provider = str(entry["provider"])
            if provider not in providers:
                raise ConfigError(
                    f"routing.{kind} 引用了未定义的 provider「{provider}」，"
                    f"请先在 providers 段定义（已有：{'、'.join(providers)}）。"
                )
            chain.append(ModelCandidate(provider=provider, model=str(entry["model"])))
        chains[kind] = chain
    return chains["heavy"], chains["light"]


def _merge_custom_providers(
    providers: dict[str, ProviderConfig],
    routing_heavy: list[ModelCandidate],
    routing_light: list[ModelCandidate],
    warnings: list[str],
    path: Path | None = None,
) -> None:
    """合并「模型卡片」在线添加的 provider（data/custom_providers.json）。

    卡片添加的不写 config.yaml（保住手维护注释），统一走这个 JSON：
    {"name": {"base_url": ..., "api_key_env": ..., "models": [...]}}。
    每个模型追加到 heavy/light 可选项（去重）；只有用户选择的首项会被调用。
    """
    import json

    path = path or (PROJECT_ROOT / "data" / "custom_providers.json")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(raw, dict):
        return
    for name, info in raw.items():
        name = str(name)
        if not isinstance(info, dict):
            continue
        if name in providers:
            # override 条目（卡片「更新」已有 provider）：启用并覆盖地址/协议，模型照补
            if not info.get("override"):
                continue
            p = providers[name]
            if info.get("base_url"):
                p.base_url = str(info["base_url"])
            if info.get("protocol"):
                p.protocol = str(info["protocol"])
            p.enabled = True
            p.api_key = _read_env(p.api_key_env)
            if not p.api_key:
                warnings.append(
                    f"自定义 provider「{name}」缺少 {p.api_key_env}；"
                    "若当前选中它，调用会直接报错，不会自动换 provider。"
                )
        else:
            base_url = str(info.get("base_url") or "")
            api_key_env = str(info.get("api_key_env") or "")
            if not base_url or not api_key_env:
                continue
            api_key = _read_env(api_key_env)
            providers[name] = ProviderConfig(
                name=name, base_url=base_url, api_key_env=api_key_env,
                enabled=True, api_key=api_key,
                protocol=str(info.get("protocol") or "openai"),
            )
            if not api_key:
                warnings.append(
                    f"自定义 provider「{name}」缺少 {api_key_env}；"
                    "若当前选中它，调用会直接报错，不会自动换 provider。"
                )
        for m in info.get("models") or []:
            cand = ModelCandidate(name, str(m))
            if cand not in routing_heavy:
                routing_heavy.append(cand)
            if cand not in routing_light:
                routing_light.append(cand)


def load_settings(
    env_path: Path = ENV_PATH,
    config_path: Path = CONFIG_PATH,
    custom_providers_path: Path | None = None,
) -> Settings:
    """加载 .env + config.yaml，返回 Settings。不校验密钥是否为空。

    custom_providers_path：卡片添加的 provider 清单，默认 data/custom_providers.json；
    测试传不存在的路径可隔绝真实文件。
    """
    load_dotenv(env_path)
    cfg = load_yaml(config_path)

    warnings: list[str] = []
    _resolve_legacy_keys(warnings)
    providers = _parse_providers(cfg, warnings)
    routing_heavy, routing_light = _parse_routing(cfg, providers)
    _merge_custom_providers(providers, routing_heavy, routing_light, warnings,
                            path=custom_providers_path)

    # 模型切换 override：固定用户持久化的选择；暂时不可用也不能静默换掉。
    try:
        from .modules.models import apply_override, load_override
        settings_tmp = Settings(providers=providers, routing_heavy=routing_heavy,
                                routing_light=routing_light)
        applied = apply_override(settings_tmp, load_override())
        if applied:
            warnings.append(f"模型 override 已应用：{'、'.join(applied)}")
    except Exception as exc:  # override 失败不阻断启动
        warnings.append(f"模型 override 应用失败（忽略）：{exc}")

    kimi = cfg.get("kimi") or {}
    push = cfg.get("push") or {}
    bitable = cfg.get("bitable") or {}
    session = cfg.get("session") or {}

    return Settings(
        feishu_app_id=_read_env("FEISHU_APP_ID"),
        feishu_app_secret=_read_env("FEISHU_APP_SECRET"),
        providers=providers,
        routing_heavy=routing_heavy,
        routing_light=routing_light,
        kimi_timeout=int(kimi.get("timeout", 60)),
        kimi_max_retries=int(kimi.get("max_retries", 2)),
        max_input_chars=int(kimi.get("max_input_chars", 60_000)),
        session_budget={str(k): int(v) for k, v in session.items()},
        warnings=warnings,
        push_daily_time=str(push.get("daily_time", "08:30")),
        push_evening_time=str(push.get("evening_time", "22:00")),
        push_chat_id=str(push.get("chat_id") or "").strip(),
        push_interests=[str(x).strip() for x in (push.get("interests") or []) if str(x).strip()],
        bitable_app_token=str(bitable.get("app_token") or "").strip(),
        bitable_tables={str(k): str(v or "").strip() for k, v in (bitable.get("tables") or {}).items()},
    )


def missing_credentials(settings: Settings) -> list[str]:
    """返回缺失项的中文说明列表；为空表示可以启动。

    大模型侧只要求当前全局选择可用。
    """
    missing = []
    if not settings.feishu_app_id:
        missing.append(f"FEISHU_APP_ID —— {REQUIRED_ENV['FEISHU_APP_ID']}")
    if not settings.feishu_app_secret:
        missing.append(f"FEISHU_APP_SECRET —— {REQUIRED_ENV['FEISHU_APP_SECRET']}")
    if not usable_candidates(settings, heavy=True) and not usable_candidates(settings, heavy=False):
        missing.append(MODEL_KEY_HINT)
    return missing


def format_missing_guide(missing: list[str]) -> str:
    """把缺失项组装成一段完整的中文指引。"""
    lines = [
        "❌ 缺少必需的密钥，无法继续。请在项目根目录创建 .env 文件并填入以下内容：",
        "",
    ]
    lines += [f"  {i}. {item}" for i, item in enumerate(missing, 1)]
    lines += [
        "",
        "可以直接复制模板：cp .env.example .env，然后逐项填写。",
        "详细准备步骤见 docs/p0-setup.md。",
    ]
    return "\n".join(lines)


def write_bitable_config(
    app_token: str,
    tables: dict[str, str],
    config_path: Path = CONFIG_PATH,
) -> None:
    """把 init 得到的 app_token 和各表 table_id 写回 config.yaml。

    采用按行正则替换而非 YAML 重新 dump，以保留文件里的中文注释。
    已有行按名替换；config.yaml 中还没有的表名（schema 新增，如精选库）
    以同样缩进追加到文件末尾（bitable.tables 是文件最后一个块，追加即落入该映射）。
    """
    text = config_path.read_text(encoding="utf-8")

    text, n = re.subn(
        r'(?m)^(\s*app_token:\s*)(?:"[^"\n]*"|[^#\n]*?)(\s*(?:#.*)?)$',
        lambda m: f'{m.group(1)}"{app_token}"{m.group(2)}',
        text,
        count=1,
    )
    if n == 0:
        raise ConfigError("config.yaml 中找不到 bitable.app_token 行，请对照样例检查格式。")

    appended: list[str] = []
    for name, table_id in tables.items():
        pattern = (
            rf'(?m)^(\s*{re.escape(name)}:\s*)'
            r'(?:"[^"\n]*"|[^#\n]*?)(\s*(?:#.*)?)$'
        )
        text, n = re.subn(
            pattern,
            lambda m, value=table_id: f'{m.group(1)}"{value}"{m.group(2)}',
            text,
            count=1,
        )
        if n == 0:
            appended.append(f'    {name}: "{table_id}"  # init 补建追加')

    if appended:
        if not text.endswith("\n"):
            text += "\n"
        text += "\n".join(appended) + "\n"

    config_path.write_text(text, encoding="utf-8")
