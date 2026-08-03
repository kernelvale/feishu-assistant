"""模型切换服务：查看并切换用户当前唯一选中的全局模型。

- 「模型」：发交互卡片（build_card）——下拉选 provider/模型 即切换，附恢复默认按钮；
  无发送通道时退化为 status_text() 文本
- 「模型 provider/model」：文本命令切换全局选择并持久化（data/model_override.json）
- 卡片回调（handle_card_action）：model_switch / model_reset，返回 toast + 刷新卡片
- 卡片在线添加 provider：探测 /models 后凭证暂存 pending_provider.json（含 key），
  确认成功、放弃（返回面板）或超过 TTL 都会清除；派生的 .env 变量名撞名直接拒绝

覆盖语义：config.yaml 的 routing 首项是默认选择；override 把用户指定模型固定到首项。
运行时只调用全局首项，后续元素仅作为下拉选项存储，不参与自动故障转移。进程内即时生效，
重启经 config 加载时再次应用 override。用户曾选中的 provider 即使暂时不可用也不会被
静默跳过，调用时会明确报错并要求用户手动处理。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from ..config import LEGACY_ENV_MAP, PROJECT_ROOT, ModelCandidate, Settings

logger = logging.getLogger(__name__)

OVERRIDE_PATH = PROJECT_ROOT / "data" / "model_override.json"
UNDO_PATH = PROJECT_ROOT / "data" / "model_switch_undo.json"  # 最近一次切换（撤销用）
CUSTOM_PATH = PROJECT_ROOT / "data" / "custom_providers.json"  # 卡片在线添加的 provider
PENDING_PATH = PROJECT_ROOT / "data" / "pending_provider.json"  # 探测后待确认的 provider（含 key，不进卡片；放弃/过期即清）
ENV_FILE = PROJECT_ROOT / ".env"
PENDING_TTL_SECONDS = 600  # 暂存有效期：含明文 key，超时即删，不长期滞留

# provider 静态模型菜单（free-tokens 支持 /v1/models 实时拉取，其余静态）
STATIC_MODELS: dict[str, list[str]] = {
    "kimi-coding": ["kimi-for-coding", "kimi-for-coding-highspeed", "k3", "k3-256k"],
    "moonshot": ["kimi-k2.5", "kimi-k2-thinking", "kimi-k2-thinking-turbo", "kimi-k2-0905-preview"],
}


def load_custom(path: Path | None = None) -> dict[str, dict]:
    """卡片添加的自定义 provider：{"name": {"base_url","api_key_env","models":[...]}}。"""
    path = path or CUSTOM_PATH
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): v for k, v in data.items() if isinstance(v, dict)}
    except (OSError, json.JSONDecodeError):
        return {}


def save_custom(custom: dict[str, dict], path: Path | None = None) -> None:
    path = path or CUSTOM_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(custom, ensure_ascii=False, indent=1), encoding="utf-8")


def load_pending(path: Path | None = None) -> dict:
    """读取暂存的待确认 provider；超过 TTL 或缺时间戳（旧版残留）视为过期：删除并返回空。"""
    path = path or PENDING_PATH
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    try:
        ts = float(data.get("ts") or 0)
    except (TypeError, ValueError):
        ts = 0.0
    if time.time() - ts > PENDING_TTL_SECONDS:
        clear_pending(path)  # 过期即清：含 key 的暂存不滞留
        return {}
    return data


def save_pending(pending: dict, path: Path | None = None) -> None:
    """暂存待确认 provider（含 api_key，只落服务端）；打时间戳供 load 按 TTL 过期。"""
    path = path or PENDING_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**pending, "ts": time.time()}, ensure_ascii=False, indent=1),
                    encoding="utf-8")


def clear_pending(path: Path | None = None) -> None:
    (path or PENDING_PATH).unlink(missing_ok=True)


def _derive_env_name(pname: str) -> str:
    """provider 名 → .env 变量名（kimi-coding → KIMI_CODING_API_KEY）。"""
    return pname.upper().replace("-", "_") + "_API_KEY"


def load_override(path: Path | None = None) -> dict[str, str]:
    path = path or OVERRIDE_PATH
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {k: str(v) for k, v in data.items() if k in ("active", "heavy", "light") and v}
    except (OSError, json.JSONDecodeError):
        return {}


def save_override(overrides: dict[str, str], path: Path | None = None) -> None:
    path = path or OVERRIDE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(overrides, ensure_ascii=False, indent=1), encoding="utf-8")


def load_undo(path: Path | None = None) -> dict[str, Any]:
    """最近一次全局模型切换记录，无则空 dict。"""
    path = path or UNDO_PATH
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_undo(entry: dict[str, Any], path: Path | None = None) -> None:
    path = path or UNDO_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")


def clear_undo(path: Path | None = None) -> None:
    (path or UNDO_PATH).unlink(missing_ok=True)


def apply_override(settings: Settings, overrides: dict[str, str] | None = None) -> list[str]:
    """把一次全局选择固定到 heavy/light 首项；兼容旧文件时优先采用 heavy。"""
    values = overrides or {}
    want = values.get("active") or values.get("heavy") or values.get("light") or ""
    if not want or "/" not in want:
        return []
    provider, model = want.split("/", 1)
    for chain in (settings.routing_heavy, settings.routing_light):
        for i, cand in enumerate(chain):
            if cand.provider == provider and cand.model == model:
                if i > 0:
                    chain.insert(0, chain.pop(i))
                break
        else:  # provider 暂不可用/已移除时也保留选择，调用层负责明确通知用户。
            chain.insert(0, ModelCandidate(provider=provider, model=model))
    migrated = "（由旧 heavy/light 选择迁移）" if "active" not in values else ""
    return [f"active={want}{migrated}"]


class ModelService:
    """模型路由的查看与切换。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _provider_models(self, name: str) -> list[str]:
        """provider 可用模型：free-tokens 实时拉 /v1/models；卡片添加的读 custom 清单；其余静态。"""
        p = self.settings.providers.get(name)
        if p is None:
            return []
        if name == "free-tokens" and p.usable():
            try:
                import httpx
                r = httpx.get(f"{p.base_url}/models",
                              headers={"Authorization": f"Bearer {p.api_key}"}, timeout=10)
                return [m["id"] for m in r.json().get("data", [])]
            except Exception as exc:
                logger.warning("拉取 free-tokens 模型列表失败（退静态）：%s", exc)
                return ["kimi-k3", "kimi-k3-highspeed"]
        custom = load_custom()
        if name in custom:
            return [str(m) for m in (custom[name].get("models") or [])]
        return STATIC_MODELS.get(name, [])

    def status_text(self) -> str:
        lines = ["🎛 当前全局模型（失败时不会自动切换）："]
        selected = self.settings.routing_heavy[0] if self.settings.routing_heavy else None
        provider = self.settings.providers.get(selected.provider) if selected else None
        mark = "✅" if provider is not None and provider.usable() else "❌"
        label = selected.label() if selected else "（未选择）"
        lines.append(f"  {mark} {label}")
        lines.append("")
        lines.append("provider 与可用模型：")
        for name, p in self.settings.providers.items():
            if not p.enabled:
                state = "⏸ disabled"
            elif not p.api_key:
                state = "⏸ enabled 但缺 key"
            else:
                state = "✅ 可用"
            models = self._provider_models(name)
            models_text = "、".join(models[:8]) + ("…" if len(models) > 8 else "")
            lines.append(f"• {name} {state}：{models_text or '（无模型菜单）'}")
        lines.append("")
        lines.append("切换：发「模型 free-tokens/gpt-5.5」")
        return "\n".join(lines)

    def switch(self, chain_name: str, target: str) -> str:
        """切换全局唯一生效模型；chain_name 仅兼容旧调用，统一按 all 处理。"""
        if chain_name not in ("all", "heavy", "light"):
            return "用法：模型 provider/model"
        if "/" not in target:
            return "格式：provider/model，如「模型 free-tokens/kimi-k3」"
        provider, model = target.split("/", 1)
        p = self.settings.providers.get(provider)
        if p is None:
            known = "、".join(self.settings.providers)
            return f"没有 provider「{provider}」（现有：{known}）"
        if not p.enabled:
            return f"provider「{provider}」是 disabled 状态（在 config.yaml 里 enabled: true 并配好 key 后才能切换）"
        if not p.api_key:
            return f"provider「{provider}」还没配 key（填进 .env 对应变量后重启或告诉我）"
        menu = self._provider_models(provider)
        if menu and model not in menu:
            hint = "、".join(menu[:8])
            return f"「{provider}」没有模型「{model}」，可用：{hint}"

        overrides = load_override()
        prev = (
            overrides.get("active")
            or (self.settings.routing_heavy[0].label() if self.settings.routing_heavy else "")
        )
        if prev != target:
            save_undo({"prev": prev})
        save_override({"active": target})
        # 进程内即时生效：heavy/light 始终同步为同一个用户选择。
        for chain in (self.settings.routing_heavy, self.settings.routing_light):
            for i, cand in enumerate(chain):
                if cand.provider == provider and cand.model == model:
                    chain.insert(0, chain.pop(i))
                    break
            else:
                chain.insert(0, ModelCandidate(provider, model))
        logger.info("全局模型切换：%s（已持久化）", target)
        hint = "回复「撤销切换」可回滚。" if prev != target else ""
        return (f"✅ 已切换全局模型为 {target}（即时生效，重启保持；失败不会自动切换）。"
                f"{hint}发「模型」可查看当前选择。")

    def undo_last(self) -> str:
        """撤销最近一次模型切换，恢复上一个全局选择。"""
        entry = load_undo()
        prev = str(entry.get("prev", ""))
        if not prev:
            return "没有可撤销的模型切换。"
        msg = self.switch("all", prev)
        if not msg.startswith("✅"):
            return msg
        clear_undo()
        logger.info("全局模型切换已撤销：%s", prev)
        return f"↩️ 已撤销：全局模型回滚为 {prev}。"

    def reset(self) -> str:
        """恢复 config.yaml 默认路由：清 override 文件 + 进程内链还原（原位替换，引用方即时生效）。"""
        from ..config import load_settings

        try:
            OVERRIDE_PATH.unlink(missing_ok=True)
            clear_undo()
        except OSError as exc:
            return f"清除 override 文件失败：{exc}"
        fresh = load_settings()  # override 已删 → 拿到的是默认链
        self.settings.routing_heavy[:] = fresh.routing_heavy
        self.settings.routing_light[:] = fresh.routing_light
        logger.info("模型路由已恢复默认")
        return "✅ 已恢复 config.yaml 默认路由（即时生效，重启保持）。"

    def add_provider(self, pname: str, base_url: str, api_key: str,
                     models_csv: str, protocol: str = "openai") -> tuple[str, bool]:
        """卡片表单添加 provider：校验 → 写 custom_providers.json + .env → 进程内热生效。

        模型加入 heavy/light 可选项（不会自动调用，下拉/文本命令明确选择后才生效）。
        protocol：openai（/chat/completions）或 anthropic（/messages）。
        派生的 .env 变量名撞名（别的 provider / 旧版兼容变量 / .env 已有非空同名）时拒绝，绝不覆盖。
        """
        pname = pname.strip().lower()
        base_url = base_url.strip().rstrip("/")
        api_key = api_key.strip()
        protocol = protocol.strip().lower() or "openai"
        models = [m.strip() for m in models_csv.replace("，", ",").split(",") if m.strip()]

        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", pname):
            return "名称只能用小写字母/数字/中划线（如 openrouter）", False
        if pname in self.settings.providers or pname in load_custom():
            return f"provider「{pname}」已存在", False
        if not base_url.startswith(("http://", "https://")):
            return "base_url 必须以 http(s):// 开头", False
        if not api_key:
            return "API key 不能为空", False
        if not models:
            return "至少填一个模型（逗号分隔）", False
        if protocol not in ("openai", "anthropic"):
            return "协议只支持 openai 或 anthropic（留空默认 openai）", False

        env_name = _derive_env_name(pname)
        conflict = self._env_name_conflict(env_name)
        if conflict:
            return (f"变量名 {env_name} 与{conflict}撞名，覆盖会串 key；"
                    "请换个 provider 名称"), False
        custom = load_custom()
        custom[pname] = {"base_url": base_url, "api_key_env": env_name,
                         "models": models, "protocol": protocol}
        save_custom(custom)
        self._upsert_env(env_name, api_key)

        # 进程内热生效
        from ..config import ProviderConfig
        self.settings.providers[pname] = ProviderConfig(
            name=pname, base_url=base_url, api_key_env=env_name, enabled=True,
            api_key=api_key, protocol=protocol)
        self._append_candidates(pname, models)
        logger.info("卡片添加 provider：%s（%d 个模型加入可选项）", pname, len(models))
        return (f"✅ 已添加 provider「{pname}」：{len(models)} 个模型加入 heavy/light 可选项，"
                "下拉明确选择后才会调用（已持久化，重启保持）"), True

    def update_provider(self, pname: str, base_url: str, api_key: str,
                        models_csv: str, protocol: str = "openai") -> tuple[str, bool]:
        """已存在的 provider（config.yaml 或卡片加过）：补/换 key 并启用，可改地址/协议/补模型。

        key 写进该 provider 自己配置的 api_key_env（不是新生成变量名）；
        持久化用 custom_providers.json 的 override 条目（config.yaml 不动）。
        """
        p = self.settings.providers.get(pname)
        if p is None:
            return f"provider「{pname}」不存在，走添加流程", False
        err = self._validate_provider_basics(pname, base_url, api_key, protocol,
                                             allow_existing=True)
        if err:
            return err, False
        models = [m.strip() for m in models_csv.replace("，", ",").split(",") if m.strip()]

        self._upsert_env(p.api_key_env, api_key)
        # 持久化 override（重启合并时覆盖 yaml 同名 provider 并启用）
        custom = load_custom()
        custom[pname] = {"base_url": base_url, "api_key_env": p.api_key_env,
                         "models": models, "protocol": protocol, "override": True}
        save_custom(custom)
        # 进程内热生效
        p.api_key = api_key
        p.enabled = True
        p.base_url = base_url
        p.protocol = protocol
        self._append_candidates(pname, models)
        logger.info("卡片更新 provider：%s（启用+补 %d 个模型）", pname, len(models))
        extra = f"，补 {len(models)} 个模型入链尾" if models else ""
        return (f"✅ 已更新 provider「{pname}」：key 已写入 .env 的 {p.api_key_env} 并启用"
                f"{extra}（已持久化，重启保持）"), True

    # ---- 内部小工具 ----

    def _env_name_conflict(self, env_name: str) -> str | None:
        """派生变量名的撞名检测；返回占用来源描述或 None（可安全写入）。

        三类占用：已有 provider（含 custom 清单）的 api_key_env；config 旧版兼容变量
        （如 KIMI_API_KEY，加载时映射成 MOONSHOT_API_KEY，写了会被别的 provider 吸走）；
        .env 里已有非空值的同名变量。命中即拒绝，绝不覆盖别人的 key。
        """
        for name, p in self.settings.providers.items():
            if p.api_key_env == env_name:
                return f"provider「{name}」的 api_key_env"
        for name, info in load_custom().items():
            if str(info.get("api_key_env") or "") == env_name:
                return f"自定义 provider「{name}」的 api_key_env"
        mapped = LEGACY_ENV_MAP.get(env_name)
        if mapped:
            return f"旧版兼容变量（加载时映射为 {mapped}）"
        try:
            lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
        except OSError:
            lines = []
        for line in lines:
            if line.startswith(f"{env_name}=") and line.split("=", 1)[1].strip():
                return ".env 中已有的非空变量"
        return None

    @staticmethod
    def _upsert_env(env_name: str, api_key: str) -> None:
        """.env 写入/更新变量；进程内同步 os.environ。失败仅告警（进程内仍生效）。"""
        try:
            lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
            for i, line in enumerate(lines):
                if line.startswith(f"{env_name}="):
                    lines[i] = f"{env_name}={api_key}"
                    break
            else:
                lines.append(f"{env_name}={api_key}")
            ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError as exc:
            logger.warning(".env 写入失败（仅进程内生效，重启丢）：%s", exc)
        os.environ[env_name] = api_key

    def _append_candidates(self, pname: str, models: list[str]) -> None:
        """模型追加到 heavy/light 可选项存储（去重）；运行时仍只调用首项。"""
        for m in models:
            cand = ModelCandidate(pname, m)
            if cand not in self.settings.routing_heavy:
                self.settings.routing_heavy.append(cand)
            if cand not in self.settings.routing_light:
                self.settings.routing_light.append(cand)

    # ---- 交互卡片（「模型」命令发面板，下拉即切换）----

    def build_card(self) -> dict:
        """模型面板：一个全局下拉，选择后所有任务统一使用。"""
        heavy = self.settings.routing_heavy
        current = heavy[0].label() if heavy else ""
        prov_bits = []
        for name, p in self.settings.providers.items():
            prov_bits.append(f"{name} " + ("✅" if p.usable() else "⏸ 未启用"))
        md = (f"**当前全局模型**：`{current}`\n"
              "同一模型承担对话、骨架、正文、审核和视觉；任务启动后冻结本次选择。\n\n"
              f"provider：{' · '.join(prov_bits)}\n"
              "下拉选择立即切换并持久化（重启保持）；API 或模型失败时只通知，不会自动切换。")

        def _select(chain_name: str, current: str) -> dict:
            options = []
            for pname, p in self.settings.providers.items():
                if not p.usable():
                    continue
                for m in self._provider_models(pname):
                    v = f"{pname}/{m}"
                    options.append({"text": {"tag": "plain_text", "content": v}, "value": v})
            el: dict[str, Any] = {
                "tag": "select_static",
                "placeholder": {"tag": "plain_text", "content": f"切换全局模型（当前 {current}）"},
                "value": {"act": "model_switch", "chain": chain_name},
                "options": options,
            }
            if any(o["value"] == current for o in options):
                el["initial_option"] = current
            return el

        return {
            "config": {"wide_screen_mode": True},
            "header": {"template": "blue",
                       "title": {"tag": "plain_text", "content": "🎛 模型路由面板"}},
            "elements": [
                {"tag": "markdown", "content": md},
                {"tag": "action", "actions": [_select("all", current)]},
                {"tag": "action", "actions": [
                    {"tag": "button",
                     "text": {"tag": "plain_text", "content": "➕ 添加 provider"},
                     "type": "primary",
                     "value": {"act": "model_add_form"}},
                    {"tag": "button",
                     "text": {"tag": "plain_text", "content": "↩️ 恢复默认路由"},
                     "type": "default",
                     "value": {"act": "model_reset"}},
                ]},
                {"tag": "note", "elements": [{"tag": "plain_text", "content":
                    "⏸ 的 provider 在 config.yaml 启用并配好 key 后才会进选项；"
                    "也可用文本命令「模型 free-tokens/gpt-5.5」"}]},
            ],
        }

    def build_add_form_card(self) -> dict:
        """添加 provider 表单卡（从面板卡点「➕ 添加 provider」翻页过来）。"""
        return {
            "config": {"wide_screen_mode": True},
            "header": {"template": "blue",
                       "title": {"tag": "plain_text", "content": "➕ 添加 provider"}},
            "elements": [
                {"tag": "markdown", "content":
                    "OpenAI 兼容端点（`protocol` 选 anthropic 时走 /messages 协议）。"
                    "key 只写 .env，不落聊天记录；模型加入 heavy/light **可选项**，"
                    "提交后回面板明确选择才会生效。\n"
                    "💡 填**已存在**的名称（如 kimi-coding、moonshot）= 补/换 key 并启用，可顺带改地址/协议。"},
                {"tag": "form", "name": "add_provider", "elements": [
                    {"tag": "input", "name": "pname", "required": True,
                     "label": {"tag": "plain_text", "content": "名称"},
                     "placeholder": {"tag": "plain_text", "content": "小写字母/数字/-，如 openrouter"}},
                    {"tag": "input", "name": "base_url", "required": True,
                     "label": {"tag": "plain_text", "content": "base_url"},
                     "placeholder": {"tag": "plain_text", "content": "openai 协议到 /v1；anthropic 不带 /v1（自动后补）"}},
                    {"tag": "input", "name": "api_key", "required": True,
                     "label": {"tag": "plain_text", "content": "API key"},
                     "placeholder": {"tag": "plain_text", "content": "sk-…"}},
                    {"tag": "input", "name": "models", "required": False,
                     "label": {"tag": "plain_text", "content": "模型（可选）"},
                     "placeholder": {"tag": "plain_text", "content": "留空=自动拉取该端点的模型列表再挑选"}},
                    {"tag": "input", "name": "protocol", "required": False,
                     "label": {"tag": "plain_text", "content": "协议（可选）"},
                     "placeholder": {"tag": "plain_text", "content": "默认 openai；Claude Code 风格端点填 anthropic"}},
                    {"tag": "button", "name": "submit_add", "text": {"tag": "plain_text", "content": "➕ 添加"},
                     "type": "primary", "action_type": "form_submit",
                     "value": {"act": "model_add_provider"}},
                ]},
                {"tag": "action", "actions": [
                    {"tag": "button",
                     "text": {"tag": "plain_text", "content": "← 返回模型面板"},
                     "type": "default",
                     "value": {"act": "model_panel"}},
                ]},
                {"tag": "note", "elements": [{"tag": "plain_text", "content":
                    "校验失败只弹提示、不清空已填内容；模型留空会先探测 /models 再给挑选卡"}]},
            ],
        }

    def build_pick_card(self, pending: dict) -> dict:
        """模型挑选卡：探测 /models 成功后列出模型，留空全选或填部分。"""
        fetched = [str(m) for m in (pending.get("fetched") or [])]
        shown = "、".join(f"`{m}`" for m in fetched[:30])
        if len(fetched) > 30:
            shown += f" …等 {len(fetched)} 个"
        return {
            "config": {"wide_screen_mode": True},
            "header": {"template": "blue",
                       "title": {"tag": "plain_text", "content": f"🔎 选择要添加的模型（{pending.get('pname', '')}）"}},
            "elements": [
                {"tag": "markdown", "content":
                    f"端点 `{pending.get('base_url', '')}`（{pending.get('protocol', 'openai')} 协议）"
                    f"探测成功，拉到 **{len(fetched)}** 个模型：\n{shown}"},
                {"tag": "form", "name": "pick_models", "elements": [
                    {"tag": "input", "name": "models_pick", "required": False,
                     "label": {"tag": "plain_text", "content": "要添加的模型"},
                     "placeholder": {"tag": "plain_text", "content": f"留空=全部 {len(fetched)} 个；或填部分，逗号分隔"}},
                    {"tag": "button", "name": "submit_pick", "text": {"tag": "plain_text", "content": "✅ 确认添加"},
                     "type": "primary", "action_type": "form_submit",
                     "value": {"act": "model_confirm_provider"}},
                ]},
                {"tag": "action", "actions": [
                    {"tag": "button",
                     "text": {"tag": "plain_text", "content": "← 返回模型面板（放弃添加）"},
                     "type": "default",
                     "value": {"act": "model_panel"}},
                ]},
            ],
        }

    # ---- 卡片动作的拆解校验与暂存 ----

    def _validate_provider_basics(self, pname: str, base_url: str, api_key: str,
                                  protocol: str, *, allow_existing: bool = False) -> str | None:
        """名称/地址/key/协议 的基础校验；返回错误文案或 None（通过）。"""
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", pname):
            return "名称只能用小写字母/数字/中划线（如 openrouter）"
        if not allow_existing and (pname in self.settings.providers or pname in load_custom()):
            return f"provider「{pname}」已存在"
        if not allow_existing:
            env_name = _derive_env_name(pname)
            conflict = self._env_name_conflict(env_name)
            if conflict:
                return f"变量名 {env_name} 与{conflict}撞名，覆盖会串 key；请换个名称"
        if not base_url.startswith(("http://", "https://")):
            return "base_url 必须以 http(s):// 开头"
        if not api_key:
            return "API key 不能为空"
        if protocol not in ("openai", "anthropic"):
            return "协议只支持 openai 或 anthropic（留空默认 openai）"
        return None

    def _probe_models(self, base_url: str, api_key: str, protocol: str) -> list[str]:
        """用刚填的凭证探测端点模型列表（GET /models，按协议带 key）。"""
        import httpx
        base = base_url.rstrip("/")
        if protocol == "anthropic":
            url = base + "/models" if base.endswith("/v1") else base + "/v1/models"
            headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
        else:
            url = base + "/models"
            headers = {"Authorization": f"Bearer {api_key}"}
        r = httpx.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        return [str(m["id"]) for m in r.json().get("data", []) if m.get("id")]

    def handle_card_action(self, act: str, chain: str = "", option: str = "",
                           form_value: dict | None = None) -> tuple[str, bool, dict | None]:
        """卡片回调入口（bot 转发）：返回 (toast, ok, 刷新后的卡片)；card=None 表示
        不刷新（表单已填内容原样保留，仅弹 toast）。"""
        if act == "model_switch":
            msg = self.switch(chain, option)
            ok = msg.startswith("✅")
            return (f"✅ 全局模型 → {option}" if ok else msg[:80]), ok, self.build_card()
        if act == "model_reset":
            return self.reset(), True, self.build_card()
        if act == "model_add_form":  # 面板 → 表单卡（翻页）
            return "📝 填写新 provider 信息", True, self.build_add_form_card()
        if act == "model_panel":  # 表单/挑选卡 → 面板卡（翻回）
            clear_pending()  # 放弃添加：清掉含 key 的暂存（确认成功路径也会清）
            return "🎛 模型路由面板", True, self.build_card()
        if act == "model_add_provider":
            fv = form_value or {}
            pname = str(fv.get("pname") or "").strip().lower()
            base_url = str(fv.get("base_url") or "").strip().rstrip("/")
            api_key = str(fv.get("api_key") or "").strip()
            models_csv = str(fv.get("models") or "").strip()
            protocol = str(fv.get("protocol") or "").strip().lower() or "openai"
            exists = pname in self.settings.providers
            err = self._validate_provider_basics(pname, base_url, api_key, protocol,
                                                 allow_existing=exists)
            if err:
                return err, False, None  # 不刷卡：已填内容保留
            if models_csv:
                if exists:  # 已存在 → 更新（补 key 启用 + 补模型）
                    msg, ok = self.update_provider(pname, base_url, api_key, models_csv,
                                                   protocol=protocol)
                else:
                    msg, ok = self.add_provider(pname, base_url, api_key, models_csv,
                                                protocol=protocol)
                if ok:
                    clear_pending()  # 直接添加成功：可能残留的探测暂存一并清掉
                return msg[:80], ok, (self.build_card() if ok else None)
            # 模型留空 → 探测 /models → 挑选卡（凭证暂存服务端，不进卡片 JSON）
            try:
                fetched = self._probe_models(base_url, api_key, protocol)
            except Exception as exc:
                logger.warning("探测模型列表失败（%s）：%s", base_url, exc)
                return (f"拉取模型列表失败（{type(exc).__name__}），"
                        "请在「模型」栏手动填写后重新提交"), False, None
            if not fetched:
                return "该端点 /models 返回空，请在「模型」栏手动填写后重新提交", False, None
            pending = {"pname": pname, "base_url": base_url, "api_key": api_key,
                       "protocol": protocol, "fetched": fetched, "update": exists}
            save_pending(pending)
            return f"🔎 探测成功：{len(fetched)} 个模型，挑选要添加的", True, \
                self.build_pick_card(pending)
        if act == "model_confirm_provider":
            pending = load_pending()
            if not pending:
                return "待添加信息已过期，请重新走「➕ 添加 provider」", False, self.build_card()
            fv = form_value or {}
            pick = str(fv.get("models_pick") or "").strip()
            fetched = [str(m) for m in (pending.get("fetched") or [])]
            if pick:
                chosen = [m.strip() for m in pick.replace("，", ",").split(",") if m.strip()]
                invalid = [m for m in chosen if m not in fetched]
                if invalid:
                    return (f"模型名不在探测列表里：{'、'.join(invalid[:3])}"
                            "（照抄上面列表）"), False, self.build_pick_card(pending)
            else:
                chosen = fetched
            if not chosen:
                return "至少选一个模型", False, self.build_pick_card(pending)
            if pending.get("update"):  # 已存在的 provider → 更新而非新建
                msg, ok = self.update_provider(
                    str(pending["pname"]), str(pending["base_url"]), str(pending["api_key"]),
                    ",".join(chosen), protocol=str(pending.get("protocol") or "openai"))
            else:
                msg, ok = self.add_provider(
                    str(pending["pname"]), str(pending["base_url"]), str(pending["api_key"]),
                    ",".join(chosen), protocol=str(pending.get("protocol") or "openai"))
            if ok:
                clear_pending()
                return msg[:80], True, self.build_card()
            return msg[:80], False, self.build_pick_card(pending)
        return f"暂不支持的动作：{act or '空'}", False, self.build_card()
