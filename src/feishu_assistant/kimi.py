"""多 provider 大模型客户端：用户显式选模 + 同模型重试。

- 所有任务都使用用户当前选中的唯一模型（routing 首项）
- 429/5xx/超时/连接错误在同一模型内指数退避重试 max_retries 次；耗尽后报错，
  绝不自动切换 provider 或模型。401/403/400 直接报错
- 流式：heavy 或总输入较长时走 stream=True（精读生成 5~12 分钟，
  非流式会把 read timeout 错当成整次生成的总时限）。
  逐 chunk 累积 content 与 reasoning_content；SDK 的 read timeout 按 chunk 间隔计，
  因此 config 的 kimi.timeout 语义是"chunk 间最大间隔"，总时长不受限。
  流中途断开按同样的重试语义处理（重试 = 整次重新生成）。
- 响应解析：兼容思考型模型的 reasoning_content —— 只把 content 当回复，
  reasoning 记入 debug 日志（可在 ChatResult.reasoning 取到）
- 空/畸形响应跨协议统一分类：空 choices、空 content、坏 JSON 都归入
  _EmptyResponseError，与 429/5xx 同族做同模型重试（不穿透原始异常）
- 日志：记录实际命中的 provider/model 与失败事件（绝不记录 key）

客户端按 provider 缓存；测试时通过 clients 参数注入 fake，无需真实 API Key。
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx
import openai

from .activity import track
from .config import ModelCandidate, ProviderConfig, Settings

logger = logging.getLogger("feishu_assistant.kimi")

# 简单启发式：命中关键词或足够长的消息使用流式长任务调用；不改变模型
HEAVY_KEYWORDS = (
    "精读", "论文", "分析", "总结", "复盘", "翻译", "对比",
    "代码", "debug", "设计", "推导", "证明", "写作", "润色",
)
HEAVY_LENGTH_THRESHOLD = 200
# 调用方即使忘记标 heavy，长上下文也必须流式。这是传输层的
# 安全阀，不改变模型、prompt 或输出格式。
AUTO_STREAM_INPUT_CHARS = 8_000

# 可在同一模型内重试的异常：429 限流、5xx、连接错误、超时；
# 网关空/畸形响应归入 _EmptyResponseError，在 except 处与本元组同族处理
RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.InternalServerError,
    openai.APIConnectionError,
    openai.APITimeoutError,
)
# 认证/权限错误（401/403）：不重试，立即通知用户
AUTH_ERRORS = (
    openai.AuthenticationError,
    openai.PermissionDeniedError,
)


class KimiError(Exception):
    """用户当前选择的大模型调用失败，信息为中文。"""


class KimiParameterError(KimiError):
    """当前模型拒绝了调用参数；调用方可改参数重试，但不得换模型。"""


class _EmptyResponseError(Exception):
    """网关返回空/畸形响应（空 choices、空 content、非法 JSON）：按可重试错误处理，耗尽后通知用户。"""


class _ReasoningOnlyError(Exception):
    """输出预算被 reasoning 吃光：同模型重试通常复现，直接通知用户。"""


# 模板哨兵：标记「共享前缀 | 本次变化」分界，用于网关前缀缓存。
# Anthropic 侧在哨兵处切 block 并挂 cache_control（实测网关接受 1h TTL 档）；
# OpenAI 侧发送前剥离职兵行，依赖网关自动前缀缓存（实测 cached_tokens 命中）。
CACHE_BREAK = "<!--CACHE_BREAK-->"
ANTHROPIC_CACHE_CONTROL = {"type": "ephemeral", "ttl": "1h"}


def _strip_cache_break(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI 协议路径：剥离哨兵行（共享内容已在最前，自动缓存即可命中）。"""
    out: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str) and CACHE_BREAK in content:
            cleaned = re.sub(
                rf"^[ \t]*{re.escape(CACHE_BREAK)}[ \t]*\n?", "", content, flags=re.M,
            )
            out.append({**msg, "content": cleaned})
        else:
            out.append(msg)
    return out


def _log_token_usage(usage: Any) -> None:
    """把 Anthropic/OpenAI 两种 usage 形态统一落一行日志，供缓存改造前后对比。"""
    if usage is None:
        return
    if isinstance(usage, dict):
        get = lambda k: usage.get(k)  # noqa: E731
        details = usage.get("prompt_tokens_details")
    else:
        get = lambda k: getattr(usage, k, None)  # noqa: E731
        details = getattr(usage, "prompt_tokens_details", None)
    inp = get("input_tokens") or get("prompt_tokens")
    out = get("output_tokens") or get("completion_tokens")
    cached = get("cache_read_input_tokens")
    if cached is None and details is not None:
        cached = (details.get("cached_tokens") if isinstance(details, dict)
                  else getattr(details, "cached_tokens", None))
    if inp is None and out is None and cached is None:
        return
    logger.info("tokens: in=%s cached=%s out=%s", inp, cached or 0, out)


# ---------------------------------------------------------------------------
# Anthropic Messages 协议（provider.protocol == "anthropic"）
# ---------------------------------------------------------------------------

def to_anthropic_payload(kwargs: dict[str, Any]) -> dict[str, Any]:
    """OpenAI chat.completions 参数 → Anthropic /messages 参数。

    差异处理：system 提成顶层参数；多模态 image_url(data:base64) → image block；
    连续同角色消息合并（Anthropic 要求角色交替）；max_tokens 必填（缺省 8192）。
    response_format / extra_body（reasoning_effort）是 OpenAI 系参数，不能原样发送。
    JSON object 模式会转换成 system 约束，避免 Anthropic 兼容端静默失去格式要求。
    """
    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []
    for msg in kwargs.get("messages") or []:
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            system_parts.append(content if isinstance(content, str) else str(content))
            continue
        if role not in ("user", "assistant"):
            continue
        if isinstance(content, list):  # 多模态：OpenAI parts → Anthropic blocks
            blocks: list[dict[str, Any]] = []
            for part in content:
                if part.get("type") == "text":
                    blocks.append({"type": "text", "text": part.get("text", "")})
                elif part.get("type") == "image_url":
                    url = str((part.get("image_url") or {}).get("url", ""))
                    m = re.match(r"^data:(image/[\w.+-]+);base64,(.+)$", url, re.DOTALL)
                    if m:
                        blocks.append({"type": "image", "source": {
                            "type": "base64", "media_type": m.group(1),
                            "data": re.sub(r"\s", "", m.group(2))}})
            messages.append({"role": role, "content": blocks})
        elif isinstance(content, str) and CACHE_BREAK in content:
            # 哨兵切分：共享前缀块挂 cache_control（网关实测接受 1h TTL），变化块原价
            prefix, suffix = content.split(CACHE_BREAK, 1)
            cb_blocks: list[dict[str, Any]] = [{
                "type": "text", "text": prefix.rstrip(),
                "cache_control": dict(ANTHROPIC_CACHE_CONTROL),
            }]
            if suffix.strip():
                cb_blocks.append({"type": "text", "text": suffix.lstrip()})
            messages.append({"role": role, "content": cb_blocks})
        else:
            messages.append({"role": role, "content": content})
    # 合并连续同角色（Anthropic 要求 user/assistant 交替）
    merged: list[dict[str, Any]] = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg["role"]:
            prev, cur = merged[-1]["content"], msg["content"]
            prev_l = prev if isinstance(prev, list) else [{"type": "text", "text": prev}]
            cur_l = cur if isinstance(cur, list) else [{"type": "text", "text": cur}]
            merged[-1]["content"] = prev_l + cur_l
        else:
            merged.append(msg)
    response_format = kwargs.get("response_format") or {}
    if isinstance(response_format, dict) and response_format.get("type") == "json_object":
        system_parts.append(
            "Return exactly one valid JSON object as the entire response. "
            "Do not include Markdown fences, commentary, or any text before or after the JSON."
        )
    payload: dict[str, Any] = {
        "model": kwargs.get("model"),
        "messages": merged,
        "max_tokens": kwargs.get("max_tokens") or 8192,
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)
    if kwargs.get("stream"):
        payload["stream"] = True
    return payload


def _raise_for_anthropic_status(resp: httpx.Response) -> None:
    """把 Anthropic 错误响应映射成 openai 异常体系，复用统一的重试语义。"""
    code = resp.status_code
    if code < 400:
        return
    try:
        resp.read()  # 流式响应（httpx.stream）必须先读体，否则 resp.text 抛 ResponseNotRead，
        body = resp.text[:200]  # 把真实的 429/500 掩成不可重试的怪错
    except httpx.StreamError:
        body = "<错误体读取失败>"
    msg = f"anthropic 协议 HTTP {code}: {body}"
    if code in (401, 403):
        raise openai.AuthenticationError(msg, response=resp, body=None)
    if code == 429:
        raise openai.RateLimitError(msg, response=resp, body=None)
    if code >= 500:
        raise openai.InternalServerError(msg, response=resp, body=None)
    raise openai.BadRequestError(msg, response=resp, body=None)


@dataclass
class ChatResult:
    """一次调用的完整结果。reasoning 为思考型模型的推理过程（可能为空）。"""

    content: str
    reasoning: str
    provider: str
    model: str
    finish_reason: str = ""
    usage: dict[str, int] | None = None


def is_heavy_task(text: str) -> bool:
    """判断一条消息是否使用流式长任务调用；模型选择不受影响。"""
    if len(text) >= HEAVY_LENGTH_THRESHOLD:
        return True
    lowered = text.lower()
    return any(kw in lowered for kw in HEAVY_KEYWORDS)


def _input_chars(messages: list[dict[str, Any]]) -> int:
    """估算请求消息大小，仅用于选择流式传输。"""
    total = 0
    for message in messages:
        content = message.get("content", "")
        total += len(content) if isinstance(content, str) else len(str(content))
    return total


class KimiClient:
    """对模型 API 的薄封装：只调用用户当前选择，失败不自动换模型。"""

    def __init__(self, settings: Settings, clients: dict[str, Any] | None = None) -> None:
        self.settings = settings
        self._clients: dict[str, Any] = dict(clients or {})  # provider 名 → client
        self._clients_lock = threading.Lock()

    def _client_for(self, provider: ProviderConfig) -> openai.OpenAI:
        """按 provider 缓存 OpenAI 客户端（各自 base_url + key）。

        max_retries=0：关闭 SDK 内置重试，由本类统一做当前模型的指数退避
        （否则两层重试叠加，实际请求数会翻几倍）。
        """
        if provider.name not in self._clients:
            with self._clients_lock:
                if provider.name not in self._clients:
                    self._clients[provider.name] = openai.OpenAI(
                        api_key=provider.api_key,
                        base_url=provider.base_url,
                        timeout=self.settings.kimi_timeout,
                        max_retries=0,
                    )
        return self._clients[provider.name]

    def _consume_stream(self, client: Any, kwargs: dict[str, Any]) -> tuple[str, str]:
        """消费流式响应，逐 chunk 累积正文与 reasoning。断流异常抛给上层按同模型重试。"""
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        chunks = 0
        usage: Any = None
        stream = client.chat.completions.create(
            **kwargs, stream=True, stream_options={"include_usage": True})
        for chunk in stream:  # 迭代期间的连接/超时异常同样按 RETRYABLE 语义处理
            chunks += 1
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = chunk_usage
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = choices[0].delta
            piece = getattr(delta, "content", None)
            think = getattr(delta, "reasoning_content", None)
            if piece:
                content_parts.append(piece)
            if think:
                reasoning_parts.append(think)
        logger.info(
            "流式完成：%d chunks，正文 %d 字符，reasoning %d 字符",
            chunks, sum(len(p) for p in content_parts), sum(len(p) for p in reasoning_parts),
        )
        _log_token_usage(usage)
        return "".join(content_parts), "".join(reasoning_parts)

    # ---- Anthropic Messages 协议（httpx 直连，不引 anthropic SDK）----

    def _anthropic_call(self, provider: ProviderConfig, kwargs: dict[str, Any]) -> tuple[str, str]:
        """Anthropic /messages 调用：非流式一次返回；流式解析 SSE content_block_delta。

        URL 约定（与 Claude Code 一致）：base_url 不带 /v1 时后补 /v1/messages；
        已带 /v1 的（老配置）只补 /messages，两种写法都兼容。
        非流式坏 JSON / 非对象 JSON 归 _EmptyResponseError，与空响应同族重试。
        """
        base = provider.base_url.rstrip("/")
        url = base + "/messages" if base.endswith("/v1") else base + "/v1/messages"
        headers = {"x-api-key": provider.api_key, "anthropic-version": "2023-06-01"}
        payload = to_anthropic_payload(kwargs)
        timeout = self.settings.kimi_timeout
        try:
            if payload.get("stream"):
                return self._anthropic_stream(url, headers, payload, timeout)
            resp = httpx.post(url, headers=headers, json=payload, timeout=timeout)
            _raise_for_anthropic_status(resp)
            data = resp.json()
        except httpx.TimeoutException as exc:
            raise openai.APITimeoutError(exc.request) from exc
        except httpx.ConnectError as exc:
            raise openai.APIConnectionError(request=exc.request) from exc
        except json.JSONDecodeError as exc:
            raise _EmptyResponseError("anthropic 返回非法 JSON") from exc
        if not isinstance(data, dict):
            raise _EmptyResponseError("anthropic 返回非对象 JSON")
        _log_token_usage(data.get("usage"))
        texts, thinks = [], []
        for block in data.get("content") or []:
            if block.get("type") == "text":
                texts.append(str(block.get("text", "")))
            elif block.get("type") == "thinking":
                thinks.append(str(block.get("thinking", "")))
        return "".join(texts), "".join(thinks)

    def _anthropic_stream(self, url: str, headers: dict[str, str], payload: dict[str, Any],
                          timeout: float) -> tuple[str, str]:
        texts: list[str] = []
        thinks: list[str] = []
        chunks = 0
        usage: dict[str, Any] = {}
        try:
            with httpx.stream("POST", url, headers=headers, json=payload,
                              timeout=timeout) as resp:
                _raise_for_anthropic_status(resp)
                for line in resp.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    try:
                        event = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
                    etype = event.get("type")
                    if etype == "content_block_delta":
                        delta = event.get("delta") or {}
                        if delta.get("type") == "text_delta":
                            piece = str(delta.get("text", ""))
                            if piece:
                                chunks += 1
                                texts.append(piece)
                        elif delta.get("type") == "thinking_delta":
                            thinks.append(str(delta.get("thinking", "")))
                    elif etype == "message_start":
                        usage.update((event.get("message") or {}).get("usage") or {})
                    elif etype == "message_delta":
                        usage.update(event.get("usage") or {})
                    elif etype == "error":
                        raise openai.InternalServerError(
                            f"anthropic 流式错误: {event.get('error')}",
                            response=resp, body=None)
        except httpx.TimeoutException as exc:
            raise openai.APITimeoutError(exc.request) from exc
        except httpx.ConnectError as exc:
            raise openai.APIConnectionError(request=exc.request) from exc
        logger.info("anthropic 流式完成：%d chunks，正文 %d 字符，thinking %d 字符",
                    chunks, sum(len(p) for p in texts), sum(len(p) for p in thinks))
        _log_token_usage(usage)
        return "".join(texts), "".join(thinks)

    def _create_one(
        self, provider: ProviderConfig, kwargs: dict[str, Any], *, stream: bool = False,
    ) -> tuple[str, str]:
        """调用单个候选，返回 (content, reasoning)。

        当前模型内指数退避重试（1s、2s……）；耗尽后原样抛出；400 转 KimiError。
        stream=True 时（heavy 路径）流式累积，重试为整次重新生成。
        provider.protocol == "anthropic" 时走 /messages 协议分支（httpx 直连）。
        """
        anthropic = provider.protocol == "anthropic"
        client = None if anthropic else self._client_for(provider)
        if not anthropic:
            # OpenAI 兼容网关（free-tokens）对显式 max_tokens 会按 ~16384 封顶，
            # 不传时网关默认预算反而更大（实测同日：不传 21175 chunks 完整出稿，
            # 传 16384/32768 均在 ~16381 chunks 处截断，reasoning 烧光正文为零）。
            # 输出预算因此只对 Anthropic 协议（max_tokens 必填）生效。
            kwargs = {k: v for k, v in kwargs.items() if k != "max_tokens"}
            kwargs["messages"] = _strip_cache_break(kwargs.get("messages") or [])
        last_err: Exception | None = None
        for attempt in range(self.settings.kimi_max_retries + 1):
            try:
                with track(f"LLM·{provider.name}/{kwargs.get('model')}"):  # 状态条活动行
                    if anthropic:
                        content, reasoning = self._anthropic_call(
                            provider, {**kwargs, "stream": stream})
                    elif stream:
                        content, reasoning = self._consume_stream(client, kwargs)
                    else:
                        try:
                            resp = client.chat.completions.create(**kwargs)
                        except json.JSONDecodeError as exc:
                            # 网关 200 + JSON 头但 body 畸形：与空响应同族，走同模型重试
                            raise _EmptyResponseError("返回非法 JSON") from exc
                        choices = getattr(resp, "choices", None) or []
                        if not choices:
                            # 网关偶发空 choices：与空内容同族，走同模型重试
                            raise _EmptyResponseError("返回空 choices")
                        msg = choices[0].message
                        content = getattr(msg, "content", None) or ""
                        reasoning = getattr(msg, "reasoning_content", None) or ""
                        _log_token_usage(getattr(resp, "usage", None))
                if not content:
                    if reasoning:
                        # 真实长论文故障：模型把整个 max_tokens 花在 thinking，重复同一模型
                        # 只会再浪费数分钟。直接停止并通知用户手动处理。
                        raise _ReasoningOnlyError(
                            f"只返回 reasoning（{len(reasoning)} 字符），正文为空"
                        )
                    raise _EmptyResponseError("返回空内容")  # 网关偶发空响应：走重试
                return content, reasoning
            except _ReasoningOnlyError:
                raise
            except (RETRYABLE_ERRORS + (_EmptyResponseError,)) as exc:
                last_err = exc
                if attempt >= self.settings.kimi_max_retries:
                    break
                delay = 2**attempt
                logger.info(
                    "provider=%s 请求失败（%s），%ds 后同模型重试（第 %d 次）",
                    provider.name, type(exc).__name__, delay, attempt + 1,
                )
                time.sleep(delay)
            except AUTH_ERRORS:
                raise  # 401/403 重试无意义，直接交给上层通知
            except openai.BadRequestError as exc:
                raise KimiParameterError(
                    f"当前模型 {provider.name}/{kwargs.get('model')} 拒绝了请求（参数问题）；"
                    f"系统未自动切换 API 或模型：{exc}"
                ) from exc
        raise last_err  # type: ignore[misc]

    def chat_full(
        self,
        messages: list[dict[str, str]],
        *,
        heavy: bool = False,
        json_mode: bool = False,
        low_reasoning: bool = False,
        max_tokens: int | None = None,
        candidates: list[ModelCandidate] | tuple[ModelCandidate, ...] | None = None,
    ) -> ChatResult:
        """只调用用户当前选择，返回完整结果（含 reasoning 与 provider/model）。

        low_reasoning：以 extra_body 传 reasoning_effort=low——部分思考型模型在复杂长 prompt 上会把默认 max_tokens
        全烧在 reasoning 导致正文为空（实测 free-tokens 网关），压低思考后 ~20s 出稿。
        max_tokens：显式抬输出预算（网关默认偏小，长骨架会被截断）。
        """
        configured = (
            list(candidates) if candidates is not None
            else list(self.settings.routing_heavy)
        )
        if not configured:
            raise KimiError("当前任务没有选择模型；系统未自动选择其他 API 或模型，请在「模型」面板手动选择。")
        cand = configured[0]
        provider = self.settings.providers.get(cand.provider)
        if provider is None:
            raise KimiError(
                f"当前选择 {cand.label()} 的 provider 不存在；系统未自动切换，请在「模型」面板重新选择。"
            )
        if not provider.enabled:
            raise KimiError(
                f"当前选择 {cand.label()} 的 provider 已禁用；系统未自动切换，请启用后重试或手动选择。"
            )
        if not provider.api_key:
            raise KimiError(
                f"当前选择 {cand.label()} 缺少 API key；系统未自动切换，请补齐凭证或手动选择。"
            )

        kwargs: dict[str, Any] = {"model": cand.model, "messages": messages}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if low_reasoning:
            kwargs["extra_body"] = {"reasoning_effort": "low"}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        # 流式与业务任务类型解耦：显式 heavy 仍然流式，任何长输入也自动
        # 流式。这样 read timeout 限制的是“相邻数据块之间卡住多久”，而不是
        # “整次推理允许多久”。
        stream = heavy or _input_chars(messages) >= AUTO_STREAM_INPUT_CHARS
        try:
            content, reasoning = self._create_one(provider, kwargs, stream=stream)
        except AUTH_ERRORS as exc:
            status = getattr(exc, "status_code", "?")
            logger.warning("当前模型失败：%s 认证失败（HTTP %s）", cand.label(), status)
            raise KimiError(
                f"当前选择 {cand.label()} 认证失败（HTTP {status}）；"
                "系统未自动切换 API 或模型，请检查 key 后重试或手动选择。"
            ) from exc
        except (RETRYABLE_ERRORS + (_EmptyResponseError,)) as exc:
            logger.warning(
                "当前模型失败：%s 重试 %d 次仍失败（%s）",
                cand.label(), self.settings.kimi_max_retries, type(exc).__name__,
            )
            raise KimiError(
                f"当前选择 {cand.label()} 调用失败（{type(exc).__name__}，"
                f"已重试 {self.settings.kimi_max_retries} 次）；系统未自动切换 API 或模型，"
                "请稍后重试、检查额度/服务状态，或在「模型」面板手动选择。"
            ) from exc
        except _ReasoningOnlyError as exc:
            logger.warning("当前模型失败：%s %s", cand.label(), exc)
            raise KimiError(
                f"当前选择 {cand.label()} 的输出预算被 reasoning 耗尽，正文为空；"
                "系统未自动切换 API 或模型，请调整模型或参数后重试。"
            ) from exc

        if reasoning:
            logger.debug("reasoning_content（%s）：%s", cand.label(), reasoning[:500])
        logger.info("模型命中：%s（heavy=%s, stream=%s）", cand.label(), heavy, stream)
        return ChatResult(content=content, reasoning=reasoning,
                          provider=cand.provider, model=cand.model)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        heavy: bool = False,
        json_mode: bool = False,
        low_reasoning: bool = False,
        max_tokens: int | None = None,
        candidates: list[ModelCandidate] | tuple[ModelCandidate, ...] | None = None,
    ) -> str:
        """通用对话，只返回正文（reasoning 丢弃，需要时用 chat_full）。"""
        return self.chat_full(messages, heavy=heavy, json_mode=json_mode,
                              low_reasoning=low_reasoning, max_tokens=max_tokens,
                              candidates=candidates).content

    def frozen(self) -> "FrozenKimiClient":
        """冻结当前全局选择；后续热切换不影响已启动任务。"""
        active = tuple(self.settings.routing_heavy[:1])
        return FrozenKimiClient(
            self, active, active,
        )

    def frozen_reading(self) -> "FrozenReadingClients":
        """按任务启动时的全局选择冻结所有精读角色。"""
        active = tuple(self.settings.routing_heavy[:1])

        return FrozenReadingClients(
            planner=FrozenKimiClient(self, active, active),
            writer=FrozenKimiClient(self, active, active),
            reviewer=FrozenKimiClient(self, active, active),
            vision=FrozenKimiClient(self, active, active),
        )

    # ---- [V7.1 清理] 已停用：生产无调用（heavy 判定已内联在 chat/session 模块）。经全项目核对生产链路零引用，保留备查 ----
    # def chat_auto(self, messages: list[dict[str, str]], user_text: str) -> str:
    #     """按 user_text 选择流式或非流式调用；始终使用同一全局模型。"""
    #     return self.chat(messages, heavy=is_heavy_task(user_text))

    def chat_json(self, messages: list[dict[str, str]], *, heavy: bool = True) -> dict:
        """JSON 模式对话，返回解析后的 dict；解析失败抛 KimiError。"""
        raw = self.chat(messages, heavy=heavy, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise KimiError(f"模型返回的内容不是合法 JSON：{raw[:200]}") from exc


class FrozenKimiClient:
    """任务级不可变路由视图；接口与 KimiClient 的常用子集一致。"""

    def __init__(
        self,
        parent: KimiClient,
        light: tuple[ModelCandidate, ...],
        heavy: tuple[ModelCandidate, ...],
    ) -> None:
        self.parent = parent
        self.light = light
        self.heavy = heavy

    def route_manifest(self) -> dict[str, list[str]]:
        return {
            "light": [cand.label() for cand in self.light],
            "heavy": [cand.label() for cand in self.heavy],
        }

    def chat_full(self, messages: list[dict[str, Any]], *, heavy: bool = False, **kwargs: Any) -> ChatResult:
        return self.parent.chat_full(
            messages, heavy=heavy,
            candidates=self.heavy if heavy else self.light,
            **kwargs,
        )

    def chat(self, messages: list[dict[str, Any]], *, heavy: bool = False, **kwargs: Any) -> str:
        return self.chat_full(messages, heavy=heavy, **kwargs).content


@dataclass(frozen=True)
class FrozenReadingClients:
    """精读任务的角色模型快照；每个角色都只有用户选择的单一模型。"""

    planner: FrozenKimiClient
    writer: FrozenKimiClient
    reviewer: FrozenKimiClient
    vision: FrozenKimiClient

    def route_manifest(self) -> dict[str, list[str]]:
        return {
            "planner": [cand.label() for cand in self.planner.light],
            "writer": [cand.label() for cand in self.writer.heavy],
            "reviewer": [cand.label() for cand in self.reviewer.heavy],
            "vision": [cand.label() for cand in self.vision.heavy],
        }
