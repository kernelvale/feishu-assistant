"""kimi.py 测试：全局模型选择、禁止自动切换、reasoning_content 解析、同模型重试。"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import openai
import pytest

from feishu_assistant.config import ModelCandidate, ProviderConfig, Settings
from feishu_assistant.kimi import KimiClient, KimiError, is_heavy_task


def make_settings(*, ft_key="sk-ft", ms_key="sk-ms", max_retries=1) -> Settings:
    """双 provider 可供手动选择；只调用链首，max_retries 默认 1。"""
    return Settings(
        providers={
            "free-tokens": ProviderConfig(
                "free-tokens", "https://free-tokens.msh.team/v1", "FREE_TOKENS_API_KEY",
                enabled=True, api_key=ft_key,
            ),
            "moonshot": ProviderConfig(
                "moonshot", "https://api.moonshot.cn/v1", "MOONSHOT_API_KEY",
                enabled=True, api_key=ms_key,
            ),
        },
        routing_heavy=[
            ModelCandidate("free-tokens", "kimi-k3"),
            ModelCandidate("moonshot", "kimi-k2-thinking"),
        ],
        routing_light=[
            ModelCandidate("free-tokens", "kimi-k3-highspeed"),
            ModelCandidate("moonshot", "kimi-k2-0905-preview"),
        ],
        kimi_max_retries=max_retries,
    )


def fake_message(content: str, reasoning: str = ""):
    msg = SimpleNamespace(content=content)
    if reasoning:
        msg.reasoning_content = reasoning
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class FakeCompletions:
    """记录调用参数，按队列抛错或返回内容。"""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if kwargs.get("stream"):
            # heavy 流式路径：把 fake_message 的正文包成单 chunk 迭代器
            content = outcome.choices[0].message.content
            reasoning = getattr(outcome.choices[0].message, "reasoning_content", None)
            delta = SimpleNamespace(content=content, reasoning_content=reasoning)
            return iter([SimpleNamespace(choices=[SimpleNamespace(delta=delta)])])
        return outcome  # 已是 fake_message 结构


class FakeOpenAI:
    def __init__(self, outcomes):
        self.completions = FakeCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)


def make_clients(ft_outcomes, ms_outcomes=None):
    clients = {"free-tokens": FakeOpenAI(ft_outcomes)}
    if ms_outcomes is not None:
        clients["moonshot"] = FakeOpenAI(ms_outcomes)
    return clients


def rate_limit_error() -> openai.RateLimitError:
    req = httpx.Request("POST", "https://x/v1/chat/completions")
    return openai.RateLimitError("rate limited", response=httpx.Response(429, request=req), body=None)


def auth_error() -> openai.AuthenticationError:
    req = httpx.Request("POST", "https://x/v1/chat/completions")
    return openai.AuthenticationError("bad key", response=httpx.Response(401, request=req), body=None)


def forbidden_error() -> openai.PermissionDeniedError:
    req = httpx.Request("POST", "https://x/v1/chat/completions")
    return openai.PermissionDeniedError("forbidden", response=httpx.Response(403, request=req), body=None)


def server_error() -> openai.InternalServerError:
    req = httpx.Request("POST", "https://x/v1/chat/completions")
    return openai.InternalServerError("boom", response=httpx.Response(500, request=req), body=None)


def bad_request_error() -> openai.BadRequestError:
    req = httpx.Request("POST", "https://x/v1/chat/completions")
    return openai.BadRequestError("bad params", response=httpx.Response(400, request=req), body=None)


MSGS = [{"role": "user", "content": "hi"}]


# ---- 链选择 ----

def test_all_calls_use_global_selected_model():
    clients = make_clients([fake_message("好的")])
    client = KimiClient(make_settings(), clients=clients)
    assert client.chat(MSGS) == "好的"
    assert clients["free-tokens"].completions.calls[0]["model"] == "kimi-k3"
    assert "moonshot" not in clients or not clients["moonshot"].completions.calls


def test_heavy_chain_uses_first_candidate():
    clients = make_clients([fake_message("深度回答")])
    client = KimiClient(make_settings(), clients=clients)
    client.chat(MSGS, heavy=True)
    assert clients["free-tokens"].completions.calls[0]["model"] == "kimi-k3"


# ---- [V7.1 清理] 已停用：对应生产代码 chat_auto 已停用。经全项目核对生产链路零引用，保留备查 ----
# def test_chat_auto_routes_by_user_text():
#     clients = make_clients([fake_message("r1"), fake_message("r2")])
#     client = KimiClient(make_settings(), clients=clients)
#     client.chat_auto(MSGS, "你好")
#     client.chat_auto(MSGS, "帮我精读这篇论文")
#     calls = clients["free-tokens"].completions.calls
#     assert [call["model"] for call in calls] == ["kimi-k3", "kimi-k3"]
#     assert is_heavy_task("复盘一下今天的面试") and not is_heavy_task("你好")


# ---- 单模型失败通知（禁止自动切换） ----

def test_retry_exhausted_notifies_without_fallback(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    clients = make_clients(
        [rate_limit_error(), rate_limit_error()],   # 链首 1+1 次全挂
        [fake_message("备用成功")],
    )
    client = KimiClient(make_settings(max_retries=1), clients=clients)

    with pytest.raises(KimiError, match="系统未自动切换"):
        client.chat(MSGS)
    assert len(clients["free-tokens"].completions.calls) == 2  # 候选内重试 1 次
    assert clients["moonshot"].completions.calls == []
    assert sleeps == [1]


def test_auth_error_notifies_without_retry_or_fallback(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    clients = make_clients([auth_error()], [fake_message("备用成功")])
    client = KimiClient(make_settings(max_retries=3), clients=clients)

    with pytest.raises(KimiError, match="认证失败"):
        client.chat(MSGS)
    assert len(clients["free-tokens"].completions.calls) == 1  # 401 不重试
    assert clients["moonshot"].completions.calls == []


@pytest.mark.parametrize("err_factory", [auth_error, forbidden_error, rate_limit_error, server_error])
def test_all_provider_failures_never_trigger_fallback(err_factory, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    clients = make_clients([err_factory()] * 3, [fake_message("备用成功")])
    client = KimiClient(make_settings(max_retries=2), clients=clients)
    with pytest.raises(KimiError, match="未自动切换"):
        client.chat(MSGS)
    assert clients["moonshot"].completions.calls == []


def test_failure_message_only_names_selected_model(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    clients = make_clients([server_error()] * 2, [auth_error()])
    client = KimiClient(make_settings(max_retries=1), clients=clients)

    with pytest.raises(KimiError, match="当前选择") as exc_info:
        client.chat(MSGS)
    assert "free-tokens" in str(exc_info.value) and "moonshot" not in str(exc_info.value)


def test_no_usable_candidate_raises_chinese_error():
    settings = make_settings(ft_key="", ms_key="")  # 全部缺 key
    client = KimiClient(settings, clients=make_clients([]))
    with pytest.raises(KimiError, match="缺少 API key"):
        client.chat(MSGS)


def test_disabled_selected_provider_is_not_skipped():
    settings = make_settings()
    settings.providers["free-tokens"].enabled = False
    clients = make_clients([], [fake_message("走备用")])
    client = KimiClient(settings, clients=clients)

    with pytest.raises(KimiError, match="已禁用"):
        client.chat(MSGS)
    assert clients["free-tokens"].completions.calls == []
    assert clients["moonshot"].completions.calls == []


def test_bad_request_no_model_switch(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    clients = make_clients([bad_request_error()], [fake_message("不该走到")])
    client = KimiClient(make_settings(), clients=clients)

    with pytest.raises(KimiError, match="未自动切换"):
        client.chat(MSGS)
    assert clients["moonshot"].completions.calls == []


def test_empty_content_retries_then_notifies_without_fallback():
    """空响应：只在当前模型内重试，耗尽后通知。"""
    clients = make_clients([fake_message(""), fake_message("")], [fake_message("备用成功")])
    client = KimiClient(make_settings(), clients=clients)  # max_retries=1
    with pytest.raises(KimiError, match="未自动切换"):
        client.chat(MSGS)
    assert len(clients["free-tokens"].completions.calls) == 2  # 候选内重试 1 次
    assert len(clients["moonshot"].completions.calls) == 0


def test_nonstream_empty_choices_retries_within_candidate(monkeypatch):
    """非流式空 choices（网关偶发）：与空内容同族，候选内重试，不再 IndexError 穿透。"""
    monkeypatch.setattr("time.sleep", lambda _: None)
    clients = make_clients(
        [SimpleNamespace(choices=[]), fake_message("重试成功")],
        [fake_message("不该命中")],
    )
    client = KimiClient(make_settings(max_retries=1), clients=clients)
    assert client.chat(MSGS) == "重试成功"
    assert len(clients["free-tokens"].completions.calls) == 2
    assert clients["moonshot"].completions.calls == []


def test_nonstream_empty_choices_exhausted_notifies_without_fallback(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    clients = make_clients([SimpleNamespace(choices=[])] * 2, [fake_message("备用成功")])
    client = KimiClient(make_settings(max_retries=1), clients=clients)
    with pytest.raises(KimiError, match="未自动切换"):
        client.chat(MSGS)
    assert len(clients["free-tokens"].completions.calls) == 2
    assert clients["moonshot"].completions.calls == []


# ---- reasoning_content 解析 ----

def test_reasoning_content_separated_from_content():
    clients = make_clients([fake_message("正式回复", reasoning="让我想想……")])
    client = KimiClient(make_settings(), clients=clients)

    result = client.chat_full(MSGS)
    assert result.content == "正式回复"
    assert result.reasoning == "让我想想……"
    assert result.provider == "free-tokens" and result.model == "kimi-k3"
    # chat() 只返回正文
    clients2 = make_clients([fake_message("正式回复", reasoning="让我想想……")])
    assert KimiClient(make_settings(), clients=clients2).chat(MSGS) == "正式回复"


def test_reasoning_only_notifies_without_retry_or_fallback():
    clients = make_clients(
        [fake_message("", reasoning="思考" * 100)],
        [fake_message("备用正文")],
    )
    client = KimiClient(make_settings(max_retries=5), clients=clients)
    with pytest.raises(KimiError, match="reasoning"):
        client.chat(MSGS)
    assert len(clients["free-tokens"].completions.calls) == 1
    assert len(clients["moonshot"].completions.calls) == 0


def test_frozen_route_ignores_later_global_route_mutation():
    settings = make_settings()
    clients = make_clients([fake_message("冻结链成功")], [fake_message("不应命中")])
    client = KimiClient(settings, clients=clients)
    frozen = client.frozen()
    settings.routing_heavy[:] = [ModelCandidate("moonshot", "later-model")]

    assert frozen.chat(MSGS) == "冻结链成功"
    assert clients["free-tokens"].completions.calls[0]["model"] == "kimi-k3"
    assert clients["moonshot"].completions.calls == []


def test_reading_roles_freeze_current_user_choices():
    settings = make_settings()
    clients = make_clients([fake_message("计划"), fake_message("正文")])
    client = KimiClient(settings, clients=clients)
    roles = client.frozen_reading()
    settings.routing_heavy[:] = [ModelCandidate("moonshot", "chat-override")]

    assert roles.planner.chat(MSGS) == "计划"
    assert roles.writer.chat(MSGS, heavy=True) == "正文"
    assert [call["model"] for call in clients["free-tokens"].completions.calls] == [
        "kimi-k3", "kimi-k3",
    ]


def test_message_without_reasoning_field_ok():
    clients = make_clients([fake_message("普通回复")])  # 无 reasoning_content 属性
    result = KimiClient(make_settings(), clients=clients).chat_full(MSGS)
    assert result.content == "普通回复" and result.reasoning == ""


# ---- JSON 模式 ----

def test_json_mode_passes_response_format_and_parses():
    clients = make_clients([fake_message('{"标题": "xxx"}')])
    client = KimiClient(make_settings(), clients=clients)
    assert client.chat_json(MSGS) == {"标题": "xxx"}
    assert clients["free-tokens"].completions.calls[0]["response_format"] == {"type": "json_object"}


def test_chat_json_invalid_json_raises_chinese_error():
    clients = make_clients([fake_message("这不是 JSON")])
    client = KimiClient(make_settings(), clients=clients)
    with pytest.raises(KimiError, match="不是合法 JSON"):
        client.chat_json(MSGS)


# ---- 候选内重试 ----

def test_retry_within_candidate_then_succeed(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    clients = make_clients([rate_limit_error(), fake_message("成功")])
    client = KimiClient(make_settings(max_retries=2), clients=clients)

    assert client.chat(MSGS) == "成功"
    assert len(clients["free-tokens"].completions.calls) == 2
    assert sleeps == [1]


# ---- Anthropic Messages 协议（provider.protocol == "anthropic"）----

def make_anthropic_settings() -> Settings:
    """anthropic 协议 provider 为当前选择；后续条目不得自动调用。"""
    return Settings(
        providers={
            "claude-x": ProviderConfig(
                "claude-x", "https://claude.example/v1", "CLAUDE_X_API_KEY",
                enabled=True, api_key="sk-cx", protocol="anthropic",
            ),
            "free-tokens": ProviderConfig(
                "free-tokens", "https://free-tokens.msh.team/v1", "FREE_TOKENS_API_KEY",
                enabled=True, api_key="sk-ft",
            ),
        },
        routing_heavy=[ModelCandidate("claude-x", "k3"), ModelCandidate("free-tokens", "kimi-k3")],
        routing_light=[ModelCandidate("claude-x", "k3"), ModelCandidate("free-tokens", "kimi-k3-highspeed")],
        kimi_max_retries=0,
    )


def anthropic_response(status: int, payload: dict) -> httpx.Response:
    return httpx.Response(status, json=payload,
                          request=httpx.Request("POST", "https://claude.example/v1/messages"))


def test_to_anthropic_payload_system_merge_and_vision():
    from feishu_assistant.kimi import to_anthropic_payload
    b64 = "aGVsbG8="  # "hello"
    payload = to_anthropic_payload({
        "model": "k3",
        "messages": [
            {"role": "system", "content": "人设一"},
            {"role": "system", "content": "人设二"},
            {"role": "user", "content": "问题一"},
            {"role": "user", "content": "问题二"},
            {"role": "user", "content": [
                {"type": "text", "text": "看图"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]},
        ],
        "response_format": {"type": "json_object"},   # OpenAI 系参数应被丢弃
        "extra_body": {"reasoning_effort": "low"},
    })
    assert payload["system"].startswith("人设一\n\n人设二\n\n")
    assert payload["max_tokens"] == 8192  # 缺省必填
    assert "response_format" not in payload and "extra_body" not in payload
    assert "exactly one valid JSON object" in payload["system"]
    # 三条连续 user 合并成一条（交替规则），内容为 text blocks + image block
    assert [m["role"] for m in payload["messages"]] == ["user"]
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "问题一"}
    assert content[1] == {"type": "text", "text": "问题二"}
    assert content[2] == {"type": "text", "text": "看图"}
    assert content[3]["type"] == "image"
    assert content[3]["source"]["media_type"] == "image/jpeg"
    assert content[3]["source"]["data"] == b64


def test_anthropic_nonstream_text_and_thinking(monkeypatch):
    calls: list[dict] = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers, "json": json})
        return anthropic_response(200, {"content": [
            {"type": "thinking", "thinking": "想一想"},
            {"type": "text", "text": "答案"},
        ]})

    monkeypatch.setattr(httpx, "post", fake_post)
    client = KimiClient(make_anthropic_settings())
    result = client.chat_full([{"role": "user", "content": "hi"}], heavy=False)
    assert result.content == "答案" and result.reasoning == "想一想"
    assert result.provider == "claude-x"
    assert calls[0]["url"] == "https://claude.example/v1/messages"
    assert calls[0]["headers"]["x-api-key"] == "sk-cx"
    assert calls[0]["headers"]["anthropic-version"] == "2023-06-01"


def test_anthropic_stream_deltas(monkeypatch):
    class FakeStreamResp:
        status_code = 200
        text = ""

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def iter_lines(self):
            yield "event: message_start"
            yield 'data: {"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"推"}}'
            yield 'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"你"}}'
            yield 'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"好"}}'
            yield 'data: {"type":"message_stop"}'

    monkeypatch.setattr(httpx, "stream",
                        lambda *a, **k: FakeStreamResp())
    client = KimiClient(make_anthropic_settings())
    result = client.chat_full([{"role": "user", "content": "hi"}], heavy=True)
    assert result.content == "你好" and result.reasoning == "推"


def test_anthropic_auth_error_notifies_without_openai_fallback(monkeypatch):
    monkeypatch.setattr(httpx, "post",
                        lambda *a, **k: anthropic_response(401, {"error": {"type": "authentication_error", "message": "bad key"}}))
    clients = make_clients([fake_message("兜底成功")])
    client = KimiClient(make_anthropic_settings(), clients=clients)
    with pytest.raises(KimiError, match="认证失败"):
        client.chat_full([{"role": "user", "content": "hi"}], heavy=False)
    assert clients["free-tokens"].completions.calls == []


def test_anthropic_rate_limit_retries_within_candidate(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    state = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        state["n"] += 1
        if state["n"] == 1:
            return anthropic_response(429, {"error": {"type": "rate_limit_error"}})
        return anthropic_response(200, {"content": [{"type": "text", "text": "重试后成功"}]})

    monkeypatch.setattr(httpx, "post", fake_post)
    s = make_anthropic_settings()
    s.kimi_max_retries = 1
    client = KimiClient(s)
    assert client.chat([{"role": "user", "content": "hi"}]) == "重试后成功"
    assert state["n"] == 2


def test_anthropic_url_v1_completion(monkeypatch):
    """URL 约定：base_url 不带 /v1 → 后补 /v1/messages；已带 /v1 → 只补 /messages。"""
    urls: list[str] = []

    def fake_post(url, headers=None, json=None, timeout=None):
        urls.append(url)
        return anthropic_response(200, {"content": [{"type": "text", "text": "ok"}]})

    monkeypatch.setattr(httpx, "post", fake_post)
    s = make_anthropic_settings()

    s.providers["claude-x"].base_url = "https://api.kimi.com/coding"  # 不带 /v1
    KimiClient(s).chat([{"role": "user", "content": "hi"}])
    assert urls[-1] == "https://api.kimi.com/coding/v1/messages"

    s.providers["claude-x"].base_url = "https://api.kimi.com/coding/v1/"  # 已带（含尾斜杠）
    KimiClient(s).chat([{"role": "user", "content": "hi"}])
    assert urls[-1] == "https://api.kimi.com/coding/v1/messages"


def anthropic_raw_response(status: int, body: bytes) -> httpx.Response:
    """非 JSON 结构化的原始响应体（模拟网关返回 HTML/数组等畸形内容）。"""
    return httpx.Response(status, content=body,
                          request=httpx.Request("POST", "https://claude.example/v1/messages"))


def test_anthropic_bad_json_retries_as_empty_response(monkeypatch):
    """anthropic 非流式坏 JSON：与空响应同族候选内重试，不再 JSONDecodeError 穿透。"""
    monkeypatch.setattr("time.sleep", lambda _: None)
    state = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        state["n"] += 1
        if state["n"] == 1:
            return anthropic_raw_response(200, b"<bad gateway>")
        return anthropic_response(200, {"content": [{"type": "text", "text": "重试后成功"}]})

    monkeypatch.setattr(httpx, "post", fake_post)
    s = make_anthropic_settings()
    s.kimi_max_retries = 1
    client = KimiClient(s)
    assert client.chat([{"role": "user", "content": "hi"}]) == "重试后成功"
    assert state["n"] == 2


def test_anthropic_non_dict_json_notifies_without_fallback(monkeypatch):
    """anthropic 返回合法 JSON 但非对象（如数组）：同族处理，耗尽后通知且不穿透。"""
    monkeypatch.setattr("time.sleep", lambda _: None)
    monkeypatch.setattr(
        httpx, "post",
        lambda *a, **k: anthropic_raw_response(200, b'["not", "a", "dict"]'))
    clients = make_clients([fake_message("备用成功")])
    client = KimiClient(make_anthropic_settings(), clients=clients)  # max_retries=0
    with pytest.raises(KimiError, match="未自动切换"):
        client.chat([{"role": "user", "content": "hi"}])
    assert clients["free-tokens"].completions.calls == []


def test_nonstream_bad_json_retries_as_empty_response(monkeypatch):
    """OpenAI 协议 200+JSON 头+畸形 body：JSONDecodeError 统一进空响应族，候选内重试不穿透。"""
    import json as _json
    monkeypatch.setattr("time.sleep", lambda _: None)
    clients = make_clients(
        [_json.JSONDecodeError("Expecting value", "<html>bad gateway</html>", 0),
         fake_message("重试成功")],
        [fake_message("不该命中")],
    )
    client = KimiClient(make_settings(max_retries=1), clients=clients)
    assert client.chat(MSGS) == "重试成功"
    assert len(clients["free-tokens"].completions.calls) == 2
    assert clients["moonshot"].completions.calls == []


def test_max_tokens_stripped_for_openai_but_kept_for_anthropic(monkeypatch):
    """OpenAI 兼容网关剥离显式 max_tokens（封顶副作用），anthropic 保留（协议必填）。"""
    ft = FakeOpenAI([fake_message("ok")])
    client = KimiClient(make_settings(max_retries=0), clients={"free-tokens": ft})
    assert client.chat(MSGS, max_tokens=32768) == "ok"
    assert "max_tokens" not in ft.completions.calls[0]

    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen["payload"] = json
        return anthropic_response(200, {"content": [{"type": "text", "text": "ok2"}]})

    monkeypatch.setattr(httpx, "post", fake_post)
    client2 = KimiClient(make_anthropic_settings())
    assert client2.chat([{"role": "user", "content": "hi"}], max_tokens=32768) == "ok2"
    assert seen["payload"]["max_tokens"] == 32768


# ---- 前缀缓存：哨兵切分与剥离 ----

def test_to_anthropic_payload_cache_break_marks_shared_prefix():
    from feishu_assistant.kimi import CACHE_BREAK, to_anthropic_payload
    payload = to_anthropic_payload({
        "model": "k3",
        "messages": [{"role": "user", "content": f"共享原文块\n{CACHE_BREAK}\n本节指令"}],
    })
    content = payload["messages"][0]["content"]
    assert content[0]["type"] == "text" and content[0]["text"] == "共享原文块"
    assert content[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert content[1] == {"type": "text", "text": "本节指令"}
    assert CACHE_BREAK not in str(payload)


def test_to_anthropic_payload_without_cache_break_unchanged():
    from feishu_assistant.kimi import to_anthropic_payload
    payload = to_anthropic_payload({
        "model": "k3",
        "messages": [{"role": "user", "content": "普通内容"}],
    })
    assert payload["messages"][0]["content"] == "普通内容"


def test_strip_cache_break_removes_only_sentinel_line():
    from feishu_assistant.kimi import CACHE_BREAK, _strip_cache_break
    messages = [
        {"role": "user", "content": f"共享原文块\n{CACHE_BREAK}\n本节指令"},
        {"role": "assistant", "content": "无哨兵"},
    ]
    cleaned = _strip_cache_break(messages)
    assert cleaned[0]["content"] == "共享原文块\n本节指令"
    assert cleaned[1]["content"] == "无哨兵"


def test_log_token_usage_accepts_both_protocol_shapes(caplog):
    import logging
    from feishu_assistant.kimi import _log_token_usage
    with caplog.at_level(logging.INFO, logger="feishu_assistant.kimi"):
        _log_token_usage({"input_tokens": 100, "cache_read_input_tokens": 90,
                          "output_tokens": 10})
        _log_token_usage({"prompt_tokens": 200, "completion_tokens": 20,
                          "prompt_tokens_details": {"cached_tokens": 180}})
        _log_token_usage(None)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("tokens:")]
    assert lines == ["tokens: in=100 cached=90 out=10", "tokens: in=200 cached=180 out=20"]


def test_anthropic_status_error_reads_streaming_body_safely():
    """流式响应的错误体未被读过：先 read 再取 text，429 必须映射成 RateLimitError
    而不是 ResponseNotRead（否则真实限流被掩成不可重试怪错，整轮任务判死）。"""
    import openai
    from feishu_assistant.kimi import _raise_for_anthropic_status

    class _Bytes(httpx.SyncByteStream):
        def __iter__(self):
            yield b'{"error": {"type": "rate_limit"}}'

    resp = httpx.Response(
        429, stream=_Bytes(),
        request=httpx.Request("POST", "https://claude.example/v1/messages"),
    )
    with pytest.raises(openai.RateLimitError):
        _raise_for_anthropic_status(resp)
