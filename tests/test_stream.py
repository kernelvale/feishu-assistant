"""kimi.py 流式化测试：chunk 累积、同模型断流重试、非流式调用、接口不变。"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import openai
import pytest

from feishu_assistant.config import ModelCandidate, ProviderConfig, Settings
from feishu_assistant.kimi import ChatResult, KimiClient, KimiError


def make_settings(max_retries=1) -> Settings:
    return Settings(
        providers={
            "free-tokens": ProviderConfig(
                "free-tokens", "https://ft/v1", "FREE_TOKENS_API_KEY", True, "sk-ft"),
            "moonshot": ProviderConfig(
                "moonshot", "https://ms/v1", "MOONSHOT_API_KEY", True, "sk-ms"),
        },
        routing_heavy=[ModelCandidate("free-tokens", "kimi-k3"),
                       ModelCandidate("moonshot", "kimi-k2-thinking")],
        routing_light=[ModelCandidate("free-tokens", "kimi-k3-highspeed"),
                       ModelCandidate("moonshot", "kimi-k2-0905-preview")],
        kimi_max_retries=max_retries,
    )


def rate_limit_error() -> openai.RateLimitError:
    req = httpx.Request("POST", "https://x/v1/chat/completions")
    return openai.RateLimitError("rate limited", response=httpx.Response(429, request=req), body=None)


def conn_error() -> openai.APIConnectionError:
    req = httpx.Request("POST", "https://x/v1/chat/completions")
    return openai.APIConnectionError(request=req)


def _chunk(content=None, reasoning=None):
    delta = SimpleNamespace(content=content, reasoning_content=reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class StreamOutcome:
    """一段流：chunks 正常产出后，可选在最后抛错（模拟断流）。"""

    def __init__(self, chunks, error=None):
        self.chunks = chunks
        self.error = error

    def iterate(self):
        yield from self.chunks
        if self.error:
            raise self.error


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if kwargs.get("stream"):
            return outcome.iterate()
        # 非流式：outcome = (content, reasoning)
        content, reasoning = outcome
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, reasoning_content=reasoning))])


class FakeOpenAI:
    def __init__(self, outcomes):
        self.completions = FakeCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)


MSGS = [{"role": "user", "content": "写个长文"}]


# ---- 流式标记与累积 ----

def test_heavy_uses_stream_and_accumulates():
    stream = StreamOutcome([
        _chunk(reasoning="先想想"),
        _chunk(content="第一"),
        _chunk(content="第二段", reasoning="再想想"),
        _chunk(content="。"),
    ])
    clients = {"free-tokens": FakeOpenAI([stream])}
    client = KimiClient(make_settings(), clients=clients)

    result = client.chat_full(MSGS, heavy=True)
    assert isinstance(result, ChatResult)
    assert result.content == "第一第二段。"
    assert result.reasoning == "先想想再想想"
    assert result.provider == "free-tokens" and result.model == "kimi-k3"
    assert clients["free-tokens"].completions.calls[0]["stream"] is True


def test_light_stays_non_stream():
    clients = {"free-tokens": FakeOpenAI([("快答", "")])}
    client = KimiClient(make_settings(), clients=clients)

    assert client.chat(MSGS, heavy=False) == "快答"
    assert "stream" not in clients["free-tokens"].completions.calls[0]


def test_long_input_auto_streams_even_when_not_marked_heavy():
    stream = StreamOutcome([_chunk(content="长输入回答")])
    clients = {"free-tokens": FakeOpenAI([stream])}
    client = KimiClient(make_settings(), clients=clients)

    messages = [{"role": "user", "content": "x" * 8_000}]
    assert client.chat(messages, heavy=False) == "长输入回答"
    assert clients["free-tokens"].completions.calls[0]["stream"] is True


def test_stream_empty_choices_tolerated():
    stream = StreamOutcome([
        SimpleNamespace(choices=[]),  # 心跳/角色帧
        _chunk(content="正文"),
    ])
    clients = {"free-tokens": FakeOpenAI([stream])}
    result = KimiClient(make_settings(), clients=clients).chat_full(MSGS, heavy=True)
    assert result.content == "正文"


# ---- 断流仅重试当前模型 ----

def test_stream_break_retries_full_regeneration(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    broken = StreamOutcome([_chunk(content="半截")], error=conn_error())
    good = StreamOutcome([_chunk(content="完整回答")])
    clients = {"free-tokens": FakeOpenAI([broken, good])}
    client = KimiClient(make_settings(max_retries=1), clients=clients)

    result = client.chat_full(MSGS, heavy=True)
    assert result.content == "完整回答"  # 重试是整次重新生成，不是续传
    assert len(clients["free-tokens"].completions.calls) == 2


def test_stream_break_exhausted_notifies_without_fallback(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    broken1 = StreamOutcome([_chunk(content="x")], error=conn_error())
    broken2 = StreamOutcome([_chunk(content="y")], error=conn_error())
    good = StreamOutcome([_chunk(content="备用完整")])
    clients = {
        "free-tokens": FakeOpenAI([broken1, broken2]),
        "moonshot": FakeOpenAI([good]),
    }
    client = KimiClient(make_settings(max_retries=1), clients=clients)

    with pytest.raises(KimiError, match="未自动切换"):
        client.chat_full(MSGS, heavy=True)
    assert len(clients["free-tokens"].completions.calls) == 2
    assert clients["moonshot"].completions.calls == []


def test_stream_all_fail_raises_chinese(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    broken = StreamOutcome([], error=conn_error())
    clients = {
        "free-tokens": FakeOpenAI([broken]),
        "moonshot": FakeOpenAI([rate_limit_error()]),
    }
    client = KimiClient(make_settings(max_retries=0), clients=clients)
    with pytest.raises(KimiError, match="当前选择"):
        client.chat_full(MSGS, heavy=True)
    assert clients["moonshot"].completions.calls == []


# ---- 空响应候选内重试 ----

def test_empty_response_retried_within_candidate(monkeypatch):
    """网关偶发空响应：候选内重试，拿到内容即成功（不再直接转移/报错）。"""
    monkeypatch.setattr("time.sleep", lambda _: None)
    clients = {"free-tokens": FakeOpenAI([StreamOutcome([_chunk(content=None)]),
                                          StreamOutcome([_chunk(content="有内容了")])])}
    client = KimiClient(make_settings(max_retries=1), clients=clients)
    assert client.chat_full(MSGS, heavy=True).content == "有内容了"
    assert len(clients["free-tokens"].completions.calls) == 2
