import asyncio

import httpx
import pytest
from pydantic import BaseModel

from app.agent.config import LLMConfig, StructuredMode
from app.agent.llm import (
    LLMUnavailableError,
    OpenAICompatibleClient,
    StructuredOutputError,
    extract_json,
    generate_structured,
    model_schema,
    user_msg,
)
from tests.agent.fakes import ScriptedLLM


class Point(BaseModel):
    x: int
    label: str


@pytest.mark.parametrize(
    "text",
    [
        '{"x": 1, "label": "a"}',
        'Вот ответ:\n```json\n{"x": 1, "label": "a"}\n```',
        '<think>{"x": 2}?</think>{"x": 1, "label": "a"}',
        '{"x": 1, "label": "a",}',
        'Конечно! {"x": 1, "label": "a {скобка}"} Надеюсь, помог.',
    ],
)
def test_extract_json_variants(text):
    assert extract_json(text) == {"x": 1, "label": "a {скобка}" if "скобка" in text else "a"}


def test_extract_json_fails_without_json():
    with pytest.raises(ValueError):
        extract_json("я не знаю")


def test_compact_schema_inlines_refs_and_keeps_title_property():
    class Inner(BaseModel):
        title: str

    class Outer(BaseModel):
        inner: Inner

    schema = model_schema(Outer)
    assert "$defs" not in schema
    assert schema["properties"]["inner"]["properties"]["title"] == {"type": "string"}
    assert "title" not in schema


def test_generate_structured_repairs_invalid_output():
    llm = ScriptedLLM({"Point": ["это не json", '{"x": "много", "label": "a"}', '{"x": 3, "label": "a"}']})
    result = asyncio.run(generate_structured(llm, [user_msg("дай точку")], Point))
    assert result == Point(x=3, label="a")
    assert len(llm.calls) == 3
    repair = llm.calls[2][1][-1].content
    assert "x" in repair and "исправленный JSON" in repair


def test_generate_structured_semantic_validation():
    llm = ScriptedLLM({"Point": ['{"x": -1, "label": "a"}', '{"x": 1, "label": "a"}']})
    result = asyncio.run(
        generate_structured(llm, [user_msg("?")], Point, validate=lambda p: ["x должен быть > 0"] if p.x < 0 else [])
    )
    assert result.x == 1


def test_generate_structured_gives_up():
    llm = ScriptedLLM({"Point": ["нет", "нет", "нет"]})
    with pytest.raises(StructuredOutputError) as e:
        asyncio.run(generate_structured(llm, [user_msg("?")], Point, max_attempts=3))
    assert e.value.raw == "нет"


def _completion(content: str) -> dict:
    return {"model": "m", "choices": [{"message": {"content": content}, "finish_reason": "stop"}], "usage": {}}


def test_client_downgrades_structured_mode_on_400():
    formats = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = __import__("json").loads(request.content)
        rf = body.get("response_format")
        formats.append(rf["type"] if rf else None)
        if rf and rf["type"] == "json_schema":
            return httpx.Response(400, json={"error": "response_format not supported"})
        return httpx.Response(200, json=_completion('{"ok": true}'))

    config = LLMConfig(base_url="http://llm/v1", backoff_base_s=0)
    http = httpx.AsyncClient(base_url="http://llm/v1/", transport=httpx.MockTransport(handler))
    client = OpenAICompatibleClient(config, http=http)
    reply = asyncio.run(client.chat([user_msg("?")], json_schema={"type": "object"}))
    assert reply.content == '{"ok": true}'
    assert formats == ["json_schema", "json_object"]
    assert client.structured_mode is StructuredMode.JSON_OBJECT


def test_client_retries_then_raises_unavailable():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(503, text="overloaded")

    config = LLMConfig(base_url="http://llm/v1", max_retries=2, backoff_base_s=0)
    http = httpx.AsyncClient(base_url="http://llm/v1/", transport=httpx.MockTransport(handler))
    client = OpenAICompatibleClient(config, http=http)
    with pytest.raises(LLMUnavailableError):
        asyncio.run(client.chat([user_msg("?")]))
    assert calls == ["/v1/chat/completions"] * 3


def test_client_recovers_after_transient_error():
    responses = iter([httpx.Response(502), httpx.Response(200, json=_completion("привет"))])
    http = httpx.AsyncClient(base_url="http://llm/v1/", transport=httpx.MockTransport(lambda r: next(responses)))
    client = OpenAICompatibleClient(LLMConfig(backoff_base_s=0), http=http)
    assert asyncio.run(client.chat([user_msg("?")])).content == "привет"
