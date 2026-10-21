from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterator, Sequence
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .client import ChatMessage, LLMClient, assistant_msg, system_msg, user_msg
from .errors import StructuredOutputError

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

SCHEMA_INSTRUCTION = (
    "Ответь одним JSON-объектом строго по JSON-схеме ниже. Без пояснений, без markdown.\nСхема: {schema}"
)
SHORT_INSTRUCTION = "Ответь одним JSON-объектом в описанном формате. Без пояснений, без markdown."
MAX_PROMPT_SCHEMA_CHARS = 4000
REPAIR_PROMPT = "Ответ не прошёл проверку:\n{errors}\nВерни только исправленный JSON-объект без пояснений."

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def extract_json(text: str) -> Any:
    text = _THINK_RE.sub("", text).strip()
    candidates = [m.group(1) for m in _FENCE_RE.finditer(text)] + [text]
    for candidate in candidates:
        for chunk in _balanced_objects(candidate):
            for variant in (chunk, _TRAILING_COMMA_RE.sub(r"\1", chunk)):
                try:
                    return json.loads(variant)
                except json.JSONDecodeError:
                    pass
    raise ValueError("в ответе нет корректного JSON-объекта")


def _balanced_objects(text: str, limit: int = 20) -> Iterator[str]:
    found = 0
    start = text.find("{")
    while start != -1 and found < limit:
        depth, in_str, escaped = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    found += 1
                    yield text[start : i + 1]
                    break
        start = text.find("{", start + 1)


def compact_schema(schema: dict[str, Any]) -> dict[str, Any]:
    defs = schema.get("$defs", {})

    def walk(node: Any, is_properties: bool = False) -> Any:
        if isinstance(node, list):
            return [walk(x) for x in node]
        if not isinstance(node, dict):
            return node
        if is_properties:
            return {name: walk(sub) for name, sub in node.items()}
        if "$ref" in node:
            target = walk(defs[node["$ref"].rsplit("/", 1)[-1]])
            rest = {k: walk(v, k == "properties") for k, v in node.items() if k not in ("$ref", "title")}
            return {**target, **rest}
        return {
            ("anyOf" if k == "oneOf" else k): walk(v, k == "properties")
            for k, v in node.items()
            if k not in ("$defs", "title", "discriminator")
        }

    return walk(schema)


def model_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    return compact_schema(model_cls.model_json_schema())


def _format_errors(error: Exception) -> list[str]:
    if isinstance(error, ValidationError):
        return [
            f"{'.'.join(str(p) for p in e['loc']) or '(корень)'}: {e['msg']}" for e in error.errors()[:6]
        ]
    return [str(error)]


async def generate_structured(
    llm: LLMClient,
    messages: Sequence[ChatMessage],
    model_cls: type[T],
    *,
    schema: dict[str, Any] | None = None,
    schema_name: str | None = None,
    max_attempts: int = 3,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    preprocess: Callable[[Any], Any] | None = None,
    validate: Callable[[T], list[str]] | None = None,
    schema_in_prompt: bool | None = None,
) -> T:
    schema = schema or model_schema(model_cls)
    name = schema_name or model_cls.__name__
    schema_text = json.dumps(schema, ensure_ascii=False)
    if schema_in_prompt is None:
        schema_in_prompt = len(schema_text) <= MAX_PROMPT_SCHEMA_CHARS
    instruction = SCHEMA_INSTRUCTION.format(schema=schema_text) if schema_in_prompt else SHORT_INSTRUCTION
    convo = list(messages)
    if convo and convo[0].role == "system":
        convo[0] = system_msg(convo[0].content + "\n\n" + instruction)
    else:
        convo.insert(0, system_msg(instruction))

    raw, errors = "", []
    for attempt in range(max_attempts):
        reply = await llm.chat(
            convo,
            json_schema=schema,
            schema_name=name,
            temperature=temperature if attempt == 0 else max(temperature, 0.3),
            max_tokens=max_tokens,
        )
        raw = reply.content
        try:
            data = extract_json(raw)
            if preprocess is not None:
                data = preprocess(data)
            obj = model_cls.model_validate(data)
            errors = validate(obj) if validate is not None else []
            if not errors:
                return obj
        except ValueError as e:
            errors = _format_errors(e)
        log.info("structured output %s rejected (attempt %d/%d): %s", name, attempt + 1, max_attempts, errors)
        convo = [*convo, assistant_msg(raw), user_msg(REPAIR_PROMPT.format(errors="\n".join(f"- {e}" for e in errors)))]
    raise StructuredOutputError(f"модель не вернула валидный {name}", raw=raw, errors=errors)
