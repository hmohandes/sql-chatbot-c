from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings, get_settings
from app.prompts import QUERY_CHECKER_PROMPT, build_system_prompt
from app.tools import TOOL_SCHEMAS, build_tool_registry, dispatch_tool

logger = logging.getLogger(__name__)

_TOOL_CALL_RE = re.compile(
    r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>",
    re.DOTALL,
)
_ARG_RE = re.compile(r"(\w+):(?:<\|\"\|>(.*?)<\|\"\|>|([^,}]*))", re.DOTALL)


@dataclass
class AgentStep:
    step: int
    kind: str
    payload: dict[str, Any]


@dataclass
class AgentResult:
    success: bool
    result: str
    error: str | None = None
    steps: list[AgentStep] = field(default_factory=list)


class GemmaSqlAgent:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.processor = None
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return

        from transformers import AutoProcessor

        token = self.settings.hf_token or None
        model_id = self.settings.model_id
        logger.info("Loading model %s", model_id)
        self.processor = AutoProcessor.from_pretrained(model_id, token=token)
        try:
            from transformers import AutoModelForMultimodalLM

            model_cls = AutoModelForMultimodalLM
        except ImportError:
            from transformers import AutoModelForImageTextToText

            model_cls = AutoModelForImageTextToText
        self.model = model_cls.from_pretrained(
            model_id,
            dtype="auto",
            device_map="auto",
            token=token,
        )
        self.model.eval()

    def run(self, question: str, debug: bool = False) -> AgentResult:
        self.load()
        steps: list[AgentStep] = []
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": build_system_prompt(self.settings.dialect, self.settings.top_k),
            },
            {"role": "user", "content": question},
        ]
        registry = build_tool_registry(self.check_query)

        last_error: str | None = None
        for iteration in range(1, self.settings.max_agent_steps + 1):
            parsed = self._generate(messages)
            content = (parsed.get("content") or "").strip()
            tool_calls = parsed.get("tool_calls") or []
            thinking = parsed.get("thinking") or parsed.get("reasoning")

            step_payload: dict[str, Any] = {
                "iteration": iteration,
                "content": content,
                "tool_calls": tool_calls,
            }
            if thinking:
                step_payload["thinking"] = thinking
            self._record(steps, debug, iteration, "model", step_payload)

            if not tool_calls:
                if content:
                    return AgentResult(success=True, result=content, steps=steps)
                last_error = "The model returned an empty response."
                break

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "tool_calls": [],
                "tool_responses": [],
            }
            if thinking:
                assistant_message["reasoning"] = thinking
            if content:
                assistant_message["content"] = content

            for call in tool_calls:
                name, arguments = _normalize_tool_call(call)
                try:
                    output = dispatch_tool(name, arguments, registry)
                except Exception as exc:
                    output = f"Error: {exc}"

                self._record(
                    steps,
                    debug,
                    iteration,
                    "tool",
                    {"name": name, "arguments": arguments, "output": output},
                )
                if output.startswith("Error:"):
                    last_error = output

                assistant_message["tool_calls"].append(
                    {"function": {"name": name, "arguments": arguments}}
                )
                assistant_message["tool_responses"].append(
                    {"name": name, "response": output}
                )

            messages.append(assistant_message)

        error = last_error or "The agent exceeded the maximum number of steps without an answer."
        return AgentResult(success=False, result=error, error=error, steps=steps)

    def check_query(self, query: str) -> str:
        prompt = QUERY_CHECKER_PROMPT.format(query=query.strip())
        messages = [
            {
                "role": "system",
                "content": "You rewrite SQL queries. Output only the final PostgreSQL query.",
            },
            {"role": "user", "content": prompt},
        ]
        parsed = self._generate(messages, use_tools=False)
        text = (parsed.get("content") or "").strip()
        return _extract_sql(text) or query.strip()

    def _generate(self, messages: list[dict[str, Any]], use_tools: bool = True) -> dict[str, Any]:
        assert self.processor is not None and self.model is not None
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "add_generation_prompt": True,
            "enable_thinking": self.settings.enable_thinking,
        }
        if use_tools:
            kwargs["tools"] = TOOL_SCHEMAS

        try:
            inputs = self.processor.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            inputs = self.processor.apply_chat_template(messages, **kwargs)

        if not hasattr(inputs, "to"):
            inputs = self.processor(text=inputs, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        import torch

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.settings.max_new_tokens,
            )

        raw = self.processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        return self._parse_response(raw, inputs["input_ids"])

    def _parse_response(self, raw: str, prefix: Any) -> dict[str, Any]:
        parsed: dict[str, Any] | None = None
        try:
            parsed = self.processor.parse_response(raw, prefix=prefix)
        except TypeError:
            try:
                parsed = self.processor.parse_response(raw, prefix="")
            except Exception:
                parsed = None
        except Exception:
            parsed = None

        if isinstance(parsed, dict) and (parsed.get("tool_calls") or parsed.get("content")):
            return parsed

        fallback = _fallback_parse(raw)
        if parsed and isinstance(parsed, dict):
            fallback.setdefault("thinking", parsed.get("thinking") or parsed.get("reasoning"))
        return fallback

    def _record(
        self,
        steps: list[AgentStep],
        debug: bool,
        step: int,
        kind: str,
        payload: dict[str, Any],
    ) -> None:
        record = AgentStep(step=step, kind=kind, payload=payload)
        steps.append(record)
        if debug:
            logger.info("agent step %s [%s]: %s", step, kind, json.dumps(payload, default=str)[:4000])


def _normalize_tool_call(call: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(call, dict):
        function = call.get("function") if "function" in call else call
        name = function.get("name") if isinstance(function, dict) else call.get("name")
        arguments = {}
        if isinstance(function, dict):
            arguments = function.get("arguments") or {}
        elif "arguments" in call:
            arguments = call.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"query": arguments}
        return str(name or ""), dict(arguments or {})
    return str(call), {}


def _fallback_parse(raw: str) -> dict[str, Any]:
    tool_calls = []
    for name, args in _TOOL_CALL_RE.findall(raw):
        arguments: dict[str, Any] = {}
        for key, quoted, bare in _ARG_RE.findall(args):
            value = quoted if quoted else bare
            arguments[key] = _cast_arg(value.strip())
        tool_calls.append(
            {
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )

    content = raw
    content = re.sub(r"<\|tool_call>.*?<tool_call\|>", "", content, flags=re.DOTALL)
    content = re.sub(r"<\|channel>thought\n.*?<channel\|>", "", content, flags=re.DOTALL)
    content = re.sub(r"<[^>]+>", "", content)
    content = content.replace("<eos>", "").strip()
    thinking_match = re.search(
        r"<\|channel>thought\n(.*?)<channel\|>", raw, flags=re.DOTALL
    )
    result: dict[str, Any] = {"role": "assistant", "content": content, "tool_calls": tool_calls}
    if thinking_match:
        result["thinking"] = thinking_match.group(1).strip()
    return result


def _cast_arg(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value.strip("'\"")


def _extract_sql(text: str) -> str:
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


_agent: GemmaSqlAgent | None = None


def get_agent() -> GemmaSqlAgent:
    global _agent
    if _agent is None:
        _agent = GemmaSqlAgent()
    return _agent
