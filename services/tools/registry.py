"""Decorator-based tool registry - the single choke point every tool call
goes through, so idempotency, timeouts, and ToolInvocation logging are
enforced once instead of per-tool. Individual tool functions in
roster_tools.py / hr_tools.py / transfer_tools.py / comms_tools.py never
implement any of that themselves; they just do the one thing they're named
for and return a plain dict.
"""

import asyncio
import hashlib
import inspect
import json
import time
from dataclasses import dataclass, field, is_dataclass, asdict
from functools import lru_cache
from typing import Any, Awaitable, Callable

from pydantic import BaseModel

DEFAULT_TOOL_TIMEOUT_MS = 1200


@lru_cache
def _signature_params(func: Callable) -> frozenset[str]:
    return frozenset(inspect.signature(func).parameters.keys())


class ToolTimeoutError(Exception):
    pass


class ToolNotFoundError(Exception):
    pass


@dataclass
class ToolSpec:
    name: str
    func: Callable[..., Awaitable[Any]]
    idempotent: bool = False
    terminal: bool = False
    timeout_ms: int = DEFAULT_TOOL_TIMEOUT_MS
    requires_slots: tuple[str, ...] = field(default_factory=tuple)


_REGISTRY: dict[str, ToolSpec] = {}


def tool(
    *,
    idempotent: bool = False,
    terminal: bool = False,
    timeout_ms: int = DEFAULT_TOOL_TIMEOUT_MS,
    requires_slots: tuple[str, ...] = (),
):
    """`terminal=True` tools (transfer_to_human, transfer_to_agent, end_call)
    end the conversation loop's turn-taking once invoked - the conversation
    agent (services/agents/conversation_agent.py) checks this flag on the
    ToolSpec, not on a per-tool-name allowlist."""

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        _REGISTRY[func.__name__] = ToolSpec(
            name=func.__name__,
            func=func,
            idempotent=idempotent,
            terminal=terminal,
            timeout_ms=timeout_ms,
            requires_slots=requires_slots,
        )
        return func

    return decorator


def get_tool(name: str) -> ToolSpec:
    if name not in _REGISTRY:
        raise ToolNotFoundError(f"No tool registered as {name!r}")
    return _REGISTRY[name]


def list_tools() -> list[str]:
    return sorted(_REGISTRY.keys())


_RESERVED_PARAMS = {"session", "call_id"}
_PY_TYPE_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    dict: "object",
    list: "array",
}


def _json_type_for(annotation: Any) -> dict[str, Any]:
    import typing

    origin = typing.get_origin(annotation)
    if origin is typing.Union:  # covers `X | None`
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        return _json_type_for(args[0]) if args else {"type": "string"}
    if origin in (list, tuple):
        (item_type,) = typing.get_args(annotation) or (str,)
        return {"type": "array", "items": _json_type_for(item_type)}
    if origin is dict:
        return {"type": "object"}
    return {"type": _PY_TYPE_TO_JSON.get(annotation, "string")}


def to_claude_tool_schema(name: str) -> dict[str, Any]:
    """Derives Claude's `tools=[...]` input schema from a tool function's
    type hints, so a new @tool doesn't also need a hand-written schema kept
    in sync by hand - one source of truth, same principle as the config
    registry's "zero Python for a sixth agent" rule."""
    spec = get_tool(name)
    sig = inspect.signature(spec.func)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for param_name, param in sig.parameters.items():
        if param_name in _RESERVED_PARAMS:
            continue
        properties[param_name] = _json_type_for(param.annotation)
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
    description = (spec.func.__doc__ or name).strip().split("\n")[0]
    return {
        "name": name,
        "description": description,
        "input_schema": {"type": "object", "properties": properties, "required": required},
    }


def canonical_idempotency_key(call_id: str, tool_name: str, args: dict[str, Any]) -> str:
    canonical = json.dumps(args, sort_keys=True, default=str)
    return hashlib.sha256(f"{call_id}:{tool_name}:{canonical}".encode()).hexdigest()


def _serialize_result(result: Any) -> dict[str, Any]:
    if isinstance(result, BaseModel):
        return result.model_dump(mode="json")
    if is_dataclass(result) and not isinstance(result, type):
        return asdict(result)
    if isinstance(result, dict):
        return result
    return {"value": result}


async def invoke_tool(
    name: str,
    *,
    call_id: str,
    session,
    turn_id: int | None = None,
    args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The only path a Conversation Agent tool_call should ever go through.
    `session` is an AsyncSession already inside the caller's transaction -
    invoke_tool does not commit; the caller's unit of work does.
    """
    import models

    args = args or {}
    spec = get_tool(name)
    idem_key = canonical_idempotency_key(call_id, name, args)

    if spec.idempotent:
        from sqlalchemy import select

        existing = await session.scalar(
            select(models.ToolInvocation).where(
                models.ToolInvocation.idempotency_key == idem_key,
                models.ToolInvocation.status == "ok",
            )
        )
        if existing is not None:
            return existing.result_json or {}

    invocation = models.ToolInvocation(
        call_id=call_id,
        turn_id=turn_id,
        tool_name=name,
        args_json=args,
        idempotency_key=idem_key,
        status="pending",
    )
    session.add(invocation)
    await session.flush()

    start = time.monotonic()
    call_kwargs = dict(args)
    func_params = _signature_params(spec.func)
    if "session" in func_params:
        call_kwargs["session"] = session
    if "call_id" in func_params:
        call_kwargs["call_id"] = call_id

    try:
        result = await asyncio.wait_for(spec.func(**call_kwargs), timeout=spec.timeout_ms / 1000)
        invocation.status = "ok"
        invocation.result_json = _serialize_result(result)
        return invocation.result_json
    except asyncio.TimeoutError as e:
        invocation.status = "timeout"
        invocation.result_json = {"error": f"timed out after {spec.timeout_ms}ms"}
        raise ToolTimeoutError(f"Tool {name!r} exceeded {spec.timeout_ms}ms") from e
    except Exception as e:
        invocation.status = "error"
        invocation.result_json = {"error": str(e)}
        raise
    finally:
        invocation.duration_ms = int((time.monotonic() - start) * 1000)
