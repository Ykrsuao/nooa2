# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Method-local strategy validators (ported from the !444 design).

Preconditions/postconditions live on the strategy *config* (method-local, not an
agent-global hook). A postcondition raises ``InvariantError`` to reject a
type-valid result — routed through the existing return_result validation-retry
channel; any other exception surfaces as an infrastructure error. Preconditions
run before generation and fail fast.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from nooa import Agent, InvariantError, strategy
from nooa.config import CodeActConfig, PredictConfig
from nooa.strategies.codeact import CodeActStrategy
from nooa.strategies.predict import PredictStrategy
from nooa.strategy_validation import normalize_conditions, run_postconditions, run_preconditions
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


def _return_result_call(result, call_id: str = "call_ret") -> ToolCall:
    return ToolCall(id=call_id, name="return_result", arguments=json.dumps({"result": result}))


def _resp(tool_calls: list) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=tool_calls,
        finish_reason="tool_calls",
    )


_DUMMY_LLM = FakeLLMClient()

# --------------------------------------------------------------------------- #
# unit: strategy_validation helpers
# --------------------------------------------------------------------------- #


def test_normalize_conditions_accepts_callables():
    f = lambda *a: None  # noqa: E731
    assert normalize_conditions("postconditions", [f]) == (f,)
    assert normalize_conditions("postconditions", None) == ()


def test_normalize_conditions_rejects_non_callable():
    with pytest.raises(ValueError):
        normalize_conditions("postconditions", [123])
    with pytest.raises(ValueError):
        normalize_conditions("postconditions", "not-iterable-of-callables")


def test_invariant_error_is_valueerror():
    assert issubclass(InvariantError, ValueError)


@pytest.mark.asyncio
async def test_run_postconditions_reraises_invariant_wraps_others():
    def raises_invariant(agent, result, call):
        raise InvariantError("bad")

    with pytest.raises(InvariantError):
        await run_postconditions(object(), 1, object(), [raises_invariant])

    def raises_other(agent, result, call):
        raise KeyError("boom")

    with pytest.raises(RuntimeError):  # non-InvariantError -> infra error
        await run_postconditions(object(), 1, object(), [raises_other])


@pytest.mark.asyncio
async def test_run_preconditions_passes_agent_and_call():
    seen = {}

    def pre(agent, call):
        seen["agent"], seen["call"] = agent, call

    a, c = object(), object()
    await run_preconditions(a, c, [pre])
    assert seen == {"agent": a, "call": c}


class _AsyncRejectBad:
    """An async callable object: no ``__name__``, ``async def __call__``."""

    async def __call__(self, agent, result, call):
        await asyncio.sleep(0)
        if result == "bad":
            raise InvariantError("'bad' is not acceptable; return 'good'.")


async def _async_raises_other(agent, result, call):
    await asyncio.sleep(0)
    raise KeyError("boom")


@pytest.mark.asyncio
async def test_async_conditions_are_awaited():
    """An async condition must actually run; before, its coroutine was never
    awaited and the check silently passed."""
    with pytest.raises(InvariantError):
        await run_postconditions(object(), "bad", object(), [_AsyncRejectBad()])
    await run_postconditions(object(), "good", object(), [_AsyncRejectBad()])

    with pytest.raises(RuntimeError, match="_async_raises_other"):
        await run_postconditions(object(), 1, object(), [_async_raises_other])


@pytest.mark.asyncio
async def test_failing_callable_object_is_named_in_the_error():
    """A callable object has no ``__name__``; reporting its failure must not
    itself fail with AttributeError and mask the real error."""

    class Broken:
        def __call__(self, agent, result, call):
            raise KeyError("boom")

    with pytest.raises(RuntimeError, match="Broken") as info:
        await run_postconditions(object(), 1, object(), [Broken()])
    assert isinstance(info.value.__cause__, KeyError)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def test_config_normalizes_conditions():
    f = lambda *a: None  # noqa: E731
    cfg = CodeActConfig(preconditions=[f], postconditions=[f])
    assert cfg.preconditions == (f,) and cfg.postconditions == (f,)
    with pytest.raises(ValueError):
        CodeActConfig(postconditions=[1])


# --------------------------------------------------------------------------- #
# end-to-end through CodeActStrategy
# --------------------------------------------------------------------------- #


def _post_reject_bad(agent, result, call):
    if result == "bad":
        raise InvariantError("'bad' is not acceptable; return 'good'.")


class _ValidatedAgent(Agent, llm=_DUMMY_LLM):
    @strategy(CodeActStrategy(config=CodeActConfig(postconditions=[_post_reject_bad])))
    async def solve(self, question: str) -> str:
        """Answer {question}."""
        ...


@pytest.mark.asyncio
async def test_postcondition_rejects_then_accepts_on_retry():
    fake = FakeLLMClient(
        scripted_responses=[
            _resp([_return_result_call("bad", "c1")]),  # InvariantError -> retry
            _resp([_return_result_call("good", "c2")]),
        ]
    )
    agent = _ValidatedAgent(llm=fake)
    assert await agent.solve("q") == "good"
    assert fake.call_count == 2  # one retry for the rejected result


@pytest.mark.asyncio
async def test_no_violation_passes_first_try():
    fake = FakeLLMClient(scripted_responses=[_resp([_return_result_call("good")])])
    agent = _ValidatedAgent(llm=fake)
    assert await agent.solve("q") == "good"
    assert fake.call_count == 1


class _PlainAgent(Agent, llm=_DUMMY_LLM):
    @strategy(CodeActStrategy(config=CodeActConfig()))
    async def solve(self, question: str) -> str:
        """Answer {question}."""
        ...


@pytest.mark.asyncio
async def test_no_postconditions_configured_is_unaffected():
    fake = FakeLLMClient(scripted_responses=[_resp([_return_result_call("bad")])])
    agent = _PlainAgent(llm=fake)
    assert await agent.solve("q") == "bad"  # no postcondition -> no rejection
    assert fake.call_count == 1


def _post_buggy(agent, result, call):
    raise TypeError("buggy postcondition")


class _BuggyAgent(Agent, llm=_DUMMY_LLM):
    @strategy(CodeActStrategy(config=CodeActConfig(postconditions=[_post_buggy])))
    async def solve(self, question: str) -> str:
        """Answer {question}."""
        ...


@pytest.mark.asyncio
async def test_buggy_postcondition_surfaces_as_infra_error_not_retry():
    """A postcondition that raises a non-InvariantError must not be swallowed as
    model-correctable feedback (it's an infra/programming error)."""
    fake = FakeLLMClient(scripted_responses=[_resp([_return_result_call("anything")])])
    agent = _BuggyAgent(llm=fake)
    with pytest.raises(RuntimeError, match="postcondition"):
        await agent.solve("q")
    assert fake.call_count == 1  # no spurious retry loop


def _post_cancel(agent, result, call):
    raise asyncio.CancelledError


@pytest.mark.asyncio
async def test_return_result_cancellation_is_reported_to_tool_hook() -> None:
    from nooa.runtime.hooks import set_hooks

    exceptions: list[BaseException | None] = []

    class Hooks:
        def before_tool_execution(self, **kwargs):
            return "return-result-context"

        def after_tool_execution(self, *, context, exception, **kwargs):
            assert context == "return-result-context"
            exceptions.append(exception)

    class CancellingAgent(Agent, llm=_DUMMY_LLM):
        @strategy(CodeActStrategy(config=CodeActConfig(postconditions=[_post_cancel])))
        async def solve(self) -> str:
            """Return a result."""
            ...

    set_hooks(Hooks())  # type: ignore[arg-type]
    try:
        agent = CancellingAgent(
            llm=FakeLLMClient(scripted_responses=[_resp([_return_result_call("done")])])
        )
        with pytest.raises(asyncio.CancelledError):
            await agent.solve()
    finally:
        set_hooks(None)

    assert len(exceptions) == 1
    assert isinstance(exceptions[0], asyncio.CancelledError)


# --------------------------------------------------------------------------- #
# explicit `return x` in a cell is a completion attempt too
# --------------------------------------------------------------------------- #


def _cell_call(code: str, call_id: str) -> ToolCall:
    return ToolCall(id=call_id, name="execute_python", arguments=json.dumps({"code": code}))


@pytest.mark.asyncio
async def test_explicit_return_is_held_to_postconditions():
    fake = FakeLLMClient(
        scripted_responses=[
            _resp([_cell_call("return 'bad'", "c1")]),  # InvariantError -> feedback
            _resp([_cell_call("return 'good'", "c2")]),
        ]
    )
    agent = _ValidatedAgent(llm=fake)
    assert await agent.solve("q") == "good"
    assert fake.call_count == 2
    second_request = json.dumps(fake.calls[1].messages, default=str)
    assert "is not acceptable; return" in second_request


@pytest.mark.asyncio
async def test_explicit_return_type_mismatch_stays_silent():
    class _IntAgent(Agent, llm=_DUMMY_LLM):
        @strategy(CodeActStrategy(config=CodeActConfig(postconditions=[_post_reject_bad])))
        async def count(self) -> int:
            """Count."""
            ...

    fake = FakeLLMClient(
        scripted_responses=[
            _resp([_cell_call("return 'not a number'", "c1")]),  # not a completion attempt
            _resp([_return_result_call(3, "c2")]),
        ]
    )
    agent = _IntAgent(llm=fake)
    assert await agent.count() == 3
    second_request = json.dumps(fake.calls[1].messages, default=str)
    assert "validation error" not in second_request


@pytest.mark.asyncio
async def test_explicit_return_buggy_postcondition_is_infra_error():
    fake = FakeLLMClient(scripted_responses=[_resp([_cell_call("return 'anything'", "c1")])])
    agent = _BuggyAgent(llm=fake)
    with pytest.raises(RuntimeError, match="postcondition"):
        await agent.solve("q")
    assert fake.call_count == 1


# --------------------------------------------------------------------------- #
# PredictStrategy honours the same conditions
# --------------------------------------------------------------------------- #


def _predict_resp(value: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content=json.dumps({"value": value}),
        tool_calls=[],
        finish_reason="stop",
    )


class _PredictValidatedAgent(Agent, llm=_DUMMY_LLM):
    @strategy(
        PredictStrategy(config=PredictConfig(max_retries=3, postconditions=[_post_reject_bad]))
    )
    async def solve(self, question: str) -> str:
        """Answer {question}."""
        ...


@pytest.mark.asyncio
async def test_predict_postcondition_rejects_then_accepts_on_retry():
    fake = FakeLLMClient(scripted_responses=[_predict_resp("bad"), _predict_resp("good")])
    agent = _PredictValidatedAgent(llm=fake)
    assert await agent.solve("q") == "good"
    assert fake.call_count == 2
    assert "is not acceptable; return" in json.dumps(fake.calls[1].messages, default=str)


@pytest.mark.asyncio
async def test_predict_postcondition_exhausts_retries():
    from nooa.errors import GenerationError

    fake = FakeLLMClient(scripted_responses=[_predict_resp("bad")] * 3)
    agent = _PredictValidatedAgent(llm=fake)
    with pytest.raises(GenerationError, match="is not acceptable"):
        await agent.solve("q")
    assert fake.call_count == 3


@pytest.mark.asyncio
async def test_predict_buggy_postcondition_is_infra_error():
    class _Agent(Agent, llm=_DUMMY_LLM):
        @strategy(PredictStrategy(config=PredictConfig(postconditions=[_post_buggy])))
        async def solve(self) -> str:
            """Answer."""
            ...

    fake = FakeLLMClient(scripted_responses=[_predict_resp("anything")])
    with pytest.raises(RuntimeError, match="postcondition"):
        await _Agent(llm=fake).solve()
    assert fake.call_count == 1


@pytest.mark.asyncio
async def test_predict_precondition_fails_fast_without_llm_call():
    def _pre_refuse(agent, call):
        raise InvariantError("not ready")

    class _Agent(Agent, llm=_DUMMY_LLM):
        @strategy(PredictStrategy(config=PredictConfig(preconditions=[_pre_refuse])))
        async def solve(self) -> str:
            """Answer."""
            ...

    fake = FakeLLMClient(scripted_responses=[_predict_resp("anything")])
    with pytest.raises(InvariantError, match="not ready"):
        await _Agent(llm=fake).solve()
    assert fake.call_count == 0


# --------------------------------------------------------------------------- #
# async conditions are enforced end to end
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_async_postcondition_enforced_by_codeact():
    class _Agent(Agent, llm=_DUMMY_LLM):
        @strategy(CodeActStrategy(config=CodeActConfig(postconditions=[_AsyncRejectBad()])))
        async def solve(self) -> str:
            """Answer."""
            ...

    fake = FakeLLMClient(
        scripted_responses=[
            _resp([_return_result_call("bad", "c1")]),
            _resp([_cell_call("return 'bad'", "c2")]),
            _resp([_return_result_call("good", "c3")]),
        ]
    )
    assert await _Agent(llm=fake).solve() == "good"
    assert fake.call_count == 3  # both 'bad' attempts were rejected


@pytest.mark.asyncio
async def test_async_postcondition_enforced_by_predict():
    class _Agent(Agent, llm=_DUMMY_LLM):
        @strategy(PredictStrategy(config=PredictConfig(postconditions=[_AsyncRejectBad()])))
        async def solve(self) -> str:
            """Answer."""
            ...

    fake = FakeLLMClient(scripted_responses=[_predict_resp("bad"), _predict_resp("good")])
    assert await _Agent(llm=fake).solve() == "good"
    assert fake.call_count == 2


@pytest.mark.asyncio
async def test_async_precondition_fails_fast():
    async def _pre_refuse(agent, call):
        await asyncio.sleep(0)
        raise InvariantError("not ready")

    class _Agent(Agent, llm=_DUMMY_LLM):
        @strategy(CodeActStrategy(config=CodeActConfig(preconditions=[_pre_refuse])))
        async def solve(self) -> str:
            """Answer."""
            ...

    fake = FakeLLMClient(scripted_responses=[_resp([_return_result_call("x")])])
    with pytest.raises(InvariantError, match="not ready"):
        await _Agent(llm=fake).solve()
    assert fake.call_count == 0
