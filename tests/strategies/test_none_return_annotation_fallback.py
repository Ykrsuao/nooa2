# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``-> None`` keeps meaning "returns nothing" when type hints cannot resolve.

``get_type_hints`` fails on an unresolvable parameter annotation, so the return
type falls back to the raw signature: a bare ``None``, or the string ``"None"``
under PEP 563. Neither may be read as a missing annotation or as a type named
"None". This module deliberately has no ``from __future__ import annotations``
so that both raw forms can be written.
"""

import json

import pytest

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.strategies.codeact import CodeActStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


def _return_nothing() -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[ToolCall(id="c1", name="return_result", arguments=json.dumps({}))],
        finish_reason="tool_calls",
    )


class _BareNone(Agent, llm=FakeLLMClient()):
    @strategy(CodeActStrategy(config=CodeActConfig()))
    async def record(self, note: "UndefinedNoteType") -> None:  # type: ignore[name-defined]  # noqa: F821
        """Record {note}."""
        ...


class _StringNone(Agent, llm=FakeLLMClient()):
    @strategy(CodeActStrategy(config=CodeActConfig()))
    async def record(self, note: "UndefinedNoteType") -> "None":  # type: ignore[name-defined]  # noqa: F821
        """Record {note}."""
        ...


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_cls", [_BareNone, _StringNone], ids=["bare", "pep563-string"])
async def test_none_return_survives_unresolvable_annotations(agent_cls):
    fake = FakeLLMClient(scripted_responses=[_return_nothing()])
    assert await agent_cls(llm=fake).record("n") is None
    assert fake.call_count == 1
