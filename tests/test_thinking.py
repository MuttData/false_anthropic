import base64

import pytest
from google.genai import types

from false_anthropic import FalseAnthropic

THOUGHT_1 = "**Counting letters**\n\nLet me spell it out. "
THOUGHT_2 = "s-t-r-a-w-b-e-r-r-y has three r."
ANSWER = "There are 3 r's in strawberry."
SIGNATURE = b"gemini-signature"

REDACTED_TRIGGER = (
    "ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB"
)


def _response(parts: list[types.Part], finish_reason=None) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        response_id="abc",
        model_version="gemini-test",
        candidates=[types.Candidate(finish_reason=finish_reason, content=types.Content(role="model", parts=parts))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10, candidates_token_count=20, thoughts_token_count=30
        ),
    )


PARTS = [
    types.Part(text=THOUGHT_1, thought=True, thought_signature=SIGNATURE),
    types.Part(text=THOUGHT_2, thought=True),
    types.Part(text=ANSWER),
]
RESPONSE = _response(PARTS, types.FinishReason.STOP)
# the same response as Gemini streams it, one part per chunk
CHUNKS = [_response([part]) for part in PARTS[:-1]] + [_response(PARTS[-1:], types.FinishReason.STOP)]


class FakeModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append((contents, config))
        return RESPONSE

    def generate_content_stream(self, model, contents, config):
        self.calls.append((contents, config))
        return iter(CHUNKS)


@pytest.fixture
def client():
    client = FalseAnthropic(api_key="fake")
    client.messages.client = type("FakeClient", (), {"models": FakeModels()})()
    return client


def _create(client, content="How many r in strawberry?", **kwargs):
    return client.messages.create(
        model="gemini-test", max_tokens=4000, messages=[{"role": "user", "content": content}], **kwargs
    )


def _stream(client, content="How many r in strawberry?", **kwargs):
    with client.messages.stream(
        model="gemini-test", max_tokens=4000, messages=[{"role": "user", "content": content}], **kwargs
    ) as stream:
        events = list(stream)
        return events, stream.get_final_message()


ENABLED = {"type": "enabled", "budget_tokens": 1024}


@pytest.mark.parametrize(
    ("thinking", "expected"),
    [
        (ENABLED, types.ThinkingConfig(include_thoughts=True, thinking_budget=1024)),
        ({"type": "adaptive"}, types.ThinkingConfig(include_thoughts=True, thinking_budget=-1)),
        ({"type": "disabled"}, None),
        (None, None),
    ],
)
def test_request_thinking_config(client, thinking, expected):
    _create(client, thinking=thinking)
    _, config = client.messages.client.models.calls[0]
    assert config.thinking_config == expected


def test_thinking_block(client):
    message = _create(client, thinking=ENABLED)
    thinking, text = message.content
    assert thinking.type == "thinking"
    assert thinking.thinking == THOUGHT_1 + THOUGHT_2
    assert thinking.signature == base64.b64encode(SIGNATURE).decode()
    assert text.type == "text" and text.text == ANSWER
    assert message.usage.output_tokens == 50  # thinking tokens count as output, like Anthropic


def test_thoughts_are_hidden_without_thinking(client):
    message = _create(client)
    assert [block.type for block in message.content] == ["text"]


def test_omitted_display(client):
    thinking, _ = _create(client, thinking={**ENABLED, "display": "omitted"}).content
    assert thinking.type == "thinking" and thinking.thinking == "" and thinking.signature


def test_magic_string_returns_redacted_thinking(client):
    redacted, text = _create(client, content=REDACTED_TRIGGER, thinking=ENABLED).content
    assert redacted.type == "redacted_thinking" and redacted.data
    assert text.text == ANSWER


@pytest.mark.parametrize("content", ["How many r in strawberry?", REDACTED_TRIGGER])
def test_stream_matches_create(client, content):
    expected = _create(client, content=content, thinking=ENABLED)
    events, final = _stream(client, content=content, thinking=ENABLED)

    assert [block.type for block in final.content] == [block.type for block in expected.content]
    if expected.content[0].type == "thinking":
        assert final.content[0].thinking == expected.content[0].thinking
        assert final.content[0].signature == expected.content[0].signature
        delta_types = [e.delta.type for e in events if e.type == "content_block_delta" and e.index == 0]
        assert delta_types == ["thinking_delta", "thinking_delta", "signature_delta"]
    assert final.content[1].text == ANSWER


def test_thinking_blocks_in_history_are_skipped(client):
    message = _create(client, thinking=ENABLED)
    client.messages.create(
        model="gemini-test",
        max_tokens=4000,
        messages=[
            {"role": "user", "content": "How many r in strawberry?"},
            {"role": "assistant", "content": message.content},
            {"role": "user", "content": "And in raspberry?"},
        ],
        thinking=ENABLED,
    )
    contents, _ = client.messages.client.models.calls[-1]
    assert [part.text for part in contents[1].parts] == [ANSWER]
