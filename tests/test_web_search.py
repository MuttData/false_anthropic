import pytest
from google.genai import types

from false_anthropic import FalseAnthropic
from false_anthropic.client import FalseAnthropicMessages

WEB_SEARCH = {"type": "web_search_20250305", "name": "web_search"}

SENTENCE_1 = "Max Verstappen ganó el campeonato de 2024."
SENTENCE_2 = "Fue su cuarto título."
ANSWER = f"{SENTENCE_1} {SENTENCE_2} ¡Increíble!"


def _segment(sentence: str) -> types.Segment:
    # Gemini's offsets are UTF-8 byte offsets within the part
    start = ANSWER.encode().index(sentence.encode())
    return types.Segment(part_index=1, start_index=start, end_index=start + len(sentence.encode()), text=sentence)


GROUNDED_RESPONSE = types.GenerateContentResponse(
    response_id="abc",
    model_version="gemini-test",
    candidates=[
        types.Candidate(
            finish_reason=types.FinishReason.STOP,
            content=types.Content(
                role="model",
                parts=[types.Part(text="Let me look this up", thought=True), types.Part(text=ANSWER)],
            ),
            grounding_metadata=types.GroundingMetadata(
                web_search_queries=["F1 champion 2024", "Verstappen titles"],
                grounding_chunks=[
                    types.GroundingChunk(web=types.GroundingChunkWeb(uri="https://example.com/a", title="example.com")),
                    types.GroundingChunk(web=types.GroundingChunkWeb(uri="https://example.org/b", title="example.org")),
                ],
                grounding_supports=[
                    types.GroundingSupport(segment=_segment(SENTENCE_1), grounding_chunk_indices=[0, 1]),
                    types.GroundingSupport(segment=_segment(SENTENCE_2), grounding_chunk_indices=[1]),
                ],
            ),
        )
    ],
    usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=10, candidates_token_count=20),
)


class FakeModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append((contents, config))
        return GROUNDED_RESPONSE


@pytest.fixture
def client():
    client = FalseAnthropic(api_key="fake")
    client.messages.client = type("FakeClient", (), {"models": FakeModels()})()
    return client


def _create(client, **kwargs):
    return client.messages.create(
        model="gemini-test",
        max_tokens=100,
        messages=[{"role": "user", "content": "Who won F1 in 2024?"}],
        tools=[WEB_SEARCH],
        **kwargs,
    )


def test_request_uses_google_search(client):
    _create(client)
    _, config = client.messages.client.models.calls[0]
    assert config.tools[0].google_search is not None


def test_unsupported_options_warn():
    with pytest.warns(UserWarning, match="allowed_domains"):
        FalseAnthropicMessages(client=None)._from_anthropic_to_genai_config(
            max_tokens=100,
            system=None,
            temperature=None,
            stop_sequences=None,
            tools=[{**WEB_SEARCH, "allowed_domains": ["example.com"]}],
            tool_choice=None,
            output_config=None,
        )


def test_response_has_search_blocks_and_citations(client):
    message = _create(client)

    assert [block.type for block in message.content] == [
        "server_tool_use", "web_search_tool_result", "server_tool_use", "web_search_tool_result",
        "text", "text", "text", "text",
    ]
    search, result = message.content[0], message.content[1]
    assert search.name == "web_search" and search.input == {"query": "F1 champion 2024"}
    assert result.tool_use_id == search.id
    assert [r.url for r in result.content] == ["https://example.com/a", "https://example.org/b"]

    texts = message.content[4:]
    assert "".join(block.text for block in texts) == ANSWER  # the thought part is left out
    assert [block.text for block in texts] == [SENTENCE_1, " ", SENTENCE_2, " ¡Increíble!"]
    assert [c.url for c in texts[0].citations] == ["https://example.com/a", "https://example.org/b"]
    assert texts[0].citations[0].cited_text == SENTENCE_1
    assert texts[1].citations is None
    assert message.usage.server_tool_use.web_search_requests == 2
    assert message.stop_reason == "end_turn"


def test_stream_replays_the_same_message(client):
    expected = _create(client)
    with client.messages.stream(
        model="gemini-test",
        max_tokens=100,
        messages=[{"role": "user", "content": "Who won F1 in 2024?"}],
        tools=[WEB_SEARCH],
    ) as stream:
        events = list(stream)
        final = stream.get_final_message()

    assert "citations_delta" in [e.delta.type for e in events if e.type == "content_block_delta"]
    assert [b.type for b in final.content] == [b.type for b in expected.content]
    assert [b.text for b in final.content if b.type == "text"] == [b.text for b in expected.content if b.type == "text"]
    assert final.content[4].citations[1].url == "https://example.org/b"
    assert final.content[0].input == {"query": "F1 champion 2024"}
    assert final.usage.server_tool_use.web_search_requests == 2


def test_search_blocks_in_history_are_skipped(client):
    message = _create(client)
    client.messages.create(
        model="gemini-test",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "Who won F1 in 2024?"},
            {"role": "assistant", "content": message.content},
            {"role": "user", "content": "And in 2023?"},
        ],
        tools=[WEB_SEARCH],
    )
    contents, _ = client.messages.client.models.calls[-1]
    assistant = contents[1]
    assert assistant.role == "model"
    assert "".join(part.text for part in assistant.parts) == ANSWER
