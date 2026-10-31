# false_anthropic

Anthropic SDK compatible interface for the Google GenAI SDK, so we can do the Anthropic certification exercises using a free tier Gemini account.

`FalseAnthropic` is a drop-in replacement for `anthropic.Anthropic`: you write the exercises with the Anthropic API shapes (messages, content blocks, tools, streaming events...), and the requests are sent to Gemini.

## Installation

With [uv](https://docs.astral.sh/uv/), straight from the repository:

```bash
uv add "false-anthropic @ git+https://github.com/MuttData/false_anthropic"
# or, with SSH access
uv add "false-anthropic @ git+ssh://git@github.com/MuttData/false_anthropic.git"
```

Pin a branch, tag or commit by appending `@<ref>` to the URL (e.g. `...false_anthropic@v0.1.0`).

With pip: `pip install "git+https://github.com/MuttData/false_anthropic"`.

## Setup

Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey) and expose it as `GEMINI_API_KEY` (or `GOOGLE_API_KEY`), e.g. in a `.env` file loaded with `python-dotenv`:

```
GEMINI_API_KEY="your-key"
```

You can also pass it explicitly: `FalseAnthropic(api_key="...")`.

## Usage

Replace the Anthropic client and use a Gemini model name; the rest of the exercise code stays the same:

```python
from false_anthropic import FalseAnthropic

client = FalseAnthropic()  # instead of anthropic.Anthropic()
model = "gemini-3.5-flash-lite"  # instead of a claude-* model

message = client.messages.create(
    model=model,
    max_tokens=1000,
    system="You are a helpful assistant",
    messages=[{"role": "user", "content": "Hi"}],
)
print(message.content[0].text)
```

See [`examples/`](examples/) for tool use, streaming and structured outputs. To run them from a clone:

```bash
uv sync
uv run examples/tool_use.py
```

## Supported

| Feature | Notes |
| --- | --- |
| `messages.create` | `model`, `max_tokens`, `messages`, `system`, `temperature`, `stop_sequences` |
| Multi-turn chats | `user` / `assistant` roles, string or text block content, `response.content` can be appended back to the history |
| Prefill | ending `messages` with an assistant message makes the model continue from it (emulated, see below) |
| Tool use | `tools`, `tool_choice` (`auto`, `any`, `tool`, `none`), `tool_use` / `tool_result` blocks, parallel tool calls, `is_error` |
| Streaming | `create(..., stream=True)` raw events and the `messages.stream()` helper (`text_stream`, `get_final_message()`, ...) |
| Structured outputs | `output_config={"format": {"type": "json_schema", ...}}`, `messages.parse(output_format=Model)` and `messages.stream(output_format=Model)` |
| MCP | client-side MCP (listing an MCP server's tools and passing them as `tools`), since it is regular tool use |

Responses are real `anthropic.types.Message` objects, with `stop_reason` and `usage` mapped from the Gemini response.

## Differences and limitations

- Gemini rejects requests ending with an assistant message, so prefill is emulated: a user turn asking the model to continue is added, any repeated prefill text is stripped from the output, and stop sequences are applied by the library (so `stop_reason="stop_sequence"` and `stop_sequence` are reported). It is reliable for typical cases like prefilling ` ```json ` with stop sequence ` ``` `, but it is the model following an instruction, not a true continuation.
- Without a prefill, `stop_reason` is never `"stop_sequence"` and `stop_sequence` is always `None`: Gemini does not report when a stop sequence was hit (the output is still cut).
- Gemini streams in bigger chunks than Claude, and tool inputs arrive whole in a single `input_json` delta.
- Structured output schemas are sent to Gemini as they are, so constraints like `minimum` / `maximum` are enforced by the model.
- `output_config["effort"]` and `tool_choice["disable_parallel_tool_use"]` are ignored.
- Not supported yet: images / PDFs / documents, extended thinking, prompt caching, server tools (web search, code execution...), the `mcp_servers` connector, `count_tokens`, batches, the async client.
- Gemini 3 models need a "thought signature" when a tool call is sent back. It is kept as an extra `gemini_thought_signature` field on the returned `ToolUseBlock`; for hand-written history a placeholder is sent instead.
