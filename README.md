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

The tests run offline: `uv run pytest`.

## Supported

| Feature | Notes |
| --- | --- |
| `messages.create` | `model`, `max_tokens`, `messages`, `system`, `temperature`, `stop_sequences` |
| Multi-turn chats | `user` / `assistant` roles, string or text block content, `response.content` can be appended back to the history |
| Prefill | ending `messages` with an assistant message makes the model continue from it (emulated, see below) |
| Tool use | `tools`, `tool_choice` (`auto`, `any`, `tool`, `none`), `tool_use` / `tool_result` blocks, parallel tool calls, `is_error` |
| Web search | `web_search_20250305` / `_20260209` / `_20260318` through Google Search, with search result blocks and citations (see below) |
| Anthropic-defined client tools | `text_editor_20250728` / `_20250429` / `_20250124`, `bash_20250124`, `memory_20250818`: pass `{"type": ..., "name": ...}` as usual, see below |
| Streaming | `create(..., stream=True)` raw events and the `messages.stream()` helper (`text_stream`, `get_final_message()`, ...) |
| Extended thinking | `thinking={"type": "enabled", "budget_tokens": N}` / `{"type": "adaptive"}` / `{"type": "disabled"}`, `display`, `thinking` / `redacted_thinking` blocks and streaming deltas (see below) |
| Structured outputs | `output_config={"format": {"type": "json_schema", ...}}`, `messages.parse(output_format=Model)` and `messages.stream(output_format=Model)` |
| Beta API | `client.beta.messages.create` / `stream` / `parse` with `betas=[...]`; the betas are accepted and ignored (see below) |
| MCP | client-side MCP (listing an MCP server's tools and passing them as `tools`), since it is regular tool use |

Responses are real `anthropic.types.Message` objects, with `stop_reason` and `usage` mapped from the Gemini response.

## Anthropic-defined client tools

The text editor, bash and memory tools are schema-less on Anthropic's API: the schema is built into Claude's model. Gemini needs one, so [`tool_schemas.py`](src/false_anthropic/tool_schemas.py) rebuilds each schema from the parameters described in Anthropic's docs, and `{"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}` is replaced with a regular tool with that schema. Gemini then calls it with the same `name` and `input` shape as Claude, so your handlers work unchanged.

| `type` | Required `name` | Commands |
| --- | --- | --- |
| `text_editor_20250728`, `text_editor_20250429` | `str_replace_based_edit_tool` | `view`, `create`, `str_replace`, `insert` (with `insert_text`); `max_characters` is supported on `20250728` |
| `text_editor_20250124` | `str_replace_editor` | the same plus `undo_edit`, and `insert` uses `new_str` |
| `bash_20250124` | `bash` | `command`, `restart` |
| `memory_20250818` | `memory` | `view`, `create`, `str_replace`, `insert`, `delete`, `rename`; Anthropic's memory system prompt is added too |

Claude is trained on these tools while Gemini only gets the schema and its description, so expect Gemini to use them a bit differently. Other server tools (code execution, web fetch...) and the computer / browser use tools are not supported.

## Web search

`{"type": "web_search_20250305", "name": "web_search"}` (or `web_search_20260209` / `web_search_20260318`) runs as Gemini's Google Search grounding, see [`web_search.py`](src/false_anthropic/web_search.py). The response is mapped to Anthropic's shape:

- each search Gemini made becomes a `server_tool_use` block followed by a `web_search_tool_result` block with the sources as `web_search_result`s,
- the answer is split into text blocks, with `web_search_result_location` citations on the parts Gemini grounded,
- `usage.server_tool_use.web_search_requests` counts the searches.

Differences: Gemini does not say which search found which source, so all sources are in the first result block; result URLs are Google redirect links and titles are usually the domain; `encrypted_content` / `encrypted_index` are empty and `cited_text` is the cited part of the answer, not a quote from the page; `max_uses`, `allowed_domains`, `blocked_domains` and `user_location` are ignored with a warning. When streaming, the response is fetched whole and then replayed as events, since Gemini only sends the sources at the end.

**Quota:** Google Search grounding has its own quota, and free tier keys may get `429 RESOURCE_EXHAUSTED` on every search request even when regular requests work. Check your limits at <https://ai.dev/rate-limit>.

## Extended thinking

Gemini's thinking models are available on the free tier, and `thinking` maps to Gemini's thinking config with the thoughts included in the response:

- `{"type": "enabled", "budget_tokens": N}` sets Gemini's `thinking_budget` to `N`, `{"type": "adaptive"}` uses Gemini's dynamic thinking (the model decides, and may not think at all on easy prompts), and `{"type": "disabled"}` (or no `thinking`) leaves Gemini's default, with the thoughts left out of the response.
- Gemini's thought summaries become a `thinking` block before the answer. `display: "omitted"` returns the block with empty `thinking`.
- Like Anthropic's API, a user message containing `ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING...` returns a `redacted_thinking` block instead, so you can test how your code handles one.
- When streaming, the thoughts arrive as `thinking_delta`s followed by a `signature_delta`.
- Thinking tokens are counted in `usage.output_tokens`, like Anthropic.

Differences: Gemini returns a summary of its thoughts, not the full reasoning (Claude 4 models summarize too). The `signature` is Gemini's thought signature when it is attached to the thought, otherwise a placeholder, and `redacted_thinking` `data` is a random value. Thinking blocks sent back in `messages` are accepted and dropped, since Gemini only needs the signatures of the tool calls (kept on the `tool_use` blocks). Anthropic's rules (budget at least 1024 and below `max_tokens`, no prefill or forced tool use with thinking, `temperature` 1) are not enforced. Whether a model can think, and its budget limits, depend on the Gemini model.

## Differences and limitations

- Gemini rejects requests ending with an assistant message, so prefill is emulated: a user turn asking the model to continue is added, any repeated prefill text is stripped from the output, and stop sequences are applied by the library (so `stop_reason="stop_sequence"` and `stop_sequence` are reported). It is reliable for typical cases like prefilling ` ```json ` with stop sequence ` ``` `, but it is the model following an instruction, not a true continuation.
- Without a prefill, `stop_reason` is never `"stop_sequence"` and `stop_sequence` is always `None`: Gemini does not report when a stop sequence was hit (the output is still cut).
- Gemini streams in bigger chunks than Claude, and tool inputs arrive whole in a single `input_json` delta.
- Structured output schemas are sent to Gemini as they are, so constraints like `minimum` / `maximum` are enforced by the model.
- `output_config["effort"]` and `tool_choice["disable_parallel_tool_use"]` are ignored.
- `betas` are ignored and `client.beta.messages` returns the regular (non `Beta*`) types, which have the same fields. `fine-grained-tool-streaming-2025-05-14` needs nothing: tool inputs are never buffered or validated, Gemini just sends each call whole.
- Not supported yet: images / PDFs / documents, `thinking={"type": "between_tools"}`, prompt caching, server tools other than web search (code execution, web fetch...), the `mcp_servers` connector, `count_tokens`, batches, the async client.
- Gemini 3 models need a "thought signature" when a tool call is sent back. It is kept as an extra `gemini_thought_signature` field on the returned `ToolUseBlock`; for hand-written history a placeholder is sent instead.
