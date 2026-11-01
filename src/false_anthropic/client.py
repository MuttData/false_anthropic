"""
The Google GenAI client exposed as an Anthropic-compatible interface.
The idea is to provide an alternative for the Anthropic certification exercises.
"""

import base64
import json
import uuid
from typing import Any, Iterable, Iterator, Literal, Mapping, NotRequired, TypedDict

import anthropic
from anthropic.lib._parse._response import parse_response
from anthropic.lib.streaming import MessageStreamManager
from anthropic.types.raw_message_delta_event import Delta
from google import genai
from pydantic import TypeAdapter
from google.genai import types

from false_anthropic import web_search
from false_anthropic.tool_schemas import resolve_client_tool


class AnthropicTextBlock(TypedDict):
    type: Literal["text"]
    text: str


class AnthropicToolUseBlock(TypedDict):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]


class AnthropicToolResultBlock(TypedDict):
    type: Literal["tool_result"]
    tool_use_id: str
    content: NotRequired[str | Iterable[AnthropicTextBlock]]
    is_error: NotRequired[bool]


AnthropicContentBlock = (
    AnthropicTextBlock
    | AnthropicToolUseBlock
    | AnthropicToolResultBlock
    | anthropic.types.TextBlock
    | anthropic.types.ToolUseBlock
)


class AnthropicMessage(TypedDict):  # anthropic manages its own type, for now lets do a typeddict
    role: Literal["user", "assistant"]
    content: str | Iterable[AnthropicContentBlock]


class AnthropicSystemBlock(TypedDict):
    type: Literal["text"]
    text: str
    cache_control: NotRequired[dict]


class AnthropicTool(TypedDict):
    name: str
    description: NotRequired[str]
    input_schema: dict[str, Any]


class AnthropicToolChoice(TypedDict):
    type: Literal["auto", "any", "tool", "none"]
    name: NotRequired[str]
    disable_parallel_tool_use: NotRequired[bool]


class AnthropicJSONOutputFormat(TypedDict):
    type: Literal["json_schema"]
    schema: dict[str, Any]


class AnthropicOutputConfig(TypedDict, total=False):
    format: AnthropicJSONOutputFormat | None
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None


class AnthropicThinkingConfig(TypedDict):
    type: Literal["enabled", "adaptive", "disabled"]
    budget_tokens: NotRequired[int]
    display: NotRequired[Literal["summarized", "omitted"] | None]


_ROLES_TO_GENAI = {"user": "user", "assistant": "model"}

# Gemini finish reasons that have no 1:1 Anthropic equivalent are treated as refusals,
# since almost all of them mean the output was blocked by a safety/policy filter.
_FINISH_REASON_TO_STOP_REASON: dict[types.FinishReason, anthropic.types.StopReason] = {
    types.FinishReason.STOP: "end_turn",
    types.FinishReason.MAX_TOKENS: "max_tokens",
    types.FinishReason.FINISH_REASON_UNSPECIFIED: "end_turn",
    types.FinishReason.OTHER: "end_turn",
    types.FinishReason.MALFORMED_FUNCTION_CALL: "end_turn",
    types.FinishReason.UNEXPECTED_TOOL_CALL: "end_turn",
    types.FinishReason.TOO_MANY_TOOL_CALLS: "end_turn",
}

_TOOL_CHOICE_TO_GENAI_MODE = {
    "auto": types.FunctionCallingConfigMode.AUTO,
    "any": types.FunctionCallingConfigMode.ANY,
    "tool": types.FunctionCallingConfigMode.ANY,
    "none": types.FunctionCallingConfigMode.NONE,
}

# Gemini 3 models reject function calls sent back without the thought signature they were
# generated with. We keep it as an extra field on the returned `ToolUseBlock`, and fall back
# to Google's documented placeholder when the history was written by hand.
_THOUGHT_SIGNATURE_FIELD = "gemini_thought_signature"
_DUMMY_THOUGHT_SIGNATURE = b"skip_thought_signature_validator"


# Gemini does not sign its thought summaries like Claude does, and thinking blocks are not sent back to Gemini,
# so thinking blocks carry this placeholder unless Gemini attached a signature to the thought part itself
_PLACEHOLDER_THINKING_SIGNATURE = base64.b64encode(b"false_anthropic_thinking").decode()

# Sending this string makes Anthropic's API return `redacted_thinking` blocks, to test how a client handles them
_REDACTED_THINKING_TRIGGER = "ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING"

ThinkingDisplay = Literal["summarized", "omitted", "redacted"]


# Gemini rejects requests ending with a model turn, so a prefill is followed by this user turn
_PREFILL_CONTINUE_PROMPT = (
    "Continue your previous response exactly from where it ends. Do not repeat any of it, output only the continuation."
)


class _PrefillContinuation:
    """Emulates an Anthropic prefill (a request ending with an assistant message), which Gemini rejects.

    The model is asked to continue the prefill text. If it repeats the prefill anyway, this class removes that echo,
    so only the continuation is returned, like Anthropic does. Stop sequences are applied here instead of by Gemini,
    otherwise one that matches an echoed prefill (e.g. prefill "```json" and stop sequence "```") would end the
    response before it starts. It also lets us report the `stop_sequence` that was hit.
    """

    def __init__(self, prefill: str, stop_sequences: list[str] | None):
        self.prefill = prefill
        self.stop_sequences = [s for s in stop_sequences or [] if s]
        self.stop_sequence: str | None = None  # the stop sequence that was hit, if any
        self._buffer = ""
        self._prefill_checked = False

    def feed(self, text: str) -> str:
        """Takes the next piece of model output and returns the part of it that is ready to be emitted."""
        if self.stop_sequence is not None:
            return ""
        self._buffer += text

        if not self._prefill_checked:
            stripped = self._buffer.lstrip()
            if len(stripped) < len(self.prefill) and self.prefill.startswith(stripped):
                return ""  # can't tell yet if the model is echoing the prefill
            if stripped.startswith(self.prefill):
                self._buffer = stripped[len(self.prefill):]
            # otherwise the model continued straight away, keep its output as is
            self._prefill_checked = True

        hits = [(self._buffer.find(s), s) for s in self.stop_sequences if s in self._buffer]
        if hits:
            position, self.stop_sequence = min(hits)
            text, self._buffer = self._buffer[:position], ""
            return text

        # hold back the tail, it could be the start of a stop sequence split across chunks
        hold_back = max((len(s) - 1 for s in self.stop_sequences), default=0)
        cut = len(self._buffer) - hold_back
        if cut <= 0:
            return ""
        text, self._buffer = self._buffer[:cut], self._buffer[cut:]
        return text

    def flush(self) -> str:
        """Returns the output held back once the model has finished."""
        if not self._prefill_checked:
            # the whole output is (a part of) the echoed prefill
            self._prefill_checked = True
            self._buffer = ""
        text, self._buffer = self._buffer, ""
        return text


class FalseStream:
    """Stand-in for `anthropic.Stream`: iterates the raw events returned by `create(..., stream=True)`."""

    response = None  # there is no Anthropic HTTP response behind this stream

    def __init__(self, events: Iterator[anthropic.types.RawMessageStreamEvent]):
        self._events = events

    def __iter__(self) -> Iterator[anthropic.types.RawMessageStreamEvent]:
        return self._events

    def __next__(self) -> anthropic.types.RawMessageStreamEvent:
        return next(self._events)

    def __enter__(self) -> "FalseStream":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._events.close()


class FalseAnthropicMessages:
    def __init__(self, client: genai.Client):
        self.client = client

    def stream(self, *, output_format: type | None = None, **kwargs: Any) -> MessageStreamManager:
        """Same arguments as `create`, plus `output_format` like `parse`.
        Use it as `with client.messages.stream(...) as stream:`.
        """
        kwargs["output_config"] = self._merge_output_format(output_format, kwargs.get("output_config"))
        # reuse the Anthropic SDK helper, so `text_stream`, `get_final_message()`, etc. behave the same
        return MessageStreamManager(
            lambda: self.create(**kwargs, stream=True),
            output_format=output_format if output_format is not None else anthropic.not_given,
        )

    def parse(self, *, output_format: type | None = None, **kwargs: Any) -> anthropic.types.ParsedMessage:
        """Same arguments as `create`, plus `output_format`: a pydantic model (or any type pydantic can validate).
        The validated object is available as `message.parsed_output`.
        """
        kwargs["output_config"] = self._merge_output_format(output_format, kwargs.get("output_config"))
        message = self.create(**kwargs)
        return parse_response(
            output_format=output_format if output_format is not None else anthropic.not_given,
            response=message,
        )

    def _merge_output_format(
        self, output_format: type | None, output_config: AnthropicOutputConfig | None
    ) -> AnthropicOutputConfig | None:
        if output_format is None:
            return output_config
        if isinstance(output_format, Mapping):
            raise TypeError("`output_format` must be a type; pass a schema dict as `output_config={'format': ...}` instead")
        # unlike Anthropic, Gemini supports the constraints in the schema (min/max, etc.), so it is sent as is
        schema = TypeAdapter(output_format).json_schema()
        return {**(output_config or {}), "format": {"type": "json_schema", "schema": schema}}

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[AnthropicMessage],
        system: str | Iterable[AnthropicSystemBlock] | None = None,
        temperature: float | None = None,
        stop_sequences: list[str] | None = None,
        tools: list[AnthropicTool] | None = None,
        tool_choice: AnthropicToolChoice | None = None,
        output_config: AnthropicOutputConfig | None = None,
        thinking: AnthropicThinkingConfig | None = None,
        stream: bool = False,
    ) -> anthropic.types.Message | FalseStream:
        thinking_display = self._thinking_display(thinking, messages)
        continuation = None
        if messages and messages[-1]["role"] == "assistant":
            continuation = _PrefillContinuation(self._prefill_text(messages[-1]), stop_sequences)
            messages = [*messages, {"role": "user", "content": _PREFILL_CONTINUE_PROMPT}]
            stop_sequences = None  # applied by the continuation

        # Gemini function responses are matched by name, Anthropic tool results only carry the id
        tool_names = {
            block["id"]: block["name"]
            for message in messages
            for block in self._blocks_from_content(message["content"])
            if block["type"] == "tool_use"
        }
        genai_messages = [self._from_anthropic_to_genai(m, tool_names) for m in messages]
        genai_messages = [m for m in genai_messages if m is not None]
        genai_config = self._from_anthropic_to_genai_config(
            max_tokens=max_tokens,
            system=system,
            temperature=temperature,
            stop_sequences=stop_sequences,
            tools=tools,
            tool_choice=tool_choice,
            output_config=output_config,
            thinking=thinking,
        )

        if stream and any(web_search.is_web_search_tool(tool) for tool in tools or []):
            # Gemini only sends the search results in the last chunk, while Anthropic sends them before the answer
            # text, so the whole response is needed first. It is then replayed as stream events.
            response = self.client.models.generate_content(
                model=model,
                contents=genai_messages,
                config=genai_config
            )
            message = self._from_genai_to_anthropic(
                response, model=model, continuation=continuation, thinking_display=thinking_display
            )
            return FalseStream(self._events_from_message(message))

        if stream:
            chunks = self.client.models.generate_content_stream(
                model=model,
                contents=genai_messages,
                config=genai_config
            )
            return FalseStream(
                self._stream_events(chunks, model=model, continuation=continuation, thinking_display=thinking_display)
            )

        response = self.client.models.generate_content(
            model=model,
            contents=genai_messages,
            config=genai_config
        )
        return self._from_genai_to_anthropic(
            response, model=model, continuation=continuation, thinking_display=thinking_display
        )

    def _thinking_display(
        self, thinking: AnthropicThinkingConfig | None, messages: list[AnthropicMessage]
    ) -> ThinkingDisplay | None:
        """How the thoughts are returned: as `thinking` blocks, with the text omitted, as `redacted_thinking` blocks,
        or not at all (None) when thinking is not enabled.
        """
        if thinking is None or thinking["type"] == "disabled":
            return None
        if thinking["type"] not in ("enabled", "adaptive"):
            raise NotImplementedError(f"Thinking type {thinking['type']!r} is not supported yet")
        user_texts = (
            block["text"]
            for message in messages
            if message["role"] == "user"
            for block in self._blocks_from_content(message["content"])
            if block["type"] == "text"
        )
        if any(_REDACTED_THINKING_TRIGGER in text for text in user_texts):
            return "redacted"
        return thinking.get("display") or "summarized"

    def _prefill_text(self, message: AnthropicMessage) -> str:
        blocks = self._blocks_from_content(message["content"])
        if any(block["type"] != "text" for block in blocks):
            raise ValueError("A final assistant message (prefill) can only contain text")
        return "".join(block["text"] for block in blocks)

    def _from_anthropic_to_genai(self, message: AnthropicMessage, tool_names: dict[str, str]) -> types.Content | None:
        parts = [self._part_from_block(block, tool_names) for block in self._blocks_from_content(message["content"])]
        parts = [part for part in parts if part is not None]
        if not parts:
            return None
        return types.Content(role=_ROLES_TO_GENAI[message["role"]], parts=parts)

    def _blocks_from_content(self, content: str | Iterable[AnthropicContentBlock]) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        # accept both plain dicts and the pydantic blocks returned in `Message.content`
        return [block if isinstance(block, dict) else block.model_dump() for block in content]

    def _part_from_block(self, block: dict[str, Any], tool_names: dict[str, str]) -> types.Part | None:
        match block["type"]:
            case "server_tool_use" | "web_search_tool_result":
                # Gemini can't take its searches back as input, the answer text carries what was found
                return None
            case "thinking" | "redacted_thinking":
                # Gemini does not take its thought summaries back as input, it only needs the thought signatures
                # (kept on the tool_use blocks)
                return None
            case "text":
                return types.Part.from_text(text=block["text"])
            case "tool_use":
                return types.Part(
                    function_call=types.FunctionCall(id=block["id"], name=block["name"], args=block["input"]),
                    thought_signature=block.get(_THOUGHT_SIGNATURE_FIELD) or _DUMMY_THOUGHT_SIGNATURE,
                )
            case "tool_result":
                tool_use_id = block["tool_use_id"]
                if tool_use_id not in tool_names:
                    raise ValueError(f"tool_result {tool_use_id!r} has no matching tool_use block in the messages")
                result = self._tool_result_text(block.get("content", ""))
                return types.Part(
                    function_response=types.FunctionResponse(
                        id=tool_use_id,
                        name=tool_names[tool_use_id],
                        response={"error": result} if block.get("is_error") else {"output": result},
                    )
                )
            case other:
                raise NotImplementedError(f"Content block type {other!r} is not supported yet")

    def _tool_result_text(self, content: str | Iterable[AnthropicTextBlock]) -> str:
        texts = []
        for block in self._blocks_from_content(content):
            if block["type"] != "text":
                raise NotImplementedError(f"Tool result block type {block['type']!r} is not supported yet")
            texts.append(block["text"])
        return "\n".join(texts)

    def _from_genai_to_anthropic(
        self,
        response: types.GenerateContentResponse,
        *,
        model: str,
        continuation: _PrefillContinuation | None = None,
        thinking_display: ThinkingDisplay | None = None,
    ) -> anthropic.types.Message:
        candidate = response.candidates[0] if response.candidates else None
        parts = candidate.content.parts if candidate and candidate.content and candidate.content.parts else []

        content: list[anthropic.types.ContentBlock] = []
        for part in parts:
            if part.function_call:
                content.append(self._tool_use_from_function_call(part))
            elif part.text and part.thought:
                # Anthropic only returns the thoughts when thinking is enabled
                if thinking_display is None:
                    continue
                # merge consecutive thought parts into a single block
                if content and content[-1].type in ("thinking", "redacted_thinking"):
                    if content[-1].type == "thinking" and thinking_display != "omitted":
                        content[-1].thinking += part.text
                else:
                    content.append(self._thinking_block(part, thinking_display))
            elif part.text:
                # merge consecutive text parts into a single block, like Anthropic does
                if content and content[-1].type == "text":
                    content[-1].text += part.text
                else:
                    content.append(anthropic.types.TextBlock(type="text", text=part.text))

        grounding = candidate.grounding_metadata if candidate else None
        if web_search.has_web_search(grounding):
            thinking_blocks = [block for block in content if block.type in ("thinking", "redacted_thinking")]
            if any(block.type == "tool_use" for block in content) or continuation is not None:
                # keep it simple, citations only for plain answers
                text_blocks = [block for block in content if block.type not in ("thinking", "redacted_thinking")]
            else:
                part_texts = [part.text if part.text and not part.thought else "" for part in parts]
                text_blocks = web_search.cited_text_blocks(part_texts, grounding)
            # like Claude, the thinking comes before the searches
            content = [*thinking_blocks, *web_search.search_blocks(grounding), *text_blocks]

        if continuation is not None:
            continued = []
            for block in content:
                if continuation.stop_sequence is not None:
                    break  # the output after a stop sequence is dropped
                if block.type == "text":
                    block.text = continuation.feed(block.text) + continuation.flush()
                continued.append(block)
            content = continued

        has_tool_use = any(block.type == "tool_use" for block in content)
        stop_sequence = continuation.stop_sequence if continuation else None
        usage = self._usage(response.usage_metadata)
        usage.server_tool_use = web_search.server_tool_usage(grounding)
        return self._message(
            response,
            model=model,
            content=content,
            stop_reason="stop_sequence" if stop_sequence else self._stop_reason(candidate, has_tool_use),
            stop_sequence=stop_sequence,
            usage=usage,
        )

    def _thinking_block(
        self, part: types.Part, thinking_display: ThinkingDisplay
    ) -> anthropic.types.ThinkingBlock | anthropic.types.RedactedThinkingBlock:
        """The block for the first thought part of a run of them (the following ones are merged into it)."""
        if thinking_display == "redacted":
            # Claude's redacted thinking is encrypted, so the content is just an opaque value here
            return anthropic.types.RedactedThinkingBlock(
                type="redacted_thinking", data=base64.b64encode(uuid.uuid4().bytes).decode()
            )
        signature = (
            base64.b64encode(part.thought_signature).decode() if part.thought_signature else _PLACEHOLDER_THINKING_SIGNATURE
        )
        return anthropic.types.ThinkingBlock(
            type="thinking", thinking="" if thinking_display == "omitted" else part.text, signature=signature
        )

    def _events_from_message(self, message: anthropic.types.Message) -> Iterator[anthropic.types.RawMessageStreamEvent]:
        """Replays a complete message as the Anthropic raw stream events that would have produced it."""
        yield anthropic.types.RawMessageStartEvent(
            type="message_start",
            message=message.model_copy(
                update={
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": anthropic.types.Usage(input_tokens=message.usage.input_tokens, output_tokens=0),
                }
            ),
        )
        for index, block in enumerate(message.content):
            if block.type == "text":
                yield anthropic.types.RawContentBlockStartEvent(
                    type="content_block_start",
                    index=index,
                    content_block=anthropic.types.TextBlock(type="text", text="", citations=[] if block.citations else None),
                )
                for citation in block.citations or []:
                    yield anthropic.types.RawContentBlockDeltaEvent(
                        type="content_block_delta",
                        index=index,
                        delta=anthropic.types.CitationsDelta(type="citations_delta", citation=citation),
                    )
                yield anthropic.types.RawContentBlockDeltaEvent(
                    type="content_block_delta",
                    index=index,
                    delta=anthropic.types.TextDelta(type="text_delta", text=block.text),
                )
            elif block.type == "thinking":
                yield anthropic.types.RawContentBlockStartEvent(
                    type="content_block_start",
                    index=index,
                    content_block=anthropic.types.ThinkingBlock(type="thinking", thinking="", signature=""),
                )
                if block.thinking:
                    yield anthropic.types.RawContentBlockDeltaEvent(
                        type="content_block_delta",
                        index=index,
                        delta=anthropic.types.ThinkingDelta(type="thinking_delta", thinking=block.thinking),
                    )
                yield anthropic.types.RawContentBlockDeltaEvent(
                    type="content_block_delta",
                    index=index,
                    delta=anthropic.types.SignatureDelta(type="signature_delta", signature=block.signature),
                )
            elif block.type in ("tool_use", "server_tool_use"):
                yield anthropic.types.RawContentBlockStartEvent(
                    type="content_block_start", index=index, content_block=block.model_copy(update={"input": {}})
                )
                yield anthropic.types.RawContentBlockDeltaEvent(
                    type="content_block_delta",
                    index=index,
                    delta=anthropic.types.InputJSONDelta(type="input_json_delta", partial_json=json.dumps(block.input)),
                )
            else:
                yield anthropic.types.RawContentBlockStartEvent(type="content_block_start", index=index, content_block=block)
            yield anthropic.types.RawContentBlockStopEvent(type="content_block_stop", index=index)

        yield anthropic.types.RawMessageDeltaEvent(
            type="message_delta",
            delta=Delta(stop_reason=message.stop_reason, stop_sequence=message.stop_sequence),
            usage=anthropic.types.MessageDeltaUsage(
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
                cache_read_input_tokens=message.usage.cache_read_input_tokens,
                server_tool_use=message.usage.server_tool_use,
            ),
        )
        yield anthropic.types.RawMessageStopEvent(type="message_stop")

    def _message(
        self,
        response: types.GenerateContentResponse,
        *,
        model: str,
        content: list[anthropic.types.ContentBlock],
        stop_reason: anthropic.types.StopReason | None,
        usage: anthropic.types.Usage,
        stop_sequence: str | None = None,
    ) -> anthropic.types.Message:
        return anthropic.types.Message(
            id=f"msg_{response.response_id or uuid.uuid4().hex}",
            type="message",
            role="assistant",
            model=response.model_version or model,
            content=content,
            stop_reason=stop_reason,
            # Gemini does not report which stop sequence was hit, only a prefill continuation knows it
            stop_sequence=stop_sequence,
            usage=usage,
        )

    def _stream_events(
        self,
        chunks: Iterator[types.GenerateContentResponse],
        *,
        model: str,
        continuation: _PrefillContinuation | None = None,
        thinking_display: ThinkingDisplay | None = None,
    ) -> Iterator[anthropic.types.RawMessageStreamEvent]:
        """Translates Gemini stream chunks into the Anthropic raw event sequence:
        message_start, (content_block_start, content_block_delta*, content_block_stop)*, message_delta, message_stop
        """
        last_chunk = None
        index = -1  # index of the current content block
        open_block = None  # the block that has not been stopped yet, if any
        has_tool_use = False

        def stop_block() -> Iterator[anthropic.types.RawMessageStreamEvent]:
            nonlocal open_block
            if open_block is None:
                return
            if open_block.type == "thinking":
                # like Claude, the signature comes last
                yield anthropic.types.RawContentBlockDeltaEvent(
                    type="content_block_delta",
                    index=index,
                    delta=anthropic.types.SignatureDelta(type="signature_delta", signature=open_block.signature),
                )
            yield anthropic.types.RawContentBlockStopEvent(type="content_block_stop", index=index)
            open_block = None

        def start_block(block: anthropic.types.ContentBlock) -> Iterator[anthropic.types.RawMessageStreamEvent]:
            nonlocal index, open_block
            yield from stop_block()
            index += 1
            open_block = block
            if block.type == "thinking":
                block = block.model_copy(update={"thinking": "", "signature": ""})
            elif block.type == "tool_use":
                block = block.model_copy(update={"input": {}})
            yield anthropic.types.RawContentBlockStartEvent(type="content_block_start", index=index, content_block=block)

        def text_events(text: str) -> Iterator[anthropic.types.RawMessageStreamEvent]:
            if not text:
                return
            if open_block is None or open_block.type != "text":
                yield from start_block(anthropic.types.TextBlock(type="text", text=""))
            yield anthropic.types.RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=index,
                delta=anthropic.types.TextDelta(type="text_delta", text=text),
            )

        def thinking_events(part: types.Part) -> Iterator[anthropic.types.RawMessageStreamEvent]:
            if open_block is None or open_block.type not in ("thinking", "redacted_thinking"):
                yield from start_block(self._thinking_block(part, thinking_display))
            if open_block.type == "thinking" and thinking_display != "omitted":
                yield anthropic.types.RawContentBlockDeltaEvent(
                    type="content_block_delta",
                    index=index,
                    delta=anthropic.types.ThinkingDelta(type="thinking_delta", thinking=part.text),
                )

        for chunk in chunks:
            if last_chunk is None:
                yield anthropic.types.RawMessageStartEvent(
                    type="message_start",
                    message=self._message(
                        chunk,
                        model=model,
                        content=[],
                        stop_reason=None,
                        usage=anthropic.types.Usage(input_tokens=self._usage(chunk.usage_metadata).input_tokens, output_tokens=0),
                    ),
                )
            last_chunk = chunk

            candidate = chunk.candidates[0] if chunk.candidates else None
            parts = candidate.content.parts if candidate and candidate.content and candidate.content.parts else []
            for part in parts:
                if part.function_call:
                    # Gemini sends each function call whole, so it becomes a full block on its own
                    has_tool_use = True
                    block = self._tool_use_from_function_call(part)
                    yield from start_block(block)
                    yield anthropic.types.RawContentBlockDeltaEvent(
                        type="content_block_delta",
                        index=index,
                        delta=anthropic.types.InputJSONDelta(type="input_json_delta", partial_json=json.dumps(block.input)),
                    )
                    yield from stop_block()
                elif part.text and part.thought:
                    if thinking_display is not None:
                        yield from thinking_events(part)
                elif part.text:
                    yield from text_events(continuation.feed(part.text) if continuation else part.text)

            if continuation is not None and continuation.stop_sequence is not None:
                chunks.close()  # the output after a stop sequence is dropped, no need to wait for it
                break

        if last_chunk is None:
            raise RuntimeError("Gemini returned an empty stream")
        if continuation is not None:
            yield from text_events(continuation.flush())
        yield from stop_block()

        # the last chunk carries the finish reason and the cumulative token usage
        candidate = last_chunk.candidates[0] if last_chunk.candidates else None
        usage = self._usage(last_chunk.usage_metadata)
        stop_sequence = continuation.stop_sequence if continuation else None
        yield anthropic.types.RawMessageDeltaEvent(
            type="message_delta",
            delta=Delta(
                stop_reason="stop_sequence" if stop_sequence else self._stop_reason(candidate, has_tool_use),
                stop_sequence=stop_sequence,
            ),
            usage=anthropic.types.MessageDeltaUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_input_tokens=usage.cache_read_input_tokens,
            ),
        )
        yield anthropic.types.RawMessageStopEvent(type="message_stop")

    def _tool_use_from_function_call(self, part: types.Part) -> anthropic.types.ToolUseBlock:
        function_call = part.function_call
        extra = {_THOUGHT_SIGNATURE_FIELD: part.thought_signature} if part.thought_signature else {}
        return anthropic.types.ToolUseBlock(
            type="tool_use",
            id=function_call.id or f"toolu_{uuid.uuid4().hex}",
            name=function_call.name,
            input=function_call.args or {},
            **extra,
        )

    def _stop_reason(self, candidate: types.Candidate | None, has_tool_use: bool) -> anthropic.types.StopReason:
        if candidate is None:
            # no candidates means the prompt itself was blocked (see `response.prompt_feedback`)
            return "refusal"
        if has_tool_use:
            return "tool_use"
        if candidate.finish_reason is None:
            return "end_turn"
        return _FINISH_REASON_TO_STOP_REASON.get(candidate.finish_reason, "refusal")

    def _usage(self, usage: types.GenerateContentResponseUsageMetadata | None) -> anthropic.types.Usage:
        if usage is None:
            return anthropic.types.Usage(input_tokens=0, output_tokens=0)
        # Anthropic counts thinking tokens as output tokens
        output_tokens = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
        return anthropic.types.Usage(
            input_tokens=usage.prompt_token_count or 0,
            output_tokens=output_tokens,
            cache_read_input_tokens=usage.cached_content_token_count,
        )

    def _from_anthropic_to_genai_config(
        self,
        *,
        max_tokens: int,
        system: str | Iterable[AnthropicSystemBlock] | None,
        temperature: float | None,
        stop_sequences: list[str] | None,
        tools: list[AnthropicTool] | None,
        tool_choice: AnthropicToolChoice | None,
        output_config: AnthropicOutputConfig | None,
        thinking: AnthropicThinkingConfig | None = None,
    ) -> types.GenerateContentConfig:
        builtin_tools = [web_search.google_search_tool(tool) for tool in tools or [] if web_search.is_web_search_tool(tool)]
        tools = [tool for tool in tools or [] if not web_search.is_web_search_tool(tool)]
        tools, tool_system_prompts = self._resolve_client_tools(tools)
        system_texts = [block["text"] for block in self._blocks_from_content(system)] if system is not None else []
        system_instruction = "\n\n".join([*system_texts, *tool_system_prompts]) or None
        # `output_config["effort"]` has no direct Gemini equivalent and is ignored
        output_format = (output_config or {}).get("format")
        if output_format is not None and output_format["type"] != "json_schema":
            raise NotImplementedError(f"Output format type {output_format['type']!r} is not supported yet")
        return types.GenerateContentConfig(
            response_mime_type="application/json" if output_format else None,
            response_json_schema=output_format["schema"] if output_format else None,
            max_output_tokens=max_tokens,
            system_instruction=system_instruction,
            temperature=temperature,
            stop_sequences=stop_sequences,
            tools=[*(self._genai_tools(tools) or []), *builtin_tools] or None,
            tool_config=self._genai_tool_config(tool_choice),
            thinking_config=self._genai_thinking_config(thinking),
            # the exercises run the tools themselves, like with the Anthropic SDK
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def _genai_thinking_config(self, thinking: AnthropicThinkingConfig | None) -> types.ThinkingConfig | None:
        if thinking is None or thinking["type"] == "disabled":
            # Gemini's default; some models can't turn thinking off, and their thoughts are left out of the response
            return None
        if thinking["type"] == "adaptive":
            # -1 is Gemini's dynamic thinking: the model decides how much to think
            return types.ThinkingConfig(include_thoughts=True, thinking_budget=-1)
        if thinking["type"] == "enabled":
            return types.ThinkingConfig(include_thoughts=True, thinking_budget=thinking["budget_tokens"])
        raise NotImplementedError(f"Thinking type {thinking['type']!r} is not supported yet")

    def _resolve_client_tools(self, tools: list[AnthropicTool] | None) -> tuple[list[AnthropicTool] | None, list[str]]:
        """Replaces the Anthropic-defined client tools (text editor, bash, memory) with regular tools carrying
        their schema, since they are schema-less on Anthropic's side. Also returns the system prompt text that
        Anthropic's API adds for them.
        """
        if not tools:
            return tools, []
        resolved, system_prompts = [], []
        for tool in tools:
            if tool.get("type", "custom") == "custom":
                resolved.append(tool)
                continue
            schema = resolve_client_tool(tool)
            resolved.append({"name": schema.name, "description": schema.description, "input_schema": schema.input_schema})
            if schema.system_prompt:
                system_prompts.append(schema.system_prompt)
        return resolved, system_prompts

    def _genai_tools(self, tools: list[AnthropicTool] | None) -> list[types.Tool] | None:
        if not tools:
            return None
        declarations = []
        for tool in tools:
            declarations.append(
                types.FunctionDeclaration(
                    name=tool["name"],
                    description=tool.get("description"),
                    parameters_json_schema=tool["input_schema"],
                )
            )
        return [types.Tool(function_declarations=declarations)]

    def _genai_tool_config(self, tool_choice: AnthropicToolChoice | None) -> types.ToolConfig | None:
        if tool_choice is None:
            return None
        # `disable_parallel_tool_use` has no Gemini equivalent and is ignored
        return types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=_TOOL_CHOICE_TO_GENAI_MODE[tool_choice["type"]],
                allowed_function_names=[tool_choice["name"]] if tool_choice["type"] == "tool" else None,
            )
        )


class FalseAnthropicBetaMessages:
    """`client.beta.messages`: same methods as `client.messages`, plus the `betas` argument.

    The betas are accepted and ignored: Gemini has no equivalent flags, and the behavior they enable either already
    matches (e.g. `fine-grained-tool-streaming-2025-05-14`, Gemini sends tool inputs unbuffered as whole calls)
    or is not supported by this library.
    """

    def __init__(self, messages: FalseAnthropicMessages):
        self._messages = messages

    def create(self, *, betas: list[str] | None = None, **kwargs: Any) -> anthropic.types.Message | FalseStream:
        return self._messages.create(**kwargs)

    def stream(self, *, betas: list[str] | None = None, **kwargs: Any) -> MessageStreamManager:
        return self._messages.stream(**kwargs)

    def parse(self, *, betas: list[str] | None = None, **kwargs: Any) -> anthropic.types.ParsedMessage:
        return self._messages.parse(**kwargs)


class FalseAnthropicBeta:
    def __init__(self, messages: FalseAnthropicMessages):
        self.messages = FalseAnthropicBetaMessages(messages)


class FalseAnthropic:
    def __init__(self, api_key: str | None = None):
        self.client = genai.Client(api_key=api_key)
        self.messages = FalseAnthropicMessages(self.client)
        self.beta = FalseAnthropicBeta(self.messages)
