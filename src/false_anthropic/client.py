"""
The Google GenAI client exposed as an Anthropic-compatible interface.
The idea is to provide an alternative for the Anthropic certification exercises.
"""

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
        stream: bool = False,
    ) -> anthropic.types.Message | FalseStream:
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
        genai_config = self._from_anthropic_to_genai_config(
            max_tokens=max_tokens,
            system=system,
            temperature=temperature,
            stop_sequences=stop_sequences,
            tools=tools,
            tool_choice=tool_choice,
            output_config=output_config,
        )

        if stream:
            chunks = self.client.models.generate_content_stream(
                model=model,
                contents=genai_messages,
                config=genai_config
            )
            return FalseStream(self._stream_events(chunks, model=model, continuation=continuation))

        response = self.client.models.generate_content(
            model=model,
            contents=genai_messages,
            config=genai_config
        )
        return self._from_genai_to_anthropic(response, model=model, continuation=continuation)

    def _prefill_text(self, message: AnthropicMessage) -> str:
        blocks = self._blocks_from_content(message["content"])
        if any(block["type"] != "text" for block in blocks):
            raise ValueError("A final assistant message (prefill) can only contain text")
        return "".join(block["text"] for block in blocks)

    def _from_anthropic_to_genai(self, message: AnthropicMessage, tool_names: dict[str, str]) -> types.Content:
        return types.Content(
            role=_ROLES_TO_GENAI[message["role"]],
            parts=[
                self._part_from_block(block, tool_names)
                for block in self._blocks_from_content(message["content"])
            ],
        )

    def _blocks_from_content(self, content: str | Iterable[AnthropicContentBlock]) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        # accept both plain dicts and the pydantic blocks returned in `Message.content`
        return [block if isinstance(block, dict) else block.model_dump() for block in content]

    def _part_from_block(self, block: dict[str, Any], tool_names: dict[str, str]) -> types.Part:
        match block["type"]:
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
    ) -> anthropic.types.Message:
        candidate = response.candidates[0] if response.candidates else None
        parts = candidate.content.parts if candidate and candidate.content and candidate.content.parts else []

        content: list[anthropic.types.ContentBlock] = []
        for part in parts:
            if part.function_call:
                content.append(self._tool_use_from_function_call(part))
            # skip the model "thought" parts, Anthropic only returns them when thinking is enabled
            elif part.text and not part.thought:
                # merge consecutive text parts into a single block, like Anthropic does
                if content and content[-1].type == "text":
                    content[-1].text += part.text
                else:
                    content.append(anthropic.types.TextBlock(type="text", text=part.text))

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
        return self._message(
            response,
            model=model,
            content=content,
            stop_reason="stop_sequence" if stop_sequence else self._stop_reason(candidate, has_tool_use),
            stop_sequence=stop_sequence,
            usage=self._usage(response.usage_metadata),
        )

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
    ) -> Iterator[anthropic.types.RawMessageStreamEvent]:
        """Translates Gemini stream chunks into the Anthropic raw event sequence:
        message_start, (content_block_start, content_block_delta*, content_block_stop)*, message_delta, message_stop
        """
        last_chunk = None
        index = -1  # index of the current content block
        open_block_type = None  # type of the block that has not been stopped yet, if any
        has_tool_use = False

        def text_events(text: str) -> Iterator[anthropic.types.RawMessageStreamEvent]:
            nonlocal index, open_block_type
            if not text:
                return
            if open_block_type != "text":
                if open_block_type is not None:
                    yield anthropic.types.RawContentBlockStopEvent(type="content_block_stop", index=index)
                index += 1
                open_block_type = "text"
                yield anthropic.types.RawContentBlockStartEvent(
                    type="content_block_start",
                    index=index,
                    content_block=anthropic.types.TextBlock(type="text", text=""),
                )
            yield anthropic.types.RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=index,
                delta=anthropic.types.TextDelta(type="text_delta", text=text),
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
                    if open_block_type is not None:
                        yield anthropic.types.RawContentBlockStopEvent(type="content_block_stop", index=index)
                    index += 1
                    has_tool_use = True
                    block = self._tool_use_from_function_call(part)
                    yield anthropic.types.RawContentBlockStartEvent(
                        type="content_block_start", index=index, content_block=block.model_copy(update={"input": {}})
                    )
                    yield anthropic.types.RawContentBlockDeltaEvent(
                        type="content_block_delta",
                        index=index,
                        delta=anthropic.types.InputJSONDelta(type="input_json_delta", partial_json=json.dumps(block.input)),
                    )
                    yield anthropic.types.RawContentBlockStopEvent(type="content_block_stop", index=index)
                    open_block_type = None
                elif part.text and not part.thought:
                    yield from text_events(continuation.feed(part.text) if continuation else part.text)

            if continuation is not None and continuation.stop_sequence is not None:
                chunks.close()  # the output after a stop sequence is dropped, no need to wait for it
                break

        if last_chunk is None:
            raise RuntimeError("Gemini returned an empty stream")
        if continuation is not None:
            yield from text_events(continuation.flush())
        if open_block_type is not None:
            yield anthropic.types.RawContentBlockStopEvent(type="content_block_stop", index=index)

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
    ) -> types.GenerateContentConfig:
        system_instruction = None
        if system is not None:
            system_instruction = "\n".join(block["text"] for block in self._blocks_from_content(system))
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
            tools=self._genai_tools(tools),
            tool_config=self._genai_tool_config(tool_choice),
            # the exercises run the tools themselves, like with the Anthropic SDK
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def _genai_tools(self, tools: list[AnthropicTool] | None) -> list[types.Tool] | None:
        if not tools:
            return None
        declarations = []
        for tool in tools:
            if tool.get("type", "custom") != "custom":
                raise NotImplementedError(f"Server tool {tool['type']!r} is not supported yet")
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


class FalseAnthropic:
    def __init__(self, api_key: str | None = None):
        self.client = genai.Client(api_key=api_key)
        self.messages = FalseAnthropicMessages(self.client)
