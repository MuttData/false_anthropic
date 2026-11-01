"""Anthropic's web search server tool, run with Gemini's Google Search grounding.

Request: the `web_search_*` tool definition becomes Gemini's `google_search` tool.

Response: Gemini returns the answer text plus `grounding_metadata`, which is mapped to Anthropic's blocks:

- each of `web_search_queries` becomes a `server_tool_use` block followed by its `web_search_tool_result` block,
  with `grounding_chunks` as the `web_search_result`s. Gemini does not say which query found which result,
  so all of them go to the first query's result block.
- `grounding_supports` (answer segments backed by some chunks) split the answer into text blocks, the supported
  ones carrying `web_search_result_location` citations, like Anthropic does.
"""

import uuid
import warnings
from typing import Any

import anthropic
from google.genai import types

WEB_SEARCH_TOOL_TYPES = {"web_search_20250305", "web_search_20260209", "web_search_20260318"}

_UNSUPPORTED_OPTIONS = ("max_uses", "allowed_domains", "blocked_domains", "user_location")


def is_web_search_tool(tool: dict[str, Any]) -> bool:
    return tool.get("type") in WEB_SEARCH_TOOL_TYPES


def google_search_tool(tool: dict[str, Any]) -> types.Tool:
    if tool.get("name", "web_search") != "web_search":
        raise ValueError(f"Tool type {tool['type']!r} must be named 'web_search', got {tool['name']!r}")
    unsupported = [option for option in _UNSUPPORTED_OPTIONS if tool.get(option)]
    if unsupported:
        warnings.warn(f"Gemini's Google Search does not support {', '.join(unsupported)}, they are ignored", stacklevel=5)
    return types.Tool(google_search=types.GoogleSearch())


def has_web_search(grounding: types.GroundingMetadata | None) -> bool:
    return bool(grounding and grounding.web_search_queries)


def search_blocks(grounding: types.GroundingMetadata) -> list[anthropic.types.ContentBlock]:
    """The `server_tool_use` / `web_search_tool_result` block pairs for the searches Gemini made."""
    results = [
        anthropic.types.WebSearchResultBlock(
            type="web_search_result",
            url=chunk.web.uri,
            title=chunk.web.title or chunk.web.domain or chunk.web.uri,
            encrypted_content="",  # Gemini does not return the page content
            page_age=None,
        )
        for chunk in grounding.grounding_chunks or []
        if chunk.web and chunk.web.uri
    ]
    blocks: list[anthropic.types.ContentBlock] = []
    for i, query in enumerate(grounding.web_search_queries):
        tool_use_id = f"srvtoolu_{uuid.uuid4().hex}"
        blocks.append(
            anthropic.types.ServerToolUseBlock(
                type="server_tool_use", id=tool_use_id, name="web_search", input={"query": query}
            )
        )
        blocks.append(
            anthropic.types.WebSearchToolResultBlock(
                type="web_search_tool_result", tool_use_id=tool_use_id, content=results if i == 0 else []
            )
        )
    return blocks


def cited_text_blocks(part_texts: list[str], grounding: types.GroundingMetadata) -> list[anthropic.types.TextBlock]:
    """Splits the answer into text blocks, with citations on the segments Gemini grounded.

    `part_texts` has the output text of each response part ("" for thoughts and non-text parts), since Gemini's
    segment offsets are UTF-8 byte offsets relative to the part at `part_index`.
    """
    part_offsets = [0]
    for text in part_texts:
        part_offsets.append(part_offsets[-1] + len(text.encode()))
    answer = "".join(part_texts).encode()
    chunks = grounding.grounding_chunks or []

    spans = []
    for support in grounding.grounding_supports or []:
        segment = support.segment
        part_index = (segment.part_index or 0) if segment else 0
        if segment is None or segment.end_index is None or part_index >= len(part_texts):
            continue
        start = part_offsets[part_index] + (segment.start_index or 0)
        end = part_offsets[part_index] + segment.end_index
        cited_text = answer[start:end].decode(errors="ignore")
        citations = [
            anthropic.types.CitationsWebSearchResultLocation(
                type="web_search_result_location",
                url=chunks[i].web.uri,
                title=chunks[i].web.title,
                # Anthropic quotes the source page here, Gemini only tells which part of the answer it supports
                cited_text=cited_text,
                encrypted_index="",
            )
            for i in support.grounding_chunk_indices or []
            if i < len(chunks) and chunks[i].web and chunks[i].web.uri
        ]
        if citations:
            spans.append((start, end, citations))

    blocks = []
    position = 0
    for start, end, citations in sorted(spans, key=lambda span: span[0]):
        if start < position:
            continue  # overlapping segments, keep the first one
        if start > position:
            blocks.append(anthropic.types.TextBlock(type="text", text=answer[position:start].decode(errors="ignore")))
        blocks.append(
            anthropic.types.TextBlock(type="text", text=answer[start:end].decode(errors="ignore"), citations=citations)
        )
        position = end
    if position < len(answer):
        blocks.append(anthropic.types.TextBlock(type="text", text=answer[position:].decode(errors="ignore")))
    return [block for block in blocks if block.text]


def server_tool_usage(grounding: types.GroundingMetadata | None) -> anthropic.types.ServerToolUsage | None:
    if not has_web_search(grounding):
        return None
    return anthropic.types.ServerToolUsage(web_search_requests=len(grounding.web_search_queries), web_fetch_requests=0)
