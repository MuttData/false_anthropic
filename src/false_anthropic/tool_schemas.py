"""Schemas for the Anthropic-defined client tools (text editor, bash, memory).

Claude's API takes these tools as just `{"type": "<tool>_<version>", "name": "<name>"}`: they are schema-less,
the schema is built into Claude's model. Gemini needs a regular function declaration, so this module rebuilds
each one from the parameters described in Anthropic's docs:

- https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool
- https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool
- https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool

Each schema is a flat object with a `command` enum (instead of one schema per command), which Gemini follows more
reliably. The tool calls Gemini makes have the same `name` and `input` shape as Claude's, so the client-side
handlers written for the exercises work unchanged.
"""

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ClientToolSchema:
    name: str
    """The name Anthropic requires for the tool, e.g. `str_replace_based_edit_tool`."""
    description: str
    input_schema: dict[str, Any]
    system_prompt: str | None = None
    """Text Anthropic's API adds to the system prompt when the tool is present, if any."""


_PATH = {"type": "string", "description": "Path to the file or directory."}
_VIEW_RANGE = {
    "type": "array",
    "items": {"type": "integer"},
    "minItems": 2,
    "maxItems": 2,
    "description": (
        "Optional for `view` on a file: `[start_line, end_line]`, 1-indexed and inclusive. "
        "`[start_line, -1]` shows everything from `start_line` to the end of the file."
    ),
}


def _text_editor(*, name: str, undo_edit: bool, insert_text_param: str, max_characters: int | None) -> ClientToolSchema:
    commands = ["view", "create", "str_replace", "insert"] + (["undo_edit"] if undo_edit else [])
    description = (
        "Tool for viewing, creating and editing files.\n"
        "* State is persistent across command calls.\n"
        "* `view`: if `path` is a file, shows its content with line numbers (like `cat -n`); "
        "if `path` is a directory, lists the files and directories inside it.\n"
        "* `create`: creates a file at `path` with the content in `file_text`.\n"
        "* `str_replace`: replaces `old_str` with `new_str` in the file at `path`. `old_str` must match EXACTLY one "
        "or more consecutive lines of the file, including whitespace and indentation, and must be unique in the "
        "file: include enough surrounding context to make it unique, or no replacement is made.\n"
        f"* `insert`: inserts `{insert_text_param}` after line `insert_line` of the file at `path` "
        "(`insert_line` 0 inserts at the beginning of the file).\n"
    )
    if undo_edit:
        description += "* `undo_edit`: reverts the last edit made to the file at `path`.\n"
    if max_characters is not None:
        description += (
            f"* File content shown by `view` is truncated to {max_characters} characters, "
            "use `view_range` to see the rest.\n"
        )

    properties: dict[str, Any] = {
        "command": {"type": "string", "enum": commands, "description": "The command to run."},
        "path": _PATH,
        "view_range": _VIEW_RANGE,
        "file_text": {"type": "string", "description": "Required for `create`: the content of the file."},
        "old_str": {"type": "string", "description": "Required for `str_replace`: the exact text to replace."},
        "new_str": {"type": "string", "description": "Required for `str_replace`: the text to replace `old_str` with."},
        "insert_line": {
            "type": "integer",
            "description": "Required for `insert`: the line number after which to insert the text.",
        },
    }
    if insert_text_param == "new_str":
        properties["new_str"]["description"] = (
            "Required for `str_replace`: the text to replace `old_str` with. Required for `insert`: the text to insert."
        )
    else:
        properties[insert_text_param] = {"type": "string", "description": "Required for `insert`: the text to insert."}

    return ClientToolSchema(
        name=name,
        description=description,
        input_schema={"type": "object", "properties": properties, "required": ["command", "path"]},
    )


def _bash() -> ClientToolSchema:
    return ClientToolSchema(
        name="bash",
        description=(
            "Run commands in a bash shell.\n"
            "* The session is persistent across calls: the working directory, environment variables and files "
            "created by a command are still there for the next one.\n"
            "* The output of the command (stdout and stderr) is returned as the result.\n"
            "* Avoid commands that need user input or run forever; run long-lived processes in the background.\n"
            "* Set `restart` to true to restart the bash session, e.g. if it is unresponsive."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to run. Required unless `restart` is true.",
                },
                "restart": {"type": "boolean", "description": "Set to true to restart the bash session."},
            },
        },
    )


# Verbatim from the memory tool docs: the API adds it to the system prompt when the tool is present
_MEMORY_SYSTEM_PROMPT = """\
IMPORTANT: ALWAYS VIEW YOUR MEMORY DIRECTORY BEFORE DOING ANYTHING ELSE.
MEMORY PROTOCOL:
1. Use the `view` command of your `memory` tool to check for earlier progress.
2. ... (work on the task) ...
   - As you make progress, record status / progress / thoughts etc in your memory.
ASSUME INTERRUPTION: Your context window might be reset at any moment, so you risk losing any progress that is not recorded in your memory directory."""


def _memory() -> ClientToolSchema:
    return ClientToolSchema(
        name="memory",
        description=(
            "Memory tool: a directory of files at `/memories` that persists across conversations. Use it to store "
            "what you learn while working (progress, decisions, user preferences) and read it back later.\n"
            "* All paths must be inside `/memories`.\n"
            "* `view`: if `path` is a directory, lists its files and directories up to 2 levels deep; if it is a "
            "file, shows its content with line numbers. Optional `view_range` for long files.\n"
            "* `create`: creates or overwrites the file at `path` with `file_text`.\n"
            "* `str_replace`: replaces `old_str` (must appear exactly once) with `new_str` in the file at `path`; "
            "omit `new_str` to delete `old_str`.\n"
            "* `insert`: inserts `insert_text` after line `insert_line` (0 for the beginning of the file).\n"
            "* `delete`: deletes the file or directory at `path`.\n"
            "* `rename`: renames or moves `old_path` to `new_path`.\n"
            "* You cannot delete or rename the `/memories` directory itself.\n"
            "* Keep the memory directory organized: update or remove outdated files instead of creating new ones."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "str_replace", "insert", "delete", "rename"],
                    "description": "The command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Path inside `/memories`. Required for every command except `rename`.",
                },
                "view_range": _VIEW_RANGE,
                "file_text": {"type": "string", "description": "Required for `create`: the content of the file."},
                "old_str": {"type": "string", "description": "Required for `str_replace`: the exact text to replace."},
                "new_str": {"type": "string", "description": "Optional for `str_replace`: the replacement text."},
                "insert_line": {
                    "type": "integer",
                    "description": "Required for `insert`: the line number after which to insert the text.",
                },
                "insert_text": {"type": "string", "description": "Required for `insert`: the text to insert."},
                "old_path": {"type": "string", "description": "Required for `rename`: the current path."},
                "new_path": {"type": "string", "description": "Required for `rename`: the new path."},
            },
            "required": ["command"],
        },
        system_prompt=_MEMORY_SYSTEM_PROMPT,
    )


# tool `type` -> builds the schema from the tool definition, for the options it supports (e.g. `max_characters`)
CLIENT_TOOLS: dict[str, Callable[[dict[str, Any]], ClientToolSchema]] = {
    "text_editor_20250728": lambda tool: _text_editor(
        name="str_replace_based_edit_tool",
        undo_edit=False,
        insert_text_param="insert_text",
        max_characters=tool.get("max_characters"),
    ),
    "text_editor_20250429": lambda tool: _text_editor(
        name="str_replace_based_edit_tool", undo_edit=False, insert_text_param="insert_text", max_characters=None
    ),
    # the original text editor used `new_str` for `insert` too, and still had `undo_edit`
    "text_editor_20250124": lambda tool: _text_editor(
        name="str_replace_editor", undo_edit=True, insert_text_param="new_str", max_characters=None
    ),
    "bash_20250124": lambda tool: _bash(),
    "memory_20250818": lambda tool: _memory(),
}


def resolve_client_tool(tool: dict[str, Any]) -> ClientToolSchema:
    """Returns the schema for an Anthropic-defined client tool definition, e.g. `{"type": "bash_20250124", "name": "bash"}`."""
    tool_type = tool["type"]
    if tool_type not in CLIENT_TOOLS:
        raise NotImplementedError(
            f"Tool type {tool_type!r} is not supported. "
            f"Supported Anthropic-defined tools: {', '.join(sorted(CLIENT_TOOLS))}"
        )
    schema = CLIENT_TOOLS[tool_type](tool)
    # like Anthropic's API, which only accepts the name the model was trained with
    if tool.get("name", schema.name) != schema.name:
        raise ValueError(f"Tool type {tool_type!r} must be named {schema.name!r}, got {tool['name']!r}")
    return schema
