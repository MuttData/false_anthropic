import typing

import pytest
from anthropic.types.beta import BetaMemoryTool20250818Command

from false_anthropic.client import FalseAnthropicMessages
from false_anthropic.tool_schemas import CLIENT_TOOLS, resolve_client_tool


def _schema(tool_type, **options):
    return CLIENT_TOOLS[tool_type]({"type": tool_type, **options})


def test_memory_schema_matches_sdk_command_types():
    command_models = typing.get_args(typing.get_args(BetaMemoryTool20250818Command)[0])
    sdk_fields = {field for model in command_models for field in model.model_fields}
    sdk_commands = {typing.get_args(model.model_fields["command"].annotation)[0] for model in command_models}

    schema = _schema("memory_20250818").input_schema
    assert set(schema["properties"]) == sdk_fields
    assert set(schema["properties"]["command"]["enum"]) == sdk_commands


# inputs from the examples in Anthropic's docs
@pytest.mark.parametrize(
    ("tool_type", "tool_input"),
    [
        ("text_editor_20250728", {"command": "view", "path": "primes.py"}),
        ("text_editor_20250728", {"command": "view", "path": "primes.py", "view_range": [1, -1]}),
        ("text_editor_20250728", {"command": "str_replace", "path": "primes.py", "old_str": "a", "new_str": "b"}),
        ("text_editor_20250728", {"command": "create", "path": "test_primes.py", "file_text": "import unittest"}),
        ("text_editor_20250728", {"command": "insert", "path": "primes.py", "insert_line": 0, "insert_text": "x"}),
        ("text_editor_20250429", {"command": "insert", "path": "primes.py", "insert_line": 0, "insert_text": "x"}),
        ("text_editor_20250124", {"command": "insert", "path": "primes.py", "insert_line": 0, "new_str": "x"}),
        ("text_editor_20250124", {"command": "undo_edit", "path": "primes.py"}),
        ("bash_20250124", {"command": "ls -la"}),
        ("bash_20250124", {"restart": True}),
        ("memory_20250818", {"command": "view", "path": "/memories"}),
        ("memory_20250818", {"command": "rename", "old_path": "/memories/a.txt", "new_path": "/memories/b.txt"}),
    ],
)
def test_documented_inputs_fit_the_schema(tool_type, tool_input):
    schema = _schema(tool_type).input_schema
    assert set(tool_input) <= set(schema["properties"])
    assert set(schema.get("required", [])) <= set(tool_input)
    command_schema = schema["properties"]["command"]
    if "enum" in command_schema:  # bash's `command` is free text
        assert tool_input["command"] in command_schema["enum"]


def test_text_editor_versions():
    assert "undo_edit" not in _schema("text_editor_20250728").input_schema["properties"]["command"]["enum"]
    assert "undo_edit" in _schema("text_editor_20250124").input_schema["properties"]["command"]["enum"]
    assert "insert_text" not in _schema("text_editor_20250124").input_schema["properties"]
    assert "10000 characters" in _schema("text_editor_20250728", max_characters=10000).description


def test_resolve_checks_the_name():
    assert resolve_client_tool({"type": "bash_20250124", "name": "bash"}).name == "bash"
    with pytest.raises(ValueError, match="must be named 'str_replace_based_edit_tool'"):
        resolve_client_tool({"type": "text_editor_20250728", "name": "str_replace_editor"})


def test_resolve_rejects_unsupported_tools():
    with pytest.raises(NotImplementedError, match="web_search_20250305"):
        resolve_client_tool({"type": "web_search_20250305", "name": "web_search"})


def test_client_replaces_tools_and_adds_the_memory_prompt():
    messages = FalseAnthropicMessages(client=None)
    custom = {"name": "get_weather", "description": "Get weather", "input_schema": {"type": "object"}}
    config = messages._from_anthropic_to_genai_config(
        max_tokens=100,
        system="Be brief",
        temperature=None,
        stop_sequences=None,
        tools=[custom, {"type": "memory_20250818", "name": "memory"}],
        tool_choice=None,
        output_config=None,
    )
    declarations = config.tools[0].function_declarations
    assert [d.name for d in declarations] == ["get_weather", "memory"]
    assert declarations[1].parameters_json_schema["properties"]["command"]["enum"][0] == "view"
    assert config.system_instruction.startswith("Be brief\n\nIMPORTANT: ALWAYS VIEW YOUR MEMORY DIRECTORY")
