"""Structured outputs, with a raw JSON schema and with a pydantic model."""

import json
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from false_anthropic import FalseAnthropic

load_dotenv()

client = FalseAnthropic()
model = "gemini-3.5-flash-lite"

# raw JSON schema
response = client.messages.create(
    model=model,
    max_tokens=500,
    messages=[{"role": "user", "content": "Extract: John Smith, john@x.com, wants the Pro plan"}],
    output_config={
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "plan": {"type": "string"},
                },
                "required": ["name", "email", "plan"],
                "additionalProperties": False,
            },
        }
    },
)
print(json.loads(response.content[0].text))


class Ingredient(BaseModel):
    name: str
    quantity: str


class Recipe(BaseModel):
    title: str
    difficulty: Literal["easy", "medium", "hard"]
    servings: int = Field(ge=1, le=12)
    ingredients: list[Ingredient]
    notes: str | None = None


# pydantic model
parsed = client.messages.parse(
    model=model,
    max_tokens=1000,
    output_format=Recipe,
    messages=[{"role": "user", "content": "Give me a simple pancake recipe"}],
)
print(parsed.parsed_output)

# pydantic model while streaming
with client.messages.stream(
    model=model,
    max_tokens=1000,
    output_format=Recipe,
    messages=[{"role": "user", "content": "Give me a simple omelette recipe"}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
    print()
    print(stream.get_final_message().parsed_output)
