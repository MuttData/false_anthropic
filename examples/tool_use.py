"""Tool use loop: the model asks for tools, we run them and send the results back."""

from dotenv import load_dotenv

from false_anthropic import FalseAnthropic

load_dotenv()

client = FalseAnthropic()
model = "gemini-3.5-flash-lite"

tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]


def get_weather(city: str) -> str:
    return f"22C and sunny in {city}"


messages = [{"role": "user", "content": "What's the weather in Paris and in Tokyo?"}]
while True:
    response = client.messages.create(model=model, max_tokens=1000, messages=messages, tools=tools)
    messages.append({"role": "assistant", "content": response.content})
    if response.stop_reason != "tool_use":
        break

    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            print(f"-> {block.name}({block.input})")
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": get_weather(**block.input)}
            )
    messages.append({"role": "user", "content": tool_results})

print(response.content[0].text)

# force a specific tool
response = client.messages.create(
    model=model,
    max_tokens=1000,
    messages=[{"role": "user", "content": "hello"}],
    tools=tools,
    tool_choice={"type": "tool", "name": "get_weather"},
)
print(response.stop_reason, response.content[-1].input)
