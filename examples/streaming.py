"""Streaming, both with raw events and with the `messages.stream()` helper."""

from dotenv import load_dotenv

from false_anthropic import FalseAnthropic

load_dotenv()

client = FalseAnthropic()
model = "gemini-3.5-flash-lite"

# raw events
stream = client.messages.create(
    model=model,
    max_tokens=500,
    messages=[{"role": "user", "content": "Write a 3 sentence story about a cat"}],
    stream=True,
)
for event in stream:
    print(event.type)

# helper with text deltas and the final accumulated message
with client.messages.stream(
    model=model,
    max_tokens=500,
    system="Be brief",
    messages=[{"role": "user", "content": "Count from 1 to 10 in words"}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
    print()
    final = stream.get_final_message()

print(final.stop_reason, final.usage)
