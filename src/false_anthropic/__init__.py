"""Anthropic SDK compatible interface for the Google GenAI SDK.

Lets you follow the Anthropic certification exercises with a free tier Gemini API key:

    from false_anthropic import FalseAnthropic

    client = FalseAnthropic()  # reads GEMINI_API_KEY / GOOGLE_API_KEY
    message = client.messages.create(
        model="gemini-3.5-flash-lite",
        max_tokens=1000,
        messages=[{"role": "user", "content": "Hi"}],
    )
    print(message.content[0].text)
"""

from false_anthropic.client import FalseAnthropic, FalseAnthropicMessages, FalseStream

__all__ = ["FalseAnthropic", "FalseAnthropicMessages", "FalseStream"]
