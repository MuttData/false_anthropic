import base64

import httpx
import pytest
from google.genai import types

from false_anthropic import FalseAnthropic

PNG = b"\x89PNG\r\n\x1a\nfake-image"
PNG_BASE64 = base64.standard_b64encode(PNG).decode()
IMAGE = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_BASE64}}

RESPONSE = types.GenerateContentResponse(
    candidates=[
        types.Candidate(
            finish_reason=types.FinishReason.STOP,
            content=types.Content(role="model", parts=[types.Part(text="A house with trees")]),
        )
    ],
)


class FakeModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append(contents)
        return RESPONSE


@pytest.fixture
def client():
    client = FalseAnthropic(api_key="fake")
    client.messages.client = type("FakeClient", (), {"models": FakeModels()})()
    return client


def _sent_contents(client, messages):
    client.messages.create(model="gemini-test", max_tokens=100, messages=messages)
    return client.messages.client.models.calls[-1]


def test_base64_image(client):
    (content,) = _sent_contents(client, [{"role": "user", "content": [IMAGE, {"type": "text", "text": "Describe it"}]}])
    image, text = content.parts
    assert image.inline_data.data == PNG
    assert image.inline_data.mime_type == "image/png"
    assert text.text == "Describe it"


def test_url_image_is_downloaded(client, monkeypatch):
    requested = []

    def fake_get(url, **kwargs):
        requested.append(url)
        return httpx.Response(
            200, content=PNG, headers={"content-type": "image/png"}, request=httpx.Request("GET", url)
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    url = "https://example.com/house.png"
    (content,) = _sent_contents(
        client, [{"role": "user", "content": [{"type": "image", "source": {"type": "url", "url": url}}]}]
    )
    assert requested == [url]
    assert content.parts[0].inline_data.data == PNG
    assert content.parts[0].inline_data.mime_type == "image/png"


def test_url_that_is_not_an_image(client, monkeypatch):
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kwargs: httpx.Response(
            200, content=b"<html>", headers={"content-type": "text/html"}, request=httpx.Request("GET", url)
        ),
    )
    with pytest.raises(ValueError, match="text/html"):
        _sent_contents(
            client,
            [{"role": "user", "content": [{"type": "image", "source": {"type": "url", "url": "https://example.com"}}]}],
        )


def test_file_source_is_not_supported(client):
    image = {"type": "image", "source": {"type": "file", "file_id": "file_123"}}
    with pytest.raises(NotImplementedError, match="'file'"):
        _sent_contents(client, [{"role": "user", "content": [image]}])


def test_image_in_tool_result(client):
    tool_use = {"type": "tool_use", "id": "toolu_1", "name": "screenshot", "input": {}}
    tool_result = {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "content": [{"type": "text", "text": "Here it is"}, IMAGE],
    }
    contents = _sent_contents(
        client,
        [
            {"role": "user", "content": "Take a screenshot"},
            {"role": "assistant", "content": [tool_use]},
            {"role": "user", "content": [tool_result]},
        ],
    )
    function_response = contents[2].parts[0].function_response
    assert function_response.response == {"output": "Here it is"}
    assert function_response.parts[0].inline_data.data == PNG
    assert function_response.parts[0].inline_data.mime_type == "image/png"
