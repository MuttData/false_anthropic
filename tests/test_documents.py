import base64

import httpx
import pytest
from google.genai import types

from false_anthropic import FalseAnthropic
from false_anthropic.documents import CITATIONS_SYSTEM_PROMPT


def _pdf(*pages: str) -> bytes:
    """A minimal PDF with one line of text per page."""
    count = len(pages)
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{' '.join(f'{4 + 2 * i} 0 R' for i in range(count))}] /Count {count} >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(pages):
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {5 + 2 * i} 0 R "
            "/Resources << /Font << /F1 3 0 R >> >> >>"
        )
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    out, offsets = "%PDF-1.4\n", []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n" + "".join(f"{o:010d} 00000 n \n" for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    return out.encode()


PDF = _pdf("Earth is the third planet. It has one moon.", "Mars is red. It has two moons.")
PDF_DOCUMENT = {
    "type": "document",
    "source": {"type": "base64", "media_type": "application/pdf", "data": base64.standard_b64encode(PDF).decode()},
}
TEXT = "The grass is green. The sky is blue."
TEXT_DOCUMENT = {
    "type": "document",
    "source": {"type": "text", "media_type": "text/plain", "data": TEXT},
    "title": "My Document",
    "context": "This is a trustworthy document.",
}
CITED = {"citations": {"enabled": True}}


class FakeModels:
    def __init__(self, answer: str = "OK"):
        self.answer = answer
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append((contents, config))
        return types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    finish_reason=types.FinishReason.STOP,
                    content=types.Content(role="model", parts=[types.Part(text=self.answer)]),
                )
            ],
        )


def _client(answer: str = "OK") -> FalseAnthropic:
    client = FalseAnthropic(api_key="fake")
    client.messages.client = type("FakeClient", (), {"models": FakeModels(answer)})()
    return client


def _create(client, *blocks, **kwargs):
    return client.messages.create(
        model="gemini-test",
        max_tokens=100,
        messages=[{"role": "user", "content": [*blocks, {"type": "text", "text": "Question?"}]}],
        **kwargs,
    )


def test_pdf_without_citations_is_sent_as_is():
    client = _client()
    _create(client, PDF_DOCUMENT)
    (content,), config = client.messages.client.models.calls[0]
    pdf, question = content.parts
    assert pdf.inline_data.data == PDF and pdf.inline_data.mime_type == "application/pdf"
    assert question.text == "Question?"
    assert config.system_instruction is None


def test_pdf_url_is_downloaded(monkeypatch):
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kwargs: httpx.Response(
            200, content=PDF, headers={"content-type": "application/octet-stream"}, request=httpx.Request("GET", url)
        ),
    )
    client = _client()
    _create(client, {"type": "document", "source": {"type": "url", "url": "https://example.com/earth.pdf"}})
    (content,), _ = client.messages.client.models.calls[0]
    assert content.parts[0].inline_data.data == PDF


def test_text_document_with_title_and_context():
    client = _client()
    _create(client, TEXT_DOCUMENT)
    (content,), _ = client.messages.client.models.calls[0]
    text = content.parts[0].text
    assert 'title="My Document"' in text and "This is a trustworthy document." in text and TEXT in text


def test_citations_on_text_document():
    answer = "The grass is green [cite: 0.0].\nThe sky is blue[cite: 0.1]. Both are colors[cite: 0.0, 0.1, 7.0]."
    client = _client(answer)
    message = _create(client, {**TEXT_DOCUMENT, **CITED})

    _, config = client.messages.client.models.calls[0]
    assert config.system_instruction == CITATIONS_SYSTEM_PROMPT
    texts = [block.text for block in message.content]
    assert texts == ["The grass is green", ".\n", "The sky is blue", ". ", "Both are colors", "."]

    grass = message.content[0].citations[0]
    assert grass.type == "char_location"
    assert (grass.cited_text, grass.start_char_index, grass.end_char_index) == ("The grass is green. ", 0, 20)
    assert grass.document_index == 0 and grass.document_title == "My Document"
    assert message.content[1].citations is None
    # consecutive sentences are one citation, and ids that don't exist are dropped
    (both,) = message.content[4].citations
    assert (both.cited_text, both.start_char_index, both.end_char_index) == (TEXT, 0, 36)


def test_citations_on_pdf_pages():
    client = _client("Mars has two moons [cite: 1.3].")
    message = _create(client, {**TEXT_DOCUMENT, **CITED}, {**PDF_DOCUMENT, **CITED})

    (contents,), _ = client.messages.client.models.calls[0]
    listing = "".join(part.text or "" for part in contents.parts)
    assert '<sentence id="1.3" page="2">It has two moons.</sentence>' in listing
    citation = message.content[0].citations[0]
    assert citation.type == "page_location"
    assert citation.document_index == 1
    assert (citation.start_page_number, citation.end_page_number) == (2, 3)
    assert citation.cited_text == "It has two moons."


def test_citations_on_custom_content():
    document = {
        "type": "document",
        "source": {
            "type": "content",
            "content": [{"type": "text", "text": "First chunk"}, {"type": "text", "text": "Second chunk"}],
        },
        **CITED,
    }
    message = _create(_client("A claim [cite: 0.0, 0.1]"), document)
    citation = message.content[0].citations[0]
    assert citation.type == "content_block_location"
    assert (citation.start_block_index, citation.end_block_index) == (0, 2)
    assert citation.cited_text == "First chunkSecond chunk"


def test_citations_must_be_on_all_or_no_documents():
    with pytest.raises(ValueError, match="all or none"):
        _create(_client(), {**TEXT_DOCUMENT, **CITED}, PDF_DOCUMENT)


def test_citations_and_structured_outputs_are_incompatible():
    with pytest.raises(ValueError, match="structured outputs"):
        _create(
            _client(),
            {**TEXT_DOCUMENT, **CITED},
            output_config={"format": {"type": "json_schema", "schema": {"type": "object"}}},
        )


def test_stream_replays_citations():
    client = _client("The grass is green [cite: 0.0].")
    expected = _create(client, {**TEXT_DOCUMENT, **CITED})
    with client.messages.stream(
        model="gemini-test",
        max_tokens=100,
        messages=[{"role": "user", "content": [{**TEXT_DOCUMENT, **CITED}, {"type": "text", "text": "Question?"}]}],
    ) as stream:
        events = list(stream)
        final = stream.get_final_message()
    assert "citations_delta" in [e.delta.type for e in events if e.type == "content_block_delta"]
    assert [(b.text, b.citations) for b in final.content] == [(b.text, b.citations) for b in expected.content]


def test_cited_answer_can_be_sent_back():
    client = _client("The grass is green [cite: 0.0].")
    message = _create(client, {**TEXT_DOCUMENT, **CITED})
    client.messages.create(
        model="gemini-test",
        max_tokens=100,
        messages=[
            {"role": "user", "content": [{**TEXT_DOCUMENT, **CITED}, {"type": "text", "text": "Question?"}]},
            {"role": "assistant", "content": message.content},
            {"role": "user", "content": "Thanks"},
        ],
    )
    contents, _ = client.messages.client.models.calls[-1]
    assert "".join(part.text for part in contents[1].parts) == "The grass is green."
