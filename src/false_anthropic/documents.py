"""Document blocks (PDF, plain text and custom content) and their citations.

Gemini reads PDFs and text, but has no document citations. When citations are enabled, they are emulated like
Anthropic's API does them internally: each document is split into sentences (or blocks, for custom content), the
model is shown them with ids and asked to cite them with markers, and the markers are then parsed into text blocks
with `char_location` / `page_location` / `content_block_location` citations. Since the cited text comes from the
document and not from the model, `cited_text` and the indices are always exact.

See https://platform.claude.com/docs/en/build-with-claude/citations
"""

import io
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

import anthropic
import pypdf
from google.genai import types

from false_anthropic import sources

CITATIONS_SYSTEM_PROMPT = """\
Some documents in this conversation are split into numbered sentences or blocks, each with an id like "0.3" \
(the document index, then the sentence or block index).
When a statement in your answer uses information from these documents, add a citation marker right after the \
statement, with the ids of the sentences or blocks that support it, like [cite: 0.3] or [cite: 0.3, 0.4]. \
Cite the exact supporting sentences or blocks, and only ids that exist. Don't mention the ids otherwise, and don't \
add a list of sources at the end."""

# a citation marker, with the spaces before it (they would be left dangling in the text)
_CITE_MARKER = re.compile(r"[ \t]*\[cite:\s*([0-9.,\s]*)\]")
# the end of a sentence: punctuation (with closing quotes or brackets) followed by whitespace, or a blank line
_SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*\s+|\n\s*\n")
# where a statement of the answer starts: at the beginning, after the end of a sentence or after a line break
_STATEMENT_START = re.compile(r"[.!?]+[\"')\]]*\s+|\n\s*|^\s+")


@dataclass
class _Chunk:
    """A citable piece of a document: a sentence, or a block of a custom content document."""

    text: str
    start: int  # character index for plain text, page number for PDFs, block index for custom content
    end: int  # exclusive


@dataclass
class Document:
    index: int
    title: str | None
    context: str | None
    citations: bool
    kind: Literal["pdf", "text", "content"]
    pdf: bytes | None = None
    text: str | None = None
    content: list[dict[str, Any]] = field(default_factory=list)  # the blocks of a custom content document
    chunks: list[_Chunk] = field(default_factory=list)

    def genai_parts(self) -> list[types.Part]:
        """The Gemini parts the document is sent as."""
        header = f'<document index="{self.index}"' + (f' title="{self.title}"' if self.title else "") + ">\n"
        if self.context:
            header += f"<context>{self.context}</context>\n"

        match self.kind:
            case "pdf":
                pdf_part = types.Part.from_bytes(data=self.pdf, mime_type="application/pdf")
                if not self.citations and not self.title and not self.context:
                    return [pdf_part]
                parts = [types.Part.from_text(text=header), pdf_part]
                closing = "</document>"
                if self.citations:
                    closing = "The text of this PDF, for citations:\n" + self._chunk_listing() + closing
                return [*parts, types.Part.from_text(text=closing)]
            case "text":
                body = self._chunk_listing() if self.citations else self.text + "\n"
                return [types.Part.from_text(text=header + body + "</document>")]
            case "content":
                parts = [types.Part.from_text(text=header)]
                chunk_index = 0  # ids are positions in `chunks`, which leave the images out
                for block in self.content:
                    if block["type"] == "image":
                        data, media_type = sources.image_bytes(block["source"])
                        parts.append(types.Part.from_bytes(data=data, mime_type=media_type))
                    elif self.citations:
                        block_id = f"{self.index}.{chunk_index}"
                        parts.append(types.Part.from_text(text=f'<block id="{block_id}">{block["text"]}</block>\n'))
                        chunk_index += 1
                    else:
                        parts.append(types.Part.from_text(text=block["text"] + "\n"))
                return [*parts, types.Part.from_text(text="</document>")]

    def _chunk_listing(self) -> str:
        if self.kind == "pdf":
            return "".join(
                f'<sentence id="{self.index}.{i}" page="{chunk.start}">{chunk.text.strip()}</sentence>\n'
                for i, chunk in enumerate(self.chunks)
            )
        return "".join(f'<sentence id="{self.index}.{i}">{chunk.text}</sentence>\n' for i, chunk in enumerate(self.chunks))

    def citation(self, chunk_indices: list[int]) -> anthropic.types.TextCitation:
        """The citation for a run of consecutive chunks."""
        first, last = self.chunks[chunk_indices[0]], self.chunks[chunk_indices[-1]]
        common = {"document_index": self.index, "document_title": self.title, "file_id": None}
        match self.kind:
            case "text":
                return anthropic.types.CitationCharLocation(
                    type="char_location",
                    cited_text=self.text[first.start:last.end],
                    start_char_index=first.start,
                    end_char_index=last.end,
                    **common,
                )
            case "pdf":
                return anthropic.types.CitationPageLocation(
                    type="page_location",
                    cited_text="".join(self.chunks[i].text for i in chunk_indices),
                    start_page_number=first.start,
                    end_page_number=last.end,
                    **common,
                )
            case "content":
                return anthropic.types.CitationContentBlockLocation(
                    type="content_block_location",
                    cited_text="".join(self.chunks[i].text for i in chunk_indices),
                    start_block_index=first.start,
                    end_block_index=last.end,
                    **common,
                )


def _sentences(text: str, offset: int = 0) -> list[tuple[str, int, int]]:
    """Splits text into sentences, keeping the whitespace after each one (like Anthropic's `cited_text`)."""
    sentences, start = [], 0
    for match in _SENTENCE_END.finditer(text):
        sentences.append((text[start:match.end()], offset + start, offset + match.end()))
        start = match.end()
    if start < len(text):
        sentences.append((text[start:], offset + start, offset + len(text)))
    return [sentence for sentence in sentences if sentence[0].strip()]


def _load(block: dict[str, Any], index: int) -> Document:
    source = block["source"]
    document = Document(
        index=index,
        title=block.get("title"),
        context=block.get("context"),
        citations=bool((block.get("citations") or {}).get("enabled")),
        kind="pdf",
    )
    match source["type"]:
        case "base64" | "url":
            document.pdf = sources.pdf_bytes(source)
            if document.citations:
                pages = pypdf.PdfReader(io.BytesIO(document.pdf)).pages
                for number, page in enumerate(pages, start=1):  # page numbers are 1-indexed
                    for text, _, _ in _sentences(page.extract_text() or ""):
                        document.chunks.append(_Chunk(text=text, start=number, end=number + 1))
        case "text":
            document.kind, document.text = "text", source["data"]
            document.chunks = [_Chunk(text, start, end) for text, start, end in _sentences(document.text)]
        case "content":
            document.kind = "content"
            content = source["content"]
            document.content = [{"type": "text", "text": content}] if isinstance(content, str) else list(content)
            # the blocks are used as they are, without further chunking; images can't be cited
            document.chunks = [
                _Chunk(text=block["text"], start=i, end=i + 1)
                for i, block in enumerate(document.content)
                if block["type"] == "text"
            ]
        case other:
            # `file` sources reference Anthropic's Files API, which has no equivalent here
            raise NotImplementedError(f"Document source type {other!r} is not supported yet")
    return document


def load_documents(blocks: Iterable[dict[str, Any]]) -> list[Document]:
    """Loads the documents among the blocks of a request, in order: their index is their position in the list."""
    documents = [_load(block, index) for index, block in enumerate(b for b in blocks if b["type"] == "document")]
    if len({document.citations for document in documents}) > 1:
        # like Anthropic's API, which returns a 400 error
        raise ValueError("Citations must be enabled on all or none of the documents in a request")
    return documents


def cited_text_blocks(text: str, documents: list[Document]) -> list[anthropic.types.TextBlock]:
    """Splits text with citation markers into text blocks, with the citations on the text before each marker."""
    blocks: list[anthropic.types.TextBlock] = []

    def add(text: str, citations: list[anthropic.types.TextCitation] | None = None) -> None:
        if not text:
            return
        if not citations and blocks and not blocks[-1].citations:
            blocks[-1].text += text  # merge consecutive uncited text
        else:
            blocks.append(anthropic.types.TextBlock(type="text", text=text, citations=citations or None))

    position = 0
    for marker in _CITE_MARKER.finditer(text):
        segment = text[position:marker.start()]
        position = marker.end()
        citations = _citations(marker.group(1), documents)
        # the cited statement starts after the last sentence end, the text before it is not cited
        sentence_ends = list(_STATEMENT_START.finditer(segment))
        start = sentence_ends[-1].end() if sentence_ends else 0
        add(segment[:start])
        add(segment[start:], citations)
    add(text[position:])
    return blocks


def _citations(ids: str, documents: list[Document]) -> list[anthropic.types.TextCitation]:
    """The citations for the ids in a marker, merging consecutive sentences of a document into one citation."""
    by_document: dict[int, set[int]] = {}
    for chunk_id in re.findall(r"(\d+)\.(\d+)", ids):
        document_index, chunk_index = map(int, chunk_id)
        if document_index < len(documents) and chunk_index < len(documents[document_index].chunks):
            by_document.setdefault(document_index, set()).add(chunk_index)  # ids that don't exist are dropped

    citations = []
    for document_index, chunk_indices in sorted(by_document.items()):
        run: list[int] = []
        for chunk_index in sorted(chunk_indices):
            if run and chunk_index != run[-1] + 1:
                citations.append(documents[document_index].citation(run))
                run = []
            run.append(chunk_index)
        citations.append(documents[document_index].citation(run))
    return citations
