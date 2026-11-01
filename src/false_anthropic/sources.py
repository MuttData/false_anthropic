"""Loads the data of image and document `source`s, which Gemini takes as inline bytes."""

import base64
from typing import Any

import httpx

# some hosts (e.g. Wikimedia) reject requests without a descriptive user agent
_USER_AGENT = "false-anthropic (+https://github.com/MuttData/false_anthropic)"


def download(url: str) -> tuple[bytes, str]:
    """Downloads a URL source, since Gemini can't fetch arbitrary URLs. Returns the data and its media type."""
    response = httpx.get(url, headers={"User-Agent": _USER_AGENT}, follow_redirects=True, timeout=30)
    response.raise_for_status()
    return response.content, response.headers.get("content-type", "").split(";")[0].strip()


def image_bytes(source: dict[str, Any]) -> tuple[bytes, str]:
    """Returns the data and media type of an image block's source."""
    match source["type"]:
        case "base64":
            return base64.b64decode(source["data"]), source["media_type"]
        case "url":
            data, media_type = download(source["url"])
            if not media_type.startswith("image/"):
                raise ValueError(f"The image URL {source['url']!r} returned {media_type or 'no'} content type")
            return data, media_type
        case other:
            # `file` sources reference Anthropic's Files API, which has no equivalent here
            raise NotImplementedError(f"Image source type {other!r} is not supported yet")


def pdf_bytes(source: dict[str, Any]) -> bytes:
    """Returns the data of a PDF document block's source."""
    match source["type"]:
        case "base64":
            data = source["data"]
            return base64.b64decode(data) if isinstance(data, str) else data
        case "url":
            # servers often send PDFs as `application/octet-stream`, so check the content instead
            data, _ = download(source["url"])
            if not data.startswith(b"%PDF"):
                raise ValueError(f"The document URL {source['url']!r} did not return a PDF")
            return data
        case other:
            raise NotImplementedError(f"PDF source type {other!r} is not supported yet")
