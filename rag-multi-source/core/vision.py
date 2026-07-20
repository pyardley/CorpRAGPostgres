"""
Image captioning for multi-modal ingestion (README "Possible
enhancements" — Multi-modal ingestion).

Images embedded in Confluence pages or attached to Jira tickets are
invisible to the indexer otherwise. :func:`caption_image` describes an
image with a vision-capable LLM so the caption becomes a normal
searchable chunk (see `app.ingestion.base.BaseIngestor._caption_image_chunks`).

Raises on failure — callers decide how to handle it (the ingestion
pipeline fails open, skipping just that image).
"""

from __future__ import annotations

import base64

from langchain_core.messages import HumanMessage

from core.llm import get_vision_llm

_CAPTION_PROMPT = (
    "Describe this image for a corporate knowledge-base search index. "
    "Focus on any text, diagrams, error messages, architecture, or data "
    "visible in the image. Be concise but specific — this caption is the "
    "only representation of the image that will be searchable."
)


def caption_image(image_bytes: bytes, mime_type: str, alt_text: str = "") -> str:
    prompt = _CAPTION_PROMPT
    if alt_text:
        prompt += f"\n\nExisting alt text (may be empty or unhelpful): {alt_text!r}"

    encoded = base64.b64encode(image_bytes).decode()
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
            },
        ]
    )
    response = get_vision_llm().invoke([message])
    content = response.content
    return content if isinstance(content, str) else str(content)
