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
    "Describe this image for a corporate knowledge-base search index. This "
    "caption is the ONLY representation of the image that will be "
    "searchable — nobody will ever see the image itself, only your text.\n\n"
    "If the image contains a table, log output, code, or any other "
    "structured/tabular data: transcribe it VERBATIM, in full — every row, "
    "every column value, every timestamp and number exactly as shown. Do "
    "not summarize or describe the columns instead of giving the values; a "
    "reader must be able to find a specific cell's exact value from your "
    "transcription alone. Use a markdown table or code block for this part.\n\n"
    "If the image is a diagram, screenshot, or architecture drawing "
    "without tabular data, describe it concisely instead — components, "
    "labels, error messages, and relationships shown.\n\n"
    "Do both if the image has both."
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
