"""
LLM-extracted entity edges (README "Possible enhancements" —
Lightweight entity graph, GraphRAG-inspired).

Deterministic edges (Jira assignee/reporter, Git commit author — see
each ingestor) come from structured fields already fetched, at zero
extra cost. :func:`extract_entities` adds an optional pass over free
text (ticket descriptions/comments, commit messages) to pull out
relationships a structured field can't capture, e.g. "this ticket
mentions service X depends on service Y".

Fails OPEN — unlike `core.live_acl`, this is a retrieval-quality/graph-
richness knob, not an authorization boundary. Any error (bad
structured-output parse, model error, timeout) is logged and `[]` is
returned, so ingestion of that resource's chunks and deterministic
edges is never blocked by an extraction failure.
"""

from __future__ import annotations

from loguru import logger
from pydantic import BaseModel, Field

from app.config import settings
from core.llm import get_entity_extraction_llm

_SYSTEM_PROMPT = (
    "Extract notable entities and relationships mentioned in this text "
    "from a corporate knowledge base (Jira tickets, Git commits) as "
    "(subject, predicate, object) triples.\n\n"
    "Focus on people, tickets/issues, repositories, and services/systems "
    "explicitly named in the text. Use short lowercase snake_case "
    "predicates (e.g. \"depends_on\", \"blocks\", \"mentions\", "
    "\"related_to\"). Only extract relationships that are clearly stated "
    "or strongly implied — do not guess or invent entities that aren't "
    "actually in the text.\n\n"
    f"Return at most {{max_edges}} edges. If nothing notable is mentioned, "
    "return an empty list."
)


class _Edge(BaseModel):
    subject: str
    predicate: str
    object: str


class _ExtractedEdges(BaseModel):
    edges: list[_Edge] = Field(default_factory=list)


def extract_entities(text: str) -> list[tuple[str, str, str]]:
    """
    Return `(subject, predicate, object)` triples extracted from `text`.

    No-ops (`return []`) when `settings.ENABLE_ENTITY_EXTRACTION_LLM` is
    False, `text` is blank, or the extraction call fails for any reason.
    """
    if not settings.ENABLE_ENTITY_EXTRACTION_LLM or not text.strip():
        return []

    try:
        llm = get_entity_extraction_llm().with_structured_output(_ExtractedEdges)
        result = llm.invoke(
            [
                (
                    "system",
                    _SYSTEM_PROMPT.format(
                        max_edges=settings.ENTITY_EXTRACTION_MAX_EDGES
                    ),
                ),
                ("human", text),
            ]
        )
        return [
            (edge.subject.strip(), edge.predicate.strip(), edge.object.strip())
            for edge in result.edges
            if edge.subject.strip() and edge.predicate.strip() and edge.object.strip()
        ][: settings.ENTITY_EXTRACTION_MAX_EDGES]
    except Exception:
        logger.warning("Entity extraction failed — skipping LLM-extracted edges")
        return []
