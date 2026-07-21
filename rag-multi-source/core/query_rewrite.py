"""
Query rewriting before retrieval (README "Possible enhancements" —
Query rewriting before retrieval, multi-query decomposition).

The chat layer normally sends the user's raw question straight to
`core.retriever.retrieve`. :func:`rewrite_query` adds an optional pass
ahead of that: one LLM call that either returns the question unchanged
(simple, single-intent questions) or decomposes it into up to
`settings.QUERY_REWRITE_MAX_SUBQUERIES` standalone search queries, each
retrieved independently and pooled by the chat layer.

Fails OPEN — unlike `core.live_acl`, this is a retrieval-quality knob,
not an authorization boundary. Any error (bad structured-output parse,
model error, timeout) is logged and `[question]` is returned, degrading
to today's single-retrieve() behaviour.
"""

from __future__ import annotations

from loguru import logger
from pydantic import BaseModel, Field

from app.config import settings
from core.llm import get_query_rewrite_llm

_SYSTEM_PROMPT = (
    "You prepare a user's question for a hybrid (vector + keyword) search "
    "engine over a corporate knowledge base (Jira, Confluence, SQL Server "
    "schemas, Git, email).\n\n"
    "If the question is already simple and single-intent, return it "
    "completely unchanged as the only item in `sub_queries`.\n\n"
    "If the question has multiple distinct parts or asks for a comparison "
    "(e.g. \"compare X and Y\", \"how does A relate to B\"), split it into "
    f"up to {{max_subqueries}} standalone search queries — each one must be "
    "understandable and searchable entirely on its own, since each is run "
    "as an independent search with no access to the others or to the "
    "original question.\n\n"
    "Preserve literal tokens EXACTLY as written — error codes, ticket keys "
    "(e.g. PROJ-123), identifiers, function/table names. Do not paraphrase "
    "or rephrase them; the keyword-search side of retrieval depends on "
    "exact matches."
)


class _SubQueries(BaseModel):
    sub_queries: list[str] = Field(
        description=(
            "1 to N standalone search queries. A single-item list "
            "(the original question, unchanged) when no decomposition is "
            "needed."
        )
    )


def rewrite_query(question: str) -> list[str]:
    """
    Return the list of search queries to retrieve for.

    No-ops (`return [question]`) when `settings.QUERY_REWRITE_ENABLED` is
    False, `question` is blank, or the rewrite call fails for any reason.
    """
    if not settings.QUERY_REWRITE_ENABLED or not question.strip():
        return [question]

    try:
        llm = get_query_rewrite_llm().with_structured_output(_SubQueries)
        result = llm.invoke(
            [
                (
                    "system",
                    _SYSTEM_PROMPT.format(
                        max_subqueries=settings.QUERY_REWRITE_MAX_SUBQUERIES
                    ),
                ),
                ("human", question),
            ]
        )
        sub_queries = [q.strip() for q in result.sub_queries if q.strip()]
        sub_queries = sub_queries[: settings.QUERY_REWRITE_MAX_SUBQUERIES]
        return sub_queries or [question]
    except Exception:
        logger.warning("Query rewrite failed — falling back to the original question")
        return [question]
