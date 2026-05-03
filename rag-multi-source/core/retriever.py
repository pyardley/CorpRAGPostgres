"""
Retriever — turns ``(query, filter_dict)`` into a list of :class:`RetrievedChunk`.

Multi-tenancy is enforced at two layers:

1. **Application-layer filter (primary).** ``filter_dict`` (built via
   :func:`core.vector_store.build_query_filter`) becomes the SQL
   ``WHERE`` clause that constrains the search to the user's
   accessible resources. There is no ``user_id`` column on the vector
   table — the filter is the original security boundary.

2. **PostgreSQL Row-Level Security (defence-in-depth).** When
   ``settings.ENABLE_RLS`` is True, the retriever also binds the
   active user_id to the per-session GUC ``app.current_user_id`` via
   :func:`app.utils.set_current_user_for_rls` so the RLS policies on
   ``vector_chunks`` / ``user_accessible_resources`` are checked by
   the database itself. If the application-layer ``WHERE`` clause is
   ever bypassed by a regression, the policies still refuse to
   surface chunks the current user wasn't granted.

Retrieval modes
---------------
``settings.RETRIEVAL_MODE`` selects between two strategies:

* ``"vector"`` — pure cosine similarity over the HNSW index. Original
  behaviour. Best for conceptual / semantic queries.
* ``"hybrid"`` — runs the vector search and a Postgres FTS keyword
  search in parallel, then fuses the two ranked lists with Reciprocal
  Rank Fusion (RRF). The keyword side is essential for rare-token
  queries (error codes, ticket numbers, identifiers like ``XYZ999``)
  where the embedding is dominated by surrounding semantic context
  rather than the literal token.

Both modes share the per-source quota, the RLS bind, and the
``hnsw.ef_search`` tuning. Hybrid mode requires the
``vector_chunks.text_search`` generated column + GIN index that
``models.migrations`` provisions automatically on first run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger
from sqlalchemy import func, literal_column, select, text

from app.config import settings
from app.utils import get_db, set_current_user_for_rls
from core.llm import get_embeddings
from core.vector_store import build_query_filter, filter_to_where
from models.vector_chunk import VectorChunk


@dataclass
class RetrievedChunk:
    """One similarity-search hit returned to the chat layer."""

    resource_id: str
    source: str
    chunk_index: int
    score: float                # cosine similarity in [-1, 1]; we expect 0..1
    text: str
    title: str
    url: str
    metadata: dict[str, Any]

    @property
    def citation_label(self) -> str:
        return self.title or self.resource_id


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _select_columns(similarity_expr):
    """
    Column projection shared by the vector and keyword searches so per-source
    quota, RRF fusion, and row-to-RetrievedChunk conversion can treat both
    result sets interchangeably.
    """
    return (
        VectorChunk.resource_id,
        VectorChunk.source,
        VectorChunk.chunk_index,
        VectorChunk.text,
        VectorChunk.title,
        VectorChunk.url,
        VectorChunk.project_key,
        VectorChunk.space_key,
        VectorChunk.db_name,
        VectorChunk.git_scope,
        VectorChunk.email_provider,
        VectorChunk.object_name,
        VectorChunk.last_updated,
        VectorChunk.extra,
        similarity_expr,
    )


def _run_vector_search(db, embedding, where_clause, score_threshold, limit):
    """Pure cosine-similarity search ordered by HNSW distance."""
    similarity_expr = (
        1 - VectorChunk.embedding.cosine_distance(embedding)
    ).label("similarity")
    distance_expr = VectorChunk.embedding.cosine_distance(embedding)

    stmt = (
        select(*_select_columns(similarity_expr))
        .where(where_clause)
        .where(similarity_expr >= score_threshold)
        .order_by(distance_expr)
        .limit(limit)
    )
    return db.execute(stmt).all()


def _run_keyword_search(db, query_text, embedding, where_clause, limit):
    """
    Postgres FTS keyword search.

    Uses the ``text_search`` generated column (managed by
    ``models.migrations``) plus ``websearch_to_tsquery`` for permissive
    user-input parsing — quoted phrases, OR-of-terms, and a leading
    ``-`` for negation all work without raising on weird syntax.

    The ``score`` field on returned rows is still cosine similarity,
    computed alongside the FTS rank, so downstream code that displays
    ``RetrievedChunk.score`` doesn't have to special-case keyword-only
    hits. Note ``score_threshold`` is *not* applied here — the whole
    point of the hybrid path is to surface lexical matches the
    embedding missed (low cosine, high lexical relevance).

    ``literal_column("text_search")`` references the generated column
    by name without requiring it to be declared on the SQLAlchemy
    model (the model stays focused on application-managed columns;
    the generated column is the migration's responsibility).
    """
    similarity_expr = (
        1 - VectorChunk.embedding.cosine_distance(embedding)
    ).label("similarity")
    text_search_col = literal_column("text_search")

    # Postgres has two ``websearch_to_tsquery`` overloads — ``(text)`` and
    # ``(regconfig, text)`` — and *no* ``(text, text)``