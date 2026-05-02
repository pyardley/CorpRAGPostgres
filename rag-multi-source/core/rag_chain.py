"""
Glue between retrieved chunks and the LLM.

Keeps the prompt and citation post-processing in one place so `app/chat.py`
stays focused on UI concerns.

The chain produces:

* The natural-language answer + de-duplicated citations.
* An **audit record** describing the LLM call (``prompt_tokens``,
  ``completion_tokens``, ``cost_usd``, ``duration_seconds``,
  ``provider``, ``model``). The chat layer feeds it into
  ``app.utils.update_session_totals`` for the running status bar.
* A list of **step timings** on ``RAGAnswer.steps``, recorded with
  ``app.utils.StepTimer``. The chat layer hands the list to
  ``app.utils.log_query_audit`` so every measured phase ends up in
  ``query_step_timings``. Steps emitted by this chain:

    - ``"build_context"``     — group hits by resource + format the
                                numbered context block.
    - ``"llm_invocation"``    — single ``llm.invoke(messages)`` call.
    - ``"post_processing"``   — extract text + build the audit record.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from app.config import settings
from app.utils import StepTimer, estimate_cost, extract_usage
from core.llm import get_llm
from core.retriever import RetrievedChunk, deduplicate_by_resource


SYSTEM_PROMPT = """You are CorporateRAG, an enterprise assistant that answers
questions strictly from the user's connected Jira tickets, Confluence pages,
SQL Server objects, and Git repositories.

Rules:
1. Use ONLY the information in the provided context. If the answer is not
   present, say so plainly — do not invent facts.
2. Cite sources inline using bracketed numeric markers like [1], [2] that
   match the numbered list provided in the context. The number you cite MUST
   be the number of the source that actually contains the fact you are
   stating — never cite [N] unless block [N] in the context supports the
   claim. If multiple sources support a claim, you may cite several, e.g.
   "X is true [1][3]".
3. Be concise and direct. Prefer bullet points for lists, fenced code blocks
   for SQL, JSON, or code snippets.
4. If the question references something that is clearly out of scope for the
   selected sources, tell the user which sources to enable."""


@dataclass
class RAGAnswer:
    """
    Result of a single answer generation.

    ``usage`` carries the audit record for this call (token counts, cost,
    duration, provider, model). It's optional purely so legacy call sites
    that build a ``RAGAnswer`` directly (e.g. the error fallback in the
    chat layer) don't have to fabricate one.

    ``steps`` carries the per-step timings the chat layer will persist to
    ``query_step_timings``. Defaults to an empty list so legacy code that
    constructs a ``RAGAnswer`` without timings still works.
    """

    answer: str
    citations: list[RetrievedChunk]
    usage: dict[str, Any] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)


def _format_context(
    hits: list[RetrievedChunk], citations: list[RetrievedChunk]
) -> str:
    """
    Build the LLM context block, numbered identically to the user-visible
    citations list.

    The retrieved ``hits`` may contain several chunks per resource; the
    ``citations`` list is the de-duplicated view shown to the user (one entry
    per resource, ordered by score). To prevent a citation-number mismatch
    where the LLM cites ``[4]`` meaning the 4th raw chunk while the user sees
    a different ``[4]``, we:

      1. Group all hit chunks by ``resource_id``.
      2. Walk ``citations`` in order, emitting one numbered block per
         resource with all of its chunks concatenated underneath the same
         number.

    Net effect: the LLM's ``[N]`` and the rendered Sources list's ``[N]``
    always point to the same resource.
    """
    grouped: dict[str, list[RetrievedChunk]] = {}
    for hit in hits:
        grouped.setdefault(hit.resource_id, []).append(hit)

    parts: list[str] = []
    for i, citation in enumerate(citations, start=1):
        chunks = grouped.get(citation.resource_id) or [citation]
        header = f"[{i}] ({citation.source}) {citation.citation_label}"
        if citation.url:
            header += f" — {citation.url}"
        body = "\n\n".join(chunk.text.strip() for chunk in chunks)
        parts.append(f"{header}\n{body}")
    return "\n\n---\n\n".join(parts)


def _current_model_name() -> str:
    """Return the configured chat model for the active provider."""
    p = settings.LLM_PROVIDER
    if p == "openai":
        return settings.OPENAI_CHAT_MODEL
    if p == "anthropic":
        return settings.ANTHROPIC_MODEL
    if p == "grok":
        return settings.GROK_MODEL
    return ""


def _build_audit_record(
    response: Any,
    duration_seconds: float,
) -> dict[str, Any]:
    """
    Assemble the audit record for one LLM call.

    Same shape as the ``query_audit_logs`` row written by
    ``app.utils.log_query_audit``, so the in-memory status bar and the
    persisted log always show the same numbers for the same call.
    """
    provider = settings.LLM_PROVIDER
    model = _current_model_name()
    usage = extract_usage(response)
    cost = estimate_cost(
        provider, model, usage["prompt_tokens"], usage["completion_tokens"]
    )
    return {
        "provider": provider,
        "model": model,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "cost_usd": cost,
        "duration_seconds": duration_seconds,
    }


def answer_question(
    question: str,
    hits: list[RetrievedChunk],
    history: Iterable[tuple[str, str]] = (),
) -> RAGAnswer:
    """
    Build the prompt, call the LLM, return answer + de-duplicated citations
    + an audit record (``RAGAnswer.usage``) the chat layer feeds into the
    status bar + a list of step timings (``RAGAnswer.steps``) the chat
    layer persists to ``query_step_timings``.

    `history` is an iterable of (role, content) tuples, where role is
    "user" or "assistant". We prepend a small window of history so follow-up
    questions work without us having to re-implement message memory.
    """
    steps: list[dict[str, Any]] = []

    if not hits:
        # No retrieval results — short-circuit before we waste an LLM
        # call. We still emit a "build_context" step (with retrieved=0)
        # so the audit shows what happened.
        with StepTimer(steps, "build_context", retrieved_count=0, citations_count=0):
            pass
        return RAGAnswer(
            answer=(
                "I couldn't find anything in the sources you have selected that "
                "answers that. Try widening your selection in the sidebar, or "
                "running an ingestion pass for the relevant project / space / "
                "database."
            ),
            citations=[],
            # No LLM call was made — emit an empty audit record so the
            # status bar still increments the prompt counter cleanly.
            usage=_build_audit_record(response=None, duration_seconds=0.0),
            steps=steps,
        )

    # Use de-duplicated, sorted citations for the visible "Sources" list, AND
    # number the LLM context the same way (one block per resource, all chunks
    # underneath the same number) so the LLM's [N] always lines up with what
    # the user sees in the Sources list.
    with StepTimer(
        steps,
        "build_context",
        retrieved_count=len(hits),
    ) as t_ctx:
        citations = deduplicate_by_resource(hits)
        context = _format_context(hits, citations)
        t_ctx.extra["citations_count"] = len(citations)
        t_ctx.extra["context_chars"] = len(context)

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    # Include up to last 6 turns of history (3 exchanges) — enough for
    # short-form follow-ups without blowing the context window.
    for role, content in list(history)[-6:]:
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            from langchain_core.messages import AIMessage

            messages.append(AIMessage(content=content))

    user_block = (
        f"Context (numbered):\n\n{context}\n\n---\n\nQuestion: {question}\n\n"
        "Answer using only the context above and cite sources with [N]."
    )
    messages.append(HumanMessage(content=user_block))

    llm = get_llm()
    logger.debug("Invoking LLM with {} context chunks", len(hits))

    # Wall-clock the LLM call only — retrieval timing is owned by the
    # chat layer (it has the full lifecycle view including the spinner).
    with StepTimer(
        steps,
        "llm_invocation",
        provider=settings.LLM_PROVIDER,
        model=_current_model_name(),
    ) as t_llm:
        started = time.perf_counter()
        response = llm.invoke(messages)
        elapsed = time.perf_counter() - started
        # Mirror the duration onto the LLM step's metadata so a single
        # ``query_step_timings`` row already carries the per-call detail
        # without having to JOIN against the audit row.
        t_llm.extra["elapsed_ms"] = int(elapsed * 1000)

    with StepTimer(steps, "post_processing") as t_post:
        text = getattr(response, "content", str(response))
        audit = _build_audit_record(response, elapsed)
        t_post.extra["prompt_tokens"] = audit["prompt_tokens"]
        t_post.extra["completion_tokens"] = audit["completion_tokens"]
        t_post.extra["total_tokens"] = audit["total_tokens"]

    logger.info(
        "[rag.chain] LLM call: provider={} model={} tokens={} (in={}, out={}) "
        "cost=${:.5f} duration={:.2f}s",
        audit["provider"],
        audit["model"],
        audit["total_tokens"],
        audit["prompt_tokens"],
        audit["completion_tokens"],
        audit["cost_usd"],
        audit["duration_seconds"],
    )

    return RAGAnswer(answer=text, citations=citations, usage=audit, steps=steps)
