"""
Glue between retrieved chunks and the LLM.

Keeps the prompt and citation post-processing in one place so `app/chat.py`
stays focused on UI concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from core.llm import get_llm
from core.retriever import RetrievedChunk, deduplicate_by_resource


SYSTEM_PROMPT = """You are CorporateRAG, an enterprise assistant that answers
questions strictly from the user's connected Jira tickets, Confluence pages
and SQL Server objects.

Rules:
1. Use ONLY the information in the provided context. If the answer is not
   present, say so plainly — do not invent facts.
2. Cite sources inline using bracketed numeric markers like [1], [2] that
   match the numbered list provided in the context.
3. Be concise and direct. Prefer bullet points for lists, fenced code blocks
   for SQL, JSON, or code snippets.
4. If the question references something that is clearly out of scope for the
   selected sources, tell the user which sources to enable."""


@dataclass
class RAGAnswer:
    answer: str
    citations: list[RetrievedChunk]


def _format_context(hits: list[RetrievedChunk]) -> str:
    parts: list[str] = []
    for i, hit in enumerate(hits, start=1):
        header = f"[{i}] ({hit.source}) {hit.citation_label}"
        if hit.url:
            header += f" — {hit.url}"
        parts.append(f"{header}\n{hit.text.strip()}")
    return "\n\n---\n\n".join(parts)


def answer_question(
    question: str,
    hits: list[RetrievedChunk],
    history: Iterable[tuple[str, str]] = (),
) -> RAGAnswer:
    """
    Build the prompt, call the LLM, return answer + de-duplicated citations.

    `history` is an iterable of (role, content) tuples, where role is
    "user" or "assistant". We prepend a small window of history so follow-up
    questions work without us having to re-implement message memory.
    """
    if not hits:
        return RAGAnswer(
            answer=(
                "I couldn't find anything in the sources you have selected that "
                "answers that. Try widening your selection in the sidebar, or "
                "running an ingestion pass for the relevant project / space / "
                "database."
            ),
            citations=[],
        )

    # Use de-duplicated, sorted citations for the visible "Sources" list, but
    # keep all chunks in the LLM context so it has more to work with.
    citations = deduplicate_by_resource(hits)
    context = _format_context(hits)

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    # Include up to last 6 turns of history (3 exchanges) — enough for
    # short-form follow-ups without blowing the context window.
    for role, content in list(history)[-6:]:
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            # langchain_core.messages has AIMessage but importing it here keeps
            # the dependency surface a touch smaller; HumanMessage with a tag
            # is treated similarly by chat models.
            from langchain_core.messages import AIMessage

            messages.append(AIMessage(content=content))

    user_block = (
        f"Context (numbered):\n\n{context}\n\n---\n\nQuestion: {question}\n\n"
        "Answer using only the context above and cite sources with [N]."
    )
    messages.append(HumanMessage(content=user_block))

    llm = get_llm()
    logger.debug("Invoking LLM with {} context chunks", len(hits))
    response = llm.invoke(messages)
    text = getattr(response, "content", str(response))

    return RAGAnswer(answer=text, citations=citations)
