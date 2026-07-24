"""
Live + static Git/GitHub import-dependency MCP tools (README "Possible
enhancements" §9 — applying the SQL impact-analysis lessons to GitHub).

Three tools, mirroring the SQL trio in `entity_graph_tools.py` /
`sql_schema_tools.py`:

* `git_dependency_graph` — static BFS over `entity_edges` rows written at
  ingestion time by `core.git_dependency_extraction.find_imports`
  (`source="git", predicate="imports"`). Parity with `sql_dependency_graph`.
* `github_file_content` — fetch a file's CURRENT content straight from
  GitHub, not the possibly-stale ingested chunk copy. Parity with
  `sql_object_definition`.
* `github_file_dependencies` — live dependency check. Parity with
  `sql_object_dependencies`, but the two directions use genuinely
  different mechanisms rather than "DMV primary / text-fallback for
  gaps", because GitHub has no DMV equivalent at all:

  - upstream ("what does this file import" / lineage): live-primary —
    fetch the file fresh, scan its imports, and recurse hop-by-hop by
    fetching each newly-discovered import live too. Always reflects the
    file's true current state (tagged `"evidence": "live-scan"`).
  - downstream ("what imports this file" / blast radius): no live
    mechanism can exist without scanning the whole repo's current
    content — the exact reasoning `sql_schema_tools.object_dependencies`
    already gives for why SQL doesn't text-fallback downstream either
    ("would mean scanning every OTHER object's text, which the static
    graph tool already does at ingestion time") applies even more
    directly here, since GitHub has no DMV to fall back on for either
    direction. Falls back entirely to the static graph
    (tagged `"evidence": "static-graph"`).

Tenancy mirrors `sql_schema_tools.py`: every call carries a `user_id`,
resolved against `user_accessible_resources` (+ live re-validation via
`core.live_acl`) before touching GitHub or the target repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from github import GithubException, RateLimitExceededException
from loguru import logger
from sqlalchemy import and_

from app.utils import get_db, list_accessible
from core.live_acl import revalidate
from mcp_server.config import mcp_settings
from mcp_server.tools._edge_bfs import bfs
from mcp_server.tools._git_engine import get_repo
from models.entity_edge import EntityEdge

# Reuses the ingestor's own live-fetch size cap rather than introducing a
# second, possibly-inconsistent global setting -- see app/config.py's
# ENABLE_GIT_DEPENDENCY_MCP_TOOLS comment.
from app.ingestion.git_ingestor import _MAX_FILE_BYTES as _MAX_LIVE_FETCH_BYTES


# ──────────────────────────────────────────────────────────────────────────────
# MCP tool descriptors
# ──────────────────────────────────────────────────────────────────────────────

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "git_dependency_graph",
        "description": (
            "Traverse the static Git import-dependency graph built at "
            "ingestion time from Python/JS/TS import statements. Use "
            "direction='downstream' for 'what breaks if I change/rename "
            "this file' (blast radius — what imports it), and "
            "direction='upstream' for 'what does this file depend on' "
            "(what it imports). Returns hop-labeled (subject, predicate, "
            "object) edges with predicate='imports'. This is a STATIC, "
            "parse-time graph — it can't see dynamic imports "
            "(importlib.import_module, dynamic require()), so treat an "
            "empty result as inconclusive, not proof of no dependency."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "git_scope": {
                    "type": "string",
                    "description": "Repo scope, e.g. 'acme/widgets@main'.",
                },
                "file_path": {
                    "type": "string",
                    "description": "Repo-relative file path, e.g. 'src/utils.py'.",
                },
                "direction": {
                    "type": "string",
                    "enum": ["upstream", "downstream", "both"],
                    "description": (
                        "'downstream' = what imports this file (impact "
                        "analysis). 'upstream' = what this file imports "
                        "(lineage). 'both' = both directions."
                    ),
                    "default": "both",
                },
                "max_hops": {
                    "type": "integer",
                    "description": "Max traversal hops (default 3, max 5).",
                    "default": 3,
                },
            },
            "required": ["git_scope", "file_path"],
        },
    },
    {
        "name": "github_file_content",
        "description": (
            "Fetch the CURRENT, live content of a file straight from "
            "GitHub — not the (possibly stale) ingested/chunked copy. "
            "Use this whenever you need to open up a called function's "
            "or imported module's actual code rather than naming it, or "
            "when a RAG-retrieved file excerpt looks truncated."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "git_scope": {
                    "type": "string",
                    "description": "Repo scope, e.g. 'acme/widgets@main'.",
                },
                "file_path": {
                    "type": "string",
                    "description": "Repo-relative file path, e.g. 'src/utils.py'.",
                },
            },
            "required": ["git_scope", "file_path"],
        },
    },
    {
        "name": "github_file_dependencies",
        "description": (
            "Check a file's dependencies against LIVE GitHub content. "
            "direction='upstream' freshly fetches the file (and, "
            "hop-by-hop, each file it imports) straight from GitHub and "
            "re-scans it — always reflects the current repo state, more "
            "authoritative than the static git_dependency_graph tool for "
            "this direction. direction='downstream' (what imports this "
            "file — blast radius) has no live equivalent (would require "
            "scanning every file in the repo), so it defers entirely to "
            "the static graph — results are tagged "
            "'evidence':'live-scan' vs 'evidence':'static-graph' so you "
            "can tell which is which."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "git_scope": {
                    "type": "string",
                    "description": "Repo scope, e.g. 'acme/widgets@main'.",
                },
                "file_path": {
                    "type": "string",
                    "description": "Repo-relative file path, e.g. 'src/utils.py'.",
                },
                "direction": {
                    "type": "string",
                    "enum": ["upstream", "downstream", "both"],
                    "description": (
                        "'downstream' = what imports this file (impact "
                        "analysis / blast radius). 'upstream' = what "
                        "this file imports (lineage). 'both' = both "
                        "directions."
                    ),
                    "default": "both",
                },
                "max_hops": {
                    "type": "integer",
                    "description": (
                        "Max traversal hops (default 3). Each upstream "
                        "hop is a live GitHub fetch, so this bounds "
                        "worst-case API calls too."
                    ),
                    "default": 3,
                },
            },
            "required": ["git_scope", "file_path"],
        },
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Result shape (mirrors mcp_server.tools.sql_tools.ToolResult)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    ok: bool
    tool: str
    data: Any = None
    markdown: str = ""
    metadata: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "tool": self.tool,
            "data": self.data,
            "markdown": self.markdown,
            "metadata": self.metadata or {},
            "error": self.error,
        }


def _check_access(user_id: str, git_scope: str) -> Optional[str]:
    """Return an error message if the user lacks access to git_scope, else None."""
    with get_db() as db:
        accessible = set(revalidate(db, user_id, "git", list_accessible(db, user_id, "git")))
    if git_scope not in accessible:
        return (
            f"Access denied: user has no granted access to repo scope "
            f"'{git_scope}'. Run a Git ingestion under your account first."
        )
    return None


def _github_error_message(exc: GithubException) -> str:
    if isinstance(exc, RateLimitExceededException):
        return f"GitHub API rate limit exceeded: {exc}. Try again later."
    return f"GitHub API error ({exc.status}): {exc.data.get('message', exc) if isinstance(exc.data, dict) else exc}"


# ──────────────────────────────────────────────────────────────────────────────
# Tool: git_dependency_graph (static)
# ──────────────────────────────────────────────────────────────────────────────

def dependency_graph(
    user_id: str,
    git_scope: str,
    file_path: str,
    direction: str = "both",
    max_hops: Optional[int] = None,
) -> ToolResult:
    """Multi-hop BFS over the static Git import graph (`entity_edges`
    rows with `source="git", predicate="imports"`), tenancy-scoped to the
    user's accessible repo scopes. Reuses the same `bfs` helper
    `entity_graph_tools.traverse_sql_dependencies` uses -- the loop
    itself has no source-specific logic."""
    if direction not in ("upstream", "downstream", "both"):
        direction = "both"
    hops = max(1, min(int(max_hops or 3), mcp_settings.MCP_GIT_MAX_DEPENDENCY_HOPS))
    start = f"{git_scope}:{file_path}"

    with get_db() as db:
        git_scopes = revalidate(db, user_id, "git", list_accessible(db, user_id, "git"))
        if not git_scopes:
            logger.info("[mcp.git_dependency_graph] user={} has no accessible git scopes", user_id)
            return ToolResult(
                ok=True,
                tool="git_dependency_graph",
                data={"edges": {}},
                markdown="_(no accessible Git repos — run a Git ingestion first)_",
                metadata={"count": 0},
            )

        base_clause = and_(
            EntityEdge.source == "git",
            EntityEdge.resource_identifier.in_(git_scopes),
            EntityEdge.predicate == "imports",
        )

        edge_sets: dict[str, list[dict[str, Any]]] = {}
        if direction in ("upstream", "both"):
            edge_sets["upstream"] = bfs(db, base_clause, start, hops, forward=True)
        if direction in ("downstream", "both"):
            edge_sets["downstream"] = bfs(db, base_clause, start, hops, forward=False)

    total = sum(len(rows) for rows in edge_sets.values())
    markdown = _render_edge_sets(file_path, edge_sets, evidence_col=False)
    logger.info(
        "[mcp.git_dependency_graph] user={} file={!r} direction={} -> {} edges",
        user_id, start, direction, total,
    )
    return ToolResult(
        ok=True,
        tool="git_dependency_graph",
        data={"file_path": file_path, "edges": edge_sets},
        markdown=markdown,
        metadata={"git_scope": git_scope, "file_path": file_path, "count": total},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tool: github_file_content (live)
# ──────────────────────────────────────────────────────────────────────────────

def file_content(user_id: str, git_scope: str, file_path: str) -> ToolResult:
    """Fetch the complete, current live content of one file."""
    access_error = _check_access(user_id, git_scope)
    if access_error:
        logger.warning("[mcp.git_file] DENY user={} scope={} reason=no-access", user_id, git_scope)
        return ToolResult(ok=False, tool="github_file_content", error=access_error)

    try:
        repo, branch = get_repo(user_id, git_scope)
        file_obj = repo.get_contents(file_path, ref=branch)
        if isinstance(file_obj, list):
            return ToolResult(
                ok=False,
                tool="github_file_content",
                error=f"'{file_path}' is a directory, not a file.",
            )
        if file_obj.size and file_obj.size > _MAX_LIVE_FETCH_BYTES:
            return ToolResult(
                ok=False,
                tool="github_file_content",
                error=(
                    f"'{file_path}' is {file_obj.size} bytes, over the "
                    f"{_MAX_LIVE_FETCH_BYTES}-byte live-fetch cap."
                ),
            )
        content = file_obj.decoded_content.decode("utf-8", errors="replace")
    except PermissionError:
        raise
    except GithubException as exc:
        return ToolResult(ok=False, tool="github_file_content", error=_github_error_message(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "[mcp.git_file] file_content FAIL user={} scope={} path={}",
            user_id, git_scope, file_path,
        )
        return ToolResult(
            ok=False, tool="github_file_content", error=f"Failed to fetch '{file_path}': {exc}"
        )

    logger.info(
        "[mcp.git_file] file_content user={} scope={} path={} chars={}",
        user_id, git_scope, file_path, len(content),
    )
    return ToolResult(
        ok=True,
        tool="github_file_content",
        data={"file_path": file_path, "content": content},
        markdown=f"**{file_path}** (live from GitHub, `{git_scope}`):\n\n```\n{content}\n```",
        metadata={"git_scope": git_scope, "file_path": file_path, "bytes": len(content)},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tool: github_file_dependencies (live upstream / static downstream)
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_content_or_none(repo, branch: str, path: str) -> Optional[str]:
    """Best-effort live fetch for one hop of the upstream scan. Returns
    None (skip this hop, not an error) for anything short of a rate
    limit -- a stale static edge pointing at a since-renamed/deleted file
    shouldn't abort the whole multi-hop trace."""
    try:
        file_obj = repo.get_contents(path, ref=branch)
        if isinstance(file_obj, list) or (file_obj.size and file_obj.size > _MAX_LIVE_FETCH_BYTES):
            return None
        return file_obj.decoded_content.decode("utf-8", errors="replace")
    except RateLimitExceededException:
        raise
    except GithubException:
        return None


def _live_upstream_scan(repo, branch: str, file_path: str, hops: int) -> list[dict[str, Any]]:
    from core.git_dependency_extraction import SUPPORTED_EXTENSIONS, find_imports

    def _ext(path: str) -> str:
        name = path.rsplit("/", 1)[-1]
        return "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""

    tree = repo.get_git_tree(branch, recursive=True)
    paths = [
        item.path for item in tree.tree
        if item.type == "blob" and _ext(item.path) in SUPPORTED_EXTENSIONS
    ]
    known_files = frozenset(paths)

    visited = {file_path}
    frontier = [file_path]
    results: list[dict[str, Any]] = []

    for hop in range(1, hops + 1):
        if not frontier:
            break
        next_frontier: list[str] = []
        for path in frontier:
            content = _fetch_content_or_none(repo, branch, path)
            if content is None:
                continue
            for subj, predicate, obj in find_imports(content, path, known_files):
                results.append(
                    {"hop": hop, "subject": subj, "predicate": predicate, "object": obj, "evidence": "live-scan"}
                )
                if obj not in visited:
                    visited.add(obj)
                    next_frontier.append(obj)
        frontier = next_frontier

    return results


def _static_downstream(user_id: str, git_scope: str, file_path: str, hops: int) -> list[dict[str, Any]]:
    """Blast-radius direction has no live mechanism (would mean scanning
    every OTHER file's current content) -- falls back entirely to the
    static graph, same reasoning `sql_schema_tools.object_dependencies`
    gives for skipping a downstream text-fallback for SQL."""
    start = f"{git_scope}:{file_path}"
    with get_db() as db:
        git_scopes = revalidate(db, user_id, "git", list_accessible(db, user_id, "git"))
        if not git_scopes:
            return []
        base_clause = and_(
            EntityEdge.source == "git",
            EntityEdge.resource_identifier.in_(git_scopes),
            EntityEdge.predicate == "imports",
        )
        rows = bfs(db, base_clause, start, hops, forward=False)
    for row in rows:
        row["evidence"] = "static-graph"
    return rows


def file_dependencies(
    user_id: str,
    git_scope: str,
    file_path: str,
    direction: str = "both",
    max_hops: Optional[int] = None,
) -> ToolResult:
    """Live upstream re-scan + static-graph downstream lookup. See module
    docstring for why the two directions use different mechanisms."""
    access_error = _check_access(user_id, git_scope)
    if access_error:
        logger.warning("[mcp.git_file] DENY user={} scope={} reason=no-access", user_id, git_scope)
        return ToolResult(ok=False, tool="github_file_dependencies", error=access_error)

    if direction not in ("upstream", "downstream", "both"):
        direction = "both"
    hops = max(1, min(int(max_hops or 3), mcp_settings.MCP_GIT_MAX_DEPENDENCY_HOPS))

    edge_sets: dict[str, list[dict[str, Any]]] = {}

    if direction in ("upstream", "both"):
        try:
            repo, branch = get_repo(user_id, git_scope)
            edge_sets["upstream"] = _live_upstream_scan(repo, branch, file_path, hops)
        except PermissionError:
            raise
        except GithubException as exc:
            return ToolResult(ok=False, tool="github_file_dependencies", error=_github_error_message(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "[mcp.git_file] file_dependencies (upstream) FAIL user={} scope={} path={}",
                user_id, git_scope, file_path,
            )
            return ToolResult(
                ok=False,
                tool="github_file_dependencies",
                error=f"Failed live dependency scan for '{file_path}': {exc}",
            )

    downstream_inconclusive = False
    if direction in ("downstream", "both"):
        from app.config import settings

        edge_sets["downstream"] = _static_downstream(user_id, git_scope, file_path, hops)
        downstream_inconclusive = not edge_sets["downstream"] and not settings.ENABLE_GIT_DEPENDENCY_GRAPH

    total = sum(len(rows) for rows in edge_sets.values())
    markdown = _render_edge_sets(file_path, edge_sets, evidence_col=True)
    if downstream_inconclusive:
        markdown += (
            "\n\n_(downstream is empty because ENABLE_GIT_DEPENDENCY_GRAPH is "
            "off — this means \"couldn't be determined\", not \"confirmed "
            "nothing imports this\".)_"
        )

    logger.info(
        "[mcp.git_file] file_dependencies user={} scope={} path={} direction={} -> {} edges",
        user_id, git_scope, file_path, direction, total,
    )
    return ToolResult(
        ok=True,
        tool="github_file_dependencies",
        data={"file_path": file_path, "edges": edge_sets},
        markdown=markdown,
        metadata={"git_scope": git_scope, "file_path": file_path, "count": total},
    )


def _render_edge_sets(
    file_path: str, edge_sets: dict[str, list[dict[str, Any]]], evidence_col: bool
) -> str:
    parts: list[str] = []
    for label, rows in edge_sets.items():
        if not rows:
            parts.append(f"**{label}** — _(no edges found)_")
            continue
        if evidence_col:
            header = (
                f"**{label}** (from {file_path}):\n\n"
                "| Hop | Subject | Predicate | Object | Evidence |\n| --- | --- | --- | --- | --- |\n"
            )
            body = "\n".join(
                f"| {r['hop']} | {r['subject']} | {r['predicate']} | {r['object']} | {r.get('evidence', '')} |"
                for r in rows
            )
        else:
            header = (
                f"**{label}** (from {file_path}):\n\n"
                "| Hop | Subject | Predicate | Object |\n| --- | --- | --- | --- |\n"
            )
            body = "\n".join(
                f"| {r['hop']} | {r['subject']} | {r['predicate']} | {r['object']} |"
                for r in rows
            )
        parts.append(header + body)
    return "\n\n".join(parts) if parts else f"_(no edges found for {file_path!r})_"
