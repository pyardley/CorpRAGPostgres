"""
CorporateRAG MCP Server package.

Hosts a lightweight FastAPI service that exposes safe, read-only tools to
the main chat agent. Today it ships SQL Server table-data tools; the
shape (``mcp_server/tools/<source>_tools.py``) is deliberately
source-pluggable so Git, Confluence and Jira tools can be added later
without touching the core server.

The implementation is HTTP/JSON over loopback rather than the canonical
MCP stdio transport. This is deliberate:

* It runs as a normal Python web service that the Streamlit app can
  start, supervise and call from the same process tree.
* It maps cleanly to the official MCP tool schema (``name``,
  ``description``, ``input_schema``) so swapping in
  ``langchain-mcp-adapters`` / Anthropic's MCP SDK later is a transport
  change, not a redesign.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
