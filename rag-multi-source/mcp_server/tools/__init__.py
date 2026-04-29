"""
MCP tool implementations.

Each module in this package corresponds to one source system (sql, git,
confluence, jira, ...) and exposes:

* a ``TOOL_SPECS`` list (MCP-style tool descriptors), and
* the actual handler callables wired up by ``mcp_server.server``.

Today only :mod:`mcp_server.tools.sql_tools` is implemented. Adding a new
source is a matter of dropping in a new module that follows the same
shape and registering it in ``mcp_server.server``.
"""
