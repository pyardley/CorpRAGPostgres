"""
Lifecycle manager for the MCP server.

The Streamlit app calls :func:`ensure_mcp_running` once on startup. That
function:

1. Probes ``http://MCP_HOST:MCP_PORT/healthz``.
2. If reachable, it assumes someone (operator, ``uvicorn``, the previous
   Streamlit run) already started the server and just primes the
   in-process client with the cached shared token.
3. Otherwise it spawns ``python -m mcp_server.server`` as a child
   process, writes a runtime token to a state file under ``.streamlit/``,
   and waits up to ~10 seconds for the health check to pass.

Why a subprocess and not ``threading.Thread(uvicorn.run, ...)``? Streamlit
re-executes the script on every interaction; running uvicorn in-process
would either (a) fight Streamlit's event loop, or (b) leak a server per
session. A child process is cleanly owned by the *first* Streamlit run
and reused by subsequent reruns / sessions.
"""

from __future__ import annotations

import atexit
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from core.mcp_client import get_mcp_client
from mcp_server.config import generate_token, mcp_settings


# ──────────────────────────────────────────────────────────────────────────────
# State files
# ──────────────────────────────────────────────────────────────────────────────

_STATE_DIR = Path(".streamlit") / "mcp"
_TOKEN_FILE = _STATE_DIR / "token"
_PID_FILE = _STATE_DIR / "pid"


def _ensure_state_dir() -> None:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)


def _read_token() -> Optional[str]:
    try:
        return _TOKEN_FILE.read_text(encoding="utf-8").strip() or None
    except FileNotFoundError:
        return None


def _write_token(token: str) -> None:
    _ensure_state_dir()
    _TOKEN_FILE.write_text(token, encoding="utf-8")


def _read_pid() -> Optional[int]:
    try:
        return int(_PID_FILE.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def _write_pid(pid: int) -> None:
    _ensure_state_dir()
    _PID_FILE.write_text(str(pid), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Process control
# ──────────────────────────────────────────────────────────────────────────────

_child: Optional[subprocess.Popen[bytes]] = None


def _spawn_server(token: str) -> subprocess.Popen[bytes]:
    """Spawn ``python -m mcp_server.server`` as a detached child."""
    env = os.environ.copy()
    env["MCP_SHARED_TOKEN"] = token
    env.setdefault("MCP_HOST", mcp_settings.MCP_HOST)
    env.setdefault("MCP_PORT", str(mcp_settings.MCP_PORT))

    creationflags = 0
    if sys.platform == "win32":
        # Detach the child so closing the Streamlit terminal doesn't
        # propagate Ctrl+C straight into the MCP server (we still
        # terminate it cleanly via atexit).
        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )

    logger.info(
        "Spawning MCP server: python -m mcp_server.server (port {})",
        mcp_settings.MCP_PORT,
    )
    proc = subprocess.Popen(
        [sys.executable, "-m", "mcp_server.server"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    _write_pid(proc.pid)
    return proc


def _terminate_child() -> None:
    global _child
    if _child is None:
        return
    try:
        if _child.poll() is None:
            logger.info("Terminating MCP server pid={}", _child.pid)
            _child.terminate()
            try:
                _child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _child.kill()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Error terminating MCP server: {}", exc)
    finally:
        _child = None
        try:
            _PID_FILE.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def _kill_pid(pid: int) -> bool:
    """Best-effort kill of an orphaned MCP server we don't own anymore."""
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/F", "/T"],
                check=False,
                capture_output=True,
                timeout=5,
            )
        else:
            os.kill(pid, 15)
        logger.info("Terminated orphan MCP server pid={}", pid)
        # Give the OS a moment to release the port.
        time.sleep(0.6)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not terminate orphan MCP pid={}: {}", pid, exc)
        return False


def _wait_for_token(deadline_s: float) -> Optional[str]:
    """Poll the token file until ``deadline_s`` (epoch). Returns token or None."""
    while time.time() < deadline_s:
        token = _read_token()
        if token:
            return token
        time.sleep(0.2)
    return None


def ensure_mcp_running(timeout_s: float = 12.0) -> bool:
    """
    Make sure the MCP server is reachable and the in-process client is
    authenticated. Returns True on success, False otherwise.

    Behaviour:
      1. **Fast path** — server is reachable AND we can resolve a token
         (env var, file written by the server on startup, or file written
         by a prior manager run): wire up the client and return True.
      2. **Recovery** — server is reachable but no token is discoverable
         (typical after upgrading to a server build that publishes its
         token: the running orphan predates the publish-on-startup change).
         Kill the orphan if we have its recorded PID, then fall through
         to step 3.
      3. **Spawn** — start ``python -m mcp_server.server`` with a fresh
         token in the env, register atexit cleanup, wait for /healthz.

    Safe to call repeatedly.
    """
    global _child

    client = get_mcp_client()

    # ── Step 1: fast path ───────────────────────────────────────────────────
    if client.healthz():
        # Server may have just booted; give it up to 2s to publish its token.
        token = (
            mcp_settings.MCP_SHARED_TOKEN
            or _read_token()
            or _wait_for_token(time.time() + 2.0)
        )
        if token:
            client.set_token(token)
            logger.info(
                "MCP server is reachable on {} and the client is "
                "authenticated.",
                client.base_url,
            )
            return True

        # ── Step 2: orphan recovery ────────────────────────────────────────
        orphan_pid = _read_pid()
        logger.warning(
            "MCP server is reachable on {} but no shared token is "
            "discoverable. Treating as orphan from a previous run "
            "(recorded pid={}). Will terminate and respawn.",
            client.base_url,
            orphan_pid,
        )
        if orphan_pid:
            _kill_pid(orphan_pid)
        # Fall through to spawn.

    # ── Step 3: spawn ───────────────────────────────────────────────────────
    token = mcp_settings.MCP_SHARED_TOKEN or generate_token()
    _write_token(token)

    _child = _spawn_server(token)
    atexit.register(_terminate_child)

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if client.healthz():
            client.set_token(token)
            logger.info("MCP server is up on {}", client.base_url)
            return True
        if _child.poll() is not None:
            logger.error(
                "MCP server child exited prematurely with code {}",
                _child.returncode,
            )
            return False
        time.sleep(0.4)

    logger.error("MCP server did not become healthy within {}s", timeout_s)
    return False


def mcp_status() -> dict[str, object]:
    """Diagnostics for the sidebar / debug panel."""
    client = get_mcp_client()
    return {
        "base_url": client.base_url,
        "healthy": client.healthz(),
        "token_set": bool(client.token),
        "child_pid": _child.pid if _child else _read_pid(),
    }
