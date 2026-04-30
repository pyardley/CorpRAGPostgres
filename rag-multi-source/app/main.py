"""
CorporateRAG - Streamlit entry point.

Run with:
    streamlit run app/main.py
"""

from __future__ import annotations

import sys
import os

# Suppress noisy startup tracebacks
# Streamlit's local-sources file watcher walks sys.modules and accesses
# `__path__` on every loaded package. The `transformers` package (pulled in
# transitively by langchain-huggingface / sentence-transformers) uses a lazy
# loader that tries to import `torchvision` whenever its submodules are
# touched - and torchvision isn't a dependency, so each touch logs a full
# traceback.
#
# Two muzzles, applied as early as possible:
#   1) `TRANSFORMERS_VERBOSITY=error` silences transformers' own warnings.
#   2) Lower the Streamlit watcher's logger to CRITICAL so its
#      "Examining the path of ..." tracebacks don't reach the console.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import logging  # noqa: E402

logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(
    logging.CRITICAL
)

# Allow imports from project root regardless of CWD
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st  # noqa: E402
from loguru import logger  # noqa: E402

# Bootstrap
from app.utils import init_db

init_db()

# Page config (must be first Streamlit call)
st.set_page_config(
    page_title="CorporateRAG",
    page_icon="\U0001F50D",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS - clean, modern look
st.markdown(
    """
    <style>
    /* Tighten sidebar padding + widen so long credential captions
       (e.g. "Full token.json") don't get clipped. Streamlit's default
       sidebar is ~21rem; bump it to 26rem and let it stretch a bit
       more on wide displays. */
    section[data-testid="stSidebar"] {
        padding-top: 1rem;
        width: 26rem !important;
        min-width: 26rem !important;
        max-width: 32rem !important;
    }
    /* The collapsed-state container also has a fixed width — keep them
       aligned so the resize handle works as expected. */
    section[data-testid="stSidebar"] > div:first-child {
        width: 26rem !important;
        min-width: 26rem !important;
    }

    /* Citation block subtle background */
    .citation-block {
        background: #f0f4ff;
        border-left: 3px solid #4f8ef7;
        padding: 0.5rem 0.75rem;
        border-radius: 4px;
        margin-top: 0.5rem;
    }

    /* Chat bubble width */
    .stChatMessage { max-width: 860px; }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Auth gate
from app.auth import current_user, login_page  # noqa: E402

if current_user() is None:
    login_page()
    st.stop()

# Authenticated app
from app.sidebar import render_sidebar  # noqa: E402
from app.chat import render_chat  # noqa: E402
from app.mcp_manager import ensure_mcp_running  # noqa: E402

# Boot the MCP server on first authenticated load. We only do this once per
# Streamlit process (cached in session_state) so reruns don't keep re-probing.
if not st.session_state.get("_mcp_bootstrapped"):
    try:
        ok = ensure_mcp_running()
        st.session_state["_mcp_bootstrapped"] = True
        if not ok:
            logger.warning(
                "MCP server failed to start — Live SQL Table Data toggle "
                "will be unavailable."
            )
    except Exception:
        logger.exception("MCP bootstrap raised; continuing without it.")
        st.session_state["_mcp_bootstrapped"] = True

selection = render_sidebar()
render_chat(selection)
