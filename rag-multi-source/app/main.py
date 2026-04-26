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
    /* Tighten sidebar padding */
    section[data-testid="stSidebar"] { padding-top: 1rem; }

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

selection = render_sidebar()
render_chat(selection)
