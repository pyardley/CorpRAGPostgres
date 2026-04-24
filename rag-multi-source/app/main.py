"""
CorporateRAG – Streamlit entry point.

Run with:
    streamlit run app/main.py
"""

from __future__ import annotations

import sys
import os

# Allow imports from project root regardless of CWD
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
from loguru import logger

# ── Bootstrap ─────────────────────────────────────────────────────────────────

from app.utils import init_db

init_db()

# ── Page config (must be first Streamlit call) ────────────────────────────────

st.set_page_config(
    page_title="CorporateRAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS – clean, modern look ──────────────────────────────────────────

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

# ── Auth gate ─────────────────────────────────────────────────────────────────

from app.auth import current_user, login_page

if current_user() is None:
    login_page()
    st.stop()

# ── Authenticated app ─────────────────────────────────────────────────────────

from app.sidebar import render_sidebar
from app.chat import render_chat

selection = render_sidebar()
render_chat(selection)
