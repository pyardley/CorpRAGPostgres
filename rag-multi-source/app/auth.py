"""Authentication: register, login, session management via Streamlit session_state."""

from __future__ import annotations

from typing import Optional

import bcrypt
import streamlit as st
from loguru import logger
from sqlalchemy.orm import Session

from app.utils import get_db
from models.user import User


# ── Password helpers ──────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ── Database helpers ──────────────────────────────────────────────────────────

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email.lower().strip()).first()


def create_user(db: Session, email: str, password: str) -> User:
    user = User(email=email.lower().strip(), password_hash=hash_password(password))
    db.add(user)
    db.flush()
    logger.info("Created new user: {}", user.email)
    return user


# ── Streamlit session helpers ─────────────────────────────────────────────────

def current_user() -> Optional[dict]:
    """Return the logged-in user dict from session_state, or None."""
    return st.session_state.get("auth_user")


def login_session(user: User) -> None:
    st.session_state["auth_user"] = {
        "id": user.id,
        "email": user.email,
    }


def logout_session() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]


# ── Streamlit pages ───────────────────────────────────────────────────────────

def login_page() -> None:
    """Render the login / register page. Redirects on success by rerunning."""
    st.set_page_config(page_title="CorporateRAG – Sign in", page_icon="🔍", layout="centered")

    st.title("🔍 CorporateRAG")
    st.caption("Unified search across Jira, Confluence, and SQL Server")

    tab_login, tab_register = st.tabs(["Sign in", "Create account"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", use_container_width=True)

        if submitted:
            if not email or not password:
                st.error("Please enter your email and password.")
                return
            with get_db() as db:
                user = get_user_by_email(db, email)
                if user is None or not verify_password(password, user.password_hash):
                    st.error("Invalid email or password.")
                    return
                if not user.is_active:
                    st.error("Account is disabled. Contact your administrator.")
                    return
                login_session(user)
            st.rerun()

    with tab_register:
        with st.form("register_form"):
            reg_email = st.text_input("Email", key="reg_email")
            reg_pwd = st.text_input("Password", type="password", key="reg_pwd")
            reg_pwd2 = st.text_input("Confirm password", type="password", key="reg_pwd2")
            reg_submitted = st.form_submit_button("Create account", use_container_width=True)

        if reg_submitted:
            if not reg_email or not reg_pwd:
                st.error("All fields are required.")
                return
            if reg_pwd != reg_pwd2:
                st.error("Passwords do not match.")
                return
            if len(reg_pwd) < 8:
                st.error("Password must be at least 8 characters.")
                return
            with get_db() as db:
                if get_user_by_email(db, reg_email) is not None:
                    st.error("An account with this email already exists.")
                    return
                user = create_user(db, reg_email, reg_pwd)
                login_session(user)
            st.success("Account created! Redirecting…")
            st.rerun()
