# core/supabase_client.py
# ─────────────────────────────────────────────
# Supabase client — singleton pattern
# Reads credentials from:
#   1. Streamlit Secrets (when deployed on Streamlit Cloud)
#   2. .env file (local development)
#   3. Environment variables (fallback)
# ─────────────────────────────────────────────

import os
from supabase import create_client, Client

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional in cloud

_client: Client | None = None


def _get_credentials() -> tuple[str, str]:
    """
    Try Streamlit secrets first, then env vars.
    Returns (url, anon_key).
    """
    url = None
    key = None

    # 1. Streamlit Secrets (Streamlit Cloud)
    try:
        import streamlit as st
        url = st.secrets.get("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_ANON_KEY")
    except Exception:
        pass

    # 2. Environment variables / .env (local)
    if not url:
        url = os.environ.get("SUPABASE_URL")
    if not key:
        key = os.environ.get("SUPABASE_ANON_KEY")

    if not url or not key:
        raise RuntimeError(
            "Missing SUPABASE_URL or SUPABASE_ANON_KEY.\n"
            "• Local: add them to your .env file.\n"
            "• Streamlit Cloud: add them in Settings → Secrets."
        )

    return url, key


def get_supabase() -> Client:
    global _client
    if _client is None:
        url, key = _get_credentials()
        _client = create_client(url, key)
    return _client


def reset_client() -> None:
    """Force re-initialization (useful after secrets update)."""
    global _client
    _client = None
