# core/supabase_client.py
# ─────────────────────────────────────────────
# Supabase client — singleton pattern
# Used by auth and all DB operations
# ─────────────────────────────────────────────

import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

_client: Client | None = None


def get_supabase() -> Client:
    global _client
    if _client is None:
        url  = os.environ.get("SUPABASE_URL")
        key  = os.environ.get("SUPABASE_ANON_KEY")
        if not url or not key:
            raise RuntimeError(
                "Missing SUPABASE_URL or SUPABASE_ANON_KEY in environment. "
                "Check your .env file."
            )
        _client = create_client(url, key)
    return _client
