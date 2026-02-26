# core/auth.py
# ─────────────────────────────────────────────
# AquaRisk Authentication & Plan Management
# Handles: register, login, logout, session,
#          plan limits, plan expiry checks
# ─────────────────────────────────────────────

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from core.supabase_client import get_supabase


# ── Plan configuration ────────────────────────────────────────

PLAN_LIMITS = {
    "professional": {
        "max_wells":            2,
        "acei_enabled":         False,
        "scenarios_enabled":    False,
        "monte_carlo_visible":  False,
        "portfolio_enabled":    False,
        "excel_export":         False,
        "pdf_report":           True,
        "min_months":           3,
        "label":                "Professional",
        "price_usd":            299,
    },
    "advanced": {
        "max_wells":            4,
        "acei_enabled":         True,
        "scenarios_enabled":    True,
        "monte_carlo_visible":  True,
        "portfolio_enabled":    False,
        "excel_export":         False,
        "pdf_report":           True,
        "min_months":           6,
        "label":                "Advanced",
        "price_usd":            790,
    },
    "enterprise": {
        "max_wells":            5,
        "acei_enabled":         True,
        "scenarios_enabled":    True,
        "monte_carlo_visible":  True,
        "portfolio_enabled":    True,
        "excel_export":         True,
        "pdf_report":           True,
        "min_months":           12,
        "label":                "Enterprise",
        "price_usd":            None,   # custom
    },
}


# ── User session dataclass ────────────────────────────────────

@dataclass
class UserSession:
    user_id:        str
    email:          str
    full_name:      str
    company:        str
    plan:           str
    plan_status:    str
    plan_expires_at: Optional[datetime]
    access_token:   str
    refresh_token:  str

    @property
    def limits(self) -> dict:
        return PLAN_LIMITS.get(self.plan, PLAN_LIMITS["professional"])

    @property
    def is_active(self) -> bool:
        if self.plan_status != "active":
            return False
        if self.plan_expires_at is None:
            return True
        now = datetime.now(timezone.utc)
        exp = self.plan_expires_at
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        return now < exp

    @property
    def days_remaining(self) -> Optional[int]:
        if self.plan_expires_at is None:
            return None
        now = datetime.now(timezone.utc)
        exp = self.plan_expires_at
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        delta = (exp - now).days
        return max(delta, 0)

    def can(self, feature: str) -> bool:
        """Check if this session's plan includes a feature."""
        return bool(self.limits.get(feature, False))

    @property
    def max_wells(self) -> int:
        return self.limits["max_wells"]


# ── Auth functions ────────────────────────────────────────────

def register_user(
    email: str,
    password: str,
    full_name: str,
    company: str,
    plan: str = "professional"
) -> tuple[Optional[UserSession], Optional[str]]:
    """
    Register a new user via Supabase Auth.
    Returns (UserSession, None) on success or (None, error_message).
    """
    sb = get_supabase()

    try:
        response = sb.auth.sign_up({
            "email":    email,
            "password": password,
            "options": {
                "data": {
                    "full_name": full_name,
                    "company":   company,
                    "plan":      plan,
                }
            }
        })

        if response.user is None:
            return None, "Registration failed. Please try again."

        # Supabase may require email confirmation depending on project settings.
        # If email confirmation is disabled, session is available immediately.
        if response.session is None:
            return None, (
                "Registration successful! Please check your email "
                "to confirm your account, then log in."
            )

        session = _build_session(response)
        return session, None

    except Exception as e:
        return None, _parse_error(e)


def login_user(
    email: str,
    password: str
) -> tuple[Optional[UserSession], Optional[str]]:
    """
    Authenticate an existing user.
    Returns (UserSession, None) on success or (None, error_message).
    """
    sb = get_supabase()

    try:
        response = sb.auth.sign_in_with_password({
            "email":    email,
            "password": password,
        })

        if response.user is None or response.session is None:
            return None, "Invalid email or password."

        session = _build_session(response)

        if not session.is_active:
            return None, (
                f"Your {session.limits['label']} plan has expired. "
                "Please contact support to renew."
            )

        return session, None

    except Exception as e:
        return None, _parse_error(e)


def logout_user() -> None:
    """Sign out from Supabase (invalidates token server-side)."""
    try:
        get_supabase().auth.sign_out()
    except Exception:
        pass  # always clear local session regardless


def refresh_session(
    refresh_token: str
) -> tuple[Optional[UserSession], Optional[str]]:
    """Refresh an expired access token using the refresh token."""
    sb = get_supabase()
    try:
        response = sb.auth.refresh_session(refresh_token)
        if response.session is None:
            return None, "Session expired. Please log in again."
        session = _build_session(response)
        return session, None
    except Exception as e:
        return None, _parse_error(e)


# ── Well management ───────────────────────────────────────────

def get_user_wells(user_id: str) -> list[dict]:
    """Fetch all wells belonging to the user."""
    sb = get_supabase()
    response = (
        sb.table("wells")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at")
        .execute()
    )
    return response.data or []


def create_well(
    user_id: str,
    well_name: str,
    location: str,
    max_wells: int
) -> tuple[Optional[dict], Optional[str]]:
    """
    Create a new well if the user hasn't reached their plan limit.
    Returns (well_dict, None) or (None, error_message).
    """
    sb = get_supabase()
    existing = get_user_wells(user_id)

    if len(existing) >= max_wells:
        return None, (
            f"Your plan allows a maximum of {max_wells} well(s). "
            "Upgrade your plan to add more."
        )

    response = (
        sb.table("wells")
        .insert({
            "user_id":   user_id,
            "well_name": well_name.strip(),
            "location":  location.strip(),
        })
        .execute()
    )

    if response.data:
        return response.data[0], None
    return None, "Failed to create well. Please try again."


def delete_well(well_id: str, user_id: str) -> Optional[str]:
    """Delete a well and all its readings. Returns error string or None."""
    sb = get_supabase()
    try:
        sb.table("wells").delete().eq("id", well_id).eq("user_id", user_id).execute()
        return None
    except Exception as e:
        return _parse_error(e)


# ── Readings management ───────────────────────────────────────

def get_well_readings(well_id: str, user_id: str) -> list[dict]:
    """Fetch all readings for a well, ordered by date."""
    sb = get_supabase()
    response = (
        sb.table("well_readings")
        .select("*")
        .eq("well_id", well_id)
        .eq("user_id", user_id)
        .order("reading_date")
        .execute()
    )
    return response.data or []


def save_readings_bulk(
    well_id: str,
    user_id: str,
    readings: list[dict]   # [{"reading_date": "YYYY-MM-DD", "water_level": float}]
) -> Optional[str]:
    """
    Replace all readings for a well with a new set.
    Deletes existing rows first, then inserts new batch.
    Returns error string or None.
    """
    sb = get_supabase()
    try:
        # Delete existing
        sb.table("well_readings").delete()\
            .eq("well_id", well_id)\
            .eq("user_id", user_id)\
            .execute()

        if not readings:
            return None

        rows = [
            {
                "well_id":      well_id,
                "user_id":      user_id,
                "reading_date": r["reading_date"],
                "water_level":  float(r["water_level"]),
                "notes":        r.get("notes", ""),
            }
            for r in readings
        ]

        sb.table("well_readings").insert(rows).execute()
        return None

    except Exception as e:
        return _parse_error(e)


# ── Analysis run tracking ─────────────────────────────────────

def log_analysis_run(
    user_id: str,
    well_id: str,
    scenario: str,
    is_dry_run: bool = False
) -> None:
    """Log an analysis run to the database (for usage tracking)."""
    try:
        get_supabase().table("analysis_runs").insert({
            "user_id":    user_id,
            "well_id":    well_id,
            "scenario":   scenario,
            "is_dry_run": is_dry_run,
        }).execute()
    except Exception:
        pass  # non-critical, don't block the user


def get_monthly_run_count(user_id: str) -> int:
    """Count real (non-dry) analysis runs this calendar month."""
    sb = get_supabase()
    now = datetime.now(timezone.utc)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    response = (
        sb.table("analysis_runs")
        .select("id", count="exact")
        .eq("user_id", user_id)
        .eq("is_dry_run", False)
        .gte("run_at", month_start.isoformat())
        .execute()
    )
    return response.count or 0


# ── Internal helpers ──────────────────────────────────────────

def _build_session(response) -> UserSession:
    """Build a UserSession from a Supabase auth response."""
    sb   = get_supabase()
    user = response.user
    sess = response.session

    # Fetch profile from DB for plan info
    profile_resp = (
        sb.table("profiles")
        .select("*")
        .eq("id", str(user.id))
        .single()
        .execute()
    )
    profile = profile_resp.data or {}

    # Parse expiry date
    expires_raw = profile.get("plan_expires_at")
    expires_at  = None
    if expires_raw:
        try:
            expires_at = datetime.fromisoformat(
                expires_raw.replace("Z", "+00:00")
            )
        except Exception:
            expires_at = None

    return UserSession(
        user_id        = str(user.id),
        email          = user.email or "",
        full_name      = profile.get("full_name") or user.user_metadata.get("full_name", ""),
        company        = profile.get("company") or user.user_metadata.get("company", ""),
        plan           = profile.get("plan", "professional"),
        plan_status    = profile.get("plan_status", "active"),
        plan_expires_at= expires_at,
        access_token   = sess.access_token,
        refresh_token  = sess.refresh_token,
    )


def _parse_error(e: Exception) -> str:
    """Extract a clean error message from Supabase exceptions."""
    msg = str(e)
    if "Invalid login credentials" in msg:
        return "Invalid email or password."
    if "Email not confirmed" in msg:
        return "Please confirm your email before logging in."
    if "User already registered" in msg:
        return "An account with this email already exists."
    if "Password should be at least" in msg:
        return "Password must be at least 6 characters."
    if "Unable to validate email" in msg:
        return "Please enter a valid email address."
    return f"An error occurred: {msg}"
