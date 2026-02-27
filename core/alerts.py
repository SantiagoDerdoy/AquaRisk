# core/alerts.py
# ─────────────────────────────────────────────
# AquaRisk — Automated Email Alert System
# Sends alerts when ACEI™ or risk metrics
# exceed user-configured thresholds.
#
# Uses Gmail SMTP (or any SMTP provider).
# Credentials stored in Streamlit Secrets / .env
# ─────────────────────────────────────────────

from __future__ import annotations
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone
from typing import Optional


# ── Load credentials ──────────────────────────

def _get_smtp_config() -> dict:
    """
    Load SMTP credentials from Streamlit Secrets or environment.
    Expected keys:
        SMTP_HOST       e.g. smtp.gmail.com
        SMTP_PORT       e.g. 587
        SMTP_USER       e.g. alerts@aquarisk.io
        SMTP_PASSWORD   app password (not account password for Gmail)
        ALERT_FROM      e.g. AquaRisk Alerts <alerts@aquarisk.io>
    """
    cfg = {}

    try:
        import streamlit as st
        cfg["host"]     = st.secrets.get("SMTP_HOST",     "")
        cfg["port"]     = int(st.secrets.get("SMTP_PORT", 587))
        cfg["user"]     = st.secrets.get("SMTP_USER",     "")
        cfg["password"] = st.secrets.get("SMTP_PASSWORD", "")
        cfg["from"]     = st.secrets.get("ALERT_FROM",    cfg["user"])
    except Exception:
        pass

    # Fallback to env vars
    if not cfg.get("host"):
        cfg["host"]     = os.environ.get("SMTP_HOST",     "smtp.gmail.com")
        cfg["port"]     = int(os.environ.get("SMTP_PORT", 587))
        cfg["user"]     = os.environ.get("SMTP_USER",     "")
        cfg["password"] = os.environ.get("SMTP_PASSWORD", "")
        cfg["from"]     = os.environ.get("ALERT_FROM",    cfg["user"])

    return cfg


def smtp_configured() -> bool:
    """Returns True if SMTP credentials are available."""
    cfg = _get_smtp_config()
    return bool(cfg.get("user") and cfg.get("password") and cfg.get("host"))


# ── Alert level config ────────────────────────

ALERT_LEVELS = {
    "watch":    {"acei_min": 40, "color": "#f59e0b", "label": "Watch",    "emoji": "🟡"},
    "warning":  {"acei_min": 60, "color": "#ef4444", "label": "Warning",  "emoji": "🟠"},
    "critical": {"acei_min": 75, "color": "#dc2626", "label": "Critical", "emoji": "🔴"},
}


def get_alert_level(acei_score: float) -> Optional[dict]:
    """Return the highest applicable alert level for a given ACEI score."""
    level = None
    for key in ["watch", "warning", "critical"]:
        if acei_score >= ALERT_LEVELS[key]["acei_min"]:
            level = {**ALERT_LEVELS[key], "key": key}
    return level


# ── Email composition ─────────────────────────

def _build_alert_email(
    recipient_name:   str,
    recipient_email:  str,
    well_name:        str,
    acei_score:       float,
    acei_category:    str,
    exceedance_prob:  float,
    risk_level:       str,
    threshold_value:  float,
    current_level:    float,
    mean_cross_months: Optional[float],
    scenario:         str,
    alert_level:      dict,
) -> MIMEMultipart:

    subject = (
        f"{alert_level['emoji']} AquaRisk {alert_level['label']}: "
        f"Well {well_name} — ACEI™ {acei_score:.1f}/100"
    )

    cross_str = (
        f"{mean_cross_months:.1f} months"
        if mean_cross_months is not None
        else "Not projected within forecast horizon"
    )

    now_str = datetime.now(timezone.utc).strftime("%B %d, %Y at %H:%M UTC")

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: 'Helvetica Neue', Arial, sans-serif; background:#060d1a; color:#cbd5e1; margin:0; padding:0; }}
  .wrapper {{ max-width:600px; margin:0 auto; background:#0c1624; border:1px solid rgba(255,255,255,0.08); border-radius:12px; overflow:hidden; }}
  .header {{ background:linear-gradient(135deg,#0e1e38,#0a1628); padding:32px 36px; border-bottom:1px solid rgba(255,255,255,0.06); }}
  .logo {{ font-family:Georgia,serif; font-size:1.4rem; color:#f1f5f9; margin-bottom:4px; }}
  .logo span {{ color:#60a5fa; }}
  .alert-badge {{ display:inline-block; background:{alert_level['color']}22; border:1px solid {alert_level['color']}55; color:{alert_level['color']}; font-size:0.75rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; padding:5px 14px; border-radius:100px; margin-top:10px; }}
  .body {{ padding:32px 36px; }}
  .greeting {{ font-size:1rem; color:#94a3b8; margin-bottom:24px; }}
  .alert-title {{ font-family:Georgia,serif; font-size:1.6rem; color:#f1f5f9; margin-bottom:8px; line-height:1.2; }}
  .alert-sub {{ font-size:0.9rem; color:#64748b; margin-bottom:28px; }}
  .metrics {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:24px; }}
  .metric {{ background:#0f1d2e; border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:16px 18px; }}
  .metric-label {{ font-size:0.65rem; color:#4a5568; text-transform:uppercase; letter-spacing:0.1em; font-weight:700; margin-bottom:6px; }}
  .metric-value {{ font-family:Georgia,serif; font-size:1.4rem; color:#e2e8f0; font-weight:600; }}
  .acei-bar-wrap {{ background:#0f1d2e; border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:18px 20px; margin-bottom:24px; }}
  .acei-bar-bg {{ background:rgba(255,255,255,0.06); border-radius:100px; height:8px; margin:10px 0 6px; overflow:hidden; }}
  .acei-bar-fill {{ height:100%; border-radius:100px; background:linear-gradient(90deg,{alert_level['color']},{alert_level['color']}88); width:{min(acei_score, 100):.1f}%; }}
  .rec-box {{ background:rgba({('239,68,68' if acei_score>=75 else '245,158,11' if acei_score>=60 else '245,158,11')},0.08); border-left:3px solid {alert_level['color']}; border-radius:8px; padding:14px 18px; margin-bottom:28px; font-size:0.875rem; color:#94a3b8; line-height:1.6; }}
  .cta {{ text-align:center; margin-bottom:28px; }}
  .cta a {{ display:inline-block; background:#2563eb; color:white; text-decoration:none; padding:13px 32px; border-radius:8px; font-weight:700; font-size:0.88rem; letter-spacing:0.04em; }}
  .footer {{ background:#080e18; border-top:1px solid rgba(255,255,255,0.05); padding:20px 36px; font-size:0.75rem; color:#374151; }}
</style>
</head>
<body>
<div class="wrapper">
  <div class="header">
    <div class="logo">Aqua<span>Risk</span></div>
    <div class="alert-badge">{alert_level['emoji']} {alert_level['label']} Alert</div>
  </div>
  <div class="body">
    <p class="greeting">Hello {recipient_name},</p>
    <div class="alert-title">Well {well_name} requires attention.</div>
    <div class="alert-sub">Scenario: {scenario.title()} · Generated {now_str}</div>

    <div class="metrics">
      <div class="metric">
        <div class="metric-label">Current Level</div>
        <div class="metric-value">{current_level:.2f} m</div>
      </div>
      <div class="metric">
        <div class="metric-label">Threshold</div>
        <div class="metric-value">{threshold_value:.1f} m</div>
      </div>
      <div class="metric">
        <div class="metric-label">Exceedance Probability</div>
        <div class="metric-value" style="color:{'#ef4444' if exceedance_prob>0.5 else '#f59e0b' if exceedance_prob>0.2 else '#10b981'}">{exceedance_prob:.1%}</div>
      </div>
      <div class="metric">
        <div class="metric-label">Risk Classification</div>
        <div class="metric-value" style="font-size:1.1rem;">{risk_level}</div>
      </div>
    </div>

    <div class="acei-bar-wrap">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <span style="font-size:0.72rem;color:#4a5568;text-transform:uppercase;letter-spacing:0.1em;font-weight:700;">ACEI™ Score</span>
        <span style="font-family:Georgia,serif;font-size:1.3rem;color:{alert_level['color']};font-weight:600;">{acei_score:.1f} / 100 — {acei_category}</span>
      </div>
      <div class="acei-bar-bg"><div class="acei-bar-fill"></div></div>
      <div style="font-size:0.75rem;color:#4a5568;">Estimated threshold crossing: <strong style="color:#94a3b8;">{cross_str}</strong></div>
    </div>

    <div class="rec-box">
      <strong style="color:#e2e8f0;">Recommended action:</strong><br>
      {_get_recommendation(acei_score)}
    </div>

    <div class="cta">
      <a href="http://localhost:8501">Open AquaRisk Dashboard →</a>
    </div>
  </div>
  <div class="footer">
    You're receiving this because automated alerts are enabled for your AquaRisk account.
    To adjust alert preferences, visit your Account settings.
    &nbsp;·&nbsp; AquaRisk Analytics &nbsp;·&nbsp; info@aquarisk.io
  </div>
</div>
</body>
</html>
"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = _get_smtp_config().get("from", "AquaRisk <alerts@aquarisk.io>")
    msg["To"]      = recipient_email
    msg.attach(MIMEText(html, "html"))
    return msg


def _get_recommendation(acei_score: float) -> str:
    if acei_score >= 75:
        return (
            "Immediate review recommended. Consider reducing extraction rates and consulting "
            "with your water manager. Prepare contingency supply plans."
        )
    elif acei_score >= 60:
        return (
            "Increase monitoring frequency. Review pumping schedules for the next 60 days "
            "and evaluate whether managed aquifer recharge is feasible."
        )
    else:
        return (
            "Monitor conditions closely. No immediate action required, but trend warrants "
            "attention. Review again when next readings are available."
        )


# ── Send function ─────────────────────────────

def send_alert_email(
    recipient_name:    str,
    recipient_email:   str,
    well_name:         str,
    acei_score:        float,
    acei_category:     str,
    exceedance_prob:   float,
    risk_level:        str,
    threshold_value:   float,
    current_level:     float,
    mean_cross_months: Optional[float],
    scenario:          str,
) -> tuple[bool, str]:
    """
    Send an alert email if ACEI score exceeds thresholds.

    Returns (success: bool, message: str)
    """
    alert_level = get_alert_level(acei_score)
    if alert_level is None:
        return False, "ACEI score below alert threshold — no email sent."

    if not smtp_configured():
        return False, (
            "Email alerts not configured. Add SMTP_HOST, SMTP_PORT, "
            "SMTP_USER, SMTP_PASSWORD to your Streamlit Secrets."
        )

    cfg = _get_smtp_config()
    msg = _build_alert_email(
        recipient_name=recipient_name,
        recipient_email=recipient_email,
        well_name=well_name,
        acei_score=acei_score,
        acei_category=acei_category,
        exceedance_prob=exceedance_prob,
        risk_level=risk_level,
        threshold_value=threshold_value,
        current_level=current_level,
        mean_cross_months=mean_cross_months,
        scenario=scenario,
        alert_level=alert_level,
    )

    try:
        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(cfg["user"], cfg["password"])
            server.sendmail(cfg["from"], recipient_email, msg.as_string())
        return True, f"Alert email sent to {recipient_email} ({alert_level['label']} level)."
    except smtplib.SMTPAuthenticationError:
        return False, "SMTP authentication failed. Check your email credentials."
    except smtplib.SMTPException as e:
        return False, f"SMTP error: {e}"
    except Exception as e:
        return False, f"Failed to send email: {e}"


# ── Supabase alert log ────────────────────────

def log_alert_sent(
    user_id:   str,
    well_id:   str,
    well_name: str,
    acei_score: float,
    alert_level: str,
) -> None:
    """Log a sent alert to Supabase for deduplication (avoid spam)."""
    try:
        from core.supabase_client import get_supabase
        get_supabase().table("alert_log").insert({
            "user_id":    user_id,
            "well_id":    well_id,
            "well_name":  well_name,
            "acei_score": acei_score,
            "alert_level": alert_level,
        }).execute()
    except Exception:
        pass  # non-critical


def was_alert_sent_recently(
    user_id:    str,
    well_id:    str,
    hours:      int = 24,
) -> bool:
    """
    Check if an alert was already sent for this well in the last N hours.
    Prevents duplicate alerts on every dashboard refresh.
    """
    try:
        from core.supabase_client import get_supabase
        from datetime import timedelta
        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=hours)
        ).isoformat()

        resp = (
            get_supabase()
            .table("alert_log")
            .select("id", count="exact")
            .eq("user_id", user_id)
            .eq("well_id", well_id)
            .gte("sent_at", cutoff)
            .execute()
        )
        return (resp.count or 0) > 0
    except Exception:
        return False  # if unsure, allow sending
