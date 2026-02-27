# core/anomaly_detector.py
# ─────────────────────────────────────────────
# AquaRisk — Anomaly Detection Engine
# Detects suspicious readings before they
# enter the forecasting model using:
#   - IQR (interquartile range) method
#   - Z-score method
#   - Rate-of-change spike detection
#   - Physical plausibility check
# ─────────────────────────────────────────────

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class AnomalyFlag:
    row_index:   int
    date:        str
    value:       float
    reason:      str
    severity:    str        # "warning" | "critical"
    suggestion:  Optional[float] = None   # interpolated replacement


@dataclass
class AnomalyReport:
    flags:          list[AnomalyFlag] = field(default_factory=list)
    clean_values:   list[float]       = field(default_factory=list)
    original_values:list[float]       = field(default_factory=list)
    n_flagged:      int               = 0
    n_critical:     int               = 0
    model_safe:     bool              = True  # False if too many anomalies

    @property
    def has_anomalies(self) -> bool:
        return len(self.flags) > 0

    @property
    def summary(self) -> str:
        if not self.flags:
            return "All readings passed validation."
        parts = [f"{self.n_flagged} anomalous reading(s) detected"]
        if self.n_critical:
            parts.append(f"{self.n_critical} critical")
        if not self.model_safe:
            parts.append("⚠ Too many anomalies — forecast reliability reduced")
        return " · ".join(parts)


def detect_anomalies(
    dates:  list[str],
    values: list[float],
    *,
    iqr_multiplier:    float = 2.5,
    zscore_threshold:  float = 3.0,
    max_monthly_change: float = 8.0,   # meters — max plausible change/month
    min_level:         float = 0.1,    # meters — physically implausible below
    max_level:         float = 200.0,  # meters — physically implausible above
) -> AnomalyReport:
    """
    Run all anomaly checks on a list of (date, value) readings.

    Parameters
    ----------
    dates             : list of "YYYY-MM-DD" strings
    values            : corresponding water level readings (m)
    iqr_multiplier    : how many IQR widths beyond Q1/Q3 = outlier
    zscore_threshold  : z-score above this = outlier
    max_monthly_change: max realistic level change between readings (m)
    min_level         : physically impossible below this
    max_level         : physically impossible above this

    Returns
    -------
    AnomalyReport with flags, clean values, and model safety assessment
    """
    if len(values) < 3:
        # Not enough data to detect anomalies statistically
        return AnomalyReport(
            clean_values=list(values),
            original_values=list(values),
            model_safe=True,
        )

    arr    = np.array(values, dtype=float)
    flags  = []
    flagged_indices = set()

    # ── 1. Physical plausibility ──────────────────────────
    for i, (v, d) in enumerate(zip(values, dates)):
        if v < min_level:
            flags.append(AnomalyFlag(
                row_index=i, date=d, value=v,
                reason=f"Value {v:.2f} m is below physical minimum ({min_level} m). Likely sensor error or data entry mistake.",
                severity="critical",
                suggestion=None,
            ))
            flagged_indices.add(i)
        elif v > max_level:
            flags.append(AnomalyFlag(
                row_index=i, date=d, value=v,
                reason=f"Value {v:.2f} m exceeds physical maximum ({max_level} m). Likely unit error (feet vs meters?).",
                severity="critical",
                suggestion=round(v * 0.3048, 2),  # feet → meters suggestion
            ))
            flagged_indices.add(i)

    # ── 2. IQR outlier detection ──────────────────────────
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr    = q3 - q1
    lower  = q1 - iqr_multiplier * iqr
    upper  = q3 + iqr_multiplier * iqr

    for i, (v, d) in enumerate(zip(values, dates)):
        if i in flagged_indices:
            continue
        if v < lower or v > upper:
            direction = "low" if v < lower else "high"
            flags.append(AnomalyFlag(
                row_index=i, date=d, value=v,
                reason=f"Statistical outlier ({direction}): {v:.2f} m is outside the expected range [{lower:.1f}–{upper:.1f} m].",
                severity="warning",
                suggestion=round(float(np.median(arr)), 2),
            ))
            flagged_indices.add(i)

    # ── 3. Z-score ────────────────────────────────────────
    mean, std = float(np.mean(arr)), float(np.std(arr))
    if std > 0:
        for i, (v, d) in enumerate(zip(values, dates)):
            if i in flagged_indices:
                continue
            z = abs((v - mean) / std)
            if z > zscore_threshold:
                flags.append(AnomalyFlag(
                    row_index=i, date=d, value=v,
                    reason=f"High z-score ({z:.1f}σ): value deviates significantly from the dataset mean ({mean:.1f} m).",
                    severity="warning",
                    suggestion=round(mean, 2),
                ))
                flagged_indices.add(i)

    # ── 4. Rate-of-change spike ───────────────────────────
    for i in range(1, len(values)):
        if i in flagged_indices or (i - 1) in flagged_indices:
            continue
        delta = abs(values[i] - values[i - 1])
        if delta > max_monthly_change:
            flags.append(AnomalyFlag(
                row_index=i, date=dates[i], value=values[i],
                reason=f"Sudden change of {delta:.1f} m from previous reading ({values[i-1]:.2f} m). "
                       f"Exceeds max plausible monthly change ({max_monthly_change} m).",
                severity="warning" if delta < max_monthly_change * 2 else "critical",
                suggestion=round(
                    float(np.interp(i, [i-1, min(i+1, len(values)-1)],
                                    [values[i-1], values[min(i+1, len(values)-1)]])), 2
                ) if i < len(values) - 1 else round(values[i-1], 2),
            ))
            flagged_indices.add(i)

    # ── Build clean values (interpolate flagged points) ──
    clean = list(values)
    clean_arr = arr.copy()

    for idx in sorted(flagged_indices):
        # Linear interpolation using nearest non-flagged neighbors
        prev_idx = next((j for j in range(idx-1, -1, -1) if j not in flagged_indices), None)
        next_idx = next((j for j in range(idx+1, len(values)) if j not in flagged_indices), None)

        if prev_idx is not None and next_idx is not None:
            clean_arr[idx] = np.interp(idx, [prev_idx, next_idx],
                                        [clean_arr[prev_idx], clean_arr[next_idx]])
        elif prev_idx is not None:
            clean_arr[idx] = clean_arr[prev_idx]
        elif next_idx is not None:
            clean_arr[idx] = clean_arr[next_idx]
        else:
            clean_arr[idx] = float(np.median(arr))

        clean[idx] = round(float(clean_arr[idx]), 3)

        # Update suggestion on flag
        for flag in flags:
            if flag.row_index == idx and flag.suggestion is None:
                flag.suggestion = clean[idx]

    # Sort flags by row index
    flags.sort(key=lambda f: f.row_index)

    n_critical  = sum(1 for f in flags if f.severity == "critical")
    n_flagged   = len(flags)
    model_safe  = (n_flagged / max(len(values), 1)) < 0.25  # >25% flagged = unreliable

    return AnomalyReport(
        flags=flags,
        clean_values=clean,
        original_values=list(values),
        n_flagged=n_flagged,
        n_critical=n_critical,
        model_safe=model_safe,
    )


def format_anomaly_for_display(flag: AnomalyFlag) -> dict:
    """Convert an AnomalyFlag to a display-friendly dict for Streamlit."""
    icon = "🔴" if flag.severity == "critical" else "🟡"
    return {
        "icon":       icon,
        "date":       flag.date,
        "value":      f"{flag.value:.2f} m",
        "issue":      flag.reason,
        "suggestion": f"{flag.suggestion:.2f} m" if flag.suggestion else "—",
        "severity":   flag.severity.title(),
    }
