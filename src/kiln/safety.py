"""Safety gate. Cedar policies (policies/safety.cedar) decide whether a schedule can be approved.

Limits live in readable, reviewable policy-as-code, and the decision is fail-closed: anything other than a
clean Allow (a rule fired, a policy error, or no decision) is a rejection.
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path

import cedarpy

from .config import CLAY_PEAK_RANGE, FEATURES, GLOBAL_LIMITS, ROOT, WET_MOISTURE_PCT
from .models import TRAIN_RANGE, predict_one

POLICY_PATH = Path(os.getenv("KILN_POLICY", ROOT / "policies" / "safety.cedar"))
_POLICY_TEXT = POLICY_PATH.read_text(encoding="utf-8")


def _parse_policies(text: str) -> list[tuple[str, str]]:
    """(kind, reason) for each policy, in file order. Cedar names them policy0, policy1, ... in that order."""
    cleaned_text = re.sub(r"//[^\n]*", "", text)
    policies = []
    
    pattern = re.compile(
        r'((?:@\w+\("[^"]*"\)\s*)*)'  # Capture annotations
        r'(permit|forbid)\s*\(',        # Capture policy effect type (permit/forbid)
        re.DOTALL
    )
    
    for match in pattern.finditer(cleaned_text):
        annotations, effect = match.groups()
        reason_match = re.search(r'@reason\("([^"]*)"\)', annotations)
        reason = reason_match.group(1) if reason_match else "unnamed rule"
        policies.append((effect, reason))
        
    return policies


_POLICIES = _parse_policies(_POLICY_TEXT)


def bounds_for(clay_body: str, moisture_pct: float) -> dict:
    """Single source of truth for allowed control ranges."""
    b = {"peak_temp": CLAY_PEAK_RANGE[clay_body], **{k: v for k, v in GLOBAL_LIMITS.items()}}
    if moisture_pct > WET_MOISTURE_PCT:
        lo, hi = b["ramp_0_573"]
        b["ramp_0_573"] = (lo, hi * 0.5)
    return b


def _dec(x: float) -> dict:
    """Cedar has no floats; its decimal type keeps up to 4 decimal places."""
    return {"__extn": {"fn": "decimal", "arg": f"{x:.4f}"}}


def _decide(attrs: dict, max_defect_pct: float):
    """Ask Cedar. Returns (allowed, fired_policy_indexes, errors)."""
    r = cedarpy.is_authorized(
        {"principal": 'Operator::"kiln-agent"', "action": 'Action::"ApproveSchedule"',
         "resource": 'Schedule::"candidate"', "context": {"max_defect_pct": _dec(max_defect_pct)}},
        _POLICY_TEXT,
        [{"uid": {"type": "Schedule", "id": "candidate"}, "attrs": attrs, "parents": []}],
    )
    errors = [str(e) for e in r.diagnostics.errors]
    fired = sorted(int(re.sub(r"\D", "", p)) for p in r.diagnostics.reasons)
    return (r.decision.value == "Allow" and not errors), fired, errors


def _reject(msg: str, prediction: dict | None = None) -> dict:
    """Standardized rejection payload factory."""
    return {
        "approved": False,
        "violations": [msg],
        "warnings": [],
        "prediction": prediction,
        "decided_by": "Cedar",
    }


def validate_schedule(schedule: dict, max_defect_pct: float | None = None) -> dict:
    missing = [k for k in FEATURES if k not in schedule]
    if missing:
        return _reject(f"missing fields: {missing}")
    try:
        s = {k: schedule[k] if k == "clay_body" else float(schedule[k]) for k in FEATURES}
    except (TypeError, ValueError):
        return _reject("non-numeric value in a numeric field")
    if not isinstance(s["clay_body"], str) or not all(math.isfinite(v) for k, v in s.items() if k != "clay_body"):
        return _reject("invalid clay_body or non-finite number")

    pred = predict_one(s)
    attrs = {k: v if k == "clay_body" else _dec(v) for k, v in s.items()}
    attrs["predicted_defect_pct"] = _dec(pred["defect_rate_pct"])
    try:
        allowed, fired, errors = _decide(attrs, 100.0 if max_defect_pct is None else float(max_defect_pct))
    except Exception as e:  # fail closed
        return _reject(f"safety policy could not be evaluated ({type(e).__name__})", pred)

    violations = [_POLICIES[i][1] for i in fired if i < len(_POLICIES) and _POLICIES[i][0] == "forbid"]
    violations += [f"policy evaluation error: {e}" for e in errors]
    if not allowed and not violations:
        violations.append("safety policy returned no decision")

    warnings = []
    for name, (lo, hi) in (TRAIN_RANGE or {}).items():
        if not (lo <= s[name] <= hi):
            warnings.append(f"{name}={s[name]} is outside the training range [{lo:.1f}, {hi:.1f}]; prediction is an extrapolation")
    return {"approved": allowed, "violations": violations, "warnings": warnings, "prediction": pred, "decided_by": "Cedar"}