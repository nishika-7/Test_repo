"""Safety tests. They run without trained models (max_defect_pct=None) unless a test sets its own prediction."""
import math

import numpy as np
import pytest

from kiln import safety
from kiln.config import CLAY_PEAK_RANGE
from kiln.safety import bounds_for, validate_schedule

GOOD = {"peak_temp": 1250, "soak_minutes": 60, "ramp_0_573": 60, "ramp_573_peak": 120, "cooling_rate": 80,
        "load_kg": 400, "num_pieces": 60, "clay_body": "stoneware", "moisture_pct": 1.5, "ambient_temp": 30}


@pytest.fixture(autouse=True)
def fast_prediction(monkeypatch):
    """Safety tests check the rules, not the ML models: use a fixed prediction so they are fast and need no trained models."""
    monkeypatch.setattr(safety, "predict_one", lambda s: {"energy_kwh": 100.0, "defect_rate_pct": 2.0})


def reasons(**changes):
    return " | ".join(validate_schedule({**GOOD, **changes})["violations"])


def test_good_schedule_is_approved():
    r = validate_schedule(GOOD)
    assert r["approved"] and r["violations"] == [] and r["decided_by"] == "Cedar"


@pytest.mark.parametrize("change, expected", [
    ({"ramp_0_573": 200}, "573 degC must be 20 to 150"),
    ({"ramp_0_573": 10}, "573 degC must be 20 to 150"),
    ({"moisture_pct": 3.0, "ramp_0_573": 100}, "wet ware"),
    ({"ramp_573_peak": 400}, "573 degC to peak"),
    ({"cooling_rate": 400}, "cooling"),
    ({"soak_minutes": 5}, "soak"),
    ({"peak_temp": 1400}, "stoneware peak"),
    ({"clay_body": "porcelain", "peak_temp": 1100}, "porcelain peak"),
    ({"clay_body": "earthenware", "peak_temp": 1250}, "earthenware peak"),
    ({"load_kg": 900}, "kiln capacity"),
    ({"load_kg": 0}, "kiln capacity"),
    ({"clay_body": "granite"}, "unknown clay body"),
])
def test_unsafe_schedules_are_rejected_with_the_right_reason(change, expected):
    r = validate_schedule({**GOOD, **change})
    assert not r["approved"]
    assert expected in " | ".join(r["violations"])


def test_every_violation_is_reported_together():
    r = validate_schedule({**GOOD, "ramp_0_573": 200, "cooling_rate": 400, "soak_minutes": 5})
    assert len(r["violations"]) == 3


@pytest.mark.parametrize("bad", [
    {k: v for k, v in GOOD.items() if k != "soak_minutes"},
    {**GOOD, "peak_temp": "hot"},
    {**GOOD, "peak_temp": math.nan},
    {**GOOD, "ramp_0_573": math.inf},
    {**GOOD, "clay_body": 5},
])
def test_malformed_input_is_rejected(bad):
    assert not validate_schedule(bad)["approved"]


def test_defect_target_uses_the_models_prediction(monkeypatch):
    monkeypatch.setattr(safety, "predict_one", lambda s: {"energy_kwh": 100.0, "defect_rate_pct": 9.0})
    assert not validate_schedule(GOOD, max_defect_pct=5.0)["approved"]
    assert "defect" in " ".join(validate_schedule(GOOD, max_defect_pct=5.0)["violations"])
    assert validate_schedule(GOOD, max_defect_pct=10.0)["approved"]


def test_policy_errors_fail_closed():
    attrs = {"clay_body": "stoneware"}  # most attributes missing -> policies error out
    allowed, _, errors = safety._decide(attrs, 5.0)
    assert not allowed and errors


def test_everything_the_optimizer_can_sample_is_allowed_by_cedar():
    """config.py (sampling ranges) and policies/safety.cedar must agree."""
    rng = np.random.default_rng(0)
    for clay in CLAY_PEAK_RANGE:
        for _ in range(150):
            moisture = float(rng.uniform(0, 12))
            b = bounds_for(clay, moisture)
            s = {k: float(rng.uniform(*b[k])) for k in ("peak_temp", "soak_minutes", "ramp_0_573", "ramp_573_peak", "cooling_rate")}
            s.update(load_kg=float(rng.uniform(1, 800)), num_pieces=60, clay_body=clay, moisture_pct=moisture, ambient_temp=30)
            r = validate_schedule(s)
            assert r["approved"], (clay, s, r["violations"])