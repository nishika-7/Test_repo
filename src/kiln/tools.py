"""The four Strands tools. Thin wrappers; the docstrings are what the LLM reads."""
from __future__ import annotations

from strands import tool

from .models import predict_one
from .optimizer import run_optimize, run_what_if
from .safety import validate_schedule
from .trace import traced


@tool
@traced
def predict(schedule: dict) -> dict:
    """Predict energy use and defect rate for ONE complete firing schedule using the trained models.

    Args:
        schedule: dict with ALL of: peak_temp (degC), soak_minutes, ramp_0_573 (degC/h),
            ramp_573_peak (degC/h), cooling_rate (degC/h), load_kg, num_pieces,
            clay_body (earthenware | stoneware | porcelain), moisture_pct, ambient_temp (degC).

    Returns:
        energy_kwh and defect_rate_pct. Does NOT check safety; use validate for that.
    """
    try:
        return predict_one(schedule)
    except KeyError as e:
        return {"error": f"missing field {e}"}


@tool
@traced
def optimize(load_kg: float, num_pieces: int, clay_body: str, moisture_pct: float,
             ambient_temp: float, max_defect_pct: float = 5.0) -> dict:
    """Generate a firing schedule that minimizes energy while keeping predicted defect rate under a target.

    Searches only inside the safe control ranges. Always run `validate` on the result before
    presenting it as a recommendation.

    Args:
        load_kg: Weight of ware in the kiln (kg).
        num_pieces: Number of pieces in the load.
        clay_body: earthenware, stoneware or porcelain.
        moisture_pct: Moisture content of the unfired ware (percent).
        ambient_temp: Ambient temperature around the kiln (degC).
        max_defect_pct: Highest acceptable predicted defect rate (percent). Default 5.

    Returns:
        recommended_schedule, predicted energy/defects, saving vs a baseline firing,
        and a few Pareto trade-off points (more energy buys fewer defects).
    """
    return run_optimize(load_kg, num_pieces, clay_body, moisture_pct, ambient_temp, max_defect_pct)


@tool
@traced
def validate(schedule: dict, max_defect_pct: float = 5.0) -> dict:
    """Approve or reject a firing schedule. Hard safety limits are enforced in code.

    Checks: all fields present, known clay body, peak temperature range for that clay,
    ramp rates (including the slow-through-573C rule and wet-ware reduction), cooling rate,
    soak time, kiln capacity, and that the predicted defect rate is under target.
    Also warns when inputs fall outside the model's training range.

    Args:
        schedule: dict with ALL of peak_temp, soak_minutes, ramp_0_573, ramp_573_peak,
            cooling_rate, load_kg, num_pieces, clay_body, moisture_pct, ambient_temp.
        max_defect_pct: Defect-rate target the schedule must meet. Default 5.

    Returns:
        approved (bool), violations (reasons for rejection), warnings, and the prediction.
        If approved is false, never present the schedule as safe.
    """
    return validate_schedule(schedule, max_defect_pct)


@tool
@traced
def what_if(base_schedule: dict, changes: dict) -> dict:
    """Compare a schedule with a modified copy, e.g. 'what if we cool faster or lower the peak by 20 degrees?'

    Args:
        base_schedule: A complete schedule (all fields, as in predict).
        changes: Only the fields to override, e.g. {"cooling_rate": 150}.

    Returns:
        Validation + prediction for both versions and the change in energy and defect rate.
    """
    return run_what_if(base_schedule, changes)