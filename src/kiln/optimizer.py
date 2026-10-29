"""Schedule search (energy minimized under a defect target) and what-if comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BASELINE_CONTROLS, CLAY_PEAK_RANGE, CONTROLS, FEATURES
from .models import MODEL_LABEL, PREDICT_FN, predict_one, rnd
from .safety import bounds_for, validate_schedule
from .trace import CHART


def run_optimize(load_kg, num_pieces, clay_body, moisture_pct, ambient_temp,
              max_defect_pct: float = 5.0, n_candidates: int = 5000, seed: int = 42) -> dict:
    if clay_body not in CLAY_PEAK_RANGE:
        return {"error": f"unknown clay_body '{clay_body}', known: {list(CLAY_PEAK_RANGE)}"}

    rng = np.random.default_rng(seed)
    b = bounds_for(clay_body, moisture_pct)
    cand = pd.DataFrame({c: rng.uniform(*b[c], n_candidates) for c in CONTROLS})
    for c, v in zip(["load_kg", "num_pieces", "moisture_pct", "ambient_temp"], [load_kg, num_pieces, moisture_pct, ambient_temp]):
        cand[c] = float(v)
    cand["clay_body"] = clay_body
    cand["energy_kwh"], cand["defect_rate_pct"] = PREDICT_FN(cand)

    # Pareto front: sorted by energy, keep points that lower the best defect rate so far.
    srt = cand.sort_values("energy_kwh")
    front, best_def = [], np.inf
    for idx, row in srt.iterrows():
        if row.defect_rate_pct < best_def:
            best_def = row.defect_rate_pct
            front.append(idx)

    feasible = cand[cand.defect_rate_pct <= max_defect_pct]
    met_target = not feasible.empty
    best = feasible.nsmallest(1, "energy_kwh").iloc[0] if met_target else cand.nsmallest(1, "defect_rate_pct").iloc[0]

    def to_schedule(row):
        return {k: (row[k] if k == "clay_body" else rnd(row[k])) for k in FEATURES}

    rec = to_schedule(best)
    check = validate_schedule(rec, max_defect_pct)
    base = {**rec, **BASELINE_CONTROLS, "peak_temp": float(np.mean(b["peak_temp"]))}
    base_pred = predict_one(base)
    saving = 100 * (base_pred["energy_kwh"] - check["prediction"]["energy_kwh"]) / base_pred["energy_kwh"]

    CHART.update(
        tested=cand[["energy_kwh", "defect_rate_pct"]].round(2).values[::25].tolist(),
        front=cand.loc[front, ["energy_kwh", "defect_rate_pct"]].round(2).values.tolist(),
    )
    pareto_pts = cand.loc[front]
    pareto_pts = pareto_pts.iloc[np.linspace(0, len(pareto_pts) - 1, min(5, len(pareto_pts))).astype(int)]
    return {
        "target_met": bool(met_target),
        "recommended_schedule": rec,
        "predicted": check["prediction"],
        "validation": {k: check[k] for k in ("approved", "violations", "warnings")},
        "baseline_prediction": base_pred,
        "energy_saving_vs_baseline_pct": rnd(saving),
        "baseline_schedule": base,
        "pareto_tradeoffs": [
            {"energy_kwh": rnd(r.energy_kwh), "defect_rate_pct": rnd(r.defect_rate_pct), "peak_temp": rnd(r.peak_temp)}
            for r in pareto_pts.itertuples()
        ],
        "model_used": MODEL_LABEL,
        "note": None if met_target else f"No candidate met the {max_defect_pct}% defect target; returning the lowest-defect schedule found.",
    }

def run_what_if(base_schedule: dict, changes: dict) -> dict:
    changed = {**base_schedule, **changes}
    before, after = validate_schedule(base_schedule), validate_schedule(changed)
    delta = None
    if before["prediction"] and after["prediction"]:
        delta = {k: rnd(after["prediction"][k] - before["prediction"][k]) for k in before["prediction"]}
    return {"base": before, "changed": after, "delta_changed_minus_base": delta}
