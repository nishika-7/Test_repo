"""Scripted demo mode: runs the same optimize and validate tools as the agent, with no LLM.

Used when KILN_LLM=scripted, or automatically when the LLM is unreachable (so a judge without Ollama or AWS
can still run the demo). It is always labelled as scripted in the reply and in the UI.
"""
from __future__ import annotations

import re

from .config import CLAY_PEAK_RANGE
from .models import MODEL_LABEL
from .tools import optimize, validate
from .trace import CHART, TRACE

_NUM = r"(\d+(?:\.\d+)?)"
EXAMPLE = "400 kg stoneware, 60 pieces, 3% moisture, 30 C ambient, defects under 5%"
_NAMES = {
    "load_kg": "the load weight in kg",
    "num_pieces": "the number of pieces",
    "clay_body": f"the clay body ({', '.join(CLAY_PEAK_RANGE)})",
    "moisture_pct": "the clay moisture in percent",
    "ambient_temp": "the ambient temperature in C",
}


def parse_request(text: str, prev: dict | None = None) -> tuple[dict, list[str]]:
    """Pull the job details out of a sentence. Anything not mentioned is taken from `prev` (the previous request),
    so a follow-up like "defects under 3%" changes only that. Returns (inputs, names of anything missing)."""
    t = text.lower()

    def grab(*patterns):
        for p in patterns:
            m = re.search(p, t)
            if m:
                return float(m.group(1))
        return None

    found = {
        "load_kg": grab(_NUM + r"\s*(?:kgs?|kilograms?)\b"),
        "num_pieces": grab(r"(\d+)\s*(?:pieces?|pcs|items?|pots?)\b"),
        "moisture_pct": grab(_NUM + r"\s*%\s*(?:clay\s*)?moisture",
                             r"moisture(?:\s*content)?\s*(?:of|at|is|=|:)?\s*" + _NUM),
        "ambient_temp": grab(_NUM + r"\s*(?:°\s*)?c\b\s*(?:ambient|room)",
                             r"(?:ambient|room)(?:\s*temp(?:erature)?)?\s*(?:of|at|is|=|:)?\s*" + _NUM),
        "max_defect_pct": grab(_NUM + r"\s*%\s*(?:defects?|defect\s*rate)", r"(?:defects?|defect\s*rate)\D{0,25}?" + _NUM + r"\s*%?"),
    }
    found["clay_body"] = next((c for c in CLAY_PEAK_RANGE if c in t), None)
    merged = {k: (v if v is not None else (prev or {}).get(k)) for k, v in found.items()}
    missing = [name for k, name in _NAMES.items() if merged[k] is None]
    if merged["max_defect_pct"] is None:
        merged["max_defect_pct"] = 5.0
    if not missing:
        merged["num_pieces"] = int(merged["num_pieces"])
    return merged, missing


def _reply(i: dict, r: dict, v: dict) -> str:
    s, P, B = r["recommended_schedule"], r["predicted"], r["baseline_prediction"]
    sav = r["energy_saving_vs_baseline_pct"]
    lines = [
        "Scripted demo (no LLM): the same optimize and validate tools the agent uses, run from a fixed script.",
        f"Request read as: {i['load_kg']:g} kg {i['clay_body']}, {i['num_pieces']} pieces, {i['moisture_pct']:g}% moisture, "
        f"{i['ambient_temp']:g} C ambient, defects at most {i['max_defect_pct']:g}%.",
        "",
        f"Schedule: peak {s['peak_temp']} C with a {s['soak_minutes']} min soak; heat to 573 C at {s['ramp_0_573']} C/h, "
        f"then to peak at {s['ramp_573_peak']} C/h; cool at {s['cooling_rate']} C/h.",
        f"Predicted: {P['energy_kwh']} kWh and {P['defect_rate_pct']}% defects, which is {abs(sav)}% "
        f"{'less' if sav >= 0 else 'more'} energy than a standard firing ({B['energy_kwh']} kWh, {B['defect_rate_pct']}% defects).",
    ]
    if r.get("note"):
        lines.append(r["note"])
    t = r["pareto_tradeoffs"][-1]
    lines.append(f"Trade-off: the lowest-defect schedule tested needs {t['energy_kwh']} kWh for {t['defect_rate_pct']}% defects.")
    lines.append("Safety check: approved by the Cedar policy." if v["approved"] else "Safety check: REJECTED. " + "; ".join(v["violations"]))
    lines += [f"Warning: {w}" for w in v["warnings"]]
    lines.append(f"Models: {MODEL_LABEL}, trained on simulated data; the defect model is less accurate than the energy model.")
    lines.append("A human operator must review this schedule before it is loaded into a real kiln controller.")
    return "\n".join(lines)


def run_scripted(text: str, prev: dict | None = None) -> dict:
    inputs, missing = parse_request(text, prev)
    if missing:
        return {"mode": "scripted", "trace": [], "result": None, "inputs": None, "chart": {},
                "reply": f"Scripted demo (no LLM). I still need {', '.join(missing)}. Try: {EXAMPLE}"}
    TRACE.clear()
    res = optimize(**inputs)
    val = validate(res["recommended_schedule"], inputs["max_defect_pct"])
    return {
        "mode": "scripted", "reply": _reply(inputs, res, val), "result": res, "inputs": inputs, "chart": dict(CHART),
        "trace": [{"tool": "optimize", "ok": bool(res["target_met"])}, {"tool": "validate", "ok": bool(val["approved"])}],
    }
