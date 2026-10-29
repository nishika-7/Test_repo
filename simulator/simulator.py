"""Ground-truth kiln simulator for the earth.csv columns.

true_fn(table) returns (energy_kwh, defect_rate_pct) for a table or dict that has
the job columns and the schedule columns below. It works on single values or arrays.

Used twice:
  1. generate.py calls it (with noise) to build the training CSV.
  2. Later, call it with noise=False to grade the optimizer's plans.
"""
import numpy as np

CLAYS = ["earthenware", "stoneware", "porcelain"]
TARGET_PEAK = np.array([1060.0, 1230.0, 1280.0])  # peak temperature that matures each clay
CRACK_SENS = np.array([0.8, 1.0, 1.3])            # porcelain cracks most easily
O2_SENS = np.array([1.2, 1.0, 0.6])               # iron-rich bodies bloat more in low oxygen

JOB_COLS = ["clay_body", "load_kg", "num_pieces", "moisture_pct", "ambient_temp"]
SCHEDULE_COLS = ["peak_temp", "soak_minutes", "ramp_0_573", "ramp_573_peak", "cooling_rate"]

# Sampling ranges. peak_temp is sampled as an offset from each clay's target peak.
SPACE = {
    "load_kg": (200, 1200), "piece_kg": (0.4, 3.0), "moisture_pct": (0, 12), "ambient_temp": (5, 45),
    "peak_offset": (-70, 50), "soak_minutes": (20, 120),
    "ramp_0_573": (40, 200), "ramp_573_peak": (60, 300), "cooling_rate": (40, 250),
}


def _col(t, k):
    return np.asarray(t[k], dtype=float)


def clay_index(clay):
    clay = np.asarray(clay)
    idx = np.select([clay == c for c in CLAYS], [0, 1, 2], default=-1)
    if (idx < 0).any():
        raise ValueError(f"clay_body must be one of {CLAYS}")
    return idx


def true_fn(t, noise=True, rng=None):
    i = clay_index(t["clay_body"])
    Tt, sens, o2s = TARGET_PEAK[i], CRACK_SENS[i], O2_SENS[i]
    peak, soak = _col(t, "peak_temp"), _col(t, "soak_minutes")
    r1, r2, cool, = _col(t, "ramp_0_573"), _col(t, "ramp_573_peak"), _col(t, "cooling_rate"),
    load, pieces, moist, amb = _col(t, "load_kg"), _col(t, "num_pieces"), _col(t, "moisture_pct"), _col(t, "ambient_temp")

    thick = np.sqrt((load / pieces) / 1.2)          # heavier pieces are thicker, so riskier
    size = np.sqrt(load / 600)                      # bigger loads heat and cool less evenly
    maturity = peak + 25 * np.log2(soak / 60)       # soaking longer is worth some extra degrees

    # Energy in kWh.
    t_heat = (573 - amb) / r1 + (peak - 573) / r2   # hours spent heating
    t_cool = (peak - 100) / cool                    # hours spent cooling
    energy = ((0.0011 * load + 0.3) * (peak - amb)  # heating the ware and the kiln
              + 35 * (t_heat + soak / 60)           # heat lost while the kiln is on
              + 3 * t_cool
              + 2.5 * load * moist / 100            # driving off water
              )  

    # Defect rate in percent: each term is a real firing risk.
    defect = (0.6
              + 0.012 * np.maximum(0, Tt - maturity) ** 1.4                 # under-fired
              + 0.015 * np.maximum(0, maturity - (Tt + 20)) ** 1.4          # over-fired, slumping
              + sens * thick * size * 0.0025 * np.maximum(0, r1 - 70) ** 1.6  # cracks at 573 C quartz inversion
              + 0.08 * thick * np.maximum(0, moist - 4) * np.maximum(0, r1 - 80) / 40  # steam bursts
              + 0.00014 * np.maximum(0, r2 - 180) ** 2                      # thermal shock on the climb
              + sens * size * 0.0002 * np.maximum(0, cool - 90) ** 2        # dunting while cooling
            )     

    if noise:
        rng = rng or np.random.default_rng()
        energy = energy * (1 + rng.normal(0, 0.015, np.shape(energy)))
        defect = defect + rng.normal(0, 0.3, np.shape(defect))
    return energy, np.clip(defect, 0.1, 60)


def sample_schedules(clay_body, n, rng):
    """n random schedules inside the ranges used for the training data."""
    u = lambda k: rng.uniform(*SPACE[k], n)
    peak = TARGET_PEAK[clay_index(clay_body)] + u("peak_offset")
    return {"peak_temp": peak, "soak_minutes": u("soak_minutes"), "ramp_0_573": u("ramp_0_573"),
            "ramp_573_peak": u("ramp_573_peak"), "cooling_rate": u("cooling_rate"),}


def best_plan(job, max_defect_pct, energy_weight, n=200_000, seed=0):
    """Near-best plan found by trying n random schedules under the TRUE function (no noise).
    Same scoring as the optimizer, so it is the fair reference to compare against."""
    rng = np.random.default_rng(seed)
    sched = sample_schedules(np.full(n, job["clay_body"]), n, rng)
    table = {**{k: np.full(n, job[k]) for k in JOB_COLS}, **sched}
    e, d = true_fn(table, noise=False)
    ok = d <= max_defect_pct
    pool = np.where(ok)[0] if ok.any() else np.arange(n)
    norm = lambda v: (v - v[pool].min()) / (np.ptp(v[pool]) or 1.0)
    score = energy_weight * norm(e) + (1 - energy_weight) * norm(d)
    b = pool[np.argmin(score[pool])]
    return {k: round(float(v[b]), 1) for k, v in sched.items()}, float(e[b]), float(d[b])


def current_schedule(clay_body):
    """A typical fixed 'house' schedule, the baseline to beat."""
    return {"peak_temp": float(TARGET_PEAK[clay_index(clay_body)]), "soak_minutes": 60.0, "ramp_0_573": 120.0,
            "ramp_573_peak": 150.0, "cooling_rate": 120.0,}


def _sanity_checks():
    job = dict(clay_body="stoneware", load_kg=600, num_pieces=300, moisture_pct=6, ambient_temp=25)
    cur = current_schedule("stoneware")
    run = lambda **o: true_fn({**job, **cur, **o}, noise=False)
    e0, d0 = run()
    assert run(ramp_0_573=70)[1] < d0, "slower warm-up through 573 C should cut defects"
    assert run(cooling_rate=70)[1] < d0, "slower cooling should cut defects"
    assert run(soak_minutes=20)[1] > d0, "short soak should raise defects (under-fired)"
    assert run(cooling_rate=70)[0] > e0, "slow cooling should cost a little more energy"
    assert true_fn({**job, **cur, "moisture_pct": 11}, noise=False)[0] > e0, "wetter load should cost more energy"
    plan, e1, d1 = best_plan(job, max_defect_pct=1.5, energy_weight=0.5)
    print(f"house schedule: {e0:.0f} kWh, {d0:.2f}% defects")
    print(f"best found:     {e1:.0f} kWh, {d1:.2f}% defects  {plan}")
    print("all sanity checks passed")


if __name__ == "__main__":
    _sanity_checks()
