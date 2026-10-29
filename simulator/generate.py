"""Build a new training CSV with the same columns as earth.csv, but from a known simulator
with low noise.

Run:  python generate.py        (writes earth_v2.csv in the current folder)
"""
import numpy as np
import pandas as pd

from simulator.simulator import CLAYS, SPACE, TARGET_PEAK, clay_index, true_fn

N_ROWS = 30_000
SEED = 42
OUT = "earth_v2.csv"


def main():
    rng = np.random.default_rng(SEED)
    n = N_ROWS
    u = lambda k: rng.uniform(*SPACE[k], n)

    clay = rng.choice(CLAYS, n)
    load = np.round(u("load_kg"), -1)
    # Job conditions and schedule values are sampled INDEPENDENTLY, so the models can learn
    # what a schedule does under each condition.
    df = pd.DataFrame({
        "cycle_id": np.arange(1, n + 1),
        "peak_temp": np.round(TARGET_PEAK[clay_index(clay)] + u("peak_offset")),
        "soak_minutes": np.round(u("soak_minutes")),
        "ramp_0_573": np.round(u("ramp_0_573")),
        "ramp_573_peak": np.round(u("ramp_573_peak")),
        "cooling_rate": np.round(u("cooling_rate")),
        "o2_percent": np.round(u("o2_percent"), 1),
        "load_kg": load,
        "num_pieces": np.maximum(1, np.round(load / u("piece_kg"))).astype(int),
        "clay_body": clay,
        "moisture_pct": np.round(u("moisture_pct"), 1),
        "ambient_temp": np.round(u("ambient_temp")),
    })

    # Targets are computed from the rounded values, so the file is self-consistent.
    energy, defect = true_fn(df, noise=True, rng=rng)
    df["energy_kwh"] = energy.round(1)
    df["defect_rate_pct"] = defect.round(2)

    clean_e, clean_d = true_fn(df, noise=False)
    assert not df.isna().any().any(), "NaNs in dataset"
    df.to_csv(OUT, index=False)

    print(f"wrote {OUT}: {df.shape[0]} rows, columns = {list(df.columns)}")
    print(df[["energy_kwh", "defect_rate_pct"]].describe().round(2))
    print("share of defect rows at the 0.1 floor:", round((df["defect_rate_pct"] <= 0.1).mean(), 3))
    # With noise this small, a good model should get close to this R2 ceiling.
    for name, clean, noisy in [("energy", clean_e, df["energy_kwh"]), ("defect", clean_d, df["defect_rate_pct"])]:
        print(f"best possible R2 for {name} (ceiling set by noise): {1 - np.var(noisy - clean) / np.var(noisy):.3f}")
    print(df.select_dtypes("number").drop(columns="cycle_id").corr()["defect_rate_pct"].drop("defect_rate_pct").round(2))


if __name__ == "__main__":
    main()
