"""Model loading and single-schedule prediction."""
from __future__ import annotations

import json
import sys

import joblib
import numpy as np
import pandas as pd

from .config import (CLAY_PEAK_RANGE, DATA_CSV, FEATURE_COLS_PATH, FEATURES, META_PATH, MODEL_DIR, OHE_PATH,
                     X_SCALER_PATH)


def _demo_surrogate(df: pd.DataFrame):
    """Stand-in so the code runs without trained models. NOT your model."""
    energy = (
        15 + 0.22 * df.load_kg * (df.peak_temp / 1200) ** 2 + 0.10 * df.soak_minutes
        + 0.02 * (df.peak_temp - df.ambient_temp) + 4000 / (df.ramp_573_peak + 50)
    )
    defect = (
        2.0 + np.clip(df.ramp_0_573 - 60, 0, None) * 0.05 * (1 + df.moisture_pct / 3)
        + np.clip(df.cooling_rate - 120, 0, None) * 0.03 + np.clip(60 - df.soak_minutes, 0, None) * 0.03
    )
    return np.asarray(energy, dtype=float), np.clip(np.asarray(defect, dtype=float), 0, 50)


def _load_target(meta: dict, name: str):
    """Returns f(X_raw, X_scaled) -> predictions for one target, for whichever model family won in training."""
    m, path = meta[name], MODEL_DIR / meta[name]["file"]
    if m["type"] == "hgb":
        model = joblib.load(path)
        return lambda X, Xs: model.predict(X)
    from tensorflow import keras  # only imported if an MLP was selected

    model, ys = keras.models.load_model(path), joblib.load(MODEL_DIR / m["scaler"])
    return lambda X, Xs: ys.inverse_transform(model.predict(Xs, batch_size=8192, verbose=0)).ravel()


def _load_models():
    """Returns (predict_fn, label). predict_fn(df with FEATURES columns) -> (energy_kwh, defect_pct) arrays.

    Rebuilds the inputs exactly as training did: one-hot clay_body, order columns as feature_columns
    (and scale with x_scaler for MLPs).
    """
    paths = [META_PATH, OHE_PATH, FEATURE_COLS_PATH]
    missing = [p.name for p in paths if not p.exists()]
    if len(missing) == len(paths):
        print("[kiln] WARNING: no trained models found - using DEMO SURROGATE, numbers are not real.", file=sys.stderr)
        return _demo_surrogate, "DEMO SURROGATE (not your trained models)"
    if missing:
        raise FileNotFoundError(f"incomplete model artifacts, missing: {missing}. Re-run training/train_model.py")

    meta = json.loads(META_PATH.read_text())
    ohe, cols = joblib.load(OHE_PATH), joblib.load(FEATURE_COLS_PATH)
    known = set(ohe.categories_[0])
    if known != set(CLAY_PEAK_RANGE):
        raise RuntimeError(f"clay_body values the encoder was trained on {sorted(known)} do not match "
                           f"CLAY_PEAK_RANGE keys {sorted(CLAY_PEAK_RANGE)}; update config.py")
    xs = joblib.load(X_SCALER_PATH) if any(v["type"] == "mlp" for v in meta.values()) else None
    pred_e, pred_d = _load_target(meta, "energy"), _load_target(meta, "defect")

    def predict(df: pd.DataFrame):
        enc = pd.DataFrame(ohe.transform(df[["clay_body"]]), columns=ohe.get_feature_names_out(["clay_body"]), index=df.index)
        X = pd.concat([df.drop(columns="clay_body"), enc], axis=1)
        lost = [c for c in cols if c not in X.columns]
        if lost:
            raise ValueError(f"missing model inputs: {lost}")
        X = X[cols]
        Xs = xs.transform(X) if xs is not None else None
        # physical bounds: an MLP can predict a negative defect rate, which an optimizer would exploit
        return np.clip(pred_e(X, Xs), 0, None).astype(float), np.clip(pred_d(X, Xs), 0, 100).astype(float)

    return predict, f"trained models (energy: {meta['energy']['type']}, defect: {meta['defect']['type']})"


PREDICT_FN, MODEL_LABEL = _load_models()
TRAIN_RANGE = None
if DATA_CSV.exists():
    _d = pd.read_csv(DATA_CSV)
    TRAIN_RANGE = {c: (float(_d[c].min()), float(_d[c].max())) for c in FEATURES if c != "clay_body" and c in _d}


def rnd(x) -> float:
    return round(float(x), 2)


def predict_one(schedule: dict) -> dict:
    df = pd.DataFrame([{k: schedule[k] for k in FEATURES}])
    e, d = PREDICT_FN(df)
    return {"energy_kwh": rnd(e[0]), "defect_rate_pct": rnd(d[0])}