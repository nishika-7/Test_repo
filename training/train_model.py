"""Train the energy and defect models, pick the better family per target on the VALIDATION split,
save only the winner, and report its TEST score once.

Run from the project root:  uv run python training/train_model.py
Outputs go to models/ (the serving code reads models/model_meta.json to know what to load).
"""
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "earth_v2.csv"
OUT = ROOT / "models"
OUT.mkdir(exist_ok=True)
TARGETS = ["energy_kwh", "defect_rate_pct"]
# The MLP is served only if its validation MAE beats boosted trees by this fraction (trees extrapolate safer).
MLP_MARGIN = float(os.getenv("KILN_MLP_MARGIN", 0.05))
keras.utils.set_random_seed(42)

# ---- data + shared preprocessing (identical for both targets) ----
data = pd.read_csv(CSV).drop(columns=["cycle_id", "o2_percent"])
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
enc = pd.DataFrame(ohe.fit_transform(data[["clay_body"]]),
                   columns=ohe.get_feature_names_out(["clay_body"]), index=data.index)
data = pd.concat([data.drop(columns="clay_body"), enc], axis=1)
X = data.drop(columns=TARGETS)
feature_columns = list(X.columns)
yE, yD = data["energy_kwh"], data["defect_rate_pct"]

X_rest, X_test, yE_rest, yE_test, yD_rest, yD_test = train_test_split(X, yE, yD, test_size=0.15, random_state=42)
X_train, X_val, yE_train, yE_val, yD_train, yD_val = train_test_split(
    X_rest, yE_rest, yD_rest, test_size=0.15 / 0.85, random_state=42)

x_scaler = StandardScaler().fit(X_train)            # used by the MLP only; fit on train only
Xtr, Xva, Xte = (x_scaler.transform(a) for a in (X_train, X_val, X_test))

joblib.dump(ohe, OUT / "ohe.joblib")
joblib.dump(feature_columns, OUT / "feature_columns.joblib")   # the optimizer/serving code needs this order
joblib.dump(x_scaler, OUT / "x_scaler.joblib")


def mae_r2(y, p):
    return mean_absolute_error(y, p), r2_score(y, p)


def train_mlp(y_train, y_val):
    ys = StandardScaler().fit(y_train.values.reshape(-1, 1))
    model = keras.Sequential([layers.Input((Xtr.shape[1],)), layers.Dense(64, activation="relu"),
                              layers.Dense(32, activation="relu"), layers.Dense(1)])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    model.fit(Xtr, ys.transform(y_train.values.reshape(-1, 1)),
              validation_data=(Xva, ys.transform(y_val.values.reshape(-1, 1))),
              epochs=200, batch_size=64, callbacks=[stop], verbose=0)
    return model, ys, lambda A: ys.inverse_transform(model.predict(A, verbose=0)).ravel()


splits = {"energy": (yE_train, yE_val, yE_test), "defect": (yD_train, yD_val, yD_test)}
meta = {}
for name, (y_train, y_val, y_test) in splits.items():
    mlp, ys, mlp_pred = train_mlp(y_train, y_val)
    hgb = HistGradientBoostingRegressor(random_state=0).fit(X_train, y_train)
    v_mlp, v_hgb = mae_r2(y_val, mlp_pred(Xva)), mae_r2(y_val, hgb.predict(X_val))
    use_mlp = v_mlp[0] < v_hgb[0] * (1 - MLP_MARGIN)
    print(f"[{name}] validation  MLP MAE={v_mlp[0]:.3f} R2={v_mlp[1]:.3f} | "
          f"boosted trees MAE={v_hgb[0]:.3f} R2={v_hgb[1]:.3f}  -> serving {'MLP' if use_mlp else 'boosted trees'}")
    if use_mlp:
        mlp.save(OUT / f"{name}.keras")
        joblib.dump(ys, OUT / f"scaler_{name}.joblib")
        t = mae_r2(y_test, mlp_pred(Xte))
        entry = {"type": "mlp", "file": f"{name}.keras", "scaler": f"scaler_{name}.joblib"}
    else:
        joblib.dump(hgb, OUT / f"{name}.joblib")
        t = mae_r2(y_test, hgb.predict(X_test))
        entry = {"type": "hgb", "file": f"{name}.joblib", "scaler": None}
    print(f"[{name}] TEST (winner only)  MAE={t[0]:.3f} R2={t[1]:.3f}")
    meta[name] = {**entry, "val_mae": round(float((v_mlp if use_mlp else v_hgb)[0]), 4),
                  "test_mae": round(float(t[0]), 4), "test_r2": round(float(t[1]), 4)}

(OUT / "model_meta.json").write_text(json.dumps(meta, indent=2))
print("saved:", OUT / "model_meta.json")