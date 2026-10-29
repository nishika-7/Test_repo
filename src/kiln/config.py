"""Feature schema, safety limits and baseline. Edit the placeholder limits before demoing."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # project root (the folder with pyproject.toml)
try:  # read settings (KILN_LLM, GROQ_API_KEY, ...) from a .env file in the project root; real environment variables win
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass
MODEL_DIR = Path(os.getenv("KILN_MODEL_DIR", ROOT / "models"))
DATA_CSV = Path(os.getenv("KILN_DATA", ROOT / "data" / "earth_v2.csv"))  # used only to warn on extrapolation

# Artifacts written by training/train_model.py; model_meta.json says which model file serves each target
META_PATH = MODEL_DIR / "model_meta.json"
OHE_PATH, X_SCALER_PATH, FEATURE_COLS_PATH = MODEL_DIR / "ohe.joblib", MODEL_DIR / "x_scaler.joblib", MODEL_DIR / "feature_columns.joblib"

# Column order from your dataset schema (cycle_id and the two targets excluded).
FEATURES = [
    "peak_temp", "soak_minutes", "ramp_0_573", "ramp_573_peak", "cooling_rate",
    "load_kg", "num_pieces", "clay_body", "moisture_pct", "ambient_temp",
]
CONTROLS = ["peak_temp", "soak_minutes", "ramp_0_573", "ramp_573_peak", "cooling_rate"]
CONDITIONS = ["load_kg", "num_pieces", "clay_body", "moisture_pct", "ambient_temp"]

# SAFETY LIMITS. These are PLACEHOLDERS (assumed units: degC and degC/hour).
# Keep policies/safety.cedar in sync: tests/test_safety.py fails if they disagree.
# Replace with values from your simulator / a ceramics reference before demoing.
CLAY_PEAK_RANGE = {              # allowed peak temperature per clay body (degC)
    "earthenware": (950, 1120),
    "stoneware": (1180, 1300),
    "porcelain": (1250, 1400),
}
GLOBAL_LIMITS = {
    "soak_minutes": (10, 240),
    "ramp_0_573": (20, 150),      # slow through quartz inversion at 573 degC to avoid cracking
    "ramp_573_peak": (30, 300),
    "cooling_rate": (20, 250),    # fast cooling risks dunting
}
WET_MOISTURE_PCT = 2.0            # above this, halve the max initial ramp (water smoking)
KILN_CAPACITY_KG = 1200

# A "typical" firing used only to report savings. Replace with your plant's real practice.
BASELINE_CONTROLS = {
    "soak_minutes": 60, "ramp_0_573": 60, "ramp_573_peak": 120,
    "cooling_rate": 80,
}