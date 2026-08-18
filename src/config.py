from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

POLYMER_COLS_IKV = [
    "PP 579s", "HP501M", "PP505", "HP 640J",
    "PP-Rezyklat", "PP-C2400", "PP-Replano",
]

MODEL_CONFIG = {
    "model_type": "ikv",
    "model_path": BASE_DIR / "models" / "ikv" / "viscosity_model.pt",
    "input_scaler_path": BASE_DIR / "models" / "ikv" / "input_scaler.pkl",
    "target_scaler_path": BASE_DIR / "models" / "ikv" / "target_scaler.pkl",
    "recipe_csv_path": BASE_DIR / "data" / "ikv_recipe_data.csv",
    "polymer_cols": POLYMER_COLS_IKV,
}
