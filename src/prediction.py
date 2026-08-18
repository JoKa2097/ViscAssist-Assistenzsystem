import numpy as np
import pandas as pd
from pathlib import Path

POLYMER_COLS = [
    "PP 579s",
    "HP501M",
    "PP505",
    "HP 640J",
    "PP-Rezyklat",
    "PP-C2400",
    "PP-Replano",
]
DEFAULT_SHEAR_RATES = np.array([51, 102, 204, 408, 815, 1630], dtype=float)
ETA_COLS = [f"eta_{i}" for i in range(1, 7)]

MATERIAL_CURVES = None
CR5P_MATERIAL_PARAMS = None
def _read_table(path):
    path = Path(path)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path, sep=";")


def _find_cr5p_column(df):
    for col in df.columns:
        if str(col).strip().lower() == "cr5p":
            return col
    raise ValueError("Keine CR5P-Spalte gefunden.")


def initialize_recipe_model(
    csv_path=None,
    material_curve_path=None,
    fit_data_path=None,
    polymer_cols=None,
    shear_rates=None,
):
    global MATERIAL_CURVES, CR5P_MATERIAL_PARAMS

    if polymer_cols is None:
        polymer_cols = POLYMER_COLS
    if shear_rates is None:
        shear_rates = DEFAULT_SHEAR_RATES

    data_path = csv_path or material_curve_path or fit_data_path
    if data_path is None:
        raise ValueError("Bitte csv_path angeben.")

    df = _read_table(data_path)

    MATERIAL_CURVES = fit_material_curves_from_recipe_data(
        df=df,
        polymer_cols=polymer_cols,
    )

    CR5P_MATERIAL_PARAMS = fit_cr5p_material_params(
        df=df,
        material_curves=MATERIAL_CURVES,
        polymer_cols=polymer_cols,
        shear_rates=shear_rates,
    )

    return {
        "material_curves": MATERIAL_CURVES,
        "cr5p_material_params": CR5P_MATERIAL_PARAMS,
        "polymer_cols": polymer_cols,
        "shear_rates": shear_rates,
    }


def fit_material_curves_from_recipe_data(df, polymer_cols=None):
    """
    Bestimmt feste Materialkurven direkt aus reinen Materialzeilen.

    Beispiel:
    PP505 = 1, alle anderen Polymere = 0, CR5P = 0
    -> eta_1 ... eta_6 = feste Materialkurve für PP505.

    Zwischenrezepturen werden später über die Mischungsregel berechnet:
    eta_blend = Summe x_i * eta_i
    """
    if polymer_cols is None:
        polymer_cols = POLYMER_COLS

    cr5p_col = _find_cr5p_column(df)

    needed = list(polymer_cols) + ETA_COLS + [cr5p_col]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Diese Spalten fehlen in der CSV: {missing}")

    df = df.copy()

    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=needed).copy()

    material_curves = {}

    for mat in polymer_cols:
        other_mats = [m for m in polymer_cols if m != mat]

        pure_rows = df[
            (np.isclose(df[mat], 1.0))
            & (np.isclose(df[cr5p_col], 0.0))
        ].copy()

        for other in other_mats:
            pure_rows = pure_rows[np.isclose(pure_rows[other], 0.0)]

        if len(pure_rows) == 0:
            raise ValueError(
                f"Keine reine Materialzeile für '{mat}' gefunden. "
                f"Erwartet: {mat}=1, alle anderen Polymere=0 und CR5P=0."
            )

        curve = pure_rows[ETA_COLS].mean(axis=0).to_numpy(dtype=float)

        if np.any(curve <= 0):
            raise ValueError(f"Materialkurve für '{mat}' enthält Werte <= 0.")

        material_curves[mat] = curve

    return material_curves


def fit_cr5p_material_params(
    df,
    material_curves,
    polymer_cols=None,
    shear_rates=None,
    ridge_alpha=1e-6,
):
    if polymer_cols is None:
        polymer_cols = POLYMER_COLS
    if shear_rates is None:
        shear_rates = DEFAULT_SHEAR_RATES

    cr5p_col = _find_cr5p_column(df)

    needed = list(polymer_cols) + ETA_COLS + [cr5p_col]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Diese Spalten fehlen in der CSV: {missing}")

    df = df.copy()

    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=needed).copy()
    df_cr = df[df[cr5p_col] > 0].copy()

    if len(df_cr) == 0:
        return {mat: {"p1": 0.0, "p2": 0.0} for mat in polymer_cols}

    A_rows = []
    b_rows = []
    ln_gamma = np.log(np.array(shear_rates, dtype=float))

    for _, row in df_cr.iterrows():
        x_cr5p = float(row[cr5p_col])
        if x_cr5p <= 0:
            continue

        polymer_sum = sum(float(row.get(mat, 0.0)) for mat in polymer_cols)
        if polymer_sum <= 0:
            continue

        x_poly = {
            mat: float(row.get(mat, 0.0)) / polymer_sum
            for mat in polymer_cols
        }

        eta_blend = np.zeros(len(shear_rates), dtype=float)
        for mat in polymer_cols:
            eta_blend += x_poly[mat] * np.array(material_curves[mat], dtype=float)

        eta_meas = np.array([float(row[col]) for col in ETA_COLS], dtype=float)

        for j in range(len(shear_rates)):
            if eta_meas[j] <= 0 or eta_blend[j] <= 0:
                continue

            y = np.log(eta_meas[j] / eta_blend[j]) / x_cr5p

            features = []

            for mat in polymer_cols:
                features.append(x_poly[mat] * ln_gamma[j])

            for mat in polymer_cols:
                features.append(x_poly[mat])

            A_rows.append(features)
            b_rows.append(y)

    if len(A_rows) < 2 * len(polymer_cols):
        raise ValueError(
            "Zu wenige gültige Datenpunkte für den materialabhängigen CR5P-Fit."
        )

    A = np.array(A_rows, dtype=float)
    b = np.array(b_rows, dtype=float)

    ATA = A.T @ A + ridge_alpha * np.eye(A.shape[1])
    ATb = A.T @ b
    coef = np.linalg.solve(ATA, ATb)

    n = len(polymer_cols)
    p1_values = coef[:n]
    p2_values = coef[n:]

    return {
        mat: {
            "p1": float(p1_values[i]),
            "p2": float(p2_values[i]),
        }
        for i, mat in enumerate(polymer_cols)
    }


def normalize_polymer_fractions(recipe, polymer_cols=None):
    if polymer_cols is None:
        polymer_cols = POLYMER_COLS

    recipe_norm = recipe.copy()

    polymer_sum = sum(float(recipe.get(mat, 0.0)) for mat in polymer_cols)

    if polymer_sum <= 0:
        raise ValueError("Die Summe der Polymeranteile muss größer 0 sein.")

    for mat in polymer_cols:
        recipe_norm[mat] = float(recipe.get(mat, 0.0)) / polymer_sum

    recipe_norm["CR5P"] = float(recipe.get("CR5P", 0.0))

    return recipe_norm


def calc_blend_viscosity(recipe, material_curves=None, polymer_cols=None):
    if polymer_cols is None:
        polymer_cols = POLYMER_COLS

    if material_curves is None:
        material_curves = MATERIAL_CURVES

    if material_curves is None:
        raise ValueError(
            "Materialkurven nicht initialisiert. Bitte initialize_recipe_model() aufrufen."
        )

    recipe_norm = normalize_polymer_fractions(recipe, polymer_cols)

    eta_blend = np.zeros(len(DEFAULT_SHEAR_RATES), dtype=float)

    for mat in polymer_cols:
        x_i = float(recipe_norm.get(mat, 0.0))
        eta_i = np.array(material_curves[mat], dtype=float)
        eta_blend += x_i * eta_i

    return eta_blend


def effective_cr5p_params(recipe, cr5p_material_params=None, polymer_cols=None):
    if polymer_cols is None:
        polymer_cols = POLYMER_COLS

    if cr5p_material_params is None:
        cr5p_material_params = CR5P_MATERIAL_PARAMS

    if cr5p_material_params is None:
        cr5p_material_params = {
            mat: {"p1": 0.0, "p2": 0.0}
            for mat in polymer_cols
        }

    recipe_norm = normalize_polymer_fractions(recipe, polymer_cols)

    p1_eff = 0.0
    p2_eff = 0.0

    for mat in polymer_cols:
        x_i = float(recipe_norm.get(mat, 0.0))
        params = cr5p_material_params.get(mat, {"p1": 0.0, "p2": 0.0})

        p1_eff += x_i * float(params["p1"])
        p2_eff += x_i * float(params["p2"])

    return p1_eff, p2_eff


def apply_cr5p_to_curve(
    eta_blend,
    shear_rates,
    x_cr5p,
    p1_eff,
    p2_eff,
):
    eta_blend = np.array(eta_blend, dtype=float)
    shear_rates = np.array(shear_rates, dtype=float)

    modifier = np.exp(
        float(x_cr5p) * (
            float(p1_eff) * np.log(shear_rates)
            + float(p2_eff)
        )
    )

    return eta_blend * modifier


def predict_curve_from_recipe(
    recipe,
    shear_rates=None,
    polymer_cols=None,
):
    if shear_rates is None:
        shear_rates = DEFAULT_SHEAR_RATES
    if polymer_cols is None:
        polymer_cols = POLYMER_COLS

    eta_blend = calc_blend_viscosity(
        recipe=recipe,
        material_curves=MATERIAL_CURVES,
        polymer_cols=polymer_cols,
    )

    x_cr5p = float(recipe.get("CR5P", 0.0))

    p1_eff, p2_eff = effective_cr5p_params(
        recipe=recipe,
        cr5p_material_params=CR5P_MATERIAL_PARAMS,
        polymer_cols=polymer_cols,
    )

    return apply_cr5p_to_curve(
        eta_blend=eta_blend,
        shear_rates=shear_rates,
        x_cr5p=x_cr5p,
        p1_eff=p1_eff,
        p2_eff=p2_eff,
    )


def print_material_curves():
    if MATERIAL_CURVES is None:
        print("Keine Materialkurven vorhanden.")
        return

    print("\nMaterialkurven:")
    for mat, curve in MATERIAL_CURVES.items():
        print(mat, curve)


def print_cr5p_params():
    if CR5P_MATERIAL_PARAMS is None:
        print("Keine CR5P-Parameter vorhanden.")
        return

    print("\nCR5P-Parameter:")
    for mat, params in CR5P_MATERIAL_PARAMS.items():
        print(f"{mat}: p1 = {params['p1']:.6f}, p2 = {params['p2']:.6f}")