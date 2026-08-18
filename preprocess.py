import pandas as pd

FEATURE_COLS_IKV = [
    "Drehmoment_mean", "Druck_mean", "T_gemessen",
    "Drehzahl_mean", "V", "T_Extruder",
]


def get_feature_cols(model_type="ikv"):
    if model_type != "ikv":
        raise ValueError("Dieses Repository enthält ausschließlich das IKV-Modell.")
    return FEATURE_COLS_IKV


def build_process_input(
    drehmoment_mean, druck_mean, t_gemessen,
    drehzahl_mean, v=None, t_extruder=None,
    model_type="ikv",
):
    row = {
        "Drehmoment_mean": drehmoment_mean,
        "Druck_mean": druck_mean,
        "T_gemessen": t_gemessen,
        "Drehzahl_mean": drehzahl_mean,
        "V": v,
        "T_Extruder": t_extruder,
    }
    cols = get_feature_cols(model_type)
    return pd.DataFrame([row])[cols]


def transform_inputs(df, scaler=None, model_type="ikv"):
    cols = get_feature_cols(model_type)
    x = df[cols].astype(float).values
    return scaler.transform(x) if scaler is not None else x
