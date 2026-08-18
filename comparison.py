import numpy as np
import pandas as pd

def compare_curves(predicted_eta, target_eta, eta_names=None, tolerance_rel=0.10):
    predicted_eta = np.asarray(predicted_eta, dtype=float).reshape(-1)
    target_eta = np.asarray(target_eta, dtype=float).reshape(-1)

    if predicted_eta.shape != target_eta.shape:
        raise ValueError("Ist- und Sollkurve müssen gleich viele Punkte enthalten.")

    if eta_names is None:
        eta_names = [f"eta_{i+1}" for i in range(len(predicted_eta))]

    abs_error = predicted_eta - target_eta
    rel_error = abs_error / (target_eta + 1e-12)

    mape = float(np.mean(np.abs(rel_error)))
    mean_abs_rel_error = float(np.mean(np.abs(rel_error)))
    mean_rel_error = float(np.mean(rel_error))
    mae = float(np.mean(np.abs(abs_error)))

    if np.all(np.abs(rel_error) <= tolerance_rel):
        status = "ok"
    elif mean_rel_error > tolerance_rel:
        status = "too_high"
    elif mean_rel_error < -tolerance_rel:
        status = "too_low"
    else:
        status = "mixed"

    details = pd.DataFrame({
        "eta": eta_names,
        "predicted": predicted_eta,
        "target": target_eta,
        "abs_error": abs_error,
        "rel_error_percent": rel_error * 100.0,
        "abs_rel_error_percent": np.abs(rel_error) * 100.0,
    })

    return {
        "status": status,
        "mae": mae,
        "mape": mape,
        "mean_abs_rel_error": mean_abs_rel_error,
        "mean_rel_error": mean_rel_error,
        "upper_abs_rel_error": float(np.mean(np.abs(rel_error[:3]))),
        "lower_abs_rel_error": float(np.mean(np.abs(rel_error[3:]))),
        "upper_rel_error": float(np.mean(rel_error[:3])),
        "lower_rel_error": float(np.mean(rel_error[3:])),
        "details": details,
    }
