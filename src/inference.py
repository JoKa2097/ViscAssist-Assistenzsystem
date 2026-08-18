from pathlib import Path
import joblib
import numpy as np
import torch
import torch.nn as nn

from src.preprocess import transform_inputs

DROPOUT = 0.3231786904737973


class IKVNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 62),
            nn.LeakyReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(62, 122),
            nn.LeakyReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(122, 32),
            nn.LeakyReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 6),
        )

    def forward(self, x):
        return self.net(x)


def _extract_state_dict(obj):
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    return obj


def load_model(model_path):
    if model_path is None:
        raise ValueError("model_path ist None.")

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Modelldatei nicht gefunden: {model_path}")

    model = IKVNet()
    state_dict = _extract_state_dict(torch.load(model_path, map_location="cpu"))

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint passt nicht zur erwarteten Modellarchitektur. "
            f"Bitte zuerst diagnose_setup.py ausführen.\n{exc}"
        ) from exc

    model.eval()
    return model


def load_scalers(input_scaler_path, target_scaler_path):
    if input_scaler_path is None or target_scaler_path is None:
        raise ValueError("Scaler-Pfad ist None.")

    input_scaler_path = Path(input_scaler_path)
    target_scaler_path = Path(target_scaler_path)

    if not input_scaler_path.exists():
        raise FileNotFoundError(f"Input-Scaler nicht gefunden: {input_scaler_path}")
    if not target_scaler_path.exists():
        raise FileNotFoundError(f"Target-Scaler nicht gefunden: {target_scaler_path}")

    return joblib.load(input_scaler_path), joblib.load(target_scaler_path)


@torch.no_grad()
def predict_eta_from_nn(
    process_input,
    model_path,
    input_scaler_path,
    target_scaler_path,
    model_type="ikv",
):
    if model_type != "ikv":
        raise ValueError("Dieses Repository enthält ausschließlich das IKV-Modell.")

    model = load_model(model_path)
    input_scaler, target_scaler = load_scalers(
        input_scaler_path, target_scaler_path
    )

    x = transform_inputs(process_input, scaler=input_scaler)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_scaled = model(x_tensor).cpu().numpy()
    y = target_scaler.inverse_transform(y_scaled)
    return np.asarray(y[0], dtype=float)
