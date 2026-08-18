import matplotlib.pyplot as plt
import numpy as np


def plot_curves(
    curve_1,
    curve_2=None,
    label_1="Ist kalibriert (KNN)",
    label_2="Soll",
    title="Fließkurve (Soll-Ist-Vergleich)",
):
    shear_rates = np.array([51, 102, 204, 408, 815, 1630], dtype=float)

    curve_1 = np.clip(np.array(curve_1, dtype=float), 1e-6, None)
    if curve_2 is not None:
        curve_2 = np.clip(np.array(curve_2, dtype=float), 1e-6, None)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=200)

    ax.plot(
        shear_rates,
        curve_1,
        marker="o",
        color="#00354E",
        label=label_1,
    )

    if curve_2 is not None:
        ax.plot(
            shear_rates,
            curve_2,
            marker="s",
            linestyle="--",
            color="#95BB20",
            label=label_2,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Scherrate [1/s]")
    ax.set_ylabel("Viskosität η [Pa·s]")
    ax.set_title(title)

    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    ax.set_xticks(shear_rates)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    ax.legend()

    return fig