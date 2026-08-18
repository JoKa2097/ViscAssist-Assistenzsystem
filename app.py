import pandas as pd
import streamlit as st

from src.config import MODEL_CONFIG
from src.prediction import initialize_recipe_model, DEFAULT_SHEAR_RATES
from src.plotting import plot_curves
from src.control_logic import (
    compute_fixed_target_curve,
    evaluate_realtime_state,
    recommend_recipe_adjustment,
)


def recipe_inputs(prefix, defaults, polymer_cols):
    """
    Zeigt ausschließlich die Materialien an, die zur aktuell
    ausgewählten Datenbasis gehören.
    """
    values = {}

    for mat in polymer_cols:
        values[mat] = st.number_input(
            f"{prefix}: {mat}",
            min_value=0.0,
            max_value=1.0,
            value=float(defaults.get(mat, 0.0)),
            step=0.0001,
            format="%.4f",
            key=f"{prefix}_{mat}",
        )

    values["CR5P"] = st.number_input(
        f"{prefix}: CR5P",
        min_value=0.0,
        max_value=0.05,
        value=float(defaults.get("CR5P", 0.0)),
        step=0.001,
        format="%.4f",
        key=f"{prefix}_CR5P",
    )

    polymer_sum = sum(float(values[mat]) for mat in polymer_cols)

    if abs(polymer_sum - 1.0) <= 1e-6:
        st.success(f"Summe {prefix}-Polymere: {polymer_sum:.4f}")
    else:
        st.warning(
            f"Summe {prefix}-Polymere: {polymer_sum:.4f} / 1,0000"
        )

    return values


def validate_recipe(recipe, polymer_cols, name):
    polymer_sum = sum(float(recipe.get(mat, 0.0)) for mat in polymer_cols)

    if abs(polymer_sum - 1.0) > 1e-6:
        st.error(
            f"{name}: Die Polymeranteile müssen 1,0000 ergeben. "
            f"Aktuell: {polymer_sum:.4f}"
        )
        st.stop()


st.set_page_config(
    page_title="ViscAssist",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 ViscAssist")
st.caption(
    "Sollkurve aus Rezepturmodell, Istkurve aus KNN, "
    "regelbasierte Rezepturempfehlung ohne PSO."
)


with st.sidebar:
    #
    # 0) Modellbasis
    

    st.header("0) Modellbasis")

    model_config = MODEL_CONFIG
    polymer_cols = list(model_config["polymer_cols"])

    st.info(
        "IKV-Datenbasis: " + ", ".join(polymer_cols)
    )

    # Rezepturmodell laden
    try:
        initialize_recipe_model(
            csv_path=model_config["recipe_csv_path"],
            shear_rates=DEFAULT_SHEAR_RATES,
            polymer_cols=polymer_cols,
        )
    except Exception as exc:
        st.error(
            "Rezepturmodell konnte nicht initialisiert werden:\n\n"
            f"{exc}"
        )
        st.stop()

    
    # Standardwerte
    

    defaults = {mat: 0.0 for mat in polymer_cols}

    if "PP 579s" in defaults:
        defaults["PP 579s"] = 0.3000

    if "PP505" in defaults:
        defaults["PP505"] = 0.7000

    # Falls PP579s/PP505 in einer Datenbasis nicht vorhanden sind:
    if abs(sum(defaults.values()) - 1.0) > 1e-6 and polymer_cols:
        defaults = {mat: 0.0 for mat in polymer_cols}
        defaults[polymer_cols[0]] = 1.0

    defaults["CR5P"] = 0.0020

    
    # 1) Soll-Rezeptur
   

    st.header("1) Soll-Rezeptur / Zielkurve")

    target_recipe = recipe_inputs(
        prefix="Soll",
        defaults=defaults,
        polymer_cols=polymer_cols,
    )

    # 2) Aktuelle Rezeptur
    
    st.header("2) Aktuelle Rezeptur")

    current_recipe = recipe_inputs(
        prefix="Aktuell",
        defaults=defaults,
        polymer_cols=polymer_cols,
    )

    # Sichtbar machen, was die Assistenz später verändern darf
    allowed_recipe_components = [
        mat
        for mat in polymer_cols
        if float(target_recipe.get(mat, 0.0)) > 1e-12
    ]

    st.caption(
        "Für die Rezepturempfehlung freigegeben: "
        + (
            ", ".join(allowed_recipe_components)
            if allowed_recipe_components
            else "keine Polymerkomponente"
        )
    )

    if float(target_recipe.get("CR5P", 0.0)) > 1e-12:
        st.caption("CR5P ist ebenfalls für eine Anpassung freigegeben.")
    else:
        st.caption("CR5P ist in der Soll-Rezeptur 0 und bleibt daher gesperrt.")

    
    # 3) Prozessdaten

    st.header("3) Aktuelle Prozessdaten / KNN-Istkurve")

    drehmoment = st.number_input(
        "Drehmoment_mean",
        value=50.0,
    )

    druck = st.number_input(
        "Druck_mean",
        value=30.0,
    )

    t_gemessen = st.number_input(
        "T_gemessen",
        value=225.0,
    )

    drehzahl = st.number_input(
        "Drehzahl_mean",
        value=300.0,
    )

    durchsatz = st.number_input(
        "V",
        value=15.0,
    )

    t_extruder = st.number_input(
        "T_Extruder",
        value=230.0,
    )

    
    # 4) Bewertung
 

    st.header("4) Bewertung")

    tolerance_rel = st.slider(
        "Toleranz Soll-Ist [%]",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
    ) / 100.0



# Rezepturen prüfen


validate_recipe(
    target_recipe,
    polymer_cols,
    "Soll-Rezeptur",
)

validate_recipe(
    current_recipe,
    polymer_cols,
    "Aktuelle Rezeptur",
)



# Prozesszustand

machine_state = {
    "Drehmoment_mean": drehmoment,
    "Druck_mean": druck,
    "T_gemessen": t_gemessen,
    "Drehzahl_mean": drehzahl,
    "V": durchsatz,
    "T_Extruder": t_extruder,
}


# Berechnung


try:

    target_state = compute_fixed_target_curve(
        target_recipe=target_recipe,
        shear_rates=DEFAULT_SHEAR_RATES,
        polymer_cols=polymer_cols,
    )

    realtime_state = evaluate_realtime_state(
        machine_state=machine_state,
        target_eta=target_state["target_eta"],
        shear_rates=DEFAULT_SHEAR_RATES,
        tolerance_rel=tolerance_rel,
        model_path=model_config["model_path"],
        input_scaler_path=model_config["input_scaler_path"],
        target_scaler_path=model_config["target_scaler_path"],
        model_type=model_config["model_type"],
    )

    recommendation = recommend_recipe_adjustment(
        current_recipe=current_recipe,
        target_recipe=target_recipe,
        target_eta=target_state["target_eta"],
        ist_eta=realtime_state["ist_eta"],
        shear_rates=DEFAULT_SHEAR_RATES,
        polymer_cols=polymer_cols,
        tolerance_rel=tolerance_rel,
        allow_cr5p=True,
    )

except Exception as exc:
    st.error(f"Fehler in der Berechnung: {exc}")
    st.stop()



# Ausgabe


col1, col2 = st.columns([2, 1])


with col1:

    st.subheader("Soll-Ist-Vergleich")

    fig = plot_curves(
        realtime_state["ist_eta"],
        target_state["target_eta"],
        label_1="Ist – Softsensor",
        label_2="Soll (Rezepturmodell)",
        title="Fließkurve: Ist vs. Soll",
    )

    st.pyplot(fig)

    st.subheader("Viskositätswerte")

    st.dataframe(
        pd.DataFrame({
            "Scherrate [1/s]":
                DEFAULT_SHEAR_RATES,

            "KNN Ist [Pa·s]":
                realtime_state["ist_eta"],

            "Soll [Pa·s]":
                target_state["target_eta"],

            "Empfohlene Rezeptur [Pa·s]":
                recommendation["recommended_eta"],
        }),
        use_container_width=True,
    )


with col2:

    st.subheader("Bewertung")

    compare_result = realtime_state["compare_result"]

    if compare_result["status"] == "ok":
        st.success("Istkurve liegt im Sollbereich.")

    elif compare_result["status"] == "too_high":
        st.error("Ist-Viskosität zu hoch.")

    elif compare_result["status"] == "too_low":
        st.warning("Ist-Viskosität zu niedrig.")

    else:
        st.info("Bereichsabhängige Abweichung.")

    st.metric(
        "MAPE Ist vs. Ziel",
        f"{compare_result['mape']:.2%}",
    )

    st.metric(
        "Mittlere abs. Abweichung",
        f"{compare_result['mean_abs_rel_error']:.2%}",
    )

    st.subheader("Rezepturempfehlung")

    st.write(recommendation["message"])

    st.caption(
        "Veränderbare Bestandteile: "
        + (
            ", ".join(recommendation["allowed_polymers"])
            if recommendation["allowed_polymers"]
            else "keine"
        )
    )

    st.dataframe(
        pd.DataFrame([
            {
                "Material": key,
                "Soll": float(target_recipe.get(key, 0.0)),
                "Aktuell": float(current_recipe.get(key, 0.0)),
                "Empfohlen": float(
                    recommendation["recommended_recipe"].get(key, 0.0)
                ),
                "Änderung": (
                    float(
                        recommendation["recommended_recipe"].get(key, 0.0)
                    )
                    - float(current_recipe.get(key, 0.0))
                ),
            }
            for key in polymer_cols + ["CR5P"]
        ]),
        use_container_width=True,
    )

    st.info(
        "Materialien mit Soll-Anteil 0 sind für die Rezeptursuche gesperrt."
    )
#python -m streamlit run "C:xxx\ViscAssist_GitHub\app.py"
