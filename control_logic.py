import numpy as np

from src.preprocess import build_process_input
from src.inference import predict_eta_from_nn
from src.prediction import (
    predict_curve_from_recipe,
    DEFAULT_SHEAR_RATES,
)
from src.comparison import compare_curves



# Grundfunktionen


def compute_fixed_target_curve(
    target_recipe,
    shear_rates=None,
    polymer_cols=None,
):
    """
    Erzeugt die feste Sollkurve aus der Soll-Rezeptur.
    """
    if shear_rates is None:
        shear_rates = DEFAULT_SHEAR_RATES

    target_eta = predict_curve_from_recipe(
        recipe=target_recipe,
        shear_rates=shear_rates,
        polymer_cols=polymer_cols,
    )

    return {
        "target_recipe": target_recipe.copy(),
        "target_eta": np.asarray(target_eta, dtype=float),
        "shear_rates": np.asarray(shear_rates, dtype=float),
    }


def evaluate_realtime_state(
    machine_state,
    target_eta,
    shear_rates=None,
    tolerance_rel=0.05,
    model_path=None,
    input_scaler_path=None,
    target_scaler_path=None,
    model_type="ikv",
):
    """
    Berechnet die Ist-Viskositätskurve mit dem ausgewählten KNN
    und vergleicht sie mit der Sollkurve.
    """
    if shear_rates is None:
        shear_rates = DEFAULT_SHEAR_RATES

    process_input = build_process_input(
        drehmoment_mean=machine_state["Drehmoment_mean"],
        druck_mean=machine_state["Druck_mean"],
        t_gemessen=machine_state["T_gemessen"],
        drehzahl_mean=machine_state["Drehzahl_mean"],
        v=machine_state.get("V"),
        t_extruder=machine_state.get("T_Extruder"),
        model_type=model_type,
    )

    ist_eta_raw = predict_eta_from_nn(
        process_input=process_input,
        model_path=model_path,
        input_scaler_path=input_scaler_path,
        target_scaler_path=target_scaler_path,
        model_type=model_type,
    )

    ist_eta = np.asarray(ist_eta_raw, dtype=float)

    compare_result = compare_curves(
        predicted_eta=ist_eta,
        target_eta=target_eta,
        eta_names=[
            f"eta_{int(g)}"
            for g in shear_rates
        ],
        tolerance_rel=tolerance_rel,
    )

    return {
        "ist_eta_raw": ist_eta.copy(),
        "ist_eta": ist_eta,
        "global_bias": 1.0,
        "target_eta": np.asarray(target_eta, dtype=float),
        "compare_result": compare_result,
        "needs_action": compare_result["status"] != "ok",
        "model_type": model_type,
    }



# Bewertungsfunktionen


def _curve_error(
    predicted_eta,
    target_eta,
):
    """
    Zielfunktion:
    70 % mittlere absolute relative Abweichung
    30 % maximale absolute relative Abweichung.

    Dadurch wird nicht nur der Mittelwert verbessert, sondern auch
    vermieden, dass eine einzelne Scherrate sehr weit daneben liegt.
    """
    predicted_eta = np.asarray(
        predicted_eta,
        dtype=float,
    )

    target_eta = np.asarray(
        target_eta,
        dtype=float,
    )

    rel = (
        predicted_eta - target_eta
    ) / (
        target_eta + 1e-12
    )

    mean_abs_rel = float(
        np.mean(np.abs(rel))
    )

    max_abs_rel = float(
        np.max(np.abs(rel))
    )

    score = (
        0.70 * mean_abs_rel
        + 0.30 * max_abs_rel
    )

    return {
        "score": float(score),
        "mape": mean_abs_rel,
        "max_abs_rel": max_abs_rel,
        "rel_vector": rel,
    }


def _recipe_change_size(
    candidate,
    current_recipe,
    allowed_polymers,
    cr5p_allowed,
):
    """
    L1-Abstand zur aktuellen Rezeptur.
    Dient nur als kleine Strafkomponente, damit bei ähnlich guter
    Kurvengüte die kleinere Rezepturänderung bevorzugt wird.
    """
    keys = list(allowed_polymers)

    if cr5p_allowed:
        keys.append("CR5P")

    return float(
        sum(
            abs(
                float(candidate.get(key, 0.0))
                - float(current_recipe.get(key, 0.0))
            )
            for key in keys
        )
    )


# Rezepturbeschränkungen


def _get_recipe_constraints(
    current_recipe,
    target_recipe,
    polymer_cols,
):
    """
    Die SOLL-Rezeptur definiert den veränderbaren Materialraum.

    Soll-Anteil > 0:
        Material darf verändert werden.

    Soll-Anteil == 0:
        Material wird gesperrt und bleibt auf seinem aktuellen Wert.
    """
    allowed_polymers = [
        mat
        for mat in polymer_cols
        if float(target_recipe.get(mat, 0.0)) > 1e-12
    ]

    if not allowed_polymers:
        raise ValueError(
            "In der Soll-Rezeptur ist kein Polymerbestandteil aktiv."
        )

    locked_polymers = [
        mat
        for mat in polymer_cols
        if mat not in allowed_polymers
    ]

    locked_values = {
        mat: float(current_recipe.get(mat, 0.0))
        for mat in locked_polymers
    }

    locked_sum = sum(
        locked_values.values()
    )

    free_polymer_sum = (
        1.0 - locked_sum
    )

    if free_polymer_sum <= 0:
        raise ValueError(
            "Für die in der Soll-Rezeptur freigegebenen Polymere "
            "steht kein Anteil zur Verfügung."
        )

    cr5p_allowed = (
        float(
            target_recipe.get(
                "CR5P",
                0.0,
            )
        ) > 1e-12
    )

    return {
        "allowed_polymers": allowed_polymers,
        "locked_polymers": locked_polymers,
        "locked_values": locked_values,
        "free_polymer_sum": free_polymer_sum,
        "cr5p_allowed": cr5p_allowed,
    }


def _normalize_allowed_polymers(
    recipe,
    allowed_polymers,
    locked_values,
):
    """
    Normiert ausschließlich die freigegebenen Polymere auf den Anteil,
    der nach Berücksichtigung der gesperrten Polymere noch zur Verfügung
    steht. Gesperrte Bestandteile werden exakt beibehalten.
    """
    result = recipe.copy()

    locked_sum = sum(
        float(v)
        for v in locked_values.values()
    )

    free_sum = (
        1.0 - locked_sum
    )

    if free_sum <= 0:
        raise ValueError(
            "Kein freier Polymeranteil für die Rezepturanpassung vorhanden."
        )

    raw_sum = sum(
        max(
            float(result.get(mat, 0.0)),
            0.0,
        )
        for mat in allowed_polymers
    )

    if raw_sum <= 0:
        raise ValueError(
            "Die Summe der freigegebenen Polymeranteile ist 0."
        )

    for mat in allowed_polymers:
        result[mat] = (
            max(
                float(result.get(mat, 0.0)),
                0.0,
            )
            / raw_sum
            * free_sum
        )

    for mat, value in locked_values.items():
        result[mat] = float(value)

    return result


def _within_max_polymer_delta(
    candidate,
    current_recipe,
    allowed_polymers,
    max_polymer_delta,
):
    """
    Verhindert unrealistisch große Einzeländerungen.
    """
    for mat in allowed_polymers:
        delta = abs(
            float(candidate.get(mat, 0.0))
            - float(current_recipe.get(mat, 0.0))
        )

        if delta > max_polymer_delta + 1e-12:
            return False

    return True


# Kandidatenerzeugung


def _generate_polymer_exchange_candidates(
    base_recipe,
    current_recipe,
    allowed_polymers,
    locked_values,
    step,
    max_polymer_delta,
):
    """
    Testet jede mögliche Verschiebung von 'step' zwischen zwei
    FREIGEGEBENEN Polymeren.

    Beispiel für zwei Polymerkomponenten:
        PP579s - step / PP505 + step
        PP505 - step / PP579s + step
    steps können auch noch angepasst werden
    """
    candidates = []

    if len(allowed_polymers) < 2:
        return candidates

    for source in allowed_polymers:
        for target in allowed_polymers:

            if source == target:
                continue

            available = float(
                base_recipe.get(
                    source,
                    0.0,
                )
            )

            actual_step = min(
                float(step),
                available,
            )

            if actual_step <= 1e-12:
                continue

            candidate = base_recipe.copy()

            candidate[source] = (
                available - actual_step
            )

            candidate[target] = (
                float(
                    candidate.get(
                        target,
                        0.0,
                    )
                )
                + actual_step
            )

            candidate = _normalize_allowed_polymers(
                recipe=candidate,
                allowed_polymers=allowed_polymers,
                locked_values=locked_values,
            )

            if not _within_max_polymer_delta(
                candidate=candidate,
                current_recipe=current_recipe,
                allowed_polymers=allowed_polymers,
                max_polymer_delta=max_polymer_delta,
            ):
                continue

            candidates.append(
                candidate
            )

    return candidates


def _generate_cr5p_candidates(
    base_recipe,
    current_recipe,
    cr5p_allowed,
    step,
    cr5p_min,
    cr5p_max,
    max_cr5p_delta,
):
    """
    Testet +/- step für CR5P.
    Nur möglich, wenn CR5P in der Soll-Rezeptur eingesetzt wird.
    steps können auch noch angepasst werden
    """
    if not cr5p_allowed:
        return []

    current_value = float(
        base_recipe.get(
            "CR5P",
            0.0,
        )
    )

    candidates = []

    for direction in (-1.0, 1.0):

        new_value = float(
            np.clip(
                current_value
                + direction * float(step),
                cr5p_min,
                cr5p_max,
            )
        )

        if abs(
            new_value - current_value
        ) <= 1e-12:
            continue

        if abs(
            new_value
            - float(
                current_recipe.get(
                    "CR5P",
                    0.0,
                )
            )
        ) > max_cr5p_delta + 1e-12:
            continue

        candidate = base_recipe.copy()

        candidate["CR5P"] = (
            new_value
        )

        candidates.append(
            candidate
        )

    return candidates


def _deduplicate_candidates(
    candidates,
    polymer_cols,
):
    """
    Entfernt numerisch identische Kandidaten.
    """
    unique = {}

    keys = list(
        polymer_cols
    ) + ["CR5P"]

    for candidate in candidates:

        signature = tuple(
            round(
                float(
                    candidate.get(
                        key,
                        0.0,
                    )
                ),
                8,
            )
            for key in keys
        )

        unique[signature] = candidate

    return list(
        unique.values()
    )


# Kandidatenbewertung

def _evaluate_candidate(
    candidate,
    target_eta,
    process_factor,
    shear_rates,
    polymer_cols,
    current_recipe,
    allowed_polymers,
    cr5p_allowed,
    change_penalty_weight,
):
    """
    Wichtigster Unterschied zur alten Logik:

    Die Rezepturkurve des Kandidaten wird NICHT direkt mit der Sollkurve
    verglichen. Stattdessen wird die aktuell beobachtete Prozessabweichung
    auf den Kandidaten übertragen:

        eta_after = eta_recipe_candidate * process_factor

    Dadurch wird tatsächlich abgeschätzt, welche Rezeptur unter dem
    aktuellen Prozesszustand die Sollkurve am besten kompensiert.
    """
    candidate_eta = np.asarray(
        predict_curve_from_recipe(
            recipe=candidate,
            shear_rates=shear_rates,
            polymer_cols=polymer_cols,
        ),
        dtype=float,
    )

    predicted_eta_after = (
        candidate_eta
        * process_factor
    )

    error = _curve_error(
        predicted_eta=predicted_eta_after,
        target_eta=target_eta,
    )

    change_size = _recipe_change_size(
        candidate=candidate,
        current_recipe=current_recipe,
        allowed_polymers=allowed_polymers,
        cr5p_allowed=cr5p_allowed,
    )

    total_score = (
        error["score"]
        + float(change_penalty_weight)
        * change_size
    )

    return {
        "recipe": candidate,
        "recipe_eta": candidate_eta,
        "predicted_eta_after": predicted_eta_after,
        "curve_error": error,
        "change_size": change_size,
        "total_score": float(total_score),
    }



# Neue Rezepturempfehlung


def recommend_recipe_adjustment(
    current_recipe,
    target_recipe,
    target_eta,
    ist_eta,
    shear_rates=None,
    polymer_cols=None,
    tolerance_rel=0.05,
    allow_cr5p=True,

    # Grob -> fein
    polymer_steps=(0.05, 0.02, 0.01),

    # CR5P ebenfalls grob -> fein
    cr5p_steps=(0.001, 0.0005),

    # Grenzen
    cr5p_min=0.0,
    cr5p_max=0.05,
    max_polymer_delta=0.20,
    max_cr5p_delta=0.005,

    # Kleine Strafkomponente für unnötig große Änderungen
    change_penalty_weight=0.002,

    # maximale Verbesserungsiterationen pro Schrittweite
    max_iterations_per_step=30,

    # Schutz gegen extreme Prozessfaktoren
    process_factor_min=0.25,
    process_factor_max=4.0,
):
    """
    Deterministische, grob-zu-feine lokale Rezeptursuche.

    Suchraum:
    Nur Polymere, die in der SOLL-Rezeptur > 0 sind.

    Bewertung:
    Für jeden Kandidaten wird die unter der aktuell beobachteten
    Prozessabweichung erwartete Kurve berechnet.

    Ziel:
    Minimierung der Abweichung zur Sollkurve bei möglichst kleiner
    Rezepturänderung.
    """

    if shear_rates is None:
        shear_rates = DEFAULT_SHEAR_RATES

    if polymer_cols is None:
        raise ValueError(
            "polymer_cols muss angegeben werden."
        )

    target_eta = np.asarray(
        target_eta,
        dtype=float,
    ).reshape(-1)

    ist_eta = np.asarray(
        ist_eta,
        dtype=float,
    ).reshape(-1)

   
    # Ist-Soll-Bewertung

    process_compare = compare_curves(
        predicted_eta=ist_eta,
        target_eta=target_eta,
        eta_names=[
            f"eta_{int(g)}"
            for g in shear_rates
        ],
        tolerance_rel=tolerance_rel,
    )

    constraints = _get_recipe_constraints(
        current_recipe=current_recipe,
        target_recipe=target_recipe,
        polymer_cols=polymer_cols,
    )

    allowed_polymers = constraints[
        "allowed_polymers"
    ]

    locked_polymers = constraints[
        "locked_polymers"
    ]

    locked_values = constraints[
        "locked_values"
    ]

    cr5p_allowed = (
        constraints["cr5p_allowed"]
        and allow_cr5p
    )

    # Aktuelle Rezeptur sauber normieren

    base_recipe = _normalize_allowed_polymers(
        recipe=current_recipe,
        allowed_polymers=allowed_polymers,
        locked_values=locked_values,
    )

    base_recipe["CR5P"] = float(
        current_recipe.get(
            "CR5P",
            0.0,
        )
    )

   
    current_recipe_eta = np.asarray(
        predict_curve_from_recipe(
            recipe=base_recipe,
            shear_rates=shear_rates,
            polymer_cols=polymer_cols,
        ),
        dtype=float,
    )

    process_factor_raw = (
        ist_eta
        / np.clip(
            current_recipe_eta,
            1e-12,
            None,
        )
    )

    process_factor = np.clip(
        process_factor_raw,
        process_factor_min,
        process_factor_max,
    )

    # Rezepturkurve, die bei diesem Prozessfaktor ideal wäre:
    compensation_target_eta = (
        target_eta
        / np.clip(
            process_factor,
            1e-12,
            None,
        )
    )

    # Falls bereits im Sollbereich: nichts verändern

    if process_compare["status"] == "ok":

        return {
            "message":
                "Istkurve liegt innerhalb der vorgegebenen Toleranz. "
                "Keine Rezepturanpassung erforderlich.",

            "recommended_recipe":
                base_recipe,

            "recommended_eta":
                current_recipe_eta,

            "predicted_eta_after":
                ist_eta.copy(),

            "changes":
                {},

            "allowed_polymers":
                allowed_polymers,

            "locked_polymers":
                locked_polymers,

            "iterations":
                0,

            "process_factor":
                process_factor,

            "process_factor_raw":
                process_factor_raw,

            "compensation_target_eta":
                compensation_target_eta,

            "objective_before":
                _curve_error(
                    predicted_eta=ist_eta,
                    target_eta=target_eta,
                ),
        }

    # Ausgangszustand bewerten

    best = _evaluate_candidate(
        candidate=base_recipe,
        target_eta=target_eta,
        process_factor=process_factor,
        shear_rates=shear_rates,
        polymer_cols=polymer_cols,
        current_recipe=current_recipe,
        allowed_polymers=allowed_polymers,
        cr5p_allowed=cr5p_allowed,
        change_penalty_weight=change_penalty_weight,
    )

    objective_before = best[
        "curve_error"
    ].copy()

    total_iterations = 0

    # GROBE -> FEINE POLYMERSUCHE


    for polymer_step in polymer_steps:

        for _ in range(
            int(max_iterations_per_step)
        ):

            candidates = [
                best["recipe"].copy()
            ]

            candidates.extend(
                _generate_polymer_exchange_candidates(
                    base_recipe=best["recipe"],
                    current_recipe=current_recipe,
                    allowed_polymers=allowed_polymers,
                    locked_values=locked_values,
                    step=float(polymer_step),
                    max_polymer_delta=max_polymer_delta,
                )
            )

            # CR5P gleichzeitig testen:
            for cr5p_step in cr5p_steps:

                candidates.extend(
                    _generate_cr5p_candidates(
                        base_recipe=best["recipe"],
                        current_recipe=current_recipe,
                        cr5p_allowed=cr5p_allowed,
                        step=float(cr5p_step),
                        cr5p_min=cr5p_min,
                        cr5p_max=cr5p_max,
                        max_cr5p_delta=max_cr5p_delta,
                    )
                )

            candidates = _deduplicate_candidates(
                candidates=candidates,
                polymer_cols=polymer_cols,
            )

            evaluated = [
                _evaluate_candidate(
                    candidate=candidate,
                    target_eta=target_eta,
                    process_factor=process_factor,
                    shear_rates=shear_rates,
                    polymer_cols=polymer_cols,
                    current_recipe=current_recipe,
                    allowed_polymers=allowed_polymers,
                    cr5p_allowed=cr5p_allowed,
                    change_penalty_weight=change_penalty_weight,
                )
                for candidate in candidates
            ]

            local_best = min(
                evaluated,
                key=lambda item: item["total_score"],
            )

            # Nur übernehmen, wenn wirklich eine Verbesserung vorliegt.
            if (
                local_best["total_score"]
                >= best["total_score"] - 1e-10
            ):
                break

            best = local_best
            total_iterations += 1


    # Sicherheitsprüfungen
    

    recommended_recipe = (
        best["recipe"].copy()
    )

    # Gesperrte Materialien dürfen sich nicht verändern.
    for mat in locked_polymers:

        original = float(
            current_recipe.get(
                mat,
                0.0,
            )
        )

        recommended = float(
            recommended_recipe.get(
                mat,
                0.0,
            )
        )

        if abs(
            original - recommended
        ) > 1e-10:

            raise RuntimeError(
                "Interner Sicherheitsfehler: "
                f"gesperrtes Material '{mat}' wurde verändert."
            )

    # Wenn CR5P in Soll-Rezeptur nicht vorkommt:
    # exakt auf aktuellem Wert halten.
    if not cr5p_allowed:

        recommended_recipe["CR5P"] = float(
            current_recipe.get(
                "CR5P",
                0.0,
            )
        )

    # Änderungen


    changes = {}

    for key in (
        list(polymer_cols)
        + ["CR5P"]
    ):

        delta = (
            float(
                recommended_recipe.get(
                    key,
                    0.0,
                )
            )
            - float(
                current_recipe.get(
                    key,
                    0.0,
                )
            )
        )

        if abs(delta) > 1e-8:
            changes[key] = delta


    # Ergebnistext

    if changes:

        before_mape = (
            objective_before["mape"]
        )

        after_mape = (
            best[
                "curve_error"
            ]["mape"]
        )

        improvement = (
            (before_mape - after_mape)
            / max(
                before_mape,
                1e-12,
            )
        )

        message = (
            "Deterministische Rezeptursuche: "
            "Innerhalb der in der Soll-Rezeptur verwendeten Bestandteile "
            "wurde eine verbesserte Rezeptur gefunden. "
            f"Modellierte MAPE-Reduktion: {improvement * 100.0:.1f} %."
        )

    else:

        message = (
            "Innerhalb des durch die Soll-Rezeptur vorgegebenen "
            "Materialraums wurde keine verbessernde Rezepturanpassung gefunden."
        )

    return {
        "message":
            message,

        "recommended_recipe":
            recommended_recipe,

        # WICHTIG:
        # reine Rezepturmodellkurve der empfohlenen Rezeptur.
        # Damit bleiben bestehende Validierungsskripte kompatibel.
        "recommended_eta":
            np.asarray(
                best["recipe_eta"],
                dtype=float,
            ),

        # zusätzlich:
        # erwartete Kurve unter der aktuell beobachteten Prozessabweichung.
        "predicted_eta_after":
            np.asarray(
                best["predicted_eta_after"],
                dtype=float,
            ),

        "changes":
            changes,

        "allowed_polymers":
            allowed_polymers,

        "locked_polymers":
            locked_polymers,

        "iterations":
            total_iterations,

        "process_factor":
            process_factor,

        "process_factor_raw":
            process_factor_raw,

        "compensation_target_eta":
            compensation_target_eta,

        "objective_before":
            objective_before,

        "objective_after":
            best["curve_error"],

        "change_size":
            best["change_size"],

        "total_score":
            best["total_score"],
    }


optimize_recipe_to_target = recommend_recipe_adjustment
