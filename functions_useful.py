from datetime import datetime
from pyomo.environ import value
import pandas as pd
import os
from pathlib import Path

"""
Dans ce script, on a des fonctions qui font ceci :
- calculer l'énergie totale consommée sur l'horizon temporel
- calculer el nombre de jours entre deux dates
- calculer, pour un scénario, l'énergie pv écrêtée sur l'horizon
- calculer l'énergie totale produite par le générateur diesel sur l'horizon
"""
def energie_totale_consomme_rule(b, horizon):
    return (sum(b.charge.p[t] for t in b.time)
            * horizon.time_step.total_seconds() / 3600
            * 20)  # en Wh


def count_days_inclusive(start_str, end_str):
    """
    Renvoie le nombre de jours entre deux dates (inclus),
    sans tenir compte des heures.

    Exemple : du 1 au 3 => 3 jours.
    """
    # Conversion string -> datetime
    start_date = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S").date()
    end_date = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S").date()

    # Nombre de jours en comptant les deux extrémités
    return (end_date - start_date).days + 1


def compute_pv_curtailment_wh(m, s, dt_s):
    """
    Calcule, pour le scénario s, l'énergie PV écrêtée sur tout l'horizon, en Wh.

    - m : modèle Pyomo
    - s : indice de scénario (élément de m.S)
    - dt_s : pas de temps en secondes (par ex. 900 s pour 15 minutes)

    Hypothèse : block_pv a été créé avec curtailable=True, donc m.pv[s].p_curt[t] existe
    et représente la puissance PV non injectée [W] à l'instant t.
    """
    pv_block = m.pv[s]

    if not hasattr(pv_block, "p_curt"):
        raise AttributeError("Le bloc PV du scénario {s} n'a pas d'attribut 'p_curt' (curtailable=False ?)")

    total_Wh = 0.0
    for t in m.time:
        p_curt_t = value(pv_block.p_curt[t])  # W
        # énergie Wh = W * (dt en heures)
        total_Wh += p_curt_t * (dt_s / 3600.0)

    return total_Wh

def compute_diesel_energy_wh(m, s, dt_s: float) -> float:
    """
    Énergie électrique produite par le générateur diesel sur l’horizon (Wh).
    dt_s : pas de temps en secondes.
    """
    if not hasattr(m, "gen"):
        return 0.0
    if (not hasattr(m.gen[s], "p")):
        return 0.0

    # Somme(P[W] * dt[h]) = Wh
    dt_h = dt_s / 3600.0
    return float(sum(value(m.gen[s].p[t]) * dt_h for t in m.time))

def scenario_metrics_df(m, prob, step_s):
    dt_h = step_s / 3600.0
    T = len(list(m.time))

    rows = []
    for s in m.S:
        # Attention: si is_served n'est pas strictement binaire (tolérances MIP),
        # on le "seuille" pour un comptage robuste.
        served_steps = sum(1 if value(m.is_served[s, t]) >= 0.5 else 0 for t in m.time)
        sat_time = served_steps / T  # fraction du temps servi

        unserved_Wh = sum(value(m.p_unserved[s, t]) * dt_h for t in m.time)  # W * h = Wh

        # Curtailment PV: on lit directement la variable du bon scénario
        pv_curt_Wh = 0.0
        if hasattr(m.pv[s], "p_curt"):
            pv_curt_Wh = sum(value(m.pv[s].p_curt[t]) * dt_h for t in m.time)

        rows.append({
            "scenario": int(s),
            "prob": float(prob[s]),
            "sat_time_%": 100.0 * sat_time,
            "unserved_Wh": float(unserved_Wh),
            "pv_curt_Wh": float(pv_curt_Wh),
        })

    df = pd.DataFrame(rows).sort_values("scenario").reset_index(drop=True)

    # valeurs attendues
    sat_expected = (df["prob"] * df["sat_time_%"]).sum()
    unserved_expected = (df["prob"] * df["unserved_Wh"]).sum()
    pvcurt_expected = (df["prob"] * df["pv_curt_Wh"]).sum()

    return df, sat_expected, unserved_expected, pvcurt_expected

def read_load_as_W(csv_path, value_candidates=("aggregate_wh","consumption_wh","value","load_wh")):
    """
    Lit un CSV à pas 15 min, cherche une colonne Wh, la convertit en W (Wh/0.25h = x4).
    Retourne la liste de puissances W.
    """
    df = pd.read_csv(csv_path, parse_dates=[0])
    # deviner la colonne de conso
    valcol = None
    for c in value_candidates:
        if c in df.columns:
            valcol = c
            break
    if valcol is None:
        valcol = df.columns[1]
    return (df[valcol].values * 4.0).tolist()

def time_index_from_horizon(horizon, time_set):
    # Index exactement comme load_data l’attend
    return pd.DatetimeIndex([horizon.map[i] for i in time_set])

def series_from_component(comp, attr_name, time_set, index):
    """Retourne une Series pandas (index=index) depuis comp.<attr_name>[t] si dispo, sinon None."""
    if not hasattr(comp, attr_name):
        return None
    var_or_param = getattr(comp, attr_name)
    try:
        vals = [float(value(var_or_param[t])) for t in time_set]
    except Exception:
        # Param/Var non indexé par t (scalaire)
        return pd.Series([float(value(var_or_param))]*len(index), index=index)
    return pd.Series(vals, index=index)

def _kwh_from_W_series(s, step_seconds):
    """Convertit une série de W en énergie kWh sur l’horizon."""
    return float(s.sum() * (step_seconds/3600.0) / 1000.0)

def _positive_part(s):
    return s.clip(lower=0.0)

def _negative_part_abs(s):
    return (-s.clip(upper=0.0))

def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def multiply_by(csv_path, factor):
    """
    Lit un fichier CSV contenant une colonne 'aggregate_wh',
    multiplie cette colonne par 'factor',
    écrit un nouveau fichier CSV modifié,
    et retourne le chemin de ce nouveau fichier.
    """
    # Lecture du CSV
    df = pd.read_csv(csv_path)

    # Vérification colonne
    if "aggregate_wh" not in df.columns:
        raise ValueError(f"La colonne 'aggregate_wh' est absente dans : {csv_path}")

    # Multiplication
    df["aggregate_wh"] = df["aggregate_wh"] * factor

    # Chemin vers le nouveau fichier
    dir_name, base_name = os.path.split(csv_path)
    name_no_ext, ext = os.path.splitext(base_name)

    new_name = f"{name_no_ext}_x{factor}{ext}"
    new_path = os.path.join(dir_name, new_name)

    # Sauvegarde
    df.to_csv(new_path, index=False)

    return new_path

def save_results_in_tab(with_diesel_generator, rows_all, pv_p_wp_fixed_list):
    results_dir = Path("results") / ("without_diesel" if with_diesel_generator == 0 else "with_diesel")
    results_dir.mkdir(parents=True, exist_ok=True)

    df_results = pd.DataFrame(rows_all)

    # Optionnel : trier pour lecture
    df_results = df_results.sort_values(["p0", "pv_fixed", "scenario"]).reset_index(drop=True)

    # Nom de fichier (tu peux adapter)
    csv_path = results_dir / f"results_pv_{min(pv_p_wp_fixed_list)}_{max(pv_p_wp_fixed_list)}.csv"
    df_results.to_csv(csv_path, index=False)

    print(f"✅ Résultats enregistrés : {csv_path}")