import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import json

from pyomo.environ import *
from tabulate import tabulate
from lms2.tools.post_processing import *
import os
import pickle

def cost_table(m, with_diesel_generator=0):
    cost_data = {
        "coût d'investissement (€)": {
            "solar panel": value(m.capex_pv),
            "batteries": value(m.capex_bat),
            "diesel generator": 0 if with_diesel_generator==0 else value(m.capex_gen),
            "total": value(m.capex_pv) + value(m.capex_bat)
        },
        "coût d'opération (€)": {
            "solar panel": value(m.opex_pv),
            "batteries": value(m.opex_bat),
            "diesel generator": 0 if with_diesel_generator==0 else value(m.opex_gen) + value(m.cost_total_fuel),
            "total": value(m.opex_pv) + value(m.opex_bat)
        },
        "coût de remplacement (€)": {
            "solar panel": 0,
            "batteries": value(m.repl_bat),
            "diesel generator": 0,
            "total": value(m.repl_bat)
        },
        "coût total (€)": {
            "solar panel": value(m.capex_pv) + value(m.opex_pv),
            "batteries": value(m.capex_bat) + value(m.opex_bat) + value(m.repl_bat),
            "diesel generator": 0 if with_diesel_generator==0 else value(m.capex_gen) + value(m.opex_gen) + value(m.cost_total_fuel),
            "total": value(m.total_cost)
        }
    }

    # Conversion en DataFrame
    df_costs = pd.DataFrame.from_dict(cost_data, orient='index')

    # Affichage du tableau avec tabulate
    print(tabulate(df_costs, headers='keys', tablefmt='grid'))
    if with_diesel_generator != 0:
        print("coût du diesel consommé : {}€".format(m.cost_total_fuel()))

def plot_results_deterministic(m, horizon, with_diesel_generator=0, file_name="test"):
    # Visualisation des résultats
    n_points = len(horizon.current)
    index_jours = np.arange(n_points) / (3600 * 24 / horizon.time_step.total_seconds())

    if with_diesel_generator == 0:
        fig, ax = plt.subplots(nrows=4, ncols=1, sharex='all', figsize=(15, 6))
        pplot(m.bat.p, m.pv.p, m.charge.p,
              ax=ax[0],
              fig=fig,
              index=index_jours,
              bbox_to_anchor=(0, -0.12, 1, 0.2),
              ylabel='Power (W)')
        pplot(m.bat.soc, ax=ax[1], fig=fig, index=index_jours,
              bbox_to_anchor=(0, -0.12, 1, 0.2), ylabel='SOC (%)')
        pplot(m.bat.e_loss, ax=ax[2], fig=fig, index=index_jours,
              bbox_to_anchor=(0, -0.12, 1, 0.2), ylabel='energy max battery')
        pplot(m.bat.emax_series, ax=ax[3], fig=fig, index=index_jours,
              bbox_to_anchor=(0, -0.12, 1, 0.2), ylabel='energy max battery')
    else:
        fig, ax = plt.subplots(nrows=4, ncols=1, sharex='all', figsize=(15, 6))
        pplot(m.bat.p, m.pv.p, m.charge.p, m.gen.p,
              ax=ax[0],
              fig=fig,
              index=index_jours,
              bbox_to_anchor=(0, -0.12, 1, 0.2),
              ylabel='Power (W)')
        pplot(m.bat.soc, ax=ax[1], fig=fig, index=index_jours,
              bbox_to_anchor=(0, -0.12, 1, 0.2), ylabel='SOC (%)')
        pplot(m.bat.tmp, ax=ax[2], fig=fig, index=index_jours,
              bbox_to_anchor=(0, -0.12, 1, 0.2), ylabel='Température (°C)')
        pplot(m.bat.emax_series, ax=ax[3], fig=fig, index=index_jours,
              bbox_to_anchor=(0, -0.12, 1, 0.2), ylabel='energy max battery')

    for axis in ax:
        axis.set_xlabel(axis.get_xlabel(), fontsize=17)
        axis.set_ylabel(axis.get_ylabel(), fontsize=17)
        axis.tick_params(axis='both', labelsize=15)

    filename = f"results_image/{file_name}.pickle"

    # --- Sauvegarder la figure ---
    with open(filename, 'wb') as f:
        pickle.dump(fig, f)
    plt.show()

# =========
#  Stochastique : exports / bilans / plots
# =========

def ensure_dir(path: str | Path) -> None:
    """Crée le répertoire s'il n'existe pas."""
    Path(path).mkdir(parents=True, exist_ok=True)


def time_index_from_horizon(horizon, time_set) -> pd.DatetimeIndex:
    """Construit un index temporel pandas aligné sur horizon.map."""
    return pd.DatetimeIndex([horizon.map[i] for i in time_set])


def series_from_component(comp, attr_name: str, time_set, index) -> Optional[pd.Series]:
    """
    Retourne une Series pandas depuis comp.<attr_name>[t] si l'attribut existe.
    - Si comp.n'a pas attr_name -> None
    - Si l'objet est scalaire -> série constante
    """
    if not hasattr(comp, attr_name):
        return None
    var_or_param = getattr(comp, attr_name)
    try:
        vals = [float(value(var_or_param[t])) for t in time_set]
    except Exception:
        # Param/Var non indexé par t (scalaire)
        return pd.Series([float(value(var_or_param))] * len(index), index=index)
    return pd.Series(vals, index=index)


def kwh_from_W_series(s: Optional[pd.Series], step_seconds: float) -> float:
    """Convertit une série de W en énergie kWh sur l’horizon."""
    if s is None:
        return 0.0
    return float(s.sum() * (step_seconds / 3600.0) / 1000.0)


def positive_part(s: Optional[pd.Series]) -> Optional[pd.Series]:
    """Retourne la partie positive de la série (x -> max(x,0))."""
    if s is None:
        return None
    return s.clip(lower=0.0)


def negative_part_abs(s: Optional[pd.Series]) -> Optional[pd.Series]:
    """Retourne la valeur absolue de la partie négative (-min(x,0))."""
    if s is None:
        return None
    return (-s.clip(upper=0.0))


def compute_and_save_cost_breakdown(m, prob: Dict, t0, out_root: str, filename: str = "cost_breakdown.json") -> Dict:
    """
    Recalcule un breakdown CAPEX / (OPEX attendu) sur l'horizon
    (comme dans main) et le sauvegarde en JSON dans out_root.
    """
    PV_YEARS = 9.64955841794
    BAT_YEARS = 9.64955841794
    s1 = list(m.S)[0]

    capex_pv_val = value(m.pv[s1].cost_inv) * value(m.pv[s1].p_wp)
    capex_bat_val = value(m.bat[s1].cost_inv) * value(m.bat[s1].emax[t0])

    opex_exp = 0.0
    for s in m.S:
        opex_pv_s = value(m.pv[s].cost_opex) * value(m.pv[s1].p_wp) * PV_YEARS
        opex_bat_s = value(m.bat[s].cost_opex) * value(m.bat[s1].emax[t0]) * BAT_YEARS
        opex_exp += prob[s] * (opex_pv_s + opex_bat_s)

    cost_breakdown = {
        "capex_pv": capex_pv_val,
        "capex_bat": capex_bat_val,
        "opex_expected": opex_exp,
        "total_objective": value(m.total_cost),
    }

    ensure_dir(out_root)
    json_path = os.path.join(out_root, filename)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cost_breakdown, f, indent=2)
    print("Saved:", json_path)

    return cost_breakdown


def export_scenario_timeseries_and_plots(m, horizon, prob: Dict, out_root: str, with_diesel_generator: int = 0) -> List[Dict]:
    """
    Pour chaque scénario s :
      - construit un DataFrame avec consumption_W, pv_W, bat_W, soc, emax, etc.
      - sauvegarde un CSV timeseries_s{s}.csv
      - génère quelques figures (puissances, SOC, emax)
      - calcule les bilans énergétiques (kWh)
    Retourne une liste de dicts 'rows_summary' (un par scénario).
    """
    ensure_dir(out_root)

    time_idx = time_index_from_horizon(horizon, m.time)
    dt_s = int(horizon.time_step.total_seconds())
    rows_summary: List[Dict] = []

    for s in m.S:
        out_dir_s = os.path.join(out_root, f"scenario_{s}")
        ensure_dir(out_dir_s)

        # Séries principales (W)
        consumption_W = series_from_component(m.consumption[s], "p", m.time, time_idx)
        pv_W = series_from_component(m.pv[s], "p", m.time, time_idx)
        bat_W = series_from_component(m.bat[s], "p", m.time, time_idx)

        # Séries optionnelles
        soc_pct = series_from_component(m.bat[s], "soc", m.time, time_idx)
        emax_Wh = series_from_component(m.bat[s], "emax", m.time, time_idx)

        # PV potentielle pour calculer l'écrêtement
        p_pot = (
            series_from_component(m.pv[s], "p_pot",        m.time, time_idx)
            or series_from_component(m.pv[s], "p_theoretical", m.time, time_idx)
            or series_from_component(m.pv[s], "p_raw",     m.time, time_idx)
            or None
        )
        if p_pot is not None and pv_W is not None:
            curtail_W = (p_pot - pv_W).clip(lower=0.0)
        else:
            curtail_W = None

        # DataFrame export
        df = pd.DataFrame({
            "consumption_W": consumption_W,
            "pv_W":          pv_W,
            "bat_W":         bat_W,
        })
        if soc_pct is not None:
            df["soc_pct"] = soc_pct
        if emax_Wh is not None:
            df["emax_Wh"] = emax_Wh
        if p_pot is not None:
            df["pv_potential_W"] = p_pot
        if curtail_W is not None:
            df["pv_curtail_W"] = curtail_W

        csv_path = os.path.join(out_dir_s, f"timeseries_s{s}.csv")
        df.to_csv(csv_path, index_label="timestamp")
        print("Saved:", csv_path)

        # Bilans énergétiques (kWh) — sur l’horizon
        consumption_kWh = kwh_from_W_series(consumption_W, dt_s)
        pv_kWh          = kwh_from_W_series(pv_W, dt_s)
        bat_dis_kWh     = kwh_from_W_series(positive_part(bat_W), dt_s)
        bat_chg_kWh     = kwh_from_W_series(negative_part_abs(bat_W), dt_s)
        curtail_kWh     = kwh_from_W_series(curtail_W, dt_s)

        rows_summary.append({
            "scenario": s,
            "probability": prob[s],
            "consumption_kWh":    consumption_kWh,
            "pv_to_load_kWh":     pv_kWh,
            "bat_discharge_kWh":  bat_dis_kWh,
            "bat_charge_kWh":     bat_chg_kWh,
            "pv_curtail_kWh":     curtail_kWh,
        })

        # Tracés (3 figures légères)
        # 1) Puissances principales
        plt.figure(figsize=(12, 3))
        if consumption_W is not None:
            plt.plot(consumption_W.index, consumption_W.values, label="consumption (W)")
        if pv_W is not None:
            plt.plot(pv_W.index, pv_W.values, label="PV to load (W)")
        if bat_W is not None:
            plt.plot(bat_W.index, bat_W.values, label="Battery p (W)")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("W")
        plt.title(f"Scenario {s} — Powers")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_s, f"plot_powers_s{s}.png"), dpi=150)
        plt.close()

        # 2) SOC (%), si dispo
        if soc_pct is not None:
            plt.figure(figsize=(12, 2.8))
            plt.plot(soc_pct.index, soc_pct.values)
            plt.xlabel("Time")
            plt.ylabel("%")
            plt.title(f"Scenario {s} — SOC (%)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir_s, f"plot_soc_s{s}.png"), dpi=150)
            plt.close()

        # 3) emax (Wh), si dispo (V3)
        if emax_Wh is not None:
            plt.figure(figsize=(12, 2.8))
            plt.plot(emax_Wh.index, emax_Wh.values)
            plt.xlabel("Time")
            plt.ylabel("Wh")
            plt.title(f"Scenario {s} — Battery emax (Wh)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir_s, f"plot_emax_s{s}.png"), dpi=150)
            plt.close()

    return rows_summary


def summarize_energy_expectation(rows_summary: List[Dict], out_root: str, summary_filename: str = "energy_summary_by_scenario.csv") -> pd.DataFrame:
    """
    À partir de rows_summary (un dict par scénario), construit le DataFrame
    récapitulatif, l'enregistre en CSV, et affiche les consos/PV attendues.
    Retourne le DataFrame.
    """
    df_summary = pd.DataFrame(rows_summary)
    df_summary["expected_kWh_contrib"] = df_summary["probability"] * df_summary["consumption_kWh"]

    csv_path = os.path.join(out_root, summary_filename)
    df_summary.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    expected_consumption_kWh = float((df_summary["probability"] * df_summary["consumption_kWh"]).sum())
    expected_pv_kWh          = float((df_summary["probability"] * df_summary["pv_to_load_kWh"]).sum())

    print(f"Expected consumption over horizon (kWh): {expected_consumption_kWh:.2f}")
    print(f"Expected PV-to-load over horizon (kWh): {expected_pv_kWh:.2f}")

    return df_summary
