import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import json

from pyomo.environ import *
from tabulate import tabulate
from lms2.tools.post_processing import *
from mpl_toolkits.mplot3d import Axes3D
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
            "diesel generator": 0 if with_diesel_generator==0 else value(m.opex_gen) + value(m.expected_fuel_cost),
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
            "diesel generator": 0 if with_diesel_generator==0 else value(m.capex_gen) + value(m.opex_gen) + value(m.expected_fuel_cost),
            "total": value(m.total_cost)
        }
    }

    # Conversion en DataFrame
    df_costs = pd.DataFrame.from_dict(cost_data, orient='index')

    # Affichage du tableau avec tabulate
    print(tabulate(df_costs, headers='keys', tablefmt='grid'))
    if with_diesel_generator != 0:
        print("coût du diesel consommé : {}€".format(m.expected_fuel_cost()))

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


def export_scenario_timeseries_and_plots(m, horizon, prob: Dict, results_root: str, pv_installed_W: int, p0_diesel_W: int, with_diesel_generator: int = 0) -> List[Dict]:
    """
    Pour chaque scénario s :
      - construit un DataFrame avec consumption_W, pv_W, bat_W, soc, emax, etc.
      - sauvegarde un CSV timeseries_s{s}.csv
      - génère quelques figures (puissances, SOC, emax)
      - calcule les bilans énergétiques (kWh)
    Retourne une liste de dicts 'rows_summary' (un par scénario).
    """
    base = Path(results_root) / ("without_diesel" if with_diesel_generator == 0 else "with_diesel")

    if with_diesel_generator == 0:
        # results/without_diesel/pv_7000W/
        out_root = base / f"pv_{pv_installed_W}W"
    else:
        # results/with_diesel/p0_2000W/pv_7000W/
        out_root = base / f"p0_{p0_diesel_W}W" / f"pv_{pv_installed_W}W"

    out_root = str(out_root)
    ensure_dir(out_root)

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
        gen_W = None
        if with_diesel_generator != 0 and hasattr(m, "gen"):
            gen_W = series_from_component(m.gen[s], "p", m.time, time_idx)

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
        if gen_W is not None:
            df["gen_W"] = gen_W
        if soc_pct is not None:
            df["soc_pct"] = soc_pct
        if emax_Wh is not None:
            df["emax_Wh"] = emax_Wh
        if p_pot is not None:
            df["pv_potential_W"] = p_pot
        if curtail_W is not None:
            df["pv_curtail_W"] = curtail_W

        csv_path = os.path.join(out_dir_s, f"timeseries.csv")
        df.to_csv(csv_path, index_label="timestamp")
        print("Saved:", csv_path)

        # Bilans énergétiques (kWh) — sur l’horizon
        consumption_kWh = kwh_from_W_series(consumption_W, dt_s)
        pv_kWh          = kwh_from_W_series(pv_W, dt_s)
        bat_dis_kWh     = kwh_from_W_series(positive_part(bat_W), dt_s)
        bat_chg_kWh     = kwh_from_W_series(negative_part_abs(bat_W), dt_s)
        curtail_kWh     = kwh_from_W_series(curtail_W, dt_s)
        gen_kWh = kwh_from_W_series(gen_W, dt_s) if gen_W is not None else 0.0

        rows_summary.append({
            "scenario": s,
            "probability": prob[s],
            "consumption_kWh":    consumption_kWh,
            "pv_to_load_kWh":     pv_kWh,
            "bat_discharge_kWh":  bat_dis_kWh,
            "bat_charge_kWh":     bat_chg_kWh,
            "pv_curtail_kWh":     curtail_kWh,
            "gen_kWh": gen_kWh,
        })

        # Tracés (3 figures légères)
        # 1) Puissances principales
        fig, ax = plt.subplots(figsize=(12, 3))
        if consumption_W is not None:
            ax.plot(consumption_W.index, consumption_W.values, label="consumption (W)")
        if pv_W is not None:
            ax.plot(pv_W.index, pv_W.values, label="PV to load (W)")
        if bat_W is not None:
            ax.plot(bat_W.index, bat_W.values, label="Battery p (W)")
        if gen_W is not None:
            ax.plot(gen_W.index, gen_W.values, label="Diesel gen (W)")
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel("W")
        ax.grid(True)
        ax.set_title(f"Scenario {s} — Powers")
        fig.tight_layout()
        fig_path = os.path.join(out_dir_s, f"plot_powers.pickle")
        with open(fig_path, "wb") as f:
            pickle.dump(fig, f)
        plt.close(fig)

        # 2) SOC (%), si dispo
        if soc_pct is not None:
            fig, ax = plt.subplots(figsize=(12, 2.8))
            ax.plot(soc_pct.index, soc_pct.values)
            ax.set_xlabel("Time")
            ax.set_ylabel("%")
            ax.grid(True)
            ax.set_title(f"Scenario {s} — SOC (%)")
            fig.tight_layout()
            fig_path = os.path.join(out_dir_s, f"plot_soc.pickle")
            with open(fig_path, "wb") as f:
                pickle.dump(fig, f)
            plt.close(fig)

        # 3) emax (Wh), si dispo (V3)
        if emax_Wh is not None:
            fig, ax = plt.subplots(figsize=(12, 2.8))
            ax.plot(emax_Wh.index, emax_Wh.values)
            ax.set_xlabel("Time")
            ax.set_ylabel("Wh")
            ax.grid(True)
            ax.set_title(f"Scenario {s} — Battery emax (Wh)")
            fig.tight_layout()
            fig_path = os.path.join(out_dir_s, f"plot_emax.pickle")
            with open(fig_path, "wb") as f:
                pickle.dump(fig, f)
            plt.close(fig)

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

def visualize(path_name):
    with open(path_name, "rb") as f:
        fig = pickle.load(f)
    fig.show()

def view_sizing_evolution_wih_diesel(p0_list, cost_list, pv_list, bat_list, out_dir):
    if cost_list is not None:
        fig, ax = plt.subplots(figsize=(12, 2.8))
        ax.plot(p0_list, cost_list, "o-")
        ax.set_xlabel("Diesel max output power (W)")
        ax.set_ylabel("Cost (€)")
        ax.grid(True)
        ax.set_title(f"Scenario — Total cost (€)")
        fig.tight_layout()
        fig_path = os.path.join(out_dir, f"plot_total_cost.pickle")
        with open(fig_path, "wb") as f:
            pickle.dump(fig, f)
        plt.close(fig)

    if pv_list is not None:
        fig, ax = plt.subplots(figsize=(12, 2.8))
        ax.plot(p0_list, pv_list, "o-")
        ax.set_xlabel("Diesel max output power (W)")
        ax.set_ylabel("Power (W)")
        ax.grid(True)
        ax.set_title(f"Scenario — PV installed power (W)")
        fig.tight_layout()
        fig_path = os.path.join(out_dir, f"plot_pv_power.pickle")
        with open(fig_path, "wb") as f:
            pickle.dump(fig, f)
        plt.close(fig)

    if bat_list is not None:
        fig, ax = plt.subplots(figsize=(12, 2.8))
        ax.plot(p0_list, bat_list, "o-")
        ax.set_xlabel("Diesel max output power (W)")
        ax.set_ylabel("Energy (Wh)")
        ax.grid(True)
        ax.set_title(f"Scenario — Battery energy (Wh)")
        fig.tight_layout()
        fig_path = os.path.join(out_dir, f"plot_bat_energy.pickle")
        with open(fig_path, "wb") as f:
            pickle.dump(fig, f)
        plt.close(fig)


def verify_pv_curt(m, horizon, s, title_prefix=None):
    """
    Affiche 2 graphiques pour vérifier la logique d'écrêtage PV :
      1) pv_curt (W) + soc_bat (%)
      2) is_full (0/1) + soc_bat (%)

    Paramètres
    ----------
    m : Pyomo ConcreteModel résolu
    horizon : SimpleHorizon
    s : scénario (élément de m.S)
    title_prefix : str optionnel
    """

    # --- index temporel pandas
    time_idx = pd.DatetimeIndex([horizon.map[t] for t in m.time])

    # --- SOC batterie (%)
    soc = series_from_component(m.bat[s], "soc", m.time, time_idx)
    if soc is None:
        raise ValueError("SOC batterie introuvable : m.bat[s].soc n'existe pas")

    if title_prefix is None:
        title_prefix = f"Scenario {s}"

    # =========================
    #  FIG 1 : pv_curt + soc
    # =========================
    pv_curt = None
    if hasattr(m.pv[s], "p_curt"):
        pv_curt = series_from_component(m.pv[s], "p_curt", m.time, time_idx)

    fig1, ax1 = plt.subplots(figsize=(14, 4))
    ax1.plot(soc.index, soc.values, label="SOC battery (%)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("SOC (%)")
    ax1.grid(True)

    if pv_curt is not None:
        ax1b = ax1.twinx()
        ax1b.plot(pv_curt.index, pv_curt.values, label="PV curtailment (W)")
        ax1b.set_ylabel("PV curtailment (W)")

        # légende combinée
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1b.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    else:
        ax1.legend(loc="upper left")

    ax1.set_title(f"{title_prefix} — pv_curt & SOC")
    fig1.tight_layout()
    plt.show()

    # =========================
    #  FIG 2 : is_full + soc
    # =========================
    is_full = None
    if hasattr(m, "is_full"):
        is_full = pd.Series(
            [int(round(value(m.is_full[s, t]))) for t in m.time],
            index=time_idx
        )

    fig2, ax2 = plt.subplots(figsize=(14, 4))
    ax2.plot(soc.index, soc.values, label="SOC battery (%)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("SOC (%)")
    ax2.grid(True)

    if is_full is not None:
        ax2b = ax2.twinx()
        ax2b.step(is_full.index, is_full.values, where="post", label="is_full (0/1)", linewidth=2)
        ax2b.set_ylabel("is_full")
        ax2b.set_yticks([0, 1])

        # légende combinée
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    else:
        ax2.legend(loc="upper left")

    ax2.set_title(f"{title_prefix} — is_full & SOC")
    fig2.tight_layout()
    plt.show()

    return fig1, fig2


def plot_lcc_and_battery_vs_pv(pv_list, lcc_list, bat_list, out_dir="results_image"):
    import os
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # --- LCC vs PV
    plt.figure()
    plt.plot(pv_list, lcc_list, marker='o')
    plt.xlabel("PV installé (W)")
    plt.ylabel("LCC / Coût total attendu (€)")
    plt.title("LCC en fonction de la puissance PV fixée")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "LCC_vs_PV.png"), dpi=200)
    plt.show()

    # --- Capacité batterie vs PV
    plt.figure()
    plt.plot(pv_list, bat_list, marker='o')
    plt.xlabel("PV installé (W)")
    plt.ylabel("Capacité batterie (Wh) (emax[t0])")
    plt.title("Capacité batterie optimale en fonction de la puissance PV fixée")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "BAT_vs_PV.png"), dpi=200)
    plt.show()


def plot_3d_surface_from_grid(
    x_list,
    y_list,
    z_grid,
    x_label="p_diesel (W)",
    y_label="PV fixé (W)",
    z_label="Value",
    title="3D surface",
    out_dir="results_image",
    filename_prefix="surface_3d"
):
    """
    Trace une surface 3D Z = f(X,Y) où :
      - x_list : liste des p_diesel (taille Nx)
      - y_list : liste des pv_fixed (taille Ny)
      - z_grid : array (Nx, Ny) avec Z[i,j] correspondant à (x_list[i], y_list[j])
                (utiliser np.nan pour les points infeasible)
    Sauvegarde une figure pickle + png.
    """
    os.makedirs(out_dir, exist_ok=True)

    X, Y = np.meshgrid(y_list, x_list)   # attention: X->PV, Y->diesel pour cohérence visuelle
    Z = np.array(z_grid, dtype=float)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")

    # masque des NaN pour éviter erreurs sur surface
    Z_masked = np.ma.masked_invalid(Z)

    surf = ax.plot_surface(X, Y, Z_masked)  # pas de couleur imposée
    ax.set_xlabel(y_label)
    ax.set_ylabel(x_label)
    ax.set_zlabel(z_label)
    ax.set_title(title)

    fig.tight_layout()

    # Sauvegardes
    pickle_path = os.path.join(out_dir, f"{filename_prefix}.pickle")
    with open(pickle_path, "wb") as f:
        pickle.dump(fig, f)

    png_path = os.path.join(out_dir, f"{filename_prefix}.png")
    fig.savefig(png_path, dpi=200)

    plt.show()
    return fig


def plot_3d_lcc_and_battery(
    p_diesel_list,
    pv_fixed_list,
    lcc_grid,
    bat_grid,
    out_dir="results_image",
    prefix="LCC_BAT_3D"
):
    """
    Produit 2 figures 3D :
      - LCC(p_diesel, pv_fixed)
      - Capacité batterie(p_diesel, pv_fixed)
    """
    plot_3d_surface_from_grid(
        x_list=p_diesel_list,
        y_list=pv_fixed_list,
        z_grid=lcc_grid,
        x_label="p_diesel max (W)",
        y_label="PV fixé (W)",
        z_label="LCC / coût total attendu (€)",
        title="LCC en fonction de (p_diesel, PV fixé)",
        out_dir=out_dir,
        filename_prefix=f"{prefix}_LCC"
    )

    plot_3d_surface_from_grid(
        x_list=p_diesel_list,
        y_list=pv_fixed_list,
        z_grid=bat_grid,
        x_label="p_diesel max (W)",
        y_label="PV fixé (W)",
        z_label="Capacité batterie (Wh)",
        title="Capacité batterie optimale en fonction de (p_diesel, PV fixé)",
        out_dir=out_dir,
        filename_prefix=f"{prefix}_BAT"
    )