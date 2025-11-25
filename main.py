# main_stochastic_nodiesel_24days.py
# - 5 scénarios de charge (24 jours, pas 15 min) à 20% chacun
# - Sans diesel
# - Batterie V3 (vieillissement) avec non-anticipation sur emax0
# - Conflit "load" corrigé -> block renommé en "demand"

from pyomo.environ import *

from lms2.core.horizon import SimpleHorizon
from lms2.tools.data_processing import read_data, load_data
from lms2.electric.sources import fixed_power_load

from block_pv import block_pv
from battery_factory import make_battery

import pandas as pd
import os
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

UB = 1e10

def _read_load_as_W(csv_path, value_candidates=("aggregate_wh","consumption_wh","value","load_wh")):
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

def _time_index_from_horizon(horizon, time_set):
    # Index exactement comme load_data l’attend
    return pd.DatetimeIndex([horizon.map[i] for i in time_set])

def _series_from_component(comp, attr_name, time_set, index):
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

if __name__ == "__main__":

    # -----------------------------
    # Fichiers d'entrée (24 jours)
    # -----------------------------
    scenario_load_files = [
        "microgrid_consumption/scenarios_24_days/24_days_example_1.csv",
        "microgrid_consumption/scenarios_24_days/24_days_example_1.csv",
        "microgrid_consumption/scenarios_24_days/24_days_example_1.csv",
        "microgrid_consumption/scenarios_24_days/24_days_example_1.csv",
        "microgrid_consumption/scenarios_24_days/24_days_example_1.csv",
    ]
    S = list(range(1, 6))
    prob = {s: 0.2 for s in S}

    irr_file = os.path.join("meteo_data", "irradiance_24_days.csv")   # cols: timestamp, Irradiance
    tmp_file = os.path.join("meteo_data", "temperature_24_days.csv")  # cols: timestamp, Temperature

    # -----------------------------
    # Horizon 24 jours (15 minutes)
    # -----------------------------
    horizon = SimpleHorizon(
        tstart="2023-01-01 00:00:00",
        tend="2023-01-24 23:45:00",
        time_step="15 minutes",
        tz="Indian/Antananarivo"
    )

    step_s = int(horizon.time_step.total_seconds())
    T = int(horizon.horizon.total_seconds())  # (24 jours - 15 min) en secondes

    m = ConcreteModel()
    m.time = RangeSet(0, T, step_s)
    t0 = m.time.first()

    # -----------------------------
    # Options techno
    # -----------------------------
    option_pv = {
        "time": m.time,
        "p_wp_min": 1, "p_wp_max": 6e6,
        "cost_inv": 1.5, "cost_opex": 0.02,
    }
    option_bat = {
        "time": m.time,
        "dt": step_s,
        "c_bat_max": 3e6, "c_bat_min": 1,
        "eta_c": 0.90, "eta_d": 0.85,
        "soc_min": 30, "soc_max": 100, "soc0": 100,
        "cost_inv": 0.12, "cost_opex": 0.0005,
        # (laisser tes paramètres V3 de vieillissement par défaut, ou les passer ici)
    }
    option_demand = { "time": m.time }  # pour fixed_power_load

    # -----------------------------
    # Blocs par scénario
    # -----------------------------
    m.S = Set(initialize=S)
    m.pv     = Block(m.S)
    m.bat    = Block(m.S)
    m.demand = Block(m.S)  # <<< RENOMMAGE pour éviter le conflit "load"

    for s in m.S:
        # PV
        block_pv(m.pv[s], curtailable=True, **option_pv)

        # Batterie V3 (vieillissement) – emax0 exposé par battery_factory.make_battery
        make_battery(m.bat[s], model=3, **option_bat)

        # Charge fixe
        fixed_power_load(m.demand[s], **option_demand)

    # -----------------------------
    # Non-anticipation (investissements communs)
    # -----------------------------
    s1 = S[0]

    @m.Constraint(m.S)
    def na_pv(_m, s):
        if s == s1: return Constraint.Skip
        return _m.pv[s].p_wp == _m.pv[s1].p_wp

    # Investissement batterie commun : emax[t0] identique
    m.bat_emax0 = Var(bounds=(option_bat["c_bat_min"], option_bat["c_bat_max"]))
    @m.Constraint(m.S)
    def na_bat(_m, s):
        return _m.bat[s].emax[t0] == _m.bat_emax0

    # -----------------------------
    # Bilan de puissance (sans diesel)
    # -----------------------------
    @m.Constraint(m.S, m.time)
    def power_balance(_m, s, t):
        return _m.bat[s].p[t] + _m.pv[s].p[t] == _m.demand[s].p[t]

    # -----------------------------
    # Chargement des données
    # -----------------------------
    irr = read_data(horizon, irr_file, usecols=["Time", "Irradiance"], tz_data="Indian/Antananarivo")
    tmp = read_data(horizon, tmp_file, usecols=["Time", "Temperature"], tz_data="Indian/Antananarivo")

    for s in m.S:
        # PV
        load_data(horizon, m.pv[s].irr, irr["Irradiance"])
        load_data(horizon, m.pv[s].tmp, tmp["Temperature"])
        # Batterie V3 : température pour vieillissement
        if hasattr(m.bat[s], "tmp"):
            load_data(horizon, m.bat[s].tmp, tmp["Temperature"])

        # Charge : convertir Wh/15min -> W (x4)
        W_vals = _read_load_as_W(scenario_load_files[s - 1])

        # Ajuster la longueur au nombre de pas (par sécurité)
        expected_pts = len(list(m.time))  # même cardinalité que l'usage de horizon.map[i]
        if len(W_vals) != expected_pts:
            if len(W_vals) > expected_pts:
                W_vals = W_vals[:expected_pts]
            else:
                W_vals = W_vals + [W_vals[-1]] * (expected_pts - len(W_vals))

        # >>> CLEF : construire l'index exactement comme load_data l'attend
        time_keys = pd.DatetimeIndex([horizon.map[i] for i in m.time])  # mêmes clés que load_data
        W_series = pd.Series(W_vals, index=time_keys)

        # Charger dans le Param Pyomo
        load_data(horizon, m.demand[s].p, W_series)

    # -----------------------------
    # Objectif (CAPEX + espérance OPEX)
    # -----------------------------
    PV_YEARS  = 9.64955841794
    BAT_YEARS = 9.64955841794

    def capex_pv(_m):
        return _m.pv[s1].cost_inv * _m.pv[s1].p_wp
    def capex_bat(_m):
        return _m.bat[s1].cost_inv * _m.bat_emax0
    def opex_pv(_m, s):
        return _m.pv[s].cost_opex * _m.pv[s1].p_wp * PV_YEARS
    def opex_bat(_m, s):
        return _m.bat[s].cost_opex * _m.bat_emax0 * BAT_YEARS
    def repl_bat(_m):
        return (_m.bat[s1].cost_inv * _m.bat_emax0 * 0.635227665282
                + _m.bat[s1].cost_inv * _m.bat_emax0 * 0.403514186739
                + _m.bat[s1].cost_inv * _m.bat_emax0 * 0.256323374751)

    @m.Objective(sense=minimize)
    def total_cost(_m):
        invest = capex_pv(_m) + capex_bat(_m)
        expected_opex = sum(prob[s] * (opex_pv(_m, s) + opex_bat(_m, s)) for s in _m.S)
        replacement_cost = repl_bat(_m)
        return invest + expected_opex + replacement_cost

    # -----------------------------
    # Solve
    # -----------------------------
    solver = SolverFactory('gurobi', tee=True, solver_io="direct")
    res = solver.solve(m, options={'MIPGap': 0.1})

    print("\n=== SOLUTION (Stoch 24 jours, sans diesel, Batt V3) ===")
    print(f"PV installé (W)        : {value(m.pv[s1].p_wp):,.1f}")
    print(f"Capacité batt emax0(Wh): {value(m.bat_emax0):,.1f}")
    print(f"Coût total attendu (€) : {value(m.total_cost):,.2f}")

    # =========================
    #  EXPORTS / BILANS / PLOTS
    # =========================035555582

    # Répertoires de sortie
    out_root = "outputs_stochastic"
    _ensure_dir(out_root)

    time_idx = _time_index_from_horizon(horizon, m.time)
    dt_s = int(horizon.time_step.total_seconds())

    # --- CAPEX/OPEX attendus (on réutilise tes fonctions / variables si elles existent)
    PV_YEARS = 9.64955841794
    BAT_YEARS = 9.64955841794
    s1 = list(m.S)[0]

    capex_pv_val = value(m.pv[s1].cost_inv) * value(m.pv[s1].p_wp)
    capex_bat_val = value(m.bat[s1].cost_inv) * value(m.bat_emax0)

    opex_exp = 0.0
    for s in m.S:
        opex_pv_s = value(m.pv[s].cost_opex) * value(m.pv[s1].p_wp) * PV_YEARS
        opex_bat_s = value(m.bat[s].cost_opex) * value(m.bat_emax0) * BAT_YEARS
        opex_exp += prob[s] * (opex_pv_s + opex_bat_s)

    '''
    ## test

    from main_deterministic_optimization import cout_total
    m.cost_inv_bat = sum([m.bat[s].cost_inv for s in S])
    m.obj = Objectif(expr=m.cost_inv_bat, sense=minimize) 
     '''

    cost_breakdown = {
        "capex_pv": capex_pv_val,
        "capex_bat": capex_bat_val,
        "opex_expected": opex_exp,
        "total_objective": value(m.total_cost),
    }
    with open(os.path.join(out_root, "cost_breakdown.json"), "w", encoding="utf-8") as f:
        json.dump(cost_breakdown, f, indent=2)
    print("Saved:", os.path.join(out_root, "cost_breakdown.json"))

    # --- Bilans et exports par scénario
    rows_summary = []
    for s in m.S:
        out_dir_s = os.path.join(out_root, f"scenario_{s}")
        _ensure_dir(out_dir_s)

        # Séries principales (W)
        demand_W = _series_from_component(m.demand[s], "p", m.time, time_idx)
        pv_W = _series_from_component(m.pv[s], "p", m.time, time_idx)  # PV injecté (après éventuelle limitation)
        bat_W = _series_from_component(m.bat[s], "p", m.time,
                                       time_idx)  # + discharge vers charge ; - charge de la batterie

        # Séries optionnelles (si dispo dans tes blocs)
        soc_pct = _series_from_component(m.bat[s], "soc", m.time, time_idx)  # %
        emax_Wh = _series_from_component(m.bat[s], "emax", m.time, time_idx)  # Wh (V3)
        # Selon l’implémentation du PV, l’une de ces colonnes peut exister -> curtailment = p_th - p (si dispo)
        p_pot = (_series_from_component(m.pv[s], "p_pot", m.time, time_idx)
                 or _series_from_component(m.pv[s], "p_theoretical", m.time, time_idx)
                 or _series_from_component(m.pv[s], "p_raw", m.time, time_idx)
                 or None)
        if p_pot is not None and pv_W is not None:
            curtail_W = (p_pot - pv_W).clip(lower=0.0)
        else:
            curtail_W = None

        # DataFrame export
        df = pd.DataFrame({"demand_W": demand_W, "pv_W": pv_W, "bat_W": bat_W})
        if soc_pct is not None: df["soc_pct"] = soc_pct
        if emax_Wh is not None: df["emax_Wh"] = emax_Wh
        if p_pot is not None:   df["pv_potential_W"] = p_pot
        if curtail_W is not None: df["pv_curtail_W"] = curtail_W

        csv_path = os.path.join(out_dir_s, f"timeseries_s{s}.csv")
        df.to_csv(csv_path, index_label="timestamp")
        print("Saved:", csv_path)

        # Bilans énergétiques (kWh) — sur l’horizon
        demand_kWh = _kwh_from_W_series(demand_W, dt_s)
        pv_kWh = _kwh_from_W_series(pv_W, dt_s) if pv_W is not None else 0.0
        bat_dis_kWh = _kwh_from_W_series(_positive_part(bat_W), dt_s) if bat_W is not None else 0.0
        bat_chg_kWh = _kwh_from_W_series(_negative_part_abs(bat_W), dt_s) if bat_W is not None else 0.0
        curtail_kWh = _kwh_from_W_series(curtail_W, dt_s) if curtail_W is not None else None

        rows_summary.append({
            "scenario": s,
            "probability": prob[s],
            "demand_kWh": demand_kWh,
            "pv_to_load_kWh": pv_kWh,
            "bat_discharge_kWh": bat_dis_kWh,
            "bat_charge_kWh": bat_chg_kWh,
            "pv_curtail_kWh": curtail_kWh,
        })

        # Tracés (3 figures légères)
        # 1) Puissances principales
        plt.figure(figsize=(12, 3))
        if demand_W is not None: plt.plot(demand_W.index, demand_W.values, label="Demand (W)")
        if pv_W is not None: plt.plot(pv_W.index, pv_W.values, label="PV to load (W)")
        if bat_W is not None: plt.plot(bat_W.index, bat_W.values, label="Battery p (W)")
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

    # Résumé bilans + espérance
    df_summary = pd.DataFrame(rows_summary)
    df_summary["expected_kWh_contrib"] = df_summary["probability"] * df_summary["demand_kWh"]
    df_summary_path = os.path.join(out_root, "energy_summary_by_scenario.csv")
    df_summary.to_csv(df_summary_path, index=False)
    print("Saved:", df_summary_path)

    # Agrégat attendu (ex : demande attendue)
    expected_demand_kWh = float((df_summary["probability"] * df_summary["demand_kWh"]).sum())
    expected_pv_kWh = float((df_summary["probability"] * df_summary["pv_to_load_kWh"]).sum())
    print(f"Expected demand over horizon (kWh): {expected_demand_kWh:.2f}")
    print(f"Expected PV-to-load over horizon (kWh): {expected_pv_kWh:.2f}")

