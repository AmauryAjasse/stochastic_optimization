# main_stochastic_nodiesel_24days.py
# - 5 scénarios de charge (24 jours, pas 15 min) à 20% chacun
# - Sans diesel
# - Batterie V3 (vieillissement) avec non-anticipation sur emax0

from pyomo.environ import *

from lms2.core.horizon import SimpleHorizon
from lms2.tools.data_processing import read_data, load_data
from lms2.electric.sources import fixed_power_load
from functions_economic import *
from functions_constraint import *
from functions_useful import *
from functions_visualisation import *

from block_pv import block_pv
from battery_factory import make_battery
from block_diesel_generator import diesel_generator, diesel_generator_V2

import pandas as pd
import os
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

UB = 1e10

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

if __name__ == "__main__":
    # -----------------------------
    # Paramètres d'entrée globaux
    # -----------------------------
    with_diesel_generator = 2                 # 0 : without diesel, 1 : diesel_V1, 2 : diesel_V2 (with SOS2)
    battery_model = 3                         # 2 : battery_V2, 3 : battery_V3
    discount_rate = 0.095                     # taux d'actualisation (0.095 pour Madagascar)
    total_duration = 20                       # durée du micro-réseau en années
    battery_replacement_years = (5, 10, 15)   # année de remplacement du pack de batterie
    time_start = "2023-01-01 00:00:00"        # début de l'horizon temporel
    time_end = "2023-01-24 23:45:00"          # fin de l'horizon temporel

    p0_list = [1, 10]                          # puissance maximale de sortie du générateur diesel
    results_p0 = []
    total_cost=[]
    pv_wp=[]
    bat_capa=[]

    for p0 in p0_list:
        print(f"\n===== Étude pour p0 = {p0} W =====")
        # -----------------------------
        # Fichiers d'entrée (24 jours)
        # -----------------------------
        scenario_load_files = [
            "microgrid_consumption/scenarios_24_days/24_days_example_1.csv",
            "microgrid_consumption/scenarios_24_days/24_days_example_2.csv",
            "microgrid_consumption/scenarios_24_days/24_days_example_3.csv",
            # multiply_by("microgrid_consumption/scenarios_24_days/24_days_example_3.csv", 1.5),
            "microgrid_consumption/scenarios_24_days/24_days_example_4.csv",
            "microgrid_consumption/scenarios_24_days/24_days_example_5.csv"
            # multiply_by("microgrid_consumption/scenarios_24_days/24_days_example_5.csv", 1.5)
        ]
        print(f"scenario files :\n" + "\n".join(scenario_load_files))

        S = list(range(1, 6))
        prob = {s: 0.2 for s in S}

        irr_file = os.path.join("meteo_data", "irradiance_24_days.csv")   # cols: timestamp, Irradiance
        tmp_file = os.path.join("meteo_data", "temperature_24_days.csv")  # cols: timestamp, Temperature

        # -----------------------------
        # Horizon 24 jours (15 minutes)
        # -----------------------------
        horizon = SimpleHorizon(tstart=time_start, tend=time_end, time_step="15 minutes", tz="Indian/Antananarivo")

        step_s = int(horizon.time_step.total_seconds())
        T = int(horizon.horizon.total_seconds())  # (24 jours - 15 min) en secondes

        m = ConcreteModel()
        m.time = RangeSet(0, T, step_s)
        t0 = m.time.first()

        # -----------------------------
        # Options techno
        # -----------------------------
        option_pv = {"time": m.time, "p_wp_min": 1, "p_wp_max": 6e6, "cost_inv": 1.5, "cost_opex": 0.02}
        option_bat = {"time": m.time, "dt": step_s, "c_bat_max": 3e6, "c_bat_min": 1, "eta_c": 0.90, "eta_d": 0.85, "soc_min": 30, "soc_max": 100, "soc0": 100, "cost_inv": 0.12, "cost_opex": 0.0005}
        option_consumption = {"time": m.time }  # pour fixed_power_load
        option_gen = {"time": m.time, "dt": step_s, "p0_min": 1, "p0_max": 1e6, "eff": 0.35, "fuel_cost": 1.2, "fuel_consumption": 0.00009639, "cost_inv": 0.7, "cost_opex": 0.03}
        option_gen_V2 = {'time': m.time, 'dt': horizon.time_step.total_seconds(), 'p0': p0, 'fuel_cost': 1.2, 'fuel_consumption': 0.00009639, 'cost_inv': 0.7, 'cost_opex': 0.03}

        # -----------------------------
        # Blocs par scénario
        # -----------------------------
        m.S = Set(initialize=S)
        m.pv     = Block(m.S)
        m.bat    = Block(m.S)
        m.consumption = Block(m.S)
        if with_diesel_generator != 0:
            m.gen = Block(m.S)

        for s in m.S:
            block_pv(m.pv[s], curtailable=True, **option_pv)
            make_battery(m.bat[s], model=3, **option_bat)
            fixed_power_load(m.consumption[s], **option_consumption)
            if with_diesel_generator == 1:
                diesel_generator(m.gen[s], **option_gen)
            elif with_diesel_generator == 2:
                diesel_generator_V2(m.gen[s], **option_gen_V2)

        # -----------------------------
        # Non-anticipation (investissements communs)
        # -----------------------------
        s1 = S[0]

        m.same_pv = Constraint(m.S, rule=lambda b, s: same_pv_rule(b, s, s_ref=s1))
        m.same_bat = Constraint(m.S, rule=lambda b, s: same_bat_rule(b, s, s_ref=s1, t0=t0))
        if with_diesel_generator == 1:
            m.same_gen = Constraint(m.S, rule=lambda b, s: same_gen_rule(b, s, s_ref=s1))

        # -----------------------------
        # Bilan de puissance (sans diesel)
        # -----------------------------

        m.bilan_puissance = Constraint(m.S, m.time, rule=lambda b, s, t: bilan_puissance_rule(b, s, t, with_diesel_generator=with_diesel_generator))

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
            W_vals = read_load_as_W(scenario_load_files[s - 1])

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
            load_data(horizon, m.consumption[s].p, W_series)

        # -----------------------------
        # Objectif (CAPEX + espérance OPEX)
        # -----------------------------

        m.capex_pv = Expression(rule=lambda b: capex_pv_rule(b))
        m.capex_bat = Expression(rule=lambda b: capex_bat_rule(b))
        m.opex_pv = Expression(rule=lambda b: opex_pv_rule(b, discount_rate=discount_rate, total_duration=total_duration))
        m.opex_bat = Expression(rule=lambda b: opex_bat_rule(b, discount_rate=discount_rate, total_duration=total_duration))
        m.repl_bat = Expression(rule=lambda b: repl_bat_rule(b, discount_rate=discount_rate, replacement_year=battery_replacement_years))
        if with_diesel_generator != 0 :
            m.capex_gen = Expression(rule=lambda b: capex_gen_rule(b))
            m.opex_gen = Expression(rule=lambda b: opex_pv_rule(b, discount_rate=discount_rate, total_duration=total_duration))
            m.expected_fuel_cost = Expression(rule=lambda b: expected_fuel_cost_rule(b, prob=prob, discount_rate=discount_rate, total_duration=total_duration))

        """The objective functions are defined in the cases without and with diesel generator."""
        m.total_cost = Objective(rule=lambda b: total_cost_rule(b, prob=prob, with_diesel_generator=with_diesel_generator, discount_rate=discount_rate, total_duration=total_duration, replacement_year=battery_replacement_years))

        # -----------------------------
        # Solve
        # -----------------------------
        solver = SolverFactory('gurobi', tee=True, solver_io="direct")
        res = solver.solve(m, options={'MIPGap': 0.1})

        print(f"\n=== SOLUTION (Stochastic optimization {count_days_inclusive(time_start, time_end)} jours, diesel version {with_diesel_generator}, battery version {battery_model}) ===")
        print(f"PV installé (W)        : {value(m.pv[s1].p_wp):,.1f}")
        print(f"Capacité batterie emax0 (Wh): {value(m.bat[s1].emax[t0]):,.1f}")
        print(f"Coût total attendu (€) : {value(m.total_cost):,.2f}\n")

        for s in m.S:
            pv_curt_Wh = compute_pv_curtailment_wh(m, s, dt_s=step_s)
            print(f"Scénario {s} : énergie PV écrêtée = {pv_curt_Wh:.1f} Wh")
        print("\n")

        total_cost.append(value(value(m.total_cost)))
        pv_wp.append(value(m.pv[s1].p_wp))
        bat_capa.append(value(m.bat[s1].emax0))

        # =========================
        #  EXPORTS / BILANS / PLOTS
        # =========================

        # Répertoires de sortie
        out_root = "outputs_stochastic"
        ensure_dir(out_root)

        # 1) Breakdown coûts (CAPEX / OPEX attendu)
        cost_breakdown = compute_and_save_cost_breakdown(m, prob=prob, t0=t0, out_root=out_root)
        # 2) Exports par scénario (timeseries + bilans + plots)
        rows_summary = export_scenario_timeseries_and_plots(m, horizon=horizon, prob=prob, out_root=out_root, with_diesel_generator=with_diesel_generator)
        # 3) Résumé multi-scénarios + espérance
        df_summary = summarize_energy_expectation(rows_summary, out_root=out_root)