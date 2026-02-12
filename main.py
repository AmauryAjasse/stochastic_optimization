from pyomo.environ import *
from pyomo.opt import TerminationCondition as TC

from lms2.core.horizon import SimpleHorizon
from lms2.tools.data_processing import read_data, load_data
from lms2.electric.sources import fixed_power_load
from functions_economic import *
from functions_constraint import *
from functions_useful import *
from functions_visualisation import *
from concurrent.futures import ProcessPoolExecutor, as_completed

from block_pv import block_pv
from battery_factory import make_battery
from block_diesel_generator import diesel_generator, diesel_generator_V2

import pandas as pd
import os
from pathlib import Path
import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import itertools
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_one_case(
        p0: int,
        pv_p_wp_fixed: int,
        scenario_load_files: list[str],
        irr_file: str,
        tmp_file: str,
        with_diesel_generator: int,
        battery_model: int,
        discount_rate: float,
        total_duration: int,
        battery_replacement_years: tuple,
        time_start: str,
        time_end: str,
        time_step: str,
        consumption_satisfaction: int,
        allow_consumption_shedding: bool,
        shedding_hour_start: int,
        shedding_hour_end: int,
        MIP_GAP: float,
        gurobi_threads: int) -> dict:
    """ On commence par créer le modèle :
        - on définit l'horizon temporel
        - on crée la liste de probabilité des scénarios
        - on crée le modèle et on fixe le temps"""
    horizon = SimpleHorizon(tstart=time_start, tend=time_end, time_step=time_step, tz="Indian/Antananarivo")

    step_s = int(horizon.time_step.total_seconds())
    T = int(horizon.horizon.total_seconds())  # (24 jours - 15 min) en secondes

    S = list(range(1, len(scenario_load_files) + 1))
    prob = {s: 1 / len(S) for s in S}

    m = ConcreteModel()
    m.time = RangeSet(0, T, step_s)
    t0 = m.time.first()

    def _allow_shed_init(m, t):
        dt = horizon.map[t]  # datetime (timezone déjà gérée par horizon)
        h = dt.hour  # 0..23
        return 1 if (shedding_hour_start <= h < shedding_hour_end) else 0

    if allow_consumption_shedding:
        m.allow_shed = Param(m.time, initialize=_allow_shed_init, within=Binary)

    """ On définit les options des différents blocs qui constituent le micro-réseau."""
    # option_pv = {"time": m.time, "p_wp_min": 1, "p_wp_max": 1e5, "cost_inv": 1.5, "cost_opex": 0.02}
    option_pv = {"time": m.time, "p_wp_fixed": pv_p_wp_fixed, "cost_inv": 1.5, "cost_opex": 0.02}
    option_bat = {"time": m.time, "dt": step_s, "c_bat_max": 1e6, "c_bat_min": 1, "eta_c": 0.90,
                  "eta_d": 0.85, "soc_min": 30, "soc_max": 100, "soc_allow_curt": 80, "soc0": 70, "socf": None,
                  "cost_inv": 0.12, "cost_opex": 0.0005}
    option_consumption = {"time": m.time}  # pour fixed_power_load
    option_gen = {"time": m.time, "dt": step_s, "eff": 0.35, "fuel_cost": 1.2,
                  "fuel_consumption": 0.00009639, "cost_inv": 0.7, "cost_opex": 0.03}
    option_gen_V2 = {'time': m.time, 'dt': horizon.time_step.total_seconds(), 'p0': p0, 'fuel_cost': 1.2,
                     'fuel_consumption': 0.00009639, 'cost_inv': 0.7, 'cost_opex': 0.03}

    # -----------------------------
    # Blocs par scénario
    # -----------------------------
    m.S = Set(initialize=S)
    m.pv = Block(m.S)
    m.bat = Block(m.S)
    m.consumption = Block(m.S)
    m.p_unserved = Var(m.S, m.time, domain=NonNegativeReals)
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
    if m.pv[s1].p_wp.is_variable_type():
        m.same_pv = Constraint(m.S, rule=lambda b, s: same_pv_rule(b, s, s_ref=s1))
    m.same_bat = Constraint(m.S, rule=lambda b, s: same_bat_rule(b, s, s_ref=s1, t0=t0))
    if with_diesel_generator == 1:
        m.same_gen = Constraint(m.S, rule=lambda b, s: same_gen_rule(b, s, s_ref=s1))

    # -----------------------------
    # Contraintes d'optimisation (bilan de puissance, gestion d'énergie)
    # -----------------------------

    # Bilan de puissance
    m.bilan_puissance = Constraint(m.S, m.time, rule=lambda b, s, t: bilan_with_shed_rule(b, s, t,
                                                                                          with_diesel_generator=with_diesel_generator))
    # On ne peut pas écrêter le solaire tant que la batterie n'a pas atteint une certaine valeur de soc (soc_allow_curt)
    M_curt, M_energy = pv_curt_bigM_rule(m, s_ref=s1, M_curt=1e6, M_energy=None)
    m.is_full = Var(m.S, m.time, domain=Binary)
    m.full_lower = Constraint(m.S, m.time,
                              rule=lambda b, s, t: full_lower_rule(b, s, t, s_ref=s1, M_energy=M_energy))
    m.full_upper = Constraint(m.S, m.time,
                              rule=lambda b, s, t: full_upper_rule(b, s, t, s_ref=s1, M_energy=M_energy))
    m.curtail_only_if_full = Constraint(m.S, m.time, rule=lambda b, s, t: curtail_only_if_full_rule(b, s, t,
                                                                                                    M_curt=M_curt))
    # On doit satisfaire la consommation au moins xx% du temps (xx = consumption_satisfaction)
    m.is_served = Var(m.S, m.time, domain=Binary)
    m.served_link = Constraint(m.S, m.time, rule=lambda b, s, t: served_link_rule(b, s, t, M_shed=1e9))
    min_served = min_served_steps(m.time, consumption_satisfaction)
    m.consumption_satisfaction = Constraint(m.S, rule=lambda b, s: satisfaction_rule_per_scenario(b, s,
                                                                                                  min_served=min_served))
    # Si on ne satisfait pas la charge, c'est obligatoirement entre shedding_hour_start et shedding_hour_end
    if allow_consumption_shedding:
        m.no_shed_outside_window = Constraint(m.S, m.time, rule=lambda b, s, t: forbid_shed_outside_window_rule(b, s, t,
                                                                                                                allow_shed=b.allow_shed))
        m.force_served_outside_window = Constraint(m.S, m.time,
                                                   rule=lambda b, s, t: force_served_outside_window_rule(b, s, t,
                                                                                                         allow_shed=b.allow_shed))
    # Si la charge n'est pas satisfaite, on ne peux pas recharger les batteries
    m.no_battery_when_shed_pos = Constraint(m.S, m.time,
                                            rule=lambda b, s, t: no_battery_when_shed_rule(b, s, t))
    m.no_battery_when_shed_neg = Constraint(m.S, m.time,
                                            rule=lambda b, s, t: no_battery_when_shed_rule_neg(b, s, t))

    # --- SOC final >= SOC initial
    m.soc_final_ge_initial = Constraint(m.S, rule=lambda b, s: soc_final_ge_initial_rule(b, s))

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

        # Charger dans le Param Pyomo
        load_data(horizon, m.consumption[s].p, pd.Series(W_vals, index=time_keys))

    # -----------------------------
    # Objectif (CAPEX + espérance OPEX)
    # -----------------------------

    m.capex_pv = Expression(rule=lambda b: capex_pv_rule(b))
    m.capex_bat = Expression(rule=lambda b: capex_bat_rule(b))
    m.opex_pv = Expression(
        rule=lambda b: opex_pv_rule(b, discount_rate=discount_rate, total_duration=total_duration))
    m.opex_bat = Expression(
        rule=lambda b: opex_bat_rule(b, discount_rate=discount_rate, total_duration=total_duration))
    m.repl_bat = Expression(rule=lambda b: repl_bat_rule(b, discount_rate=discount_rate,
                                                         replacement_year=battery_replacement_years))
    if with_diesel_generator != 0:
        m.capex_gen = Expression(rule=lambda b: capex_gen_rule(b))
        m.opex_gen = Expression(
            rule=lambda b: opex_gen_rule(b, discount_rate=discount_rate, total_duration=total_duration))
        m.expected_fuel_cost = Expression(
            rule=lambda b: expected_fuel_cost_rule(b, prob=prob, discount_rate=discount_rate,
                                                   total_duration=total_duration))

    """The objective functions are defined in the cases without and with diesel generator."""
    m.total_cost = Objective(
        rule=lambda b: total_cost_rule(b, prob=prob, with_diesel_generator=with_diesel_generator,
                                       discount_rate=discount_rate, total_duration=total_duration,
                                       replacement_year=battery_replacement_years))

    # -----------------------------
    # Solve
    # -----------------------------
    t_solve_start = datetime.datetime.now()
    solver = SolverFactory('gurobi', solver_io="direct")
    res = solver.solve(m, options={"MIPGap": MIP_GAP,
                                             "Threads": gurobi_threads})  # pour voir le détail, rajouter tee=True entre m et options
    t_solve_end = datetime.datetime.now()

    dt_solve = (t_solve_end - t_solve_start).total_seconds()
    print(f"Temps pour solve avec p0={p0}W et p_wp={pv_p_wp_fixed}W : {dt_solve:,.2f} s")

    out_root = os.path.join("outputs_stochastic", f"p0_{p0}_pv_{pv_p_wp_fixed}")

    # df_met, sat_exp, unserved_exp, pvcurt_exp = scenario_metrics_df(m, prob=prob, step_s=step_s)

    # =========================
    #  EXPORTS / BILANS / PLOTS
    # =========================

    # On affiche un message seulement si l'optimisation ne fonctionne pas
    tc = res.solver.termination_condition
    if tc not in (TC.optimal, TC.feasible):
        return {"p0": p0, "pv_fixed": pv_p_wp_fixed, "status": str(tc)}

    # --- coûts (communs au run, on les répète sur chaque ligne scénario)
    capex_total = float(value(m.capex_pv) + value(m.capex_bat))
    opex_total = float(value(m.opex_pv) + value(m.opex_bat))
    repl_bat = float(value(m.repl_bat))
    fuel_cost = 0.0
    if with_diesel_generator != 0:
        capex_total += float(value(m.capex_gen))
        opex_total += float(value(m.opex_gen))
        fuel_cost = float(value(m.expected_fuel_cost))

    # --- métriques par scénario (taux satisfaction + PV écrêté)
    df_met, sat_exp, unserved_exp, pvcurt_exp = scenario_metrics_df(m, prob=prob, step_s=step_s)

    # Exports par scénario (timeseries + bilans + plots)
    rows_summary = export_scenario_timeseries_and_plots(m, horizon=horizon, prob=prob, results_root="results", pv_installed_W=pv_p_wp_fixed, p0_diesel_W=p0, with_diesel_generator=with_diesel_generator)

    # Construire une ligne par scénario avec exactement les colonnes voulues
    rows = []
    for _, row in df_met.iterrows():
        rows.append({
            "p0": p0,
            "pv_fixed": pv_p_wp_fixed,
            "total_cost": float(value(m.total_cost)),
            "bat_emax_t0": float(value(m.bat[s1].emax[t0])),
            "pv_wp": float(value(m.pv[s1].p_wp)),
            "scenario": int(row["scenario"]),  # <- colonne scénario demandée
            "pv_curt_Wh": float(row["pv_curt_Wh"]),  # <- PV écrêté dans le scénario
            "sat_time_pct": float(row["sat_time_%"]),  # <- taux satisfaction (%) dans le scénario
            "unserved_Wh": float(row["unserved_Wh"]),  # énergie non servie
            "capex_total_EUR": capex_total,
            "opex_total_EUR": opex_total,
            "repl_bat_EUR": repl_bat,
            "fuel_cost_EUR": fuel_cost,
        })

    return {"rows": rows}




if __name__ == "__main__":
    # -----------------------------
    # Paramètres d'entrée globaux
    # -----------------------------
    with_diesel_generator = 0                 # 0 : without diesel, 1 : diesel_V1, 2 : diesel_V2 (with SOS2)
    battery_model = 3                         # 2 : battery_V2, 3 : battery_V3
    discount_rate = 0.095                     # taux d'actualisation (0.095 pour Madagascar)
    total_duration = 20                       # durée du micro-réseau en années
    battery_replacement_years = (5, 10, 15)   # année de remplacement du pack de batterie
    time_start = "2023-01-01 00:00:00"        # début de l'horizon temporel
    time_end = "2023-01-05 23:00:00"          # fin de l'horizon temporel
    time_step = "1 hour"                      # on peut mettre "15 min" ou "30 min" aussi
    consumption_satisfaction = 90             # % du temps où la consommation est satisfaite
    UB = 1e10

    allow_consumption_shedding = False                   # si True, on doit satisfaire la charge entre shed_hour_start et shed_hour_end
    shedding_hour_start = 0                       # heure de départ autorisation de non satisfaction de la charge
    shedding_hour_end = 4                         # heure de fin autorisation de non satisfaction de la charge

    MIP_GAP = 0.8
    MAX_WORKERS = 4
    GUROBI_THREADS = 5

    pv_p_wp_fixed_list = list(range(5000, 20000, 1000))     # puissance installée en W
    p0_list=[0] if with_diesel_generator==0 else list(range(2000, 6000, 2000))

    jobs = [(p0, pv) for p0 in p0_list for pv in pv_p_wp_fixed_list]
    # results_p0 = []
    # total_cost_list=[]
    # pv_wp_list=[]
    # bat_capa_list=[]

    rows_all = []

    # lcc_grid = np.full((len(p0_list), len(pv_p_wp_fixed_list)), np.nan)
    # bat_grid = np.full((len(p0_list), len(pv_p_wp_fixed_list)), np.nan)
    t_start_total = datetime.datetime.now()

    # for i_p0, p0 in enumerate(p0_list):
    #     print(f"\n===== Étude pour p0 = {p0} W ==================================================================================================================================================================")
    #     for j_pv, pv_p_wp_fixed in enumerate(pv_p_wp_fixed_list):
    #         print(f"\n===== Étude pour PV fixé = {pv_p_wp_fixed} W (p0={p0}) =====")


    # -----------------------------
    # Fichiers d'entrée (24 jours)
    # -----------------------------
    scenario_load_files = [
                    "microgrid_consumption/scenarios_24_days/24_days_example_1_1h.csv",
                    "microgrid_consumption/scenarios_24_days/24_days_example_2_1h.csv"
                    # "microgrid_consumption/scenarios_24_days/24_days_example_3_1h.csv",
                    # "microgrid_consumption/scenarios_24_days/24_days_example_4_1h.csv",
                    # "microgrid_consumption/scenarios_24_days/24_days_example_5_1h.csv"

                    # multiply_by("microgrid_consumption/scenarios_24_days/24_days_example_1.csv", 0.2),
                    # multiply_by("microgrid_consumption/scenarios_24_days/24_days_example_2.csv", 0.3),
                    # multiply_by("microgrid_consumption/scenarios_24_days/24_days_example_3.csv", 0.2),
                    # multiply_by("microgrid_consumption/scenarios_24_days/24_days_example_4.csv", 0.2),
                    # multiply_by("microgrid_consumption/scenarios_24_days/24_days_example_5.csv", 0.3)
                ]
    print(f"scenario files :\n" + "\n".join(scenario_load_files))

    irr_file = os.path.join("meteo_data", "irradiance_24_days_1h.csv")     # cols: timestamp, Irradiance
    tmp_file = os.path.join("meteo_data", "temperature_24_days_1h.csv")    # cols: timestamp, Temperature

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [
            ex.submit(
                run_one_case,
                p0, pv,
                scenario_load_files,
                irr_file, tmp_file,
                with_diesel_generator,
                battery_model,
                discount_rate,
                total_duration,
                battery_replacement_years,
                time_start, time_end, time_step,
                consumption_satisfaction,
                allow_consumption_shedding,
                shedding_hour_start, shedding_hour_end,
                MIP_GAP,
                GUROBI_THREADS
            )
            for (p0, pv) in jobs
        ]

        for fut in as_completed(futures):
            r = fut.result()
            if r.get("status", "ok") != "ok":
                print(f"[FAIL] pv_fixed={r['pv_fixed']}W -> {r['status']}")
                continue
            # print(f"[DONE] pv_fixed={r['pv_fixed']}W -> total_cost={r['total_cost']:.2f} € | out={r['out_dir']}")

            rows_all.extend(r["rows"])


    save_results_in_tab(with_diesel_generator, rows_all, pv_p_wp_fixed_list)

    rows_all.sort(key=lambda d: d["pv_fixed"])
    pv_wp_list      = [r["pv_wp"] for r in rows_all]
    total_cost_list = [r["total_cost"] for r in rows_all]
    bat_capa_list   = [r["bat_emax_t0"] for r in rows_all]

    # view_sizing_evolution_wih_diesel(p0_list, total_cost_list, pv_wp_list, bat_capa_list, "results_image")

    # Dans le cas où on a une seule valeur de p0_diesel et où on fait varier p_wp_fixed
    # plot_lcc_and_battery_vs_pv(pv_wp_list, total_cost_list, bat_capa_list, out_dir="results_image")

    # plot_3d_lcc_and_battery(
    #     p_diesel_list=p0_list,
    #     pv_fixed_list=pv_p_wp_fixed_list,
    #     lcc_grid=lcc_grid,
    #     bat_grid=bat_grid,
    #     out_dir="results_image",
    #     prefix="study"
    # )

    t_end_total = datetime.datetime.now()
    dt_total_script = (t_end_total - t_start_total).total_seconds()
    print(f"Temps total du script : {dt_total_script:,.2f} s")