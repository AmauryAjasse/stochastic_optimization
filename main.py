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
    scenario_probabilities = {s: 1 / len(S) for s in S}
    # scenario_probabilities = {
    #     1: 0.10,  # S1 faible
    #     2: 0.20,  # S2 bas
    #     3: 0.40,  # S3 référence
    #     4: 0.20,  # S4 croissance
    #     5: 0.10  # S5 forte croissance
    # }
    if abs(sum(scenario_probabilities.values()) - 1.0) > 1e-9:
        raise ValueError("La somme des probabilités des scénarios doit être égale à 1.")

    m = ConcreteModel()
    m.time = RangeSet(0, T, step_s)
    t0 = m.time.first()

    t_list = list(m.time)
    day_of_t = {t: horizon.map[t].date() for t in t_list}  # date() = AAAA-MM-JJ

    days = sorted(set(day_of_t.values()))
    m.DAYS = Set(initialize=days, ordered=True)

    def _init_T_of_day(m, d):
        return [t for t in t_list if day_of_t[t] == d]

    m.T_of_day = Set(m.DAYS, initialize=_init_T_of_day, ordered=True)

    # min 80% du temps servi par jour
    daily_satisfaction_pct = 80.0

    def _init_min_served_day(m, d):
        n = len(list(m.T_of_day[d]))
        return int(math.ceil((daily_satisfaction_pct / 100.0) * n))

    m.min_served_day = Param(m.DAYS, initialize=_init_min_served_day, within=NonNegativeIntegers)

    def _allow_shed_init(m, t):
        dt = horizon.map[t]  # datetime (timezone déjà gérée par horizon)
        h = dt.hour  # 0..23
        return 1 if (shedding_hour_start <= h < shedding_hour_end) else 0

    if allow_consumption_shedding:
        m.allow_shed = Param(m.time, initialize=_allow_shed_init, within=Binary)

    """ On définit les options des différents blocs qui constituent le micro-réseau."""
    # option_pv = {"time": m.time, "p_wp_min": 1, "p_wp_max": 1e5, "cost_inv": 1.5, "cost_opex": 0.02}
    option_pv = {"time": m.time, "p_wp_fixed": pv_p_wp_fixed, "cost_inv": 1.0, "cost_opex": 0.009}
    option_bat = {"time": m.time, "dt": step_s, "c_bat_max": 1e6, "c_bat_min": 1, "eta_c": 0.93,
                  "eta_d": 0.92, "soc_min": 30, "soc_max": 100, "soc_allow_curt": 80, "soc0": 70, "socf": None,
                  "cost_inv": 0.12, "cost_opex": 0.0005}
    option_consumption = {"time": m.time}  # pour fixed_power_load
    option_gen = {"time": m.time, "dt": step_s, "eff": 0.35, "fuel_cost": 1.2,
                  "fuel_consumption": 0.00009639, "cost_inv": 0.7, "cost_opex": 0.03}
    option_gen_V2 = {'time': m.time, 'dt': horizon.time_step.total_seconds(), 'p0': p0, 'fuel_cost': 12.0,
                     'fuel_consumption': 0.00009639, 'cost_inv': 0.7, 'cost_opex': 0.03}

    # -----------------------------
    # Blocs par scénario
    # -----------------------------
    m.S = Set(initialize=S)
    m.prob = Param(m.S, initialize=scenario_probabilities, within=NonNegativeReals)
    m.pv = Block(m.S)
    m.bat = Block(m.S)
    m.consumption = Block(m.S)
    m.p_unserved = Var(m.S, m.time, domain=NonNegativeReals)

    VOLL = 5.0  # exemple : 5 €/kWh (à ajuster)
    dt_h = step_s / 3600.0

    m.unserved_kWh = Expression(m.S, rule=lambda b, s: sum(b.p_unserved[s, t] * dt_h / 1000.0 for t in b.time))
    # m.cost_unserved = Expression(m.S, rule=lambda b, s: VOLL * b.unserved_kWh[s])
    # Espérance d'énergie non servie en kWh
    # m.expected_unserved_kWh = Expression(rule=lambda b: sum(prob[s] * sum(b.p_unserved[s, t] * dt_h / 1000.0 for t in b.time) for s in b.S))
    # Coût associé
    # m.cost_unserved = Expression(rule=lambda b: VOLL * b.expected_unserved_kWh)

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

    # on doit satisfaire 80% de l'énergie tous les jours
    m.daily_satisfaction = Constraint(m.S, m.DAYS, rule=lambda b, s, d: satisfaction_rule_per_day_per_scenario(b, s, d, b.min_served_day, b.T_of_day))

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
        m.expected_fuel_cost = Expression(m.S, rule=lambda b, s: expected_fuel_cost_rule(b, s, discount_rate=discount_rate, total_duration=total_duration))

    m.total_cost = Expression(m.S,
        rule=lambda b, s: total_cost_rule(b, s, with_diesel_generator=with_diesel_generator,
                                       discount_rate=discount_rate, total_duration=total_duration,
                                       replacement_year=battery_replacement_years))

    """The objective functions are defined in the cases without and with diesel generator."""
    # m.obj = Objective(
    #     rule=lambda b: total_cost_rule(b, prob=prob, with_diesel_generator=with_diesel_generator,
    #                                    discount_rate=discount_rate, total_duration=total_duration,
    #                                    replacement_year=battery_replacement_years) + b.cost_unserved)
    # m.obj = Objective(rule=lambda b: sum(b.total_cost[s] for s in b.S))
    m.obj = Objective(rule=lambda b: sum(b.prob[s] * b.total_cost[s] for s in b.S))
    # m.obj = Objective(rule=lambda b: sum(b.total_cost[s] for s in b.S) + b.cost_unserved)

    # --------------------------------------------
    # Solve
    # --------------------------------------------
    # print("soc0:", value(m.bat[s1].soc0), "socf:", value(m.bat[s1].socf))
    t_solve_start = datetime.datetime.now()
    solver = SolverFactory('gurobi', solver_io="direct")
    res = solver.solve(m, options={"MIPGap": MIP_GAP,
                                             "Threads": gurobi_threads})  # pour voir le détail, rajouter tee=True entre m et options
    t_solve_end = datetime.datetime.now()

    dt_solve = (t_solve_end - t_solve_start).total_seconds()
    print(f"Temps pour solve avec p0={p0}W et p_wp={pv_p_wp_fixed}W : {dt_solve:,.2f} s")

    # print("\n===== Solution optimale =====")
    # print(f"Puissance PV retenue : {value(m.pv_p_wp_fixed):.0f} W")
    # print(f"Capacité batterie    : {value(m.bat[s1].emax[t0]):.0f} Wh")
    print(f"Coût total           : {value(m.obj):.2f} €")
    # print("\nChoix PV :")
    # for k in m.KPV:
    #     if value(m.y_pv[k]) > 0.5:
    #         print(f"  indice {k} -> {pv_p_wp_list[k]} W")
    # if with_diesel_generator == 2:
    #     print(f"Puissance diesel retenue : {value(m.p0_selected):.0f} W")
    #     print("\nChoix diesel :")
    #     for k in m.KGEN:
    #         if value(m.y_gen[k]) > 0.5:
    #             print(f"  indice {k} -> {p0_list[k]} W")

    out_root = os.path.join("outputs_stochastic", f"p0_{p0}_pv_{pv_p_wp_fixed}")

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
        # fuel_cost = float(value(m.expected_fuel_cost[sc]))

    # --- métriques par scénario (taux satisfaction + PV écrêté)
    # df_met, sat_exp, unserved_exp, pvcurt_exp = scenario_metrics_df(m, step_s=step_s)
    df_met = scenario_metrics_df(m, step_s=step_s)

    # --- métriques d'énergie de charge à partir des CSV de scénario
    load_metrics = {}
    for s in S:  # S = [1..nb_scenarios]
        total_Wh, pct_9_16 = load_energy_metrics_from_csv(scenario_load_files[s - 1], start_h=9, end_h=16)
        load_metrics[int(s)] = {"load_total_Wh": total_Wh, "load_pct_9_16": pct_9_16}

    # Exports par scénario (timeseries + bilans + plots)
    rows_summary = export_scenario_timeseries_and_plots(m, horizon=horizon, results_root="results", pv_installed_W=pv_p_wp_fixed, p0_diesel_W=p0, with_diesel_generator=with_diesel_generator)
    summary_by_s = {d["scenario"]: d for d in rows_summary}

    # Construire une ligne par scénario avec exactement les colonnes voulues
    rows = []
    for _, row in df_met.iterrows():
        sc = int(row["scenario"])
        sm = summary_by_s.get(sc, {})
        rows.append({
            "p0": p0,
            "pv_fixed": pv_p_wp_fixed,
            "total_cost": float(value(m.total_cost[sc])),
            "bat_emax_t0": float(value(m.bat[s1].emax[t0])),
            "pv_wp": float(value(m.pv[s1].p_wp)),
            "scenario": sc,  # <- colonne scénario demandée
            "load_total_Wh": float(load_metrics[sc]["load_total_Wh"]),
            "load_pct_energy_9_16": float(load_metrics[sc]["load_pct_9_16"]),
            "pv_curt_Wh": float(row["pv_curt_Wh"]),  # <- PV écrêté dans le scénario
            "sat_time_pct": float(row["sat_time_%"]),  # <- taux satisfaction (%) dans le scénario
            "unserved_Wh": float(row["unserved_Wh"]),  # énergie non servie
            "capex_total_EUR": capex_total,
            "opex_total_EUR": opex_total,
            "repl_bat_EUR": repl_bat,
            "fuel_cost_EUR": 0.0 if with_diesel_generator == 0 else float(value(m.expected_fuel_cost[sc])),
            "diesel_efficiency_mean_active": float(sm.get("diesel_efficiency_mean_active", np.nan)),
            "battery_capacity_final_pct": float(sm.get("battery_capacity_final_pct", np.nan)),
            "diesel_usage_rate_pct": float(sm.get("diesel_usage_rate_pct", np.nan)),
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
    time_end = "2023-01-24 23:00:00"          # fin de l'horizon temporel
    time_step = "1 hour"                      # on peut mettre "15 min" ou "30 min" aussi
    consumption_satisfaction = 100             # % du temps où la consommation est satisfaite
    UB = 1e10

    allow_consumption_shedding = False            # si True, on doit satisfaire la charge entre shed_hour_start et shed_hour_end
    shedding_hour_start = 0                       # heure de départ autorisation de non satisfaction de la charge
    shedding_hour_end = 4                         # heure de fin autorisation de non satisfaction de la charge

    MIP_GAP = 0.1
    MAX_WORKERS = 4
    GUROBI_THREADS = 5

    pv_p_wp_fixed_list = list(range(43000, 44000, 1000))     # puissance installée en W
    p0_list=[0] if with_diesel_generator==0 else list(range(1000, 51000, 10000))

    jobs = [(p0, pv) for p0 in p0_list for pv in pv_p_wp_fixed_list]

    rows_all = []

    t_start_total = datetime.datetime.now()

    # -----------------------------
    # Fichiers d'entrée (24 jours)
    # -----------------------------
    scenario_load_files = [
                    "microgrid_consumption/Real_consumption/one_year_formatted.csv"
                    # "microgrid_consumption/microgrid_consumption_examples/24_days_s1_1h.csv",
                    # "microgrid_consumption/microgrid_consumption_examples/24_days_s2_1h.csv",
                    # "microgrid_consumption/microgrid_consumption_examples/24_days_s3_1h.csv",
                    # "microgrid_consumption/microgrid_consumption_examples/24_days_s4_1h.csv",
                    # "microgrid_consumption/microgrid_consumption_examples/24_days_s5_1h.csv"
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

    t_end_total = datetime.datetime.now()
    dt_total_script = (t_end_total - t_start_total).total_seconds()
    print(f"Temps total du script : {dt_total_script:,.2f} s")