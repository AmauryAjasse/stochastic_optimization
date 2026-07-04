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

def manuscript_results_table(m, s, t0, step_s, consumption_satisfaction, with_diesel_generator=0,
                             co2_pv_kg_per_W=0.5,
                             co2_bat_kg_per_Wh=0.065,
                             co2_gen_kg_per_W=0.0,
                             co2_diesel_kg_per_kWh=267e-6,
                             nb_menages=54):
    """
    Calcule les indicateurs du tableau du manuscrit pour un scénario s.
    Les CO2 sont à renseigner selon tes références bibliographiques.
    """

    dt_h = step_s / 3600.0

    pv_installed_W = value(m.pv[s].p_wp)
    bat_capacity_Wh = value(m.bat[s].emax[t0])
    lcc_total = value(m.total_cost[s])

    load_Wh = sum(value(m.consumption[s].p[t]) * dt_h for t in m.time)
    pv_used_Wh = sum(value(m.pv[s].p[t]) * dt_h for t in m.time)

    unserved_Wh = sum(value(m.p_unserved[s, t]) * dt_h for t in m.time)

    n_heures_demandees = len(list(m.time))
    n_heures_fournies = sum(1 for t in m.time if value(m.p_unserved[s, t]) <= 1e-6)

    asai_impose = consumption_satisfaction / 100.0
    asai_reel = n_heures_fournies / n_heures_demandees
    part_energie_satisfaite = (load_Wh - unserved_Wh) / load_Wh if load_Wh > 0 else float("nan")

    pv_curt_Wh = 0.0
    if hasattr(m.pv[s], "p_curt"):
        pv_curt_Wh = sum(value(m.pv[s].p_curt[t]) * dt_h for t in m.time)

    pv_available_Wh = pv_used_Wh + pv_curt_Wh

    if with_diesel_generator == 2:
        gen_installed_W = value(m.p0_selected)
        gen_Wh = sum(
            value(m.gen[s, k].p[t]) * dt_h
            for k in m.KGEN for t in m.time
        )
        gen_hours_per_day = sum(
            1 for t in m.time
            if sum(value(m.gen[s, k].p[t]) for k in m.KGEN) > 1e-6
        ) * dt_h / len(list(m.DAYS))
    else:
        gen_installed_W = None
        gen_Wh = 0.0
        gen_hours_per_day = None

    lcoe = lcc_total / ((load_Wh / 1000.0) * 20*365/24) if load_Wh > 0 else float("nan")

    r_pv = pv_used_Wh / load_Wh if load_Wh > 0 else float("nan")
    r_gen = gen_Wh / load_Wh if load_Wh > 0 else float("nan")
    r_curt = pv_curt_Wh / pv_available_Wh if pv_available_Wh > 0 else 0.0

    co2_total = (
        co2_pv_kg_per_W * pv_installed_W
        + co2_bat_kg_per_Wh * bat_capacity_Wh * 2
        + (0.0 if gen_installed_W is None else co2_gen_kg_per_W * gen_installed_W)
    )

    gen_kWh = gen_Wh / 1000.0
    co2_cons = co2_diesel_kg_per_kWh * gen_kWh

    co2_cons_hab = co2_total / (nb_menages * 4.4 * 20)

    return {
        "P_pv_p_W": pv_installed_W,
        "E_bat_max_Wh": bat_capacity_Wh,
        "P_gen_0_W": gen_installed_W,
        "L_gen_jour_h_per_day": gen_hours_per_day,
        "LCC_total_EUR": lcc_total,
        "LCOE_EUR_per_kWh": lcoe,
        "R_PV": r_pv,
        "R_gen": r_gen,
        "R_curt": r_curt,
        "CO2_total_kg": co2_total,
        "CO2_cons_kg": co2_cons,
        "CO2_cons_kg_per_hab": co2_cons_hab,
        "ASAI_impose": asai_impose,
        "ASAI_reel": asai_reel,
        "Part_energie_satisfaite": part_energie_satisfaite,
    }

def run_one_case(
        p0_list: list[int],
        pv_p_wp_list: list[int],
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
    option_pv = {"time": m.time, "p_wp_min": min(pv_p_wp_list), "p_wp_max": max(pv_p_wp_list), "cost_inv": 1.0, "cost_opex": 0.009}
    option_bat = {"time": m.time, "dt": step_s, "c_bat_max": 1e9, "c_bat_min": 1, "eta_c": 0.93,
                  "eta_d": 0.92, "soc_min": 30, "soc_max": 100, "soc_allow_curt": 80, "soc0": 70, "socf": None, "c_rate_max": 0.25,
                  "cost_inv": 0.12, "cost_opex": 0.0005}
    option_consumption = {"time": m.time}  # pour fixed_power_load
    option_gen = {"time": m.time, "dt": step_s, "eff": 0.35, "fuel_cost": 1.2,
                  "fuel_consumption": 0.00009639, "cost_inv": 0.7, "cost_opex": 0.03}
    option_gen_V2 = {'time': m.time, 'dt': step_s, 'fuel_cost': 1.2,
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

    # if with_diesel_generator != 0:
    #     m.gen = Block(m.S)
    if with_diesel_generator == 2:
        m.KGEN = RangeSet(0, len(p0_list) - 1)
        m.y_gen = Var(m.KGEN, domain=Binary)
        m.one_gen_choice = Constraint(expr=sum(m.y_gen[k] for k in m.KGEN) == 1)
        m.gen = Block(m.S, m.KGEN)

    # -----------------------------
    # Choix discret de la puissance PV
    # -----------------------------
    m.KPV = RangeSet(0, len(pv_p_wp_list) - 1)
    m.y_pv = Var(m.KPV, domain=Binary)
    m.one_pv_choice = Constraint(expr=sum(m.y_pv[k] for k in m.KPV) == 1)
    m.pv_selected = Expression(expr=sum(pv_p_wp_list[k] * m.y_pv[k] for k in m.KPV))
    # m.force_discrete_pv = Constraint(expr=m.pv[s].p_wp == m.pv_selected)

    for s in m.S:
        block_pv(m.pv[s], curtailable=True, **option_pv)
        make_battery(m.bat[s], model=battery_model, **option_bat)
        fixed_power_load(m.consumption[s], **option_consumption)

        m.pv[s].force_discrete_pv = Constraint(expr=m.pv[s].p_wp == m.pv_selected)

        if with_diesel_generator == 2:
            for k in m.KGEN:
                option_gen_V2_k = dict(option_gen_V2)
                option_gen_V2_k["p0"] = p0_list[k]
                diesel_generator_V2(m.gen[s, k], **option_gen_V2_k)

    # -----------------------------
    # Non-anticipation (investissements communs)
    # -----------------------------
    s1 = S[0]
    if m.pv[s1].p_wp.is_variable_type():
        m.same_pv = Constraint(m.S, rule=lambda b, s: same_pv_rule(b, s, s_ref=s1))
    m.same_bat = Constraint(m.S, rule=lambda b, s: same_bat_rule(b, s, s_ref=s1, t0=t0))
    # if with_diesel_generator == 1:
    #     m.same_gen = Constraint(m.S, rule=lambda b, s: same_gen_rule(b, s, s_ref=s1))

    # -----------------------------
    # Contraintes d'optimisation (bilan de puissance, gestion d'énergie)
    # -----------------------------

    # Bilan de puissance
    # m.bilan_puissance = Constraint(m.S, m.time, rule=lambda b, s, t: bilan_with_shed_rule(b, s, t,
    #                                                                                       with_diesel_generator=with_diesel_generator))
    if with_diesel_generator == 0:
        m.bilan_puissance = Constraint(
            m.S, m.time,
            rule=lambda b, s, t:
            b.bat[s].p[t] + b.pv[s].p[t] + b.p_unserved[s, t] == b.consumption[s].p[t]
        )

    elif with_diesel_generator == 2:
        m.bilan_puissance = Constraint(
            m.S, m.time,
            rule=lambda b, s, t:
            b.bat[s].p[t]
            + b.pv[s].p[t]
            + sum(b.gen[s, k].p[t] for k in b.KGEN)
            + b.p_unserved[s, t]
            == b.consumption[s].p[t]
        )

    # Activation d'un seul générateur diesel
    if with_diesel_generator == 2:
        m.gen_activation = Constraint(m.S, m.KGEN, m.time, rule=lambda b, s, k, t: b.gen[s, k].p[t] <= p0_list[k] * b.y_gen[k])

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
    # if with_diesel_generator != 0:
    #     m.capex_gen = Expression(rule=lambda b: capex_gen_rule(b))
    #     m.opex_gen = Expression(
    #         rule=lambda b: opex_gen_rule(b, discount_rate=discount_rate, total_duration=total_duration))
    #     m.expected_fuel_cost = Expression(m.S, rule=lambda b, s: expected_fuel_cost_rule(b, s, discount_rate=discount_rate, total_duration=total_duration))
    if with_diesel_generator == 2:
        annuity_factor = sum(1 / (1 + discount_rate) ** i for i in range(0, total_duration))

        m.p0_selected = Expression(expr=sum(p0_list[k] * m.y_gen[k] for k in m.KGEN))

        m.capex_gen = Expression(expr=0.7 * m.p0_selected)

        m.opex_gen = Expression(expr=0.03 * m.p0_selected * annuity_factor)

        m.expected_fuel_cost = Expression(m.S, rule=lambda b, s:
            sum(
                b.gen[s, k].fuel_cost
                * b.gen[s, k].fuel_consumption
                * b.gen[s, k].e_th[t]
                * annuity_factor
                for k in b.KGEN for t in b.time
            )
        )

    # m.total_cost = Expression(m.S,
    #     rule=lambda b, s: total_cost_rule(b, s, with_diesel_generator=with_diesel_generator,
    #                                    discount_rate=discount_rate, total_duration=total_duration,
    #                                    replacement_year=battery_replacement_years))
    if with_diesel_generator == 0:
        m.total_cost = Expression(
            m.S,
            rule=lambda b, s:
            b.capex_pv + b.capex_bat
            + b.opex_pv + b.opex_bat
            + b.repl_bat
        )

    elif with_diesel_generator == 2:
        m.total_cost = Expression(
            m.S,
            rule=lambda b, s:
            b.capex_pv + b.capex_bat + b.capex_gen
            + b.opex_pv + b.opex_bat + b.opex_gen
            + b.repl_bat
            + b.expected_fuel_cost[s]
        )

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
    print(f"Temps de résolution : {dt_solve:,.2f} s")

    # out_root = os.path.join("outputs_stochastic", f"p0_{p0}_pv_{pv_p_wp_fixed}")

    # =========================
    #  EXPORTS / BILANS / PLOTS
    # =========================

    # On affiche un message seulement si l'optimisation ne fonctionne pas
    tc = res.solver.termination_condition
    if tc not in (TC.optimal, TC.feasible):
        return {"status": str(tc)}

    print("\n===== Solution optimale =====")
    print(f"Puissance PV retenue : {value(m.pv_selected):.0f} W")
    print(f"Capacité batterie    : {value(m.bat[s1].emax[t0]):.0f} Wh")
    print(f"Coût total           : {value(m.obj):.2f} €")
    print("\nChoix PV :")
    for k in m.KPV:
        if value(m.y_pv[k]) > 0.5:
            print(f"  indice {k} -> {pv_p_wp_list[k]} W")
    if with_diesel_generator == 2:
        print(f"Puissance diesel retenue : {value(m.p0_selected):.0f} W")
        print("\nChoix diesel :")
        for k in m.KGEN:
            if value(m.y_gen[k]) > 0.5:
                print(f"  indice {k} -> {p0_list[k]} W")

    table_results = manuscript_results_table(
        m=m,
        s=s1,
        t0=t0,
        step_s=step_s,
        consumption_satisfaction=consumption_satisfaction,
        with_diesel_generator=with_diesel_generator,
        co2_pv_kg_per_W=0.5,
        co2_bat_kg_per_Wh=0.065,
        co2_gen_kg_per_W=0.0,
        co2_diesel_kg_per_kWh=267e-6
    )

    print("\n===== Tableau manuscrit =====")
    for key, val in table_results.items():
        if val is None:
            print(f"{key:25s}: N.A.")
        else:
            print(f"{key:25s}: {val:.4f}")


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
    # rows_summary = export_scenario_timeseries_and_plots(m, horizon=horizon, results_root="results", pv_installed_W=pv_p_wp_fixed, p0_diesel_W=p0, with_diesel_generator=with_diesel_generator)
    pv_opt = float(value(m.pv[s1].p_wp))
    p0_opt = 0.0 if with_diesel_generator == 0 else float(value(m.p0_selected))

    rows_summary = export_scenario_timeseries_and_plots(m, horizon=horizon, results_root="results", pv_installed_W=pv_opt, p0_diesel_W=p0_opt, with_diesel_generator=with_diesel_generator)
    summary_by_s = {d["scenario"]: d for d in rows_summary}

    # Construire une ligne par scénario avec exactement les colonnes voulues
    rows = []
    for _, row in df_met.iterrows():
        sc = int(row["scenario"])
        sm = summary_by_s.get(sc, {})
        rows.append({
            "p0": p0_opt,
            "pv_fixed": pv_opt,
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
            "P_pv_p_W": table_results["P_pv_p_W"],
            "E_bat_max_Wh": table_results["E_bat_max_Wh"],
            "P_gen_0_W": table_results["P_gen_0_W"],
            "L_gen_jour_h_per_day": table_results["L_gen_jour_h_per_day"],
            "LCC_total_EUR": table_results["LCC_total_EUR"],
            "LCOE_EUR_per_kWh": table_results["LCOE_EUR_per_kWh"],
            "R_PV": table_results["R_PV"],
            "R_gen": table_results["R_gen"],
            "R_curt": table_results["R_curt"],
            "CO2_total_kg": table_results["CO2_total_kg"],
            "CO2_cons_kg": table_results["CO2_cons_kg"],
            "ASAI_impose": table_results["ASAI_impose"],
            "ASAI_reel": table_results["ASAI_reel"],
            "Part_energie_satisfaite": table_results["Part_energie_satisfaite"],
        })

    return {"rows": rows}




if __name__ == "__main__":
    # -----------------------------
    # Paramètres d'entrée globaux
    # -----------------------------
    with_diesel_generator = 2                 # 0 : without diesel, 1 : diesel_V1, 2 : diesel_V2 (with SOS2)
    battery_model = 3                         # 2 : battery_V2, 3 : battery_V3
    discount_rate = 0.095                     # taux d'actualisation (0.095 pour Madagascar)
    total_duration = 20                       # durée du micro-réseau en années
    battery_replacement_years = (10,)          # année de remplacement du pack de batterie
    time_start = "2023-02-01 00:00:00"        # début de l'horizon temporel
    time_end = "2023-02-24 23:00:00"          # fin de l'horizon temporel
    time_step = "1 hour"                      # on peut mettre "15 min" ou "30 min" aussi
    consumption_satisfaction = 90             # % du temps où la consommation est satisfaite
    UB = 1e10

    allow_consumption_shedding = False            # si True, on doit satisfaire la charge entre shed_hour_start et shed_hour_end
    shedding_hour_start = 0                       # heure de départ autorisation de non satisfaction de la charge
    shedding_hour_end = 4                         # heure de fin autorisation de non satisfaction de la charge

    MIP_GAP = 0.9
    MAX_WORKERS = 4
    GUROBI_THREADS = 5

    # pv_p_wp_fixed_list = list(range(100000, 101000, 10000))     # puissance installée en W
    # p0_list=[0] if with_diesel_generator==0 else list(range(1000, 51000, 10000))
    #
    # jobs = [(p0, pv) for p0 in p0_list for pv in pv_p_wp_fixed_list]
    pv_p_wp_list = list(range(1000, 2001000, 1000))
    p0_list = [0] if with_diesel_generator == 0 else list(range(1000, 51000, 1000))

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

    # irr_file = os.path.join("meteo_data", "irradiance_24_days_1h.csv")     # cols: timestamp, Irradiance
    # tmp_file = os.path.join("meteo_data", "temperature_24_days_1h.csv")    # cols: timestamp, Temperature
    irr_file = os.path.join("meteo_data", "irradiance_2023_1h.csv")
    tmp_file = os.path.join("meteo_data", "temperature_2023_1h.csv")
    # with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
    #     futures = [
    #         ex.submit(
    #             run_one_case,
    #             p0, pv,
    #             scenario_load_files,
    #             irr_file, tmp_file,
    #             with_diesel_generator,
    #             battery_model,
    #             discount_rate,
    #             total_duration,
    #             battery_replacement_years,
    #             time_start, time_end, time_step,
    #             consumption_satisfaction,
    #             allow_consumption_shedding,
    #             shedding_hour_start, shedding_hour_end,
    #             MIP_GAP,
    #             GUROBI_THREADS
    #         )
    #         for (p0, pv) in jobs
    #     ]
    #
    #     for fut in as_completed(futures):
    #         r = fut.result()
    #         if r.get("status", "ok") != "ok":
    #             print(f"[FAIL] pv_fixed={r['pv_fixed']}W -> {r['status']}")
    #             continue
    #         # print(f"[DONE] pv_fixed={r['pv_fixed']}W -> total_cost={r['total_cost']:.2f} € | out={r['out_dir']}")
    #
    #         rows_all.extend(r["rows"])

    r = run_one_case(
        p0_list,
        pv_p_wp_list,
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

    if r.get("status", "ok") != "ok":
        print(f"[FAIL] -> {r['status']}")
    else:
        rows_all.extend(r["rows"])

    save_results_in_tab(with_diesel_generator, rows_all, pv_p_wp_list)

    t_end_total = datetime.datetime.now()
    dt_total_script = (t_end_total - t_start_total).total_seconds()
    print(f"Temps total du script : {dt_total_script:,.2f} s")