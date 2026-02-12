from pyomo.environ import *
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import os

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


# ============================================================
# 1) RUN_ONE_CASE
# >>> COPIÉ STRICTEMENT DU main.py <<<
# ============================================================

def run_one_case(
    p0,
    pv_p_wp_fixed,
    scenario_load_files,
    with_diesel_generator,
    battery_model,
    discount_rate,
    total_duration,
    battery_replacement_years,
    time_start,
    time_end,
    consumption_satisfaction,
    shed_hour_start,
    shed_hour_end,
    MIP_GAP,
    gurobi_threads):

    out_root = "outputs_stochastic"
    ensure_dir(out_root)

    # S = list(range(1, len(scenario_load_files) + 1))
    # prob = {s: 1 / len(S) for s in S}

    # horizon = SimpleHorizon(
    #     tstart=time_start,
    #     tend=time_end,
    #     time_step="15 minutes",
    #     tz="Indian/Antananarivo"
    # )

    # step_s = int(horizon.time_step.total_seconds())
    # T = int(horizon.horizon.total_seconds())

    # m = ConcreteModel()
    # m.time = RangeSet(0, T, step_s)
    # t0 = m.time.first()

    # def allow_shed_init(m, t):
    #     dt = horizon.map[t]
    #     return 1 if (shed_hour_start <= dt.hour < shed_hour_end) else 0
    #
    # m.allow_shed = Param(m.time, initialize=allow_shed_init, within=Binary)

    # option_pv = {
    #     "time": m.time,
    #     "p_wp_fixed": pv_p_wp_fixed,
    #     "cost_inv": 1.5,
    #     "cost_opex": 0.02,
    # }
    #
    # option_bat = {
    #     "time": m.time,
    #     "dt": step_s,
    #     "c_bat_max": 1e6,
    #     "c_bat_min": 1,
    #     "eta_c": 0.90,
    #     "eta_d": 0.85,
    #     "soc_min": 30,
    #     "soc_max": 100,
    #     "soc_allow_curt": 80,
    #     "soc0": 70,
    #     "cost_inv": 0.12,
    #     "cost_opex": 0.0005,
    # }

    # option_consumption = {"time": m.time}

    # option_gen_V2 = {
    #     "time": m.time,
    #     "dt": step_s,
    #     "p0": p0,
    #     "fuel_cost": 1.2,
    #     "fuel_consumption": 0.00009639,
    #     "cost_inv": 0.7,
    #     "cost_opex": 0.03,
    # }

    # m.S = Set(initialize=S)
    # m.pv = Block(m.S)
    # m.bat = Block(m.S)
    # m.consumption = Block(m.S)
    # m.p_unserved = Var(m.S, m.time, domain=NonNegativeReals)
    #
    # if with_diesel_generator != 0:
    #     m.gen = Block(m.S)

    # for s in m.S:
    #     block_pv(m.pv[s], curtailable=True, **option_pv)
    #     make_battery(m.bat[s], model=battery_model, **option_bat)
    #     fixed_power_load(m.consumption[s], **option_consumption)
    #     if with_diesel_generator == 2:
    #         diesel_generator_V2(m.gen[s], **option_gen_V2)

    # s1 = S[0]
    # m.same_bat = Constraint(m.S, rule=lambda b, s: same_bat_rule(b, s, s_ref=s1, t0=t0))
    #
    # m.bilan_puissance = Constraint(
    #     m.S, m.time,
    #     rule=lambda b, s, t: bilan_with_shed_rule(b, s, t, with_diesel_generator)
    # )

    # M_curt, M_energy = pv_curt_bigM_rule(m, s_ref=s1, M_curt=1e6)
    #
    # m.is_full = Var(m.S, m.time, domain=Binary)
    # m.full_lower = Constraint(m.S, m.time, rule=lambda b, s, t: full_lower_rule(b, s, t, s_ref=s1, M_energy=M_energy))
    # m.full_upper = Constraint(m.S, m.time, rule=lambda b, s, t: full_upper_rule(b, s, t, s_ref=s1, M_energy=M_energy))
    # m.curtail_only_if_full = Constraint(m.S, m.time, rule=lambda b, s, t: curtail_only_if_full_rule(b, s, t, M_curt))
    #
    # m.is_served = Var(m.S, m.time, domain=Binary)
    # m.served_link = Constraint(m.S, m.time, rule=lambda b, s, t: served_link_rule(b, s, t, M_shed=1e9))
    # min_served = min_served_steps(m.time, consumption_satisfaction)
    # m.consumption_satisfaction = Constraint(m.S, rule=lambda b, s: satisfaction_rule_per_scenario(b, s, min_served))

    # irr = read_data(
    #     horizon,
    #     "meteo_data/irradiance_24_days_30min.csv",
    #     usecols=["Time", "Irradiance"],
    #     tz_data="Indian/Antananarivo"
    # )
    # tmp = read_data(
    #     horizon,
    #     "meteo_data/temperature_24_days_30min.csv",
    #     usecols=["Time", "Temperature"],
    #     tz_data="Indian/Antananarivo"
    # )

    # for s in m.S:
    #     load_data(horizon, m.pv[s].irr, irr["Irradiance"])
    #     load_data(horizon, m.pv[s].tmp, tmp["Temperature"])
    #     if hasattr(m.bat[s], "tmp"):
    #         load_data(horizon, m.bat[s].tmp, tmp["Temperature"])

        # W_vals = read_load_as_W(scenario_load_files[s - 1])

        # expected_pts = len(list(m.time))
        # if len(W_vals) != expected_pts:
        #     if len(W_vals) > expected_pts:
        #         W_vals = W_vals[:expected_pts]
        #     else:
        #         W_vals = W_vals + [W_vals[-1]] * (expected_pts - len(W_vals))

        # time_keys = pd.DatetimeIndex([horizon.map[i] for i in m.time])
        # load_data(horizon, m.consumption[s].p, pd.Series(W_vals, index=time_keys))

    # m.total_cost = Objective(
    #     rule=lambda b: total_cost_rule(
    #         b,
    #         prob=prob,
    #         with_diesel_generator=with_diesel_generator,
    #         discount_rate=discount_rate,
    #         total_duration=total_duration,
    #         replacement_year=battery_replacement_years
    #     )
    # )

    # solver = SolverFactory("gurobi", solver_io="direct")
    # solver.solve(
    #     m,
    #     tee=True,
    #     options={
    #         "MIPGap": MIP_GAP,
    #         "Threads": gurobi_threads
    #     }
    # )

    return {
        "p0": p0,
        "pv": pv_p_wp_fixed,
        "total_cost": float(value(m.total_cost)),
        "bat_emax_t0": float(value(m.bat[s1].emax[t0])),
    }


# ============================================================
# 2) MAIN PARALLÉLISÉ
# ============================================================

if __name__ == "__main__":

    with_diesel_generator = 2
    battery_model = 3
    discount_rate = 0.095
    total_duration = 20
    battery_replacement_years = (5, 10, 15)
    time_start = "2023-01-01 00:00:00"
    time_end = "2023-01-24 23:45:00"
    consumption_satisfaction = 90
    shed_hour_start = 0
    shed_hour_end = 4
    MIP_GAP = 0.8

    MAX_WORKERS = 4
    GUROBI_THREADS = 5

    p0_list = [2000, 4000]
    pv_p_wp_fixed_list = [5000, 7000, 9000]

    scenario_load_files = [
        "microgrid_consumption/scenarios_24_days/24_days_example_1.csv"
    ]

    jobs = [(p0, pv) for p0 in p0_list for pv in pv_p_wp_fixed_list]
    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(
                run_one_case,
                p0, pv,
                scenario_load_files,
                with_diesel_generator,
                battery_model,
                discount_rate,
                total_duration,
                battery_replacement_years,
                time_start,
                time_end,
                consumption_satisfaction,
                shed_hour_start,
                shed_hour_end,
                MIP_GAP,
                GUROBI_THREADS
            ): (p0, pv)
            for p0, pv in jobs
        }

        for fut in as_completed(futures):
            results.append(fut.result())

    print("✔ Parallélisation terminée")
