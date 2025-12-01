from pyomo.environ import *
from pyomo.network import Port

from lms2.tools.data_processing import read_data, load_data
from lms2.core.horizon import SimpleHorizon
from lms2.electric.sources import pv_panel, fixed_power_load, power_source, scalable_power_source
from lms2.tools.post_processing import *

from block_pv import *
from block_battery import *
from battery_factory import make_battery
from functions_constraint import bilan_puissance_rule
import functions_visualisation as visu
from functions_economic import *
from functions_useful import *
from block_diesel_generator import *

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
from tabulate import tabulate
import os
import numpy as np
import pandas as pd
import datetime
import pickle
import os
import logging
from pathlib import Path

UB=10e6 # Upper Bound by default

if __name__ == '__main__':
    with_diesel_generator = 0                 # 0 : without diesel, 1 : diesel_V1, 2 : diesel_V2 (with SOS2)
    battery_model = 3                         # 2 : battery_V2, 3 : battery_V3
    discount_rate = 0.095                     # taux d'actualisation (0.095 pour Madagascar)
    total_duration = 20                       # durée du micro-réseau en années
    battery_replacement_years = [5, 10, 15]   # année de remplacement du pack de batterie

    # p0=1 # puissance nominale du générateur diesel en W
    p0_list = [7000]
    total_cost=[]
    pv_wp=[]
    bat_capa=[]


    for i in range(len(p0_list)):
        m = ConcreteModel()
        horizon = SimpleHorizon(tstart='2023-01-01 00:00:00', tend='2023-01-24 23:45:00', time_step='15 minutes', tz='Indian/Antananarivo')
        m.time = RangeSet(0, horizon.horizon.total_seconds(), horizon.horizon.total_seconds()/(horizon.horizon.total_seconds()/900))

        t1 = datetime.datetime.now()
        irradiance_kw_2019 = read_data(horizon, os.path.join(os.getcwd(), 'meteo_data', 'irradiance_24_days.csv'), usecols=[0, 1],
                              tz_data='Indian/Antananarivo')
        temperature_2019 = read_data(horizon, os.path.join(os.getcwd(), 'meteo_data', 'temperature_24_days.csv'), usecols=[0, 1],
                              tz_data='Indian/Antananarivo')
        charge_2019 = read_data(horizon, os.path.join(os.getcwd(), 'microgrid_consumption', 'scenarios_24_days', '24_days_example_1.csv'), usecols=["timestamp", "aggregate_wh"],  tz_data='Indian/Antananarivo')
        charge_2019_W = 1000 * charge_2019/(horizon.time_step.total_seconds()/3600)
        print('Elapsed time read data: ' + str(datetime.datetime.now() - t1))

        t2 = datetime.datetime.now()
        option_charge   = {'time': m.time}
        option_pv       = {'time': m.time, 'p_wp_min': 1, 'p_wp_max': 6e6, 'cost_inv': 1.5, 'cost_opex': 0.02}
        option_bat      = {'time': m.time, 'dt': horizon.time_step.total_seconds(), 'c_bat_max': 3e6, 'c_bat_min': 1, 'eta_c': 0.9, 'eta_d': 0.85, 'soc_min': 30, 'soc_max': 100, 'soc0': 100, 'cost_inv': 0.12, 'cost_opex': 0.0005}
        option_gen      = {'time': m.time, 'dt': horizon.time_step.total_seconds(), 'p0_min': 1, 'p0_max': 1e6, 'eff': 0.35, 'fuel_cost': 1.5, 'fuel_consumption': 0.00009639, 'cost_inv': 0.7, 'cost_opex': 0.03}
        option_gen_V2   = {'time': m.time, 'dt': horizon.time_step.total_seconds(), 'p0':p0_list[i], 'fuel_cost': 1.2, 'fuel_consumption': 0.00009639, 'cost_inv': 0.7, 'cost_opex': 0.03}

        m.charge        = Block(rule=lambda x: fixed_power_load(x, **option_charge))
        m.pv            = Block(rule=lambda x: block_pv(x, curtailable=True, **option_pv))
        m.bat           = Block(rule=lambda x: make_battery(x, model=battery_model, **option_bat))
        if with_diesel_generator == 1 :
            m.gen = Block(rule=lambda x: diesel_generator(x, **option_gen))
        elif with_diesel_generator == 2 :
            m.gen = Block(rule=lambda x: diesel_generator_V2(x, **option_gen_V2))


        """The power balance constraints are defined in the cases without and with diesel generator."""
        m.bilan_puissance = Constraint(m.time, rule = lambda b, t: bilan_puissance_rule(b, t, with_diesel_generator=with_diesel_generator))

        """The objective functions are defined in the cases without and with diesel generator."""
        m.total_cost = Objective(rule = lambda b: cout_total_rule(b, with_diesel_generator=with_diesel_generator, discount_rate=discount_rate, total_duration=total_duration, replacement_year=battery_replacement_years))

        print('Elapsed time instantiation model: ' + str(datetime.datetime.now() - t2))

        t3 = datetime.datetime.now()
        # Lire le CSV 24 jours (colonnes: timestamp, aggregate_wh)
        df_load = pd.read_csv(os.path.join(os.getcwd(), 'microgrid_consumption/scenarios_24_days/24_days_example_1.csv'), parse_dates=[0])

        # Convertir Wh/15min -> W (puissance moyenne sur le pas) : W = Wh / 0.25h = 4 * Wh
        val_col = 'aggregate_wh' if 'aggregate_wh' in df_load.columns else df_load.columns[1]
        W_vals = (df_load[val_col].values * 4.0).tolist()

        # Sécuriser la longueur = nb de pas de l’horizon
        expected_pts = len([t for t in m.time])
        if len(W_vals) != expected_pts:
            if len(W_vals) > expected_pts:
                W_vals = W_vals[:expected_pts]
            else:
                W_vals = W_vals + [W_vals[-1]] * (expected_pts - len(W_vals))

        # Construire l’index EXACT utilisé par load_data (clés = horizon.map[i])
        time_keys = pd.DatetimeIndex([horizon.map[i] for i in m.time])
        W_series = pd.Series(W_vals, index=time_keys)

        # Charger dans le Param de charge
        load_data(horizon, m.charge.p, W_series)

        load_data(horizon, m.pv.irr, irradiance_kw_2019['Irradiance'])
        load_data(horizon, m.pv.tmp, temperature_2019['Temperature'])
        load_data(horizon, m.bat.tmp, temperature_2019['Temperature'])
        # load_data(horizon, m.charge.p, charge_2019_W['kilowatt_hours'])
        print('Elapsed time load data: ' + str(datetime.datetime.now() - t3))

        t4 = datetime.datetime.now()
        sol = SolverFactory('gurobi', tee=True, solver_io="direct")
        res = sol.solve(m, options={'MIPGap': 0.1})
        print('Elapsed time solve: ' + str(datetime.datetime.now() - t4))
        print('TEMPS TOTAL : ' + str(datetime.datetime.now() - t1))


        """ AFFICHAGE DES RESULTATS """
        if with_diesel_generator ==0:
            print("La puissance installée des panneaux photovoltaïques vaut {} W,\n et la capacité installée du pack batterie vaut {} Wh.".format(value(m.pv.p_wp), value(m.bat.emax0)))

        else:
            print("La puissance installée des panneaux photovoltaïques vaut {} W,\n la capacité installée du pack batterie vaut {} Wh,\n et la puissance du générateur Diesel vaut {} W".format(value(m.pv.p_wp), value(m.bat.emax0), value(m.gen.p0)))
        print("Le coût total est de {} €".format(value(m.total_cost)))

        """On calcule les expressions de tous les coûts qu'on va ensuite vouloir afficher dans le tableau récapitulatif."""
        m.capex_pv = Expression(rule=lambda b: capex_pv_rule(b))
        m.capex_bat = Expression(rule=lambda b: capex_bat_rule(b))
        m.opex_pv = Expression(rule=lambda b: opex_pv_rule(b, discount_rate=discount_rate, total_duration=total_duration))
        m.opex_bat = Expression(rule=lambda b: opex_bat_rule(b, discount_rate=discount_rate, total_duration=total_duration))
        m.repl_bat = Expression(rule=lambda b: repl_bat_rule(b, discount_rate=discount_rate, replacement_year=battery_replacement_years))

        m.energie_totale_consomme = Expression(rule=lambda b: energie_totale_consomme_rule(b, horizon=horizon))


        if with_diesel_generator != 0:
            @m.Expression()
            def capex_gen(b):
                return b.gen.cost_inv * b.gen.p0.value

            @m.Expression()
            def opex_gen(b):
                return b.gen.cost_opex * b.gen.p0.value * 9.64955841794

            @m.Expression()
            def cost_total_fuel(b):
                return sum(b.gen.e_th[t] * b.gen.fuel_cost * b.gen.fuel_consumption for t in b.time) * 9.64955841794

        print("L'énergie totale consommée vaut {} Wh".format(value(m.energie_totale_consomme)))
        lcoe_value = value(value(m.total_cost)) / value(value(m.energie_totale_consomme))
        print("LCOE : {:.8f} €/kWh".format(lcoe_value))

        """On visualise les coûts détaillés dans un tableau"""
        visu.cost_table(m, with_diesel_generator=with_diesel_generator)

        total_cost.append(value(m.total_cost))
        pv_wp.append(value(m.pv.p_wp))
        bat_capa.append(value(m.bat.emax0))

        # Visualisation des résultats
        visu.plot_results_deterministic(m, horizon, with_diesel_generator=with_diesel_generator)

