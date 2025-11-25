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
from functions_economic import *
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
        horizon = SimpleHorizon(tstart='2023-01-01 04:00:00', tend='2023-01-24 20:45:00', time_step='15 minutes', tz='Indian/Antananarivo')
        m.time = RangeSet(0, horizon.horizon.total_seconds(), horizon.horizon.total_seconds()/(horizon.horizon.total_seconds()/900))

        t1 = datetime.datetime.now()
        irradiance_kw_2019 = read_data(horizon, os.path.join(os.getcwd(), 'meteo_data', 'irradiance_solcast_formatted.csv'), usecols=[1, 2],
                              tz_data='Indian/Antananarivo')
        temperature_2019 = read_data(horizon, os.path.join(os.getcwd(), 'meteo_data', 'temperature_solcast_formatted.csv'), usecols=[1, 2],
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
        m.total_cost = Objective(rule = lambda b: cout_total_rule(b, with_diesel_generator=with_diesel_generator))

        print('Elapsed time instantiation model: ' + str(datetime.datetime.now() - t2))

        t3 = datetime.datetime.now()
        # Lire le CSV 24 jours (colonnes: timestamp, aggregate_wh)
        df_load = pd.read_csv(os.path.join(os.getcwd(), 'microgrid_consumption', 'scenarios_24_days', '24_days_example_1.csv'),
                              parse_dates=[0])

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
            print("La puissance installée des panneaux photovoltaïques vaut {} W,\n la capacité installée du pack batterie vaut {} Wh,\n et la puissance du générateur Diesel vaut {} W".format(m.pv.p_wp.value, m.bat.emax0, m.gen.p0.value))
        print("Le coût total est de {} €".format(value(m.total_cost)))

        """On calcule les expressions de tous les coûts qu'on va ensuite vouloir afficher dans le tableau récapitulatif."""
        m.capex_pv = Expression(rule=lambda b: capex_pv_rule(b))
        m.capex_bat = Expression(rule=lambda b: capex_bat_rule(b))
        m.opex_pv = Expression(rule=lambda b: opex_pv_rule(b, discount_rate=discount_rate, total_duration=total_duration))
        m.opex_bat = Expression(rule=lambda b: opex_bat_rule(b, discount_rate=discount_rate, total_duration=total_duration))
        m.repl_bat = Expression(rule=lambda b: repl_bat_rule(b, discount_rate=discount_rate, replacement_year=battery_replacement_years))

        @m.Expression()
        def energie_totale_consomme(b):
            return (sum(b.charge.p[t] for t in b.time)
                    * horizon.time_step.total_seconds() / 3600
                    * 20)  # en Wh

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

        print("L'énergie totale consommée vaut {} Wh".format(m.energie_totale_consomme()))
        lcoe_value = value(value(m.total_cost)) / value(m.energie_totale_consomme())
        print("LCOE : {:.8f} €/kWh".format(lcoe_value))

        """Cost details without and with diesel generation."""
        if with_diesel_generator == 0:
            cost_data = {
                "coût d'investissement (€)": {
                    "solar panel": m.capex_pv(),
                    "batteries": m.capex_bat(),
                    "diesel generator": 0,
                    "total": m.capex_pv() + m.capex_bat()
                },
                "coût d'opération (€)": {
                    "solar panel": m.opex_pv(),
                    "batteries": m.opex_bat(),
                    "diesel generator": 0,
                    "total": m.opex_pv() + m.opex_bat()
                },
                "coût de remplacement (€)": {
                    "solar panel": 0,
                    "batteries": m.repl_bat(),
                    "diesel generator": 0,
                    "total": m.repl_bat()
                },
                "coût total (€)": {
                    "solar panel": m.capex_pv() + m.opex_pv(),
                    "batteries": m.capex_bat() + m.opex_bat() + m.repl_bat(),
                    "diesel generator": 0,
                    "total": m.total_cost()
                }
            }
        else:
            cost_data = {
                "coût d'investissement (€)": {
                    "solar panel": m.capex_pv(),
                    "batteries": m.capex_bat(),
                    "diesel generator": m.capex_gen(),
                    "total": m.capex_pv() + m.capex_bat() + m.capex_gen()
                },
                "coût d'opération (€)": {
                    "solar panel": m.opex_pv(),
                    "batteries": m.opex_bat(),
                    "diesel generator": m.opex_gen() + m.cost_total_fuel(),
                    "total": m.opex_pv() + m.opex_bat() + m.opex_gen() + m.cost_total_fuel()
                },
                "coût de remplacement (€)": {
                    "solar panel": 0,
                    "batteries": m.repl_bat(),
                    "diesel generator": 0,
                    "total": m.repl_bat()
                },
                "coût total (€)": {
                    "solar panel": m.capex_pv() + m.opex_pv(),
                    "batteries": m.capex_bat() + m.opex_bat() + m.repl_bat(),
                    "diesel generator": m.capex_gen() + m.opex_gen() + m.cost_total_fuel(),
                    "total": m.total_cost()
                }
            }

        # Conversion en DataFrame
        df_costs = pd.DataFrame.from_dict(cost_data, orient='index')

        # Affichage du tableau avec tabulate
        print(tabulate(df_costs, headers='keys', tablefmt='grid'))
        if with_diesel_generator != 0:
            print("coût du diesel consommé : {}€".format(m.cost_total_fuel()))

        total_cost.append(value(m.total_cost()))
        pv_wp.append(value(m.pv.p_wp))
        bat_capa.append(value(m.bat.emax0))

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

        # pplot(m.pv.p_curt, ax=ax[1][0], title='Puissance PV perdue', fig=fig, index=index_jours,
        #       bbox_to_anchor=(0, -0.2, 1, 0.2), ylabel='Puissance (W)')

        # if with_diesel_generator==0:
        #     pplot(m.pv.p, ax=ax[1][1], title='Puissance PV', fig=fig, index=index_jours,
        #           bbox_to_anchor=(0, -0.2, 1, 0.2), ylabel='Puissance (W)')
        # pplot(m.gen.p, ax=ax[2], fig=fig, index=index_jours,
        #       bbox_to_anchor=(0, -0.2, 1, 0.2), ylabel='Diesel power (W)')


        # --- Adapter la taille de police des titres et des ticks ---
        for axis in ax:
            axis.set_xlabel(axis.get_xlabel(), fontsize=17)
            axis.set_ylabel(axis.get_ylabel(), fontsize=17)
            axis.tick_params(axis='both', labelsize=15)

        # --- Création du dossier s'il n'existe pas ---
        os.makedirs('results_image', exist_ok=True)

        # --- Définir le nom du fichier ---
        if with_diesel_generator == 0:
            diesel_power = 0
        else:
            diesel_power = p0_list[i]

        filename = f"results_image/temporal_data_diesel_{diesel_power}W.pickle"

        # --- Sauvegarder la figure ---
        with open(filename, 'wb') as f:
            pickle.dump(fig, f)
        plt.show()

    # fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 10), sharex=True)
    #
    # # 1. Coût total
    # axs[0].plot(p0_list, total_cost, marker='o')
    # axs[0].set_ylabel("Coût total (€)")
    #
    # # 2. Puissance PV installée
    # axs[1].plot(p0_list, pv_wp, marker='o')
    # axs[1].set_ylabel("Puissance PV (W)")
    #
    # # 3. Capacité batterie installée
    # axs[2].plot(p0_list, bat_capa, marker='o')
    # axs[2].set_ylabel("Capacité batterie (Wh)")
    # axs[2].set_xlabel(r'$P_{\mathrm{gen,max}}$ [W]')
    #
    # # Améliorations visuelles
    # for ax in axs:
    #     ax.grid(True)
    #
    # plt.tight_layout()
    # plt.show()