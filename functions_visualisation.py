import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd

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