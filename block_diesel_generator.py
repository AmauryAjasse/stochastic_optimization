from pyomo.environ import *
from pyomo.core import NonNegativeReals, Binary, PositiveReals, Reals, Any
from pyomo.network import Port
from pyomo.environ import Piecewise

from lms2.tools.data_processing import read_data, load_data
from lms2.core.horizon import SimpleHorizon
from lms2.electric.sources import pv_panel, fixed_power_load, power_source, scalable_power_source
from lms2.tools.post_processing import *

from block_pv import *
from block_battery import *

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import os
import numpy as np
import datetime
import logging

UB = 10e6 # Upper Bound by default


def diesel_generator(gen, **options):
    """
        Modèle de générateur diesel.

        Instanciation options:
            - p_max : Puissance maximale du générateur (kW)
            - p_min : Puissance minimale du générateur (kW)
            - eff : Rendement du générateur (typiquement 30-40%)
            - fuel_cost : Coût du carburant par litre (€/L)
            - fuel_consumption : Consommation spécifique (L/kWh)
            - cost_inv : Coût d'investissement (€/kW)
            - cost_opex : Coût d'exploitation (€/kW par an)

        Variables:
            - p : Puissance fournie par le générateur (kW)
            - fuel_used : Carburant consommé (L/h)
    """

    time                = options.get('time', RangeSet(0, 1))
    dt                  = options.get('dt', 1.0)

    p0_max              = options.get('p0_max', 1e8)
    p0_min              = options.get('p0_min', 1)
    eff                 = options.get('eff', 0.35)
    fuel_cost           = options.get('fuel_cost', 1.2)                    # €/L
    fuel_consumption    = options.get('fuel_consumption', 0.00025)         # L/Wh
    cost_inv            = options.get('cost_inv', 0.2)                     # €/W
    cost_opex           = options.get('cost_opex', 0.01)                   # €/W/an

    gen.p0              = Var(initialize=p0_min, within=Reals, bounds=(p0_min, p0_max))
    gen.e_th            = Var(time, within=Reals, initialize=0)
    gen.p               = Var(time, initialize=0, within=NonNegativeReals, bounds=(0, p0_max))
    gen.u               = Var(time, initialize=0, within=Binary)

    gen.eff              = Param(default=eff, doc='Efficacité du générateur diesel')
    gen.fuel_cost        = Param(default=fuel_cost, doc='Coût du diesel en €/L')
    gen.fuel_consumption = Param(default=fuel_consumption, doc='Diesel consommé pour produire de l\'énergie en L/Wh')
    gen.cost_inv         = Param(default=cost_inv, doc='Coût d\'investissement d\'un générateur diesel en €/Wh')
    gen.cost_opex        = Param(default=cost_opex, doc='Coût d\'opération d\'un générateur diesel en €/Wh')

    @gen.Constraint(time)
    def power_relation1(g, t):
        return g.p[t] <= gen.p0
        # return g.p[t] ==0
    @gen.Constraint(time)
    def fuel_usage_constraint(g, t):
        return g.e_th[t] * g.eff == g.p[t] * (dt / 3600)

    return gen


def diesel_generator_V2(gen, **options):
    """
    Modèle de générateur diesel avec rendement interpolé par morceaux,
    utilisant Piecewise + SOS2.

    Variables :
        - p : puissance délivrée (W)
        - eta : rendement en fonction de p/p0
        - fuel_used : carburant consommé (L/h)
    """

    # === Paramètres ===
    time               = options.get('time', RangeSet(0, 1))
    dt                 = options.get('dt', 900)  # durée du pas de temps en secondes
    p_min              = options.get('p_min', 0)
    p0                 = options.get('p0', 1000)  # Puissance nominale (W)
    fuel_cost          = options.get('fuel_cost', 1.2)  # €/L
    fuel_consumption   = options.get('fuel_consumption', 0.00025)  # L/Wh
    cost_inv           = options.get('cost_inv', 0.2)  # €/W
    cost_opex          = options.get('cost_opex', 0.01)  # €/W/an

    # === Données de linéarisation ===
    norm_P_points = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]   # p/p0
    eta_points    = [0.0, 0.188, 0.260, 0.299, 0.323, 0.339]  # rendement

    # === Paramètres ===
    gen.p0               = Param(initialize=p0, within=PositiveReals)
    gen.fuel_cost        = Param(default=fuel_cost)
    gen.fuel_consumption = Param(default=fuel_consumption)
    gen.cost_inv         = Param(default=cost_inv)
    gen.cost_opex        = Param(default=cost_opex)

    # === Variables ===
    gen.p         = Var(time, bounds=(p_min, p0), within=NonNegativeReals)   # Puissance réelle [W]
    gen.p_norm    = Var(time, bounds=(0, 1), within=NonNegativeReals)        # Puissance normalisée [0, 1]
    gen.eta       = Var(time, bounds=(0, 1), within=NonNegativeReals)        # Rendement diesel
    gen.e_th      = Var(time, within=NonNegativeReals)                       # Energie thermique consommée [Wh]

    # === Contraintes ===

    # Normalisation : p_norm[t] = p[t] / p0
    @gen.Constraint(time)
    def def_p_norm(g, t):
        return g.p_norm[t] == g.p[t] / g.p0
        # return g.p_norm[t] == 0

    # Interpolation par morceaux du rendement eta = f(p_norm)
    gen.pw = Piecewise(
        time,
        gen.eta,
        gen.p_norm,
        pw_pts=norm_P_points,
        f_rule=eta_points,
        pw_constr_type='EQ',
        pw_repn='SOS2'
    )

    # Consommation de carburant
    @gen.Constraint(time)
    def power_th(g, t):
        return g.e_th[t] * g.eta[t] == g.p[t] * (dt / 3600)

    # plt.figure(figsize=(6, 4))
    # plt.plot([x * 100 for x in norm_P_points], eta_points, marker='o')
    # plt.xlabel("Puissance (% de p0)")
    # plt.ylabel("Rendement η")
    # plt.title("Courbe de rendement du générateur diesel")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    return gen
