from pyomo.environ import *
from pyomo.core import NonNegativeReals, Binary, PositiveReals, Reals, Any
from pyomo.network import Port

from lms2.tools.data_processing import read_data, load_data
from lms2.core.horizon import SimpleHorizon
from lms2.electric.sources import pv_panel, fixed_power_load, power_source, scalable_power_source
from lms2.tools.post_processing import *

import math
import os
import matplotlib.pyplot as plt
import numpy as np
import datetime
import logging

def block_pv(b, curtailable=False, **kwargs):
    """Photovoltaic panel model.

    This PV model is limited in power, and it can also be curtailed, setting the key-word parameter curtailable`=True.
    In this case, a positive portion of the generated power, named p_curt, can be curtailed.

    Instanciation options:
        - p_wp_max :     pv maximum installed power (W)
        - p_wp_min :     pv minimum installed power (W)
        - cost_inv :     pv investment cost (€/W)
        - cost_opex :    pv operation and maintenance cost (€/W/year)
    """

    p_wp_max       = kwargs.get('p_wp_max', 1e8)
    p_wp_min       = kwargs.get('p_wp_min', 1)
    time           = kwargs.get('time', RangeSet(0, 1))

    cost_inv       = kwargs.pop('cost_inv', 1.5)
    cost_opex      = kwargs.pop('cost_opex', 0.01)

    b.cost_inv     = Param(initialize=cost_inv)
    b.cost_opex    = Param(initialize=cost_opex)
    b.tmp_n        = Param(initialize=298)
    b.irr_n        = Param(initialize=1000)
    b.gamma        = Param(initialize=-0.004)
    b.noct         = Param(initialize=45)

    b.irr          = Param(time, mutable=True, default=0)
    b.tmp          = Param(time, mutable=True, default=0)

    b.p_wp         = Var(initialize=p_wp_max, within=PositiveReals, bounds=(p_wp_min, p_wp_max))

    b.p            = Var(time, initialize=p_wp_max, within=PositiveReals, bounds=(0, 1e8))

    if curtailable:
        b.p_curt       = Var(time, within=NonNegativeReals, bounds=(0, 1e8))
        @b.Constraint(time)
        def p_wp_constraint(b, t):
            return b.p[t] + b.p_curt[t] == (b.irr[t] / b.irr_n) * b.p_wp * (
                        1 + b.gamma * (b.tmp[t] + (b.noct - 20) * b.irr[t] / 800 + 273 - b.tmp_n))

    else:
        @b.Constraint(time)
        def p_wp_constraint(b, t):
            return b.p[t] == (b.irr[t] / b.irr_n) * b.p_wp * (1 + b.gamma * (b.tmp[t] + (b.noct - 20) * b.irr[t] / 800 + 273 - b.tmp_n))

    return b