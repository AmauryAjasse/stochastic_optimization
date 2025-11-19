from pyomo.environ import *
from pyomo.core import NonNegativeReals, Binary, PositiveReals, Reals, Any
from pyomo.network import Port

from lms2.tools.data_processing import read_data, load_data
from lms2.core.horizon import SimpleHorizon
from lms2.electric.sources import pv_panel, fixed_power_load, power_source, scalable_power_source
from lms2.tools.post_processing import *

import math
import os
import numpy as np
import datetime
import logging

UB=10e6 # Upper Bound by default

def battery_v2(bat, **options):
    """
    Bilinear battery Model.

    This battery is limited in power, variation of power, state of charge and energy. One can fix initial and final
    state of charge.
    Efficiency for charge and discharge are considered.
    It exposes one power port using source convention.

    Instanciation options:
        - c_bat :       battery capacity (kWh)
        - c_bat_max :   battery maximal capacity, default : +inf (only if c_bat is None)
        - c_bat_min :   battery minimal capacity, default : 0 (only if c_bat is None)
        - p_max :       maximal charging power, default : +inf (>0)
        - p_min :       maximal descharging power, default : +inf (>0)
        - soc_min :     minimal soc, default : 0 (>0)
        - soc_max :     maximal soc, default : 100 (>0)
        - soc0 :        initial SOC, default: 50 (0<soc0<100)
        - socf :        final SOC, defalut: 50 (0<socf<100)
        - eta_c :       charging efficiency, default : 1 (<1 and >0)
        - eta_d :       descharging efficiency, default : 1 (<1 and >0)

    """
    time        = options.get('time', RangeSet(0, 1))
    dt          = options.pop('dt', 1.0)

    c_bat       = options.pop('c_bat', None)
    c_bat_max   = options.pop('c_bat_max', UB)
    c_bat_min   = options.pop('c_bat_min', 0)
    p_max       = options.get('p_max', UB)
    p_min       = options.get('p_min', UB)
    soc_min     = options.pop('soc_min', 0)
    soc_max     = options.pop('soc_max', 100)
    soc0        = options.pop('soc0', 50)
    socf        = options.pop('socf', 50)
    eta_c       = options.pop('eta_c', 1)
    eta_d       = options.pop('eta_d', 1)

    cost_inv    = options.pop('cost_inv', 0.3) # €/Wh
    cost_opex   = options.pop('cost_opex', 0.005) # €/Wh/year

    if c_bat is None:
        assert c_bat_max is not None, 'User should either set c_bat or (c_bat_min and c_bat_max)'
        assert c_bat_min is not None, 'User should either set c_bat or (c_bat_min and c_bat_max)'
        assert c_bat_min < c_bat_max, "You should have c_bat_min < c_bat_max"
        bat.emax = Var(initialize=c_bat_min, doc='maximal energy (Wh)', bounds=(c_bat_min, c_bat_max))
        bat.emin = Param(default=0, doc='minimum energy (Wh)', mutable=True, within=NonNegativeReals)
    else:
        c_bat_max = c_bat
        c_bat_min = c_bat
        assert c_bat >= 0, 'Battery capacity should not be negative'
        logger.info('options c_bat_min and c_bat_max have no effect since c_bat is fixed.')
        bat.emax = Param(default=c_bat, doc='maximal energy', mutable=True, within=Reals)
        bat.emin = Param(default=0, doc='minimum energy (Wh)', mutable=True, within=NonNegativeReals)

    assert 1 >= eta_c > 0, 'eta_c should be positif, smaller than 1'
    assert 1 >= eta_d > 0, 'eta_d should be positif, smaller than 1'

    bat.pc        = Var(time, doc='Charging power', within=NonNegativeReals, initialize=0)
    bat.pd        = Var(time, doc='Discharging power', within=NonNegativeReals, initialize=0)
    bat.u         = Var(time, doc='Binary variable', within=Binary, initialize=0)

    bat.socmin    = Param(default=soc_min, doc='Minimum SOC (%)', mutable=True, within=Reals)
    bat.pinit     = Param(default=None, doc='initial output power of the battery (default : None)', mutable=True, within=Any)
    bat.socmax    = Param(default=soc_max, doc='Maximum SOC (%)', mutable=True, within=Reals)
    bat.soc0      = Param(default=soc0, doc='initial state', mutable=True, within=Any)
    bat.socf      = Param(default=socf, doc='final state', mutable=True, within=Any)
    bat.dpdmax    = Param(default=UB, doc='maximal discharging power', mutable=True, within=Reals)
    bat.dpcmax    = Param(default=UB, doc='maximal charging power', mutable=True, within=Reals)
    bat.pcmax     = Param(default=p_min, doc='maximal charging power', mutable=True, within=NonNegativeReals)
    bat.pdmax     = Param(default=p_max, doc='maximal discharging power', mutable=True, within=NonNegativeReals)
    bat.etac      = Param(default=eta_c, doc='Charging efficiency', mutable=True, within=Reals)
    bat.etad      = Param(default=eta_d, doc='Discharging efficiency', mutable=True, within=Reals)
    bat.dt        = Param(default=dt, doc=f'Time step')

    bat.cost_inv  = Param(default=cost_inv, doc='Investment cost (€/Wh)', mutable=True, within=NonNegativeReals)
    bat.cost_opex = Param(default=cost_opex, doc='Operation cost (€/Wh/year)', mutable=True, within=NonNegativeReals)

    def _init_e(m, t):
        if m.soc0.value is not None:
            return m.soc0 * m.emax / 100
        else:
            return 50

    bat.p = Var(time, doc='Energy derivative with respect to time (kW)', initialize=0)
    bat.e = Var(time, doc='Energy in battery (kWh)', initialize=_init_e, bounds=(0, c_bat_max))

    bat.outlet = Port(initialize={'f': (bat.p, Port.Extensive, {'include_splitfrac': False})},
                      doc='output power of the battery (kW), using source convention')

    # initializing pinit should not be done, since it can introduce infeasibility in case of moving horizon
    @bat.Constraint(time, doc='Initialize power')
    def _p_init(m, t):
        if m.pinit.value is not None:
            if t == time.first():
                return m.p[t] == m.pinit
        return Constraint.Skip

    @bat.Constraint(time, doc='Minimal energy constraint')
    def _e_min(m, t):
        if m.emin.value is None:
            return Constraint.Skip
        return m.e[t] >= m.emin

    @bat.Constraint(time, doc='Maximal energy constraint')
    def _e_max(m, t):
        if m.emax.value is None:
            return Constraint.Skip
        return m.e[t] <= m.emax

    @bat.Constraint(time, doc='Power bounds constraint')
    def _pmax(m, t):
        if m.pcmax.value is None:
            return Constraint.Skip
        else:
            return -m.pcmax, m.p[t], m.pdmax

    @bat.Constraint(time, doc='Initial soc constraint')
    def _soc_init(m, t):
        if m.soc0.value is None:
            return Constraint.Skip
        else:
            if t == time.first():
                return m.e[t] == m.soc0 * m.emax / 100
            else:
                return Constraint.Skip

    @bat.Constraint(time, doc='Final soc constraint')
    def _soc_final(m, t):
        if m.socf.value is None:
            return Constraint.Skip
        else:
            if t == time.last():
                return m.e[t] == m.socf * m.emax / 100
            else:
                return Constraint.Skip

    @bat.Constraint(time, doc='Minimal state of charge constraint')
    def soc_min_constraint(m, t):
        if m.socmin.value is None:
            return Constraint.Skip
        return m.e[t] >= m.socmin * m.emax / 100

    @bat.Constraint(time, doc='Maximal state of charge constraint')
    def soc_max_constraint(m, t):
        if m.socmax.value is None:
            return Constraint.Skip
        return m.e[t] <= m.socmax * m.emax / 100

    @bat.Constraint(time, doc='Discharging power bound')
    def _pdmax(bat, t):
        if bat.pdmax.value is None:
            return Constraint.Skip
        return bat.pd[t] - bat.u[t] * bat.pdmax <= 0

    @bat.Constraint(time, doc='Charging power bound')
    def _pcmax(bat, t):
        if bat.pcmax.value is None:
            return Constraint.Skip
        return bat.pc[t] + bat.u[t] * bat.pcmax <= bat.pcmax

    @bat.Constraint(time, doc='Energy balance constraint')
    def energy_balance(m, t):
        if t == time.first():
            return m.e[t] == m.soc0 * m.emax / 100
        return m.e[t] == m.e[t - m.dt] + (m.pc[t] * m.etac - m.pd[t] / m.etad) * m.dt / 3600

    @bat.Constraint(time, doc='Power balance constraint')
    def charging_balance(m, t):
        return m.p[t] == - m.pc[t] + m.pd[t]

    bat.soc = Expression(time, rule=lambda m, t: 100 * m.e[t] / m.emax, doc='Expression of the state of charge')

    return bat

def battery_v3(bat, **options):
    """
    Bilinear battery Model.

    This battery is limited in power, variation of power, state of charge and energy. One can fix initial and final
    state of charge.
    Efficiency for charge and discharge are considered.
    It exposes one power port using source convention.

    Instanciation options:
        - c_bat :       battery capacity (kWh)
        - c_bat_max :   battery maximal capacity, default : +inf (only if c_bat is None)
        - c_bat_min :   battery minimal capacity, default : 0 (only if c_bat is None)
        - p_max :       maximal charging power, default : +inf (>0)
        - p_min :       maximal descharging power, default : +inf (>0)
        - soc_min :     minimal soc, default : 0 (>0)
        - soc_max :     maximal soc, default : 100 (>0)
        - soc0 :        initial SOC, default: 50 (0<soc0<100)
        - socf :        final SOC, defalut: 50 (0<socf<100)
        - eta_c :       charging efficiency, default : 1 (<1 and >0)
        - eta_d :       descharging efficiency, default : 1 (<1 and >0)

    """
    time        = options.get('time', RangeSet(0, 1))
    dt          = options.pop('dt', 1.0)

    c_bat       = options.pop('c_bat', None)
    c_bat_max   = options.pop('c_bat_max', UB)
    c_bat_min   = options.pop('c_bat_min', 0)
    p_max       = options.get('p_max', UB)
    p_min       = options.get('p_min', UB)
    soc_min     = options.pop('soc_min', 0)
    soc_max     = options.pop('soc_max', 100)
    soc0        = options.pop('soc0', 50)
    socf        = options.pop('socf', 50)
    eta_c       = options.pop('eta_c', 1)
    eta_d       = options.pop('eta_d', 1)

    cost_inv    = options.pop('cost_inv', 0.3) # €/Wh
    cost_opex   = options.pop('cost_opex', 0.005) # €/Wh/year

    # Paramètres de vieillissement (article CARDOSO 2018)
    Q_bar       = options.pop('Q_bar', 0.2)         # perte de capacité max autorisée durant la durée de vie (sans dimension)
    L           = options.pop('L', 10)              # durée de vie ciblée en années
    E_r         = options.pop('E_r', 1.0)           # référence de définition des coefficients de vieillissement

    alpha       = options.pop('alpha', 5.04e-6)     # coefficient de vieillissement cyclique en kW^{-1}K^{-2}
    beta        = options.pop('beta', -2.998e-3)    # coefficient de vieillissement cyclique en kW^{-1}K^{-1}
    gamma       = options.pop('gamma', 0.446)       # coefficient de vieillissement cyclique en kW^{-1}
    delta       = options.pop('delta', -6.7e-3)     # coefficient de vieillissement cyclique en K^{-1}h
    epsilon     = options.pop('epsilon', 2.35)      # coefficient de vieillissement cyclique en h

    upsilon     = options.pop('upsilon', 4944)      # coefficient de vieillissement calendaire en months^{-1/2}
    Ea          = options.pop('Ea', 24500)          # énergie d'activation (loi d'Arrhenius) en J/mol
    R           = options.pop('R', 8.314)           # constante universelle des gaz parfaits en J/(mol.K)
    i_prime     = options.pop('i_prime', 0.2)

    if c_bat is None:
        assert c_bat_max is not None, 'User should either set c_bat or (c_bat_min and c_bat_max)'
        assert c_bat_min is not None, 'User should either set c_bat or (c_bat_min and c_bat_max)'
        assert c_bat_min < c_bat_max, "You should have c_bat_min < c_bat_max"
        bat.emax = Var(time, initialize=c_bat_max, doc='maximal energy', bounds=(c_bat_min, c_bat_max))
        bat.emin = Param(default=0, doc='minimum energy (Wh)', mutable=True, within=NonNegativeReals)
    else:
        c_bat_max = c_bat
        c_bat_min = c_bat
        assert c_bat >= 0, 'Battery capacity should not be negative'
        logger.info('options c_bat_min and c_bat_max have no effect since c_bat is fixed.')
        bat.emax = Param(default=UB, doc='maximal energy', mutable=True, within=Reals)
        bat.emin = Param(default=0, doc='minimum energy (Wh)', mutable=True, within=NonNegativeReals)

    assert 1 >= eta_c > 0, 'eta_c should be positif, smaller than 1'
    assert 1 >= eta_d > 0, 'eta_d should be positif, smaller than 1'

    bat.pc        = Var(time, doc='Charging power', within=NonNegativeReals, initialize=0)
    bat.pd        = Var(time, doc='Discharging power', within=NonNegativeReals, initialize=0)
    bat.u         = Var(time, doc='Binary variable', within=Binary, initialize=0)

    bat.socmin    = Param(default=soc_min, doc='Minimum SOC (%)', mutable=True, within=Reals)
    bat.pinit     = Param(default=None, doc='initial output power of the battery (default : None)', mutable=True, within=Any)
    bat.socmax    = Param(default=soc_max, doc='Maximum SOC (%)', mutable=True, within=Reals)
    bat.soc0      = Param(default=soc0, doc='initial state', mutable=True, within=Any)
    bat.socf      = Param(default=socf, doc='final state', mutable=True, within=Any)
    bat.dpdmax    = Param(default=UB, doc='maximal discharging power', mutable=True, within=Reals)
    bat.dpcmax    = Param(default=UB, doc='maximal charging power', mutable=True, within=Reals)
    bat.pcmax     = Param(default=p_min, doc='maximal charging power', mutable=True, within=NonNegativeReals)
    bat.pdmax     = Param(default=p_max, doc='maximal discharging power', mutable=True, within=NonNegativeReals)
    bat.etac      = Param(default=eta_c, doc='Charging efficiency', mutable=True, within=Reals)
    bat.etad      = Param(default=eta_d, doc='Discharging efficiency', mutable=True, within=Reals)
    bat.dt        = Param(default=dt, doc=f'Time step')

    bat.cost_inv  = Param(default=cost_inv, doc='Investment cost (€/Wh)', mutable=True, within=NonNegativeReals)
    bat.cost_opex = Param(default=cost_opex, doc='Operation cost (€/Wh/year)', mutable=True, within=NonNegativeReals)

    bat.tmp              = Param(time, mutable=True, default=0)
    # calendar_coeff = ((alpha * (bat.tmp[t] + 273.15) ** 2 + beta * (bat.tmp[t] + 273.15) + gamma) * exp(
    #     (delta * (bat.tmp[t] + 273.15) + epsilon) * i_prime) for t in time)
    # cycling_coeff = (upsilon * exp(-Ea / (R * (bat.tmp[t] + 273.15))) * np.sqrt(i_prime) for t in time)

    @bat.Expression(time, doc='Initialize calendar aging coefficient')
    def calendar_coeff(m, t):
        T_K = m.tmp[t] + 273.15
        return ((alpha * T_K**2 + beta * T_K + gamma) *
                exp((delta * T_K + epsilon) * i_prime))

    @bat.Expression(time, doc='Initialize cycling aging coefficient')
    def cycling_coeff(m, t):
        T_K = m.tmp[t] + 273.15
        return (upsilon * exp(-Ea / (R * T_K)) * np.sqrt(i_prime))


    def _init_e(m, t):
        if m.soc0.value is not None:
            return m.soc0 * m.emax[0] / 100
        else:
            return 50

    bat.p = Var(time, doc='Energy derivative with respect to time (kW)', initialize=0)
    bat.e = Var(time, doc='Energy in battery (kWh)', initialize=_init_e, bounds=(0, c_bat_max))
    bat.e_loss = Var(time, doc='Loss fo maximal stored energy', within=NonNegativeReals, initialize=0)

    bat.outlet = Port(initialize={'f': (bat.p, Port.Extensive, {'include_splitfrac': False})},
                      doc='output power of the battery (kW), using source convention')

    # initializing pinit should not be done, since it can introduce infeasibility in case of moving horizon
    @bat.Constraint(time, doc='Initialize power')
    def _p_init(m, t):
        if m.pinit.value is not None:
            if t == time.first():
                return m.p[t] == m.pinit
        return Constraint.Skip

    @bat.Constraint(time, doc='Loss of energy max stored constraint')
    def energy_loss(m, t):
        return m.e_loss[t] == m.calendar_coeff[t] * m.pd[t] + m.cycling_coeff[t]

    # @bat.Constraint(time, doc='Loss of energy max stored constraint')
    # def energy_loss(m, t):
    #     return m.e_loss[t] == 1e-10 * m.pd[t] + 1e-10

    @bat.Constraint(time, doc='Loss of energy max stored constraint')
    def energy_max_stored(m, t):
        if t == time.first():
            return m.emax[t] == m.emax[0]
        return m.emax[t] == m.emax[t - m.dt] - m.calendar_coeff[t] * m.pd[t] - m.cycling_coeff[t]

    # @bat.Constraint(time, doc='Loss of energy max stored constraint')
    # def energy_max_stored(m, t):
    #     if t == time.first():
    #         return m.emax[t] == m.emax[0]
    #     return m.emax[t] == m.emax[t - m.dt] - 1e-9 * m.pd[t] - 1e-9

    @bat.Constraint(time, doc='Minimal energy constraint')
    def _e_min(m, t):
        if m.emin.value is None:
            return Constraint.Skip
        return m.e[t] >= m.emin

    @bat.Constraint(time, doc='Maximal energy constraint')
    def _e_max(m, t):
        if m.emax[0] is None:
            return Constraint.Skip
        return m.e[t] <= m.emax[t] + m.e_loss[t]

    @bat.Constraint(time, doc='Power bounds constraint')
    def _pmax(m, t):
        if m.pcmax.value is None:
            return Constraint.Skip
        else:
            return -m.pcmax, m.p[t], m.pdmax

    @bat.Constraint(time, doc='Initial soc constraint')
    def _soc_init(m, t):
        if m.soc0.value is None:
            return Constraint.Skip
        else:
            if t == time.first():
                return m.e[t] == m.soc0 * m.emax[t] / 100
            else:
                return Constraint.Skip

    @bat.Constraint(time, doc='Final soc constraint')
    def _soc_final(m, t):
        if m.socf.value is None:
            return Constraint.Skip
        else:
            if t == time.last():
                return m.e[t] == m.socf * m.emax[t] / 100
            else:
                return Constraint.Skip

    @bat.Constraint(time, doc='Minimal state of charge constraint')
    def soc_min_constraint(m, t):
        if m.socmin.value is None:
            return Constraint.Skip
        return m.e[t] >= m.socmin * m.emax[t] / 100

    @bat.Constraint(time, doc='Maximal state of charge constraint')
    def soc_max_constraint(m, t):
        if m.socmax.value is None:
            return Constraint.Skip
        return m.e[t] <= m.socmax * m.emax[t] / 100

    @bat.Constraint(time, doc='Discharging power bound')
    def _pdmax(bat, t):
        if bat.pdmax.value is None:
            return Constraint.Skip
        return bat.pd[t] - bat.u[t] * bat.pdmax <= 0

    @bat.Constraint(time, doc='Charging power bound')
    def _pcmax(bat, t):
        if bat.pcmax.value is None:
            return Constraint.Skip
        return bat.pc[t] + bat.u[t] * bat.pcmax <= bat.pcmax

    @bat.Constraint(time, doc='Energy balance constraint')
    def energy_balance(m, t):
        if t == time.first():
            return m.e[t] == m.soc0 * m.emax[t] / 100
        return m.e[t] == m.e[t - m.dt] + (m.pc[t] * m.etac - m.pd[t] / m.etad) * m.dt / 3600

    @bat.Constraint(time, doc='Power balance constraint')
    def charging_balance(m, t):
        return m.p[t] == - m.pc[t] + m.pd[t]

    bat.soc = Expression(time, rule=lambda m, t: 100 * m.e[t] / m.emax[t], doc='Expression of the state of charge')

    return bat



