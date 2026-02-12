from pyomo.environ import *
import math

"""Dans ce script, on met toutes les fonctions contraintes qui peuvent être utilisées lors de mon optimisation,
qu'elle soit déterministe ou stochastique."""

def same_pv_rule(m, s, s_ref):
    """On impose d'avoir le même nombre de panneaux photovoltaïques pour tous les scénarios."""
    if s == s_ref: return Constraint.Skip
    return m.pv[s].p_wp == m.pv[s_ref].p_wp

def same_bat_rule(m, s, s_ref, t0=0):
    """On impose d'avoir le même nombre de batteries pour tous les scénarios."""
    if s==s_ref: return Constraint.Skip
    return m.bat[s].emax[t0] == m.bat[s_ref].emax[t0]

def same_gen_rule(m, s, s_ref):
    """On impose d'avoir le même nombre de batteries pour tous les scénarios."""
    if s==s_ref: return Constraint.Skip
    return m.gen[s].p0 == m.gen[s_ref].p0

def bilan_puissance_rule(b, s, t, with_diesel_generator=0):
    """C'est la contrainte de bilan de puissance qui est scindée en 2 en fonction de la présence ou non du
    générateur diesel.
    Sans générateur : la production des panneaux photovoltaïques et des batteries doivent satisfaire la consommation
    Avec générateur : la production des panneaux photovoltaïques, des batteries et du générateur diesel doivent satisfaire la consommation"""
    if with_diesel_generator == 0:
        return b.bat[s].p[t] + b.pv[s].p[t] == b.consumption[s].p[t]
    else:
        return b.bat[s].p[t] + b.pv[s].p[t] + b.gen[s].p[t] == b.consumption[s].p[t]

def bilan_with_shed_rule(b, s, t, with_diesel_generator=0):
    if with_diesel_generator == 0:
        return b.bat[s].p[t] + b.pv[s].p[t] + b.p_unserved[s, t] == b.consumption[s].p[t]
    else:
        return b.bat[s].p[t] + b.pv[s].p[t] + b.gen[s].p[t] + b.p_unserved[s, t] == b.consumption[s].p[t]

# On ne peut pas écrêter le solaire tant que la batterie n'a pas atteint une certaine valeur de soc (soc_allow_curt) pour les 4 fonctions suivantes
def pv_curt_bigM_rule(m, s_ref, M_curt=1e6, M_energy=None):
    """
    Calcule/retourne (M_curt, M_energy) utilisés par les contraintes 'no curtailment before SOC threshold'.
    """
    # M_curt : borne sup sur p_curt (W)
    if M_curt is None:
        try:
            M_curt = value(m.pv[s_ref].p_wp)
        except Exception:
            M_curt = 1e6

    # M_energy : borne sup sur l'énergie batterie (Wh)
    if M_energy is None:
        try:
            t0 = m.time.first()
            M_energy = value(m.bat[s_ref].emax[t0])
        except Exception:
            M_energy = 1e8

    return float(M_curt), float(M_energy)


def full_lower_rule(b, s, t, s_ref, M_energy):
    """
    Si is_full=1 => e >= soc_allow_curt% * emax (sinon relaxé par Big-M).
    """
    return b.bat[s].e[t] >= (value(b.bat[s_ref].soc_allow_curt) / 100.0) * b.bat[s].emax[t] - M_energy * (1 - b.is_full[s, t])


def full_upper_rule(b, s, t, s_ref, M_energy):
    """
    Si is_full=0 => e <= soc_allow_curt% * emax (sinon relaxé par Big-M).
    """
    return b.bat[s].e[t] <= (value(b.bat[s_ref].soc_allow_curt) / 100.0) * b.bat[s].emax[t] + M_energy * (b.is_full[s, t])


def curtail_only_if_full_rule(b, s, t, M_curt):
    """
    Si is_full=0 => p_curt = 0 ; si is_full=1 => p_curt <= M_curt
    """
    return b.pv[s].p_curt[t] <= M_curt * b.is_full[s, t]


# On doit satisfaire la consommation au moins xx% du temps (xx = consumption_satisfaction) pour les 4 fonctions suivantes
def min_served_steps(time_set, consumption_satisfaction):
    """Nombre minimal de pas de temps servis."""
    nT = len(list(time_set))
    return int(math.ceil((consumption_satisfaction / 100.0) * nT))


def served_link_rule(b, s, t, M_shed):
    """
    Lien Big-M entre p_unserved et is_served.
    - Si is_served = 1 => p_unserved <= 0 => donc p_unserved = 0 (car >=0)
    - Si is_served = 0 => p_unserved <= M_shed (libre)
    """
    return b.p_unserved[s, t] <= M_shed * (1 - b.is_served[s, t])


def satisfaction_rule_per_scenario(b, s, min_served):
    """Au moins min_served pas de temps servis dans chaque scénario."""
    return sum(b.is_served[s, t] for t in b.time) >= min_served


def satisfaction_rule_weighted(b, min_served, prob):
    """Version pondérée (en espérance) si besoin."""
    return sum(prob[s] * sum(b.is_served[s, t] for t in b.time) for s in b.S) >= min_served


# on force la batterie à n'écrêter la consommation qu'à certaines heures
def forbid_shed_outside_window_rule(b, s, t, allow_shed):
    """
    Interdit le délestage (p_unserved=0) si allow_shed[t] = 0
    allow_shed : Param (0/1) indexé par m.time
    """
    if int(allow_shed[t]) == 1:
        return Constraint.Skip
    return b.p_unserved[s, t] == 0

def force_served_outside_window_rule(b, s, t, allow_shed):
    """
    Force is_served=1 si allow_shed[t] = 0
    (optionnel mais cohérent avec served_link_rule)
    """
    if int(allow_shed[t]) == 1:
        return Constraint.Skip
    return b.is_served[s, t] == 1

# lorsqu'on ne satisfait pas la consommation, on ne peut pas recharger la batterie
def no_battery_when_shed_rule(b, s, t):
    M_bat=1e6
    return b.bat[s].p[t] <= M_bat * b.is_served[s, t]

def no_battery_when_shed_rule_neg(b, s, t):
    M_bat = 1e6
    return -b.bat[s].p[t] <= M_bat * b.is_served[s, t]

def soc_final_ge_initial_rule(b, s):
    """
    Impose e_final >= e_initial pour la batterie
    """
    t0 = b.time.first()
    t_end = b.time.last()
    return b.bat[s].e[t_end] >= b.bat[s].e[t0]