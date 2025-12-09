from pyomo.environ import Constraint

"""Dans ce script, on met toutes les fonctions contraintes qui peuvent être utilisées lors de mon optimisation,
qu'elle soit déterministe ou stochastique."""
def bilan_puissance_rule(b, s, t, with_diesel_generator=0):
    """C'est la contrainte de bilan de puissance qui est scindée en 2 en fonction de la présence ou non du
    générateur diesel.
    Sans générateur : la production des panneaux photovoltaïques et des batteries doivent satisfaire la consommation
    Avec générateur : la production des panneaux photovoltaïques, des batteries et du générateur diesel doivent satisfaire la consommation"""
    if with_diesel_generator == 0:
        return b.bat[s].p[t] + b.pv[s].p[t] == b.consumption[s].p[t]
    else:
        return b.bat[s].p[t] + b.pv[s].p[t] + b.gen[s].p[t] == b.consumption[s].p[t]


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