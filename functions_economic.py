"""Dans ce script, on met toutes les fonctions liées aux calculs économiques qui peuvent être utilisées lors
de mon optimisation, qu'elle soit déterministe ou stochastique."""


def capex_pv_rule(b):
    """Calcule le CAPEX des modules phovoltaïques en multipliant le coût d'investissement des panneaux en €/W par
        la puissance installée en W. Ce coût est commun à tous les scénarios."""
    s1 = list(b.S)[0]
    return b.pv[s1].cost_inv * b.pv[s1].p_wp


def capex_bat_rule(b):
    """Calcule le CAPEX des batteries en multipliant le coût d'investissement des batteries en €/Wh par
        l'énergie maximale qui peut être stockée en Wh. Ce coût est commun à tous les scénarios."""
    s1 = list(b.S)[0]
    return b.bat[s1].cost_inv * b.bat[s1].emax0

def capex_gen_rule(b):
    """Calcule le CAPEX du générateur diesel en multipliant le coût d'investissement du générateur en €/W par
    la puissance maximale de sortie en W. Ce coût est commun à tous les scénarios."""
    s1 = list(b.S)[0]
    return b.gen[s1].cost_inv * b.gen[s1].p0

def opex_pv_rule(b, discount_rate=1, total_duration=20):
    """Calcule le OPEX des modules phovoltaïques en multipliant le coût d'opération des panneaux en €/W/an par
    la puissance installée en W et par un facteur faisant intervenir le taux d'actualisation sur 20 ans.
    Ce coût est commun à tous les scénarios."""
    s1 = list(b.S)[0]
    return b.pv[s1].cost_opex * b.pv[s1].p_wp * sum(1 / (1+discount_rate)**i for i in range(0, total_duration))


def opex_bat_rule(b, discount_rate=1, total_duration=20):
    """Calcule le OPEX des batteries en multipliant le coût d'opération des batteries en €/Wh/an par
    l'énergie maximale qui peut être stockée en Wh et par un facteur faisant intervenir le taux d'actualisation.
    Ce coût est commun à tous les scénarios."""
    s1 = list(b.S)[0]
    return b.bat[s1].cost_opex * b.bat[s1].emax0 * sum(1 / (1+discount_rate)**i for i in range(0, total_duration))

def opex_gen_rule(b, discount_rate=1, total_duration=20):
    """Calcule le OPEX du générateur diesel en multipliant le coût d'opération du générateur en €/W/an par
    la puissance maximale de sortie en W et par un facteur faisant intervenir le taux d'actualisation.
    Ce coût est commun à tous les scénarios."""
    s1 = list(b.S)[0]
    return b.gen[s1].cost_opex * b.gen[s1].p0 * sum(1 / (1 + discount_rate) ** i for i in range(0, total_duration))

def repl_bat_rule(b, discount_rate=1, replacement_year=(5, 10, 15)):
    """Calcule le coût de remplacement des batteries en multipliant le coût d'investissement des batteries en €/Wh par
        l'énergie maximale qui peut être stockée en Wh et par un facteur faisant intervenir le taux
        d'actualisation. On rentre aussi les années. Ce coût est commun à tous les scénarios."""
    s1 = list(b.S)[0]
    return sum(b.bat[s1].cost_inv * b.bat[s1].emax0 / (1 + discount_rate)**i for i in replacement_year)

def expected_fuel_cost_rule(b, prob, discount_rate=1, total_duration=20):
    """Coût du carburant consommé. C'est le seul coût qui dépend de la gestion d'énergie et qui est donc propre à
    chaque scénario."""
    expected_fuel = sum(prob[s] * sum(b.gen[s].e_th[t] * b.gen[s].fuel_cost * b.gen[s].fuel_consumption for t in b.time) for s in b.S)
    return expected_fuel * sum(1 / (1 + discount_rate) ** i for i in range(0, total_duration))

def total_cost_rule(b, prob, with_diesel_generator=0, discount_rate=1, total_duration=20, replacement_year=[5, 10, 15]):
    if with_diesel_generator == 0:
        capex = capex_pv_rule(b) + capex_bat_rule(b)
        opex = opex_pv_rule(b, discount_rate, total_duration) + opex_bat_rule(b, discount_rate, total_duration)
        repl = repl_bat_rule(b, discount_rate, replacement_year)
        return capex + opex + repl

    else:
        capex = capex_pv_rule(b) + capex_bat_rule(b) + capex_gen_rule(b)
        opex = opex_pv_rule(b, discount_rate, total_duration) + opex_bat_rule(b, discount_rate, total_duration) + opex_gen_rule(b, discount_rate, total_duration)
        repl = repl_bat_rule(b, discount_rate, replacement_year)
        fuel = expected_fuel_cost_rule(b, prob, discount_rate, total_duration)
        return capex + opex + repl + fuel