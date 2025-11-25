"""Dans ce script, on met toutes les fonctions liées aux calculs économiques qui peuvent être utilisées lors
de mon optimisation, qu'elle soit déterministe ou stochastique."""


def capex_pv_rule(b):
    """Calcule le CAPEX des modules phovoltaïques en multipliant le coût d'investissement des panneaux en €/W par
    la puissance installée en W."""
    return b.pv.cost_inv * b.pv.p_wp.value


def capex_bat_rule(b):
    """Calcule le CAPEX des batteries en multipliant le coût d'investissement des batteries en €/Wh par
        l'énergie maximale qui peut être stockée en Wh."""
    return b.bat.cost_inv * b.bat.emax0

def opex_pv_rule(b, discount_rate=1, total_duration=20):
    """Calcule le OPEX des modules phovoltaïques en multipliant le coût d'opération des panneaux en €/W/an par
    la puissance installée en W et par un facteur faisant intervenir le taux d'actualisation sur 20 ans."""
    return b.pv.cost_opex * b.pv.p_wp.value * sum(1 / (1+discount_rate)**i for i in range(0, total_duration))


def opex_bat_rule(b, discount_rate=1, total_duration=20):
    """Calcule le OPEX des batteries en multipliant le coût d'opération des batteries en €/Wh/an par
        l'énergie maximale qui peut être stockée en Wh et par un facteur faisant intervenir le taux
        d'actualisation."""
    return b.bat.cost_opex * b.bat.emax0 * sum(1 / (1+discount_rate)**i for i in range(0, total_duration))


def repl_bat_rule(b, discount_rate=1, replacement_year=[5]):
    """Calcule le coût de remplacement des batteries en multipliant le coût d'investissement des batteries en €/Wh par
        l'énergie maximale qui peut être stockée en Wh et par un facteur faisant intervenir le taux
        d'actualisation. On rentre aussi les années"""
    return sum(b.bat.cost_inv * b.bat.emax0 / (1 + discount_rate)**i for i in replacement_year)

def cout_total_rule(b, with_diesel_generator=0):
    if with_diesel_generator == 0:
        return (b.pv.cost_inv * b.pv.p_wp  # invest pv
            + b.pv.cost_opex * b.pv.p_wp * 9.64955841794  # maintenance pv
            + b.bat.cost_inv * b.bat.emax0  # invest batt
            + b.bat.cost_opex * b.bat.emax0 * 9.64955841794  # maintenance batt
            + b.bat.cost_inv * b.bat.emax0 * 0.635227665282  # replacement batt annee 5
            + b.bat.cost_inv * b.bat.emax0 * 0.403514186739  # replacement batt annee 10
            + b.bat.cost_inv * b.bat.emax0 * 0.256323374751  # replacement batt annee 15
            )
    else:
        return (b.pv.cost_inv * b.pv.p_wp  # invest pv
                + b.pv.cost_opex * b.pv.p_wp * 9.64955841794  # maintenance pv
                + b.bat.cost_inv * b.bat.emax0  # invest batt
                + b.bat.cost_opex * b.bat.emax0 * 9.64955841794  # maintenance batt
                + b.bat.cost_inv * b.bat.emax0 * 0.635227665282  # replacement batt annee 5
                + b.bat.cost_inv * b.bat.emax0 * 0.403514186739  # replacement batt annee 10
                + b.bat.cost_inv * b.bat.emax0 * 0.256323374751  # replacement batt annee 15
                + b.gen.cost_inv * b.gen.p0  # invest diesel
                + b.gen.cost_opex * b.gen.p0 * 9.64955841794  # maintenance diesel
                + sum(b.gen.e_th[t] * b.gen.fuel_cost * b.gen.fuel_consumption for t in b.time) * 9.64955841794 # consumption diesel
                )