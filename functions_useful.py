from datetime import datetime
from pyomo.environ import value

def energie_totale_consomme_rule(b, horizon):
    return (sum(b.charge.p[t] for t in b.time)
            * horizon.time_step.total_seconds() / 3600
            * 20)  # en Wh


def count_days_inclusive(start_str, end_str):
    """
    Renvoie le nombre de jours entre deux dates (inclus),
    sans tenir compte des heures.

    Exemple : du 1 au 3 => 3 jours.
    """
    # Conversion string -> datetime
    start_date = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S").date()
    end_date = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S").date()

    # Nombre de jours en comptant les deux extrémités
    return (end_date - start_date).days + 1


def compute_pv_curtailment_wh(m, s, dt_s):
    """
    Calcule, pour le scénario s, l'énergie PV écrêtée sur tout l'horizon, en Wh.

    - m : modèle Pyomo
    - s : indice de scénario (élément de m.S)
    - dt_s : pas de temps en secondes (par ex. 900 s pour 15 minutes)

    Hypothèse : block_pv a été créé avec curtailable=True, donc m.pv[s].p_curt[t] existe
    et représente la puissance PV non injectée [W] à l'instant t.
    """
    pv_block = m.pv[s]

    if not hasattr(pv_block, "p_curt"):
        raise AttributeError("Le bloc PV du scénario {s} n'a pas d'attribut 'p_curt' (curtailable=False ?)")

    total_Wh = 0.0
    for t in m.time:
        p_curt_t = value(pv_block.p_curt[t])  # W
        # énergie Wh = W * (dt en heures)
        total_Wh += p_curt_t * (dt_s / 3600.0)

    return total_Wh