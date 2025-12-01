def energie_totale_consomme_rule(b, horizon):
    return (sum(b.charge.p[t] for t in b.time)
            * horizon.time_step.total_seconds() / 3600
            * 20)  # en Wh