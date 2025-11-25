"""Dans ce script, on met toutes les fonctions contraintes qui peuvent être utilisées lors de mon optimisation,
qu'elle soit déterministe ou stochastique."""
def bilan_puissance_rule(b, t, with_diesel_generator=0):
    if with_diesel_generator == 0:
        return b.bat.p[t] + b.pv.p[t] == b.charge.p[t]
    else:
        return b.bat.p[t] + b.pv.p[t] + b.gen.p[t] == b.charge.p[t]