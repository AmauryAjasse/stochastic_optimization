import matplotlib.pyplot as plt
import numpy as np
import random
from math import sqrt, pi
from datetime import date, time, datetime, timedelta
import matplotlib.dates as mdates

"""Voici les profils de conso d'origine qui viennent des articles de la littérature scientifique."""
azimoh = [5000, 5000, 5000, 5000, 5000, 5000, 45000, 37000, 40000, 41000, 39000, 44000, 53000, 52000, 57000, 52000, 56000, 35000, 26000, 27000, 47000, 38000, 5000, 5000] # W for 300 consumers
blodgett_entesopia = [76, 67, 39, 19, 19, 25, 73, 104, 163, 180, 303, 379, 410, 379, 284, 362, 368, 679, 752, 617, 413, 239, 152, 95] # W for 19 consumers
blodgett_barsaloi = [12, 9, 6, 2, 10, 29, 30, 53, 64, 77, 76, 67, 58, 43, 39, 39, 37, 103, 350, 414, 356, 199, 91, 33] # W for 23 customers
williams = [2.9, 1.9, 1.8, 1.7, 1.7, 1.9, 2.7, 2.1, 2.1, 2.1, 2, 1.9, 2, 2.3, 2.4, 2.8, 2.6, 2.5, 2.4, 9.2, 19.2, 21.8, 15.2, 6.5] # percent of total consumption of 90 Wh/day for 1 consumer
otieno = [2.8, 2, 1.7, 1.4, 1.3, 1.3, 1.7, 1.4, 1.9, 2.5, 6.4, 4, 4.7, 4.6, 4.2, 4.2, 5, 4.6, 4.6, 8.3, 12, 12.1, 9, 5.6] # W for one consumer
okundamiya = [3, 2.6, 2.6, 2.5, 8.4, 12.8, 14.1, 12.7, 10.7, 11, 12.6, 13.7, 17.7, 13.3, 10.8, 10.2, 10.4, 16.8, 31.3, 25.7, 17.3, 12.3, 7.7, 2.6] # W for 15 consumers
odou = [23, 14.1, 14.1, 14.1, 14.1, 16.7, 20.5, 2.9, 2.3, 2, 2.1, 1.8, 1.8, 2.1, 2.4, 3.7, 3.2, 4, 13.2, 39.6, 47.1, 46.7, 46, 30] # kW for 50 consumers

"""On prend les profils d'origine et on les mets à chaque fois pour un seul consommateur."""
azimoh_ok = [i / 300 for i in azimoh]
blodgett_entesopia_ok = [i / 19 for i in blodgett_entesopia]
blodgett_barsaloi_ok = [i / 23 for i in blodgett_barsaloi]
williams_ok = [i * (90/113.7) for i in williams] # 90 car 90Wh par jour et 113.7 car c'est la somme du pourcentage total de la liste d'origine
otieno_ok = otieno
okundamiya_ok = [i / 15 for i in okundamiya]
odou_ok = [i / 50 for i in odou]

"""On normalise les profils en pourcentage de la consommation totale."""
azimoh_percent = [i / sum(azimoh_ok) for i in azimoh_ok]
blodgett_entesopia_percent = [i / sum(blodgett_entesopia_ok) for i in blodgett_entesopia_ok]
blodgett_barsaloi_percent = [i / sum(blodgett_barsaloi_ok) for i in blodgett_barsaloi_ok]
williams_percent = [i / sum(williams_ok) for i in williams_ok]
otieno_percent = [i / sum(otieno_ok) for i in otieno_ok]
okundamiya_percent = [i / sum(okundamiya_ok) for i in okundamiya_ok]
odou_percent = [i / sum(odou_ok) for i in odou_ok]


profiles_dict_ok = {
    "Blodgett Entesopia": blodgett_entesopia_ok,
    "Blodgett Barsaloi": blodgett_barsaloi_ok,
    "Williams": williams_ok,
    "Otieno": otieno_ok,
    "Okundamiya": okundamiya_ok,
    "Odou": odou_ok
}
profiles_dict_percent = {
    "Blodgett Entesopia": blodgett_entesopia_percent,
    "Blodgett Barsaloi": blodgett_barsaloi_percent,
    "Williams": williams_percent,
    "Otieno": otieno_percent,
    "Okundamiya": okundamiya_percent,
    "Odou": odou_percent
}

def plot_profiles(profiles_dict, normalized=True):
    """
    Affiche tous les profils contenus dans un dictionnaire de profils.

    Args:
        profiles_dict (dict): dictionnaire {nom: liste de 24 valeurs}
    """
    heures = list(range(24))  # 0h à 23h

    plt.figure(figsize=(12, 6))

    for label, data in profiles_dict.items():
        plt.plot(heures, data, marker="o", label=label)

    plt.xlabel("Time (h)")
    if normalized :
        plt.ylabel("Normalized energy consumption")
    else :
        plt.ylabel("Energy consumption (Wh)")
    plt.title("Load profiles per consumer")
    plt.xticks(range(0, 24))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_hourly_boxplots(profiles_dict_percent: dict, percent: bool = True, show_points: bool = True, overlay_mean: bool = True, figsize=(13, 6)):
    """
    Affiche 24 boxplots (un par heure) construits à partir des 6 profils du dictionnaire.

    Args:
        profiles_dict_percent: dict {nom_profil: liste_de_24_valeurs_normalisées}
        percent: si True, affiche en % (multiplie par 100), sinon en fraction (0..1).
        show_points: si True, superpose les 6 points bruts (jitter léger) pour chaque heure.
        overlay_mean: si True, ajoute la courbe de la moyenne par heure.
        figsize: taille de la figure matplotlib.

    Returns:
        None
    """
    # Empilement: (n_profils x 24)
    arr = np.vstack([vals for vals in profiles_dict_percent.values()])  # shape: (6, 24)
    if percent:
        arr_plot = arr * 100.0
        ylabel = "Part de la consommation journalière (%)"
    else:
        arr_plot = arr
        ylabel = "Part de la consommation journalière (fraction)"

    # Préparer les échantillons par heure pour boxplot: liste de 24 tableaux (chaque tableau: 6 valeurs)
    hourly_samples = [arr_plot[:, h] for h in range(arr_plot.shape[1])]

    fig, ax = plt.subplots(figsize=figsize)

    # Boxplots positionnés sur 0..23
    ax.boxplot(
        hourly_samples,
        positions=np.arange(24),
        widths=0.6,
        showmeans=True  # triangle du mean par défaut
        # (on laisse les couleurs/styles par défaut)
    )

    # Points individuels (jitter léger pour visualiser les 6 échantillons)
    if show_points:
        n = arr_plot.shape[0]  # 6 profils
        # offsets symétriques et déterministes (évite l'aléa visuel)
        offsets = np.linspace(-0.18, 0.18, n)
        for h in range(24):
            ax.scatter(
                h + offsets,
                np.sort(arr_plot[:, h]),
                s=25,
                alpha=0.9,
                zorder=3
            )

    # Courbe de la moyenne horaire (optionnelle)
    if overlay_mean:
        mean_hour = arr_plot.mean(axis=0)
        ax.plot(np.arange(24), mean_hour, marker="o", linewidth=2, label="Moyenne horaire")
        ax.legend(fontsize=18)

    ax.set_xticks(np.arange(24))
    ax.set_xlabel("Heure", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    # ax.set_title("Boxplots horaires des parts de consommation (6 profils)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    # Taille de la police des graduations
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.show()

def mean_per_hour(profiles: dict):
    """
    Calcule la moyenne heure par heure sur toutes les listes d'un dictionnaire de profils.

    Args:
        profiles (dict): dictionnaire {nom: liste de 24 valeurs}

    Returns:
        list: liste 'mean' des moyennes heure par heure
    """
    n_profiles = len(profiles)
    n_hours = len(next(iter(profiles.values())))  # suppose que toutes les listes ont même longueur

    mean = []
    for h in range(n_hours):
        somme = sum(profiles[name][h] for name in profiles)
        mean.append(somme / n_profiles)

    return mean

profiles_dict_percent_with_mean = {
    "Blodgett Entesopia": blodgett_entesopia_percent,
    "Blodgett Barsaloi": blodgett_barsaloi_percent,
    "Williams": williams_percent,
    "Otieno": otieno_percent,
    "Okundamiya": okundamiya_percent,
    "Odou": odou_percent,
    "Moyenne": mean_per_hour(profiles_dict_percent)
}

def show_single_profile(name: str, profiles: dict, dt_hours: float = 1.0, as_bar: bool = False, metrics=True):
    """
    Affiche le profil 'name' issu du dictionnaire 'profiles' et retourne les métriques.

    Args:
        name (str): Nom du profil à tracer (clé du dict).
        profiles (dict): Dictionnaire {nom: liste_de_24_valeurs}.
        dt_hours (float): Pas de temps en heures (1.0 si données horaires).
        as_bar (bool): Si True, affiche un histogramme ; sinon une courbe.

    Returns:
        dict: {"mean_power_W": float, "daily_energy_Wh": float}
    """
    if name not in profiles:
        raise ValueError(f"Profil '{name}' introuvable. Profils dispo: {list(profiles.keys())}")

    data = profiles[name]
    n_hours = len(data)
    if n_hours != 24:
        # On autorise d'autres longueurs mais on prévient
        print(f"⚠️ Le profil '{name}' contient {n_hours} points (pas forcément horaires).")

    heures = list(range(n_hours))

    # Calculs
    mean_power_W = sum(data) / n_hours
    daily_energy_Wh = sum(v * dt_hours for v in data)  # Wh si v en W et dt_hours en h

    # Tracé
    plt.figure(figsize=(10, 5))
    if as_bar:
        plt.bar(heures, data)
    else:
        plt.plot(heures, data, marker="o")
    plt.title(f"Mean profile: {name}")
    plt.xlabel("Time (h)")
    plt.ylabel("Normalized energy consumption")
    plt.xticks(range(0, n_hours))
    plt.grid(True, linestyle="--", alpha=0.6)
    # On ajoute les métriques dans un petit texte
    if metrics:
        txt = f"Moyenne: {mean_power_W:.2f} W\nÉnergie journalière: {daily_energy_Wh:.2f} Wh"
        plt.gcf().text(0.99, 0.02, txt, ha="right", va="bottom")
    plt.tight_layout()
    plt.show()

    return {"mean_power_W": mean_power_W, "daily_energy_Wh": daily_energy_Wh}

def fit_hourly_gaussians(profiles_dict_percent, min_sigma=1e-9):
    """
    Estime une gaussienne par heure (mu, sigma) à partir des 6 profils normalisés.
    Retourne un dict: {0:{'mu':..., 'sigma':...}, ..., 23:{...}}
    """
    # Empile les profils en matrice (n_profils x 24)
    arr = np.vstack([profiles_dict_percent[name] for name in profiles_dict_percent])
    # Moyenne et écart-type par heure (axis=0), ddof=1 = écart-type sans biais
    mus = arr.mean(axis=0)
    sigmas = arr.std(axis=0, ddof=1)
    # Evite sigma=0 (échantillon trop homogène)
    sigmas = np.maximum(sigmas, min_sigma)
    # Dictionnaire heure -> paramètres
    return {h: {"mu": float(mus[h]), "sigma": float(sigmas[h])} for h in range(24)}


def plot_hour_with_gaussian(profiles_dict_percent: dict, gaussians: dict, hour: int, show_hist: bool = False):
    """
    Affiche, pour une heure donnée (0..23), la gaussienne ajustée et les 6 valeurs observées.

    Args:
        profiles_dict_percent: dict {nom_profil: liste_de_24_valeurs_normalisées}
        gaussians: dict {heure: {"mu": float, "sigma": float}} (tel que renvoyé par fit_hourly_gaussians)
        hour: heure à tracer (0..23)
        show_hist: si True, ajoute un histogramme (densité) des 6 valeurs
    """
    if not (0 <= hour <= 23):
        raise ValueError("L'heure doit être comprise entre 0 et 23.")
    if hour not in gaussians:
        raise ValueError(f"Aucune gaussienne pour l'heure {hour} dans 'gaussians'.")

    # Récupère les 6 valeurs à l'heure 'hour'
    vals = np.array([v[hour] for v in profiles_dict_percent.values()], dtype=float)
    mu = float(gaussians[hour]["mu"])
    sigma = float(gaussians[hour]["sigma"])
    if sigma <= 0:
        raise ValueError(f"sigma nul ou négatif pour l'heure {hour}.")

    # Grille pour la PDF
    lo = max(0.0, min(vals.min(), mu - 4 * sigma))
    hi = max(vals.max(), mu + 4 * sigma)
    if hi <= lo:
        # cas pathologique : on élargit un peu la fenêtre
        hi = lo + (1e-6 if lo == 0 else 0.01 * abs(lo))

    x = np.linspace(lo, hi, 600)
    pdf = (1.0 / (sigma * sqrt(2 * pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    # Tracé
    plt.figure(figsize=(8, 5))
    plt.plot(x, pdf, label=f"Gaussienne N({mu:.4f}, {sigma:.4f}²)")

    if show_hist:
        # Histogramme en densité (peu parlant avec 6 points, mais parfois utile)
        nbins = min(6, max(3, len(vals)))
        plt.hist(vals, bins=nbins, density=True, alpha=0.3, edgecolor="black", label="Histogramme (échantillons)")

    # Rug plot + points pour visualiser les 6 valeurs
    y0 = -0.03 * pdf.max()  # petite marge sous l'axe pour la "jauge"
    for v in vals:
        plt.plot([v, v], [y0, 0], linewidth=2)
    plt.scatter(vals, np.zeros_like(vals), zorder=3, label="Échantillons (6 profils)")

    plt.xlabel("Part de la consommation journalière (même unité que les profils)")
    plt.ylabel("Densité (PDF)")
    plt.title(f"Heure {hour:02d} — valeurs observées et gaussienne ajustée")
    plt.legend()
    plt.ylim(bottom=y0 * 1.5)
    plt.tight_layout()
    plt.show()

"""On a différentes représentations des gaussiennes."""

"""Représentation 1 : pleins de petits sous-graphes qui permettent d'afficher toutes les gaussiennes et de comparer 
les formes directement """

def _pdf_grid(gaussians: dict, x_min=None, x_max=None, n=400):
    mus = np.array([gaussians[h]['mu'] for h in range(24)])
    sigmas = np.array([gaussians[h]['sigma'] for h in range(24)])
    lo = 0.0 if x_min is None else x_min
    hi = float(np.max(mus + 4*sigmas)) if x_max is None else x_max
    x = np.linspace(lo, hi, n)
    pdf = np.zeros((n, 24))
    for h in range(24):
        s = sigmas[h]
        pdf[:, h] = (1.0/(s*np.sqrt(2*np.pi))) * np.exp(-0.5*((x - mus[h])/s)**2)
    return x, pdf, mus, sigmas

def plot_gaussians_small_multiples(gaussians: dict, x_max: float = 0.2):
    x, pdf, mus, sigmas = _pdf_grid(gaussians)
    fig, axes = plt.subplots(4, 6, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    for h in range(24):
        ax = axes[h]
        ax.plot(x, pdf[:, h])
        ax.axvline(mus[h], linestyle="--", linewidth=1)
        ax.set_xlim(0.0, x_max)
        ax.set_title(f"{h:02d}h")
    fig.suptitle("Gaussiennes horaires (petits multiples)")
    axes[18].set_xlabel("Part de conso journalière")
    axes[0].set_ylabel("Densité (PDF)")
    plt.tight_layout()
    plt.show()

"""Représentation 2 : on représente une courbe qui correspond à la médiane des gaussiennes et autour on représente
les bandes 50% et 90%"""
def plot_gaussians_fan(gaussians: dict):
    z05, z25, z75, z95 = -1.64485362695, -0.67448975, 0.67448975, 1.64485362695
    mus = np.array([gaussians[h]['mu'] for h in range(24)])
    sig = np.array([gaussians[h]['sigma'] for h in range(24)])
    h = np.arange(24)
    p05 = mus + sig*z05; p25 = mus + sig*z25; p50 = mus; p75 = mus + sig*z75; p95 = mus + sig*z95

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(h, p05, p95, alpha=0.2, label="Bande 90%")
    ax.fill_between(h, p25, p75, alpha=0.4, label="Bande 50%")
    ax.plot(h, p50, linewidth=2, label="Médiane (μ)")
    ax.set_xticks(h)
    ax.set_xlabel("Heure")
    ax.set_ylabel("Part de conso journalière")
    ax.set_title("Quantiles horaires (fan chart) sous N(μ, σ)")
    ax.legend()
    plt.tight_layout()
    plt.show()

"""Représentation 3 : on représente une heatmap pour voir les densités des gaussiennes en couleur"""
def plot_gaussians_heatmap(gaussians: dict):
    x, pdf, _, _ = _pdf_grid(gaussians, n=300)
    # pdf: (n_x, 24) -> transpose pour imshow (y, x)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pdf.T, aspect='auto', origin='lower',
                   extent=[x[0], x[-1], 0, 23])
    ax.set_xlabel("Part de conso journalière")
    ax.set_ylabel("Heure")
    ax.set_yticks(range(0, 24, 3))
    ax.set_title("Densité des gaussiennes horaires (heatmap)")
    fig.colorbar(im, ax=ax, label="Densité")
    plt.tight_layout()
    plt.show()

"""On crée le profil de charge sur une journée pour un ménage"""
# def sample_profile_from_gaussians(gaussians, seed=None, clip_zero=True, renormalize=True, plot=False):
#     """
#     Tire un profil journalier (24h) en échantillonnant N(mu_h, sigma_h) pour chaque heure,
#     puis le convertit en énergie réelle par heure (Wh) de sorte que la somme journalière = DAILY_ENERGY_WH.
#
#     Args:
#         gaussians: dict {heure: {"mu": float, "sigma": float}}
#         seed: graine RNG
#         clip_zero: coupe à 0 les valeurs négatives
#         renormalize: conservé pour compatibilité ; le scaling final garantit de toute façon la somme = DAILY_ENERGY_WH
#         plot: si True, trace l'énergie horaire (Wh)
#     Returns:
#         list[float]: 24 valeurs en Wh (somme = DAILY_ENERGY_WH)
#     """
#     rng = np.random.default_rng(seed)
#     x = np.array([rng.normal(gaussians[h]["mu"], gaussians[h]["sigma"]) for h in range(24)], dtype=float)
#     if clip_zero:
#         x = np.clip(x, 0, None)
#
#     s = x.sum()
#     if s <= 0:
#         # fallback: profil uniforme si tout est nul après clipping
#         energy = np.full(24, DAILY_ENERGY_WH / 24.0)
#     else:
#         # Énergie horaire en Wh, somme journalière fixée à DAILY_ENERGY_WH
#         # (équivalent à renormaliser puis multiplier par DAILY_ENERGY_WH)
#         energy = (x / s) * DAILY_ENERGY_WH
#
#     if plot:
#         heures = np.arange(24)
#         plt.figure(figsize=(10, 4))
#         plt.plot(heures, energy, marker="o")
#         plt.xticks(heures)
#         plt.xlabel("Time (h)")
#         plt.ylabel("Energy consumption (Wh)")
#         plt.title(f"Load profile — daily consumption = {sum(energy):.1f} Wh")
#         plt.grid(True, linestyle="--", alpha=0.5)
#         plt.tight_layout()
#         plt.show()
#
#     return energy.tolist()

def sample_profile_from_gaussians(
        gaussians,
        mean_daily_kwh,
        std_daily_kwh,
        seed=None,
        clip_zero=True,
        renormalize=True,
        plot=False):
    """
    Tire un profil journalier 24h, avec une consommation journalière totale aléatoire.

    La forme horaire est tirée à partir des gaussiennes horaires.
    La somme journalière est tirée dans une loi normale :
        N(mean_daily_kwh, std_daily_kwh)

    Args:
        gaussians: dict {heure: {"mu": float, "sigma": float}}
        mean_daily_kwh: moyenne de consommation journalière par consommateur (kWh/jour)
        std_daily_kwh: écart-type de consommation journalière par consommateur (kWh/jour)
        seed: graine RNG
        clip_zero: coupe à 0 les valeurs négatives horaires
        renormalize: conservé pour compatibilité
        plot: si True, trace l'énergie horaire (Wh)

    Returns:
        list[float]: 24 valeurs en Wh
    """

    rng = np.random.default_rng(seed)

    # Tirage de la consommation journalière totale
    daily_energy_kwh = rng.normal(mean_daily_kwh, std_daily_kwh)

    # Sécurité : une consommation ne peut pas être négative
    daily_energy_kwh = max(daily_energy_kwh, 0)

    # Conversion kWh -> Wh
    daily_energy_wh = daily_energy_kwh * 1000

    # Tirage de la forme horaire du profil
    x = np.array([
        rng.normal(gaussians[h]["mu"], gaussians[h]["sigma"])
        for h in range(24)
    ], dtype=float)

    if clip_zero:
        x = np.clip(x, 0, None)

    s = x.sum()

    if s <= 0:
        energy = np.full(24, daily_energy_wh / 24.0)
    else:
        energy = (x / s) * daily_energy_wh

    if plot:
        heures = np.arange(24)
        plt.figure(figsize=(10, 4))
        plt.plot(heures, energy, marker="o")
        plt.xticks(np.arange(0, 24, 2), fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel("Temps (h)", fontsize=16)
        plt.ylabel("Energie consommée (Wh)", fontsize=16)
        plt.title(f"Load profile — daily consumption = {sum(energy):.1f} Wh")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    return energy.tolist()

"""On crée le profil de charge sur une année pour un ménage"""
def sample_annual_profile_from_gaussians(gaussians, mean_daily_kwh= 0.154, std_daily_kwh=0.162, seed=None, clip_zero=True, renormalize=True, plot=False):
    """
    Construit un profil domestique annuel (365 jours) en concaténant 365 profils
    tirés via sample_profile_from_gaussians(...). L'affichage (plot=True) met l'axe des x en DATES.

    Args:
        gaussians: dict {heure: {"mu": float, "sigma": float}}
        seed: graine globale
        clip_zero, renormalize, plot: mêmes rôles que dans sample_profile_from_gaussians

    Returns:
        list[float]: 24*365 valeurs (Wh/heure). La somme ≈ 365 * DAILY_ENERGY_WH.
    """
    days = 365
    annual = []

    if seed is None:
        for _ in range(days):
            daily = sample_profile_from_gaussians(
                gaussians,
                seed=None,
                mean_daily_kwh=mean_daily_kwh,
                std_daily_kwh=std_daily_kwh,
                clip_zero=clip_zero,
                renormalize=renormalize,
                plot=False
            )
            annual.extend(daily)
    else:
        rng = np.random.default_rng(seed)
        day_seeds = rng.integers(0, 2 ** 32 - 1, size=days, dtype=np.uint32)
        for ds in day_seeds:
            daily = sample_profile_from_gaussians(
                gaussians,
                seed=int(ds),
                clip_zero=clip_zero,
                renormalize=renormalize,
                plot=False
            )
            annual.extend(daily)

    # if plot:
    #     # Axe des x en DATES (pas en heures)
    #     t0 = datetime.combine(START_DATE, time(0, 0))
    #     ts = [t0 + timedelta(hours=i) for i in range(24 * days)]
    #
    #     fig, ax = plt.subplots(figsize=(14, 4))
    #     ax.plot(ts, annual, linewidth=0.9)
    #     ax.set_xlabel("Date", fontsize=16)
    #     ax.set_ylabel("Energie consommée (Wh)", fontsize=16)
    #     # ax.set_title(f"Domestic annual profile - one consumer")
    #
    #     # Ticks lisibles: mois en majeur, semaines en mineur
    #     ax.xaxis.set_major_locator(mdates.MonthLocator())
    #     ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    #     ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
    #
    #     fig.autofmt_xdate()
    #     ax.grid(True, linestyle="--", alpha=0.4)
    #     plt.tight_layout()
    #     plt.xticks(fontsize=16)
    #     plt.yticks(fontsize=16)
    #     plt.show()

    if plot:
        # Axe des x en DATES (pas en heures)
        t0 = datetime.combine(START_DATE, time(0, 0))
        ts = [t0 + timedelta(hours=i) for i in range(24 * days)]

        # =========================
        # Graphe 1 : année complète
        # =========================
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(ts, annual, linewidth=0.9)

        ax.set_xlabel("Date", fontsize=16)
        ax.set_ylabel("Energie consommée (Wh)", fontsize=16)

        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_minor_locator(
            mdates.WeekdayLocator(byweekday=mdates.MO, interval=1)
        )

        fig.autofmt_xdate()
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(axis="both", labelsize=16)

        plt.tight_layout()
        plt.show()

        # ================================================
        # Graphe 2 : zoom du 6 février matin au 12 février soir
        # ================================================
        zoom_start = datetime.combine(
            START_DATE.replace(month=2, day=6),
            time(0, 0)
        )

        zoom_end = datetime.combine(
            START_DATE.replace(month=2, day=12),
            time(23, 0)
        )

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(ts, annual, linewidth=0.9)

        ax.set_xlim(zoom_start, zoom_end)

        ax.set_xlabel("Date", fontsize=20)
        ax.set_ylabel("Energie consommée (Wh)", fontsize=20)

        # Graduation principale : un tick par jour
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

        # Graduation secondaire : toutes les 6 heures
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))

        fig.autofmt_xdate()
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(axis="both", labelsize=20)

        plt.tight_layout()
        plt.show()

    return annual

"""On crée le profil de charge sur une année pour tout le micro-réseau"""
def sample_microgrid_annual_profile_from_gaussians(gaussians, seed=None, clip_zero=True, renormalize=True, plot=False, return_households=False):
    """
    Construit la courbe annuelle d'un micro-réseau en sommant N_HOUSEHOLDS
    profils annuels (un par ménage) tirés depuis les gaussiennes horaires.

    Args:
        gaussians: dict {heure: {"mu": float, "sigma": float}}
        seed: graine globale (None => non reproductible). On dérive une graine par ménage.
        clip_zero, renormalize, plot: identiques à sample_profile_from_gaussians / _annual_...
        return_households (bool): si True, renvoie aussi la liste des profils par ménage.

    Returns:
        aggregate (list[float]): 24*365 valeurs Wh/heure (somme ≈ N_HOUSEHOLDS * 365 * DAILY_ENERGY_WH)
        households (list[list[float]] | optionnel): N_HOUSEHOLDS listes de 24*365 Wh
    """
    days = 365
    n = N_HOUSEHOLDS

    # Graine par ménage (si seed fourni) pour obtenir des profils différents mais reproductibles
    if seed is not None:
        rng = np.random.default_rng(seed)
        hh_seeds = rng.integers(0, 2**32 - 1, size=n, dtype=np.uint32)
    else:
        hh_seeds = [None] * n

    # Profils annuels par ménage
    households = []
    for i in range(n):
        prof = sample_annual_profile_from_gaussians(
            gaussians,
            seed=None if seed is None else int(hh_seeds[i]),
            clip_zero=clip_zero,
            renormalize=renormalize,
            plot=False
        )
        households.append(prof)

    # Somme heure par heure (8760 points)
    arr = np.array(households)               # shape: (n, 8760)
    aggregate = arr.sum(axis=0).tolist()     # shape: (8760,)

    if plot:
        # Axe des x en dates (cohérent avec ta fonction annuelle)
        t0 = datetime.combine(START_DATE, time(0, 0))
        ts = [t0 + timedelta(hours=i) for i in range(24 * days)]

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(ts, aggregate, linewidth=1.2, label=f"Microgrid (N={n})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Energy consumption (Wh)")
        total_wh = sum(aggregate)
        ax.set_title(f"Annual consumption — microgrid ({n} domestic consumers)\nTotal = {total_wh:.0f} Wh")

        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
        fig.autofmt_xdate()
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        plt.tight_layout()
        plt.show()

    if return_households:
        return aggregate, households
    return aggregate

# Exécution de l'application
if __name__ == '__main__':
    """Variables globales"""
    DAILY_ENERGY_WH = 120.0  # Wh par ménage et par jour
    START_DATE = date(2023, 1, 1)  # date de début de l'année (modifiable globalement)
    N_HOUSEHOLDS = 15  # nombre de ménages dans le micro-réseau

    """ On commence par afficher tous les profils sans la moyenne."""
    # plot_profiles(profiles_dict_ok, normalized=False)
    """On affiche également les profils normalisés (toujours sans la moyenne)."""
    # plot_profiles(profiles_dict_percent, normalized=True)
    """On affiche un boxplot à chaque heure plutôt que les courbes de base"""
    # plot_hourly_boxplots(profiles_dict_percent, show_points=False)

    """On affiche à nouveau les profils normalisés mais en affichant également la moyenne heure par heure."""
    # plot_profiles(profiles_dict_percent_with_mean)

    """On affiche uniquement le profil normalisé moyen."""
    # show_single_profile("Moyenne", profiles_dict_percent_with_mean, metrics=False)

    """On crée un dictionnaire contenant les informations des gaussiennes pour chaque heure."""
    gaussians = fit_hourly_gaussians(profiles_dict_percent)

    """On affiche une gaussienne avec les valeurs de consommation pour une heure donnée."""
    # plot_hour_with_gaussian(profiles_dict_percent, gaussians, hour=20, show_hist=False)

    """On compare toutes les gaussiennes."""
    """Représentation 1 : toutes les gaussiennes en plein de sous-graphes"""
    # plot_gaussians_small_multiples(gaussians)

    """Représentation 2 : on représente une courbe qui correspond à la médiane des gaussiennes et autour on représente
    les bandes 50% et 90%"""
    # plot_gaussians_fan(gaussians)

    """Représentation 3 : on représente une heatmap pour voir les densités des gaussiennes en couleur"""
    # plot_gaussians_heatmap(gaussians)

    """On créé le profil de charge sur une journée pour un ménage"""
    # one_day_example = sample_profile_from_gaussians(gaussians, mean_daily_kwh= 0.154, std_daily_kwh=0.162, plot=True)

    """On crée le profil de charge sur une année pour un ménage"""
    one_year_example = sample_annual_profile_from_gaussians(gaussians, mean_daily_kwh= 0.154, std_daily_kwh=0.162, seed=None, clip_zero=True, renormalize=True, plot=True)

    """On crée le profil de charge sur une année pour tout le micro-réseau"""
    # sample_microgrid_annual_profile_from_gaussians(gaussians, plot=True)