import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence
from datetime import date, datetime, timedelta
import matplotlib.dates as mdates

school_odou = [30, 30, 30, 30, 30, 30,
          80, 30, 155, 155, 375, 320,
          220, 120, 170, 350, 350, 375,
          105, 135, 110, 110, 60, 30]

health_center_odou = [150, 150, 150, 150, 150, 270,
                      300, 150, 175, 175, 295, 240,
                      440, 440, 440, 440, 440, 465,
                      325, 325, 300, 180, 180, 150]

water_pump_odou = [0 ,0, 0, 0, 0, 0,
                   0, 0, 2200, 2200, 2200, 2200,
                   2200, 2200, 2200, 2200, 2200, 2200,
                   0, 0, 0, 0, 0, 0]

mills_odou = [0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 37500,
              37500, 37500, 37500, 37500, 37500, 0,
              0, 0, 0, 0, 0, 0]

church_odou = [40, 40, 40, 40, 40, 40,
               40, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0,
               0, 120, 120, 120, 120, 40]

church_williams = [2.34, 1.81, 1.52, 1.1, 0.97, 1.73,
                   2.26, 1.73, 2.15, 3.62, 5.8, 6.8,
                   5.88, 3.75, 4.46, 6.56, 8.55, 10.13,
                   5.56, 4.12, 6.38, 7.19, 5.38, 3.75]

def random_decortiquerie_daily_profile(
    power_w: float = 5000.0,
    window_start_h: float = 8.0,
    window_end_h: float = 18.0,
    target_hours: float = 3.0,
    jitter_hours: float = 0.5,
    n_blocks: Optional[int] = None,   # None => tirage aléatoire
    min_block: float = 0.25,       # 0.25 h = 15 min
    seed: Optional[int] = None,
    plot: bool = False,
    return_segments: bool = False,
):
    """
    Génère un profil journalier (24 valeurs) d'énergie (Wh) pour une décortiquerie
    avec plusieurs créneaux non contigus dans la fenêtre [window_start_h, window_end_h].

    - Puissance constante power_w (W)
    - Durée totale ~ target_hours ± jitter_hours (bornée à la fenêtre)
    - Répartition en 'n_blocks' segments (tirés si None), chacun ≥ min_block
    - Sortie: liste de 24 valeurs (Wh). Optionnel: retour des segments (start, end).

    Returns:
        energy_wh (list[float]) [, segments (list[tuple[start,end]])]
    """
    rng = np.random.default_rng(seed)

    # Longueur de fenêtre et durée totale
    win_len = max(0.0, window_end_h - window_start_h)
    dur = rng.uniform(target_hours - jitter_hours, target_hours + jitter_hours)
    dur = float(np.clip(dur, min_block, win_len))

    # Nombre de blocs
    max_possible_blocks = max(1, int(np.floor(dur / min_block)))
    if n_blocks is None:
        # tirage aléatoire entre 1 et min(4, max_possible_blocks)
        n_blocks = int(rng.integers(1, min(4, max_possible_blocks) + 1))
    else:
        n_blocks = int(np.clip(n_blocks, 1, max_possible_blocks))

    # Longueurs des blocs (Dirichlet pour répartir la durée, avec contrainte min_block)
    if n_blocks == 1:
        lengths = [dur]
    else:
        base = dur - n_blocks * min_block
        if base <= 1e-9:
            lengths = [min_block] * n_blocks
        else:
            w = rng.dirichlet(np.ones(n_blocks))
            lengths = (min_block + w * base).tolist()

    # Gaps (espaces vides) entre et autour des blocs, pour rester dans la fenêtre
    total_gap = win_len - dur
    n_gaps = n_blocks + 1
    if total_gap > 0:
        g = rng.dirichlet(np.ones(n_gaps))
        gaps = (g * total_gap).tolist()
    else:
        gaps = [0.0] * n_gaps

    # Construction des segments ordonnés dans la fenêtre
    segments = []
    t = window_start_h + gaps[0]
    for i in range(n_blocks):
        seg_start = t
        seg_end = t + lengths[i]
        segments.append((seg_start, seg_end))
        t = seg_end + gaps[i + 1]

    # Énergie par heure = somme des recouvrements des segments avec chaque [h, h+1]
    energy_wh = []
    for h in range(24):
        slot_start, slot_end = float(h), float(h + 1)
        overlap = 0.0
        for a, b in segments:
            overlap += max(0.0, min(b, slot_end) - max(a, slot_start))
        energy_wh.append(overlap * power_w)

    # Plot
    if plot:
        heures = np.arange(24)
        plt.figure(figsize=(10, 4))
        plt.bar(heures, energy_wh)
        plt.xticks(heures)
        plt.xlabel("Time (h)")
        plt.ylabel("Energy consumption (Wh)")
        tot = sum(energy_wh)
        plt.title(f"Water pump — {n_blocks} bloc(s), total = {tot:.0f} Wh")
        plt.grid(True, axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    return (energy_wh, segments) if return_segments else energy_wh

def decortiquerie_annual_profile_2023(
    power_w: float = 5000.0,
    window_start_h: float = 8.0,
    window_end_h: float = 18.0,
    target_hours: float = 3.0,
    jitter_hours: float = 0.5,
    n_blocks: Optional[int] = None,
    min_block: float = 0.25,
    seed: Optional[int] = None,
    plot: bool = False,
) -> list:
    """
    Construit le profil horaire (Wh) sur 2023 :
      - Jan–Avr & Nov–Déc : jours ouvrés => profil aléatoire via random_decortiquerie_daily_profile
      - Mai–Oct : 0
      - Week-ends : 0
    Renvoie une liste de 24*365 valeurs (Wh).
    """
    start = date(2023, 1, 1)
    end = date(2024, 1, 1)
    n_days = (end - start).days  # 365
    annual = []
    day_dates = []

    # Générateur pour dériver une seed par jour si 'seed' est donné
    day_rng = np.random.default_rng(seed) if seed is not None else None

    for d in range(n_days):
        cur = start + timedelta(days=d)
        day_dates.append(cur)

        # Mois off : Mai (5) -> Oct (10)
        if 5 <= cur.month <= 10:
            annual.extend([0.0] * 24)
            continue

        # Week-ends off (samedi=5, dimanche=6)
        if cur.weekday() >= 5:
            annual.extend([0.0] * 24)
            continue

        # Jour ouvré en mois actif : on génère un profil
        day_seed = None if day_rng is None else int(day_rng.integers(0, 2**32 - 1))
        daily = random_decortiquerie_daily_profile(
            power_w=power_w,
            window_start_h=window_start_h,
            window_end_h=window_end_h,
            target_hours=target_hours,
            jitter_hours=jitter_hours,
            n_blocks=n_blocks,
            min_block=min_block,
            seed=day_seed,
            plot=False,
        )
        annual.extend(daily)

    if plot:
        # --- PROFIL HORAIRE ---
        t0 = datetime(2023, 1, 1, 0, 0)
        ts = [t0 + timedelta(hours=i) for i in range(len(annual))]

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(ts, annual, linewidth=0.8)
        ax.set_xlabel("Date")
        ax.set_ylabel("Energy consumption (Wh)")
        ax.set_title("Mills — annual profile 2023 (0 from may to oct. and week-ends)")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

    return annual

def water_pump_annual_profile(
    year: int = 2023,
    power_w: float = 3000.0,
    window_start_h: float = 7.0,
    window_end_h: float = 22.0,
    target_hours: float = 4.0,
    jitter_hours: float = 0.5,
    n_blocks: Optional[int] = None,   # None = nombre de blocs aléatoire
    min_block: float = 0.25,          # 15 min mini par bloc
    seed: Optional[int] = None,       # graine globale (pour reproductibilité)
    plot: bool = False,
) -> list:
    """
    Construit le profil horaire (Wh) de la pompe à eau sur une année complète.
    - Utilisée tous les jours (week-ends inclus, aucun mois exclu).
    - Chaque jour est généré via random_decortiquerie_daily_profile(...) avec les paramètres ci-dessus.

    Returns:
        list[float]: 24 * nb_jours valeurs (Wh/heure).
    """
    # bornes annuelles
    start = date(year, 1, 1)
    end = date(year + 1, 1, 1)
    n_days = (end - start).days

    # générateur de sous-seeds par jour (si seed globale fournie)
    day_rng = np.random.default_rng(seed) if seed is not None else None

    annual = []
    for d in range(n_days):
        day_seed = None if day_rng is None else int(day_rng.integers(0, 2**32 - 1))
        daily = random_decortiquerie_daily_profile(
            power_w=power_w,
            window_start_h=window_start_h,
            window_end_h=window_end_h,
            target_hours=target_hours,
            jitter_hours=jitter_hours,
            n_blocks=n_blocks,
            min_block=min_block,
            seed=day_seed,
            plot=False,  # on trace à l'échelle annuelle ci-dessous si demandé
        )
        annual.extend(daily)

    if plot:
        t0 = datetime(year, 1, 1, 0, 0)
        ts = [t0 + timedelta(hours=i) for i in range(len(annual))]

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(ts, annual, linewidth=0.8)
        ax.set_xlabel("Date")
        ax.set_ylabel("Energy consumption (Wh)")
        ax.set_title(f"Water pump — annual profile {year}")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

    return annual

def school_annual_profile(
    base_profile: Sequence[float],
    year: int = 2023,
    noise_amp: float = 100.0,        # bruit horaire uniforme dans [-noise_amp, +noise_amp]
    seed: Optional[int] = None,      # graine optionnelle (reproductibilité)
    clip_at_zero: bool = True,       # empêche les valeurs négatives après bruit
    plot: bool = False               # trace le profil horaire annuel si True
) -> list:
    """
    Construit un profil horaire (Wh) sur l'année 'year' à partir d'un profil journalier de 24 valeurs.
    - Jours ouvrés (lundi-vendredi) : base_profile + bruit uniforme horaire ±noise_amp
    - Week-ends (samedi-dimanche) : 0
    - Renvoie une liste de longueur 24 * nb_jours (8760 pour 2023)

    Args:
        base_profile: séquence de 24 valeurs (Wh/heure) pour la journée type.
        year: année cible.
        noise_amp: amplitude du bruit horaire (uniforme dans [-noise_amp, +noise_amp]).
        seed: graine RNG pour reproductibilité (None => aléatoire).
        clip_at_zero: tronque à 0 après bruit si besoin.
        plot: si True, trace le profil horaire annuel.

    Returns:
        list[float]: profil horaire annuel (Wh).
    """
    if len(base_profile) != 24:
        raise ValueError("base_profile doit contenir 24 valeurs (0h..23h).")

    rng = np.random.default_rng(seed)

    start = date(year, 1, 1)
    end = date(year + 1, 1, 1)
    n_days = (end - start).days

    annual = []
    for d in range(n_days):
        cur = start + timedelta(days=d)
        if cur.weekday() >= 5:  # 5=Samedi, 6=Dimanche
            # Week-end: pas de conso
            annual.extend([0.0] * 24)
        else:
            base = np.asarray(base_profile, dtype=float)
            noise = rng.uniform(-noise_amp, +noise_amp, size=24)
            day = base + noise
            if clip_at_zero:
                day = np.clip(day, 0.0, None)
            annual.extend(day.tolist())

    if plot:
        t0 = datetime(year, 1, 1, 0, 0)
        ts = [t0 + timedelta(hours=i) for i in range(len(annual))]
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(ts, annual, linewidth=0.8)
        ax.set_xlabel("Date")
        ax.set_ylabel("Energy consumption (Wh)")
        ax.set_title(f"School — annual profile {year} (0 on week-end)")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

    return annual

def health_center_annual_profile(
    base_profile: Sequence[float],
    year: int = 2023,
    noise_amp: float = 100.0,        # bruit horaire uniforme dans [-noise_amp, +noise_amp]
    seed: Optional[int] = None,      # graine optionnelle (reproductibilité)
    clip_at_zero: bool = True,       # évite les valeurs négatives après bruit
    plot: bool = False               # trace le profil horaire annuel si True
) -> list:
    """
    Construit un profil horaire (Wh) sur l'année 'year' à partir d'un profil journalier (24 valeurs).
    - Health-center: fonctionne TOUS LES JOURS (lundi..dimanche).
    - Chaque jour = base_profile + bruit uniforme horaire ±noise_amp.
    - Renvoie une liste de longueur 24 * nb_jours.

    Args:
        base_profile: 24 valeurs (Wh/heure) pour la journée type.
        year: année cible.
        noise_amp: amplitude du bruit horaire (uniforme dans [-noise_amp, +noise_amp]).
        seed: graine RNG (None => aléatoire).
        clip_at_zero: tronque à 0 après bruit si nécessaire.
        plot: si True, trace le profil horaire annuel.

    Returns:
        list[float]: profil horaire annuel (Wh).
    """
    if len(base_profile) != 24:
        raise ValueError("base_profile doit contenir 24 valeurs (0h..23h).")

    rng = np.random.default_rng(seed)

    start = date(year, 1, 1)
    end = date(year + 1, 1, 1)
    n_days = (end - start).days

    annual = []
    for _ in range(n_days):
        base = np.asarray(base_profile, dtype=float)
        noise = rng.uniform(-noise_amp, +noise_amp, size=24)
        day = base + noise
        if clip_at_zero:
            day = np.clip(day, 0.0, None)
        annual.extend(day.tolist())

    if plot:
        t0 = datetime(year, 1, 1, 0, 0)
        ts = [t0 + timedelta(hours=i) for i in range(len(annual))]
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(ts, annual, linewidth=0.8)
        ax.set_xlabel("Date")
        ax.set_ylabel("Energy consumption (Wh)")
        ax.set_title(f"Health-center — annual profile {year}")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

    return annual

def church_annual_profile(
        base_profile: Sequence[float],
        year: int = 2023,
        noise_amp: float = 20.0,  # bruit horaire uniforme dans [-20, +20] Wh
        seed: Optional[int] = None,  # graine optionnelle (reproductibilité)
        clip_at_zero: bool = True,  # évite les valeurs négatives après bruit
        plot: bool = False  # trace le profil horaire annuel si True
) -> list:
    """
    Construit un profil horaire (Wh) sur l'année 'year' à partir d'un profil journalier (24 valeurs).
    - Église : fonctionne TOUS LES JOURS (lundi..dimanche).
    - Chaque jour = base_profile + bruit uniforme horaire ±noise_amp.
    - Renvoie une liste de longueur 24 * nb_jours.
    """
    if len(base_profile) != 24:
        raise ValueError("base_profile doit contenir 24 valeurs (0h..23h).")

    rng = np.random.default_rng(seed)

    start = date(year, 1, 1)
    end = date(year + 1, 1, 1)
    n_days = (end - start).days

    annual = []
    for _ in range(n_days):
        base = np.asarray(base_profile, dtype=float)
        noise = rng.uniform(-noise_amp, +noise_amp, size=24)
        day = base + noise
        if clip_at_zero:
            day = np.clip(day, 0.0, None)
        annual.extend(day.tolist())

    if plot:
        t0 = datetime(year, 1, 1, 0, 0)
        ts = [t0 + timedelta(hours=i) for i in range(len(annual))]
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(ts, annual, linewidth=0.8)
        ax.set_xlabel("Date")
        ax.set_ylabel("Energy consumption (Wh)")
        ax.set_title(f"CHURCH — annual profile {year}")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

    return annual

# Exécution de l'application
if __name__ == '__main__':
    """MILLS"""
    """On fait la représentation du profil des mills sur une journée."""
    # 3 blocs (ex: ~0.5 h, 1.5 h, 1.0 h)
    # prof, segs = random_decortiquerie_daily_profile(n_blocks=3, plot=True, return_segments=True)
    # print("Segments (start,end) h:", segs)

    # Blocs aléatoires (1 à 4), durée ~3h ± 0.5h
    # mills = random_decortiquerie_daily_profile(plot=True)

    """Maintenant le profil des mills sur une année"""
    # annual = decortiquerie_annual_profile_2023(seed=2025, plot=True)

    """WATER PUMP"""
    """On fait la représentation du profil des water pumps sur une journée."""
    # water_pump = random_decortiquerie_daily_profile(power_w=3000.0,window_start_h=7.0, window_end_h=22.0, target_hours=4.0, jitter_hours=0.5, plot=True)

    """Maintenant le profil des water pumps sur une année"""
    # annual_wp = water_pump_annual_profile(year=2023, power_w=3000.0, window_start_h=7.0, window_end_h=22.0, target_hours=4.0, jitter_hours=0.5, plot=True)

    """SCHOOL"""
    # annual_school = school_annual_profile(school_odou, year=2023, noise_amp=100.0, seed=2025, plot=True)

    """HEALTH CENTER"""
    # annual_hc = health_center_annual_profile(health_center_odou, year=2023, noise_amp=100.0, seed=2025, plot=True)

    """PLACE OF WORSHIP"""
    annual_church = church_annual_profile(church_odou, year=2023, noise_amp=20.0, seed=2025, plot=True)

