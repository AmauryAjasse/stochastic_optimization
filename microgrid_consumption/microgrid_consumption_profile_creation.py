
"""
microgrid_consumption_profiles.py — v2 (15‑min export + plotting)

But
----
Créer un profil de consommation annuel (pas 15 min) d’un micro‑réseau en
additionnant :
- les charges domestiques (N ménages) générées par domestic_loads_profiles.py ;
- les charges communautaires (école, centre de santé, église, pompe à eau, décortiquerie/mill)
  générées par community_wide_loads_profiles.py.

Nouveautés vs v1
----------------
- Conversion en pas de 15 minutes (35040 points) à partir des profils horaires (8760).
- Fonction unique `create_microgrid_profile_15min(...)` qui :
  * construit l’agrégat sur l’année,
  * convertit en 15 min,
  * sauvegarde le profil (CSV) dans le dossier `microgrid_consumption_examples/` avec un nom au choix,
  * et trace la série temporelle.

Entrée (cfg minimale)
---------------------
cfg = {
    "year": 2023,
    "households": 50,
    "community": {
        "school": 1,
        "health_center": 0,
        "church": 0,
        "water_pump": 1,
        "mills": 0
    },
    # Optionnel : paramètres détaillés
    # "domestic_params": {...},
    # "community_params": {...}
}

Sorties
-------
- aggregate_15min : liste de 35040 valeurs (Wh par pas de 15 min)
- details_15min   : dict { "domestic_total": [...], "school": [...], ... } en 15 min (si requested)

NB
--
Les fonctions domestiques/communautaires utilisées proviennent des scripts utilisateur :
`domestic_loads_profiles.py` et `community_wide_loads_profiles.py`.
"""

from typing import Dict, Tuple, List, Optional
import numpy as np

# Import des générateurs existants (fichiers fournis par l'utilisateur)
from microgrid_consumption import community_wide_loads_profiles as cl, domestic_loads_profiles as dl


# -------------------------------
#     Utilities (internal)
# -------------------------------

def _hours_to_quarter_hours(series_hourly: List[float]) -> List[float]:
    """
    Convertit un profil horaire (Wh/h) en pas de 15 minutes en supposant
    une répartition uniforme au sein de l'heure.
    -> Chaque heure h (Wh) est divisée en 4 quarts d'heure de Wh/4.
    """
    arr = np.asarray(series_hourly, dtype=float)
    # Répéter chaque valeur 4 fois puis diviser par 4 pour conserver l'énergie
    qh = np.repeat(arr, 4) / 4.0
    return qh.tolist()


def _build_domestic_annual(
    n_households: int,
    year: int = 2023,
    seed: Optional[int] = None,
    gaussians: Optional[dict] = None,
) -> List[float]:
    """
    Construit la somme des profils annuels DOMESTIQUES (8760 points) pour n ménages.
    On réutilise la logique existante du module 'dl'.
    """
    if gaussians is None:
        # Utilise tes gaussiennes horaires calculées sur les profils de référence
        gaussians = dl.fit_hourly_gaussians(dl.profiles_dict_percent)

    total = np.zeros(24 * 365, dtype=float)
    # Seeds indépendants par ménage pour reproductibilité
    if seed is not None:
        rng = np.random.default_rng(seed)
        hh_seeds = rng.integers(0, 2**32 - 1, size=n_households)
    else:
        hh_seeds = [None] * n_households

    for i in range(n_households):
        prof = dl.sample_annual_profile_from_gaussians(
            gaussians,
            seed=None if hh_seeds[i] is None else int(hh_seeds[i]),
            plot=False
        )
        total += np.asarray(prof, dtype=float)

    return total.tolist()


def _replicate(series: List[float], count: int) -> List[float]:
    """Répète/ajoute 'series' 'count' fois (somme)."""
    if count <= 0:
        return [0.0] * len(series)
    arr = np.asarray(series, dtype=float)
    return (arr * count).tolist()


def _community_component(name: str, count: int, year: int, params: Dict) -> List[float]:
    """
    Construit un composant communautaire annuel (8760 points) selon 'name'.
    """
    if count <= 0:
        return [0.0] * (24 * 365)

    name_l = name.lower()

    if name_l == "school":
        base = params.get("base", "odou")
        noise_amp = float(params.get("noise_amp", 100.0))
        seed = params.get("seed", None)
        if base.lower() == "odou":
            base_profile = cl.school_odou
        else:
            raise ValueError("Seule la base 'odou' est disponible pour school.")
        one = cl.school_annual_profile(base_profile, year=year, noise_amp=noise_amp, seed=seed, plot=False)
        return _replicate(one, count)

    if name_l == "health_center":
        base = params.get("base", "odou")
        noise_amp = float(params.get("noise_amp", 100.0))
        seed = params.get("seed", None)
        if base.lower() == "odou":
            base_profile = cl.health_center_odou
        else:
            raise ValueError("Seule la base 'odou' est disponible pour health_center.")
        one = cl.health_center_annual_profile(base_profile, year=year, noise_amp=noise_amp, seed=seed, plot=False)
        return _replicate(one, count)

    if name_l == "church":
        base = params.get("base", "odou")
        noise_amp = float(params.get("noise_amp", 20.0))
        seed = params.get("seed", None)
        if base.lower() == "odou":
            base_profile = cl.church_odou
        elif base.lower() == "williams":
            base_profile = cl.church_williams
        else:
            raise ValueError("Bases disponibles pour church: 'odou' ou 'williams'.")
        one = cl.church_annual_profile(base_profile, year=year, noise_amp=noise_amp, seed=seed, plot=False)
        return _replicate(one, count)

    if name_l == "water_pump":
        kw = {
            "year": year,
            "power_w": float(params.get("power_w", 3000.0)),
            "window_start_h": float(params.get("window_start_h", 7.0)),
            "window_end_h": float(params.get("window_end_h", 22.0)),
            "target_hours": float(params.get("target_hours", 4.0)),
            "jitter_hours": float(params.get("jitter_hours", 0.5)),
            "n_blocks": params.get("n_blocks", None),
            "min_block": float(params.get("min_block", 0.25)),
            "seed": params.get("seed", None),
            "plot": False,
        }
        one = cl.water_pump_annual_profile(**kw)
        return _replicate(one, count)

    if name_l == "mills":
        kw = {
            "power_w": float(params.get("power_w", 5000.0)),
            "window_start_h": float(params.get("window_start_h", 8.0)),
            "window_end_h": float(params.get("window_end_h", 18.0)),
            "target_hours": float(params.get("target_hours", 3.0)),
            "jitter_hours": float(params.get("jitter_hours", 0.5)),
            "n_blocks": params.get("n_blocks", None),
            "min_block": float(params.get("min_block", 0.25)),
            "seed": params.get("seed", None),
            "plot": False,
        }
        one = cl.decortiquerie_annual_profile_2023(**kw)
        return _replicate(one, count)

    raise ValueError(f"Type communautaire inconnu: '{name}'")


def build_microgrid_annual_profile(
    cfg: Dict,
    return_components: bool = True
) -> Tuple[List[float], Dict[str, List[float]]]:
    """
    Construit le profil *horaire* (8760) du micro-réseau et, en option, les composants.
    """
    year = int(cfg.get("year", 2023))
    n_households = int(cfg.get("households", 0))

    # Params facultatifs
    domestic_params = cfg.get("domestic_params", {}) or {}
    community_counts = (cfg.get("community", {}) or {})
    community_params = (cfg.get("community_params", {}) or {})

    # ---- DOMESTIC ----
    seed_domestic = domestic_params.get("seed", None)
    gaussians = domestic_params.get("gaussians", None)
    domestic_total = _build_domestic_annual(
        n_households=n_households,
        year=year,
        seed=seed_domestic,
        gaussians=gaussians,
    )

    # ---- COMMUNITY ----
    components = {"domestic_total": domestic_total}
    for name in ["school", "health_center", "church", "water_pump", "mills"]:
        count = int(community_counts.get(name, 0))
        params = community_params.get(name, {}) if isinstance(community_params, dict) else {}
        comp = _community_component(name, count, year, params)
        components[name] = comp

    # ---- AGRÉGAT HORAIRE ----
    agg_hourly = np.zeros(24 * 365, dtype=float)
    for series in components.values():
        agg_hourly += np.asarray(series, dtype=float)

    return (agg_hourly.tolist(), components) if return_components else (agg_hourly.tolist(), {})


def create_microgrid_profile_15min(
    cfg: Dict,
    filename: str,
    outdir: str = "microgrid_consumption_examples",
    save_components: bool = False,
    plot: bool = True,
) -> Tuple[List[float], Optional[Dict[str, List[float]]], str]:
    """
    Construit l'agrégat 15 min (35040), sauvegarde le CSV et (optionnel) trace la figure.

    Args
    ----
    cfg : dictionnaire de configuration (voir docstring en tête de fichier).
    filename : nom du fichier (ex: 'village_A_2023.csv').
    outdir : dossier de sortie (créé s'il n'existe pas).
    save_components : si True, enregistre aussi les colonnes par composante.
    plot : si True, affiche un tracé de la série 15 min.

    Returns
    -------
    aggregate_15min : liste de 35040 valeurs (Wh/15min)
    details_15min   : dict {comp: liste 35040} si save_components=True, sinon None
    filepath        : chemin du fichier CSV sauvegardé
    """
    year = int(cfg.get("year", 2023))

    # 1) Profil *horaire* + composants
    agg_hourly, comps_hourly = build_microgrid_annual_profile(cfg, return_components=True)

    # 2) Conversion en 15 min
    agg_qh = _hours_to_quarter_hours(agg_hourly)

    details_qh = None
    if save_components:
        details_qh = {k: _hours_to_quarter_hours(v) for k, v in comps_hourly.items()}

    # 3) DataFrame + index temporel (UTC naïf)
    n_qh = 24 * 4 * 365  # 35040
    start = f"{year}-01-01 00:00:00"
    index = pd.date_range(start=start, periods=n_qh, freq="15min")

    data = {"aggregate_wh": agg_qh}
    if save_components and details_qh is not None:
        for k, v in details_qh.items():
            data[k] = v

    df = pd.DataFrame(data, index=index)

    # 4) Sauvegarde
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    filepath = str(out_path / filename)
    df.to_csv(filepath, index_label="timestamp")

    # 5) Tracé
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.plot(df.index, df["aggregate_wh"])
        mean_daily = df["aggregate_wh"].sum() / 365.0
        plt.title(f"Microgrid annual load — 15‑min profile (mean daily = {mean_daily:.1f} Wh)")
        plt.xlabel("Time")
        plt.ylabel("Energy per 15‑min (Wh)")
        plt.tight_layout()
        plt.show()

    return agg_qh, details_qh, filepath

import pandas as pd
from pathlib import Path

def extract_first_and_15th_days(input_csv: str, output_csv: str, time_col: str = "timestamp", value_col: str = "aggregate_wh"):
    """
    Crée un nouveau CSV ne contenant que les jours 1 et 15 de chaque mois,
    puis réécrit ces 24 jours avec des dates consécutives du 1er au 24 janvier.

    Args:
        input_csv : chemin du CSV source (avec colonnes temps + consommation)
        output_csv : chemin du nouveau CSV à créer
        time_col : nom de la colonne temporelle dans le fichier d'entrée (défaut: 'timestamp')
        value_col : nom de la colonne de consommation (défaut: 'aggregate_wh')
    """

    # 1. Charger le fichier
    df = pd.read_csv(input_csv, parse_dates=[time_col])
    df = df.sort_values(time_col)

    # 2. Extraire le jour du mois
    df["day"] = df[time_col].dt.day
    df["month"] = df[time_col].dt.month

    # 3. Garder seulement les 1er et 15e jours de chaque mois
    df_selected = df[df["day"].isin([1, 15])].copy()

    # Vérification qu'on a bien 24 jours (2 par mois)
    n_days = df_selected[time_col].dt.date.nunique()
    if n_days != 24:
        print(f"⚠️ Attention : {n_days} jours trouvés au lieu de 24 (certains mois manquent peut-être).")

    # 4. Réindexer les jours pour créer des dates consécutives
    # On garde les intervalles d'origine (par ex. 15min)
    df_selected = df_selected.reset_index(drop=True)
    freq = pd.infer_freq(df_selected[time_col])
    if freq is None:
        # si la fréquence ne peut être déduite, on suppose 15min
        freq = "15min"

    n_points = len(df_selected)
    new_index = pd.date_range(start="2023-01-01", periods=n_points, freq=freq)
    df_selected[time_col] = new_index

    # 5. Supprimer les colonnes auxiliaires
    df_selected = df_selected[[time_col, value_col]]

    # 6. Sauvegarder le nouveau CSV
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_selected.to_csv(output_csv, index=False)

    print(f"✅ Nouveau CSV créé : {output_csv}")
    print(f"   {len(df_selected):,} points de données (du {new_index[0].date()} au {new_index[-1].date()})")

    return df_selected


# -------------------------------
#        Example (optional)
# -------------------------------
if __name__ == "__main__":
    # Exemple d'utilisation : l'utilisateur peut changer filename à la volée.
    cfg = {
        "year": 2023,
        "households": 15,
        "community": {"school": 1, "health_center": 1, "church": 1, "water_pump": 1, "mills": 0},
        "domestic_params": {
            # "seed": 2025,
            # "gaussians": <dict de gaussiennes horaires si tu veux override>
        },
        "community_params": {
            "school": {"base": "odou", "noise_amp": 100.0},
            "health_center": {"base": "odou", "noise_amp": 100.0},
            "church": {"base": "odou", "noise_amp": 20.0},
            "water_pump": {"power_w": 3000.0, "window_start_h": 7.0, "window_end_h": 22.0,
                           "target_hours": 4.0, "jitter_hours": 0.5},
            "mills": {"power_w": 5000.0, "window_start_h": 8.0, "window_end_h": 18.0,
                      "target_hours": 3.0, "jitter_hours": 0.5}
        }
    }

    # Le nom de fichier est à choisir par l'utilisateur :
    # _filename = "one_year_example_5.csv"
    # agg_qh, details_qh, path = create_microgrid_profile_15min(cfg, filename=_filename, save_components=False, plot=True)
    # print(f"Saved: {path}  |  points={len(agg_qh)}  |  annual Wh={sum(agg_qh):.1f}")

    extract_first_and_15th_days(
        input_csv="microgrid_consumption_examples/one_year_example_5.csv",
        output_csv="microgrid_consumption_examples/24_days_example_5.csv",
        time_col="timestamp",
        value_col="aggregate_wh")
