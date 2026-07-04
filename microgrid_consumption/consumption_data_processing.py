import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from typing import Optional, Union
import numpy as np

# ==========
# Paramètres
# ==========

# Nom du dossier contenant les scénarios (à adapter si besoin)
SCENARIOS_DIR_NAME = "microgrid_consumption_examples"

# Dossier racine du projet : ici on suppose que ce script est dans "microgrid_consumption"
BASE_DIR = Path(__file__).resolve().parent
SCENARIOS_DIR = BASE_DIR / SCENARIOS_DIR_NAME


# ==========================
# Fonctions utilitaires I/O
# ==========================

def load_all_scenarios():
    """
    Charge tous les fichiers CSV du dossier SCENARIOS_DIR.
    Retourne un dictionnaire {nom_scenario: dataframe}.
    On parse 'timestamp' comme datetime.
    """
    if not SCENARIOS_DIR.exists():
        raise FileNotFoundError(f"Le dossier des scénarios n'existe pas : {SCENARIOS_DIR}")

    csv_files = sorted(SCENARIOS_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Aucun fichier .csv trouvé dans {SCENARIOS_DIR}")

    scenarios = {}
    for csv_path in csv_files:
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        scenarios[csv_path.stem] = df

    return scenarios


def _compute_duration_and_timestep(df):
    """
    Calcule la durée totale et le pas de temps moyen d'un scénario.
    Retour :
        duration (Timedelta),
        timestep_mean (Timedelta)
    """
    ts = df["timestamp"].sort_values()
    dt = ts.diff().dropna()

    if dt.empty:
        # Un seul point -> durée nulle, pas de temps non défini
        return pd.Timedelta(0), pd.NaT

    timestep_mean = dt.mean()
    # Durée = dernière - première + 1 pas de temps moyen
    duration = ts.iloc[-1] - ts.iloc[0] + timestep_mean
    return duration, timestep_mean


# ============================================
# 1) Durée totale & pas de temps moyen/scénario
# ============================================

def summarize_durations_and_timesteps():
    """
    Affiche la durée totale et le pas de temps moyen pour chaque scénario.
    Optionnellement, produit deux graphiques barres :
        - durée en jours
        - pas de temps moyen en minutes
    """
    scenarios = load_all_scenarios()

    results = []
    for name, df in scenarios.items():
        duration, timestep_mean = _compute_duration_and_timestep(df)
        duration_days = duration.total_seconds() / 86400 if duration is not pd.NaT else float("nan")
        timestep_min = timestep_mean.total_seconds() / 60 if pd.notna(timestep_mean) else float("nan")
        results.append((name, duration_days, timestep_min))

    # ===== Tableau tabulate =====
    print("\n=== Durée totale et pas de temps moyen par scénario ===\n")
    headers = ["Scénario", "Durée (jours)", "Δt moyen (min)"]

    print(tabulate(results, headers=headers, floatfmt=".2f", tablefmt="psql"))
    print("\n")


# ==================================================
# 2) Énergie moyenne consommée sur la durée/scénario
# ==================================================

def summarize_average_energy(show_plot=True):
    """
    Calcule et affiche, pour chaque scénario :
        - l'énergie totale (kWh)
        - la durée en jours
        - l'énergie moyenne par jour (kWh/jour)

    Suppose que la colonne 'aggregate_wh' contient l'énergie consommée
    sur chaque pas de temps (en Wh).

    Produit un graphique en barres de l'énergie moyenne par jour (kWh/jour).
    """
    scenarios = load_all_scenarios()

    results = []
    print("=== Énergie moyenne consommée par scénario ===")
    for name, df in scenarios.items():
        duration, _ = _compute_duration_and_timestep(df)
        duration_days = duration.total_seconds() / 86400 if duration is not pd.NaT else float("nan")

        total_Wh = df["aggregate_wh"].sum()
        total_kWh = total_Wh / 1000.0

        if duration_days > 0:
            avg_energy_kWh_per_day = total_kWh / duration_days
        else:
            avg_energy_kWh_per_day = float("nan")

        results.append((name, total_kWh, duration_days, avg_energy_kWh_per_day))

    # ----- Tableau tabulate -----
    headers = ["Scénario", "Total (kWh)", "Durée (jours)", "Énergie moyenne (kWh/jour)"]

    print(tabulate(results, headers=headers, floatfmt=".2f", tablefmt="psql"))

    if show_plot:
        scenario_names = [r[0] for r in results]
        avg_kWh_per_day_list = [r[3] for r in results]

        plt.figure()
        plt.bar(scenario_names, avg_kWh_per_day_list)
        plt.ylabel("Énergie moyenne (kWh/jour)")
        plt.title("Énergie moyenne consommée par scénario")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()


# ===============================================================
# 3) % d'énergie consommée 06–18h et 18–22h pour chaque scénario
# ===============================================================

def summarize_energy_by_time_windows(show_plot=True):
    """
    Pour chaque scénario, calcule le pourcentage d'énergie consommée :
        - entre 06:00 et 18:00
        - entre 18:00 et 22:00
        - le reste du temps (pour info)

    On considère que 'aggregate_wh' est l'énergie associée au timestamp
    de la ligne.

    Produit un graphique en barres empilées (stacked bar) montrant, pour
    chaque scénario, la répartition (% du total) :
        - 06–18h
        - 18–22h
        - Autres heures
    """
    scenarios = load_all_scenarios()

    results = []

    print("=== Répartition de l'énergie par fenêtres horaires ===")
    for name, df in scenarios.items():
        ts = df["timestamp"]
        # heure sous forme décimale (ex: 6h15 -> 6.25)
        hour_float = ts.dt.hour + ts.dt.minute / 60.0

        mask_day = (hour_float >= 6) & (hour_float < 18)
        mask_evening = (hour_float >= 18) & (hour_float < 22)

        E_total = df["aggregate_wh"].sum()
        if E_total <= 0:
            # Éviter division par zéro
            pct_day = pct_evening = pct_other = float("nan")
        else:
            E_day = df.loc[mask_day, "aggregate_wh"].sum()
            E_evening = df.loc[mask_evening, "aggregate_wh"].sum()
            E_other = E_total - E_day - E_evening

            pct_day = 100.0 * E_day / E_total
            pct_evening = 100.0 * E_evening / E_total
            pct_other = 100.0 * E_other / E_total

        results.append((name, pct_day, pct_evening, pct_other))

        print(f"- {name}:")
        print(f"    06–18h  : {pct_day:.1f} % de l'énergie totale")
        print(f"    18–22h  : {pct_evening:.1f} % de l'énergie totale")
        print(f"    Autres  : {pct_other:.1f} % de l'énergie totale")

    if show_plot:
        scenario_names = [r[0] for r in results]
        pct_day_list = [r[1] for r in results]
        pct_evening_list = [r[2] for r in results]
        pct_other_list = [r[3] for r in results]

        # Barres empilées
        plt.figure()
        bottom_evening = pct_day_list
        bottom_other = [d + e for d, e in zip(pct_day_list, pct_evening_list)]

        plt.bar(scenario_names, pct_day_list, label="06–18h")
        plt.bar(scenario_names, pct_evening_list, bottom=bottom_evening, label="18–22h")
        plt.bar(scenario_names, pct_other_list, bottom=bottom_other, label="Autres heures")

        plt.ylabel("Part de l'énergie (%)")
        plt.title("Répartition de l'énergie par fenêtres horaires")
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.show()


# ===============================================================
# 4) passage de courbe sur une année à courbe sur 24 jours
# ===============================================================
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

def create_synthetic_load_profile_like_24days(
    ref_csv_path: str,
    out_csv_path: str,
    timestep: str = "15min",              # "15min" | "30min" | "1h"
    mean_daily_energy_wh: float = 20000,  # Wh/jour
    window_start_h: float = 9.0,          # heure décimale (ex: 9.5 = 09:30)
    window_end_h: float = 16.0,           # heure décimale
    window_energy_frac: float = 0.95,     # fraction entre 0 et 1 (95% -> 0.95)
    start_date: str = "2023-01-01",
    noise_frac: float = 0.0,              # 0.0 = profil parfaitement plat ; ex: 0.10 = ±10% (bruit)
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Génère un profil synthétique (Wh par pas) sur N jours, où N est le même
    que dans ref_csv_path (ex: 24_days_example_1.csv), et exporte un CSV
    au format: timestamp,aggregate_wh (même format que 24_days_example_1.csv).

    - mean_daily_energy_wh : énergie moyenne consommée par jour (Wh/jour)
    - window_energy_frac : part de l'énergie journalière consommée entre window_start_h et window_end_h
      (ex: 0.95 => 95% de l'énergie entre 9h et 16h)
    - timestep : "15min", "30min" ou "1h"
    - window_start_h / window_end_h : heures décimales (ex: 9.0, 16.0, 18.5)
    - noise_frac : ajoute un léger bruit multiplicatif par pas (optionnel)
    """
    # ---------- checks ----------
    timestep = timestep.lower().strip()
    allowed = {"15min": 15, "30min": 30, "1h": 60, "60min": 60}
    if timestep not in allowed:
        raise ValueError(f"timestep='{timestep}' invalide. Valeurs possibles: '15min', '30min', '1h'.")

    dt_min = allowed[timestep]
    if mean_daily_energy_wh <= 0:
        raise ValueError("mean_daily_energy_wh doit être > 0.")
    if not (0.0 < window_energy_frac < 1.0):
        raise ValueError("window_energy_frac doit être entre 0 et 1 (ex: 0.95).")
    if not (0.0 <= window_start_h < 24.0) or not (0.0 <= window_end_h < 24.0):
        raise ValueError("window_start_h et window_end_h doivent être dans [0, 24).")
    if window_end_h <= window_start_h:
        raise ValueError("Pour l’instant, on suppose window_end_h > window_start_h (fenêtre dans la journée).")
    if noise_frac < 0:
        raise ValueError("noise_frac doit être >= 0.")

    # ---------- 1) lire le CSV de référence pour récupérer N jours ----------
    df_ref = pd.read_csv(ref_csv_path)
    if not {"timestamp", "aggregate_wh"}.issubset(df_ref.columns):
        raise ValueError(
            f"Colonnes attendues: timestamp, aggregate_wh. Colonnes trouvées: {set(df_ref.columns)}"
        )

    df_ref["timestamp"] = pd.to_datetime(df_ref["timestamp"])
    n_days = df_ref["timestamp"].dt.date.nunique()
    if n_days <= 0:
        raise ValueError("Impossible de déterminer le nombre de jours dans le CSV de référence.")

    # ---------- 2) construire l'index temporel sur N jours ----------
    steps_per_day = int((24 * 60) / dt_min)
    n_points = n_days * steps_per_day
    freq_str = "15min" if dt_min == 15 else ("30min" if dt_min == 30 else "1h")
    index = pd.date_range(start=f"{start_date} 00:00:00", periods=n_points, freq=freq_str)

    # ---------- 3) répartir l'énergie journalière ----------
    # Fenêtre [start, end) en minutes
    w_start_min = int(round(window_start_h * 60))
    w_end_min = int(round(window_end_h * 60))
    # indices de pas dans une journée
    start_k = int(np.floor(w_start_min / dt_min))
    end_k = int(np.floor(w_end_min / dt_min))
    if end_k <= start_k:
        raise ValueError("Fenêtre trop petite (end_k <= start_k) avec ce pas de temps. Ajuste les heures ou le timestep.")

    n_in = end_k - start_k
    n_out = steps_per_day - n_in
    if n_out <= 0:
        raise ValueError("La fenêtre couvre toute la journée : mets une fenêtre plus petite.")

    e_in = mean_daily_energy_wh * window_energy_frac
    e_out = mean_daily_energy_wh * (1.0 - window_energy_frac)

    base_in = e_in / n_in
    base_out = e_out / n_out

    # Profil journalier "plat" (2 niveaux)
    day_profile = np.full(steps_per_day, base_out, dtype=float)
    day_profile[start_k:end_k] = base_in

    # ---------- 4) répéter sur N jours + bruit optionnel ----------
    profile = np.tile(day_profile, n_days)

    if noise_frac > 0.0:
        rng = np.random.default_rng(seed)
        # bruit multiplicatif centré sur 1 : (1 + eps), eps ~ U[-noise_frac, +noise_frac]
        eps = rng.uniform(-noise_frac, +noise_frac, size=profile.shape[0])
        profile = profile * (1.0 + eps)
        profile = np.clip(profile, 0.0, None)

        # renormalisation : conserver EXACTEMENT mean_daily_energy_wh par jour
        profile_reshaped = profile.reshape((n_days, steps_per_day))
        sums = profile_reshaped.sum(axis=1)
        # éviter division par 0
        scale = np.where(sums > 0, mean_daily_energy_wh / sums, 1.0)
        profile = (profile_reshaped * scale[:, None]).reshape(-1)

    # ---------- 5) export CSV au même format ----------
    df_out = pd.DataFrame({
        "timestamp": index.strftime("%Y-%m-%d %H:%M:%S"),
        "aggregate_wh": profile
    })

    Path(os.path.dirname(out_csv_path) or ".").mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv_path, index=False)

    return df_out


# =====================
# Exemple d'utilisation
# =====================

if __name__ == "__main__":
    summarize_durations_and_timesteps()
    summarize_average_energy(show_plot=False)
    summarize_energy_by_time_windows(show_plot=False)

    # extract_first_and_15th_days(
    #     input_csv="scenarios_one_year/one_year_example_1.csv",
    #     output_csv="scenarios_24_days/24_days_example_1.csv",
    #     time_col="timestamp",
    #     value_col="aggregate_wh")

    # df = create_synthetic_load_profile_like_24days(
    #     ref_csv_path="scenarios_24_days/24_days_example_1.csv",
    #     out_csv_path="scenarios_24_days/24_days_manual_example_1h.csv",
    #     timestep="1h",
    #     mean_daily_energy_wh=24000,  # 24 kWh/j
    #     window_start_h=9.0,
    #     window_end_h=16.0,
    #     window_energy_frac=0.95,
    #     noise_frac=0.05,  # optionnel
    #     seed=123
    # )
