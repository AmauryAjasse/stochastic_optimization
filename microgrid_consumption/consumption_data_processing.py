import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ==========
# Paramètres
# ==========

# Nom du dossier contenant les scénarios (à adapter si besoin)
SCENARIOS_DIR_NAME = "scenarios_one_year"

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

def summarize_durations_and_timesteps(show_plot=True):
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

    print("=== Durée totale et pas de temps moyen par scénario ===")
    for name, duration_days, timestep_min in results:
        print(f"- {name}:")
        print(f"    Durée totale ≈ {duration_days:.2f} jours")
        print(f"    Pas de temps moyen ≈ {timestep_min:.2f} minutes")

    if show_plot:
        scenario_names = [r[0] for r in results]
        duration_days_list = [r[1] for r in results]
        timestep_min_list = [r[2] for r in results]

        # Durée en jours
        plt.figure()
        plt.bar(scenario_names, duration_days_list)
        plt.ylabel("Durée totale (jours)")
        plt.title("Durée totale par scénario")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Pas de temps moyen en minutes
        plt.figure()
        plt.bar(scenario_names, timestep_min_list)
        plt.ylabel("Pas de temps moyen (minutes)")
        plt.title("Pas de temps moyen par scénario")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        plt.show()


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

        print(f"- {name}:")
        print(f"    Énergie totale ≈ {total_kWh:.2f} kWh")
        print(f"    Durée ≈ {duration_days:.2f} jours")
        print(f"    Énergie moyenne ≈ {avg_energy_kWh_per_day:.2f} kWh/jour")

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


# =====================
# Exemple d'utilisation
# =====================

if __name__ == "__main__":
    summarize_durations_and_timesteps(show_plot=False)
    # summarize_average_energy(show_plot=False)
    # summarize_energy_by_time_windows(show_plot=False)

    # extract_first_and_15th_days(
    #     input_csv="scenarios_one_year/one_year_example_1.csv",
    #     output_csv="scenarios_24_days/24_days_example_1.csv",
    #     time_col="timestamp",
    #     value_col="aggregate_wh")
