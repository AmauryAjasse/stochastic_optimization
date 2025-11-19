import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd

def plot_example(values, dt_hours, time_on: str = "x"):
    """
    Trace un profil sur une journée à partir d'une liste de valeurs et d'un pas de temps.

    Paramètres
    ----------
    values : list[float] | array-like
        Les valeurs échantillonnées sur la journée (ex: puissance, consommation, etc.).
    dt_hours : float
        Pas de temps en heures. Exemples:
          - 1.0  -> 24 points (horaire)
          - 0.25 -> 96 points (quart d'heure)
    time_on : {"x", "y"}
        Où placer l'axe du temps : "x" (abscisse, par défaut) ou "y" (ordonnée).

    Remarques
    ---------
    - Si len(values) = 24/dt_hours (à un arrondi près), l'axe temporel est borné à 0–24 h.
    - Sinon, on trace quand même, avec un avertissement, sur la durée len(values)*dt_hours.
    """
    if dt_hours <= 0:
        raise ValueError("dt_hours doit être strictement positif.")

    y = np.asarray(values, dtype=float)
    n = y.size
    if n == 0:
        raise ValueError("La liste 'values' est vide.")

    # Axe temps en heures
    t = np.arange(n) * dt_hours  # 0, dt, 2*dt, ...

    # Vérification "profil sur une journée"
    expected = 24.0 / dt_hours
    expected_int = int(round(expected))
    is_full_day = abs(expected - expected_int) < 1e-6 and (n == expected_int)

    if not is_full_day:
        print(
            f"⚠️ Avertissement: {n} points pour dt={dt_hours} h "
            f"(attendu ~{round(expected)} pour couvrir 24 h). "
            f"Durée tracée: {t[-1]:.2f} h.",
            flush=True
        )

    # Petite fonction pour étiquettes HH:MM
    def hhmm(hours_float):
        total_min = int(round(hours_float * 60))
        hh = (total_min // 60) % 24
        mm = total_min % 60
        return f"{hh:02d}:{mm:02d}"

    major_ticks = [0, 6, 12, 18, 24]

    fig, ax = plt.subplots()

    if time_on.lower() == "y":
        # Temps en ordonnée
        ax.plot(y, t, marker='o' if n <= 48 else None)
        ax.set_xlabel("Energie consommée (kWh)")
        ax.set_ylabel("Heure de la journée")
        if is_full_day:
            ax.set_ylim(0, 24)
            ax.set_yticks(major_ticks)
            ax.set_yticklabels([hhmm(h) for h in major_ticks])
        else:
            ax.set_yticks(np.linspace(0, t[-1], 5))
            ax.set_yticklabels([hhmm(h) for h in np.linspace(0, t[-1], 5)])
    else:
        # Temps en abscisse (par défaut)
        ax.plot(t, y, marker='o' if n <= 48 else None)
        ax.set_xlabel("Heure de la journée")
        ax.set_ylabel("Energie consommée (kWh)")
        if is_full_day:
            ax.set_xlim(0, 24)
            ax.set_xticks(major_ticks)
            ax.set_xticklabels([hhmm(h) for h in major_ticks])
        else:
            ax.set_xticks(np.linspace(0, t[-1], 5))
            ax.set_xticklabels([hhmm(h) for h in np.linspace(0, t[-1], 5)])

    ax.grid(True, alpha=0.3)
    ax.set_title("Profil sur 24h - Milling 10 kW")
    plt.tight_layout()
    plt.show()


# # --- Exemples d'utilisation ---
# if __name__ == "__main__":
#     # Exemple 1 : pas horaire (24 points)
#     vals_heure = [0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 2.5, 2.5, 0, 5, 0, 0, 0, 0, 0, 0]
#     plot_example(vals_heure, 1.0)              # temps en X (classique)
#     # plot_example(vals_heure, 1.0, time_on="y")  # temps en Y si vous préférez
#
#     # Exemple 2 : pas quart d'heure (96 points)
#     x = np.linspace(0, 2*np.pi, 96, endpoint=False)
#     vals_qh = [0, 0, 0, 0,
#                0, 0, 0, 0,
#                0, 0, 0, 0,
#                0, 0, 0, 0,
#                0, 0, 0, 0,
#                0, 0, 0, 0,
#                0, 0, 0, 0,
#                0, 0, 0, 0,
#                0, 0, 0, 0,
#                10, 10, 0, 0,
#                0, 10, 0, 10,
#                0, 0, 0, 0,
#                0, 0, 0, 0,
#                0, 0, 0, 0,
#                0, 10, 0, 0,
#                10, 0, 0, 0,
#                0, 0, 0, 0,
#                0, 0, 10, 10,
#                0, 0, 0, 0,
#                0, 0, 0, 0,
#                0, 0, 0, 0,
#                0, 0, 0, 0,
#                0, 0, 0, 0,
#                0, 0, 0, 0,]
#     plot_example(vals_qh, 0.25)

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

extract_first_and_15th_days(
    input_csv="meteo_data/temperature_solcast_formatted.csv",
    output_csv="meteo_data/temperature_24_days.csv",
    time_col="Time",
    value_col="Temperature")
