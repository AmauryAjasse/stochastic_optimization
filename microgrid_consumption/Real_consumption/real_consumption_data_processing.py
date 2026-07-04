import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from typing import Optional, Union
import numpy as np


def format_energy_file(input_path, output_path):
    df = pd.read_csv(input_path, sep=';')

    # Conversion en datetime (très important)
    df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)

    # Conversion kWh → Wh
    df['kilowatt_hours'] = df['kilowatt_hours'] * 1000

    # Renommage
    df = df.rename(columns={
        'datetime': 'timestamp',
        'kilowatt_hours': 'aggregate_wh'
    })

    # Format EXACT comme ton fichier de référence
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Ordre des colonnes
    df = df[['timestamp', 'aggregate_wh']]

    # Sauvegarde
    df.to_csv(output_path, index=False)

    return df

format_energy_file(
    input_path="energy_total_2023.csv",
    output_path="one_year_formatted.csv"
)

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
    input_csv="one_year_formatted.csv",
    output_csv="24_days_formatted.csv",
    time_col="timestamp",
    value_col="aggregate_wh")