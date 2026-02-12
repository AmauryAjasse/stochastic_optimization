import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# def extract_first_and_15th_days(input_csv: str, output_csv: str, time_col: str = "timestamp", value_col: str = "aggregate_wh"):
#     """
#     Crée un nouveau CSV ne contenant que les jours 1 et 15 de chaque mois,
#     puis réécrit ces 24 jours avec des dates consécutives du 1er au 24 janvier.
#
#     Args:
#         input_csv : chemin du CSV source (avec colonnes temps + consommation)
#         output_csv : chemin du nouveau CSV à créer
#         time_col : nom de la colonne temporelle dans le fichier d'entrée (défaut: 'timestamp')
#         value_col : nom de la colonne de consommation (défaut: 'aggregate_wh')
#     """
#
#     # 1. Charger le fichier
#     df = pd.read_csv(input_csv, parse_dates=[time_col])
#     df = df.sort_values(time_col)
#
#     # 2. Extraire le jour du mois
#     df["day"] = df[time_col].dt.day
#     df["month"] = df[time_col].dt.month
#
#     # 3. Garder seulement les 1er et 15e jours de chaque mois
#     df_selected = df[df["day"].isin([1, 15])].copy()
#
#     # Vérification qu'on a bien 24 jours (2 par mois)
#     n_days = df_selected[time_col].dt.date.nunique()
#     if n_days != 24:
#         print(f"⚠️ Attention : {n_days} jours trouvés au lieu de 24 (certains mois manquent peut-être).")
#
#     # 4. Réindexer les jours pour créer des dates consécutives
#     # On garde les intervalles d'origine (par ex. 15min)
#     df_selected = df_selected.reset_index(drop=True)
#     freq = pd.infer_freq(df_selected[time_col])
#     if freq is None:
#         # si la fréquence ne peut être déduite, on suppose 15min
#         freq = "15min"
#
#     n_points = len(df_selected)
#     new_index = pd.date_range(start="2023-01-01", periods=n_points, freq=freq)
#     df_selected[time_col] = new_index
#
#     # 5. Supprimer les colonnes auxiliaires
#     df_selected = df_selected[[time_col, value_col]]
#
#     # 6. Sauvegarder le nouveau CSV
#     Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
#     df_selected.to_csv(output_csv, index=False)
#
#     print(f"✅ Nouveau CSV créé : {output_csv}")
#     print(f"   {len(df_selected):,} points de données (du {new_index[0].date()} au {new_index[-1].date()})")
#
#     return df_selected


def extract_first_and_15th_days(
    input_csv: str,
    output_csv: str,
    time_col: str = "Time",
    value_col: str = "Irradiance",
    freq: str = "15min",
    fill_value: float = 0.0,
):
    """
    1) Garde uniquement les jours 1 et 15 de chaque mois (sur le fichier Solcast formaté)
    2) Complète chaque journée sur une grille complète (00:00 -> 23:45) au pas 'freq'
       (les points manquants sont remplis par fill_value, typiquement 0 pour l'irradiance)
    3) Recolle ces 24 jours les uns après les autres en réécrivant une timeline consécutive
       à partir du 2023-01-01 00:00:00.
    """
    df = pd.read_csv(input_csv)
    if time_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"Colonnes attendues: {time_col} et {value_col}. Colonnes trouvées: {list(df.columns)}")

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)

    # Sélection jours 1 et 15
    df_sel = df[df[time_col].dt.day.isin([1, 15])][[time_col, value_col]].copy()

    # Construire une journée complète pour chaque date sélectionnée
    df_sel["date"] = df_sel[time_col].dt.normalize()

    days = []
    for d, g in df_sel.groupby("date"):
        g = g.set_index(time_col)[[value_col]].sort_index()

        # grille complète 00:00 -> 23:45 au pas freq
        day_index = pd.date_range(start=d, end=d + pd.Timedelta(days=1) - pd.Timedelta(freq), freq=freq)

        g_full = g.reindex(day_index)
        g_full[value_col] = g_full[value_col].fillna(fill_value)  # irradiance manquante -> 0
        g_full = g_full.reset_index().rename(columns={"index": time_col})

        days.append(g_full)

    df_full = pd.concat(days, ignore_index=True)

    # Vérif jours uniques
    n_days = df_full[time_col].dt.normalize().nunique()
    if n_days != 24:
        print(f"⚠️ Attention : {n_days} jours obtenus au lieu de 24 (mois manquants ou données incomplètes).")

    # Recollement consécutif (24 jours à partir du 1er janvier)
    n_points = len(df_full)
    new_index = pd.date_range(start="2023-01-01 00:00:00", periods=n_points, freq=freq)
    df_full[time_col] = new_index

    # Sauvegarde
    df_full = df_full[[time_col, value_col]]
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_full.to_csv(output_csv, index=False)

    print(f"✅ Nouveau CSV créé : {output_csv} ({len(df_full):,} points)")
    return df_full

# extract_first_and_15th_days(
#     input_csv="irradiance_solcast_formatted.csv",
#     output_csv="irradiance_24_days.csv",
#     time_col="Time",
#     value_col="Irradiance")

def sample_meteo_15min_to_30min(irradiance_csv: str,
                                temperature_csv: str,
                                time_col: str = "Time",
                                irr_col: str = "Irradiance",
                                temp_col: str = "Temperature") -> None:
    """
    Crée des fichiers météo au pas 30min à partir de fichiers 15min, SANS agrégation :
    on conserve strictement les valeurs existantes aux timestamps 00 et 30 minutes
    (ex: 16:00 dans le nouveau fichier == 16:00 dans l'ancien).

    Sorties (même dossier) :
      irradiance_x.csv  -> irradiance_x_30min.csv
      temperature_x.csv -> temperature_x_30min.csv
    """

    def _strict_keep_every_30min(csv_path: str, value_col: str) -> str:
        df = pd.read_csv(csv_path)

        if time_col not in df.columns or value_col not in df.columns:
            raise ValueError(
                f"Fichier {csv_path} : colonnes attendues '{time_col}' et '{value_col}', "
                f"colonnes trouvées = {list(df.columns)}"
            )

        # parse dates
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col)

        # on garde uniquement les lignes sur des timestamps alignés 30 min (minute 0 ou 30)
        mask = df[time_col].dt.minute.isin([0, 30]) & (df[time_col].dt.second == 0)
        df_30 = df.loc[mask, [time_col, value_col]].copy()

        # sécurité : si ton fichier contient des pas 15 min réguliers, tu peux aussi vérifier
        # que les minutes restantes sont bien 0/15/30/45
        # (je ne bloque pas ici, mais tu peux le faire si tu veux)

        # format timestamp identique
        df_30[time_col] = df_30[time_col].dt.strftime("%Y-%m-%d %H:%M:%S")

        base, ext = os.path.splitext(csv_path)
        out_path = f"{base}_30min{ext}"
        df_30.to_csv(out_path, index=False)
        return out_path

    out_irr = _strict_keep_every_30min(irradiance_csv, irr_col)
    out_tmp = _strict_keep_every_30min(temperature_csv, temp_col)

    print(f"✅ Irradiance 30min (strict) créé : {out_irr}")
    print(f"✅ Température 30min (strict) créé : {out_tmp}")


# sample_meteo_15min_to_30min("irradiance_24_days.csv",
#                              "temperature_24_days.csv")

def sample_meteo_15min_to_1h(
    irradiance_csv: str,
    temperature_csv: str,
    time_col: str = "Time",
    irr_col: str = "Irradiance",
    temp_col: str = "Temperature",
) -> None:
    """
    Crée des fichiers météo au pas 1h à partir de fichiers 15min, SANS agrégation :
    on conserve strictement les valeurs existantes aux timestamps pile à l’heure
    (ex: 16:00 dans le nouveau fichier == 16:00 dans l'ancien).

    Sorties (même dossier) :
      irradiance_x.csv  -> irradiance_x_1h.csv
      temperature_x.csv -> temperature_x_1h.csv
    """

    def _strict_keep_every_1h(csv_path: str, value_col: str) -> str:
        df = pd.read_csv(csv_path)

        if time_col not in df.columns or value_col not in df.columns:
            raise ValueError(
                f"Fichier {csv_path} : colonnes attendues '{time_col}' et '{value_col}', "
                f"colonnes trouvées = {list(df.columns)}"
            )

        # parsing datetime + tri
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col)

        # on garde uniquement les timestamps pile à l'heure
        mask = (
            (df[time_col].dt.minute == 0)
            & (df[time_col].dt.second == 0)
        )
        df_1h = df.loc[mask, [time_col, value_col]].copy()

        # format timestamp identique à l'entrée
        df_1h[time_col] = df_1h[time_col].dt.strftime("%Y-%m-%d %H:%M:%S")

        base, ext = os.path.splitext(csv_path)
        out_path = f"{base}_1h{ext}"
        df_1h.to_csv(out_path, index=False)

        return out_path

    out_irr = _strict_keep_every_1h(irradiance_csv, irr_col)
    out_tmp = _strict_keep_every_1h(temperature_csv, temp_col)

    print(f"✅ Irradiance 1h (strict) créé : {out_irr}")
    print(f"✅ Température 1h (strict) créé : {out_tmp}")

sample_meteo_15min_to_1h("irradiance_24_days.csv",
                             "temperature_24_days.csv")