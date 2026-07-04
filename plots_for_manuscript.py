import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import json

from pyomo.environ import *
from tabulate import tabulate
from lms2.tools.post_processing import *
from mpl_toolkits.mplot3d import Axes3D
import os
import pickle
import re

pd.set_option('display.max_rows', None)      # toutes les lignes
pd.set_option('display.max_columns', None)   # toutes les colonnes
pd.set_option('display.width', None)         # largeur illimitée
pd.set_option('display.max_colwidth', None)  # contenu complet des colonnes

def plot_from_csv(file_path, x_col, y_col, sort_x=True):
    """
    Trace y_col en fonction de x_col à partir d'un fichier CSV.

    Paramètres
    ----------
    file_path : str
        Chemin du fichier CSV
    x_col : str
        Nom de la colonne à mettre en abscisse
    y_col : str
        Nom de la colonne à mettre en ordonnée
    sort_x : bool
        Trie les données selon x (recommandé pour des courbes propres)
    """

    # Lecture du fichier
    df = pd.read_csv(file_path)

    # Vérification des colonnes
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Colonnes disponibles : {df.columns.tolist()}")

    # Tri optionnel
    if sort_x:
        df = df.sort_values(by=x_col)

    # Plot
    plt.figure()
    plt.plot(df[x_col], df[y_col])

    # Labels
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} en fonction de {x_col}")

    plt.grid()
    plt.show()

def plot_multiple_csv_same_graph(file_paths, x_col, y_col, sort_x=True):
    """
    Trace y_col en fonction de x_col pour plusieurs fichiers CSV sur le même graphe.

    Paramètres
    ----------
    file_paths : list[str]
        Liste des chemins vers les fichiers CSV
    x_col : str
        Nom de la colonne à mettre en abscisse
    y_col : str
        Nom de la colonne à mettre en ordonnée
    sort_x : bool
        Trie les données selon x dans chaque fichier
    """

    plt.figure()

    for file_path in file_paths:
        # Lecture du fichier
        df = pd.read_csv(file_path)

        # Vérification des colonnes
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(
                f"Dans le fichier {file_path}, colonnes disponibles : {df.columns.tolist()}"
            )

        # Tri optionnel
        if sort_x:
            df = df.sort_values(by=x_col)

        # Récupération du label depuis le nom du fichier
        file_name = os.path.basename(file_path)
        match = re.search(r's(\d+)', file_name.lower())

        if match:
            label = f"{match.group(1)}%"
        else:
            label = file_name  # fallback si pas de s85, s90, etc.

        if "deterministic" in file_name.lower():
            linestyle = "--"   # pointillé
        else:
            linestyle = "-"    # trait plein

        # Tracé
        plt.plot(df[x_col], df[y_col], label=label, linestyle=linestyle)

    # Labels
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} en fonction de {x_col}")
    plt.legend()
    plt.grid()
    plt.show()

def plot_cost_heatmap_mean_by_scenarios(
    df: pd.DataFrame,
    out_dir: Path,
    x_col: str = "pv_wp",
    y_col: str = "p0",
    value_col: str = "total_cost",
    scenario_col: str = "scenario",
    filename: str = "HEATMAP_mean_total_cost.png",
    title: str = "Coût total moyen (moyenne sur scénarios) en fonction de (p0, p_wp)",
    annotate: bool = True,
) -> None:
    """
    Heatmap en cases :
      - X = p_wp (colonne x_col, ex: 'pv_wp')
      - Y = p0  (colonne y_col, ex: 'p0')
      - Valeur = moyenne de value_col sur les scénarios (scenario_col) pour chaque (p0, p_wp)

    Couleurs : vert = moins cher, rouge = plus cher.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    required = {x_col, y_col, value_col, scenario_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans df: {missing}")

    # 1) moyenne sur scénarios pour chaque couple (p0, p_wp)
    df_mean = (
        df.groupby([y_col, x_col], as_index=False)[value_col]
          .mean()
          .rename(columns={value_col: f"{value_col}_mean"})
    )

    # 2) pivot -> matrice (index=p0, columns=p_wp)
    y_list = sorted(df_mean[y_col].unique().tolist())
    x_list = sorted(df_mean[x_col].unique().tolist())

    pivot = df_mean.pivot(index=y_col, columns=x_col, values=f"{value_col}_mean")
    pivot = pivot.reindex(index=y_list, columns=x_list)

    Z = pivot.to_numpy(dtype=float)

    # 3) plot heatmap
    plt.figure(figsize=(12, 7))

    cmap = plt.cm.get_cmap("RdYlGn_r").copy()  # vert pour faible, rouge pour fort
    cmap.set_bad(color="lightgray")  # cases manquantes

    Z_masked = np.ma.masked_invalid(Z)

    im = plt.imshow(
        Z_masked,
        aspect="auto",
        origin="lower",     # p0 croissant vers le haut
        cmap=cmap,
        interpolation="nearest",
    )

    plt.title(title)
    plt.xlabel("p_wp (W)")
    plt.ylabel("p0 (W)")

    # ticks
    plt.xticks(range(len(x_list)), [str(int(x)) for x in x_list], rotation=45, ha="right")
    plt.yticks(range(len(y_list)), [str(int(y)) for y in y_list])

    cbar = plt.colorbar(im)
    cbar.set_label("Coût total moyen (€)")

    # 4) annotation valeur dans chaque case
    if annotate:
        # format en € sans décimales (tu peux changer)
        for i in range(len(y_list)):
            for j in range(len(x_list)):
                val = Z[i, j]
                if np.isfinite(val):
                    plt.text(j, i, f"{val:,.2f}".replace(",", " "),
                             ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200)
    plt.show()




if __name__ == "__main__":
    """Pour plot juste un seul fichier"""
    # plot_from_csv(
    #     file_path="results/without_diesel/results_pv_100000_150000_s85_deterministic.csv",
    #     x_col="pv_fixed",
    #     y_col="total_cost"
    # )

    """Pour plot avec plusieurs fichiers et voir l'influence du taux de satisfaction de la charge"""
    # file_paths = [
    #     "results/without_diesel/results_pv_100000_150000_s85.csv",
    #     "results/without_diesel/results_pv_100000_150000_s90.csv",
    #     "results/without_diesel/results_pv_100000_150000_s95.csv",
    #     "results/without_diesel/results_pv_100000_200000_s100.csv",
    #     "results/without_diesel/results_pv_100000_150000_s85_deterministic_s5.csv",
    #     "results/without_diesel/results_pv_100000_150000_s90_deterministic_s5.csv",
    #     "results/without_diesel/results_pv_100000_150000_s95_deterministic_s5.csv",
    #     "results/without_diesel/results_pv_100000_200000_s100_deterministic_s5.csv"
    # ]
    #
    # plot_multiple_csv_same_graph(
    #     file_paths=file_paths,
    #     x_col="pv_fixed",
    #     y_col="total_cost"
    # )

    """Pour plot avec diesel heatmap colorée"""
    df = pd.read_csv("results/with_diesel/results_pv_1000_29000_s85.csv")
    out_dir = Path("results/with_diesel/plots_heatmap")

    plot_cost_heatmap_mean_by_scenarios(
        df=df,
        out_dir=out_dir,
        x_col="pv_wp",  # abscisse
        y_col="p0",  # ordonnée
        value_col="total_cost",
        scenario_col="scenario",
        filename="HEATMAP_mean_total_cost_p0_vs_pwp_s85.png",
    )

    """Pour connaitre le pourcentage d'énergie consommée qui provient du générateur diesel et du PV"""
    # df = compute_diesel_pv_shares("results/with_diesel/results_pv_1000_19000_s85.csv")
    #
    # cols_to_show = [
    #     "scenario",
    #     "p0",
    #     "pv_fixed",
    #     "total_cost",
    #     "load_total_Wh",
    #     "fuel_cost_EUR",
    #     "diesel_thermal_energy_Wh",
    #     "diesel_electrical_energy_Wh",
    #     "diesel_share_of_load_pct",
    #     "pv_share_of_load_pct"
    # ]
    #
    # print(df[cols_to_show])
