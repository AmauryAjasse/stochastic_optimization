from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import pickle


def load_without_diesel_results(project_root: Path,
                                filename: str = "results_pv_5000_19000.csv") -> pd.DataFrame:
    """
    Charge le CSV 'without_diesel' et agrège pour obtenir 1 ligne par pv_fixed
    (car le CSV contient 1 ligne par scénario).
    """
    csv_path = project_root / "results" / "without_diesel" / filename
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"p0", "pv_fixed", "scenario", "total_cost", "bat_emax_t0", "pv_wp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV: {missing}")

    # 1 point par couple (p0, pv_fixed) -> on prend la 1ère ligne (les valeurs sont identiques entre scénarios
    # pour total_cost / pv_wp / bat_emax_t0 dans ton modèle)
    df_agg = (
        df.sort_values(["p0", "pv_fixed", "scenario"])
          .drop_duplicates(subset=["p0", "pv_fixed"], keep="first")
          .sort_values(["p0", "pv_fixed"])
          .reset_index(drop=True)
    )

    return df_agg


def plot_lcc_vs_pv(df_agg: pd.DataFrame, out_dir: Path,
                   x_col: str = "pv_wp", y_col: str = "total_cost",
                   filename: str = "LCC_vs_PV.png") -> None:
    """
    Plot LCC (total_cost) en fonction de la puissance PV installée (pv_wp).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    x = df_agg[x_col].astype(float).to_list()
    y = df_agg[y_col].astype(float).to_list()

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel("PV installé (W)")
    plt.ylabel("LCC / Coût total attendu (€)")
    plt.title("LCC en fonction de la puissance PV fixée")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200)



def plot_battery_capacity_vs_pv(df_agg: pd.DataFrame, out_dir: Path,
                                x_col: str = "pv_wp", y_col: str = "bat_emax_t0",
                                filename: str = "BAT_vs_PV.png") -> None:
    """
    Plot capacité batterie (bat_emax_t0) en fonction de la puissance PV installée (pv_wp).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    x = df_agg[x_col].astype(float).to_list()
    y = df_agg[y_col].astype(float).to_list()

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel("PV installé (W)")
    plt.ylabel("Capacité batterie (Wh) (emax[t0])")
    plt.title("Capacité batterie optimale en fonction de la puissance PV fixée")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200)

def plot_3d_surface_from_grid(
    x_list,
    y_list,
    z_grid,
    x_label="p_diesel (W)",
    y_label="PV fixé (W)",
    z_label="Value",
    title="3D surface",
    out_dir="results_image",
    filename_prefix="surface_3d"
):
    """
    Trace une surface 3D Z = f(X,Y) où :
      - x_list : liste des p_diesel (taille Nx)
      - y_list : liste des pv_fixed (taille Ny)
      - z_grid : array (Nx, Ny) avec Z[i,j] correspondant à (x_list[i], y_list[j])
                (utiliser np.nan pour les points infeasible)
    Sauvegarde une figure pickle + png.
    """
    os.makedirs(out_dir, exist_ok=True)

    X, Y = np.meshgrid(y_list, x_list)   # attention: X->PV, Y->diesel pour cohérence visuelle
    Z = np.array(z_grid, dtype=float)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")

    # masque des NaN pour éviter erreurs sur surface
    Z_masked = np.ma.masked_invalid(Z)

    surf = ax.plot_surface(X, Y, Z_masked)  # pas de couleur imposée
    ax.set_xlabel(y_label)
    ax.set_ylabel(x_label)
    ax.set_zlabel(z_label)
    ax.set_title(title)

    fig.tight_layout()

    # Sauvegardes
    pickle_path = os.path.join(out_dir, f"{filename_prefix}.pickle")
    with open(pickle_path, "wb") as f:
        pickle.dump(fig, f)

    png_path = os.path.join(out_dir, f"{filename_prefix}.png")
    fig.savefig(png_path, dpi=200)

    return fig


def plot_3d_lcc_and_battery(
    p_diesel_list,
    pv_fixed_list,
    lcc_grid,
    bat_grid,
    out_dir="results_image",
    prefix="LCC_BAT_3D"
):
    """
    Produit 2 figures 3D :
      - LCC(p_diesel, pv_fixed)
      - Capacité batterie(p_diesel, pv_fixed)
    """
    plot_3d_surface_from_grid(
        x_list=p_diesel_list,
        y_list=pv_fixed_list,
        z_grid=lcc_grid,
        x_label="p_diesel max (W)",
        y_label="PV fixé (W)",
        z_label="LCC / coût total attendu (€)",
        title="LCC en fonction de (p_diesel, PV fixé)",
        out_dir=out_dir,
        filename_prefix=f"{prefix}_LCC"
    )

    plot_3d_surface_from_grid(
        x_list=p_diesel_list,
        y_list=pv_fixed_list,
        z_grid=bat_grid,
        x_label="p_diesel max (W)",
        y_label="PV fixé (W)",
        z_label="Capacité batterie (Wh)",
        title="Capacité batterie optimale en fonction de (p_diesel, PV fixé)",
        out_dir=out_dir,
        filename_prefix=f"{prefix}_BAT"
    )

def load_with_diesel_results_agg(project_root: Path,
                                filename: str = "results_pv_5000_29000.csv") -> pd.DataFrame:
    """
    Charge le CSV results/with_diesel/<filename> et agrège à 1 ligne par (p0, pv_fixed)
    (car le CSV contient une ligne par scénario).
    """
    csv_path = project_root / "results" / "with_diesel" / filename
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"p0", "pv_fixed", "scenario", "total_cost", "bat_emax_t0"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV: {missing}")

    # 1 point par (p0, pv_fixed) : total_cost / bat_emax_t0 sont identiques entre scénarios
    df_agg = (
        df.sort_values(["p0", "pv_fixed", "scenario"])
          .drop_duplicates(subset=["p0", "pv_fixed"], keep="first")
          .sort_values(["p0", "pv_fixed"])
          .reset_index(drop=True)
    )

    return df_agg


def build_grids_for_3d(df_agg: pd.DataFrame):
    """
    Construit p_diesel_list, pv_fixed_list, lcc_grid, bat_grid
    (avec np.nan si un point (p0,pv) n'existe pas).
    """
    p_diesel_list = sorted(df_agg["p0"].unique().tolist())
    pv_fixed_list = sorted(df_agg["pv_fixed"].unique().tolist())

    # pivot -> matrices (index=p0, columns=pv_fixed)
    lcc_pivot = df_agg.pivot(index="p0", columns="pv_fixed", values="total_cost")
    bat_pivot = df_agg.pivot(index="p0", columns="pv_fixed", values="bat_emax_t0")

    # Réindex pour garantir l'ordre + remplir manquants par NaN
    lcc_pivot = lcc_pivot.reindex(index=p_diesel_list, columns=pv_fixed_list)
    bat_pivot = bat_pivot.reindex(index=p_diesel_list, columns=pv_fixed_list)

    lcc_grid = lcc_pivot.to_numpy(dtype=float)
    bat_grid = bat_pivot.to_numpy(dtype=float)

    return p_diesel_list, pv_fixed_list, lcc_grid, bat_grid


def plot_3d_from_with_diesel_csv(project_root: Path,
                                csv_name: str = "results_pv_5000_29000.csv",
                                out_subdir: str = "plots_3d",
                                prefix: str = "study") -> None:
    df_agg = load_with_diesel_results_agg(project_root, filename=csv_name)
    p_diesel_list, pv_fixed_list, lcc_grid, bat_grid = build_grids_for_3d(df_agg)

    out_dir = project_root / "results" / "with_diesel" / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_3d_lcc_and_battery(
        p_diesel_list=p_diesel_list,
        pv_fixed_list=pv_fixed_list,
        lcc_grid=lcc_grid,
        bat_grid=bat_grid,
        out_dir=str(out_dir),
        prefix=prefix
    )

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
                    plt.text(j, i, f"{val:,.0f}".replace(",", " "),
                             ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200)

def add_diesel_efficiency_and_final_battery_capacity(
    project_root: Path,
    csv_name: str = "results_pv_1000_19000.csv",
    timeseries_filename: str = "timeseries.csv",
    save: bool = True,
) -> pd.DataFrame:
    """
    Ajoute 2 colonnes au CSV with_diesel :
        - diesel_efficiency_mean  : moyenne temporelle de (gen_W / p0)
        - battery_capacity_final_Wh : dernière valeur de emax_Wh

    Le timeseries est lu dans :
        results/with_diesel/p0_xxxxW/pv_yyyyW/timeseries.csv
    """

    # -----------------------------
    # 1) Chargement du CSV principal
    # -----------------------------
    csv_path = project_root / "results" / "with_diesel" / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable : {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"p0", "pv_fixed", "scenario"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")

    diesel_eff_list = []
    bat_final_list = []
    diesel_usage_rate_list = []

    # -----------------------------
    # 2) Boucle sur chaque ligne
    # -----------------------------
    for _, row in df.iterrows():

        p0 = int(row["p0"])
        pv = int(row["pv_fixed"])
        scen = row["scenario"]

        # robuste si scenario est lu comme float (ex: 1.0)
        try:
            scen_int = int(float(scen))
        except Exception:
            scen_int = int(scen)

        ts_path = (
            project_root
            / "results"
            / "with_diesel"
            / f"p0_{p0}W"
            / f"pv_{pv}W"
            / f"scenario_{scen_int}"
            / timeseries_filename
        )

        if not ts_path.exists():
            diesel_eff_list.append(np.nan)
            bat_final_list.append(np.nan)
            continue

        ts = pd.read_csv(ts_path)

        # -----------------------------
        # 3) Efficacité moyenne diesel
        # -----------------------------
        if "gen_W" not in ts.columns:
            raise ValueError(f"'gen_W' absent dans {ts_path}")

        gen = ts["gen_W"].astype(float)

        if p0 > 0:
            gen_active = gen[gen > 0]

            if len(gen_active) > 0:
                diesel_eff_mean = (gen_active / p0).mean()
            else:
                diesel_eff_mean = np.nan  # jamais utilisé
            diesel_usage_rate = len(gen_active) / len(gen)
        else:
            diesel_eff_mean = np.nan
            diesel_usage_rate = np.nan



        # -----------------------------
        # 4) Capacité batterie finale
        # -----------------------------
        if "emax_Wh" not in ts.columns:
            raise ValueError(f"'emax_Wh' absent dans {ts_path}")

        battery_initial = ts["emax_Wh"].iloc[0]
        battery_final = ts["emax_Wh"].iloc[-1]

        diesel_eff_list.append(diesel_eff_mean)
        bat_final_list.append(battery_final/battery_initial)
        diesel_usage_rate_list.append(diesel_usage_rate)

    # -----------------------------
    # 5) Ajout des colonnes
    # -----------------------------
    df["diesel_efficiency_mean"] = diesel_eff_list
    df["battery_capacity_final"] = bat_final_list
    df["diesel_usage_rate"] = diesel_usage_rate_list

    # -----------------------------
    # 6) Sauvegarde
    # -----------------------------
    if save:
        df.to_csv(csv_path, index=False)

    return df

import pandas as pd
import os

def extract_min_total_cost_per_scenario(input_csv_path, output_csv_path=None):
    """
    Extracts the row with the minimum total cost for each scenario
    from a results CSV file.

    Parameters
    ----------
    input_csv_path : str
        Path to the input CSV file (e.g., results_pv_1000_22000.csv)

    output_csv_path : str or None
        Path to save the output CSV file.
        If None, the file will be saved next to the input file
        with suffix '_min_per_scenario.csv'

    Returns
    -------
    pd.DataFrame
        DataFrame containing one row per scenario (minimum total cost)
    """

    # Load CSV (adapt separator if needed)
    df = pd.read_csv(input_csv_path)

    # ----------- Identify key columns automatically -----------

    # Try to detect scenario column
    scenario_col = None
    for col in df.columns:
        if "scenario" in col.lower():
            scenario_col = col
            break

    if scenario_col is None:
        raise ValueError("No scenario column found in CSV.")

    # Try to detect total cost column
    cost_col = None
    for col in df.columns:
        if "total" in col.lower() and "cost" in col.lower():
            cost_col = col
            break

    if cost_col is None:
        raise ValueError("No total cost column found in CSV.")

    # ----------- Select minimum cost per scenario -----------

    idx = df.groupby(scenario_col)[cost_col].idxmin()
    df_min = df.loc[idx].sort_values(by=scenario_col)

    # ----------- Save output file -----------

    if output_csv_path is None:
        base, ext = os.path.splitext(input_csv_path)
        output_csv_path = base + "_min_per_scenario.csv"

    df_min.to_csv(output_csv_path, index=False)

    print(f"Saved file with {len(df_min)} rows to:")
    print(output_csv_path)

    return df_min


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]

    df_agg = load_without_diesel_results(project_root, filename="results_pv_20000_39000.csv")

    out_dir = project_root / "results" / "without_diesel" / "plots"

    # df_updated = add_diesel_efficiency_and_final_battery_capacity(
    #     project_root=project_root,
    #     csv_name="results_pv_1000_19000.csv",
    # )


    # plot_lcc_vs_pv(df_agg, out_dir=out_dir)
    # plot_battery_capacity_vs_pv(df_agg, out_dir=out_dir)

    # plot_3d_from_with_diesel_csv(
    #     project_root=project_root,
    #     csv_name="results_pv_1000_22000.csv",
    #     out_subdir="plots_3d",
    #     prefix="study"
    # )

    df = pd.read_csv(project_root / "results" / "with_diesel" / "results_pv_1000_19000_s100.csv")
    out_dir = project_root / "results" / "with_diesel" / "plots_heatmap"


    plot_cost_heatmap_mean_by_scenarios(
        df=df,
        out_dir=out_dir,
        x_col="pv_wp",  # abscisse
        y_col="p0",  # ordonnée
        value_col="total_cost",
        scenario_col="scenario",
        filename="HEATMAP_mean_total_cost_p0_vs_pwp.png",
    )

    # extract_min_total_cost_per_scenario(input_csv_path="with_diesel/results_pv_1000_22000_182timebatt.csv")

    plt.show()
