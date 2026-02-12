# results/results_processing.py

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


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




if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]

    df_agg = load_without_diesel_results(project_root, filename="results_pv_5000_19000.csv")

    out_dir = project_root / "results" / "without_diesel" / "plots"

    plot_lcc_vs_pv(df_agg, out_dir=out_dir)
    plot_battery_capacity_vs_pv(df_agg, out_dir=out_dir)
    plt.show()

    print(f"✅ Plots sauvegardés dans: {out_dir}")

