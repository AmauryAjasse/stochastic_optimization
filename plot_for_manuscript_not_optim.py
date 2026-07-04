import matplotlib.pyplot as plt
import pandas as pd

electricity_access_data = {"Monde" : [72.7, 73.2, 73.5, 74.7, 75.9, 76.5, 77.2, 77.8, 78.4, 79.1, 79.9, 80.8, 81.4, 82.2, 83.2, 84.5, 85.4, 86.8, 88.7, 89.7, 90.3, 90.4, 90.4, 90.7],
                           "Burundi": [4, 4.1, 4.3, 4.4, 4.5, 4.7, 4.8, 4.9, 5, 5.2, 5.3, 5.6, 6.5, 7.4, 8.3, 9.1, 10, 10.1, 10.2, 10.2, 10.2, 10.2, 10.1, 10.3],
                           "République Centrafricaine": [1, 1.1, 1.3, 1.4, 1.5, 1.8, 1.8, 1.9, 2, 2.2, 2.3, 2.4, 2.5, 2.5, 2.6, 3, 3.3, 3.6, 3.9, 4.4, 5.2, 6.1, 6.1, 6.4],
                           "R. D. Congo": [6.7, 6.7, 6.8, 6.8, 6.8, 6.9, 6.9, 6.9, 6.9, 7, 7, 7.1, 7.2, 7.3, 7.4, 7.5, 7.9, 8.3, 8.6, 8.7, 8.7, 9.3, 9.8, 10.5],
                           "Malawi": [5, 5.4, 5.8, 6.4, 7, 7.4, 7.8, 8.2, 8.6, 9, 8.7, 8.9, 9, 8.7, 11.9, 10.8, 11.7, 12.6, 11.8, 11, 11, 11.4, 12, 12.1],
                           "Madagascar": [8, 8.2, 8.3, 11.7, 15, 15, 15.5, 16, 17.4, 17.9, 18.4, 18.9, 20.2, 21.6, 22.9, 24.2, 25.5, 26.8, 28.1, 29.4, 30.7, 32, 32, 33],
                           "Algérie": [98, 98.3, 98.5, 98.3, 98.1, 98.3, 98.6, 98.8, 99.1, 99.3, 99.3, 99.4, 99, 99, 99.9, 99.1, 99.1, 99.1, 99.1, 99.1, 99.1, 99.1, 100, 100],
                           "Ethiopie": [4.7, 7.3, 9.9, 12.4, 15, 16.1, 17.3, 18.4, 19.6, 20.7, 21.9, 23, 23.2, 23.3, 28.5, 33.7, 37.4, 41.9, 44, 48.7, 51.5, 52.3, 53.1, 54.3],
                           "Tchad": [2, 2.4, 2.8, 3.1, 3.5, 3.5, 3.6, 3.6, 3.6, 3.6, 3.7, 3.7, 4.7, 5.7, 6.7, 7.7, 7.7, 7.8, 7.8, 7.8, 7.8, 7.8, 7.8, 7.8],
                           "Kenya": [11, 15, 17, 18, 19, 21, 23, 24, 25, 28, 30, 32, 35, 37, 41, 47, 52, 59, 65, 69, 71, 72, 75, 79.1]}

annees = list(range(2000, 2024))

# def plot_electricity_access(electricity_access_data, annees):
#     cm_to_inch = 1 / 2.54
#
#     plt.figure(figsize=(15 * cm_to_inch, 10 * cm_to_inch))
#
#     df = pd.DataFrame(electricity_access_data, index=annees)
#
#     for country in df.columns:
#         plt.plot(df.index, df[country], label=country)
#
#     plt.xlabel("Year", fontsize=11)
#     plt.ylabel("Access to electricity (%)", fontsize=11)
#
#     plt.xticks(fontsize=11)
#     plt.yticks(fontsize=11)
#
#     plt.legend(fontsize=11, loc='center left', bbox_to_anchor=(1.02, 0.5))
#     plt.grid(True, alpha=0.3)
#
#     plt.tight_layout()
#     plt.show()
def plot_electricity_access(electricity_access_data, annees):
    cm_to_inch = 1 / 2.54

    # 15 cm pour le graphe + 5 cm pour la légende
    fig, ax = plt.subplots(figsize=(20 * cm_to_inch, 11 * cm_to_inch))

    df = pd.DataFrame(electricity_access_data, index=annees)

    for country in df.columns:
        ax.plot(df.index, df[country], label=country)

    ax.set_xlabel("Année", fontsize=11)
    ax.set_ylabel("Taux d'accès à l'électricité (%)", fontsize=11)

    ax.tick_params(axis='both', labelsize=11)

    ax.grid(True, alpha=0.3)

    # La zone du graphe occupe seulement 75% de la largeur
    plt.subplots_adjust(right=0.65)

    # Légende à droite
    ax.legend(
        fontsize=10,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5)
    )

    plt.show()


if __name__ == "__main__":
    plot_electricity_access(electricity_access_data, annees)
