"""
Plots the commits per week of several software.
This is based on Github history:
https://github.com/SGpp/SGpp/graphs/contributors?all=1
https://github.com/cossan-working-group/OpenCossan/graphs/contributors?all=1
https://github.com/UCL-CCS/EasyVVUQ/graphs/contributors?all=1
https://github.com/snl-dakota/dakota/graphs/contributors?all=1
https://github.com/lanl/GPMSA/graphs/contributors?all=1
https://github.com/openturns/openturns/graphs/contributors?all=1
https://github.com/llnl/psuade/graphs/contributors?all=1
https://github.com/libqueso/queso/graphs/contributors?all=1
https://github.com/idaholab/raven/graphs/contributors?all=1
https://github.com/SURGroup/UQpy/graphs/contributors?all=1
https://github.com/jonathf/chaospy/graphs/contributors?all=1
https://github.com/sandialabs/UQTk/graphs/contributors?all=1

"""

# %%
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import itertools

# %%
def tracer_evolution_commits(
    repertoire_data,
    frequence="W",
    fenetre_lissage=None,
    figsize=(8, 4),
    glob_pattern="*Commits_over_time.csv",
    cleanup_pattern="_Commits_over_time.csv",
):
    """
    Lit les fichiers de commits dans le répertoire spécifié, trie les logiciels
    par nombre total de commits décroissant, et trace leur évolution temporelle.

    Parameters:
    -----------
    repertoire_data : str ou Path
        Chemin vers le répertoire contenant les fichiers *_Commits_over_time.csv
    frequence : str, default 'W'
        Fréquence de regroupement des données ('W', 'M', 'MS', 'Y', 'YS', etc.).
    fenetre_lissage : int, optional
        Taille de la fenêtre pour la moyenne mobile. Si None, aucun lissage n'est appliqué.
    figsize : tuple, default (8, 4)
        Dimensions de la figure Matplotlib.
    """
    # 1. Recherche des fichiers
    pattern = os.path.join(repertoire_data, glob_pattern)
    fichiers = glob.glob(pattern)

    print(f"Found {len(fichiers)} files:")
    print(fichiers)

    df_liste = []

    # 2. Lecture et extraction des données
    for fichier in fichiers:
        if os.path.exists(fichier):
            nom_fichier = Path(fichier).name
            nom_logiciel = nom_fichier.replace(cleanup_pattern, "")

            # Lecture du fichier CSV
            df = pd.read_csv(fichier, sep=";")

            # Conversion de la colonne des dates en objets datetime
            df["Week of"] = pd.to_datetime(df["Week of"])

            # Ajout du nom du logiciel pour identifier les données
            df["Logiciel"] = nom_logiciel

            df_liste.append(df)

    # 3. Consolidation, rééchantillonnage et tri des données
    if df_liste:
        df_global = pd.concat(df_liste, ignore_index=True)

        # Pivot des données
        df_pivot = df_global.pivot(
            index="Week of", columns="Logiciel", values="Commits"
        )

        # Remplacement des valeurs manquantes avant regroupement
        df_pivot = df_pivot.fillna(0)

        # Rééchantillonnage (resampling) selon la fréquence et somme des commits
        df_resampled = df_pivot.resample(frequence).sum()

        # Calcul du nombre total de commits par logiciel pour établir le classement
        totaux_commits = df_resampled.sum().sort_values(ascending=False)
        
        # Réorganisation des colonnes du DataFrame selon le classement décroissant
        df_resampled = df_resampled[totaux_commits.index]

        # Application du lissage si spécifié (après le tri pour conserver l'ordre)
        titre_lissage = ""
        if fenetre_lissage is not None and fenetre_lissage > 1:
            df_resampled = df_resampled.rolling(
                window=fenetre_lissage, min_periods=1
            ).mean()
            titre_lissage = f" (Moyenne mobile sur {fenetre_lissage} périodes)"

        # Détermination du label de l'axe des ordonnées et du titre
        dict_freq = {
            "W": "par semaine",
            "M": "par mois",
            "MS": "par mois",
            "Y": "par an",
            "YS": "par an",
        }
        label_temps = dict_freq.get(frequence, "")

        # 4. Représentation graphique avec Matplotlib
        plt.figure(figsize=figsize)

        # Définition des styles de lignes disponibles et création d'un cycleur
        styles_lignes = ["-", "--", "-.", ":"]
        cycle_styles = itertools.cycle(styles_lignes)

        # Tracé des courbes dans l'ordre du classement
        for logiciel in df_resampled.columns:
            style_courant = next(cycle_styles)
            total_logiciel = int(totaux_commits[logiciel])
            
            plt.plot(
                df_resampled.index,
                df_resampled[logiciel],
                label=f"{logiciel} ({total_logiciel} commits)",
                linestyle=style_courant,
                linewidth=1.7,
            )

        plt.title(f"Évolution du nombre de commits {label_temps}{titre_lissage}")
        plt.xlabel("Date")
        plt.ylabel(f"Nombre de commits {label_temps}")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(title="Logiciels (Total)", loc="upper left", bbox_to_anchor=(1.0, 1.0))
        plt.tight_layout()

        # Affichage du graphique
        plt.show()
    else:
        print("Aucun fichier correspondant n'a été trouvé.")

# %%
tracer_evolution_commits("../data")

# %%
tracer_evolution_commits("../data", frequence="M")

# %%
tracer_evolution_commits("../data", frequence="Y")

# %%
tracer_evolution_commits("../data", frequence="M", fenetre_lissage=8)

# %%
tracer_evolution_commits("../data", frequence="Y", fenetre_lissage=2)
plt.savefig("../figures/Github_commits.png", bbox_inches="tight")
plt.savefig("../figures/Github_commits.pdf", bbox_inches="tight")

# %%
