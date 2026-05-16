# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import squarify

# %%
# 1. Chargement des données
# Remplacez 'logiciels_cloc.csv' par le chemin réel de votre fichier
df = pd.read_csv("logiciels_cloc.csv")

# %%
# Nettoyage des espaces potentiels dans les noms de colonnes ou de chaînes
df.columns = df.columns.str.strip()
df["logiciel"] = df["logiciel"].str.strip()
df["langage"] = df["langage"].str.strip()

# %%
# ==============================================================================
# GRAPHique 1 : Nombre total de lignes de code par logiciel
# ==============================================================================
totaux_code = df.groupby("logiciel")["code"].sum().sort_values(ascending=False)

plt.figure(figsize=(8, 4))
bars = plt.bar(totaux_code.index, totaux_code.values, color="skyblue", edgecolor="grey")

plt.title("Nombre total de lignes de code par logiciel", fontsize=14, fontweight="bold")
plt.xlabel("Logiciels")
plt.ylabel("Lignes de code")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Ajout des valeurs au-dessus des barres
maximum_cloc = 0
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval + (yval * 0.01),
        f"{yval:,}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    maximum_cloc = max(maximum_cloc, yval)
plt.ylim(0, maximum_cloc * 1.1)
plt.savefig("../figures/github_count_cloc_total.png")
plt.savefig("../figures/github_count_cloc_total.pdf")
plt.show()

# %%
# ==============================================================================
# GRAPHique 2 : Cartes proportionnelles (Treemaps) par projet
# ==============================================================================
logiciels_unique = df["logiciel"].unique()

for log in logiciels_unique:
    df_log = df[df["logiciel"] == log].copy()

    # Tri par volume de code décroissant
    df_log = df_log.sort_values(by="code", ascending=False)

    # Calcul du pourcentage pour filtrer ou étiqueter proprement
    total_log = df_log["code"].sum()
    df_log["pourcentage"] = (df_log["code"] / total_log) * 100

    # Pour éviter la surcharge visuelle, regrouper les langages < 1.5% sous "Autres"
    seuil = 1.5
    principaux = df_log[df_log["pourcentage"] >= seuil].copy()
    autres = df_log[df_log["pourcentage"] < seuil]

    if not autres.empty:
        nouvelle_ligne = pd.DataFrame(
            {
                "logiciel": [log],
                "langage": ["Autres"],
                "fichiers": [autres["fichiers"].sum()],
                "vides": [autres["vides"].sum()],
                "commentaires": [autres["commentaires"].sum()],
                "code": [autres["code"].sum()],
                "pourcentage": [autres["pourcentage"].sum()],
            }
        )
        df_visualisation = pd.concat([principaux, nouvelle_ligne], ignore_index=True)
    else:
        df_visualisation = principaux

    # Préparation des étiquettes (Nom + Pourcentage)
    labels = [
        f"{row['langage']}\n{row['code']:,} l.\n({row['pourcentage']:.1f}%)"
        for _, row in df_visualisation.iterrows()
    ]

    # Génération d'une palette de couleurs distinctes
    couleurs = plt.cm.tab20(np.linspace(0, 1, len(df_visualisation)))

    # Création de la figure pour le treemap du logiciel courant
    plt.figure(figsize=(5, 5))
    squarify.plot(
        sizes=df_visualisation["code"],
        label=labels,
        color=couleurs,
        alpha=0.8,
        edgecolor="white",
        linewidth=2,
        text_kwargs={"fontsize": 10, "weight": "bold"},
    )

    plt.title(
        f"Répartition des langages dans le projet : {log}\n(Total : {total_log:,} lignes de code)",
        fontsize=14,
        fontweight="bold",
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"../figures/github_count_cloc_{log}_fractions.png")
    plt.savefig(f"../figures/github_count_cloc_{log}_fractions.pdf")
    plt.show()
# %%
