"""
Visualisation de la stratification des multi-indices pour le chaos polynomial.

Ce script a pour objectif de représenter graphiquement la répartition des
multi-indices dans un plan bidimensionnel, en fonction d'une fonction
d'énumération anisotrope hyperbolique. Il permet de visualiser comment les
indices sont regroupés par strates et comment la norme q pondérée définit
les frontières de ces ensembles au sein de la base polynomiale.

La mise en œuvre s'appuie sur la bibliothèque OpenTURNS pour la gestion des
fonctions d'énumération et la création d'objets graphiques. Le code calcule
le rang de chaque multi-indice, lui attribue une couleur selon sa strate
d'appartenance, et superpose les lignes de niveau théoriques calculées via une
fonction symbolique paramétrique de la norme q.
"""

# %%
import openturns as ot
import openturns.viewer as otv
import plot_hyperbolic_anisotropic_rule_lib as pharlib
import tabulate

# %%
weights = [1.0, 0.5]  # Cas isotrope pour la clarté visuelle
q = 0.7
graph, multindex_table, maximum_q_norm = pharlib.plot_hyperbolic_anisotropic_rule(
    weights, q, maximum_strata_index=8, maximum_marginal_index=8.0
)
view = otv.View(graph, axes_kw={"aspect": "equal"}, figure_kw={"figsize": (5, 4)})

# %%
tabulate.tabulate(
    multindex_table,
    headers=["Indice", "Couche", "Degré", "Multi-indice", "q-norm"],
    tablefmt="html",
)

# %%
