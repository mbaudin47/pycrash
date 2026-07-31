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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tabulate


# %%
def draw_stratas_custom(
    enumeration_function, maximum_strata_index, offset_text=[0.1, 0.1], font_size=1
):
    """
    Représente les multi-indices colorés par strates avec étiquettes de rang.
    """
    cmap = plt.colormaps["viridis"]
    colors = [
        mcolors.to_hex(cmap(i / (maximum_strata_index - 1)))
        for i in range(maximum_strata_index)
    ]

    graph = ot.Graph(
        "Stratification (q={})".format(enumeration_function.getQ()),
        "alpha_1",
        "alpha_2",
        True,
    )

    layers_levels = []
    global_index = 0
    offset_indices = 0
    multindex_table = []
    maximum_marginal_index = [0, 0]
    maximum_q_norm = 0.0

    for strata_index in range(maximum_strata_index):
        strata_cardinal = enumeration_function.getStrataCardinal(strata_index)
        indices_list = [
            enumeration_function(idx + offset_indices) for idx in range(strata_cardinal)
        ]

        q = enumeration_function.getQ()
        weights = enumeration_function.getWeight()
        q_norm_function = build_q_norm_function(weights, q)

        for j in range(len(indices_list)):
            multiindex = indices_list[j]
            q_norm = q_norm_function(multiindex)[0]
            layers_levels.append(q_norm)

            # Ajout du texte pour chaque point (rang d'énumération)
            # On place le texte avec le décalage spécifié
            text_pos = [multiindex[0] + offset_text[0], multiindex[1] + offset_text[1]]
            text_pos = ot.Sample([text_pos])
            description = ot.Description(1)
            description[0] = str(global_index)
            label = ot.Text(text_pos, description)
            label.setTextSize(font_size)
            label.setColor("black")
            graph.add(label)

            # Ajoute le multi-indice dans la table
            multindex_table.append(
                [global_index, strata_index, sum(multiindex), multiindex, q_norm]
            )

            # Met à jour les indices marginaux
            maximum_marginal_index[0] = max(maximum_marginal_index[0], multiindex[0])
            maximum_marginal_index[1] = max(maximum_marginal_index[1], multiindex[1])

            # Met à jour la q-Norme maximale
            maximum_q_norm = max(maximum_q_norm, q_norm)

            global_index += 1

        # Dessin du nuage de points de la strate
        cloud = ot.Cloud(ot.Sample(indices_list))
        cloud.setPointStyle("circle")
        cloud.setColor(colors[strata_index])
        cloud.setLegend("Strate {}".format(strata_index))
        graph.add(cloud)

        offset_indices += strata_cardinal

    graph.setIntegerXTick(True)
    graph.setIntegerYTick(True)
    levels = sorted(list(set(layers_levels)))
    return graph, levels, multindex_table, maximum_marginal_index, maximum_q_norm


# %%
def build_q_norm_function(weights, q):
    """Returns a parametric function representing the $q$-norm of a
    two-dimensional vector weighted by the provided coefficients."""
    if any(w <= 0 for w in weights):
        raise ValueError("All weights must be strictly positive.")
    if len(weights) != 2:
        raise ValueError(f"The number of weights should be 2, but is {len(weights)}")
    if q < 0.0 or q > 1.0:
        raise ValueError(f"The parameter q should be between 0 and 1, but is {q}")
    q_norm_function = ot.SymbolicFunction(
        ["x1", "x2", "w1", "w2", "q"], ["((w1 * x1)^q + (w2 * x2)^q)^(1 / q)"]
    )
    q_norm_parametric = ot.ParametricFunction(
        q_norm_function, [2, 3, 4], [weights[0], weights[1], q]
    )
    return q_norm_parametric


# %%
def draw_qnorm_contour(weights, q, levels, maximum_marginal_index):
    """
    Trace les lignes de niveau de la norme q pondérée.
    """
    f = build_q_norm_function(weights, q)
    contour = (
        f.draw([0.0, 0.0], [maximum_marginal_index, maximum_marginal_index])
        .getDrawable(0)
        .getImplementation()
    )
    contour.setLevels(levels)
    return contour


# %%
# Paramètres de l'analyse
def plot_hyperbolic_anisotropic_rule(
    weights, q, maximum_strata_index=8, maximum_marginal_index=10.0
):
    # Création de la planche graphique (Grid)
    dim = len(weights)
    graph = ot.Graph("", r"$\alpha_1$", r"$\alpha_2$", True)
    graph.setTitle("q={}, w={}".format(q, weights))

    # Répartition des indices
    enumeration_function = ot.HyperbolicAnisotropicEnumerateFunction(weights, q)
    curve, levels, multindex_table, _, maximum_q_norm = draw_stratas_custom(
        enumeration_function, maximum_strata_index
    )
    graph.add(curve)

    # Lignes de niveau
    graph.add(draw_qnorm_contour(weights, q, levels, maximum_marginal_index))
    return graph, multindex_table, maximum_q_norm
