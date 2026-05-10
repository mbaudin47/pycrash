# %%
import openturns as ot
import openturns.viewer as otv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# %%
def draw_stratas_custom(enumeration_function, maximum_strata_index):
    """
    Représente les multi-indices colorés par strates.
    """
    cmap = plt.colormaps["viridis"]
    colors = [mcolors.to_hex(cmap(i / (maximum_strata_index - 1))) for i in range(maximum_strata_index)]
    
    graph = ot.Graph("Stratification (q={})".format(enumeration_function.getQ()), "alpha_1", "alpha_2", True)
    
    layers_levels = []
    offset = 0
    for strata_index in range(maximum_strata_index):
        strata_cardinal = enumeration_function.getStrataCardinal(strata_index)
        # Récupération des indices de la strate actuelle
        indices_list = [enumeration_function(idx + offset) for idx in range(strata_cardinal)]
        offset += strata_cardinal

        # Print the q-Norm of points in this layer
        q = enumeration_function.getQ()
        weights = enumeration_function.getWeight()
        q_norm_function = build_q_norm_function(weights, q)
        for j in range(len(indices_list)):
            multiindex = indices_list[j]
            q_norm = q_norm_function(multiindex)[0]
            print(f"Layer #{strata_index}, j={j}, multiindex={multiindex}, q-Norm={q_norm:.2f}")
            layers_levels.append(q_norm)

        # Draw the cloud
        cloud = ot.Cloud(ot.Sample(indices_list))
        cloud.setPointStyle("circle")
        cloud.setColor(colors[strata_index])
        cloud.setLegend("Strate {}".format(strata_index))
        graph.add(cloud)
        
    graph.setIntegerXTick(True)
    graph.setIntegerYTick(True)
    levels = list(set(layers_levels))
    levels.sort()
    return graph, levels

# %%
def build_q_norm_function(weights, q):
    q_norm_function = ot.SymbolicFunction(["x1", "x2", "w1", "w2", "q"], ["((w1 * x1)^(q) + (w2 * x2)^(q))^(1/q)"])
    q_norm_parametric = ot.ParametricFunction(q_norm_function, [2, 3, 4], [weights[0], weights[1], q])
    return q_norm_parametric

# %%
def draw_qnorm_contour(weights, q, levels):
    """
    Trace les lignes de niveau de la norme q pondérée.
    """
    f = build_q_norm_function(weights, q)    
    contour = f.draw([0.0, 0.0], [5.0, 5.0]).getDrawable(0).getImplementation()
    contour.setLevels(levels)
    return contour

# %%
# Paramètres de l'analyse
max_strata = 8

# %%
# Création de la planche graphique (Grid)
q= 0.7
weights = [1.0, 1.0] # Cas isotrope pour la clarté visuelle
dim = len(weights)
graph = ot.Graph("Isolignes", "x1", "x2", True)
graph.setTitle("Isolignes (q={}, w={})".format(q, weights))

# Répartition des indices
enumeration_function = ot.HyperbolicAnisotropicEnumerateFunction(weights, q)
curve, levels = draw_stratas_custom(enumeration_function, max_strata)
graph.add(curve)

# Lignes de niveau
graph.add(draw_qnorm_contour(weights, q, levels))


view = otv.View(graph, axes_kw={"aspect": "equal"}, figure_kw={"figsize": (5, 4)})

# %%
