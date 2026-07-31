"""
Banc d'essai comparatif de la vitesse des fonctions d'énumération.

Ce script a pour objectif de quantifier et de comparer la puissance de calcul
des différentes stratégies d'énumération d'OpenTURNS en fonction de la taille
de la base et de la dimension du problème. Il permet de mesurer précisément
la dégradation des performances lors du passage d'une règle linéaire à une
règle hyperbolique, tout en analysant la sensibilité du temps de génération
à la valeur de la quasi-norme $q$.

La mise en œuvre repose sur une procédure de montée en charge itérative qui
augmente la taille de la base jusqu'à atteindre un seuil temporel critique.
Pour chaque configuration, le script calcule le débit de production de
multi-indices (exprimé en milliers d'indices par seconde) et génère des
graphiques de synthèse en échelle logarithmique. Ces résultats sont exportés
sous forme de fichiers PDF pour différentes dimensions, permettant une analyse
visuelle de la complexité algorithmique des fonctions testées.

References
----------

https://github.com/openturns/openturns/issues/2971

Output
------

+ 1. With linear enumeration function
basis size =  1000
elapsed = 0.0 (s)

+ 2. With hyperbolic enumeration function
basis size =  1000
elapsed = 0.3 (s)

+ dimension =  2
N=2000, Elapsed time: 0.0 (s), performance: 763.086 (indices/s)
N=4000, Elapsed time: 0.0 (s), performance: 664.181 (indices/s)
N=8000, Elapsed time: 0.0 (s), performance: 629.291 (indices/s)
N=16000, Elapsed time: 0.0 (s), performance: 574.400 (indices/s)
N=32000, Elapsed time: 0.1 (s), performance: 505.041 (indices/s)
N=64000, Elapsed time: 0.1 (s), performance: 443.127 (indices/s)
N=128000, Elapsed time: 0.3 (s), performance: 370.329 (indices/s)
N=256000, Elapsed time: 0.9 (s), performance: 299.762 (indices/s)
N=512000, Elapsed time: 2.2 (s), performance: 235.814 (indices/s)
N=1024000, Elapsed time: 5.7 (s), performance: 179.844 (indices/s)
Quasi-norm parameter =  0.2
N=2000, Elapsed time: 5.6 (s), performance: 0.357 (indices/s)
Quasi-norm parameter =  0.5
N=2000, Elapsed time: 2.6 (s), performance: 0.760 (indices/s)
N=4000, Elapsed time: 9.8 (s), performance: 0.410 (indices/s)
Quasi-norm parameter =  0.9
N=2000, Elapsed time: 1.3 (s), performance: 1.525 (indices/s)
N=4000, Elapsed time: 6.6 (s), performance: 0.604 (indices/s)
Quasi-norm parameter =  1.0
N=2000, Elapsed time: 1.3 (s), performance: 1.593 (indices/s)
N=4000, Elapsed time: 5.4 (s), performance: 0.744 (indices/s)

+ dimension =  3
N=2000, Elapsed time: 0.0 (s), performance: 824.676
N=4000, Elapsed time: 0.0 (s), performance: 761.701
N=8000, Elapsed time: 0.0 (s), performance: 762.237
N=16000, Elapsed time: 0.0 (s), performance: 707.019
N=32000, Elapsed time: 0.0 (s), performance: 699.772
N=64000, Elapsed time: 0.1 (s), performance: 648.209
N=128000, Elapsed time: 0.2 (s), performance: 636.261
N=256000, Elapsed time: 0.4 (s), performance: 593.127
N=512000, Elapsed time: 0.9 (s), performance: 560.030
N=1024000, Elapsed time: 2.0 (s), performance: 508.199
N=2048000, Elapsed time: 4.5 (s), performance: 453.276
Quasi-norm parameter =  0.2
N=2000, Elapsed time: 5.7 (s), performance: 0.348
Quasi-norm parameter =  0.5
N=2000, Elapsed time: 2.8 (s), performance: 0.724
N=4000, Elapsed time: 9.7 (s), performance: 0.412
Quasi-norm parameter =  0.9
N=2000, Elapsed time: 1.2 (s), performance: 1.634
N=4000, Elapsed time: 6.9 (s), performance: 0.576
Quasi-norm parameter =  1.0
N=2000, Elapsed time: 1.2 (s), performance: 1.719
N=4000, Elapsed time: 5.6 (s), performance: 0.719

+ dimension =  5
N=2000, Elapsed time: 0.0 (s), performance: 805.822
N=4000, Elapsed time: 0.0 (s), performance: 783.250
N=8000, Elapsed time: 0.0 (s), performance: 779.574
N=16000, Elapsed time: 0.0 (s), performance: 791.808
N=32000, Elapsed time: 0.0 (s), performance: 776.058
N=64000, Elapsed time: 0.1 (s), performance: 777.864
N=128000, Elapsed time: 0.2 (s), performance: 756.672
N=256000, Elapsed time: 0.3 (s), performance: 739.163
N=512000, Elapsed time: 0.7 (s), performance: 719.500
N=1024000, Elapsed time: 1.5 (s), performance: 704.644
N=2048000, Elapsed time: 3.0 (s), performance: 673.271
N=4096000, Elapsed time: 6.2 (s), performance: 658.906
Quasi-norm parameter =  0.2
N=2000, Elapsed time: 5.5 (s), performance: 0.365
Quasi-norm parameter =  0.5
N=2000, Elapsed time: 2.7 (s), performance: 0.745
N=4000, Elapsed time: 9.6 (s), performance: 0.415
Quasi-norm parameter =  0.9
N=2000, Elapsed time: 1.1 (s), performance: 1.761
N=4000, Elapsed time: 7.1 (s), performance: 0.563
Quasi-norm parameter =  1.0
N=2000, Elapsed time: 1.1 (s), performance: 1.810
N=4000, Elapsed time: 5.7 (s), performance: 0.706

+ dimension =  10
N=2000, Elapsed time: 0.0 (s), performance: 806.054 (indices/s)
N=4000, Elapsed time: 0.0 (s), performance: 790.856 (indices/s)
N=8000, Elapsed time: 0.0 (s), performance: 790.725 (indices/s)
N=16000, Elapsed time: 0.0 (s), performance: 793.211 (indices/s)
N=32000, Elapsed time: 0.0 (s), performance: 799.524 (indices/s)
N=64000, Elapsed time: 0.1 (s), performance: 800.213 (indices/s)
N=128000, Elapsed time: 0.2 (s), performance: 788.159 (indices/s)
N=256000, Elapsed time: 0.3 (s), performance: 773.868 (indices/s)
N=512000, Elapsed time: 0.7 (s), performance: 760.571 (indices/s)
N=1024000, Elapsed time: 1.4 (s), performance: 742.372 (indices/s)
N=2048000, Elapsed time: 2.8 (s), performance: 732.845 (indices/s)
N=4096000, Elapsed time: 5.6 (s), performance: 735.505 (indices/s)
Quasi-norm parameter =  0.2
N=2000, Elapsed time: 5.1 (s), performance: 0.389 (indices/s)
Quasi-norm parameter =  0.5
N=2000, Elapsed time: 2.7 (s), performance: 0.739 (indices/s)
N=4000, Elapsed time: 9.9 (s), performance: 0.404 (indices/s)
Quasi-norm parameter =  0.9
N=2000, Elapsed time: 1.1 (s), performance: 1.760 (indices/s)
N=4000, Elapsed time: 7.1 (s), performance: 0.565 (indices/s)
Quasi-norm parameter =  1.0
N=2000, Elapsed time: 1.1 (s), performance: 1.788 (indices/s)
N=4000, Elapsed time: 5.5 (s), performance: 0.729 (indices/s)

"""

# %%
import openturns as ot
import time
import openturns.viewer as otv


# %%
def benchmarkEnumeration(
    enumerateFunction,
    max_time=4.0,
    basisSize_factor=2.0,
    minimum_basisSize=2,
    number_of_points_per_second_factor=1.0e6,
    maximum_number_of_iterations=30,
):
    """
    perform a benchmark of the enumerateFunction method.

    At each iteration, the sample size increases, which increases the
    elapsed time.
    The algorithm increases the sample size until the elapsed time gets greater than
    the maximum time.

    Parameters
    ----------
    max_time : float, optional
        The maximum number of seconds to wait before stopping the
        algorithm. The default is 4.0.
    basisSize_factor : float, > 1.0
        The factor which multiplies the sample size at each iteration.
    minimum_basisSize : int, default = 2
        The minimum sample size.
    number_of_points_per_second_factor : float, > 1.0, default = 1.e6
        The factor which is used to measure the performance.
        The default is 1.0e6, which measures number of million
        points per seconds.
    maximum_number_of_iterations : int, default = 30
        The maximum number of iterations.

    Returns
    -------
    basisSizeList : np.array(niter)
        The size of Monte Carlo samples.
    timeList : np.array(niter)
        The elapsed time (s).
    performanceList : np.array(niter)
        The number of Million Monte Carlo samples divided by the elapsed time (s).

    """
    #
    basisSizeList = list()
    timeList = list()
    performanceList = list()
    basisSize = minimum_basisSize
    for i in range(maximum_number_of_iterations):
        basisSize = int(basisSize_factor * basisSize)
        start_time = time.time()
        for i in range(basisSize):
            _ = enumerateFunction(i)
        end_time = time.time()
        time_sample = end_time - start_time
        if time_sample == 0.0:
            # Performance is too small, skip sample size - Reset
            basisSizeList = list()
            timeList = list()
            performanceList = list()
            continue
        performanceList_sample = (
            basisSize / time_sample / number_of_points_per_second_factor
        )
        print(
            f"N={basisSize}, Elapsed time: {time_sample:.1f} (s), performance: {performanceList_sample:.3f} (indices/s)"
        )
        # Store the data
        basisSizeList.append(basisSize)
        timeList.append(time_sample)
        performanceList.append(performanceList_sample)
        if time_sample > max_time:
            break
    return (basisSizeList, timeList, performanceList)


# %%
def TimeEnumerationFunction(enumerateFunction, basisSize):
    t1 = time.time()
    for i in range(basisSize):
        _ = enumerateFunction(i)
    t2 = time.time()
    elapsed = t2 - t1
    return elapsed


# %%
def makeCurveCloud(dataX, dataY, color, legend, pointStyle, lineStyle):
    """
    Create a curve-cloud graph.

    Parameters
    ----------
    dataX : ot.Sample(size, 1)
        The X data.
    dataY : ot.Sample(size, 1)
        The Y data.
    color : str
        The color.
    legend : str
        The legend.
    pointStyle : str
        The point style.
    lineStyle : str
        The line style.

    Returns
    -------
    graph : ot.Graph
        The graph.

    """
    graph = ot.Graph()
    cloud = ot.Cloud(dataX, dataY)
    cloud.setPointStyle(pointStyle)
    cloud.setLegend(legend)
    cloud.setColor(color)
    graph.add(cloud)
    curve = ot.Curve(dataX, dataY)
    curve.setLineStyle(lineStyle)
    curve.setLegend("")
    curve.setColor(color)
    graph.add(curve)
    return graph


# %%
def benchmarkHyperbolicRule(
    weight, quasiNorm, minimum_basisSize, number_of_points_per_second_factor
):
    enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(weight, quasiNorm)
    result = benchmarkEnumeration(
        enumerateFunction,
        minimum_basisSize=minimum_basisSize,
        number_of_points_per_second_factor=number_of_points_per_second_factor,
    )
    return result


# %%
def MakeBenchmarkEnumerationRule(
    dimension,
    quasiNormList=[0.2, 0.5, 0.9, 1.0],
    number_of_points_per_second_factor=1.0e3,
    minimum_basisSize=1000,
):
    # 1. Benchmark linear enumeration
    enumerateFunction = ot.LinearEnumerateFunction(dimension)
    basisSizeListLinear, timeListLinear, performanceListLinear = benchmarkEnumeration(
        enumerateFunction,
        minimum_basisSize=minimum_basisSize,
        number_of_points_per_second_factor=number_of_points_per_second_factor,
    )

    # 2. Benchmark hyperbolic enumeration
    hyperbolicResultList = []
    for quasiNorm in quasiNormList:
        print("Quasi-norm parameter = ", quasiNorm)
        result = benchmarkHyperbolicRule(
            weight,
            quasiNorm,
            minimum_basisSize=minimum_basisSize,
            number_of_points_per_second_factor=number_of_points_per_second_factor,
        )
        hyperbolicResultList.append(result)

    # 3. Make a plot
    pointStyleList = list(ot.Drawable.GetValidPointStyles())
    pointStyleList.remove("circle")
    pointStyleList.remove("diamond")
    pointStyleList.remove("bullet")
    pointStyleList.remove("dot")
    pointStyleList.remove("none")
    lineStyleList = list(ot.Drawable.GetValidLineStyles())
    lineStyleList.remove("blank")
    palette = ot.Drawable.BuildDefaultPalette(1 + len(quasiNormList))
    graph = ot.Graph(
        f"Performance of enumerate functions, dimension = {dimension}",
        "Basis size",
        "Thousand Multi-indices / s",
        True,
    )
    curveCloud = makeCurveCloud(
        basisSizeListLinear,
        performanceListLinear,
        palette[0],
        "Linear",
        pointStyleList[0],
        lineStyleList[0],
    )
    graph.add(curveCloud)
    index = 0
    for hyperbolic_result in hyperbolicResultList:
        quasiNorm = quasiNormList[index]
        (
            basisSizeListHyperbolic,
            timeListHyperbolic,
            performanceListHyperbolic,
        ) = hyperbolic_result
        curveCloud = makeCurveCloud(
            basisSizeListHyperbolic,
            performanceListHyperbolic,
            palette[1 + index],
            "Hyperbolic, q = %.2f" % (quasiNorm),
            pointStyleList[1 + index],
            lineStyleList[(1 + index) % 6],
        )
        graph.add(curveCloud)
        index += 1
    graph.setLogScale(ot.GraphImplementation.LOGXY)
    graph.setLegendPosition("topright")
    return graph


# %%
dimension = 20
basisSize = 1000

# %%
print("+ 1. With linear enumeration function")
enumerateFunction = ot.LinearEnumerateFunction(dimension)
print("basis size = ", basisSize)
elapsed = TimeEnumerationFunction(enumerateFunction, basisSize)
print("elapsed = %.1f (s)" % (elapsed))


# %%
print("+ 2. With hyperbolic enumeration function")
quasiNorm = 1.0
weight = [1.0] * dimension
enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(weight, quasiNorm)
print("basis size = ", basisSize)
elapsed = TimeEnumerationFunction(enumerateFunction, basisSize)
print("elapsed = %.1f (s)" % (elapsed))


# %%
for dimension in [2, 3, 5, 10, 20]:
    print("+ dimension = ", dimension)
    graph = MakeBenchmarkEnumerationRule(dimension)
    view = otv.View(
        graph,
        figure_kw={"figsize": (5.0, 3.0)},
        legend_kw={"bbox_to_anchor": (1.0, 1.0), "loc": "upper left"},
    )
    filename = "figures/benchmark-hyperbolic-enumeration-dimension-%d.pdf" % (dimension)
    view.getFigure().savefig(filename, bbox_inches="tight")
    otv.View.ShowAll()
