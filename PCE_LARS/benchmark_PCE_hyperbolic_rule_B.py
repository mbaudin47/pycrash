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
            "N=%d, Elapsed time: %.1f (s), performance: %.3f"
            % (basisSize, time_sample, performanceList_sample)
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
    (basisSizeListLinear, timeListLinear, performanceListLinear) = benchmarkEnumeration(
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
        "Performance of enumerate functions, dimension = %d" % (dimension),
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
    filename = "benchmark-hyperbolic-enumeration-dimension-%d.pdf" % (dimension)
    view.getFigure().savefig(filename, bbox_inches="tight")
    otv.View.ShowAll()