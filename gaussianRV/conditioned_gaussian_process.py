# -*- coding: utf-8 -*-
"""
Experiments on conditioned gaussian processes.
Adapted from:
https://openturns.github.io/openturns/latest/auto_meta_modeling/kriging_metamodel/plot_kriging_simulate.html#sphx-glr-auto-meta-modeling-kriging-metamodel-plot-kriging-simulate-py

Kriging : generate trajectories from a metamodel
"""

# %%
import openturns as ot
import openturns.viewer as viewer
from matplotlib import pylab as plt


def plot_data_train(x_train, y_train):
    """Plot the data (x_train,y_train) as a Cloud, in red"""
    graph_train = ot.Cloud(x_train, y_train)
    graph_train.setColor("red")
    graph_train.setLegend("Data")
    return graph_train


# %%
def plot_data_test(x_test, y_test):
    """Plot the data (x_test,y_test) as a Curve, in dashed black"""
    graphF = ot.Curve(x_test, y_test)
    graphF.setLegend("Exact")
    graphF.setColor("black")
    graphF.setLineStyle("dashed")
    return graphF


ot.Log.Show(ot.Log.NONE)

# %%
g = ot.SymbolicFunction(["x"], ["sin(x)"])

# %%
grid = [2.0, 4.0, 6.0, 8.0]
x_train = [[x] for x in [2.0, 4.0, 6.0, 8.0, 11.0, 11.5]]
y_train = g(x_train)
n_train = len(x_train)
n_train

# %%
# In order to compare the function and its metamodel, we use a test (i.e. validation) design of experiments made of a regular grid of 100 points from 0 to 12. Then we convert this grid into a `Sample` and we compute the outputs of the function on this sample.

# %%
xmin = 0.0
xmax = 12.0
n_test = 13
step = (xmax - xmin) / (n_test - 1)
myRegularGrid = ot.RegularGrid(xmin, step, n_test)
x_test = myRegularGrid.getVertices()
print(x_test)
y_test = g(x_test)


dimension = 1
basis = ot.ConstantBasisFactory(dimension).build()
covarianceModel = ot.MaternModel([1.0] * dimension, 1.5)
algo = ot.KrigingAlgorithm(x_train, y_train, covarianceModel, basis)
algo.run()
krigingResult = algo.getResult()

krigeageMM = krigingResult.getMetaModel()
y_test_MM = krigeageMM(x_test)

process = ot.ConditionedGaussianProcess(krigingResult, myRegularGrid)

# %%
trajectories = process.getSample(10)

graph = trajectories.drawMarginal()
graph.add(plot_data_test(x_test, y_test))
graph.add(plot_data_train(x_train, y_train))
graph.setAxes(True)
graph.setXTitle("X")
graph.setYTitle("Y")
graph.setLegendPosition("topright")
graph.setTitle("10 simulated trajectories")
view = viewer.View(graph)
plt.show()
