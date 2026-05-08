"""
Check the OrthogonalMatchingPursuitPCE class with several outputs.

This script is based on a function which has two outputs :
- marginal 1: Ishigami,
- marginal 2: G-Sobol'.
"""

# %%
import openturns as ot
import openturns.viewer as otv
import OrthogonalMatchingPursuitPCE as omppce
import numpy as np
import otbenchmark as otb


# %%
# Implement a function with input dimension 3 
# and output dimension 2.
# The first marginal is Ishigami and the second is G-Sobol'.
class MultiOutputFunction(ot.OpenTURNSPythonFunction):
    def __init__(self):
        input_dimension = 3
        super().__init__(input_dimension, 2)
        ishigami_problem = otb.IshigamiSensitivity()
        self.ishigami_model = ishigami_problem.getFunction()
        gsobol_problem = otb.GSobolSensitivity()
        self.gsobol_model = gsobol_problem.getFunction()
        self.distribution = ot.JointDistribution(
            [ot.Uniform(0.0, 1.0)] * input_dimension
        )
        self.ishigami_distribution = ishigami_problem.getInputDistribution()
        self.gsobol_distribution = gsobol_problem.getInputDistribution()
        self.ishigami_transformation = ot.DistributionTransformation(
            self.distribution,
            self.ishigami_distribution,
        )
        self.gsobol_transformation = ot.DistributionTransformation(
            self.distribution,
            self.gsobol_distribution,
        )

    def getInputDistribution(self):
        return self.distribution

    def _exec(self, X):
        # Evaluate Ishigami
        scaledX = self.ishigami_transformation(X)
        Y1 = self.ishigami_model(scaledX)[0]
        # Evaluate G-Sobol'
        scaledX = self.gsobol_transformation(X)
        Y2 = self.gsobol_model(scaledX)[0]
        return [Y1, Y2]


# %%
# Create the function
multioutput_function = MultiOutputFunction()
distribution = multioutput_function.getInputDistribution()
model = ot.Function(multioutput_function)
x = distribution.getRealization()
y = model(x)
print(y)

# %%
# Create a (X, Y) sample
sample_size = 1000
experiment = ot.MonteCarloExperiment(distribution, sample_size)
input_sample, wX = experiment.generateWithWeights()
output_sample = model(input_sample)
print("Entrées (X) :\n", input_sample[:5])
print("Sorties calculées (Y) :\n", output_sample[:5])

# %%
# Visualize (X, Y)
graph = ot.VisualTest.DrawPairsXY(input_sample, output_sample)
graph.setTitle("Multi-output function: Ishigami, G-Sobol'")
view = otv.View(graph, figure_kw={"figsize": (6.0, 4.0)})
otv.View.ShowAll()

# %%
# Create basis
input_dimension = distribution.getDimension()
basis = ot.OrthogonalProductPolynomialFactory(
    [distribution.getMarginal(i) for i in range(input_dimension)]
)


# %%
# Compute PCE
maximumBasisSize = 100
algo = omppce.OrthogonalMatchingPursuitPCE(
    input_sample,
    output_sample,
    distribution,
    basis,
    maximumBasisSize,
    wX=wX,
    verbose=False,
    leastSquaresMethodName="Cholesky",
    minAbsCorrelation=0.0,  # Arbitrary early stopping
)
algo.run()

# %%
# Display outputs
print("Active Indices:", algo.getActiveIndices())
selection_history = algo.getSelectionHistory()
print("Selection History:", selection_history)
fitting_score_history = algo.getFittingScoreHistory()
print("Fitting Score History:", fitting_score_history)
result = algo.getResult()
result


# %%
# Validation
input_test = distribution.getSample(1000)
output_test = model(input_test)
meta_model = result.getMetaModel()
validation = ot.MetaModelValidation(output_test, meta_model(input_test))
output_dimension = model.getOutputDimension()
r2Score = validation.computeR2Score()
for i in range(output_dimension):
    print(f"Q2[{i}] = {r2Score[i]:.6f}")

# %%
error_factor = ot.ResourceMap.GetAsScalar("SparseMethod-MaximumErrorFactor")


# %%
# Plot the KFold score as a function of the iterations for each output
output_names = ["Ishigami", "G-Sobol'"]
fitting = algo.getFittingAlgorithm()
grid = ot.GridLayout(1, 2)
for i in range(output_dimension):
    marginal_fitting_score = fitting_score_history.getMarginal(i).asPoint()
    min_index = np.argmin(marginal_fitting_score)
    fitting_score_min = min(marginal_fitting_score)
    graph = ot.Graph(
        f"OMP with {fitting.getClassName()}, {output_names[i]}",
        "Iteration",
        f"{fitting.getClassName()} score",
        True,
    )
    number_of_selected_coefficients = len(fitting_score_history)
    cloud = ot.Cloud(range(number_of_selected_coefficients), marginal_fitting_score)
    graph.add(cloud)
    graph.setLogScale(ot.GraphImplementation.LOGY)
    # Plot min corrected score
    cloud = ot.Cloud([min_index], [fitting_score_min])
    cloud.setPointStyle("circle")
    cloud.setLegend("Min")
    graph.add(cloud)
    # Plot error factor
    curve = ot.Curve(
        [0, number_of_selected_coefficients], [error_factor * fitting_score_min] * 2
    )
    curve.setLineWidth(2.0)
    curve.setLegend("Treshold")
    graph.add(curve)
    if i > 0:
        graph.setYTitle("")
    grid.setGraph(0, i, graph)

view = otv.View(grid, figure_kw={"figsize": (8.0, 3.0)})
view.save("check_OMP_multioutput.png")

# %%
