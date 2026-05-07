# %%
import openturns as ot
import OrthogonalMatchingPursuitPCE as omppce
from openturns.usecases import ishigami_function
import openturns.viewer as otv

# %%
ot.RandomGenerator.SetSeed(0)

# %%
im = ishigami_function.IshigamiModel()
sample_size = 200
experiment = ot.MonteCarloExperiment(im.inputDistribution, sample_size)
input_sample, wX = experiment.generateWithWeights()
output_sample = im.model(input_sample)

# %%
# Create basis
input_dimension = im.inputDistribution.getDimension()
basis = ot.OrthogonalProductPolynomialFactory(
    [im.inputDistribution.getMarginal(i) for i in range(input_dimension)]
)

# %%
maximumBasisSize = 100
print(f"Number of coefficients = {maximumBasisSize}")

# %%
# [Markdown]
# KFold : Cholesky
# CorrectedLeaveOneOut : Nécessite la diagonal de l'inverse de la 
# matrice H > Choisir SVD

# %%
# Set minAbsCorrelation to zero to see all path.
algo = omppce.OrthogonalMatchingPursuitPCE(
    input_sample,
    output_sample,
    im.inputDistribution,
    basis,
    maximumBasisSize,
    wX=wX,
    verbose=True,
    leastSquaresMethodName="Cholesky",
    minAbsCorrelation=0.0,  # Arbitrary early stopping
)
algo.run()

# %%
# Display outputs
print("Active Indices:", algo.getActiveIndices())
print("Selection History:", algo.getSelectionHistory())
fitting_score_history = algo.getFittingScoreHistory()
print("Fitting Score History:", fitting_score_history)
result = algo.getResult()
result

# %%
fitting = algo.getFittingAlgorithm()

# %%
input_test = im.inputDistribution.getSample(1000)
output_test = im.model(input_test)
meta_model = result.getMetaModel()
validation = ot.MetaModelValidation(output_test, meta_model(input_test))
print(f"Q2 = {validation.computeR2Score()[0]:.15f}")


# %%
def argmin(liste):
    # This can be avoided if using np.argmin.
    # But we want to show that Numpy can be avoided here,
    # and rely only on OpenTURNS for the OMP algorithm.
    if not liste:
        return None

    indice_min = 0
    valeur_min = liste[0]

    for i in range(1, len(liste)):
        if liste[i] < valeur_min:
            valeur_min = liste[i]
            indice_min = i

    return indice_min


# %%
threshold = ot.ResourceMap.GetAsScalar("SparseMethod-ErrorThreshold")
error_factor = ot.ResourceMap.GetAsScalar("SparseMethod-MaximumErrorFactor")
marginal_fitting_score = fitting_score_history.getMarginal(0).asPoint()
min_index = argmin(marginal_fitting_score)
fitting_score_min = min(marginal_fitting_score)

#%%
graph = ot.Graph(
    f"OMP with {fitting.getClassName()}",
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
view = otv.View(graph)
view.save("OrthogonalMatchingPursuitPCE.png")

# %%
