# %%
import openturns as ot
from openturns.usecases import ishigami_function
import openturns.viewer as otv


# %%
class LeastAngleRegressionStepwisePCE:
    def __init__(
        self,
        input_sample,
        output_sample,
        distribution,
        basis,
        maximumBasisSize=10,
        wX=None,
        leastSquaresMethodName="SVD",
        fittingAlgorithm=None,
        kParameter=10,
        minAbsCorrelation=0.0,
        verbose=False,
    ):
        """
        Create a polynomial chaos by Least Angle Regression Stepwise (LARS).

        Parameters
        ----------
        input_sample : ot.Sample(size, input_dimension)
            The input sample.
        output_sample : ot.Sample(size, output_dimension)
            The output sample.
        distribution : ot.Distribution(input_dimension)
            The distribution of the input.
        basis : ot.OrthogonalBasis
            The orthogonal basis of functions.
        maximumBasisSize : int, optional
            The maximum number of active basis functions.
        wX : ot.Point(size), optional
            The quadrature weights. The default is None.
        leastSquaresMethodName : str
            The least squares resolution method.
        fittingAlgorithm : FittingAlgorithm or None
            The fitting algorithm used to monitor the sparse basis selection.
            Uses KFold by default.
        kParameter : int
            The number of folds when fittingAlgorithm="KFold".
        minAbsCorrelation : float
            Stop if the best absolute correlation is below this threshold.
        verbose : bool
            If True, print the progression of the algorithm.
        """
        self.input_sample = input_sample
        self.output_sample = output_sample
        self.distribution = distribution
        self.basis = basis
        self.wX = wX
        self.leastSquaresMethodName = leastSquaresMethodName
        if fittingAlgorithm is None:
            self.fittingAlgorithm = ot.KFold(kParameter)
        else:
            self.fittingAlgorithm = fittingAlgorithm
        self.maximumBasisSize = maximumBasisSize
        self.minAbsCorrelation = minAbsCorrelation
        self.verbose = verbose

        self.result = None
        self.activeIndices = None
        self.selectionHistory = []
        self.fittingScoreHistory = []

    def run(self):
        """
        Create the functional chaos metamodel by Least Angle Regression Stepwise.
        """
        # Setup
        transformation = ot.DistributionTransformation(
            self.distribution, self.basis.getMeasure()
        )
        standard_input = transformation(self.input_sample)
        sample_size = standard_input.getSize()
        output_dimension = self.output_sample.getDimension()

        # Create a list of functions
        functions = [self.basis.build(i) for i in range(self.maximumBasisSize)]
        designProxy = ot.DesignProxy(standard_input, functions)

        # Precompute evaluated basis functions as a list of Points
        X = [functions[j](standard_input).asPoint() for j in range(self.maximumBasisSize)]

        fitting = self.fittingAlgorithm

        coefficients_map = {}
        self.selectionHistory = []
        self.fittingScoreHistory = []

        for output_index in range(output_dimension):
            if self.verbose:
                print(f"--- Output marginal {output_index} ---")

            marginal_output = self.output_sample.getMarginal(output_index)
            sample_mean = marginal_output.computeMean()[0]

            # Initialisation
            list_of_active_functions = [0]
            marginal_selection = [0]

            leastSquaresMethod = ot.LeastSquaresMethod.Build(
                self.leastSquaresMethodName, designProxy, list_of_active_functions
            )

            # Compute initial fitting score
            fitting_score = fitting.run(
                standard_input,
                marginal_output,
                ot.Point(sample_size, 1) / sample_size,
                functions,
                list_of_active_functions,
            )

            if self.verbose:
                print(f"  Fitting score = {fitting_score:.4e}")

            marginal_fitting_scores = [fitting_score]

            # Update residuals and initial coefficients
            residuals = marginal_output.asPoint() - ot.Point(sample_size, sample_mean)
            coefficients_dict = {0: sample_mean}

            # Loop stops either when max basis size is reached, or when no degrees of freedom are left
            max_iterations = min(sample_size, self.maximumBasisSize) - 1
            
            for i in range(max_iterations):
                if self.verbose:
                    print(f"Current active indices ({len(list_of_active_functions)})= {list_of_active_functions}")

                # 1. Compute correlations
                v = [X[j].dot(residuals) / sample_size for j in range(self.maximumBasisSize)]

                # 2. Find candidate with maximum absolute correlation with the residual
                C = 0.0
                best_basis_function_index = None

                for j in range(self.maximumBasisSize):
                    if j in list_of_active_functions:
                        continue
                    current_absolute_correlation = abs(v[j])
                    if current_absolute_correlation > C:
                        best_basis_function_index = j
                        C = current_absolute_correlation

                if best_basis_function_index is None:
                    break

                if self.verbose:
                    print(
                        f"  Best index = {best_basis_function_index} "
                        f"with max. abs. corr. = {C:.4e}"
                    )

                # Early stopping criterion
                if C < self.minAbsCorrelation:
                    if self.verbose:
                        print(f"  Stopping early: maximum absolute correlation ({C:.4e}) "
                              f"is below the threshold ({self.minAbsCorrelation:.4e}).")
                    break

                # Add the best candidate to the active set
                list_of_active_functions.append(best_basis_function_index)
                marginal_selection.append(best_basis_function_index)
                coefficients_dict[best_basis_function_index] = 0.0

                # 3. Compute the OLS direction for the updated active set
                leastSquaresMethod = ot.LeastSquaresMethod.Build(
                    self.leastSquaresMethodName, designProxy, list_of_active_functions
                )
                c_ols = leastSquaresMethod.solve(marginal_output.asPoint())

                c_curr = ot.Point(len(list_of_active_functions), 0.0)
                for idx, active_idx in enumerate(list_of_active_functions):
                    c_curr[idx] = coefficients_dict[active_idx]

                d = c_ols - c_curr

                # Compute X_A * d
                X_A_d = ot.Point(sample_size, 0.0)
                for idx, active_idx in enumerate(list_of_active_functions):
                    X_A_d += X[active_idx] * d[idx]

                # 4. Compute inner products with direction
                a = [X[j].dot(X_A_d) / sample_size for j in range(self.maximumBasisSize)]

                # 5. Find step size gamma
                gamma = 1.0
                for k in range(self.maximumBasisSize):
                    if k not in list_of_active_functions:
                        # Forward evaluation
                        den_plus = C - a[k]
                        if den_plus > 1e-12:
                            g_plus = (C - v[k]) / den_plus
                            if 0 < g_plus < gamma:
                                gamma = g_plus

                        # Backward evaluation
                        den_minus = C + a[k]
                        if den_minus > 1e-12:
                            g_minus = (C + v[k]) / den_minus
                            if 0 < g_minus < gamma:
                                gamma = g_minus

                # 6. Update coefficients and residuals
                c_next = c_curr + d * gamma
                for idx, active_idx in enumerate(list_of_active_functions):
                    coefficients_dict[active_idx] = c_next[idx]

                residuals -= X_A_d * gamma

                # 7. Compute CV score (evaluates the OLS model of the active set matching LARS-OLS approach)
                fitting_score = fitting.run(leastSquaresMethod, marginal_output)

                if self.verbose:
                    print(f"  Fitting score = {fitting_score:.4e}")

                marginal_fitting_scores.append(fitting_score)

            # Store the coefficients for this output marginal
            for j in range(len(list_of_active_functions)):
                idx = list_of_active_functions[j]
                if idx not in coefficients_map:
                    coefficients_map[idx] = ot.Point(output_dimension, 0.0)
                coefficients_map[idx][output_index] = coefficients_dict[idx]

            self.selectionHistory.append(marginal_selection)
            self.fittingScoreHistory.append(marginal_fitting_scores)

        # Unpack histories if the output is 1D to preserve backwards compatibility
        if output_dimension == 1:
            self.selectionHistory = self.selectionHistory[0]
            self.fittingScoreHistory = self.fittingScoreHistory[0]

        # Merge active indices and build the final samples and functions
        sorted_indices = sorted(coefficients_map.keys())
        self.activeIndices = ot.Indices(sorted_indices)

        coefficient_list = [coefficients_map[idx] for idx in sorted_indices]
        coefficient_sample = ot.Sample(coefficient_list)

        final_functions = [self.basis.build(idx) for idx in sorted_indices]

        # Create the result
        self.result = ot.FunctionalChaosResult(
            self.input_sample,
            self.output_sample,
            self.distribution,
            transformation,
            transformation.inverse(),
            self.basis,
            self.activeIndices,
            coefficient_sample,
            final_functions,
        )

    def getResult(self):
        """
        Return the functional chaos result.
        """
        if self.result is None:
            self.run()
        return self.result

    def getActiveIndices(self):
        """
        Return the active basis indices.
        """
        if self.result is None:
            self.run()
        return self.activeIndices

    def getSelectionHistory(self):
        """
        Return the LARS selection history.
        """
        if self.result is None:
            self.run()
        return self.selectionHistory

    def getFittingScoreHistory(self):
        """
        Return the fitting score history.
        """
        if self.result is None:
            self.run()
        return self.fittingScoreHistory

    def getFittingAlgorithm(self):
        return self.fittingAlgorithm


# %%
ot.RandomGenerator.SetSeed(0)

# %%
im = ishigami_function.IshigamiModel()
sample_size = 200
input_sample = im.inputDistribution.getSample(sample_size)
output_sample = im.model(input_sample)

# %%
# Create basis
basis = ot.OrthogonalProductPolynomialFactory(
    [
        im.inputDistribution.getMarginal(i)
        for i in range(im.inputDistribution.getDimension())
    ]
)

# %%
maximumBasisSize = 100
print(f"Number of coefficients = {maximumBasisSize}")

# %%
# Execution using the new LARS algorithm
algo = LeastAngleRegressionStepwisePCE(
    input_sample,
    output_sample,
    im.inputDistribution,
    basis,
    maximumBasisSize,
    verbose=True,
    minAbsCorrelation=1.0e-2  # Arbitrary early stopping
)
algo.run()

# %%
# Display outputs
print("Active Indices:", algo.getActiveIndices())
print("Selection History:", algo.getSelectionHistory())
fitting_score_list = algo.getFittingScoreHistory()
print("Fitting Score History:", fitting_score_list)
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
min_index = argmin(fitting_score_list)
fitting_score_min = min(fitting_score_list)
graph = ot.Graph(
    f"{fitting.getClassName()}", "Iteration", f"{fitting.getClassName()} score", True
)
number_of_selected_coefficients = len(fitting_score_list)
cloud = ot.Cloud(range(number_of_selected_coefficients), fitting_score_list)
graph.add(cloud)
graph.setLogScale(ot.GraphImplementation.LOGY)
# Plot min corrected score
cloud = ot.Cloud([min_index], [fitting_score_min])
cloud.setPointStyle("circle")
cloud.setLegend("Min")
graph.add(cloud)
# Plot error factor
curve = ot.Curve([0, number_of_selected_coefficients], [error_factor * fitting_score_min] * 2)
curve.setLineWidth(2.0)
curve.setLegend("Treshold")
graph.add(curve)
view = otv.View(graph)
view.save("LeastAngleRegressionStepwisePCE.png")

# %%