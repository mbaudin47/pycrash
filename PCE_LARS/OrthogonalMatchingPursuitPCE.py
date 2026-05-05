# %%
import openturns as ot
from openturns.usecases import ishigami_function


# %%
class OrthogonalMatchingPursuitPCE:
    def __init__(
        self,
        input_sample,
        output_sample,
        distribution,
        basis,
        totalDegree,
        wX=None,
        leastSquaresMethodName="SVD",
        fittingAlgorithm=None,
        kParameter=10,
        maxActiveFunctions=10,
        minAbsCorrelation=0.0,
        verbose=False,
    ):
        """
        Create a polynomial chaos by Orthogonal Matching Pursuit.

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
        totalDegree : int
            The maximum total degree.
        wX : ot.Point(size), optional
            The quadrature weights. The default is None.
        leastSquaresMethodName : str
            The least squares resolution method.
        fittingAlgorithm : FittingAlgorithm or None
            The fitting algorithm used to monitor the sparse basis selection.
            Uses KFold by default.
        kParameter : int
            The number of folds when fittingAlgorithm="KFold".
        maxActiveFunctions : int, optional
            The maximum number of active basis functions.
            If None, all basis functions are candidates.
        minAbsCorrelation : float
            Stop if the best absolute correlation is below this threshold.
        includeConstant : bool
            If True, start with the constant basis function.
        verbose : bool
            If True, print the progression of the algorithm.

        Returns
        -------
        None.
        """
        self.input_sample = input_sample
        self.output_sample = output_sample
        self.distribution = distribution
        self.basis = basis
        self.totalDegree = totalDegree
        self.wX = wX
        self.leastSquaresMethodName = leastSquaresMethodName
        if fittingAlgorithm is None:
            self.fittingAlgorithm = ot.KFold(kParameter)
        else:
            self.fittingAlgorithm = fittingAlgorithm
        self.maxActiveFunctions = maxActiveFunctions
        self.minAbsCorrelation = minAbsCorrelation
        self.verbose = verbose


        self.result = None
        self.activeIndices = None
        self.selectionHistory = []
        self.fittingScoreHistory = []

    def run(self):
        """
        Create the functional chaos metamodel by Orthogonal Matching Pursuit.

        Returns
        -------
        None.
        """
        # Setup
        enumerateFunction = self.basis.getEnumerateFunction()
        strataIndex = enumerateFunction.getMaximumDegreeStrataIndex(self.totalDegree)
        maximumBasisSize = enumerateFunction.getStrataCumulatedCardinal(strataIndex)
        transformation = ot.DistributionTransformation(self.distribution, self.basis.getMeasure())
        standard_input = transformation(self.input_sample)
        # Compute coefficients
        sample_size = standard_input.getSize()
        transformation = ot.DistributionTransformation(im.inputDistribution, basis.getMeasure())
        standard_input = transformation(input_sample)
        # Create a list of functions
        functions = [basis.build(i) for i in range(self.maxActiveFunctions)]
        designProxy = ot.DesignProxy(standard_input, functions)
        # Initialisation
        list_of_active_functions = [0]  # Initialize with constant basis
        leastSquaresMethod = ot.LeastSquaresMethod.Build(
            self.leastSquaresMethodName, designProxy, list_of_active_functions
        )
        residuals = output_sample.asPoint()
        # Select your best fitting algorithm
        # fitting = ot.CorrectedLeaveOneOut()
        kParameter = 10
        fitting = ot.KFold(kParameter)
        # Compute initial fitting score
        fitting_score = fitting.run(
            standard_input,
            output_sample,
            ot.Point(sample_size, 1) / sample_size,
            functions,
            list_of_active_functions,
        )
        print(f"  Fitting score = {fitting_score:.4e}")
        fitting_score_list = [fitting_score]
        # Update residuals
        residuals -= ot.Point(sample_size, output_sample.computeMean()[0])
        # TODO: Repeat this for output each marginal
        for i in range(self.maxActiveFunctions - 1):
            # Find candidate with maximum absolute correlation with the residual
            print(f"Current active indices = {list_of_active_functions}")
            maximum_absolute_correlation = 0.0
            best_basis_function_index = None
            for j in range(self.maxActiveFunctions):
                if j in list_of_active_functions:
                    # Skip this basis (already active)
                    continue
                current_basis_function = basis.build(j)
                basis_function_value = current_basis_function(standard_input)
                current_absolute_correlation = (
                    abs(residuals.dot(basis_function_value.asPoint())) / sample_size
                )
                if current_absolute_correlation > maximum_absolute_correlation:
                    best_basis_function_index = j
                    maximum_absolute_correlation = current_absolute_correlation
            print(
                f"  Best index = {best_basis_function_index} "
                f"with max. abs. corr. = {maximum_absolute_correlation:.4e}"
            )
            # Update the LS method
            leastSquaresMethod.update([best_basis_function_index], list_of_active_functions, [])
            # Add the best candidate to the active set
            list_of_active_functions.append(best_basis_function_index)
            # Update the coefficients
            coefficients = leastSquaresMethod.solve(self.output_sample.asPoint())
            # Update the residuals
            designMatrix = leastSquaresMethod.computeWeightedDesign()
            residuals = self.output_sample.asPoint() - designMatrix * coefficients
            # Compute corrected leave-out score
            # After https://github.com/openturns/openturns/issues/2948
            fitting_score = fitting.run(leastSquaresMethod, self.output_sample)
            print(f"  Fitting score = {fitting_score:.4e}")
            fitting_score_list.append(fitting_score)

        coefficientSample = ot.Sample.BuildFromPoint(coefficients)

        # Create the result
        functions = [basis.build(i) for i in list_of_active_functions]
        result = ot.FunctionalChaosResult(
            input_sample,
            output_sample,
            im.inputDistribution,
            transformation,
            transformation.inverse(),
            basis,
            list_of_active_functions,
            coefficientSample,
            functions,
        )

    def getResult(self):
        """
        Return the functional chaos result.

        Returns
        -------
        result : ot.FunctionalChaosResult
            The metamodel.
        """
        if self.result is None:
            self.run()
        return self.result

    def getActiveIndices(self):
        """
        Return the active basis indices.

        Returns
        -------
        activeIndices : ot.Indices
            The selected basis indices.
        """
        if self.result is None:
            self.run()
        return self.activeIndices

    def getSelectionHistory(self):
        """
        Return the OMP selection history.

        Returns
        -------
        selectionHistory : list
            The iteration history.
        """
        if self.result is None:
            self.run()
        return self.selectionHistory

    def getFittingScoreHistory(self):
        """
        Return the fitting score history.

        Returns
        -------
        fittingScoreHistory : list
            The fitting scores.
        """
        if self.result is None:
            self.run()
        return self.fittingScoreHistory


# %%
# [Markdown]
# Notice that this script does not use Numpy _at all_.

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
basis

# %%
degree = 4
algo = OrthogonalMatchingPursuitPCE(
    input_sample, output_sample, im.inputDistribution, basis, degree
)
algo.run()
result=algo.getResult()
result

# %%
