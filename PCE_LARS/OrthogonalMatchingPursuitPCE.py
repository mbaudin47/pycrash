"""Implements the OMP selection method of a polynomial chaos expansion algorithm in Python.

This implements 2 algorithms:
- Algorithm B.1 Orthogonal matching pursuit (OMP) page 628 of (Lüthen, et al., 2021),
- Algorithm B.1 with with CV using Corrected Leave-One-Out or K-Fold.

Reference
---------
- Lüthen, N., Marelli, S., & Sudret, B. (2021).
  Sparse polynomial chaos expansions: Literature survey and benchmark.
  SIAM/ASA Journal on Uncertainty Quantification, 9(2), 593-649.
- https://gist.github.com/mbaudin47/87a09578aef2e38b498f2f5c5cda193b
"""

# %%
import openturns as ot


# %%
class OrthogonalMatchingPursuitPCE:
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
        Create the functional chaos metamodel by Orthogonal Matching Pursuit.
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

        # Precompute the entire design matrix
        X = designProxy.computeDesign(range(self.maximumBasisSize))

        coefficients_map = {}
        self.selectionHistory = []
        self.fittingScoreHistory = ot.Sample(self.maximumBasisSize - 1, output_dimension)

        # Initialize the list of active functions over all outputs
        list_of_active_functions = [0]

        for output_index in range(output_dimension):
            if self.verbose:
                print(f"--- Output marginal {output_index} ---")

            marginal_output = self.output_sample.getMarginal(output_index)

            # Initialisation for current output
            marginal_selection = [0]

            leastSquaresMethod = ot.LeastSquaresMethod.Build(
                self.leastSquaresMethodName, designProxy, self.wX, marginal_selection
            )
            residuals = marginal_output.asPoint()

            # Compute initial fitting score
            fitting_score = self.fittingAlgorithm.run(
                leastSquaresMethod,
                marginal_output,
            )

            if self.verbose:
                print(f"  Fitting score = {fitting_score:.4e}")

            marginal_fitting_scores = [fitting_score]

            # Update residuals
            residuals -= ot.Point(sample_size, marginal_output.computeMean()[0])
            coefficients = ot.Point([marginal_output.computeMean()[0]])

            for i in range(self.maximumBasisSize - 1):
                if self.verbose:
                    print(
                        f"Current active indices ({len(list_of_active_functions)})= {list_of_active_functions}"
                    )
                maximum_absolute_correlation = 0.0
                best_basis_function_index = None

                # 1. Compute correlations
                v = (X.transpose() * residuals) / sample_size

                # 2. Find candidate with maximum absolute correlation with the residual
                for j in range(self.maximumBasisSize):
                    if j in list_of_active_functions:
                        # Skip this basis (already active)
                        continue
                    current_absolute_correlation = abs(v[j]) / sample_size
                    if current_absolute_correlation > maximum_absolute_correlation:
                        best_basis_function_index = j
                        maximum_absolute_correlation = current_absolute_correlation

                if self.verbose:
                    print(
                        f"  Best index = {best_basis_function_index} "
                        f"with max. abs. corr. = {maximum_absolute_correlation:.4e}"
                    )
                # 3. Early stopping criterion ---
                if maximum_absolute_correlation < self.minAbsCorrelation:
                    if self.verbose:
                        print(
                            f"  Stopping early: maximum absolute correlation ({maximum_absolute_correlation:.4e}) "
                            f"is below the threshold ({self.minAbsCorrelation:.4e})."
                        )
                    break

                # 4. Update the LS method
                leastSquaresMethod.update(
                    [best_basis_function_index], marginal_selection, []
                )

                # 5. Add the best candidate to the active set
                list_of_active_functions.append(best_basis_function_index)
                marginal_selection.append(best_basis_function_index)

                # 6. Update the coefficients
                coefficients = leastSquaresMethod.solve(marginal_output.asPoint())

                # 7. Update the residuals
                designMatrix = leastSquaresMethod.computeWeightedDesign()
                residuals = marginal_output.asPoint() - designMatrix * coefficients

                # 8. Compute corrected leave-out score
                fitting_score = self.fittingAlgorithm.run(
                    leastSquaresMethod, marginal_output
                )

                if self.verbose:
                    print(f"  Fitting score = {fitting_score:.4e}")

                self.fittingScoreHistory[i, output_index] = fitting_score

            # Store the coefficients for this output marginal
            for j in range(len(list_of_active_functions)):
                idx = list_of_active_functions[j]
                if idx not in coefficients_map:
                    coefficients_map[idx] = ot.Point(output_dimension, 0.0)
                coefficients_map[idx][output_index] = coefficients[j]

            self.selectionHistory.append(marginal_selection)

        # Merge active indices and build the final samples and functions
        sorted_indices = sorted(coefficients_map.keys())
        self.activeIndices = ot.Indices(sorted_indices)

        coefficient_list = [coefficients_map[idx] for idx in sorted_indices]
        coefficient_sample = ot.Sample(coefficient_list)

        final_functions = [functions[idx] for idx in sorted_indices]

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
        Return the OMP selection history.
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
