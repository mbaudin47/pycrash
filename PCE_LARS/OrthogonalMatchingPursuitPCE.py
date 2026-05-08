"""Implements the OMP selection method of a polynomial chaos expansion algorithm in Python.

This implements the Algorithm B.1 Orthogonal matching pursuit (OMP)
page 628 of (Lüthen, et al., 2021).

TODO
----
- Implement Algorithm B.1 with with CV using Corrected Leave-One-Out or K-Fold.
  Currently, the K-Fold score is evaluated, but not used in the algorithm.
  Instead the early stopping rule is based on a threshold on the
  maximum absolute correlation.

Reference
---------
- Lüthen, N., Marelli, S., & Sudret, B. (2021).
  Sparse polynomial chaos expansions: Literature survey and benchmark.
  SIAM/ASA Journal on Uncertainty Quantification, 9(2), 593-649.
- https://gist.github.com/mbaudin47/87a09578aef2e38b498f2f5c5cda193b
"""

import openturns as ot


class OrthogonalMatchingPursuitPCE:
    def __init__(
        self,
        input_sample,
        output_sample,
        distribution,
        basis,
        candidateBasisSize=10,
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
        candidateBasisSize : int, optional
            The number of candidate basis functions.
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
        sample_size = input_sample.getSize()
        if output_sample.getSize() != sample_size:
            raise ValueError(
                f"Input sample has size {sample_size} but output sample has size {output_sample.getSize()}."
            )
        input_dimension = input_sample.getDimension()
        if distribution.getDimension() != input_dimension:
            raise ValueError(
                f"Distribution has dimension {distribution.getDimension()} but input sample has dimension {input_dimension}."
            )
        if wX is None:
            wX = ot.Point(sample_size, 1.0 / sample_size)
        if not all(abs(w - 1.0 / sample_size) < 1.0e-14 for w in wX):
            raise NotImplementedError("Non-uniform weights are not yet supported.")
        self.input_sample = input_sample
        self.output_sample = output_sample
        self.distribution = distribution
        self.basis = basis
        self.wX = wX
        self.leastSquaresMethodName = leastSquaresMethodName
        if fittingAlgorithm is None:
            if kParameter > sample_size:
                raise ValueError(
                    f"K-Fold parameter is {kParameter} but sample size is {sample_size}."
                )
            if kParameter < 2:
                raise ValueError(
                    f"K-Fold parameter is {kParameter} but should be at least 2."
                )
            self.fittingAlgorithm = ot.KFold(kParameter)
        else:
            self.fittingAlgorithm = fittingAlgorithm
        self.candidateBasisSize = candidateBasisSize
        self.minAbsCorrelation = minAbsCorrelation
        self.verbose = verbose

        self.result = None
        self.activeIndices = None
        self.selectionHistory = []
        self.fittingScoreHistory = None

    def run(self):
        """
        Create the functional chaos metamodel by Orthogonal Matching Pursuit.
        """
        if self.result is not None:
            return

        # Setup
        transformation = ot.DistributionTransformation(
            self.distribution, self.basis.getMeasure()
        )
        standard_input = transformation(self.input_sample)
        sample_size = standard_input.getSize()
        output_dimension = self.output_sample.getDimension()

        # Create a list of functions
        functions = [self.basis.build(i) for i in range(self.candidateBasisSize)]
        designProxy = ot.DesignProxy(standard_input, functions)

        # Precompute the entire design matrix
        X = designProxy.computeDesign(range(self.candidateBasisSize))
        XT = X.transpose()

        coefficients_map = {}
        self.fittingScoreHistory = ot.Sample(self.candidateBasisSize, output_dimension)

        for output_index in range(output_dimension):
            if self.verbose:
                print(f"--- Output marginal {output_index} ---")

            marginal_output = self.output_sample.getMarginal(output_index)

            # Initialisation for current output
            marginal_selection = [0]

            leastSquaresMethod = ot.LeastSquaresMethod.Build(
                self.leastSquaresMethodName, designProxy, self.wX, marginal_selection
            )
            rightHandSide = marginal_output.asPoint()

            # Compute initial fitting score
            fitting_score = self.fittingAlgorithm.run(
                leastSquaresMethod,
                marginal_output,
            )

            if self.verbose:
                print(f"  Fitting score = {fitting_score:.4e}")

            self.fittingScoreHistory[0, output_index] = fitting_score

            # Set residuals
            # Note: With non equal weights, the output sample mean must be weighted,
            # i.e. the next line is wrong.
            # This would be sum_i w_i * y_i
            marginal_output_mean = marginal_output.computeMean()[0]
            residuals = rightHandSide - ot.Point(sample_size, marginal_output_mean)

            # Initialize coefficients (useful in case of early stopping)
            coefficients = [marginal_output_mean]

            for i in range(self.candidateBasisSize - 1):
                if self.verbose:
                    print(
                        f"Current active indices ({len(marginal_selection)})={marginal_selection}"
                    )

                # 1. Compute correlations
                # Note: With non equal weights, the sample correlation must be weighted,
                # i.e. the next line is wrong.
                # This would be c_j = sum_i w_i * psi_j(x_i) * r_i
                v = (XT * residuals) / sample_size

                # 2. Find candidate with maximum absolute correlation with the residual
                maximum_absolute_correlation = 0.0
                best_basis_function_index = None
                for j in range(self.candidateBasisSize):
                    if j in marginal_selection:
                        # Skip this basis (already active for the current marginal)
                        continue
                    current_absolute_correlation = abs(v[j])
                    if current_absolute_correlation > maximum_absolute_correlation:
                        best_basis_function_index = j
                        maximum_absolute_correlation = current_absolute_correlation

                if self.verbose:
                    print(
                        f"  Best index = {best_basis_function_index} "
                        f"with max. abs. corr. = {maximum_absolute_correlation:.4e}"
                    )
                # 3. Early stopping criterion ---
                if (
                    best_basis_function_index is None
                    or maximum_absolute_correlation < self.minAbsCorrelation
                ):
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
                marginal_selection.append(best_basis_function_index)

                # 6. Update the coefficients
                coefficients = leastSquaresMethod.solve(rightHandSide)

                # 7. Update the residuals
                designMatrix = leastSquaresMethod.computeWeightedDesign()
                # Question: Would designMatrix = X.getMarginal(marginal_selection) work?
                residuals = rightHandSide - designMatrix * coefficients

                # 8. Compute corrected leave-out score
                fitting_score = self.fittingAlgorithm.run(
                    leastSquaresMethod, marginal_output
                )

                if self.verbose:
                    print(f"  Fitting score = {fitting_score:.4e}")

                self.fittingScoreHistory[i + 1, output_index] = fitting_score

            # Store the coefficients for this output marginal
            for j in range(len(marginal_selection)):
                idx = marginal_selection[j]
                if idx not in coefficients_map:
                    coefficients_map[idx] = ot.Point(output_dimension, 0.0)
                coefficients_map[idx][output_index] = coefficients[j]

            self.selectionHistory.append(marginal_selection.copy())

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
