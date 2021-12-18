#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a polynomial chaos. 
"""

import openturns as ot

def BuildAdaptiveBasis(distribution_collection, quasi_norm = 1.0):
    """
    Create a multivariate basis using a collection of distributions.
    
    If quasi_norm is equal to 1, then a linear enumeration rule is 
    used. 
    This rule is also called "graded reverse-lexicographic ordering".
    
    If quasi_norm is not equal to 1, then an hyperbolic enumeration 
    rule is used.

    We often use values of quasi_norm in the [0.5, 1.0] range. 
    Small values create very sparse models, with high marginal 
    degree and very few interactions. 

    Parameters
    ----------
    distribution_collection : list(ot.Distribution)
        The list of marginal, univariate, distributions.
    quasi_norm : float, optional, in (0.0, 1.0]
        The quasi-norm of the Hyperbolic enumeratation rule. The default is 1.0.

    Returns
    -------
    multivariateBasis : OrthogonalProductPolynomialFactory
        The multivariate basis.

    """
    input_dimension = len(distribution_collection)
    if quasi_norm == 1.0:
        multivariateBasis = ot.OrthogonalProductPolynomialFactory(distribution_collection)
    else:
        polynomial_collection = ot.PolynomialFamilyCollection(input_dimension)
        for i in range(input_dimension):
            polynomial_collection[i] = ot.StandardDistributionPolynomialFactory(distribution_collection[i])
        enumerate_function = ot.HyperbolicAnisotropicEnumerateFunction(input_dimension, quasi_norm)
        multivariateBasis = ot.OrthogonalProductPolynomialFactory(polynomial_collection, enumerate_function)
    return multivariateBasis


def compute_strata_from_degree(
    enumerateFunction, total_degree=10, maximum_strata_index=100
):
    """
    Compute the minimum index of the strata corresponding to a given total degree.
    
    This function is used when we consider an adaptive basis such 
    as the hyperbolic enumeration rule.
    This is the smallest index of a strata which contains a multiindex 
    having the required total degree.
    Usually, this function is used with a large maximumu strata index, so that 
    a large number of stratas are explored until the required total degree 
    is reached.

    Parameters
    ----------
    enumerateFunction : ot.EnumerateFunction()
        The enumerate function.
    total_degree : int
        The total degree to reach.
    maximum_strata_index : int
        The maximum number of strata to try.

    Returns
    -------
    degree_strata_index : int
        The index of the strata containing a multiindex having given total 
        degree.
        It is lower of equal to maximum_strata_index.
        It may be equal to maximum_strata_index if no multiindex 
        has reached the given total degree.

    """
    is_degree_reached = False
    degree_strata_index = 0
    for strata_index in range(1 + maximum_strata_index):
        if is_degree_reached:
            break
        strata_cardinal = enumerateFunction.getStrataCardinal(strata_index)
        cumulated_cardinal = enumerateFunction.getStrataCumulatedCardinal(strata_index)
        number_of_indices_in_strata = cumulated_cardinal - strata_cardinal
        for i in range(number_of_indices_in_strata, cumulated_cardinal):
            multiindex = enumerateFunction(i)
            multiindex_degree = sum(multiindex)
            if multiindex_degree >= total_degree:
                is_degree_reached = True
                break
            else:
                degree_strata_index = strata_index
    return degree_strata_index


class PolynomialChaosFactory:

    def BuildBasisFromDistributionCollection(distribution_collection, quasi_norm = 1.0):
        """
        Build multivariate basis from a collection of distributions

        Parameters
        ----------
        distribution_collection : list(ot.Distribution)
            The list of input marginal distributions.
        quasi_norm : float, in [0.0, 1.0]
            If not equal to 1, the quasi-norm of the Hyperbolic rule.
            Default is 1, which corresponds to the linear enumeration rule.

        Returns
        -------
        distribution : ot.Distribution(input_dimension)
            The input distribution.
        multivariateBasis : ot.OrthogonalProductPolynomialFactory()
            The multivariate polynomial basis.

        """
        distribution = ot.ComposedDistribution(distribution_collection)
        multivariateBasis = BuildAdaptiveBasis(distribution_collection, quasi_norm)
        return distribution, multivariateBasis

    def BuildBasisFromData(input_sample, quasi_norm = 1.0):
        """
        Build multivariate basis from an input sample
        
        This method may produce a Distribution with dependency, 
        if the sample has any.

        Parameters
        ----------
        input_sample : ot.Sample(size, input_dimension)
            The input sample.
        quasi_norm : float, in [0.0, 1.0]
            If not equal to 1, the quasi-norm of the Hyperbolic rule.
            Default is 1, which corresponds to the linear enumeration rule.

        Returns
        -------
        distribution : ot.Distribution(input_dimension)
            The input distribution.
        multivariateBasis : ot.OrthogonalProductPolynomialFactory()
            The multivariate polynomial basis.

        """
        input_dimension = input_sample.getDimension()
        distribution = ot.FunctionalChaosAlgorithm.BuildDistribution(input_sample)
        distribution_collection = [distribution.getMarginal(i) for i in range(input_dimension)]
        multivariateBasis = BuildAdaptiveBasis(distribution_collection, quasi_norm)
        return distribution, multivariateBasis

    def BuildBasisFromKernelSmoothing(input_sample, quasi_norm = 1.0):
        """
        Build multivariate basis from an input sample using KDE

        Parameters
        ----------
        input_sample : ot.Sample(size, input_dimension)
            The input sample.
        quasi_norm : float, in [0.0, 1.0]
            If not equal to 1, the quasi-norm of the Hyperbolic rule.
            Default is 1, which corresponds to the linear enumeration rule.

        Returns
        -------
        distribution : ot.Distribution(input_dimension)
            The input distribution.
        multivariateBasis : ot.OrthogonalProductPolynomialFactory()
            The multivariate polynomial basis.

        """
        input_dimension = input_sample.getDimension()
        distribution = ot.KernelSmoothing().build(input_sample)
        distribution_collection = [distribution.getMarginal(i) for i in range(input_dimension)]
        multivariateBasis = BuildAdaptiveBasis(distribution_collection, quasi_norm)
        return distribution, multivariateBasis
    
    def BuildBasisFromHistogram(input_sample, quasi_norm = 1.0):
        """
        Build multivariate basis from an input sample using histogram.
        
        The created distribution has independent marginals. 

        Parameters
        ----------
        input_sample : ot.Sample(size, input_dimension)
            The input sample.
        quasi_norm : float, in [0.0, 1.0]
            If not equal to 1, the quasi-norm of the Hyperbolic rule.
            Default is 1, which corresponds to the linear enumeration rule.

        Returns
        -------
        distribution : ot.Distribution(input_dimension)
            The input distribution.
        multivariateBasis : ot.OrthogonalProductPolynomialFactory()
            The multivariate polynomial basis.

        """
        input_dimension = input_sample.getDimension()
        distribution_collection = []
        for i in range(input_dimension):
            marginal = ot.HistogramFactory().build(input_sample[:, i])
            distribution_collection.append(marginal)
        distribution = ot.ComposedDistribution(distribution_collection)
        multivariateBasis = BuildAdaptiveBasis(distribution_collection, quasi_norm)
        return distribution, multivariateBasis


    def __init__(self, totalDegree, multivariateBasis, distribution):
        """
        Create a polynomial chaos.
        
        Creating a polynomial chaos requires two ingredients:
        - the adaptive basis, which specify how to create the 
        functionnal basis ; We create that basis using an enumeration rule.
        - the projection strategy, which specify how to compute the 
        coefficients; We can compute them from either regression 
        of integration.

        Parameters
        ----------
        multivariateBasis : ot.Basis
            The multivariate orthogonal polynomial basis
        totalDegree : int
            The maximum total polynomial degree.
            The total polynomial degree is the sum of the marginal degrees
        distribution : ot.Distribution
            The distribution of the input random vector

        Returns
        -------
        None.

        """
        self.totalDegree = totalDegree
        self.multivariateBasis = multivariateBasis
        self.distribution = distribution
        self.maximum_strata_index=100
        return

    def setMaximumStrataIndex(self, maximum_strata_index):
        """
        Set the maximum index of a strata when using an adaptive basis.
        
        This method is usually used when using an hyperbolic enumeration 
        rule.
        
        The parameter is used when computing the number of terms in the 
        basis. 
        It is the maximum number of strata which are explored when searching
        for the strata which achieves the required total degree, if any.
        See compute_strata_from_degree for further details.

        Parameters
        ----------
        maximum_strata_index : int, greater than 1
            The maximum index of a strata.

        Returns
        -------
        None.

        """
        self.maximum_strata_index=maximum_strata_index
        return None

    def _createAdaptiveStrategy(self):
        enumerateFunction = self.multivariateBasis.getEnumerateFunction()
        adaptive_basis_name = enumerateFunction.getClassName()
        if adaptive_basis_name == 'HyperbolicAnisotropicEnumerateFunction':
            strata_index = compute_strata_from_degree(
                enumerateFunction, self.total_degree, self.maximum_strata_index
            )
            number_of_terms_in_basis = enumerateFunction.getStrataCumulatedCardinal(strata_index)
        else:
            number_of_terms_in_basis = enumerateFunction.getStrataCumulatedCardinal(self.totalDegree)
        adaptiveStrategy = ot.FixedStrategy(self.multivariateBasis, number_of_terms_in_basis)
        return adaptiveStrategy

    def buildFromRegression(self, inputTrain, outputTrain, 
                            use_model_selection=True):
        """
        Create a sparse polynomial chaos with least squares.
    
        * Uses the enumeration rule from multivariateBasis.
        * Uses LeastSquaresStrategy to compute the coefficients from
        linear least squares.
        * Uses LeastSquaresMetaModelSelectionFactory to select the polynomials
        in the basis using least angle regression stepwise (LARS)
        * Uses FixedStrategy to keep all coefficients that LARS has selected,
        up to the given maximum total degree.
    
        Parameters
        ----------
        inputTrain : ot.Sample(n)
            The input training design of experiments with n points
        outputTrain : ot.Sample(n)
            The input training design of experiments with n points
        use_model_selection : bool
            Set to True to use LARS model selection.
            Set to False to compute full polynomial chaos.
    
        Returns
        -------
        result : ot.FunctionalChaosResult
            The polynomial chaos result
        """
        # 1. Create the projection strategy
        if use_model_selection:
            # LARS model selection
            selectionAlgorithm = ot.LeastSquaresMetaModelSelectionFactory()
        else:
            # No model selection
            selectionAlgorithm = ot.PenalizedLeastSquaresAlgorithmFactory()
        projectionStrategy = ot.LeastSquaresStrategy(
            inputTrain, outputTrain, selectionAlgorithm
        )
        # 2. Create the adaptive basis
        adaptiveStrategy = self._createAdaptiveStrategy()
        # 3. Create the polynomial chaos
        chaosalgo = ot.FunctionalChaosAlgorithm(
            inputTrain,
            outputTrain,
            self.distribution,
            adaptiveStrategy,
            projectionStrategy,
        )
        return chaosalgo

    def buildFullChaosFromIntegration(self, g_function):
        """
        Create a full polynomial chaos with integration based on Gaussian quadrature.
    
        * Uses the enumeration rule from multivariateBasis.
        * Uses GaussProductExperiment to create a design of experiments using
        a given total degree.
        * Uses IntegrationStrategy to compute the coefficients using
        integration.
        * Uses FixedStrategy to keep all coefficients.
    
        When the number of input variables
        or the required marginal degree is large, the design of experiments is
        very large.
    
        Parameters
        ----------
        function : ot.Function
            The function to create the metamodel from.
    
        Returns
        -------
        result : ot.FunctionalChaosResult
            The polynomial chaos result
        """

        # 1. Create the adaptive basis
        adaptiveStrategy = self._createAdaptiveStrategy()
        # 2. Create the projection strategy
        distribution_measure = self.multivariateBasis.getMeasure()
        dim_input = g_function.getInputDimension()
        totalDegreeList = [self.totalDegree] * dim_input
        experiment = ot.GaussProductExperiment(distribution_measure, totalDegreeList)
        projectionStrategy = ot.IntegrationStrategy(experiment)
        # 3. Create the polynomial chaos
        chaosalgo = ot.FunctionalChaosAlgorithm(
            g_function, self.distribution, adaptiveStrategy, projectionStrategy
        )
        return chaosalgo

