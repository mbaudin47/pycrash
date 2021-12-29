#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a polynomial chaos. 
"""

import openturns as ot


class SuperHistogramFactory:
    def __init__(self):
        """
        Create a SuperHistogramFactory.
        """
        return None

    def build(self, sample):
        """
        Build a multivariate histogram, with independent marginals.

        TODO: integrate that into our favorite library.

        Parameters
        ----------
        sample : ot.Sample(size, dimension)
            The sample.

        Returns
        -------
        distribution : ot.Distribution(dimension)
            The distribution.

        """
        dimension = sample.getDimension()
        distribution_collection = []
        for i in range(dimension):
            marginal = ot.HistogramFactory().build(sample[:, i])
            distribution_collection.append(marginal)
        distribution = ot.ComposedDistribution(distribution_collection)
        return distribution


class MultivariateBasisFactory:
    def __init__(self, distribution):
        self.distribution = distribution
        input_dimension = distribution.getDimension()
        self.distribution_collection = [
            self.distribution.getMarginal(i) for i in range(input_dimension)
        ]

    def build(self):
        """
        Create a multivariate basis using a collection of distributions.
        
        A linear enumeration rule is used. 
    
        Parameters
        ----------
    
        Returns
        -------
        multivariateBasis : OrthogonalProductPolynomialFactory
            The multivariate basis.
    
        """
        multivariateBasis = ot.OrthogonalProductPolynomialFactory(
            self.distribution_collection
        )
        return multivariateBasis

    def buildAdaptive(self, quasi_norm=1.0):
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
        input_dimension = len(self.distribution_collection)
        if quasi_norm == 1.0:
            multivariateBasis = ot.OrthogonalProductPolynomialFactory(
                self.distribution_collection
            )
        else:
            polynomial_collection = ot.PolynomialFamilyCollection(input_dimension)
            for i in range(input_dimension):
                polynomial_collection[i] = ot.StandardDistributionPolynomialFactory(
                    self.distribution_collection[i]
                )
            enumerate_function = ot.HyperbolicAnisotropicEnumerateFunction(
                input_dimension, quasi_norm
            )
            multivariateBasis = ot.OrthogonalProductPolynomialFactory(
                polynomial_collection, enumerate_function
            )
        return multivariateBasis


class PolynomialChaosFactory:
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
        self.maximum_strata_index = 100
        return

    def _compute_strata_from_degree(self,):
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
    
        Returns
        -------
        degree_strata_index : int
            The index of the strata containing a multiindex having given total 
            degree.
            It is lower of equal to maximum_strata_index.
            It may be equal to maximum_strata_index if no multiindex 
            has reached the given total degree.
    
        """
        enumerateFunction = self.multivariateBasis.getEnumerateFunction()
        is_degree_reached = False
        degree_strata_index = 0
        for strata_index in range(1 + self.maximum_strata_index):
            if is_degree_reached:
                break
            strata_cardinal = enumerateFunction.getStrataCardinal(strata_index)
            cumulated_cardinal = enumerateFunction.getStrataCumulatedCardinal(
                strata_index
            )
            number_of_indices_in_strata = cumulated_cardinal - strata_cardinal
            for i in range(number_of_indices_in_strata, cumulated_cardinal):
                multiindex = enumerateFunction(i)
                multiindex_degree = sum(multiindex)
                if multiindex_degree >= self.total_degree:
                    is_degree_reached = True
                    break
                else:
                    degree_strata_index = strata_index
        return degree_strata_index

    def setMaximumStrataIndex(self, maximum_strata_index):
        """
        Set the maximum index of a strata when using an adaptive basis.
        
        This method is usually used when using an hyperbolic enumeration 
        rule.
        
        The parameter is used when computing the number of terms in the 
        basis. 
        It is the maximum number of strata which are explored when searching
        for the strata which achieves the required total degree, if any.
        See _compute_strata_from_degree for further details.

        Parameters
        ----------
        maximum_strata_index : int, greater than 1
            The maximum index of a strata.

        Returns
        -------
        None.

        """
        self.maximum_strata_index = maximum_strata_index
        return None

    def _createAdaptiveStrategy(self):
        enumerateFunction = self.multivariateBasis.getEnumerateFunction()
        adaptive_basis_name = enumerateFunction.getClassName()
        if adaptive_basis_name == "HyperbolicAnisotropicEnumerateFunction":
            strata_index = self._compute_strata_from_degree()
            number_of_terms_in_basis = enumerateFunction.getStrataCumulatedCardinal(
                strata_index
            )
        else:
            number_of_terms_in_basis = enumerateFunction.getStrataCumulatedCardinal(
                self.totalDegree
            )
        adaptiveStrategy = ot.FixedStrategy(
            self.multivariateBasis, number_of_terms_in_basis
        )
        return adaptiveStrategy

    def buildFromRegression(self, inputTrain, outputTrain, use_model_selection=True):
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

    def buildFullChaosFromIntegration(self, g_function, experiment=None):
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
        # 2. Create the experiment
        if experiment is None:
            distribution_measure = self.multivariateBasis.getMeasure()
            dim_input = g_function.getInputDimension()
            totalDegreeList = [self.totalDegree] * dim_input
            experiment = ot.GaussProductExperiment(
                distribution_measure, totalDegreeList
            )
        # 3. Create the projection strategy
        projectionStrategy = ot.IntegrationStrategy(experiment)
        # 4. Create the polynomial chaos
        chaosalgo = ot.FunctionalChaosAlgorithm(
            g_function, self.distribution, adaptiveStrategy, projectionStrategy
        )
        return chaosalgo
