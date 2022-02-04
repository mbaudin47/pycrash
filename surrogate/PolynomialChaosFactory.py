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

        TODO: integrate that into our favorite library.

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
        return

    def _createAdaptiveStrategy(self):
        # Create a FixedStrategy from the total degree
        enumerateFunction = self.multivariateBasis.getEnumerateFunction()
        strata_index = enumerateFunction.getMaximumDegreeStrataIndex(self.totalDegree)
        number_of_terms_in_basis = enumerateFunction.getStrataCumulatedCardinal(
            strata_index
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

    def buildFullChaosFromIntegration(
        self, g_function, experiment, adaptiveStrategy=None
    ):
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
        if adaptiveStrategy is None:
            adaptiveStrategy = self._createAdaptiveStrategy()
        projectionStrategy = ot.IntegrationStrategy(experiment)
        chaosalgo = ot.FunctionalChaosAlgorithm(
            g_function, self.distribution, adaptiveStrategy, projectionStrategy
        )
        return chaosalgo
