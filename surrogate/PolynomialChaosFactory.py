#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a polynomial chaos. 
"""

import openturns as ot


class PolynomialChaosFactory:
    def __init__(self, totalDegree, multivariateBasis, distribution):
        """
        Create a polynomial chaos.

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

    def buildFromRegression(self, inputTrain, outputTrain, is_sparse=True):
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
    
        Returns
        -------
        result : ot.FunctionalChaosResult
            The polynomial chaos result
        """
        if is_sparse:
            # LARS model selection
            selectionAlgorithm = ot.LeastSquaresMetaModelSelectionFactory()
        else:
            # No model selection
            selectionAlgorithm = ot.PenalizedLeastSquaresAlgorithmFactory()
        projectionStrategy = ot.LeastSquaresStrategy(
            inputTrain, outputTrain, selectionAlgorithm
        )
        enumfunc = self.multivariateBasis.getEnumerateFunction()
        P = enumfunc.getStrataCumulatedCardinal(self.totalDegree)
        adaptiveStrategy = ot.FixedStrategy(self.multivariateBasis, P)
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

        enumfunc = self.multivariateBasis.getEnumerateFunction()
        P = enumfunc.getStrataCumulatedCardinal(self.totalDegree)
        adaptiveStrategy = ot.FixedStrategy(self.multivariateBasis, P)
        distribution_measure = self.multivariateBasis.getMeasure()
        dim_input = g_function.getInputDimension()
        totalDegreeList = [self.totalDegree] * dim_input
        experiment = ot.GaussProductExperiment(distribution_measure, totalDegreeList)
        projectionStrategy = ot.IntegrationStrategy(experiment)
        chaosalgo = ot.FunctionalChaosAlgorithm(
            g_function, self.distribution, adaptiveStrategy, projectionStrategy
        )
        return chaosalgo
