#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Another implementation of the polynomial chaos by integration.

Authors : Sofiane Haddad, Régis Lebrun (Airbus)
With small additions from Michaël Baudin:
    - removed unused optional arguments
    - use the total degree instead of the number of terms in basis

TODO: fill the metamodel field of the FunctionalChaosResult    
TODO: show how to optionnaly use GaussProductQuadrature.
TODO: show how to optionnaly use CleaningStrategy.
TODO: Show how to optionally use a non-polynomial basis
TODO: Show how it could be extended to another selection method, such as 
using the part of variance of each coefficient.
"""


import openturns as ot

# Here we use some numpy tricks to be implemented in OT
import numpy as np


class IntegrationFCE:
    def __init__(
        self, input_sample, output_sample, distribution, basis, totalDegree, wX=None,
    ):
        """
        Create a polynomial chaos by integration.
        
        This reduces to making the weighted dot product of the 
        weighted vector w to the design matrix.

        Parameters
        ----------
        input_sample : ot.Sample(size, input_dimension)
            The input sample.
        output_sample : ot.Sample(size, output_dimension)
            The output sample.
        distribution : ot.Distribution(input_dimension)
            The distributino of the input.
        basis : ot.OrthogonalBasis()
            The orthogonal basis of functions.
        totalDegree : int
            The maximum total degree.
        wX : ot.Point(size), optional
            The quadrature weights. The default is None.

        Returns
        -------
        None.

        """
        self.input_sample = input_sample
        if wX is None:
            self.wX = ot.Point(input_sample.getSize(), 1.0 / input_sample.getSize())
        else:
            self.wX = wX
        self.output_sample = output_sample
        self.distribution = distribution
        self.basis = basis
        self.totalDegree = totalDegree
        self.result = None

    def run(self):
        """
        Create the functional chaos metamodel.
        
        The algorithm estimates the coefficients of the functional chaos 
        expansion using integration. 
        This involves a component-by-component product of the outputs 
        with the weights, then a row vector-by-matrix product. 

        Returns
        -------
        None.

        """
        enumerateFunction = self.basis.getEnumerateFunction()
        strata_index = enumerateFunction.getMaximumDegreeStrataIndex(self.totalDegree)
        maximumBasisSize = enumerateFunction.getStrataCumulatedCardinal(strata_index)
        transformation = ot.DistributionTransformation(distribution, basis.getMeasure())
        XTransformed = transformation(self.input_sample)
        indices = ot.Indices(maximumBasisSize)
        indices.fill()
        functions = [basis.build(i) for i in indices]
        designProxy = ot.DesignProxy(XTransformed, functions)
        designMatrix = np.array(ot.Matrix(designProxy.computeDesign(indices)))
        weightedOutput = np.multiply(np.array(self.output_sample).T, np.array(self.wX))
        coefficients = np.dot(weightedOutput, designMatrix).T
        self.result = ot.FunctionalChaosResult(
            ot.Function(),
            self.distribution,
            transformation,
            transformation.inverse(),
            ot.Function(),
            self.basis,
            indices,
            coefficients,
            functions,
            [-1.0],
            [-1.0],
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


if __name__ == "__main__":
    from math import pi
    from time import time

    a = 7.0
    b = 0.1
    inputVariables = ["xi1", "xi2", "xi3"]
    formula = [
        "sin(xi1) + ("
        + str(a)
        + ") * (sin(xi2)) ^ 2 + ("
        + str(b)
        + ") * xi3^4 * sin(xi1)"
    ]
    model = ot.SymbolicFunction(inputVariables, formula)

    # Create the input distribution
    dimension = len(inputVariables)
    distribution = ot.ComposedDistribution([ot.Uniform(-pi, pi)] * dimension)

    # Create the orthogonal basis
    enumerateFunction = ot.LinearEnumerateFunction(dimension)
    basis = ot.OrthogonalProductPolynomialFactory(
        [ot.LegendreFactory()] * dimension, enumerateFunction
    )

    # Create the input/output database
    size = 10000
    input_sample = distribution.getSample(size)
    output_sample = model(input_sample)
    totalDegree = 5
    algo = IntegrationFCE(input_sample, output_sample, distribution, basis, totalDegree)

    t0 = time()
    result = algo.getResult()
    t1 = time()
    # print("result=", result)
    print("t=", t1 - t0, "s")
    enumerateFunction = basis.getEnumerateFunction()
    strata_index = enumerateFunction.getMaximumDegreeStrataIndex(totalDegree)
    basisSize = enumerateFunction.getStrataCumulatedCardinal(strata_index)
    algo = ot.FunctionalChaosAlgorithm(
        input_sample,
        output_sample,
        distribution,
        ot.FixedStrategy(basis, basisSize),
        ot.IntegrationStrategy(),
    )
    t0 = time()
    algo.run()
    result = algo.getResult()
    t1 = time()
    # print("result=", result)
    print("t=", t1 - t0, "s")
