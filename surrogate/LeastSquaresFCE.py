#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Another implementation of the polynomial chaos by regression.

Authors : Sofiane Haddad, Régis Lebrun (Airbus)
With small additions from Michaël Baudin:
    - removed unused optional arguments
    - use the total degree instead of the number of terms in basis
    
TODO: fill the metamodel field of the FunctionalChaosResult    
TODO: Show how to optionally use LARS
TODO: Show how to optionally use a non-polynomial basis
TODO: Show how it could be extended to ridge regression
"""


import openturns as ot


class LeastSquaresFCE:
    def __init__(
        self,
        input_sample,
        output_sample,
        distribution,
        basis,
        totalDegree,
        wX=None,
        leastSquaresMethod="SVD",
    ):
        """
        Create a polynomial chaos by least squares.
        
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
        leastSquaresMethod : ot.LeastSquaresMethod()
            The resolution method.

        Returns
        -------
        None.

        """
        self.input_sample = input_sample
        self.wX = wX
        self.output_sample = output_sample
        self.distribution = distribution
        self.basis = basis
        self.totalDegree = totalDegree
        self.leastSquaresMethod = leastSquaresMethod
        self.result = None

    def run(self):
        """
        Create the functional chaos metamodel.
        
        The algorithm estimates the coefficients of the functional chaos 
        expansion using least squares. 
        This involves using a least squares method. 

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
        if self.wX is None:
            leastSquaresMethod = ot.LeastSquaresMethod.Build(
                self.leastSquaresMethod, designProxy, indices
            )
        else:
            leastSquaresMethod = ot.LeastSquaresMethod.Build(
                self.leastSquaresMethod, designProxy, self.wX, indices
            )
        outputDimension = self.output_sample.getDimension()
        coefficients = ot.Sample(maximumBasisSize, outputDimension)
        for j in range(outputDimension):
            coeffsJ = leastSquaresMethod.solve(output_sample.getMarginal(j).asPoint())
            for i in range(maximumBasisSize):
                coefficients[i, j] = coeffsJ[i]
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
    algo = LeastSquaresFCE(
        input_sample, output_sample, distribution, basis, totalDegree
    )

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
        ot.LeastSquaresStrategy(),
    )
    t0 = time()
    algo.run()
    result = algo.getResult()
    t1 = time()
    # print("result=", result)
    print("t=", t1 - t0, "s")
