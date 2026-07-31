#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Another implementation of the polynomial chaos by least squares.

Authors : Sofiane Haddad, Régis Lebrun (Airbus)
With small additions from Michaël Baudin:
    - removed unused optional arguments
    - use the total degree instead of the number of terms in basis
    
TODO: fill the metamodel field of the FunctionalChaosResult    
TODO: Show how to optionally use LARS
TODO: Show how to optionally use a non-polynomial basis
TODO: Show how it could be extended to ridge regression
"""

# %%
import openturns as ot
from openturns.usecases import ishigami_function
from math import pi
from time import time


# %%
class LeastSquaresFCE:
    def __init__(
        self,
        input_sample,
        output_sample,
        distribution,
        basis,
        basis_dimension,
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
        basis_dimension : int
            The number of coefficients.
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
        self.basis_dimension = basis_dimension
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
        transformation = ot.DistributionTransformation(self.distribution, self.basis.getMeasure())
        standard_input = transformation(self.input_sample)
        indices = ot.Indices(self.basis_dimension)
        indices.fill()
        functions = [self.basis.build(i) for i in indices]
        designProxy = ot.DesignProxy(standard_input, functions)
        if self.wX is None:
            leastSquaresMethod = ot.LeastSquaresMethod.Build(
                self.leastSquaresMethod, designProxy, indices
            )
        else:
            leastSquaresMethod = ot.LeastSquaresMethod.Build(
                self.leastSquaresMethod, designProxy, self.wX, indices
            )
        outputDimension = self.output_sample.getDimension()
        coefficients = ot.Sample(self.basis_dimension, outputDimension)
        for j in range(outputDimension):
            coeffsJ = leastSquaresMethod.solve(output_sample.getMarginal(j).asPoint())
            for i in range(self.basis_dimension):
                coefficients[i, j] = coeffsJ[i]
        # Create the result
        self.result = ot.FunctionalChaosResult(
            input_sample,
            output_sample,
            self.distribution,
            transformation,
            transformation.inverse(),
            self.basis,
            indices,
            coefficients,
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

# %%
# Create the Ishigami model
im = ishigami_function.IshigamiModel()
dimension = im.inputDistribution.getDimension()
model = im.model
distribution = im.inputDistribution

# %%
# Create the basis
enumerateFunction = ot.LinearEnumerateFunction(dimension)
basis = ot.OrthogonalProductPolynomialFactory(
    [ot.LegendreFactory()] * dimension, enumerateFunction
)

# %%
# Create the input/output database
size = 10000
input_sample = distribution.getSample(size)
output_sample = model(input_sample)

# %%
totalDegree = 7
enumerateFunction = basis.getEnumerateFunction()
basis_dimension = enumerateFunction.getBasisSizeFromTotalDegree(totalDegree)
print(f"Number of coefficients = {basis_dimension}")


# %%
algo = LeastSquaresFCE(
    input_sample, output_sample, distribution, basis, basis_dimension
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

# %%
result

# %%
input_test = im.inputDistribution.getSample(1000)
output_test = im.model(input_test)
meta_model = result.getMetaModel()
validation = ot.MetaModelValidation(output_test, meta_model(input_test))
print(f"Q2 = {validation.computeR2Score()[0]:.15f}")

# %%
