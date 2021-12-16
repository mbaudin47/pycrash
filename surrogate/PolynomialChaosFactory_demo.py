#!/usr/bin/env python
# coding: utf-8

"""
# Polynômes de chaos : application au cas de la poutre encastrée

Résumé

Dans ce notebook, nous présentons la décomposition en chaos polynomial 
du cas de la poutre encastrée. Nous montrons comment créer un polynôme 
du chaos creux de manière simplifiée.

Ce notebook est adapté de :

* https://github.com/mbaudin47/otsupgalilee-eleve/blob/master/5-Chaos/Exercice-chaos-cantilever-beam.ipynb

# Model definition

Output
------
Case #1: sparse chaos from regression
    Q2 = 100.00%
Size of basis = 126, Nb. coeff = 67, sparse rate = 0.47
Case #1-bis: sparse chaos from regression, with hyperbolic rule
    Q2 = 100.00%
Size of basis = 27, Nb. coeff = 67, sparse rate = -1.48
Case #2: full chaos from regression
    Q2 = 99.92%
Size of basis = 27, Nb. coeff = 27, sparse rate = 0.00
Case #3: chaos from integration
    Q2 = 99.97%
Size of basis = 27, Nb. coeff = 27, sparse rate = 0.00
Case #4: data-given polynomial chaos
Build distribution from : BuildBasisFromDistributionCollection
quasi-norm =  0.5
    Q2 = 99.96%
Build distribution from : BuildBasisFromData
quasi-norm =  0.5
    Q2 = 99.93%
Build distribution from : BuildBasisFromHistogram
quasi-norm =  0.5
    Q2 = 99.97%
Build distribution from : BuildBasisFromKernelSmoothing
quasi-norm =  0.5
    Q2 = 97.71%
Build distribution from : BuildBasisFromDistributionCollection
quasi-norm =  1.0
    Q2 = 100.00%
Build distribution from : BuildBasisFromData
quasi-norm =  1.0
    Q2 = 99.96%
Build distribution from : BuildBasisFromHistogram
quasi-norm =  1.0
    Q2 = 100.00%
Build distribution from : BuildBasisFromKernelSmoothing
quasi-norm =  1.0
    Q2 = 97.54%
"""

import openturns as ot
import PolynomialChaosFactory as pcf
import openturns.viewer as otv


def printSparsityRate(multivariateBasis, totalDegree, chaosResult):
    """Compute the sparsity rate, assuming a FixedStrategy."""
    # TODO : this does not work with Hyperbolic rule: why?
    # Get P, the maximum possible number of coefficients
    enumfunc = multivariateBasis.getEnumerateFunction()
    P = enumfunc.getStrataCumulatedCardinal(totalDegree)
    # Get number of coefficients in the selection
    indices = chaosResult.getIndices()
    nbcoeffs = indices.getSize()
    # Compute rate
    sparsityRate = 1.0 - nbcoeffs / P
    print(
        "Size of basis = %d, Nb. coeff = %d, sparse rate = %.2f"
        % (P, nbcoeffs, sparsityRate)
    )
    return sparsityRate


# Validate the metamodel
def validate_polynomial_chaos(myDistribution, g, result, test_sample_size = 1000):
    metamodel = result.getMetaModel()
    inputTest = myDistribution.getSample(test_sample_size)
    outputTest = g(inputTest)
    val = ot.MetaModelValidation(inputTest, outputTest, metamodel)
    Q2 = val.computePredictivityFactor()[0]
    graph = val.drawValidation()
    print("    Q2 = %.2f%%" % (Q2 * 100))
    graph.setTitle("Q2=%.2f%%" % (Q2 * 100))
    view = otv.View(graph, figure_kw={"figsize": (4.0, 3.0)})
    return view


ot.RandomGenerator.SetSeed(1976)

dist_E = ot.Beta(0.9, 2.2, 2.8e7, 4.8e7)
dist_E.setDescription(["E"])
F_para = ot.LogNormalMuSigma(3.0e4, 9.0e3, 15.0e3)  # in N
dist_F = ot.ParametrizedDistribution(F_para)
dist_F.setDescription(["F"])
dist_L = ot.Uniform(250.0, 260.0)  # in cm
dist_L.setDescription(["L"])
dist_I = ot.Beta(2.5, 1.5, 310.0, 450.0)  # in cm^4
dist_I.setDescription(["I"])

myDistribution = ot.ComposedDistribution([dist_E, dist_F, dist_L, dist_I])

input_dimension = 4  # dimension of the input
output_dimension = 1  # dimension of the output


def function_beam(X):
    E, F, L, I = X
    Y = F * (L ** 3) / (3 * E * I)
    return [Y]


g = ot.PythonFunction(input_dimension, output_dimension, function_beam)
g.setInputDescription(myDistribution.getDescription())

# On crée la base polynomiale multivariée par tensorisation de polynômes univariés avec
# la règle d'énumération linéaire par défaut.
multivariateBasis = ot.OrthogonalProductPolynomialFactory(
    [dist_E, dist_F, dist_L, dist_I]
)

# Generate an training sample of size N with MC simulation (or retrieve the
# design from experimental data).

training_sample_size = 200  # Size of the training design of experiments

inputTrain = myDistribution.getSample(training_sample_size)
outputTrain = g(inputTrain)

totalDegree = 5  # Maximum total polynomial degree

print("Case #1: sparse chaos from regression")
factory = pcf.PolynomialChaosFactory(totalDegree, multivariateBasis, myDistribution)
chaosalgo = factory.buildFromRegression(inputTrain, outputTrain)
chaosalgo.run()
result = chaosalgo.getResult()
validate_polynomial_chaos(myDistribution, g, result)
printSparsityRate(multivariateBasis, totalDegree, result)

print("Case #1-bis: sparse chaos from regression, with hyperbolic rule")
distribution_collection = [dist_E, dist_F, dist_L, dist_I]
quasi_norm = 0.5
multivariateBasis = pcf.BuildBasis(distribution_collection, quasi_norm)
chaosalgo = factory.buildFromRegression(inputTrain, outputTrain)
factory = pcf.PolynomialChaosFactory(totalDegree, multivariateBasis, myDistribution)
chaosalgo.run()
result = chaosalgo.getResult()
validate_polynomial_chaos(myDistribution, g, result)
printSparsityRate(multivariateBasis, totalDegree, result)

print("Case #2: full chaos from regression")
factory = pcf.PolynomialChaosFactory(totalDegree, multivariateBasis, myDistribution)
chaosalgo = factory.buildFromRegression(inputTrain, outputTrain, False)
chaosalgo.run()
result = chaosalgo.getResult()
validate_polynomial_chaos(myDistribution, g, result)
printSparsityRate(multivariateBasis, totalDegree, result)

# Create a sparse polynomial chaos decomposition from integration
print("Case #3: chaos from integration")
factory = pcf.PolynomialChaosFactory(totalDegree, multivariateBasis, myDistribution)
chaosalgo = factory.buildFullChaosFromIntegration(g)
chaosalgo.run()
result = chaosalgo.getResult()
validate_polynomial_chaos(myDistribution, g, result)
printSparsityRate(multivariateBasis, totalDegree, result)

print("Case #4: data-given polynomial chaos")
for quasi_norm in [0.5, 1.0]:
    for select_given_data_method in ["BuildBasisFromDistributionCollection", 
                                     "BuildBasisFromData", 
                                     "BuildBasisFromHistogram",
                                     "BuildBasisFromKernelSmoothing"]:
        if select_given_data_method == "BuildBasisFromDistributionCollection":
            # Given data 1 : BuildBasisFromDistributionCollection (best case scenario)
            distribution_collection = [dist_E, dist_F, dist_L, dist_I]
            myDistribution, multivariateBasis = pcf.PolynomialChaosFactory.BuildBasisFromDistributionCollection(distribution_collection, quasi_norm)
        elif select_given_data_method == "BuildBasisFromData":
            # Given data 2 : FromData
            myDistribution, multivariateBasis = pcf.PolynomialChaosFactory.BuildBasisFromData(inputTrain, quasi_norm)
        elif select_given_data_method == "BuildBasisFromHistogram":
            # Given data 3 : use Histograms, assuming independent marginals
            myDistribution, multivariateBasis = pcf.PolynomialChaosFactory.BuildBasisFromHistogram(inputTrain, quasi_norm)
        elif select_given_data_method == "BuildBasisFromKernelSmoothing":
            # Given data 4 : Kernel Smoothing
            myDistribution, multivariateBasis = pcf.PolynomialChaosFactory.BuildBasisFromKernelSmoothing(inputTrain, quasi_norm)
        else:
            raise ValueError("Unknown given data method %s" % (select_given_data_method))
        print("Build distribution from :", select_given_data_method)
        print("quasi-norm = ", quasi_norm)
        factory = pcf.PolynomialChaosFactory(totalDegree, multivariateBasis, myDistribution)
        chaosalgo = factory.buildFromRegression(inputTrain, outputTrain)
        chaosalgo.run()
        result = chaosalgo.getResult()
        validate_polynomial_chaos(myDistribution, g, result)
