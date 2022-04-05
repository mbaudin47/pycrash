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
Sample size =  1000
Case #1: sparse chaos from regression
    Q2 = 100.00%
    Size of basis = 126, Nb. coeff = 90, sparse rate = 0.29
Case #1-bis: sparse chaos from regression, with hyperbolic rule
    Q2 = 99.99%
    Size of basis = 27, Nb. coeff = 24, sparse rate = 0.11
Case #2: full chaos from regression
    Q2 = 100.00%
    Size of basis = 126, Nb. coeff = 126, sparse rate = 0.00
Case #3: data-given polynomial chaos (regression)
Build distribution from : ComposedDistribution
    Q2 = 100.00%
    Size of basis = 126, Nb. coeff = 90, sparse rate = 0.29
Build distribution from : BuildDistribution
    Q2 = 99.99%
    Size of basis = 126, Nb. coeff = 41, sparse rate = 0.67
Build distribution from : Histogram
    Q2 = 91.02%
    Size of basis = 126, Nb. coeff = 4, sparse rate = 0.97
Build distribution from : KernelSmoothing
    Q2 = 92.69%
    Size of basis = 126, Nb. coeff = 4, sparse rate = 0.97

"""

import openturns as ot
import PolynomialChaosFactory as pcf
import openturns.viewer as otv


def printSparsityRate(multivariateBasis, totalDegree, chaosResult):
    """Compute the sparsity rate, assuming a FixedStrategy."""
    # Get P, the maximum possible number of coefficients
    enumfunc = multivariateBasis.getEnumerateFunction()
    number_of_terms_in_basis = enumfunc.getStrataCumulatedCardinal(totalDegree)
    # Get number of coefficients in the selection
    indices = chaosResult.getIndices()
    nbcoeffs = indices.getSize()
    # Compute rate
    sparsityRate = 1.0 - nbcoeffs / number_of_terms_in_basis
    print(
        "    Size of basis = %d, Nb. coeff = %d, sparse rate = %.2f"
        % (number_of_terms_in_basis, nbcoeffs, sparsityRate)
    )
    return sparsityRate


# Validate the metamodel
def validate_polynomial_chaos(
    myDistribution, g_function, result, test_sample_size=1000
):
    metamodel = result.getMetaModel()
    inputTest = myDistribution.getSample(test_sample_size)
    outputTest = g_function(inputTest)
    val = ot.MetaModelValidation(inputTest, outputTest, metamodel)
    Q2 = val.computePredictivityFactor()[0]
    graph = val.drawValidation()
    print("    Q2 = %.2f%%" % (Q2 * 100))
    graph.setTitle("Q2=%.2f%%" % (Q2 * 100))
    view = otv.View(graph, figure_kw={"figsize": (4.0, 3.0)})
    return view


ot.Log.Show(ot.Log.NONE)
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

distribution_collection = [dist_E, dist_F, dist_L, dist_I]
myDistribution = ot.ComposedDistribution(distribution_collection)

input_dimension = 4  # dimension of the input
output_dimension = 1  # dimension of the output


def function_beam(X):
    E, F, L, I = X
    Y = F * (L ** 3) / (3 * E * I)
    return [Y]


g_function = ot.PythonFunction(input_dimension, output_dimension, function_beam)
g_function.setInputDescription(myDistribution.getDescription())

# Generate an training sample of size N with MC simulation (or retrieve the
# design from experimental data).
training_sample_size = 1000  # Size of the training design of experiments
print("Sample size = ", training_sample_size)
inputTrain = myDistribution.getSample(training_sample_size)
outputTrain = g_function(inputTrain)

totalDegree = 5  # Maximum total polynomial degree

# Case 1: linear enumerate function
linear_multivariate_basis = ot.OrthogonalProductPolynomialFactory(
    distribution_collection
)

# Case 2: hyperbolic enumerate function
quasi_norm = 0.5
polynomial_collection = ot.PolynomialFamilyCollection(input_dimension)
for i in range(input_dimension):
    polynomial_collection[i] = ot.StandardDistributionPolynomialFactory(
        distribution_collection[i]
    )
hyperbolic_enumerate_function = ot.HyperbolicAnisotropicEnumerateFunction(
    input_dimension, quasi_norm
)
hyperbolic_multivariate_basis = ot.OrthogonalProductPolynomialFactory(
    polynomial_collection, hyperbolic_enumerate_function
)

print("Case #1: sparse chaos from regression")
factory = pcf.PolynomialChaosFactory(
    totalDegree, linear_multivariate_basis, myDistribution
)
chaosalgo = factory.buildFromRegression(inputTrain, outputTrain)
chaosalgo.run()
result = chaosalgo.getResult()
validate_polynomial_chaos(myDistribution, g_function, result)
printSparsityRate(linear_multivariate_basis, totalDegree, result)

print("Case #1-bis: sparse chaos from regression, with hyperbolic rule")
factory = pcf.PolynomialChaosFactory(
    totalDegree, hyperbolic_multivariate_basis, myDistribution
)
chaosalgo = factory.buildFromRegression(inputTrain, outputTrain)
chaosalgo.run()
result = chaosalgo.getResult()
validate_polynomial_chaos(myDistribution, g_function, result)
printSparsityRate(hyperbolic_multivariate_basis, totalDegree, result)

print("Case #2: full chaos from regression")
multivariateBasis = ot.OrthogonalProductPolynomialFactory(distribution_collection)
factory = pcf.PolynomialChaosFactory(
    totalDegree, linear_multivariate_basis, myDistribution
)
chaosalgo = factory.buildFromRegression(inputTrain, outputTrain, False)
chaosalgo.run()
result = chaosalgo.getResult()
validate_polynomial_chaos(myDistribution, g_function, result)
printSparsityRate(multivariateBasis, totalDegree, result)

print("Case #4: data-given polynomial chaos (regression)")
for select_given_data_method in [
    "ComposedDistribution",
    "BuildDistribution",
    "Histogram",
    "KernelSmoothing",
]:
    if select_given_data_method == "ComposedDistribution":
        # Given data 1 : BuildAdaptiveBasisFromDistributionCollection (best case scenario)
        myDistribution = ot.ComposedDistribution(distribution_collection)
    elif select_given_data_method == "BuildDistribution":
        # Given data 2 : FromData
        myDistribution = ot.FunctionalChaosAlgorithm.BuildDistribution(inputTrain)
    elif select_given_data_method == "Histogram":
        # Given data 3 : use Histograms, assuming independent marginals
        histogram_factory = pcf.SuperHistogramFactory()
        myDistribution = histogram_factory.build(inputTrain)
    elif select_given_data_method == "KernelSmoothing":
        # Given data 4 : Kernel Smoothing
        distribution = ot.KernelSmoothing().build(inputTrain)
    else:
        raise ValueError("Unknown given data method %s" % (select_given_data_method))
    print(
        "Build distribution from :", select_given_data_method,
    )
    factory = pcf.PolynomialChaosFactory(
        totalDegree, linear_multivariate_basis, myDistribution
    )
    chaosalgo = factory.buildFromRegression(inputTrain, outputTrain)
    chaosalgo.run()
    result = chaosalgo.getResult()
    validate_polynomial_chaos(myDistribution, g_function, result)
    printSparsityRate(multivariateBasis, totalDegree, result)
