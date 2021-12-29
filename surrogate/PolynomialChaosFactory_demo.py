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
    Size of basis = 1001, Nb. coeff = 68, sparse rate = 0.93
Case #1-bis: sparse chaos from regression, with hyperbolic rule
    Q2 = 99.99%
    Size of basis = 69, Nb. coeff = 26, sparse rate = 0.62
Case #2: full chaos from regression
    Q2 = -291.92%
    Size of basis = 1001, Nb. coeff = 1001, sparse rate = 0.00
Case #3: chaos from integration
    Q2 = 100.00%
    Size of basis = 1001, Nb. coeff = 1001, sparse rate = 0.00
Case #4: data-given polynomial chaos (regression)
Build distribution from : ComposedDistribution  quasi-norm =  0.5
    Q2 = 99.98%
    Size of basis = 69, Nb. coeff = 26, sparse rate = 0.62
Build distribution from : BuildDistribution  quasi-norm =  0.5
    Q2 = 99.98%
    Size of basis = 69, Nb. coeff = 33, sparse rate = 0.52
Build distribution from : Histogram  quasi-norm =  0.5
    Q2 = 99.99%
    Size of basis = 69, Nb. coeff = 26, sparse rate = 0.62
Build distribution from : KernelSmoothing  quasi-norm =  0.5
    Q2 = 99.99%
    Size of basis = 69, Nb. coeff = 26, sparse rate = 0.62
Build distribution from : ComposedDistribution  quasi-norm =  1.0
    Q2 = 100.00%
    Size of basis = 1001, Nb. coeff = 68, sparse rate = 0.93
Build distribution from : BuildDistribution  quasi-norm =  1.0
    Q2 = 99.97%
    Size of basis = 1001, Nb. coeff = 35, sparse rate = 0.97
Build distribution from : Histogram  quasi-norm =  1.0
    Q2 = 100.00%
    Size of basis = 1001, Nb. coeff = 54, sparse rate = 0.95
Build distribution from : KernelSmoothing  quasi-norm =  1.0
    Q2 = 100.00%
    Size of basis = 1001, Nb. coeff = 54, sparse rate = 0.95

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
training_sample_size = 200  # Size of the training design of experiments
inputTrain = myDistribution.getSample(training_sample_size)
outputTrain = g_function(inputTrain)

totalDegree = 10  # Maximum total polynomial degree

print("Case #1: sparse chaos from regression")
basis_factory = pcf.MultivariateBasisFactory(myDistribution)
multivariateBasis = basis_factory.build()
factory = pcf.PolynomialChaosFactory(totalDegree, multivariateBasis, myDistribution)
chaosalgo = factory.buildFromRegression(inputTrain, outputTrain)
chaosalgo.run()
result = chaosalgo.getResult()
validate_polynomial_chaos(myDistribution, g_function, result)
printSparsityRate(multivariateBasis, totalDegree, result)

print("Case #1-bis: sparse chaos from regression, with hyperbolic rule")
quasi_norm = 0.5
basis_factory = pcf.MultivariateBasisFactory(myDistribution)
multivariateBasis = basis_factory.buildAdaptive(quasi_norm)
factory = pcf.PolynomialChaosFactory(totalDegree, multivariateBasis, myDistribution)
chaosalgo = factory.buildFromRegression(inputTrain, outputTrain)
chaosalgo.run()
result = chaosalgo.getResult()
validate_polynomial_chaos(myDistribution, g_function, result)
printSparsityRate(multivariateBasis, totalDegree, result)

print("Case #2: full chaos from regression")
basis_factory = pcf.MultivariateBasisFactory(myDistribution)
multivariateBasis = basis_factory.build()
factory = pcf.PolynomialChaosFactory(totalDegree, multivariateBasis, myDistribution)
chaosalgo = factory.buildFromRegression(inputTrain, outputTrain, False)
chaosalgo.run()
result = chaosalgo.getResult()
validate_polynomial_chaos(myDistribution, g_function, result)
printSparsityRate(multivariateBasis, totalDegree, result)

# Create a polynomial chaos decomposition from integration with Gauss rule
print("Case #3: chaos from integration (Gauss rule)")
basis_factory = pcf.MultivariateBasisFactory(myDistribution)
multivariateBasis = basis_factory.build()
factory = pcf.PolynomialChaosFactory(totalDegree, multivariateBasis, myDistribution)
chaosalgo = factory.buildFullChaosFromIntegration(g_function)
chaosalgo.run()
result = chaosalgo.getResult()
validate_polynomial_chaos(myDistribution, g_function, result)
printSparsityRate(multivariateBasis, totalDegree, result)

# Create a polynomial chaos decomposition from integration with sampling
print("Case #3-bis: chaos from integration (sampling)")
basis_factory = pcf.MultivariateBasisFactory(myDistribution)
multivariateBasis = basis_factory.build()
sequence = ot.SobolSequence()
dimension = myDistribution.getDimension()
factory = pcf.PolynomialChaosFactory(totalDegree, multivariateBasis, myDistribution)
for experiment in [
    ot.MonteCarloExperiment(myDistribution, training_sample_size),
    ot.LHSExperiment(myDistribution, training_sample_size),
    ot.LowDiscrepancyExperiment(sequence, myDistribution, dimension),
]:
    name = experiment.getClassName()
    print("Experiment : ", name)
    chaosalgo = factory.buildFullChaosFromIntegration(g_function, experiment)
    chaosalgo.run()
    result = chaosalgo.getResult()
    validate_polynomial_chaos(myDistribution, g_function, result)
    printSparsityRate(multivariateBasis, totalDegree, result)

print("Case #4: data-given polynomial chaos (regression)")
for quasi_norm in [0.5, 1.0]:
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
            raise ValueError(
                "Unknown given data method %s" % (select_given_data_method)
            )
        print(
            "Build distribution from :",
            select_given_data_method,
            " quasi-norm = ",
            quasi_norm,
        )
        basis_factory = pcf.MultivariateBasisFactory(myDistribution)
        multivariateBasis = basis_factory.buildAdaptive(quasi_norm)
        factory = pcf.PolynomialChaosFactory(
            totalDegree, multivariateBasis, myDistribution
        )
        chaosalgo = factory.buildFromRegression(inputTrain, outputTrain)
        chaosalgo.run()
        result = chaosalgo.getResult()
        validate_polynomial_chaos(myDistribution, g_function, result)
        printSparsityRate(multivariateBasis, totalDegree, result)
