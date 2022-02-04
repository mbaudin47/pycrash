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
Sample size =  4000
Case #1: full chaos from integration (Gauss rule)
    Q2 = 100.00%
    Size of basis = 126, Nb. coeff = 126, sparse rate = 0.00
Case #2: full chaos from integration (sampling)
Experiment :  MonteCarloExperiment
    Q2 = 52.86%
    Size of basis = 126, Nb. coeff = 126, sparse rate = 0.00
Experiment :  LHSExperiment
    Q2 = 68.35%
    Size of basis = 126, Nb. coeff = 126, sparse rate = 0.00
Experiment :  LowDiscrepancyExperiment
    Q2 = 58.06%
    Size of basis = 126, Nb. coeff = 126, sparse rate = 0.00
Case #3: sparse chaos from integration (Gauss rule)
    Q2 = 100.00%
    Size of basis = 126, Nb. coeff = 31, sparse rate = 0.75
Case #4: sparse chaos from integration (sampling)
Experiment :  MonteCarloExperiment
    Q2 = 83.77%
    Size of basis = 126, Nb. coeff = 5, sparse rate = 0.96
Experiment :  LHSExperiment
    Q2 = 92.47%
    Size of basis = 126, Nb. coeff = 5, sparse rate = 0.96
Experiment :  LowDiscrepancyExperiment
    Q2 = 94.66%
    Size of basis = 126, Nb. coeff = 6, sparse rate = 0.95
Case #5: find sparse hyperparameters by naive cross-validation and LDS
5 0.1
    Q2 = 94.93%
5 0.01
    Q2 = 97.05%
5 0.001
    Q2 = 96.64%
5 0.0001
    Q2 = 77.25%
10 0.1
    Q2 = 93.34%
10 0.01
    Q2 = -169.30%
10 0.001
    Q2 = 97.02%
10 0.0001
    Q2 = 90.86%
15 0.1
    Q2 = 95.98%
15 0.01
    Q2 = 87.55%
15 0.001
    Q2 = 96.89%
15 0.0001
    Q2 = 90.25%
20 0.1
    Q2 = 91.79%
20 0.01
    Q2 = 96.51%
20 0.001
    Q2 = 86.98%
20 0.0001
    Q2 = 94.87%
"""

import openturns as ot
import PolynomialChaosFactory as pcf
import openturns.viewer as otv
import itertools

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
training_sample_size = 4000  # Size of the training design of experiments
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

#
print("Case #1: full chaos from integration (Gauss rule)")
multivariateBasis = ot.OrthogonalProductPolynomialFactory(distribution_collection)
factory = pcf.PolynomialChaosFactory(
    totalDegree, linear_multivariate_basis, myDistribution
)
# The integration is in the standard space
distribution_standard = multivariateBasis.getMeasure()
dim_input = g_function.getInputDimension()
totalDegreeList = [totalDegree] * dim_input
experiment = ot.GaussProductExperiment(distribution_standard, totalDegreeList)
chaosalgo = factory.buildFullChaosFromIntegration(g_function, experiment)
chaosalgo.run()
result = chaosalgo.getResult()
validate_polynomial_chaos(myDistribution, g_function, result)
printSparsityRate(linear_multivariate_basis, totalDegree, result)


#
print("Case #2: full chaos from integration (sampling)")
multivariateBasis = ot.OrthogonalProductPolynomialFactory(distribution_collection)
sequence = ot.SobolSequence()
dimension = myDistribution.getDimension()
factory = pcf.PolynomialChaosFactory(
    totalDegree, linear_multivariate_basis, myDistribution
)
for experiment in [
    ot.MonteCarloExperiment(distribution_standard, training_sample_size),
    ot.LHSExperiment(distribution_standard, training_sample_size),
    ot.LowDiscrepancyExperiment(sequence, distribution_standard, training_sample_size),
]:
    name = experiment.getClassName()
    print("Experiment : ", name)
    chaosalgo = factory.buildFullChaosFromIntegration(g_function, experiment)
    chaosalgo.run()
    result = chaosalgo.getResult()
    validate_polynomial_chaos(myDistribution, g_function, result)
    printSparsityRate(multivariateBasis, totalDegree, result)

#
print("Case #3: sparse chaos from integration (Gauss rule)")
multivariateBasis = ot.OrthogonalProductPolynomialFactory(distribution_collection)
factory = pcf.PolynomialChaosFactory(
    totalDegree, linear_multivariate_basis, myDistribution
)
dim_input = g_function.getInputDimension()
totalDegreeList = [totalDegree] * dim_input
experiment = ot.GaussProductExperiment(distribution_standard, totalDegreeList)
maximumConsideredTerms = 100
mostSignificant = 30
significanceFactor = 1.0e-10
adaptiveStrategy = ot.CleaningStrategy(
    multivariateBasis,
    maximumConsideredTerms,
    mostSignificant,
    significanceFactor,
    True,
)
chaosalgo = factory.buildFullChaosFromIntegration(
    g_function, experiment, adaptiveStrategy
)
chaosalgo.run()
result = chaosalgo.getResult()
validate_polynomial_chaos(myDistribution, g_function, result)
printSparsityRate(linear_multivariate_basis, totalDegree, result)

#
print("Case #4: sparse chaos from integration (sampling)")
multivariateBasis = ot.OrthogonalProductPolynomialFactory(distribution_collection)
sequence = ot.SobolSequence()
dimension = myDistribution.getDimension()
maximumConsideredTerms = 150
mostSignificant = 5
significanceFactor = 1.0e-1
adaptiveStrategy = ot.CleaningStrategy(
    multivariateBasis,
    maximumConsideredTerms,
    mostSignificant,
    significanceFactor,
    True,
)
factory = pcf.PolynomialChaosFactory(
    totalDegree, linear_multivariate_basis, myDistribution
)
for experiment in [
    ot.MonteCarloExperiment(distribution_standard, training_sample_size),
    ot.LHSExperiment(distribution_standard, training_sample_size),
    ot.LowDiscrepancyExperiment(sequence, distribution_standard, training_sample_size),
]:
    name = experiment.getClassName()
    print("Experiment : ", name)
    chaosalgo = factory.buildFullChaosFromIntegration(
        g_function, experiment, adaptiveStrategy
    )
    chaosalgo.run()
    result = chaosalgo.getResult()
    validate_polynomial_chaos(myDistribution, g_function, result)
    printSparsityRate(multivariateBasis, totalDegree, result)

#
print("Case #5: find sparse hyperparameters by naive cross-validation and LDS")
experiment = ot.LowDiscrepancyExperiment(sequence, distribution_standard, training_sample_size)
maximumConsideredTerms = 150
mostSignificant_list = [5, 10, 15, 20]
significanceFactor_list = [1.0e-1, 1.e-2, 1.e-3, 1.e-4]
for mostSignificant, significanceFactor in itertools.product(mostSignificant_list, significanceFactor_list):
    adaptiveStrategy = ot.CleaningStrategy(
        multivariateBasis,
        maximumConsideredTerms,
        mostSignificant,
        significanceFactor,
        True,
    )    
    chaosalgo = factory.buildFullChaosFromIntegration(
        g_function, experiment, adaptiveStrategy
    )
    chaosalgo.run()
    result = chaosalgo.getResult()
    print(mostSignificant, significanceFactor)
    validate_polynomial_chaos(myDistribution, g_function, result)

