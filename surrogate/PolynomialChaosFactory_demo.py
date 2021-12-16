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
<PolynomialChaosFactory.PolynomialChaosFactory object at 0x7f4d418beb80>
Size of basis = 126, Nb. coeff = 80, sparse rate = 0.37
Case #2: full chaos from regression
<PolynomialChaosFactory.PolynomialChaosFactory object at 0x7f4d418be550>
Size of basis = 126, Nb. coeff = 126, sparse rate = 0.00
Case #3: chaos from integration
Size of basis = 35, Nb. coeff = 35, sparse rate = 0.00
"""

import openturns as ot
import PolynomialChaosFactory as pcf
import openturns.viewer as otv


def printSparsityRate(multivariateBasis, totalDegree, chaosResult):
    """Compute the sparsity rate, assuming a FixedStrategy."""
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
def validate_polynomial_chaos(test_sample_size, myDistribution, g, result):
    metamodel = result.getMetaModel()
    inputTest = myDistribution.getSample(test_sample_size)
    outputTest = g(inputTest)
    val = ot.MetaModelValidation(inputTest, outputTest, metamodel)
    Q2 = val.computePredictivityFactor()[0]
    graph = val.drawValidation()
    graph.setTitle("Q2=%.2f%%" % (Q2 * 100))
    view = otv.View(graph, figure_kw={"figsize": (4.0, 3.0)})
    return view


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

dim_input = 4  # dimension of the input
dim_output = 1  # dimension of the output


def function_beam(X):
    E, F, L, I = X
    Y = F * (L ** 3) / (3 * E * I)
    return [Y]


g = ot.PythonFunction(dim_input, dim_output, function_beam)
g.setInputDescription(myDistribution.getDescription())

# Create a sparse polynomial chaos decomposition from least squares

# Pour simplifier l'utilisation du polynôme du chaos, nous avons créé la fonction
# suivante qui permet de créer un polynôme du chaos creux par moindres carrés linéaires
# avec un nombre réduit de paramètres.

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
test_sample_size = 1000
validate_polynomial_chaos(test_sample_size, myDistribution, g, result)

printSparsityRate(multivariateBasis, totalDegree, result)

print("Case #2: full chaos from regression")
factory = pcf.PolynomialChaosFactory(totalDegree, multivariateBasis, myDistribution)
chaosalgo = factory.buildFromRegression(inputTrain, outputTrain, False)
chaosalgo.run()
result = chaosalgo.getResult()
test_sample_size = 1000
validate_polynomial_chaos(test_sample_size, myDistribution, g, result)

printSparsityRate(multivariateBasis, totalDegree, result)

# Create a sparse polynomial chaos decomposition from integration
print("Case #3: chaos from integration")

totalDegree = 3  # Total polynomial degree
factory = pcf.PolynomialChaosFactory(totalDegree, multivariateBasis, myDistribution)
chaosalgo = factory.buildFullChaosFromIntegration(g)
chaosalgo.run()
result = chaosalgo.getResult()

# La méthode `getMetaModel` retourne une fonction permettant d'évaluer le métamodèle.
metamodel = result.getMetaModel()
x = myDistribution.getMean()
y = metamodel(x)

test_sample_size = 1000
validate_polynomial_chaos(test_sample_size, myDistribution, g, result)

printSparsityRate(multivariateBasis, totalDegree, result)
