# -*- coding: utf-8 -*-
"""
Cas Cogénération
Fonction G : Analytique
PDF des 3 variables d'entrée X
Matrice de scatter plot de l'échantillon de l'entrée X
Tendance centrale : moyenne, écart-type
Algorithme : MonteCarlo

Objectif : paramétrer la précision d'estimation de la moyenne 
en fonction d'un coefficient de variation.

Reference
http://www.epixanalytics.com/modelassist/CrystalBall/Model_Assist.htm#Montecarlo/Precision_control_feature.htm
"""
import openturns as ot
from centraldispersion import (
centralDispersionByMonteCarlo, 
centralDispersionPrintResults, 
centralDispersionPrintParameters
)

# 1. The function G
inputs=["Q","E","C"]
formula=["1-Q/(E/(1-0.05)/0.54+C/0.8)"]
g = ot.SymbolicFunction(inputs,formula)
g = ot.MemoizeFunction(g)

# 2. Random vector definition
Q = ot.Normal(10200,100)
E = ot.Normal(3000,15)
C = ot.Normal(4000,60)
Q.setDescription(["Energie primaire"])
E.setDescription(["Energie electrique"])
C.setDescription(["Energie thermique"])

# 4. Create the joint distribution function, 
#    the output and the event. 
inputDistribution = ot.ComposedDistribution([Q, E, C])

# 5. Create the input random vector
inputRandomVector = ot.RandomVector(inputDistribution)

outputRandomVector = ot.RandomVector(g, inputRandomVector)

# User parameters
blocksize=1000
maxiter=100
maxcov=0.001 # Criteria B
maxcalls=maxiter * blocksize # Criteria A
maxelapsetime=5 # Criteria C
alpha=0.05 # Niveau de confiance de l'intervalle

# 6. Estimate expectation with algorithm
print("\ncentralDispersionByMonteCarlo")
centralDispersionPrintParameters(blocksize, maxcov, maxcalls,maxelapsetime,alpha)

outputSample, criteria=centralDispersionByMonteCarlo(outputRandomVector, blocksize,maxcov,maxcalls,maxelapsetime)
if (criteria==1):
    print("Reached number of calls")
elif (criteria==2):
    print("Reached required precision")
elif (criteria==3):
    print("Reached maximum elapsed time")
    
centralDispersionPrintResults(outputSample,alpha)

# 7. Estimate expectation with ExpectationSimulationAlgorithm
print("\nExpectationSimulationAlgorithm")
g.clearHistory()
ot.Log.Show(ot.Log.DBG)
algo = ot.ExpectationSimulationAlgorithm(outputRandomVector)
algo.setMaximumOuterSampling(maxiter)
algo.setBlockSize(blocksize)
algo.setMaximumCoefficientOfVariation(maxcov)
algo.run()
result = algo.getResult()
expectation = result.getExpectationEstimate()
print("Mean by ESA = %f " % expectation[0])
print("CV(Mean) = %.2f %% " % (100*result.getCoefficientOfVariation()[0]))
print("Number of function calls = %d" % (g.getInputHistory().getSize()))

