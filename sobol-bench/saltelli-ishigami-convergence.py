# -*- coding: utf-8 -*-
"""
Estimates the sensitivity indices of Ishigami test function.
Validation of the convergence of several estimators to the 
exact solution. 
The expected behaviour should be the one of a Monte-Carlo simulation, 
i.e. the absolute error must decrease as 1/sqrt(n).

References
Introduction to sensitivity analysis with NISP
Michael Baudin (EDF), Jean-Marc Martinez (CEA)
Version 0.5, February 2014
Section 4.5 "Summary of the results"
"""

import openturns as ot
from math import pi
import numpy as np
import pylab as pl

def ishigamisa(a,b):
    var = 1.0/2 + a**2/8 + b*pi**4/5 + b**2*pi**8/18
    S1 = (1.0/2 + b*pi**4/5+b**2*pi**8/50)/var
    S2 = (a**2/8)/var
    S3 = 0
    S13 = b**2*pi**8/2*(1.0/9-1.0/25)/var
    exact = {
            'expectation' : a/2, 
            'variance' : var,
            'S1' : (1.0/2 + b*pi**4/5+b**2*pi**8.0/50)/var,
            'S2' : (a**2/8)/var, 
            'S3' : 0,
            'S12' : 0,
            'S23' : 0,
            'S13' : S13,
            'S123' : 0,
            'ST1' : S1 + S13,
            'ST2' : S2,
            'ST3' : S3 + S13
            }
    return exact

a = 7.
b = 0.1
exact = ishigamisa(a,b)
#print(exact)
print("S=%.4f, %.4f, %.4f" % (exact['S1'],exact['S2'],exact['S3']))
print("ST=%.4f, %.4f, %.4f" % (exact['ST1'],exact['ST2'],exact['ST3']))

# Create the model and input distribution
formula = ['sin(X1)+7*sin(X2)^2+0.1*X3^4*sin(X1)']
input_names = ['X1', 'X2', 'X3']
model = ot.SymbolicFunction(input_names, formula)
dist_X = ot.Uniform(-pi, pi)
distribution = ot.ComposedDistribution([dist_X]*3)
dimension = distribution.getDimension()
# Create X/Y data
ot.RandomGenerator.SetSeed(0)
size = 10000
inputDesign = ot.SobolIndicesExperiment(distribution, size, True).generate()
outputDesign = model(inputDesign)
# Compute first order indices using the Saltelli estimator
sensitivityAnalysis = ot.SaltelliSensitivityAlgorithm(inputDesign, outputDesign, size)
first_order = sensitivityAnalysis.getFirstOrderIndices()
total_order = sensitivityAnalysis.getTotalOrderIndices()
print("S = %s " % (str(first_order)))
print("ST = %s " % (str(total_order)))

#
def mySaltelliSobolEstimator(distribution, size,model):
    inputDesign = ot.SobolIndicesExperiment(distribution, size, True).generate()
    outputDesign = model(inputDesign)
    sensitivityAnalysis = ot.SaltelliSensitivityAlgorithm(inputDesign, outputDesign, size)
    return sensitivityAnalysis

def myJansenSobolEstimator(distribution, size,model):
    inputDesign = ot.SobolIndicesExperiment(distribution, size, True).generate()
    outputDesign = model(inputDesign)
    sensitivityAnalysis = ot.JansenSensitivityAlgorithm(inputDesign, outputDesign, size)
    return sensitivityAnalysis

def myMauntzKucherenkoSobolEstimator(distribution, size,model):
    inputDesign = ot.SobolIndicesExperiment(distribution, size, True).generate()
    outputDesign = model(inputDesign)
    sensitivityAnalysis = ot.MauntzKucherenkoSensitivityAlgorithm(inputDesign, outputDesign, size)
    return sensitivityAnalysis

def myMartinezSobolEstimator(distribution, size,model):
    inputDesign = ot.SobolIndicesExperiment(distribution, size, True).generate()
    outputDesign = model(inputDesign)
    sensitivityAnalysis = ot.MartinezSensitivityAlgorithm(inputDesign, outputDesign, size)
    return sensitivityAnalysis

def mySobolLowDiscrepancyExperiment(distribution, size,model):
    sequence = ot.SobolSequence(distribution.getDimension())
    experiment = ot.LowDiscrepancyExperiment(sequence, distribution, size)
    sensitivity_algorithm = ot.SaltelliSensitivityAlgorithm(experiment, model)
    return sensitivity_algorithm

def myHaltonLowDiscrepancyExperiment(distribution, size,model):
    sequence = ot.HaltonSequence(distribution.getDimension())
    experiment = ot.LowDiscrepancyExperiment(sequence, distribution, size)
    sensitivity_algorithm = ot.SaltelliSensitivityAlgorithm(experiment, model)
    return sensitivity_algorithm

def mySaltelliSobolRandomizedSequence(distribution, size,model):
    sequence = ot.SobolSequence(distribution.getDimension())
    experiment = ot.LowDiscrepancyExperiment(sequence, distribution, size)
    experiment.setRandomize(True)
    sensitivity_algorithm = ot.SaltelliSensitivityAlgorithm(experiment, model)
    return sensitivity_algorithm

def myLHSExperiment(distribution, size,model):
    experiment = ot.LHSExperiment(distribution, size)
    # lhs.setAlwaysShuffle(True) # randomized = works
    sensitivity_algorithm = ot.SaltelliSensitivityAlgorithm(experiment, model)
    return sensitivity_algorithm

def myMonteCarloExperiment(distribution, size,model):
    experiment = ot.MonteCarloExperiment(distribution, size)
    sensitivity_algorithm = ot.SaltelliSensitivityAlgorithm(experiment, model)
    return sensitivity_algorithm

def myImportanceSamplingExperiment(distribution, size,model):
    experiment = ot.ImportanceSamplingExperiment(distribution, distribution,size)
    sensitivity_algorithm = ot.SaltelliSensitivityAlgorithm(experiment, model)
    # Fails : this is good
    return sensitivity_algorithm

def myOptimalLHSExperiment(distribution, size,model):
    # Build standard randomized LHS algorithm
    lhs = ot.LHSExperiment(distribution, size)
    #lhs.setAlwaysShuffle(False) # randomized
    # Defining space fillings
    spaceFilling = ot.SpaceFillingC2()
    # RandomBruteForce MonteCarlo with N designs (LHS with C2 optimization)
    N = 10000
    optimalLHSAlgorithm = ot.MonteCarloLHS(lhs, N, spaceFilling)
    experiment = optimalLHSAlgorithm.getLHS()
    sensitivity_algorithm = ot.SaltelliSensitivityAlgorithm(experiment, model)
    return sensitivity_algorithm


def mySaltelliSamplingMethodQMCIndicesExperiment(distribution, size,model):
    ot.ResourceMap.SetAsString('SobolIndicesExperiment-SamplingMethod', 'QMC')
    inputDesign = ot.SobolIndicesExperiment(distribution, size, True).generate()
    outputDesign = model(inputDesign)
    sensitivity_algorithm = ot.SaltelliSensitivityAlgorithm(inputDesign, outputDesign, size)
    return sensitivity_algorithm

def mySaltelliSamplingMethodLHSIndicesExperiment(distribution, size,model):
    ot.ResourceMap.SetAsString('SobolIndicesExperiment-SamplingMethod', 'LHS')
    inputDesign = ot.SobolIndicesExperiment(distribution, size, True).generate()
    outputDesign = model(inputDesign)
    sensitivity_algorithm = ot.SaltelliSensitivityAlgorithm(inputDesign, outputDesign, size)
    return sensitivity_algorithm

def myBenchmark(myestimator,title):
    n = 18
    abserr = np.zeros((n,2*dimension))
    sizearray = np.zeros((n))
    size = 8
    for i in range(n):
        sizearray[i] = size
        print("i=%d, size = %d" % (i, size))
        sensitivityAnalysis = myestimator(distribution,size,model)
        first_order = sensitivityAnalysis.getFirstOrderIndices()
        total_order = sensitivityAnalysis.getTotalOrderIndices()
        abserr[i,0] = abs(first_order[0] - exact['S1'])
        abserr[i,1] = abs(first_order[1] - exact['S2'])
        abserr[i,2] = abs(first_order[2] - exact['S3'])
        abserr[i,3] = abs(total_order[0] - exact['ST1'])
        abserr[i,4] = abs(total_order[1] - exact['ST2'])
        abserr[i,5] = abs(total_order[2] - exact['ST3'])
        size = size * 2
    
    # Figure
    pl.figure()
    pl.plot(sizearray,abserr[:,0],"-",label="S1")
    pl.plot(sizearray,abserr[:,1],":",label="S2")
    pl.plot(sizearray,abserr[:,2],"--",label="S3")
    pl.plot(sizearray,abserr[:,3],"-+",label="ST1")
    pl.plot(sizearray,abserr[:,4],"-*",label="ST2")
    pl.plot(sizearray,abserr[:,5],"-^",label="ST3")
    pl.xlabel("Iterations")
    pl.ylabel("Absolute error")
    pl.legend()
    pl.xscale("log")
    pl.yscale("log")
    pl.title(title)
    pl.savefig(title + ".pdf")
    #pl.savefig(title + ".png")



myBenchmark(myLHSExperiment,"LHSExperiment")

'''
myBenchmark(myOptimalLHSExperiment,"OptimalLHSExperiment")
myBenchmark(myMonteCarloExperiment,"MonteCarloExperiment")
myBenchmark(mySaltelliSobolRandomizedSequence,"SaltelliSobolRandomizeSequence")
myBenchmark(myHaltonLowDiscrepancyExperiment,"HaltonLowDiscrepancyExperiment")
myBenchmark(mySobolLowDiscrepancyExperiment,"SobolLowDiscrepancyExperiment")
myBenchmark(mySaltelliSamplingMethodQMCIndicesExperiment,"SaltelliSamplingMethodQMC")
myBenchmark(mySaltelliSamplingMethodLHSIndicesExperiment,"SaltelliSamplingMethodLHS")
myBenchmark(mySaltelliSobolEstimator,"SaltelliSensitivityAlgorithm")
myBenchmark(myJansenSobolEstimator,"JansenSensitivityAlgorithm")
myBenchmark(myMauntzKucherenkoSobolEstimator,"MauntzKucherenkoSensitivityAlgorithm")
myBenchmark(myMartinezSobolEstimator,"MartinezSensitivityAlgorithm")
'''
