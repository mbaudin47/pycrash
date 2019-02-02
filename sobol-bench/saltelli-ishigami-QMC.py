# -*- coding: utf-8 -*-
"""
Estimates the sensitivity indices of Ishigami test function.
Manually generates the input design of experiments that 
the SaltelliSensitivityAlgorithm class expects. 
In other words, computes the inputDesign variable in the following 
statement :
    
salg = ot.SaltelliSensitivityAlgorithm(inputDesign, outputDesign, size)

The goal of this script is to detect bugs with the estimate of 
Sobol' indices with low discrepancy sequence, as revealed by the 
following ticket 

https://github.com/openturns/openturns/issues/1000

The trick is to use a low discrepancy sequence with twice the dimension.

Thanks to 
* Bertrand Iooss
* Régis Lebrun

References
Introduction to sensitivity analysis with NISP
Michael Baudin (EDF), Jean-Marc Martinez (CEA)
Version 0.5, February 2014
Section 4.5 "Summary of the results"
"""

import openturns as ot
from openturns.viewer import View
from math import pi
import pylab as pl
import numpy as np

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
#
inputDesign = ot.SobolIndicesExperiment(distribution, size, True).generate()
outputDesign = model(inputDesign)
# Compute first order indices using the Saltelli estimator
sensitivityAnalysis = ot.SaltelliSensitivityAlgorithm(inputDesign, outputDesign, size)
first_order = sensitivityAnalysis.getFirstOrderIndices()
total_order = sensitivityAnalysis.getTotalOrderIndices()
print("S = %s " % (str(first_order)))
print("ST = %s " % (str(total_order)))

def generateSampleKernel(A,B,computeSecondOrder=False):
    '''
    Given A and B, generates the required sample.
    '''
    design = ot.Sample(A) # Create copy
    design.add(B)
    dimension = distribution.getDimension()
    # Compute designs of type Saltelli/Martinez for 1st order
    for p in range(dimension):
        E = ot.Sample(A) # E=A
        E[:, p] = B[:, p] # p-th column of E := p-th column of B
        design.add(E)
    # Special case for dim=2: do not add the C sample
    if (computeSecondOrder & (dimension != 2)):
        for p in range(dimension):
            C = ot.Sample(B) # C=B
            C[:, p] = A[:, p] # p-th column of E := p-th column of A
            design.add(C)
    return design

def generateByMonteCarlo(distribution, size, computeSecondOrder=False):
    '''
    Generates the input DOE for the estimator of the Sobol' sensitivity 
    indices.
    Uses a plain, simple, raw, Monte-Carlo DOE.
    '''
    # Monte-Carlo sampling
    A = distribution.getSample(size) # A
    B = distribution.getSample(size) # B
    # Uses the kernel to generate the sample
    design = generateSampleKernel(A,B,computeSecondOrder)
    return design

def generateByLowDiscrepancy(distribution, size, computeSecondOrder=False):
    '''
    Generates the input DOE for the estimator of the Sobol' sensitivity 
    indices.
    Uses a Low Discrepancy sequence.
    '''
    dimension = distribution.getDimension()
    # Create a doubled distribution
    marginalList = [distribution.getMarginal(p) for p in range(dimension)]
    twiceDistribution = ot.ComposedDistribution(marginalList*2)
    # Generates a low discrepancy sequence in twice the dimension
    sequence = ot.SobolSequence(2*dimension)
    experiment = ot.LowDiscrepancyExperiment(sequence, twiceDistribution, size)
    fullDesign = experiment.generate()
    # Split the A and B designs
    A = fullDesign[:,0:dimension] # A
    B = fullDesign[:,dimension:2*dimension] # B
    # Uses the kernel to generate the sample
    design = generateSampleKernel(A,B,computeSecondOrder)
    return design

def generateByLHS(distribution, size, computeSecondOrder=False):
    '''
    Generates the input DOE for the estimator of the Sobol' sensitivity 
    indices.
    Uses a LHS design.
    '''
    dimension = distribution.getDimension()
    # Create a doubled distribution
    marginalList = [distribution.getMarginal(p) for p in range(dimension)]
    twiceDistribution = ot.ComposedDistribution(marginalList*2)
    # Generates a LHS in twice the dimension
    experiment = ot.LHSExperiment(twiceDistribution, size)
    fullDesign = experiment.generate()
    # Split the A and B designs
    A = fullDesign[:,0:dimension] # A
    B = fullDesign[:,dimension:2*dimension] # B
    # Uses the kernel to generate the sample
    design = generateSampleKernel(A,B,computeSecondOrder)
    return design

def myMonteCarloExperiment(distribution, size,model):
    inputDesign = generateByMonteCarlo(distribution, size)
    outputDesign = model(inputDesign)
    salg = ot.SaltelliSensitivityAlgorithm(inputDesign, outputDesign, size)
    return salg

def myLowDiscrepancyExperiment(distribution, size,model):
    inputDesign = generateByLowDiscrepancy(distribution, size)
    outputDesign = model(inputDesign)
    salg = ot.SaltelliSensitivityAlgorithm(inputDesign, outputDesign, size)
    return salg

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
    pl.savefig(title + ".png")

inputDesign = generateByMonteCarlo(distribution, size)
outputDesign = model(inputDesign)
salg = ot.SaltelliSensitivityAlgorithm(inputDesign, outputDesign, size)
first_order = salg.getFirstOrderIndices()
total_order = salg.getTotalOrderIndices()
print("S = %s " % (str(first_order)))
print("ST = %s " % (str(total_order)))

myBenchmark(myMonteCarloExperiment,"MonteCarloSampling")

inputDesign = generateByLowDiscrepancy(distribution, size)
outputDesign = model(inputDesign)
salg = ot.SaltelliSensitivityAlgorithm(inputDesign, outputDesign, size)
first_order = salg.getFirstOrderIndices()
total_order = salg.getTotalOrderIndices()
print("S = %s " % (str(first_order)))
print("ST = %s " % (str(total_order)))

myBenchmark(myLowDiscrepancyExperiment,"LowDiscrepancySampling-fixed")

inputDesign = generateByLHS(distribution, size)
outputDesign = model(inputDesign)
salg = ot.SaltelliSensitivityAlgorithm(inputDesign, outputDesign, size)
first_order = salg.getFirstOrderIndices()
total_order = salg.getTotalOrderIndices()
print("S = %s " % (str(first_order)))
print("ST = %s " % (str(total_order)))

myBenchmark(myLowDiscrepancyExperiment,"LHSSampling-fixed")
