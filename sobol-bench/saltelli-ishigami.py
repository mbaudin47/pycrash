# -*- coding: utf-8 -*-
"""
Estimates the sensitivity indices of Ishigami test function.

References
Introduction to sensitivity analysis with NISP
Michael Baudin (EDF), Jean-Marc Martinez (CEA)
Version 0.5, February 2014
Section 4.5 "Summary of the results"
"""

import openturns as ot
from openturns.viewer import View
from math import pi

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
# Draw indices
View(sensitivityAnalysis.draw())




