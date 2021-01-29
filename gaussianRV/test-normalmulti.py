#! /usr/bin/env python

from __future__ import print_function
import openturns as ot
import numpy as np

# Case 1 : OK
R = ot.CorrelationMatrix(2, [1.0, 0.05, 0.05, 1.0])
copula = ot.NormalCopula(R)

# Case 2 : KO : X2=X1
if False:
    R = ot.CorrelationMatrix(2, [1.0, 1.0, 1.0, 1.0])
    copula = ot.NormalCopula(R)

# Case 3 : KO
if False:
    R = ot.CorrelationMatrix(3, [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    print(R)
    copula = ot.NormalCopula(R)

# Case 4 : KO
if False:
    R = ot.CorrelationMatrix(3, [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    copula = ot.NormalCopula(R)

# Utilisation
marginals = [ot.Normal(), ot.Uniform()]
mydist = ot.ComposedDistribution(marginals)
X = mydist.getSample(10)


# Case 2 : workaround
N = 10
X1 = marginals[0].getSample(N)
p1 = marginals[0].computeCDF(X1)
PP1 = np.array(p1)
X2 = marginals[1].computeQuantile(PP1[:, 0])
X = ot.NumericalSample(N, 2)
X[:, 0] = X1
X[:, 1] = X2

# Select independent components.
R = ot.CorrelationMatrix(3, [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
print(R)
copula = ot.NormalCopula(R)
#
dimension = R.getDimension()
remaining_indices = []
equal_indices = []
for i in range(dimension):
    for j in range(0, i):
        print("i=", i, ", j=", j, "R[i,j]=", R[i, j])
        if R[i, j] == 1.0:
            equal_indices.append([i, j])
