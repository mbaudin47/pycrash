# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 16:47:29 2021.

@author: C61372
"""
import openturns as ot

# Liste des marginales
marginals = [ot.Normal(), ot.Uniform(), ot.Beta()]

# Select independent components.
R = ot.CorrelationMatrix(3, [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
print(R)

# Calcule la SVD
singular_values, U, VT = R.computeSVD()
print("singular_values=", singular_values)

# copula = ot.NormalCopula(R)
#
dimension = R.getDimension()
nondegenerate_indices = list(range(dimension))
equal_indices = []
for i in range(dimension):
    for j in range(0, i):
        print("i=", i, ", j=", j, "R[i,j]=", R[i, j])
        if R[i, j] == 1.0:
            equal_indices.append([i, j])
            nondegenerate_indices.pop(j)
print("Equal marginals : ", equal_indices)
print("remaining_indices= ", nondegenerate_indices)

# Extrait la sous-matrice non dégénérée
nondegenerate_dimension = len(nondegenerate_indices)
nondegenerate_R = ot.CorrelationMatrix(nondegenerate_dimension)
nondegenerate_R[0, 1] = R[1, 2]
print(nondegenerate_R)

# Crée la liste des marginales non dégénérées
nondegenerate_marginals = []
for i in range(nondegenerate_dimension):
    k = nondegenerate_indices[i]
    nondegenerate_marginals.append(marginals[k])

# Simule la copule non dégénérée
nondegenerate_copula = ot.NormalCopula(nondegenerate_R)
nondegenerate_X = ot.ComposedDistribution(nondegenerate_marginals, nondegenerate_copula)
sample_size = 10
nondegenerate_sample = nondegenerate_X.getSample(sample_size)

# Crée l'échantillon
degenerate_dimension = dimension - nondegenerate_dimension
sample = ot.Sample(sample_size, dimension)
for i in range(sample_size):
    # Copie ce qui est non dégénéré
    for j in range(nondegenerate_dimension):
        k = nondegenerate_indices[j]
        sample[i, k] = nondegenerate_sample[i, j]
    # Copie ce qui est dégénéré
    for j in range(degenerate_dimension):
        k1 = equal_indices[j][0]
        k2 = equal_indices[j][1]
        sample[i, k2] = sample[i, k1]

print(sample)
