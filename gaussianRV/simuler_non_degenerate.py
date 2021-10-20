# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 17:10:10 2021.

@author: C61372
"""

import openturns as ot
import openturns.viewer as otv

ot.RandomGenerator.SetSeed(77)

# Liste des marginales
marginal_collection = [ot.Normal(), ot.Uniform(), ot.Beta()]

# Dépendance
R = ot.CorrelationMatrix(3, [1.0, 0.5, 0.3, 0.5, 1.0, 0.9, 0.3, 0.9, 1.0])
print(R)
copula = ot.NormalCopula(R)
distribution = ot.ComposedDistribution(marginal_collection, copula)

# Simulation
sample_size = 100
sample = distribution.getSample(sample_size)
print(sample)

graph = ot.VisualTest.DrawPairs(sample)
graph.setTitle("With getSample()")
otv.View(graph)

# Create a gaussian random vector
dimension = R.getDimension()
mu = ot.Point(dimension)
sigma = ot.Point(dimension, 1.0)
distribution = ot.Normal(mu, sigma, R)
sample_z = distribution.getSample(sample_size)
print(sample_z)

graph = ot.VisualTest.DrawPairs(sample)
graph.setTitle("Normal sample")
otv.View(graph)

# Simulation pas à pas


def get_realization(marginal_collection, R):
    """Generate a new realization with Gaussian copula."""
    dimension = R.getDimension()
    # Generate a realization without correlation
    Z_distribution = ot.Normal(dimension)
    z_decorrelated = Z_distribution.getRealization()
    # Compute correlated Normal realization
    L = R.computeCholesky()
    U = L.transpose()
    z_correlated = U.solveLinearSystem(z_decorrelated)
    # Apply marginals
    x_correlated = ot.Point(dimension)
    for i in range(dimension):
        marginal = marginal_collection[i]
        cdf = ot.Normal().computeCDF(z_correlated[i])
        x_correlated[i] = marginal.computeQuantile(cdf)[0]
    return x_correlated, z_correlated


ot.RandomGenerator.SetSeed(77)

# Generate a sample
dimension = R.getDimension()
sample = ot.Sample(sample_size, dimension)
sample_z = ot.Sample(sample_size, dimension)
for i in range(sample_size):
    x_correlated, z_correlated = get_realization(marginal_collection, R)
    sample[i] = x_correlated
    sample_z[i] = z_correlated
#
graph = ot.VisualTest.DrawPairs(sample)
graph.setTitle("With step-by-step: X")
otv.View(graph)

graph = ot.VisualTest.DrawPairs(sample_z)
graph.setTitle("With step-by-step: Z")
otv.View(graph)
