# -*- coding: utf-8 -*-
"""
A degenerate gaussian random vector example.
Adapted from 
https://nbviewer.org/github/mbaudin47/pycrash/blob/master/gaussianRV/simulate_degenerate_gaussian_random_vector.ipynb

"""
import openturns as ot
import openturns.viewer as otv

ot.Log.Show(ot.Log.ALL)
dimension = 3
R = ot.CorrelationMatrix(3)
R[0, 1] = 1.0
print("R = ")
print(R)

mu = [0.0] * dimension
sigma = [1.0] * dimension
nrv = ot.Normal(mu, sigma, R)
sample_size = 100
sample = nrv.getSample(sample_size)
print(sample)

def view_sample(sample, title, figure_size=5.0):
    graph = ot.VisualTest.DrawPairs(sample)
    graph.setTitle(title)
    figure = otv.View(graph).getFigure()
    figure.set_figheight(figure_size)
    figure.set_figwidth(figure_size)
    return graph

graph = view_sample(sample, "With SVD")

"""
C = nrv.getCovariance()
L = C.computeCholesky()
RuntimeError: NotSymmetricDefinitePositiveException : Error: the matrix is not definite positive.
"""
