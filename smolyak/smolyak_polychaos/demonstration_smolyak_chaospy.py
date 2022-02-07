#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiments with chaospy's Smolyak.

Adapted from 
https://chaospy.readthedocs.io/en/latest/user_guide/fundamentals/quadrature_integration.htm

Output
------
J(Normal(mu=1.5, sigma=0.2), Uniform(lower=0.1, upper=0.2))
+ Monte-Carlo quadrature
+ Gaussian quadrature
gauss_nodes = 
[[0.74991206 0.1025446 ]
 [0.74991206 0.11292344]
 [0.74991206 0.12970774]
 [0.74991206 0.15      ]
 [0.74991206 0.17029226]
 [0.74991206 0.18707656]
 [0.74991206 0.1974554 ]
 [1.02664812 0.1025446 ]
 [1.02664812 0.11292344]
 [1.02664812 0.12970774]
 [1.02664812 0.15      ]
 [1.02664812 0.17029226]
 [1.02664812 0.18707656]
 [1.02664812 0.1974554 ]
 [1.26911892 0.1025446 ]
 [1.26911892 0.11292344]
 [1.26911892 0.12970774]
 [1.26911892 0.15      ]
 [1.26911892 0.17029226]
 [1.26911892 0.18707656]
 [1.26911892 0.1974554 ]
 [1.5        0.1025446 ]
 [1.5        0.11292344]
 [1.5        0.12970774]
 [1.5        0.15      ]
 [1.5        0.17029226]
 [1.5        0.18707656]
 [1.5        0.1974554 ]
 [1.73088108 0.1025446 ]
 [1.73088108 0.11292344]
 [1.73088108 0.12970774]
 [1.73088108 0.15      ]
 [1.73088108 0.17029226]
 [1.73088108 0.18707656]
 [1.73088108 0.1974554 ]
 [1.97335188 0.1025446 ]
 [1.97335188 0.11292344]
 [1.97335188 0.12970774]
 [1.97335188 0.15      ]
 [1.97335188 0.17029226]
 [1.97335188 0.18707656]
 [1.97335188 0.1974554 ]
 [2.25008794 0.1025446 ]
 [2.25008794 0.11292344]
 [2.25008794 0.12970774]
 [2.25008794 0.15      ]
 [2.25008794 0.17029226]
 [2.25008794 0.18707656]
 [2.25008794 0.1974554 ]]
gauss_weights =  [3.54962871e-05 7.66768775e-05 1.04672762e-04 1.14577002e-04
 1.04672762e-04 7.66768775e-05 3.54962871e-05 1.99129258e-03
 4.30146670e-03 5.87199710e-03 6.42761121e-03 5.87199710e-03
 4.30146670e-03 1.99129258e-03 1.55461708e-02 3.35818738e-02
 4.58431227e-02 5.01808439e-02 4.58431227e-02 3.35818738e-02
 1.55461708e-02 2.95965637e-02 6.39326609e-02 8.72754401e-02
 9.55335277e-02 8.72754401e-02 6.39326609e-02 2.95965637e-02
 1.55461708e-02 3.35818738e-02 4.58431227e-02 5.01808439e-02
 4.58431227e-02 3.35818738e-02 1.55461708e-02 1.99129258e-03
 4.30146670e-03 5.87199710e-03 6.42761121e-03 5.87199710e-03
 4.30146670e-03 1.99129258e-03 3.54962871e-05 7.66768775e-05
 1.04672762e-04 1.14577002e-04 1.04672762e-04 7.66768775e-05
 3.54962871e-05]
+ Weightless quadrature
nodes = 
[[-1.         -0.70710678  0.          0.70710678  1.        ]]
weights =  [0.06666667 0.53333333 0.8        0.53333333 0.06666667]
+ Smolyak sparse-grid
sparse_nodes = 
[[1.03311716 0.15      ]
 [1.15358984 0.125     ]
 [1.15358984 0.15      ]
 [1.15358984 0.175     ]
 [1.3        0.11464466]
 [1.3        0.125     ]
 [1.3        0.15      ]
 [1.3        0.175     ]
 [1.3        0.18535534]
 [1.35160724 0.15      ]
 [1.5        0.10954915]
 [1.5        0.11464466]
 [1.5        0.125     ]
 [1.5        0.13454915]
 [1.5        0.15      ]
 [1.5        0.16545085]
 [1.5        0.175     ]
 [1.5        0.18535534]
 [1.5        0.19045085]
 [1.64839276 0.15      ]
 [1.7        0.11464466]
 [1.7        0.125     ]
 [1.7        0.15      ]
 [1.7        0.175     ]
 [1.7        0.18535534]
 [1.84641016 0.125     ]
 [1.84641016 0.15      ]
 [1.84641016 0.175     ]
 [1.96688284 0.15      ]]
sparse_weights =  [ 0.04587585  0.08333333 -0.16666667  0.08333333  0.14285714 -0.25
  0.21428571 -0.25        0.14285714  0.45412415  0.187887   -0.28571429
  0.33333333  0.312113   -1.0952381   0.312113    0.33333333 -0.28571429
  0.187887    0.45412415  0.14285714 -0.25        0.21428571 -0.25
  0.14285714  0.08333333 -0.16666667  0.08333333  0.04587585]
sum(weights) =  1.0000000000000002
"""
import numpy
import chaospy
from problem_formulation import joint
from matplotlib import pyplot
from problem_formulation import model_solver

print(joint)

#
print("+ Monte-Carlo quadrature")
nodes = joint.sample(500)
weights = numpy.repeat(1 / 500, 500)

pyplot.scatter(*nodes)
pyplot.show()

evaluations = numpy.array([model_solver(node) for node in nodes.T])
estimate = numpy.sum(weights * evaluations.T, axis=-1)

#
print("+ Gaussian quadrature")
gauss_nodes, gauss_weights = chaospy.generate_quadrature(6, joint, rule="gaussian")

print("gauss_nodes = ")
print(gauss_nodes.T)
print("gauss_weights = ", gauss_weights)

pyplot.scatter(*gauss_nodes, s=gauss_weights * 1e4)
pyplot.show()

#
print("+ Weightless quadrature")
# grid_nodes, grid_weights = chaospy.generate_quadrature(
#    3, joint, rule=["genz_keister_24", "fejer_2"], growth=True)

grid_nodes, grid_weights = chaospy.generate_quadrature(
    3, joint, rule=["gaussian", "fejer"], growth=True
)

pyplot.scatter(*grid_nodes, s=grid_weights * 6e3)
pyplot.show()

interval = (-1, 1)
nodes, weights = chaospy.generate_quadrature(4, interval, rule="clenshaw_curtis")
print("nodes = ")
print(nodes)
print("weights = ", weights)

#
print("+ Smolyak sparse-grid")

sparse_nodes, sparse_weights = chaospy.generate_quadrature(
    3, joint, rule=["gaussian", "fejer"], sparse=True
)

print("sparse_nodes = ")
print(sparse_nodes.T)
print("sparse_weights = ", sparse_weights)

size = sparse_nodes.shape[1]
idx = sparse_weights > 0
factor = 5.0e2
pyplot.title("Smolyak, n = %d" % (size))
pyplot.scatter(*sparse_nodes[:, idx], s=sparse_weights[idx] * factor)
pyplot.scatter(*sparse_nodes[:, ~idx], s=-sparse_weights[~idx] * factor, color="grey")
pyplot.show()
print("sum(weights) = ", sum(sparse_weights))
