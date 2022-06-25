#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiments with chaospy's Smolyak.

Adapted from 
https://chaospy.readthedocs.io/en/latest/user_guide/fundamentals/quadrature_integration.htm

Output
------

"""
import chaospy
import matplotlib.pyplot as pl

dimension = 5
def g_function(x):
    """
    Evaluates the exponential integrand function.

    Parameters
    ----------
    x : ot.Point

    Returns
    -------
    y : ot.Point
    """
    value = (1.0 + 1.0 / dimension) ** dimension
    for i in range(dimension):
        value *= x[i] ** (1.0 / dimension)
    return value

integral = 1.0
print("Exact integral = ", integral)
name = "ExponentialProduct"

def smolyakQuadrature(level, marginal_rule, dimension, g_function):
    distribution = chaospy.Iid(chaospy.Uniform(0.0, 1.0), dimension)
    rule = [marginal_rule] * dimension
    nodes, weights = chaospy.generate_quadrature(
        level, distribution, rule=rule, sparse=True
    )
    size = nodes.shape[1]
    approximate_integral = 0.0
    for i in range(size):
        node = nodes[:, i]
        g_value = g_function(node)
        approximate_integral += g_value * weights[i]
    return approximate_integral, size


# Test level 3
level = 3
marginal_rule = "gaussian"
approximate_integral, size = smolyakQuadrature(level, marginal_rule, dimension, 
                                               g_function)
print("approximate I = ", approximate_integral)

def computeSmolyakConvergence(level_max, marginal_rule, dimension, 
                              g_function):
    # Increase the level
    size_list = []
    abs_err_list = []
    for level in range(level_max):
        approximate_integral, size = smolyakQuadrature(level, marginal_rule, dimension, 
                                                       g_function)
        abs_err = abs(integral - approximate_integral)
        print("level = %d, size = %d, approx. I = %.6f, abs.err. = %.2e" % (level, size, approximate_integral, abs_err))
        size_list.append(size)
        abs_err_list.append(abs_err)
    return size_list, abs_err_list

level_max = 10

# Plot
pl.figure(figsize = (3.0, 2.0))
pl.title("Smolyak quadrature")
#
marginal_rule = "gaussian"
size_list, abs_err_list = computeSmolyakConvergence(
    level_max, marginal_rule, dimension, 
    g_function)
pl.plot(size_list, abs_err_list, "o", label="Gauss")
#
marginal_rule = "clenshaw_curtis"
size_list, abs_err_list = computeSmolyakConvergence(
    level_max, marginal_rule, dimension, 
    g_function)
pl.plot(size_list, abs_err_list, "^", label="Clenshaw-Curtis")
#
marginal_rule = "fejer"
size_list, abs_err_list = computeSmolyakConvergence(
    level_max, marginal_rule, dimension, 
    g_function)
pl.plot(size_list, abs_err_list, "v", label="Fejér")
#
pl.xlabel("$n$")
pl.ylabel("$e_{abs}$")
pl.legend(bbox_to_anchor=(1.0, 1.0))
pl.xscale("log")
pl.yscale("log")
pl.savefig("smolyak_convergence_chaospy.pdf", bbox_inches="tight")

