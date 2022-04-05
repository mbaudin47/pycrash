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
from matplotlib import pyplot

def plot_smolyak_nodes(vector_distribution, rule, level, verbose=False, factor = 5.0e2):
    if len(vector_distribution) != len(rule):
        raise ValueError("Dimension of distribution is %d, but there are %d quadrature rules" % (
            len(vector_distribution), len(rule)))
    sparse_nodes, sparse_weights = chaospy.generate_quadrature(
        level, vector_distribution, rule=rule, sparse=True
    )
    if verbose:
        print("rule = ", rule)
        print("level = ", level)
        print("sparse_nodes = ")
        print(sparse_nodes.T)
        print("sparse_weights = ", sparse_weights)
        print("sum(weights) = ", sum(sparse_weights))
    
    size = sparse_nodes.shape[1]
    idx = sparse_weights > 0
    pyplot.title("Smolyak [%s], level = %d, n = %d" % (",".join(rule), level, size))
    pyplot.scatter(*sparse_nodes[:, idx], s=sparse_weights[idx] * factor)
    pyplot.scatter(*sparse_nodes[:, ~idx], s=-sparse_weights[~idx] * factor, color="grey")
    pyplot.axis("square")
    pyplot.xlabel("$x_1$")
    pyplot.ylabel("$x_2$")
    epsilon = 0.1
    pyplot.xlim(-1.0 - epsilon, 1.0 + epsilon)
    pyplot.ylim(-1.0 - epsilon, 1.0 + epsilon)
    return None

def print_smolyak_nodes(vector_distribution, rule, level, verbose=False, format_nodes="%-20s", format_markdown=True):
    if len(vector_distribution) != len(rule):
        raise ValueError("Dimension of distribution is %d, but there are %d quadrature rules" % (
            len(vector_distribution), len(rule)))
    nodes, weights = chaospy.generate_quadrature(
        level, vector_distribution, rule=rule, sparse=True
    )
    dimension = nodes.shape[0]
    size = nodes.shape[1]
    for i in range(size):
        x_str = ["%.4f" % (nodes[j, i]) for j in range(dimension)]
        x_print = ", ".join(x_str)
        final_x_print = format_nodes % x_print
        if format_markdown:
            print("| %d | [%s] | %.4f |" % (i, final_x_print, weights[i]))
        else:
            print("x[%d] : [%s], w[%d] = %.4f" % (i, final_x_print, i, weights[i]))
    return None

def plot_smolyak_grid(number_of_rows, number_of_columns, vector_distribution, rule, 
                      factor):
    level = 0
    for i in range(number_of_rows):
        for j in range(number_of_columns):
            index = level + 1
            pyplot.subplot(number_of_rows, number_of_columns, index)
            plot_smolyak_nodes(vector_distribution, rule, level, verbose=False, factor=factor)
            pyplot.title("level = %d" % (level))
            level += 1
    return None


x1 = chaospy.Uniform(-1.0, 1.0)
x2 = chaospy.Uniform(-1.0, 1.0)
vector_distribution = chaospy.J(x1, x2)
print("vector_distribution distribution")
print(vector_distribution)
#
print("+ Smolyak sparse-grid")

rule=["gaussian", "gaussian"]
level = 3
plot_smolyak_nodes(vector_distribution, rule, level, verbose=False)
pyplot.show()

#
print("+ Plot different levels for Smolyak-Legendre")
factor = 100.0
number_of_rows = 2
number_of_columns = 3
rule=["gaussian", "gaussian"]
print("rule = ", rule)
plot_smolyak_grid(number_of_rows, number_of_columns, vector_distribution, rule, 
                      factor)
pyplot.subplots_adjust(wspace=0.3, hspace = 0.7)
pyplot.suptitle("Smolyak-Legendre")
pyplot.show()

#
print("+ Plot different levels for Smolyak-Clenshaw-Curtis")
factor = 100.0
number_of_rows = 2
number_of_columns = 3
rule=["clenshaw_curtis", "clenshaw_curtis"]
print("rule = ", rule)
plot_smolyak_grid(number_of_rows, number_of_columns, vector_distribution, rule, 
                      factor)
pyplot.subplots_adjust(wspace=0.3, hspace = 0.7)
pyplot.suptitle("Smolyak-Clenshaw-Curtis")
pyplot.show()

#
print("+ Plot different levels for Smolyak-Fejér")
factor = 100.0
number_of_rows = 2
number_of_columns = 3
rule=["fejer", "fejer"]
print("rule = ", rule)
plot_smolyak_grid(number_of_rows, number_of_columns, vector_distribution, rule, 
                      factor)
pyplot.subplots_adjust(wspace=0.3, hspace = 0.7)
pyplot.suptitle("Smolyak-Fejér")
pyplot.show()

#
print("+ Print nodes and weights for Smolyak-Legendre (dimension 2)")
rule=["gaussian", "gaussian"]
print("rule = ", rule)
for level in range(1, 4):
    print("Level = ", level)
    print_smolyak_nodes(vector_distribution, rule, level, verbose=False, format_nodes="%-20s")

print("+ Print nodes and weights for Smolyak-Clenshaw-Curtis (dimension 2)")
rule=["clenshaw_curtis", "clenshaw_curtis"]
print("rule = ", rule)
for level in range(1, 4):
    print("Level = ", level)
    print_smolyak_nodes(vector_distribution, rule, level, verbose=False, format_nodes="%-20s")

print("+ Print nodes and weights for Smolyak-Fejér (dimension 2)")
rule=["fejer", "fejer"]
print("rule = ", rule)
for level in range(1, 4):
    print("Level = ", level)
    print_smolyak_nodes(vector_distribution, rule, level, verbose=False, format_nodes="%-20s")

print("+ Print nodes and weights for Smolyak-Clenshaw-Curtis (dimension 1)")
rule=["clenshaw_curtis"]
x1 = chaospy.Uniform(-1.0, 1.0)
vector_distribution = chaospy.J(x1)
print("rule = ", rule)
for level in range(1, 6):
    print("Level = ", level)
    print_smolyak_nodes(vector_distribution, rule, level, verbose=False, format_nodes="%-20s")

print("+ Print nodes and weights for Smolyak with uniform distribution (dimension 1)")
level = 5
x1 = chaospy.Uniform(-5.0, 5.0)
vector_distribution = chaospy.J(x1)
for rule in (["clenshaw_curtis"], ["fejer"]):
    print("rule = ", rule)
    print("Level = ", level)
    print_smolyak_nodes(vector_distribution, rule, level, verbose=False, format_nodes="%-20s")

print("+ Print nodes and weights for Smolyak with normal distribution (dimension 1)")
level = 5
x1 = chaospy.Normal(1.0, 4.0)
vector_distribution = chaospy.J(x1)
for rule in (["clenshaw_curtis"], ["fejer"]):
    print("rule = ", rule)
    print("Level = ", level)
    print_smolyak_nodes(vector_distribution, rule, level, verbose=False, format_nodes="%-20s")
