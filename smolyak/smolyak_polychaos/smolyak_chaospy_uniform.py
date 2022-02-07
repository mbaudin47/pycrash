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

def plot_smolyak_nodes(rule, level, verbose=False, factor = 5.0e2):
    sparse_nodes, sparse_weights = chaospy.generate_quadrature(
        level, joint, rule=rule, sparse=True
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

def print_smolyak_nodes(rule, level, verbose=False, format_nodes="%-20s", format_markdown=True):
    nodes, weights = chaospy.generate_quadrature(
        level, joint, rule=rule, sparse=True
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


x1 = chaospy.Uniform(-1.0, 1.0)
x2 = chaospy.Uniform(-1.0, 1.0)
joint = chaospy.J(x1, x2)
print("joint distribution")
print(joint)
#
print("+ Smolyak sparse-grid")

rule=["gaussian", "gaussian"]
level = 3
plot_smolyak_nodes(rule, level, verbose=False)
pyplot.show()

#
print("+ Plot different levels")
factor = 100.0
level = 0
for i in range(2):
    for j in range(3):
        index = level + 1
        pyplot.subplot(2, 3, index)
        plot_smolyak_nodes(rule, level, verbose=False, factor=factor)
        pyplot.title("level = %d" % (level))
        level += 1
pyplot.subplots_adjust(wspace=0.3, hspace = 0.7)
pyplot.suptitle("Smolyak-Legendre")
pyplot.show()

#
print("+ Print nodes and weights")
rule=["gaussian", "gaussian"]
print("rule = ", rule)
for level in range(1, 4):
    print("Level = ", level)
    print_smolyak_nodes(rule, level, verbose=False, format_nodes="%-20s")

rule=["clenshaw_curtis", "clenshaw_curtis"]
print("rule = ", rule)
for level in range(1, 4):
    print("Level = ", level)
    print_smolyak_nodes(rule, level, verbose=False, format_nodes="%-20s")
