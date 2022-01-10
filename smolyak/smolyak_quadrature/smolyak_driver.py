#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute Smolyak quadrature.
"""
import openturns as ot
import openturns.viewer as otv
import os

# Run command
dimension = 3
k_stage = 5
command = "./smolyak_driver %d %d" % (dimension, k_stage)
os.system(command)

# Read weights and nodes
weights_filename = "weights.txt"
with open(weights_filename, "r") as fp:
    for count, line in enumerate(fp):
        pass
size = count + 1
print("Size : ", size)
w = ot.Point(size)
execfile(weights_filename)
x = ot.Sample(size, dimension)
nodes_filename = "nodes.txt"
execfile(nodes_filename)

# Plot
graph = ot.VisualTest.DrawPairs(x)
view = otv.View(graph, figure_kw={"figsize": (5.0, 5.0)})
view.getFigure().savefig("smokyak_3_5.pdf", bbox_inches="tight")
