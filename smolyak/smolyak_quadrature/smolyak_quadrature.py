#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 22:24:51 2022

@author: devel
"""

import openturns as ot
import openturns.viewer as otv
import smolyak
import pylab as pl

dimension = 3
k_stage = 5
experiment = smolyak.SmolyakExperiment(dimension, k_stage)
sample, weights = experiment.generateWithWeights()

# Plot nodes
graph = ot.VisualTest.DrawPairs(sample)
view = otv.View(graph, figure_kw={"figsize": (5.0, 5.0)})
view.getFigure().savefig("smokyak_quadrature_3_5.pdf", bbox_inches="tight")

# Plot in dimension 2, at various levels
def draw_experiment_2D(k_stage):
    dimension = 2
    experiment = smolyak.SmolyakExperiment(dimension, k_stage)
    sample, weights = experiment.generateWithWeights()

    # Plot nodes
    graph = ot.Graph("k=%d" % (k_stage), r"$x_1$", r"$x_2$", True)
    cloud = ot.Cloud(sample)
    graph.add(cloud)
    return graph


nb_rows = 3
nb_columns = 3
grid = ot.GridLayout(nb_rows, nb_columns)
k_stage = 0
epsilon = 0.1
for i in range(nb_rows):
    for j in range(nb_columns):
        graph = draw_experiment_2D(k_stage)
        graph.setBoundingBox(ot.Interval([0.0 - epsilon] * 2, [1.0 + epsilon] * 2))
        if i != nb_rows - 1:
            graph.setXTitle("")
        if j != 0:
            graph.setYTitle("")
        grid.setGraph(i, j, graph)
        k_stage += 1
view = otv.View(grid, figure_kw={"figsize": (7.0, 6.0)})
pl.subplots_adjust(wspace=0.5, hspace=0.6)
view.getFigure().savefig("smokyak_quadrature_dim_2.pdf", bbox_inches="tight")

# Number of nodes
k_stage_max = 8
dimension_max = 8
graph = ot.Graph("Smolyak quadrature", r"$k$", r"$n$", True, "topleft")
for dimension in range(1, dimension_max):
    number_of_nodes = ot.Sample(k_stage_max, 1)
    for k_stage in range(k_stage_max):
        experiment = smolyak.SmolyakExperiment(dimension, k_stage)
        sample, weights = experiment.generateWithWeights()
        size = sample.getSize()
        number_of_nodes[k_stage, 0] = size
    cloud = ot.Cloud(ot.Sample.BuildFromPoint(range(k_stage_max)), number_of_nodes)
    cloud.setLegend("$d = %d$" % (dimension))
    cloud.setPointStyle("bullet")
    graph.add(cloud)
graph.setColors(ot.Drawable().BuildDefaultPalette(dimension_max - 1))
graph.setLogScale(ot.GraphImplementation.LOGY)
view = otv.View(
    graph,
    figure_kw={"figsize": (4.0, 3.0)},
    legend_kw={"bbox_to_anchor": (1.0, 1.0), "loc": "upper left"},
)
view.getFigure().savefig("smokyak_quadrature_k_vs_d.pdf", bbox_inches="tight")
