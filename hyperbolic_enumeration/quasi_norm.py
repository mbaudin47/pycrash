#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the quasi-norm of a vector.

References
----------

G. Blatman. Adaptive sparse polynomial chaos expansions for uncertainty propa-
gation and sensitivity analysis. PhD thesis, Blaise Pascal University-Clermont II,
France, 2009.

|x| = (sum x[i] ** q) ** (1 / q)

But:

a^b = exp(b * log(a))

Hence:

a^(1/q) = exp(log(a) / q)
    
"""


import openturns as ot
import openturns.viewer as otv
import numpy as np



class QuasiNorm(ot.OpenTURNSPythonFunction):
    def __init__(self, dimension, quasi_norm_parameter):
        super(QuasiNorm, self).__init__(dimension, 1)
        self.quasi_norm_parameter = quasi_norm_parameter

    def _exec(self, x):
        x = ot.Point(x)
        dimension = x.getDimension()
        norm = 0.0
        for i in range(dimension):
            norm += x[i] ** self.quasi_norm_parameter
        norm = np.exp(np.log(norm) / self.quasi_norm_parameter)
        return [norm]


def my_viewer(graph):
    legends = graph.getLegends()
    # Truncate the legends
    number_of_legends = len(legends)
    for i in range(number_of_legends):
        legends[i] = legends[i][0:4]
    graph.setLegendPosition("")
    view = otv.View(graph, figure_kw={"figsize": (2.0, 1.5)})
    figure = view.getFigure()
    # figure.legend(legends, bbox_to_anchor=(1.25, 0.9))
    ax = figure.get_axes()[0]
    ax.set_aspect("equal")
    ticks = [0.0, 0.5, 1.0]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    return figure


quasi_norm_parameter = 1.0
quasi_norm_1 = ot.Function(QuasiNorm(2, quasi_norm_parameter))

x = ot.Point([0.5, 0.5])
print("x=", x)
normx = quasi_norm_1(x)
print("|x|=", normx)


def plot_semi_norm(quasi_norm_parameter):
    number_of_contours = 5
    quasi_norm = ot.Function(QuasiNorm(2, quasi_norm_parameter))
    ot.ResourceMap.SetAsUnsignedInteger(
        "Contour-DefaultLevelsNumber", number_of_contours
    )
    graph = quasi_norm.draw([0.0, 0.0], [1.0, 1.0], [50] * 2)
    graph.setColors(ot.DrawableImplementation.BuildDefaultPalette(number_of_contours))
    graph.setXTitle(r"$x_1$")
    graph.setYTitle(r"$x_2$")
    graph.setTitle("$q=%.2f$" % (quasi_norm_parameter))
    figure = my_viewer(graph)
    return figure


figure = plot_semi_norm(1.0)
figure.savefig("figures/chaos-quasi_norm_1.pdf", bbox_inches="tight")

figure = plot_semi_norm(0.75)
figure.savefig("figures/chaos-quasi_norm_075.pdf", bbox_inches="tight")

figure = plot_semi_norm(0.5)
figure.savefig("figures/chaos-quasi_norm_05.pdf", bbox_inches="tight")

figure = plot_semi_norm(0.25)
figure.savefig("figures/chaos-quasi_norm_025.pdf", bbox_inches="tight")
