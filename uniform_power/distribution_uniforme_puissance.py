#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse la loi de U^n où U ~ U(a, b)
"""
import openturns as ot
import openturns.viewer as otv
import numpy as np
import pylab as pl

def computeApproximateYMax(fitted_distribution, alpha = 0.1):
    x01 = fitted_distribution.computeQuantile(alpha, False)
    y01 = fitted_distribution.computePDF(x01)
    x99 = fitted_distribution.computeQuantile(alpha, True)
    y91 = fitted_distribution.computePDF(x99)
    x50 = fitted_distribution.computeQuantile(0.5)
    y50 = fitted_distribution.computePDF(x50)
    y_max = max(y01, y91, y50)
    return y_max

def plotPowersOfUniform(n_list, U, size = 1000000, 
                        y_min = -0.2, 
                        y_max = 3.0
                        ):
    sample = U.getSample(size)
    n_number = len(n_list)
    grid = ot.GridLayout(1, n_number)
    for index in range(n_number):
        n = n_list[index]
        power_sample = np.array(sample) ** n
        fitted_distribution = ot.KernelSmoothing().build(power_sample)
        graph = fitted_distribution.drawPDF()
        graph.setTitle("n = %d" % (n))
        graph.setXTitle("Y")
        graph.setLegends([""])
        if index > 0:
            graph.setYTitle("")
        # Adjust y bounding box
        if index > 0:
            bounding_box = graph.getBoundingBox()
            upper_bound = bounding_box.getUpperBound()
            upper_bound[1] = y_max
            bounding_box.setUpperBound(upper_bound)
            lower_bound = bounding_box.getLowerBound()
            lower_bound[1] = y_min
            bounding_box.setLowerBound(lower_bound)
            graph.setBoundingBox(bounding_box)
        grid.setGraph(0, index, graph)
    
    grid.setTitle("%s" % (U))
    return grid


distribution_list = [ot.Uniform(-1.0, 1.0), ot.Uniform(0.0, 1.0), ot.Uniform(-1.0, 0.0)]                    
for U in distribution_list:
    n_list = [1, 2, 3, 4, 5, 6]
    grid = plotPowersOfUniform(n_list, U)
    n_number = len(n_list)
    otv.View(grid, figure_kw={"figsize": (n_number * 2.0, 3.0)})
    pl.subplots_adjust(wspace = 0.3, top=  0.8)
