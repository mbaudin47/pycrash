#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive numerical differentiation
Authors: R. S. Stepleman and N. D. Winarsky
Journal: Math. Comp. 33 (1979), 1257-1264 
"""
import numpy as np
import openturns as ot
import openturns.viewer as otv

g = ot.SymbolicFunction(["x"], ["exp(-x / 1.e6)"])

def central_finite_difference(g, x, h):
    x1 = x + h
    x2 = x - h
    y1 = g([x1])
    y2 = g([x2])
    g_fd = (y1 - y2) / (x1 + (- x2) ) # Magic trick?
    # g_fd = (y1 - y2) / (2.0 * h)
    return g_fd

x = 1.0
number_of_points = 1000
h_array = ot.Sample([[x] for x in np.logspace(-7.0, 5.0, number_of_points)])
error_array = ot.Sample(number_of_points, 1)
for i in range(number_of_points):
    g_gradient = g.gradient([x])
    h = h_array[i, 0]
    g_fd = central_finite_difference(g, x, h)
    error_array[i, 0] = abs(g_fd[0] - g_gradient[0, 0])

graph = ot.Graph("Finite difference", "h", "Error", True)
curve = ot.Curve(h_array, error_array)
graph.add(curve)
graph.setLogScale(ot.GraphImplementation.LOGXY)
otv.View(graph)

# Algorithm to detect h*
h0 = 1.e5
h_previous = h0
g_fd_previous = central_finite_difference(g, x, h_previous)
diff_previous = np.inf
for i in range(53):
    h_current = h_previous / 2.0
    g_fd_current = central_finite_difference(g, x, h_current)
    diff_current = abs(g_fd_current[0] - g_fd_previous[0])
    print("i=%d, h=%.4e, |FD(h_current) - FD(h_previous)=%.4e" % (
        i, h_current, diff_current))
    if diff_previous < diff_current:
        print("Stop!")
        break
    g_fd_previous = g_fd_current
    h_previous = h_current
    diff_previous = diff_current

print("Optimum h=", h_current)
