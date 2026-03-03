# -*- coding: utf-8 -*-
# Copyright (C) 2018 - Michael Baudin

import pylab as pl
import openturns as ot
import numpy as np


"""
Une librairie pour l'assimilation de donnée.
"""


def plotModelVsData(theta, xobserved, yobserved, obsFunction):
    # Modele vs Donnees
    y = obsFunction(theta)
    pl.plot(xobserved, yobserved, "bo", label="Data")
    pl.plot(xobserved, y, "ro", label="Model")
    pl.legend(loc=2)
    return None


def plotModelVsDataBeforeAndAFter(thetaB, thetaStar, xobserved, yobserved, obsFunction):
    # Plot the fit
    pl.plot(xobserved, yobserved, "bo", label="Data")
    y = obsFunction(thetaB)
    pl.plot(xobserved, y, "ro", label="Before DA")
    y = obsFunction(thetaStar)
    pl.plot(xobserved, y, "go", label="After DA")
    pl.legend(loc=2)
    return None


def plotObservationsVsPredictions(theta, yobserved, obsFunction):
    # Plot the Ymodel vs Yobservations
    pl.plot(yobserved, yobserved, "r-")
    y = obsFunction(theta)
    pl.plot(yobserved, y, "bo")
    pl.xlabel("Observation")
    pl.ylabel("Prediction")
    pl.legend(loc=2)
    return None


def plotResiduals(theta, yobserved, obsFunction, sigmaY=None):
    # Plot the distribution of the residuals
    y = obsFunction(theta)
    r = y - yobserved
    r = np.array(r)
    r = r.flatten()
    r = ot.Sample([[v] for v in r])
    myGraph = ot.HistogramFactory().build(r).drawPDF()
    # Loi normale ajustée
    fittedNormal = ot.NormalFactory().build(r)
    mu = fittedNormal.getParameter()[0]
    sigma = fittedNormal.getParameter()[1]
    ng = fittedNormal.drawPDF()
    ng.setColors(["blue"])
    myGraph.add(ng)
    # Loi normale sur les observations
    if sigmaY != None:
        obsDistr = ot.Normal(0.0, sigmaY)
        og = obsDistr.drawPDF()
        og.setColors(["green"])
        myGraph.add(og)
    #
    if sigmaY != None:
        myGraph.setLegends(["Data", "Normal data fit", "Normal(0,%.4e)" % (sigmaY)])
    else:
        myGraph.setLegends(["Data", "Normal data fit"])
    myGraph.setTitle("Residuals analysis - Mu=%.4e, Sigma=%.4e" % (mu, sigma))
    myGraph.setXTitle("Residuals")
    myGraph.setYTitle("Probability distribution function")
    return myGraph


def plotObservationsVsPredictionsBeforeAfter(theta0, thetaStar, yobserved, obsFunction):
    # Plot the Ymodel vs Yobservations
    pl.plot(yobserved, yobserved, "b-")
    y = obsFunction(theta0)
    pl.plot(yobserved, y, "ro", label="Before")
    y = obsFunction(thetaStar)
    pl.plot(yobserved, y, "go", label="After")
    pl.xlabel("Observation")
    pl.ylabel("Prediction")
    pl.legend(loc=2)
    return None
