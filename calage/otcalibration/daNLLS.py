# -*- coding: utf-8 -*-
# Copyright (C) 2018 - Michael Baudin

import openturns as ot
import numpy as np
from openturns.viewer import View
from . import dalib
from .daCalibrationFunction import DaCalibrationFunction

"""
Une librairie pour l'assimilation de données/
* NLLS
"""


class DaNLLS:
    def __init__(self):
        self.theta0 = None
        self.Xobservations = None
        self.Yobservations = None
        self.optimAlgo = 3
        self.nfeval = None
        self.boundsMin = None
        self.boundsMax = None
        self.observationFunction = None
        self.labelsTheta = None
        self.computeThetaByBootstrap = True
        self.thetaStarBootstrap = None
        self.bootstrapSize = 100
        self.confidenceLevel = 0.95

    def setConfidenceLevel(self, confidenceLevel):
        self.confidenceLevel = confidenceLevel
        return None

    def setComputeThetaByBootstrap(self, computeThetaByBootstrap):
        self.computeThetaByBootstrap = computeThetaByBootstrap
        return None

    def setBootstrapSize(self, bootstrapSize):
        self.bootstrapSize = bootstrapSize
        return None

    def setLabelsTheta(self, labelsTheta):
        self.labelsTheta = labelsTheta
        return None

    def setObservationFunction(self, observationFunction):
        self.observationFunction = observationFunction
        return None

    def setTheta0(self, theta0):
        self.theta0 = theta0
        return None

    def setXobservations(self, Xobservations):
        # Configure the inputs
        self.Xobservations = Xobservations
        return None

    def setYobservations(self, Yobservations):
        self.Yobservations = Yobservations
        return None

    def setOptimAlgo(self, optimAlgo):
        # optimAlgo = 1 : Cobyla
        # optimAlgo = 2 : Cobyla + Multi-Start
        # optimAlgo = 3 : scipy.optimize.least_squares
        self.optimAlgo = optimAlgo
        return None

    def setBoundsMin(self, boundsMin):
        self.boundsMin = boundsMin
        return None

    def setBoundsMax(self, boundsMax):
        self.boundsMax = boundsMax
        return None

    # Définit la fonction des résidus
    def residualFunction(self, theta):
        # Calcule les prédictions
        y = self.observationFunction(theta)
        # convertit en array
        y = np.array(y)
        # Calcule les résidus
        r = y - self.Yobservations
        r = r.flatten()
        return r

    def costFunction(self, theta):
        # Fonction coût pour l'assimilation de données
        # Calcule les résidus
        r = self.residualFunction(theta)
        # Fait la somme des carrés
        sumOfSquares = sum(r ** 2)
        return [sumOfSquares]

    def setAlgorithm(self, optimAlgo):
        self.optimAlgo = optimAlgo
        return None

    def getAlgorithmName(self):
        if self.optimAlgo == 1:
            methodName = "Cobyla"
        elif self.optimAlgo == 2:
            methodName = "Multi-Start"
        else:
            methodName = "Least-Squares"
        return methodName

    def run(self):
        # Configure les observations X dans la fonction
        self.observationFunction.setXobservations(self.Xobservations)
        #
        bounds = ot.Interval(self.boundsMin, self.boundsMax)
        dimCalage = len(self.theta0)
        # define the problem
        def internalCostFunction(theta):
            return self.costFunction(theta)

        objective = ot.PythonFunction(dimCalage, 1, internalCostFunction)
        problem = ot.OptimizationProblem(objective)
        problem.setMinimization(True)
        problem.setBounds(bounds)
        #
        if self.optimAlgo == 1:
            maximumIteration = 1000
            algo = ot.Cobyla()
            algo.setProblem(problem)
            algo.setMaximumIterationNumber(maximumIteration)
            algo.setStartingPoint(self.theta0)
            # Lance l'optimisation
            algo.run()
            # retrieve results
            self.result = algo.getResult()
            self.thetaStar = self.result.getOptimalPoint()
            # Get the number of function evaluations
            self.nfeval = objective.getEvaluationCallsNumber()
        elif self.optimAlgo == 2:
            # Tente un multistart
            npoints = 50
            maximumIteration = 100
            problem.setBounds(bounds)
            # solve the problem
            algoLocal = ot.Cobyla()
            algoLocal.setProblem(problem)
            algoLocal.setMaximumIterationNumber(maximumIteration)
            bounds = ot.Interval(self.boundsMin, self.boundsMax)
            algoLocal.setStartingPoint(self.theta0)
            # Génère l'échantillon des points de départ
            startingPoints = ot.Sample(npoints, dimCalage)
            for j in range(dimCalage):
                myUnif = ot.Uniform(self.boundsMin[j], self.boundsMax[j])
                mySampleJ = myUnif.getSample(npoints)
                for i in range(npoints):
                    startingPoints[i, j] = mySampleJ[i, 0]
            # Configure les points de départ
            algo = ot.MultiStart(algoLocal, startingPoints)
            # Lance l'optimisation
            algo.run()
            # retrieve results
            self.result = algo.getResult()
            self.thetaStar = self.result.getOptimalPoint()
            # Get the number of function evaluations
            self.nfeval = objective.getEvaluationCallsNumber()
        else:
            # least_squares
            from scipy.optimize import least_squares

            myBounds = (self.boundsMin, self.boundsMax)

            def internalResidualFunction(theta):
                return self.residualFunction(theta)

            res_LS = least_squares(
                internalResidualFunction, self.theta0, bounds=myBounds
            )
            self.thetaStar = res_LS.x
            self.nfeval = res_LS.nfev
        #
        # Compute Theta distribution by bootstrap,
        # if required
        if self.computeThetaByBootstrap:
            self.computeThetaDistribution()
        return None

    def getThetaStar(self):
        return self.thetaStar

    def drawOptimalValueHistory(self):
        View(self.result.drawOptimalValueHistory())
        return None

    def getNfeval(self):
        return self.nfeval

    def computeBilateralConfidenceInterval(self):
        alpha = (1 - self.confidenceLevel) / 2
        lb = self.thetaStarBootstrap.computeQuantilePerComponent(alpha)
        ub = self.thetaStarBootstrap.computeQuantilePerComponent(1 - alpha)
        c = ot.Interval(lb, ub)
        return c

    def printResult(self):
        # Affiche le coût avant optimisation
        dimCalage = len(self.theta0)
        costInitial = self.costFunction(self.theta0)
        print("Theta0=")
        for i in range(dimCalage):
            print("\t%s 0[%d]=%.3e" % (self.labelsTheta[i], i, self.theta0[i]))
        print("Cout initial=%.3e" % (costInitial[0]))

        # Affiche le nom de la méthode
        methodName = self.getAlgorithmName()
        print("Methode : %s" % (methodName))

        # Affiche l'optimum
        print("Theta*=")
        for i in range(dimCalage):
            print("\t%s *[%d]=%.3e" % (self.labelsTheta[i], i, self.thetaStar[i]))
        # Intervalle de confiance
        print("Confidence interval at level =%.4f" % (self.confidenceLevel))
        c = self.computeBilateralConfidenceInterval()
        lb = c.getLowerBound()
        ub = c.getUpperBound()
        for i in range(dimCalage):
            print("\tTheta[%d] in [%.4e,%.4e]" % (i, lb[i], ub[i]))
        # Coût optimal
        costFinal = self.costFunction(self.thetaStar)
        print("Cout final=%.3e" % (costFinal[0]))
        print("Nombre d'évaluations=%d" % (self.nfeval))
        return None

    def generateBootstrapSample(self):
        # TODO : vectorise ce bazard, malgré OT !
        nbobs = self.Yobservations.getSize()
        bootstrapIndices = np.random.randint(0, nbobs, (nbobs, 1))
        # Génère X
        dimX = self.Xobservations.getDimension()
        XBootstrap = ot.Sample(nbobs, dimX)
        for i in range(nbobs):
            for j in range(dimX):
                k = int(bootstrapIndices[i, 0])
                XBootstrap[i, j] = self.Xobservations[k, j]
        # Génère Y
        YBootstrap = ot.Sample(nbobs, 1)
        for i in range(nbobs):
            k = int(bootstrapIndices[i, 0])
            YBootstrap[i, j] = self.Yobservations[k]
        return XBootstrap, YBootstrap

    def plotThetaDistribution(self, allLabels=None):
        dimCalage = len(self.theta0)
        for i in range(dimCalage):
            histoGraph = (
                ot.HistogramFactory().build(self.thetaStarBootstrap[:, i]).drawPDF()
            )
            histoGraph.setXTitle(self.labelsTheta[i])
            histoGraph.setYTitle("Frequence")
            # Ajouter la normale associée
            factory = ot.NormalFactory()
            loiThetaI = factory.build(self.thetaStarBootstrap[:, i])
            mu = loiThetaI.getParameter()[0]
            sigma = loiThetaI.getParameter()[1]
            pdf_graph = loiThetaI.drawPDF()
            pdf_graph.setColors(["blue"])
            histoGraph.add(pdf_graph)
            histoGraph.setLegends(["Empirique", "Normale"])
            histoGraph.setAutomaticBoundingBox(True)
            histoGraph.setTitle(
                "Bootstrap - N= %d, μ=%.4f, σ=%.4f" % (self.bootstrapSize, mu, sigma)
            )
            View(histoGraph)
        return None

    def computeThetaDistribution(self):
        # Créée une nouvelle fonction d'observation
        # (évite de modifier les observations X de la fonction
        # de calage de self)
        mycfB = DaCalibrationFunction(
            self.observationFunction.modelFunction,
            self.observationFunction.XobservationsIndices,
            self.observationFunction.calibratedIndices,
            self.observationFunction.numberXobservations,
        )
        # Generate a bootstrap sample of the coefficients
        mydaB = DaNLLS()
        mydaB.setTheta0(self.theta0)
        mydaB.setObservationFunction(mycfB)
        mydaB.setBoundsMin(self.boundsMin)
        mydaB.setBoundsMax(self.boundsMax)
        mydaB.setAlgorithm(self.optimAlgo)
        # Disable the distribution computation
        # (otherwise, this creates an infinite loop)
        mydaB.setComputeThetaByBootstrap(False)
        dimCalage = len(self.theta0)
        self.thetaStarBootstrap = ot.Sample(self.bootstrapSize, dimCalage)
        for i in range(self.bootstrapSize):
            XBootstrap, YBootstrap = self.generateBootstrapSample()
            # Résout le problème bootstrap
            mydaB.setXobservations(XBootstrap)
            mydaB.setYobservations(YBootstrap)
            mydaB.run()
            self.thetaStarBootstrap[i, :] = mydaB.getThetaStar()
        return None

    def plotObservationsVsPredictionsBeforeAfter(self):
        dalib.plotObservationsVsPredictionsBeforeAfter(
            self.theta0, self.thetaStar, self.Yobservations, self.observationFunction
        )
        return None

    def plotModelVsDataBeforeAndAFter(self):
        dalib.plotModelVsDataBeforeAndAFter(
            self.theta0,
            self.thetaStar,
            self.observationFunction.Xobservations,
            self.Yobservations,
            self.observationFunction,
        )
        return None

    def plotResidualsAfter(self):
        myGraph = dalib.plotResiduals(
            self.thetaStar, self.Yobservations, self.observationFunction
        )
        return myGraph
