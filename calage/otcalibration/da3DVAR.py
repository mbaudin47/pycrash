# -*- coding: utf-8 -*-
# Copyright (C) 2018 - Michael Baudin

import openturns as ot
import numpy as np
from openturns.viewer import View
from . import dalib

"""
Une librairie pour l'assimilation de données/
* 3DVar
"""


class Da3DVAR:
    def __init__(self):
        self.thetaB = None
        self.Xobservations = None
        self.Yobservations = None
        self.covarianceTheta = None
        self.covarianceY = None
        self.optimAlgo = 1
        self.nfeval = None
        self.boundsMin = None
        self.boundsMax = None
        self.observationFunction = None
        self.sigmaTheta = None
        self.sigmaY = None
        self.labelsTheta = None

    def setLabelsTheta(self, labelsTheta):
        self.labelsTheta = labelsTheta
        return None

    def setObservationFunction(self, observationFunction):
        self.observationFunction = observationFunction
        return None

    def setThetaB(self, thetaB):
        self.thetaB = thetaB
        return None

    def setXobservations(self, Xobservations):
        # Configure the inputs
        self.Xobservations = Xobservations
        return None

    def setYobservations(self, Yobservations):
        self.Yobservations = Yobservations
        return None

    def setSigmaTheta(self, sigmaTheta):
        self.sigmaTheta = sigmaTheta
        self.computeSigmaTheta()
        return None

    def computeSigmaTheta(self):
        # sigmaTheta : une liste d'écarts-types
        dimCalage = len(self.sigmaTheta)
        self.covarianceTheta = ot.CovarianceMatrix(dimCalage)
        for i in range(dimCalage):
            varianceThetaI = self.sigmaTheta[i] ** 2
            self.covarianceTheta[i, i] = varianceThetaI
        return None

    def setSigmaY(self, sigmaY):
        self.sigmaY = sigmaY
        self.computeSigmaY()
        return None

    def computeSigmaY(self):
        nbobs = len(self.Yobservations)
        varianceY = self.sigmaY ** 2
        covarianceDiagonale = varianceY * np.eye(nbobs)
        self.covarianceY = ot.CovarianceMatrix(covarianceDiagonale)
        return None

    def setCovarianceY(self, covarianceY):
        self.covarianceY = covarianceY
        return None

    def setOptimAlgo(self, optimAlgo):
        self.optimAlgo = optimAlgo
        return None

    def setBoundsMin(self, boundsMin):
        self.boundsMin = boundsMin
        return None

    def setBoundsMax(self, boundsMax):
        self.boundsMax = boundsMax
        return None

    def setDiagonalSigmaY(self, sigmaY):
        nbobs = len(self.Yobservations)
        varianceY = sigmaY ** 2
        covarianceDiagonale = varianceY * np.eye(nbobs)
        self.covarianceY = ot.CovarianceMatrix(covarianceDiagonale)
        return None

    def setDiagonalSigmaTheta(self, sigmaTheta):
        # sigmaTheta : une liste d'écarts-types
        dimCalage = len(sigmaTheta)
        self.covarianceTheta = ot.CovarianceMatrix(dimCalage)
        for i in range(dimCalage):
            varianceThetaI = sigmaTheta[i] ** 2
            self.covarianceTheta[i, i] = varianceThetaI
        return None

    def squaredMahalanobis(self, x, y, covariance):
        # Squared Mahalanobis distance
        # TODO : voir si il est plus malin de résoudre le système d'équations linéaires ou
        # bien d'inverser directement la matrice de covariance.
        # (puisqu'elle est définie positive, il n'y a pas de permutation des lignes)
        # TODO : voir si l'utilisation des Hmat apporte quelque chose.
        x = ot.Point(np.ravel(np.array(x)))
        y = ot.Point(np.ravel(np.array(y)))
        delta = x - y
        z = covariance.solveLinearSystem(delta)
        m = delta.dot(z)
        return m

    def costFunction(self, theta):
        # Fonction coût pour l'assimilation de données
        # theta : les paramètres à caler
        # monModelePhysique : le modèle
        # Yobservations : les observations
        # thetaB : l'ébauche
        # BI : l'inverse de la matrice de covariance de l'ébauche
        # RI : l'inverse de la matrice de covariance des observations
        # Résolution de la partie Ebauche
        Jb = 0.5 * self.squaredMahalanobis(self.thetaB, theta, self.covarianceTheta)
        # Résolution de la partie Observations
        Ypredictions = self.observationFunction.obsFunction(theta)
        Jo = 0.5 * self.squaredMahalanobis(
            self.Yobservations, Ypredictions, self.covarianceY
        )
        # Somme les deux composantes
        J = Jb + Jo
        return [J]

    def setAlgorithm(self, optimAlgo):
        self.optimAlgo = optimAlgo
        return None

    def getAlgorithmName(self):
        if self.optimAlgo == 1:
            methodName = "Cobyla"
        else:
            methodName = "Multi-Start"
        return methodName

    def run(self):
        bounds = ot.Interval(self.boundsMin, self.boundsMax)
        dimCalage = len(self.thetaB)
        self.observationFunction.setXobservations(self.Xobservations)

        # TODO : implémenter le calcul du gradient
        # la dérivée d'une somme de carrés est exacte,
        # seule la dérivée de logisticSolution par rapport à a et b
        # est à faire par diff. finies.
        #
        def internalCostFunction(theta):
            return self.costFunction(theta)

        # define the problem
        objective = ot.PythonFunction(dimCalage, 1, internalCostFunction)
        problem = ot.OptimizationProblem(objective)
        problem.setMinimization(True)
        problem.setBounds(bounds)
        if self.optimAlgo == 1:
            # solve the problem
            maximumIteration = 100
            algo = ot.NLopt(problem, "LD_LBFGS")
            algo.setMaximumIterationNumber(maximumIteration)
            algo.setStartingPoint(self.thetaB)
            algo.run()
            # retrieve results
            self.result = algo.getResult()
            self.thetaStar = self.result.getOptimalPoint()
        else:
            # Tente un multistart
            npoints = 50
            maximumIteration = 100
            problem.setBounds(bounds)
            #
            algo = ot.NLopt(problem, "LD_LBFGS")
            algo.setMaximumIterationNumber(maximumIteration)
            algo.setStartingPoint(self.thetaB)
            #
            # Génère l'échantillon des points de départ
            startingPoints = ot.Sample(npoints, dimCalage)
            for j in range(dimCalage):
                myUnif = ot.Uniform(self.boundsMin[j], self.boundsMax[j])
                mySampleJ = myUnif.getSample(npoints)
                for i in range(npoints):
                    startingPoints[i, j] = mySampleJ[i, 0]
            # Configure les points de départ
            algoMS = ot.MultiStart(algo, startingPoints)
            algoMS.run()
            # retrieve results
            self.result = algoMS.getResult()
            self.thetaStar = self.result.getOptimalPoint()
        self.nfeval = objective.getEvaluationCallsNumber()
        return None

    def getThetaStar(self):
        return self.thetaStar

    def drawOptimalValueHistory(self):
        View(self.result.drawOptimalValueHistory())
        return None

    def getNfeval(self):
        return self.nfeval

    def printResult(self):
        # Affiche le coût avant optimisation
        dimCalage = len(self.thetaB)
        print("ThetaB=")
        for i in range(dimCalage):
            print("\t%s B[%d]=%f" % (self.labelsTheta[i], i, self.thetaB[i]))
        costInitial = self.costFunction(self.thetaB)
        print("Cout initial=%.4e" % (costInitial[0]))

        # Affiche le nom de la méthode
        methodName = self.getAlgorithmName()
        print("Methode : %s" % (methodName))
        # Affiche l'optimum
        print("Theta*=")
        for i in range(dimCalage):
            print("\t%s *[%d]=%f" % (self.labelsTheta[i], i, self.thetaStar[i]))
        costFinal = self.costFunction(self.thetaStar)
        print("Cout final=%.4e" % (costFinal[0]))
        print("Nombre d'évaluations=%d" % (self.nfeval))
        return None

    def plotObservationsVsPredictionsBeforeAfter(self):
        dalib.plotObservationsVsPredictionsBeforeAfter(
            self.thetaB, self.thetaStar, self.Yobservations, self.observationFunction
        )
        return None

    def plotModelVsDataBeforeAndAFter(self):
        dalib.plotModelVsDataBeforeAndAFter(
            self.thetaB,
            self.thetaStar,
            self.Xobservations,
            self.Yobservations,
            self.observationFunction,
        )
        return None

    def plotResidualsAfter(self):
        myGraph = dalib.plotResiduals(
            self.thetaStar, self.Yobservations, self.observationFunction, self.sigmaY
        )
        return myGraph

    def plotPrior(self):
        # Plot the distribution of the prior vs posterior
        dimCalage = len(self.thetaB)
        for iTheta in range(dimCalage):
            mu = self.thetaB[iTheta]
            sigma = np.sqrt(self.covarianceTheta[iTheta, iTheta])
            lawPrior = ot.Normal(mu, sigma)
            myGraph = lawPrior.drawPDF()
            myGraph.setXTitle("%s" % (self.labelsTheta[iTheta]))
            myGraph.setLegends([""])
            myGraph.setTitle("Prior")
            View(myGraph).show()
            # TODO : estimer la loi à posteriori par MCMC
        return None
