# -*- coding: utf-8 -*-
# Copyright (C) 2018 - Michael Baudin

import openturns as ot
import numpy as np
from openturns.viewer import View

"""
Une librairie pour l'assimilation de données/
BLUE
"""


class DaBLUE:
    def __init__(self):
        self.thetaB = None
        self.Yobservations = None
        self.covarianceTheta = None
        self.covarianceY = None
        self.K = None
        self.Xobservations = None
        self.sigmaTheta = None
        self.sigmaY = None
        self.labelsTheta = None
        self.observationFunctionValue = None
        self.observationGradientValue = None
        self.confidenceLevel = 0.95

    def setConfidenceLevel(self, confidenceLevel):
        self.confidenceLevel = confidenceLevel
        return None

    def setLabelsTheta(self, labelsTheta):
        self.labelsTheta = labelsTheta
        return None

    def setObservationFunctionValue(self, observationFunctionValue):
        self.observationFunctionValue = observationFunctionValue
        return None

    def setObservationGradientValue(self, observationGradientValue):
        self.observationGradientValue = observationGradientValue
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

    def squaredMahalanobis(self, x, y, covariance):
        # Squared Mahalanobis distance
        # TODO : voir si il est plus malin de résoudre le système d'équations linéaires ou
        # bien d'inverser directement la matrice de covariance.
        # (puisqu'elle est définie positive, il n'y a pas de permutation des lignes)
        # TODO : voir si l'utilisation des Hmat apporte quelque chose.
        x = np.array(x)
        y = np.array(y)
        delta = x - y
        delta = delta.flatten()
        z = covariance.solveLinearSystem(delta)
        m = np.dot(delta, z)
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
        # L'hypothèse est que le modèle est linéaire
        deltaTheta = theta - self.thetaB
        YfunPoint = ot.Point(np.array(self.observationFunctionValue).flatten())
        Ypredictions = (
            YfunPoint + self.observationGradientValue.transpose() * deltaTheta
        )
        YobsPoint = ot.Point(np.array(self.Yobservations).flatten())
        Jo = 0.5 * self.squaredMahalanobis(YobsPoint, Ypredictions, self.covarianceY)
        # Somme les deux composantes
        J = Jb + Jo
        return [J]

    def run(self):
        # Résout le BLUE
        dimCalage = len(self.thetaB)
        H = np.array(self.observationGradientValue)
        H = H.T
        # Calcule l'inverse des matrices de covariance (brutal !)
        B = np.array(self.covarianceTheta)
        IB = np.linalg.inv(B)
        R = np.array(self.covarianceY)
        IR = np.linalg.inv(R)
        # Voir Eq 4.11, page 22 dans la revue
        Bplus = IB + np.dot(H.T, np.dot(IR, H))
        IBplus = np.linalg.inv(Bplus)
        self.K = np.dot(IBplus, np.dot(H.T, IR))
        """
        Si H est linéaire, on a y = np.dot(J,thetaB)
        Si H est affine, on a H(t) = H(thetaB) + np.dot(J,t-thetaB)
        où J est la Jacobienne de H
        """
        YobsPoint = ot.Point(np.array(self.Yobservations).flatten())
        YfunPoint = ot.Point(np.array(self.observationFunctionValue).flatten())
        deltaY = YobsPoint - YfunPoint
        deltaTheta = np.dot(self.K, deltaY)
        self.thetaStar = self.thetaB + deltaTheta
        # Calcule la matrice de covariance de theta
        # Voir eq 4.7, page 22, dans la revue
        I = np.eye(dimCalage)
        L = I * np.dot(self.K, H)
        self.covarianceThetaStar = np.dot(L, np.dot(B, L.T)) + np.dot(
            self.K, np.dot(R, self.K.T)
        )
        return None

    def getThetaStar(self):
        return self.thetaStar

    def getCovarianceThetaStar(self):
        return self.covarianceThetaStar

    def getK(self):
        return self.K

    def plotThetaDistribution(self, allLabels=None):
        # Plot the distribution of theta
        dimCalage = len(self.thetaStar)
        for i in range(dimCalage):
            # Loi à posteriori
            thetaPosterior = ot.Normal(
                self.thetaStar[i], np.sqrt(self.covarianceThetaStar[i, i])
            )
            myGraph = thetaPosterior.drawPDF()
            myGraph.setColors(["green"])
            # Loi à priori
            thetaPrior = ot.Normal(self.thetaB[i], np.sqrt(self.covarianceTheta[i, i]))
            pg = thetaPrior.drawPDF()
            pg.setColors(["red"])
            myGraph.add(pg)
            #
            myGraph.setXTitle(self.labelsTheta[i])
            myGraph.setYTitle("PDF")
            myGraph.setLegends(["Posterior", "Prior"])
            myGraph.setTitle("Theta PDF")
            View(myGraph)
        return None

    def computeBilateralConfidenceInterval(self):
        dimCalage = len(self.thetaB)
        lb = ot.Point(dimCalage)
        ub = ot.Point(dimCalage)
        for i in range(dimCalage):
            Theta_i = ot.Normal(
                self.thetaStar[i], np.sqrt(self.covarianceThetaStar[i, i])
            )
            ci = Theta_i.computeBilateralConfidenceInterval(self.confidenceLevel)
            lb[i] = ci.getLowerBound()[0]
            ub[i] = ci.getUpperBound()[0]
        c = ot.Interval(lb, ub)
        return c

    def printResult(self):
        # Affiche le coût avant optimisation
        dimCalage = len(self.thetaB)
        print("ThetaB=")
        for i in range(dimCalage):
            print("%s B[%d]=%f" % (self.labelsTheta[i], i, self.thetaB[i]))
        costInitial = self.costFunction(self.thetaB)
        print("Cout initial=%.4e" % (costInitial[0]))
        # Affiche l'optimum
        print("Theta*=")
        for i in range(dimCalage):
            print("%s *[%d]=%f" % (self.labelsTheta[i], i, self.thetaStar[i]))
        costFinal = self.costFunction(self.thetaStar)
        print("Cout final=%.4e" % (costFinal[0]))
        if costFinal[0] > costInitial[0]:
            print("!Attention : le coût a augmenté!")
        # Intervalle de confiance
        print("Confidence interval at level =%.4f" % (self.confidenceLevel))
        c = self.computeBilateralConfidenceInterval()
        lb = c.getLowerBound()
        ub = c.getUpperBound()
        for i in range(dimCalage):
            print("\tTheta[%d] in [%.4e,%.4e]" % (i, lb[i], ub[i]))
        return None
