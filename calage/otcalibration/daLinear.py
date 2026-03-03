# -*- coding: utf-8 -*-
# Copyright (C) 2018 - Michael Baudin

import openturns as ot
import numpy as np
from openturns.viewer import View

"""
Une librairie pour l'assimilation de données/
Moindres carrés linéaires
"""


class DaLinear:
    def __init__(self):
        self.thetaB = None  # Point de référence pour calculer la matrice Jacobienne
        self.Yobservations = None  # Vecteur des observations
        self.covarianceY = None  # Covariance des erreurs d'observation
        self.mymodel = None
        self.sigmaY = None  # Ecart-type des erreurs d'observation
        self.labelsTheta = None
        self.covarianceThetaStar = None
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

    def setYobservations(self, Yobservations):
        self.Yobservations = Yobservations
        return None

    def getSigmaY(self):
        return self.sigmaY

    def costFunction(self, theta):
        # Fonction coût pour le calage linéaire
        # theta : les paramètres à caler
        # monModelePhysique : le modèle
        # Yobservations : les observations
        # thetaB : l'ébauche
        deltaTheta = theta - self.thetaB
        YfunPoint = ot.Point(np.array(self.observationFunctionValue).flatten())
        Ypredictions = (
            YfunPoint + self.observationGradientValue.transpose() * deltaTheta
        )
        YobsPoint = ot.Point(np.array(self.Yobservations).flatten())
        r = YobsPoint - Ypredictions  # Résidu
        C = 0.5 * np.linalg.norm(r, 2) ** 2
        return [C]

    def run(self):
        # Résout le problème de moindres carrés linéaires
        dimCalage = len(self.thetaB)
        nbobs = len(self.Yobservations)
        H = self.observationGradientValue.transpose()
        """
        On a H(theta) = H(thetaB) + np.dot(J,theta-thetaB)
        où J est la Jacobienne de H
        """
        # Convertit en Point
        YfunPoint = ot.Point(np.array(self.observationFunctionValue).flatten())
        Yobservations = ot.Point(np.array(self.Yobservations).flatten())
        deltaY = Yobservations - YfunPoint
        """
        LMF = ot.LinearModelFactory()
        plante car pas ne n'ai apparemment pas R sur le poste.
        En attendant, méthode QR.
        """
        Q1, R1 = H.computeQR()
        z = Q1.transpose() * deltaY
        deltaTheta = R1.solveLinearSystem(z)
        self.thetaStar = self.thetaB + np.array(deltaTheta)
        # Calcul des résidus
        Ypredictions = (
            YfunPoint + self.observationGradientValue.transpose() * deltaTheta
        )
        r = Yobservations - Ypredictions
        # Estimateur de l'écart-type de l'erreur d'observation
        self.sigmaY = np.linalg.norm(r, 2) / np.sqrt(nbobs - dimCalage)
        # Calcule l'inverse de la matrice de Gram
        invGramBySVD = True
        if invGramBySVD:
            S, U, VT = H.computeSVD()
            p = len(self.thetaStar)
            Sm = ot.Matrix(p, p)
            for i in range(p):
                Sm[i, i] = 1.0 / S[i] ** 2
            invGram = VT.transpose() * Sm * VT
        else:
            JTJ = H.transpose() * H
            invGram = np.linalg.inv(JTJ)
        self.covarianceThetaStar = self.sigmaY ** 2 * invGram
        return None

    def getThetaStar(self):
        return self.thetaStar

    def getCovarianceThetaStar(self):
        return self.covarianceThetaStar

    def plotThetaDistribution(self, allLabels=None):
        # Plot the distribution of theta
        dimCalage = len(self.thetaB)
        for i in range(dimCalage):
            # Loi à posteriori
            thetaPosterior = ot.Normal(
                self.thetaStar[i], np.sqrt(self.covarianceThetaStar[i, i])
            )
            myGraph = thetaPosterior.drawPDF()
            myGraph.setColors(["green"])
            #
            myGraph.setXTitle(self.labelsTheta[i])
            myGraph.setYTitle("PDF")
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
        print("Number of observations=%d" % (len(self.Yobservations)))
        print("Number of parameters=%d" % (len(self.thetaB)))
        dimCalage = len(self.thetaB)
        print("ThetaB=")
        for i in range(dimCalage):
            print("\t%s B[%d]=%.4e" % (self.labelsTheta[i], i, self.thetaB[i]))
        print("SigmaY=%.4e" % (self.sigmaY))
        costInitial = self.costFunction(self.thetaB)
        print("Cout initial=%.4e" % (costInitial[0]))
        # Affiche l'optimum
        print("Theta*=")
        for i in range(dimCalage):
            sigmaBetai = np.sqrt(self.covarianceThetaStar[i, i])
            print(
                "\t%s*[%d]=%.4e +- %.4e"
                % (self.labelsTheta[i], i, self.thetaStar[i], sigmaBetai)
            )
        costFinal = self.costFunction(self.thetaStar)
        # Intervalle de confiance
        print("Confidence interval at level =%.4f" % (self.confidenceLevel))
        c = self.computeBilateralConfidenceInterval()
        lb = c.getLowerBound()
        ub = c.getUpperBound()
        for i in range(dimCalage):
            print("\tTheta[%d] in [%.4e,%.4e]" % (i, lb[i], ub[i]))
        #
        print("Cout final=%.4e" % (costFinal[0]))
        if costFinal[0] > costInitial[0]:
            print("!Attention : le coût a augmenté!")
        return None
