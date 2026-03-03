#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A collection of functions for calibration
"""

import openturns as ot
import numpy as np


def gaussianLinearCalibrationFromCholesky(
    model,
    inputObservations,
    outputObservations,
    candidate,
    parameterCovariance,
    errorCovariance,
    verbose=False,
):
    """
    Compute the solution of the linear Gaussian calibration from Cholesky decomposition.

    Parameters
    ----------
    model : ot.Function
        The function to calibrate.
    inputObservations : ot.Sample(sampleSize, inputDimension)
        The observed inputs.
    outputObservations : ot.Sample(sampleSize, outputDimension)
        The observed outputs.
    candidate : ot.Point(parameterDimension)
        The reference (or initial) parameter value.
    parameterCovariance : ot.CovarianceMatrix(parameterDimension)
        The covariance matrix of the parameter.
    errorCovariance : ot.CovarianceMatrix()
        The covariance of the observations errors.
    verbose : bool
        If True, print intermediate messages.

    Returns
    -------
    calibratedParameter : ot.Point(parameterDimension)
        The MAP estimator of the calibration problem.
    parameterCovariance : ot.CovarianceMatrix(parameterDimension)
        The posterior covariance matrix of the parameter.

    """
    parameterDimension = candidate.getDimension()
    size = inputObservations.getSize()
    # Compute model observations
    model.setParameter(candidate)
    modelObservations = model(inputObservations)
    # Compute residuals
    residuals = outputObservations - modelObservations
    outputDimension = modelObservations.getDimension()
    # Stack the residuals by observation,
    #   deltay = [residuals[i,0],residuals[i,1],...,residuals[i,outputDimension]]
    deltay = ot.Point(size * outputDimension)
    for i in range(size):
        for j in range(outputDimension):
            deltay[i * outputDimension + j] = residuals[i, j]
    # Compute J
    transposedGradientObservations = ot.Matrix(
        parameterDimension, size * outputDimension
    )
    for i in range(size):
        g = model.parameterGradient(inputObservations[i])
        for j in range(outputDimension):
            for k in range(parameterDimension):
                transposedGradientObservations[k, i * outputDimension + j] = g[k, j]
    gradientObservations = transposedGradientObservations.transpose()
    # Compute R
    observationDimension = errorCovariance.getDimension()
    R = ot.CovarianceMatrix(deltay.getSize())
    for i in range(size):
        for j in range(observationDimension):
            for k in range(observationDimension):
                R[
                    i * observationDimension + j, i * observationDimension + k
                ] = errorCovariance[j, k]
    # Create B, R, inv(B), inv(R)
    B = ot.CovarianceMatrix(parameterCovariance)
    LB = B.computeCholesky()
    LR = R.computeCholesky()
    #
    ILB = ot.IdentityMatrix(parameterDimension)
    invLB = LB.solveLinearSystem(ILB)
    invLRJ = LR.solveLinearSystem(gradientObservations)
    # Compute Abar
    Abar = ot.Matrix(parameterDimension + size * outputDimension, parameterDimension)
    Abar[0:parameterDimension, 0:parameterDimension] = invLB
    for i in range(size):
        for j in range(outputDimension):
            for k in range(parameterDimension):
                Abar[i * outputDimension + j + parameterDimension, k] = -invLRJ[
                    i * outputDimension + j, k
                ]
    #
    invLRz = LR.solveLinearSystem(deltay)
    # Compute ybar
    ybar = ot.Point(parameterDimension + size * outputDimension)
    for i in range(size):
        for j in range(outputDimension):
            ybar[i * outputDimension + j + parameterDimension] = -invLRz[
                i * outputDimension + j
            ]
    # Solve the least squares problem
    if verbose:
        print("log10(Cond(Abar))=%.2f" % (np.log10(np.linalg.cond(Abar))))
    method = ot.SVDMethod(Abar)
    parameterDelta = method.solve(ybar)
    calibratedParameter = candidate + parameterDelta
    parameterCovariance = method.getGramInverse()
    return calibratedParameter, parameterCovariance


def squaredMahalanobis(x, y, covariance):
    """
    Squared Mahalanobis distance.

    This is:

        (x - y)' A^{-1} (x - y)

    where A is the covariance matrix.

    Parameters
    ----------
    x : ot.Point(dimension)
        The first point.
    y : ot.Point(dimension)
        The second point.
    covariance : ot.CovarianceMatrix(dimension)
        The covariance matrix.

    Returns
    -------
    m : float
        The Mahalanobis distance.
    """
    delta = x - y
    z = covariance.solveLinearSystem(delta)
    m = delta.dot(z)
    return m


def gaussianCalibrationCostFunction(
    theta,
    candidate,
    parameterCovariance,
    modelObservations,
    gradientObservations,
    outputObservations,
    globalErrorCovariance,
):
    """
    Cost function for linear gaussian calibration.

    This is:

        0.5 * SquaredMahalanobis(y - H(theta), R)
        + 0.5 * SquaredMahalanobis(theta - mu, B)

    where
    y is the vector of output observations,
    H(theta) is the output predictions of the model,
    R is the covariance matrix of the observations errors,
    B is the covariance matrix of the parameter,
    theta is the parameter,
    mu is the mean of the prior distribution of the parameter.

    Reference
    ---------
    http://openturns.github.io/openturns/master/theory/data_analysis/gaussian_calibration.html

    Parameters
    ----------
    theta : ot.Point(parameterDimension)
        The value of the parameter.
    candidate : ot.Point(parameterDimension)
        The mean of the prior Gaussian distribution (i.e. the background).
    parameterCovariance : ot.CovarianceMatrix(parameterDimension)
        The covariance matrix of the prior Gaussian distribution.
    modelObservations : ot.Sample(numberOfObservations, 1)
        The sample of outputs of the model.
    gradientObservations : ot.Matrix(numberOfObservations, parameterDimension)
        The Jacobian matrix of the model with respect to the parameter.
    outputObservations : ot.Sample(numberOfObservations, 1)
        The sample of observed outputs.
    globalErrorCovariance : ot.CovarianceMatrix(numberOfObservations)
        The global covariance matrix of the Gaussian distribution of the observations.

    Returns
    -------
    list(J, Jb, Jo)
        J = Jb + Jo : float, the value of the cost function.
        Jb : float, the part of the cost function from background.
        Jo : float, the part of the cost function from observations.
    """
    # 1. Background part of the cost function
    Jb = 0.5 * squaredMahalanobis(candidate, theta, parameterCovariance)
    # 2. Observation part of the cost function
    deltaTheta = theta - candidate
    # Observation values of the model
    YfunPoint = modelObservations.asPoint()
    # Predicted values of the model based on linearization
    Ypredictions = YfunPoint + gradientObservations * deltaTheta
    YobsPoint = outputObservations.asPoint()
    Jo = 0.5 * squaredMahalanobis(YobsPoint, Ypredictions, globalErrorCovariance)
    # 3. Sum of the two parts
    J = Jb + Jo
    return [J, Jb, Jo]


def gaussianLinearCalibrationFromKalman(
    model,
    inputObservations,
    outputObservations,
    candidate,
    parameterCovariance,
    errorCovariance,
    verbose=False,
):
    """
    Compute the solution of the linear Gaussian calibration from Kalman matrix.

    Parameters
    ----------
    model : ot.Function
        The function to calibrate.
    inputObservations : ot.Sample(sampleSize, inputDimension)
        The observed inputs.
    outputObservations : ot.Sample(sampleSize, outputDimension)
        The observed outputs.
    candidate : ot.Point(parameterDimension)
        The reference (or initial) parameter value.
    parameterCovariance : ot.CovarianceMatrix(parameterDimension)
        The covariance matrix of the parameter.
    errorCovariance : ot.CovarianceMatrix()
        The covariance of the observations errors.
    verbose : bool
        If True, print intermediate messages.

    Returns
    -------
    calibratedParameter : ot.Point(parameterDimension)
        The MAP estimator of the calibration problem.
    parameterCovariance : ot.CovarianceMatrix(parameterDimension)
        The posterior covariance matrix of the parameter.

    """
    parameterDimension = candidate.getDimension()
    size = inputObservations.getSize()
    # Compute model observations
    model.setParameter(candidate)
    modelObservations = model(inputObservations)
    # Compute residuals
    residuals = outputObservations - modelObservations
    outputDimension = modelObservations.getDimension()
    # Stack the residuals by observation,
    #   deltay = [residuals[i,0],residuals[i,1],...,residuals[i,outputDimension]]
    deltay = ot.Point(size * outputDimension)
    for i in range(size):
        for j in range(outputDimension):
            deltay[i * outputDimension + j] = residuals[i, j]
    # Compute J
    transposedGradientObservations = ot.Matrix(
        parameterDimension, size * outputDimension
    )
    for i in range(size):
        g = model.parameterGradient(inputObservations[i])
        for j in range(outputDimension):
            for k in range(parameterDimension):
                transposedGradientObservations[k, i * outputDimension + j] = g[k, j]
    gradientObservations = transposedGradientObservations.transpose()
    if verbose:
        print(
            "log10(Cond(Jacobian))=%.2f"
            % (np.log10(np.linalg.cond(gradientObservations)))
        )
    # Compute R
    observationDimension = errorCovariance.getDimension()
    R = ot.CovarianceMatrix(deltay.getSize())
    for i in range(size):
        for j in range(observationDimension):
            for k in range(observationDimension):
                R[
                    i * observationDimension + j, i * observationDimension + k
                ] = errorCovariance[j, k]
    # Compute B, inverse of B
    B = ot.CovarianceMatrix(parameterCovariance)
    IB = ot.IdentityMatrix(parameterDimension)
    invB = B.solveLinearSystem(IB)
    # Compute inverse of R
    IR = ot.IdentityMatrix(R.getNbRows())
    invR = R.solveLinearSystem(IR)
    #
    C = gradientObservations.transpose() * invR
    invA = invB + C * gradientObservations
    K = invA.solveLinearSystem(C)
    if verbose:
        print("log10(Cond(Kalman))=%.2f" % (np.log10(np.linalg.cond(K))))
    thetaStar = candidate + K * deltay
    #
    L = IB - K * gradientObservations
    covarianceThetaStar = K * R * K.transpose() + L * B * L.transpose()
    covarianceThetaStar = ot.CovarianceMatrix(covarianceThetaStar)
    return thetaStar, covarianceThetaStar


def gaussianLinearCalibrationFromCholeskyFixed(
    model,
    inputObservations,
    outputObservations,
    candidate,
    parameterCovariance,
    errorCovariance,
    verbose=False,
):
    """
    Compute the solution of the linear Gaussian calibration from Cholesky decomposition.

    Parameters
    ----------
    model : ot.Function
        The function to calibrate.
    inputObservations : ot.Sample(sampleSize, inputDimension)
        The observed inputs.
    outputObservations : ot.Sample(sampleSize, outputDimension)
        The observed outputs.
    candidate : ot.Point(parameterDimension)
        The reference (or initial) parameter value.
    parameterCovariance : ot.CovarianceMatrix(parameterDimension)
        The covariance matrix of the parameter.
    errorCovariance : ot.CovarianceMatrix()
        The covariance of the observations errors.
    verbose : bool
        If True, print intermediate messages.

    Returns
    -------
    calibratedParameter : ot.Point(parameterDimension)
        The MAP estimator of the calibration problem.
    parameterCovariance : ot.CovarianceMatrix(parameterDimension)
        The posterior covariance matrix of the parameter.

    """
    parameterDimension = candidate.getDimension()
    size = inputObservations.getSize()
    # Compute model observations
    model.setParameter(candidate)
    modelObservations = model(inputObservations)
    # Compute residuals
    residuals = outputObservations - modelObservations
    outputDimension = modelObservations.getDimension()
    # Stack the residuals by observation,
    #   deltay = [residuals[i,0],residuals[i,1],...,residuals[i,outputDimension]]
    deltay = ot.Point(size * outputDimension)
    for i in range(size):
        for j in range(outputDimension):
            deltay[i * outputDimension + j] = residuals[i, j]
    # Compute J
    transposedGradientObservations = ot.Matrix(
        parameterDimension, size * outputDimension
    )
    for i in range(size):
        g = model.parameterGradient(inputObservations[i])
        for j in range(outputDimension):
            for k in range(parameterDimension):
                transposedGradientObservations[k, i * outputDimension + j] = g[k, j]
    gradientObservations = transposedGradientObservations.transpose()
    # Compute R
    observationDimension = errorCovariance.getDimension()
    R = ot.CovarianceMatrix(deltay.getSize())
    for i in range(size):
        for j in range(observationDimension):
            for k in range(observationDimension):
                R[
                    i * observationDimension + j, i * observationDimension + k
                ] = errorCovariance[j, k]
    # Create B, R, inv(B), inv(R)
    B = ot.CovarianceMatrix(parameterCovariance)
    LB = B.computeCholesky()
    LR = R.computeCholesky()
    #
    ILB = ot.IdentityMatrix(parameterDimension)
    invLB = LB.solveLinearSystem(ILB)
    invLRJ = LR.solveLinearSystem(gradientObservations)
    # Compute Abar
    Abar = ot.Matrix(parameterDimension + size * outputDimension, parameterDimension)
    Abar[0:parameterDimension, 0:parameterDimension] = invLB
    for i in range(size):
        for j in range(outputDimension):
            for k in range(parameterDimension):
                Abar[i * outputDimension + j + parameterDimension, k] = -invLRJ[
                    i * outputDimension + j, k
                ]
    #
    invLRz = LR.solveLinearSystem(deltay)
    # Compute ybar
    ybar = ot.Point(parameterDimension + size * outputDimension)
    for i in range(size):
        for j in range(outputDimension):
            ybar[i * outputDimension + j + parameterDimension] = -invLRz[
                i * outputDimension + j
            ]
    # Solve the least squares problem
    if verbose:
        print("log10(Cond(Abar))=%.2f" % (np.log10(np.linalg.cond(Abar))))
    method = ot.SVDMethod(Abar)
    parameterDelta = method.solve(ybar)
    calibratedParameter = candidate + parameterDelta
    varianceOutput = deltay.normSquare() / (size - parameterDimension)
    if verbose:
        print("varianceOutput=", varianceOutput)
    parameterCovariance = ot.CovarianceMatrix(
        np.array(varianceOutput * method.getGramInverse())
    )
    return calibratedParameter, parameterCovariance


def computeJacobianParameterMatrix(model, inputObservations):
    """
    Compute the Jacobian matrix of the model with respect to the parameters.

    Parameters
    ----------
    model : ot.Function
        The function to calibrate.
    inputObservations : ot.Sample(sampleSize, inputDimension)
        The input observations.

    Returns
    -------
    gradientObservations : ot.Matrix(sampleSize, parameterDimension)
        The jacobian matrix of the model w.r. to the parameters.

    """
    parameterDimension = model.getParameterDimension()
    outputDimension = model.getOutputDimension()
    sampleSize = inputObservations.getSize()
    transposedGradientObservations = ot.Matrix(
        parameterDimension, sampleSize * outputDimension
    )
    for i in range(sampleSize):
        g = model.parameterGradient(inputObservations[i])
        for j in range(outputDimension):
            for k in range(parameterDimension):
                transposedGradientObservations[k, i * outputDimension + j] = g[k, j]
    gradientObservations = transposedGradientObservations.transpose()
    return gradientObservations


def computeGlobalObservationsCovarianceMatrixFromLocalCovariance(
    numberOfObservations, errorCovariance
):
    """
    Compute the global covariance matrix of the observations from the local.

    Parameters
    ----------
    numberOfObservations : int
        The number of observations.
    errorCovariance : ot.Covariance(outputDimension)
        The local covariance matrix of the observations.

    Returns
    -------
    globalCovarianceMatrix : ot.Covariance(outputDimension * observationDimension)
        The global covariance matrix of the observations.

    """
    observationDimension = errorCovariance.getDimension()
    globalCovarianceMatrix = ot.CovarianceMatrix(
        numberOfObservations * observationDimension
    )
    for i in range(numberOfObservations):
        for j in range(observationDimension):
            for k in range(observationDimension):
                globalCovarianceMatrix[
                    i * observationDimension + j, i * observationDimension + k
                ] = errorCovariance[j, k]
    return globalCovarianceMatrix


def computeKalmanMatrix(
    model,
    inputObservations,
    candidate,
    parameterCovariance,
    errorCovariance,
    verbose=False,
):
    """
    Compute the Kalman matrix.

    Parameters
    ----------
    model : ot.Function
        The function to calibrate.
    inputObservations : ot.Sample(sampleSize, inputDimension)
        The observed inputs.
    candidate : ot.Point(parameterDimension)
        The reference (or initial) parameter value.
    parameterCovariance : ot.CovarianceMatrix(parameterDimension)
        The covariance matrix of the parameter.
    errorCovariance : ot.CovarianceMatrix()
        The covariance of the observations errors.
    verbose : bool
        If True, print intermediate messages.

    Returns
    -------
    K : ot.Matrix(parameterDimension, size * outputDimension)
        The Kalman matrix.
    """
    parameterDimension = candidate.getDimension()
    size = inputObservations.getSize()
    if verbose:
        print("size = ", size)
    # Compute model observations
    model.setParameter(candidate)
    outputDimension = model.getOutputDimension()
    if verbose:
        print("outputDimension = ", outputDimension)
    # Compute J
    transposedGradientObservations = ot.Matrix(
        parameterDimension, size * outputDimension
    )
    for i in range(size):
        g = model.parameterGradient(inputObservations[i])
        for j in range(outputDimension):
            for k in range(parameterDimension):
                transposedGradientObservations[k, i * outputDimension + j] = g[k, j]
    gradientObservations = transposedGradientObservations.transpose()
    if verbose:
        print(
            "log10(Cond(Jacobian))=%.2f"
            % (np.log10(np.linalg.cond(gradientObservations)))
        )
    # Compute R
    observationDimension = errorCovariance.getDimension()
    if verbose:
        print("observationDimension = ", observationDimension)
    R = ot.CovarianceMatrix(size * outputDimension)
    for i in range(size):
        for j in range(observationDimension):
            for k in range(observationDimension):
                R[
                    i * observationDimension + j, i * observationDimension + k
                ] = errorCovariance[j, k]
    # Compute B, inverse of B
    B = ot.CovarianceMatrix(parameterCovariance)
    IB = ot.IdentityMatrix(parameterDimension)
    invB = B.solveLinearSystem(IB)
    # Compute inverse of R
    IR = ot.IdentityMatrix(size * outputDimension)
    invR = R.solveLinearSystem(IR)
    #
    C = gradientObservations.transpose() * invR
    invA = invB + C * gradientObservations
    K = invA.solveLinearSystem(C)
    return K
