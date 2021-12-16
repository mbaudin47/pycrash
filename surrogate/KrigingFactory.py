#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
An attempt to make the kriging useful in practice. 
"""
import openturns as ot
import scipy.spatial
import numpy as np

class KrigingFactory():
    def __init__(self, X_train, Y_train, covarianceModel, basis):
        """
        Create a kriging algorithm. 

        Parameters
        ----------
        X_train : ot.Sample
            The input DOE.
        Y_train : ot.Sample
            The output DOE.
        covarianceModel : ot.CovarianceModel
            The covariance model.
        basis : ot.Basis
            The trend.

        Returns
        -------
        None.

        """
        self.X_train = X_train
        self.Y_train = Y_train
        self.covarianceModel = covarianceModel
        self.basis = basis
        return

    def buildFromBounds(self, minimum_scale_bounds, maximum_scale_bounds,
                        number_of_multistart_points = 20):
        """
        Create a KrigingAlgorithm using correlation bounds.

        We set the initial value of the scale parameter of the 
        covariance model so that the initial point of the optimization 
        algorithm is the upper bound. 

        We set the bounds of the optimization algorithm for the 
        correlation lengths of the covariance model to the 
        given lower and upper bounds. 

        We use a MultiStart algorithm with the given number of points 
        as starting points for the optimization of the correlation lengths.
        The starting points are created from a Sobol' sequence.

        Parameters
        ----------
        minimum_scale_bounds : ot.Point(input_dimension)
            The lower bounds of the correlation length.
        maximum_scale_bounds : ot.Point(input_dimension)
            The upper bounds of the correlation length.
        number_of_multistart_points : int, optional
            The number of points in the MultiStart. The default is 20.

        Returns
        -------
        algorithm : ot.KrigingAlgorithm
            A kriging algorithm.

        References
        ----------
        Numerical issues in maximum likelihood parameter 
        estimation for Gaussian process regression. 
        S. Basak, S. Petit, J. Bect, E. Vasquez.
        (2021)
        """
        self.covarianceModel.setScale(minimum_scale_bounds)  # Trick A
        algorithm = ot.KrigingAlgorithm(self.X_train, self.Y_train, self.covarianceModel,
                                        self.basis)
        print("Set bounds of scale optimization.")
        scaleOptimizationBounds = ot.Interval(minimum_scale_bounds, maximum_scale_bounds)
        algorithm.setOptimizationBounds(scaleOptimizationBounds)  # Trick B
        # Configure Multistart
        solver = algorithm.getOptimizationAlgorithm()
        solverImplementation = solver.getImplementation()
        local_solver = solverImplementation.getClassName()
        print("Local solver=", local_solver)
        sequence = ot.SobolSequence()
        restart = True
        # Create distribution
        distribution_collection = []
        input_dimension = self.X_train.getDimension()
        for i in range(input_dimension):
            marginal = ot.Uniform(minimum_scale_bounds[i], maximum_scale_bounds[i])
            distribution_collection.append(marginal)
        resample_distribution = ot.ComposedDistribution(distribution_collection)
        experiment = ot.LowDiscrepancyExperiment(sequence, resample_distribution,
                                                 number_of_multistart_points, restart)
        starting_points = experiment.generate()
        print("starting_points")
        print(starting_points)
        multiStartSolver = ot.MultiStart(solver, starting_points)
        algorithm.setOptimizationAlgorithm(multiStartSolver)
        return algorithm

    def buildFromQuantiles(self, distribution, alphaLower,
                        number_of_multistart_points = 20):
        """
        Create a KrigingAlgorithm using quantiles of the distribution as bounds.

        We compute the bounds of the correlation lengths to be optimized 
        while maximizing the log-likelihood. 
        These bounds are computed based on the componentwise quantiles of the 
        distribution. 
        The upper value of alpha is 1 - alphaLower. 

        Parameters
        ----------
        distribution : ot.Distribution
            The distribution of the points to create the Multistart.
        alphaLower : float, in [0.0, 0.5]
            The probability value of the lower quantile.
        number_of_multistart_points : int, optional
            The number of MultiStart points. The default is 20.

        Raises
        ------
        ValueError
            alphaLower must be in [0.0, 0.5].

        Returns
        -------
        algorithm : ot.KrigingAlgorithm
            A kriging algorithm.

        """
        if alphaLower < 0.0 or alphaLower > 0.5:
            raise ValueError("The parameter alpha=%s is not in [0, 0.5]" % (alphaLower))
        alphaLower = 0.01
        alphaUpper = 1.0 - alphaLower
        input_dimension = distribution.getDimension()
        minimum_scale_bounds = ot.Point(input_dimension)
        maximum_scale_bounds = ot.Point(input_dimension)
        for i in range(input_dimension):
            marginal = distribution.getMarginal(i)
            minimum_scale_bounds[i] = marginal.computeQuantile(alphaLower)[0]
            maximum_scale_bounds[i] = marginal.computeQuantile(alphaUpper)[0]
        print("minimum_scale_bounds=", minimum_scale_bounds)
        print("maximum_scale_bounds=", maximum_scale_bounds)
        algorithm = self.buildFromBounds(minimum_scale_bounds, maximum_scale_bounds,
                                number_of_multistart_points = 20)
        return algorithm
    
    def buildFromSample(self, scale_min_factor = 0.1, scale_max_factor = 10.0, 
                        number_of_multistart_points = 20):
        """
        Create a KrigingAlgorithm using the input sample.

        We set the bounds of the optimization algorithm for the 
        correlation lengths of the covariance model to the 
        given lower and upper bounds. 

        We use a MultiStart algorithm with the given number of points 
        as starting points for the optimization of the correlation lengths.
        The starting points are created from a Sobol' sequence.

        Parameters
        ----------
        scale_min_factor : float
            The multiplier of the scale parameter for the lower bound.
        scale_max_factor : float
            The multiplier of the scale parameter for the upper bound.
        number_of_multistart_points : int, optional
            The number of points in the MultiStart. The default is 20.

        Returns
        -------
        algorithm : ot.KrigingAlgorithm
            A kriging algorithm.

        References
        ----------
        Numerical issues in maximum likelihood parameter 
        estimation for Gaussian process regression. 
        S. Basak, S. Petit, J. Bect, E. Vasquez.
        (2021)
        """
        input_dimension = self.X_train.getDimension()
        minimum_scale_bounds = ot.Point(input_dimension)
        maximum_scale_bounds = ot.Point(input_dimension)
        for i in range(input_dimension):
            dist = scipy.spatial.distance.pdist(self.X_train[:,i])
            print("dist", dist)
            minimum_scale_bounds[i] = scale_min_factor * np.min(dist)
            maximum_scale_bounds[i] = scale_max_factor * np.max(dist)
        algorithm = self.buildFromBounds(minimum_scale_bounds, maximum_scale_bounds, 
                                         number_of_multistart_points)
        return algorithm

