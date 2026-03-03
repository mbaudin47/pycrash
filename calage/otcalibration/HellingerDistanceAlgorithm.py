#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 22:50:00 2021

@author: devel
"""

import numpy as np
import openturns as ot


class HellingerDistanceAlgorithm:
    def __init__(self, distribution_1, distribution_2):
        """
        Create a Hellinger distance algorithm.

        Parameters
        ----------
        distribution_1 : ot.Distribution
            The first distribution.
        distribution_2 : ot.Distribution
            The second distribution.

        Raises
        ------
        ValueError
            The dimension of the two distributions must be equal.

        Returns
        -------
        None.
        """
        self.distribution_1 = distribution_1
        self.distribution_2 = distribution_2
        dimension = distribution_1.getDimension()
        dimension_2 = distribution_2.getDimension()
        if dimension_2 != dimension:
            raise ValueError(
                "The dimension of distribution 1 is %d but"
                " distribution 2 has dimension %d" % (dimension, dimension_2)
            )

    def compute_by_integration(self, integration_algorithm):
        """
        Compute the Hellinger distance.

        Parameters
        ----------
        integration_algorithm : ot.IntegrationAlgorithm
            The integration algorithm.

        Returns
        -------
        hellinger_distance: float
            The Hellinger distance.

        """
        if not self.distribution_1.isContinuous():
            raise ValueError("The distribution 1 must be continuous.")
        if not self.distribution_2.isContinuous():
            raise ValueError("The distribution 2 must be continuous.")

        def compute_integrand(x):
            pdf_1 = self.distribution_1.computePDF(x)
            pdf_2 = self.distribution_2.computePDF(x)
            y = (np.sqrt(pdf_1) - np.sqrt(pdf_2)) ** 2
            return [y]

        dimension = self.distribution_1.getDimension()
        integrand = ot.PythonFunction(dimension, 1, compute_integrand)
        # Set integration lower bound
        range_1 = self.distribution_1.getRange()
        range_2 = self.distribution_2.getRange()
        lower_bound_1 = range_1.getLowerBound()
        lower_bound_2 = range_2.getLowerBound()
        lower_bound = ot.Point(
            [min(lower_bound_1[i], lower_bound_2[i]) for i in range(dimension)]
        )
        # Set integration upper bound
        upper_bound_1 = range_1.getUpperBound()
        upper_bound_2 = range_2.getUpperBound()
        upper_bound = ot.Point(
            [max(upper_bound_1[i], upper_bound_2[i]) for i in range(dimension)]
        )
        interval = ot.Interval(lower_bound, upper_bound)
        integral = integration_algorithm.integrate(integrand, interval)[0]
        hellinger_distance = np.sqrt(integral) / np.sqrt(2.0)
        return hellinger_distance

    def compute_by_summation(self):
        """
        Compute the Hellinger distance.

        Parameters
        ----------
        NONE

        Returns
        -------
        hellinger_distance: float
            The Hellinger distance.

        """
        if self.distribution_1.isContinuous():
            raise ValueError("The distribution 1 must not be continuous.")
        if self.distribution_2.isContinuous():
            raise ValueError("The distribution 2 must not be continuous.")

        def compute_integrand(x):
            pdf_1 = self.distribution_1.computePDF(x)
            pdf_2 = self.distribution_2.computePDF(x)
            y = (np.sqrt(pdf_1) - np.sqrt(pdf_2)) ** 2
            return y

        support_1 = self.distribution_1.getSupport()
        support_2 = self.distribution_2.getSupport()
        if support_1 != support_2:
            raise ValueError("The two distributions must have the same support.")
        integral = 0.0
        for x in support_1:
            integral += compute_integrand(x)
        hellinger_distance = np.sqrt(integral) / np.sqrt(2.0)
        return hellinger_distance
