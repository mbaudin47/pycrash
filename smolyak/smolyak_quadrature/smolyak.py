#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 22:17:09 2022

@author: devel
"""

import openturns as ot
import os


class SmolyakExperiment:
    def __init__(self, dimension, k_stage):
        """
        Create a Smolyak experiment for quadrature.

        Reference:

        Knut Petras,
        Smolyak Cubature of Given Polynomial Degree with Few Nodes
        for Increasing Dimension,
        Numerische Mathematik,
        Volume 93, Number 4, February 2003, pages 729-753.

        Parameters
        ----------
        dimension : int
            The dimension of the experiment.
        k_stage : int
            The number of stages.

        Returns
        -------
        None.

        """
        self.dimension = dimension
        self.k_stage = k_stage

    def generateWithWeights(self):
        """
        Create a Smolyak quadrature.

        Returns
        -------
        x : ot.Sample(size, dimension)
            The nodes.
        w : ot.Point(size)
            The weights.

        """
        # Code fails with k = 0
        if self.k_stage == 0:
            sample = ot.Sample([[0.5] * self.dimension])
            weights = ot.Point([0.5])
            return sample, weights

        # Run command
        command = "./smolyak_driver %d %d" % (self.dimension, self.k_stage)
        os.system(command)

        # Read weights and nodes
        weights_filename = "weights.txt"
        with open(weights_filename, "r") as fp:
            for count, line in enumerate(fp):
                pass
        size = count + 1
        # Read weights
        weights = ot.Point(size)
        f = open(weights_filename, "r")
        lines = f.readlines()
        i = 0
        for row in lines:
            index = row.find("=")
            weights[i] = float(row[1 + index : -1])
            i += 1
        # Read nodes
        sample = ot.Sample(size, self.dimension)
        nodes_filename = "nodes.txt"
        f = open(nodes_filename, "r")
        lines = f.readlines()
        row_index = 0
        i = 0
        j = 0
        for row in lines:
            index = row.find("=")
            sample[i, j] = float(row[1 + index : -1])
            row_index += 1
            if row_index % self.dimension == 0:
                j = 0
                i += 1
            else:
                j += 1
        return sample, weights
