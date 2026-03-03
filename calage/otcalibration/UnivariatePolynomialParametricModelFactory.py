#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Univariate parametric polynomial in canonical basis.
"""

import openturns as ot

class UnivariatePolynomialParametricModelFactory:
    def __init__(self):
        """
        Create univariate polynomials in canonical basis.

        Returns
        -------
        None.

        """
        return None
        
    def build(self, polynomialDegree):
        """
        Return a ParametricFunction of a polynomial with given degree.
        
        The coefficients of the polynomials are parameters of the function.

        Parameters
        ----------
        polynomialDegree : int
            The polynomial degree.

        Returns
        -------
        model : ot.ParametricFunction
            The polynomial function.

        """
        # Create a degree 10 polynomial function
        inputList = ["x"]
        for i in range(1 + polynomialDegree):
            inputList.append("b%d" % (i))
        formulaString = "b0"
        for i in range(1, 1 + polynomialDegree):
            formulaString += " + b%d * x^%d" % (i, i)
        g = ot.SymbolicFunction(inputList, [formulaString])
        
        # Create model : x > p(x) with b as parameter
        indices = list(range(1, 2 + polynomialDegree))
        parameter = ot.Point(len(indices))
        model = ot.ParametricFunction(g, indices, parameter)
        return model
