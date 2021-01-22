# -*- coding: utf-8 -*-
"""
This script provides wrong implementations of the square root function.

The goal : find bugs using each function as a black-box:
    PLEASE DO NOT READ THE CODE IN square_root.py.

The game : load this module and find x which shows the bug in the output y = sqrt(x).

Here is a template of your script:

import square_root
x = 0.0  # Find the critical value ...
expected_output = 0.0  # ... and the expected output ...
computed_output = square_root.sqrt1(x)
# ... such that this fails!
np.testing.assert_allclose(computed_output, expected_output)
"""

import numpy as np


def sqrt1(x):
    """
    Compute square root - method 1.

    Parameters
    ----------
    x : float
        The input.

    Returns
    -------
    y : float
        The square root
    """
    y = np.sqrt(x)
    z = np.around(y, 9)
    return z


def sqrt2(x):
    """
    Compute square root - method 2.

    Parameters
    ----------
    x : float
        The input.

    Returns
    -------
    y : float
        The square root
    """
    y = np.sqrt(x)
    z = np.around(y, 0)
    return z


def sqrt3(x):
    """
    Compute square root - method 3.

    Parameters
    ----------
    x : float
        The input.

    Returns
    -------
    y : float
        The square root
    """
    y = np.sqrt(x)
    if (x - int(x) != 0.0):
        y = np.around(y, 0)
    return y


def sqrt4(x):
    """
    Compute square root - method 4.

    Parameters
    ----------
    x : float
        The input.

    Returns
    -------
    y : float
        The square root
    """
    y = 1.0
    for i in range(100):
        # print("y=", y)
        z = (y + x / y) / 2.0
        if abs(z - y) < 1.e-16 * max(z, y):
            break
        y = z
    return y


def sqrt5(x):
    """
    Compute square root - method 5.

    Parameters
    ----------
    x : float
        The input.

    Returns
    -------
    y : float
        The square root
    """
    y = np.sqrt(x)
    return y


if __name__ == "__main__":
    print("Unit testing.")
    rtol = 1.e-16
    for function_to_test in [sqrt1, sqrt2, sqrt3, sqrt4, sqrt5]:
        print("Testing : ", function_to_test)
        testing_data = [[0.0, 0.0],                   # 0
                        [1.0, 1.0],                   # 1
                        [1.5, 1.224744871391589049],  # 2
                        [2.0, 1.414213562373095048],  # 3
                        [1.e308, 1.e154]              # 4
                        ]
        number_of_tests = len(testing_data)
        for i in range(number_of_tests):
            try:
                print("    Test #", i)
                x = testing_data[i][0]
                expected_output = testing_data[i][1]
                computed_output = function_to_test(x)
                np.testing.assert_allclose(
                    computed_output, expected_output, rtol=rtol)
            except AssertionError:
                print("    - Testing function : fail on test #", i)
