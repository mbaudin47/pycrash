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
    """
    There are three parts:
        1. True unit test with hard-coded values
        2. Eco consistency test
        3. Exceptional tests
    """
    # The relative tolerance should be used in general.
    # Its value may be set to a multiple of the rounding error.
    # The factor 10 reflects what is lost because of the implementation.
    # Increasing this coefficient means a less accurate implementation.
    loss_coefficient = 1.0
    rtol = loss_coefficient * np.sys.float_info.epsilon
    for function_to_test in [sqrt1, sqrt2, sqrt3, sqrt4, sqrt5]:
        # 1. True unit tests
        print("Testing : ", function_to_test)
        testing_data = [[0.0, 0.0],                   # 0 : basic
                        [1.0, 1.0],                   # 1 : basic
                        [2.0, 1.414213562373095048],  # 2 : minimal real test
                        [1.5, 1.224744871391589049],  # 3 : non integer input
                        [1.e308, 1.e154]              # 4 : extreme float
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
        # 2. Consistency unit test.
        """
        This is the "eco" test, in the sense that it does not require to have
        actual function values hard-coded in the unit test.
        It may reveal catastropic failures, provided the composition of the
        backward inversion and the forward evaluation does not hide accuracy losses
        because what this test actually checks is not the function:

            y = sqrt(x)

        under test, but the composition:

            z = square ( sqrt ( x ) )

        and check if z is close to x.
        While the original function sqrt may have locally poor condition number
        for some values of x, the composition may have a different behaviour:
        * composition may have a poor condition for a x for which the condition
        number of the function is correct,
        * composition may have a good condition for a x for which the condition
        number of the function is poor.
        """
        number_of_consistency_tests = 7
        data = np.linspace(0.0, 5.0, number_of_consistency_tests)
        for i in range(number_of_consistency_tests):
            try:
                print("    Test #", i)
                x = data[i]
                computed_output = function_to_test(x)
                squared = computed_output ** 2
                np.testing.assert_allclose(squared, x, rtol=rtol)
            except AssertionError:
                print("    - Testing function : fail on consistency test #", i)
        # 3. Check that errors are correctly generated.
        # Generating an exception may or may not be wanted.
        try:
            y = function_to_test(-1.0)
            assert(np.isnan(y))
            with np.testing.assert_raises(TypeError):
                y = function_to_test("bla")
                y = function_to_test(1 + 1j)
        except AssertionError:
            print("    - Testing function : fail on error test")
        except ZeroDivisionError:
            print("    - Testing function : fail on error test")
    """
    Wrap-up
    -------
    * Use a unit test framework.
      70s development methods were great at this time, but that was 50 years ago.
    * Do not use a feature which is untested. This will save you (lots of) time.
      Writing a unit test is way faster. Not using a package is a more professional
      attitude than using the first package found on the internet or, worse,
      developed in hurry.
      If I do not have the time to test the package, I usually use the following
      function:

def my_super_complicated_algorithm():
    # ...
    # smart, complicated, creative, with excellent peer-review journal paper
    # ...
    return 12.

      then convince my client that the good answer is 12.
    * Do not integrate a development without a unit test.
      Do not review a code without a unit test.
      Proving that the code works is the job of the developer, not the one of the
      reviewer. If the function is untested, it does not work.
    * Do not close an issue without writing a new unit test.
      This is because the issue reveals a bug in the unit tests (and a bug in the code).
      Furthermore, changes in the code may happen which break it again in the future:
      the unit test will make this impossible.
    * To see the quality of a code, read and run the unit tests.
      This is because quality cannot be achieved without unit tests. No tests means
      no quality. Poor tests means poor quality.
      The unit tests may run well on a system, but may fail on another system because
      of the specific implementation of the system, e.g. the dependencies may
      behave differently.
    """
