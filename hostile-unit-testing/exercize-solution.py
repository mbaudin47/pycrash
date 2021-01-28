# -*- coding: utf-8 -*-
"""
This script provides an implementation to test the square root function.

    PLEASE DO NOT READ THE CODE IN exercize-solution.py

before attempting the exercize.
"""

from square_root import sqrt5
import numpy as np

# Case 1
x = 0.0
expected_output = 0.0
computed_output = sqrt5(x)
np.testing.assert_allclose(computed_output, expected_output, atol=0.0)

# Case 2
x = 1.0
expected_output = 1.0
computed_output = sqrt5(x)
np.testing.assert_allclose(computed_output, expected_output, rtol=0.0)

# Case 3
x = 4.0
expected_output = 2.0
computed_output = sqrt5(x)
np.testing.assert_allclose(computed_output, expected_output, rtol=0.0)

# Case 4
x = 2.0
expected_output = 1.41421356237309505
computed_output = sqrt5(x)
np.testing.assert_allclose(computed_output, expected_output, rtol=1.e-16)

# Case 5 - extreme (large)
x = 1.e308
expected_output = 1.e154
computed_output = sqrt5(x)
np.testing.assert_allclose(computed_output, expected_output, rtol=1.e-16)

# Case 6 - extreme (close to zero)
x = 1.e-308
expected_output = 1.e-154
computed_output = sqrt5(x)
np.testing.assert_allclose(computed_output, expected_output, rtol=1.e-16)

# Case 7 - extreme (close to zero - subnormal)
x = 1.e-322
expected_output = 1.e-161
computed_output = sqrt5(x)
np.testing.assert_allclose(computed_output, expected_output, rtol=1.e-16)

# Case 8 - Test a set of numbers (consistency with square)
for x in np.logspace(-322, 308, 100):
    computed_output = sqrt5(x)
    squared_output = computed_output ** 2
    np.testing.assert_allclose(squared_output, x, rtol=1.e-15)

# Case 9 - Test a set of small integers - exact output is expected (requires time)
for x in np.arange(0.0, 1.e4):
    squared = x ** 2
    computed_output = sqrt5(squared)
    np.testing.assert_allclose(computed_output, x, rtol=0.0)

"""
There are many other tests to perform in practice.
* Performance.
* Exceptional floats : INF, NAN.
* Errors (e.g. non positive number).
* Other types : complex, arrays, etc...
"""
