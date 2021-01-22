# -*- coding: utf-8 -*-
"""
This script provides an exercize to test the square root function.

The pitch:
We provide 5 implementations of the square root function. 
Unfortunately, the doc was lost. 
Moreover, an hostile spy put bugs in the code: 
we do not know which function is correct and which function is 
wrong. 

The goal: find bugs using each function as a black-box:
    PLEASE DO NOT READ THE CODE IN square_root.py.

The game: find x which shows the bug in the output y = sqrt(x).

Here is a template of your script:
"""

from square_root import (sqrt1, sqrt2, sqrt3, sqrt4, sqrt5) 
import numpy as np

x = TODO  # Find the critical value ...
expected_output = TODO  # ... and the expected output ...
computed_output = square_root.sqrt1(x)
# ... such that this fails!
np.testing.assert_allclose(computed_output, expected_output)

"""
The goal is to replace "TODO" with valid code.

Questions
---------

* Look at the help page of the assert_allclose function in numpy:
https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html

How to set the relative tolerance?
How to set the absolute tolerance?
What are the points x for which the relative tolerance should be used?
What are the points x for which the absolute tolerance should be used?
What should be the value of the relative tolerance that we should 
use?
What should be the value of the absolute tolerance that we should 
use?

* Test the output value at x = 0.

* Can you find three other simple x values that may be tested 
with a minimum effort?

* Test the output value at x = 2.
In order to get the expected output, use a symbolic 
computation software (e.g. www.wolframalpha.com) 
or a numerical scientific software (e.g. www.scilab.org).

* Test the output value at an extreme floating point number.

* Test the speed.

* Conclude: what has not been tested in this current session?
Hint: think outside the box!
"""
