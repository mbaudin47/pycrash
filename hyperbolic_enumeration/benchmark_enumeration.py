"""
https://github.com/openturns/openturns/issues/2971

Output
------

Dimension = 20
basisSize = 5000

+ 1. LinearEnumerateFunction
basis size =  5000
elapsed = 0.0 (s)
Speed=656611.67 coeffs/s

+ 2. HyperbolicAnisotropicEnumerateFunction
basis size =  5000
elapsed = 11.9 (s)
Speed=420.72 coeffs/s

+ 3. HyperbolicAnisotropicEnumerateFunction, 0.7
basis size =  5000
elapsed = 14.6 (s)
Speed=343.25 coeffs/s

+ 4. HyperbolicEnumerateFunction, 0.7
basis size =  5000
elapsed = 14.6 (s)
Speed=342.20 coeffs/s

"""

# %%
import openturns as ot
import time
import openturns.viewer as otv


# %%
print(f"OT Version: {ot.__version__}")

# %%
def TimeEnumerationFunction(enumerateFunction, basisSize):
    t1 = time.time()
    for i in range(basisSize):
        _ = enumerateFunction(i)
    t2 = time.time()
    elapsed = t2 - t1
    return elapsed



# %%
dimension = 20
basisSize = 5000
print(f"Dimension = {dimension}")
print(f"basisSize = {basisSize}")

# %%
print("")
print("+ 1. LinearEnumerateFunction")
enumerateFunction = ot.LinearEnumerateFunction(dimension)
print("basis size = ", basisSize)
elapsed = TimeEnumerationFunction(enumerateFunction, basisSize)
print("elapsed = %.1f (s)" % (elapsed))
print(f"Speed={basisSize / elapsed:.0f} coeffs/s")


# %%
print("")
print("+ 2. HyperbolicAnisotropicEnumerateFunction, q=1.0")
quasiNorm = 1.0
weight = [1.0] * dimension
enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(weight, quasiNorm)
print("basis size = ", basisSize)
elapsed = TimeEnumerationFunction(enumerateFunction, basisSize)
print("elapsed = %.1f (s)" % (elapsed))
print(f"Speed={basisSize / elapsed:.0f} coeffs/s")

# %%
print("")
print("+ 3. HyperbolicAnisotropicEnumerateFunction, q=0.7")
quasiNorm = 0.7
enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(dimension, quasiNorm)
print("basis size = ", basisSize)
elapsed = TimeEnumerationFunction(enumerateFunction, basisSize)
print("elapsed = %.1f (s)" % (elapsed))
print(f"Speed={basisSize / elapsed:.0f} coeffs/s")

# %%
print("")
print("+ 4. HyperbolicEnumerateFunction, 0.7")
quasiNorm = 0.7
enumerateFunction = ot.HyperbolicEnumerateFunction(dimension, quasiNorm)
print("basis size = ", basisSize)
elapsed = TimeEnumerationFunction(enumerateFunction, basisSize)
print("elapsed = %.1f (s)" % (elapsed))
print(f"Speed={basisSize / elapsed:.0f} coeffs/s")
