"""
https://github.com/openturns/openturns/issues/2971

Output
------

LinearEnumerateFunction
Elapsed = 0.0011 (s)
Nb. Coeffs = 3003

HyperbolicAnisotropicEnumerateFunction(1.0)
Elapsed = 2.3327 (s)
Nb. Coeffs = 3003

"""
# %%
import time
import openturns as ot

# %%
print(f"LinearEnumerateFunction")
t1 = time.time()
enumerateFunction = ot.LinearEnumerateFunction(10)
nbCoeffs = enumerateFunction.getBasisSizeFromTotalDegree(5)
t2 = time.time()
print(f"Elapsed = {t2 - t1:.4f} (s)")
print(f"Nb. Coeffs = {nbCoeffs}")

# %%
print(f"HyperbolicAnisotropicEnumerateFunction(10, 1.0)")
t1 = time.time()
enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(10, 1.0)
nbCoeffs = enumerateFunction.getBasisSizeFromTotalDegree(5)
t2 = time.time()
print(f"Elapsed = {t2 - t1:.4f} (s)")
print(f"Nb. Coeffs = {nbCoeffs}")

# %%
print(f"HyperbolicAnisotropicEnumerateFunction(20, 0.7)")
t1 = time.time()
enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(20, 0.7)
nbCoeffs = enumerateFunction.getBasisSizeFromTotalDegree(4)
t2 = time.time()
print(f"Elapsed = {t2 - t1:.4f} (s)")
print(f"Nb. Coeffs = {nbCoeffs}")

# %%
