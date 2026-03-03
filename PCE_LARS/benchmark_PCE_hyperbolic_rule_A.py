# %%
import time
import openturns as ot

# %%
t1 = time.time()
enumerateFunction = ot.LinearEnumerateFunction(10)
nbCoeffs = enumerateFunction.getBasisSizeFromTotalDegree(5)
t2 = time.time()
print(f"Elapsed = {t2 - t1:.4f} (s)")
print(f"Nb. Coeffs = {nbCoeffs}")

# %%
t1 = time.time()
enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(10, 1.0)
nbCoeffs = enumerateFunction.getBasisSizeFromTotalDegree(5)
t2 = time.time()
print(f"Elapsed = {t2 - t1:.4f} (s)")
print(f"Nb. Coeffs = {nbCoeffs}")