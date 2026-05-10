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
print("")
print(f"+ LinearEnumerateFunction")
t1 = time.time()
enumerateFunction = ot.LinearEnumerateFunction(10)
nbCoeffs = enumerateFunction.getBasisSizeFromTotalDegree(5)
t2 = time.time()
elapsed = t2 - t1
print(f"Elapsed = {elapsed:.4f} (s)")
print(f"Nb. Coeffs = {nbCoeffs}")
print(f"Speed={nbCoeffs / elapsed:.2f} coeffs/s")

# %%
print("")
print(f"+ HyperbolicAnisotropicEnumerateFunction(10, 1.0)")
t1 = time.time()
enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(10, 1.0)
nbCoeffs = enumerateFunction.getBasisSizeFromTotalDegree(5)
t2 = time.time()
elapsed = t2 - t1
print(f"Elapsed = {elapsed:.4f} (s)")
print(f"Nb. Coeffs = {nbCoeffs}")
print(f"Speed={nbCoeffs / elapsed:.2f} coeffs/s")

# %%
print("")
print(f"+ HyperbolicAnisotropicEnumerateFunction(20, 0.7)")
t1 = time.time()
enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(20, 0.7)
nbCoeffs = enumerateFunction.getBasisSizeFromTotalDegree(5)
t2 = time.time()
elapsed = t2 - t1
print(f"Elapsed = {elapsed:.4f} (s)")
print(f"Nb. Coeffs = {nbCoeffs}")
print(f"Speed={nbCoeffs / elapsed:.2f} coeffs/s")

# %%
print("")
print(f"+ HyperbolicEnumerateFunction(20, 0.7) (New!)")
t1 = time.time()
enumerateFunction = ot.HyperbolicEnumerateFunction(20, 0.7)
nbCoeffs = enumerateFunction.getBasisSizeFromTotalDegree(5)
t2 = time.time()
elapsed = t2 - t1
print(f"Elapsed = {elapsed:.4f} (s)")
print(f"Nb. Coeffs = {nbCoeffs}")
print(f"Speed={nbCoeffs / elapsed:.2f} coeffs/s")

# %%
