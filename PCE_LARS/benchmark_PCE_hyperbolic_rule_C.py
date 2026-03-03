# %%
import openturns as ot
import time

# %%
# Decompose the time
enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(10, 1.0)

# %%
print("Part 1: getMaximumDegreeStrataIndex()")
maximumDegree = 5
t1 = time.time()
idx = enumerateFunction.getMaximumDegreeStrataIndex(maximumDegree)
t2 = time.time()
print(f"Elapsed = {t2 - t1:.4f} (s)")
print(f"idx = {idx}")

# %%
print("Part 2: getStrataCumulatedCardinal()")
t1 = time.time()
nbCoeffs = enumerateFunction.getStrataCumulatedCardinal(idx)
t2 = time.time()
print(f"Elapsed = {t2 - t1:.4f} (s)")
print(f"nbCoeffs = {nbCoeffs}")