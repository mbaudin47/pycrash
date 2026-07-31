"""
Décomposition du temps de calcul pour l'indexation des strates.

Ce script a pour objectif d'isoler et de mesurer séparément les deux étapes
critiques de la détermination de la taille d'une base polynomiale : la
recherche de l'indice de strate maximal correspondant à un degré donné et la
récupération du cardinal cumulé associé. Cette approche permet d'identifier si
le coût computationnel d'OpenTURNS provient de l'exploration de l'espace des
degrés ou de la gestion interne de la structure de données des strates.

La mise en œuvre consiste à chronométrer indépendamment les appels aux méthodes
`getMaximumDegreeStrataIndex` et `getStrataCumulatedCardinal` pour une
configuration à 10 dimensions sous une norme $q$ unitaire. En séparant ces deux
opérations, le script met en évidence l'effet du cache interne de la
bibliothèque, où le calcul initial de l'indice de strate absorbe la majeure
partie du temps d'exécution, rendant l'accès ultérieur au cardinal quasi
instantané.

References
----------
https://github.com/openturns/openturns/issues/2971

Part 1: getMaximumDegreeStrataIndex()
Elapsed = 1.9049 (s)
idx = 5

Part 2: getStrataCumulatedCardinal()
Elapsed = 0.0000 (s)
nbCoeffs = 3003
"""

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
