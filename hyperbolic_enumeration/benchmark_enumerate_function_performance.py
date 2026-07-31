"""
Évaluation des performances de calcul du cardinal des bases polynomiales.

Ce script a pour objectif de mesurer le temps requis par différentes classes
d'énumération d'OpenTURNS pour déterminer la taille d'une base à partir d'un
degré total donné. Il permet de quantifier l'efficacité computationnelle de la
méthode `getBasisSizeFromTotalDegree` lorsqu'elle est soumise à des contraintes
linéaires ou hyperboliques, mettant ainsi en évidence les différences de coût
entre les algorithmes standards et les nouvelles implémentations.

La mise en œuvre consiste à comparer systématiquement quatre configurations :
une énumération linéaire classique et trois variantes hyperboliques (isotropes
et anisotropes) avec des paramètres de norme $q$ distincts. Pour chaque test,
le script enregistre le temps de calcul précis, le nombre de coefficients
générés et le débit résultant, facilitant ainsi l'identification de gains de
performance potentiels apportés par les versions récentes de la bibliothèque.

References
----------

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
