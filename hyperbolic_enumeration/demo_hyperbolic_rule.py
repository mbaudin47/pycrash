"""
Illustration du comportement des fonctions d'énumération anisotropes.

Ce script a pour objectif de mettre en évidence l'influence des paramètres de
norme $q$ et des poids associés aux dimensions sur l'ordre de génération des
multi-indices. Il permet de visualiser concrètement comment une quasi-norme
inférieure à l'unité ou une pondération hétérogène modifie la hiérarchie
classique des degrés, favorisant certaines directions ou configurations au
sein de la base polynomiale.

La mise en œuvre repose sur l'affichage séquentiel des premiers multi-indices
produits par la classe HyperbolicAnisotropicEnumerateFunction d'OpenTURNS. À
travers trois exemples distincts (variation de la norme $q$ en dimension 2,
application de poids différenciés en dimension 3 et extension à une dimension
plus élevée), le code vérifie l'ordre d'apparition des indices par rapport à
la somme de leurs degrés marginaux.

References
----------

https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.HyperbolicAnisotropicEnumerateFunction.html

Output
------

HyperbolicAnisotropicEnumerateFunction, dimension 2, q=0.5
[0,0]
[1,0]
[0,1]
[2,0]
[0,2]
[3,0]
[0,3]
[1,1]
[4,0]
[0,4]

HyperbolicAnisotropicEnumerateFunction, dimension 3, w = [1, 2, 4]
i= 0 enum= [0,0,0]
i= 1 enum= [1,0,0]
i= 2 enum= [0,1,0]
i= 3 enum= [2,0,0]
i= 4 enum= [3,0,0]
i= 5 enum= [0,0,1]
i= 6 enum= [0,2,0]
i= 7 enum= [4,0,0]
i= 8 enum= [5,0,0]
i= 9 enum= [0,3,0]
i= 10 enum= [6,0,0]
i= 11 enum= [7,0,0]
i= 12 enum= [0,0,2]
i= 13 enum= [0,4,0]
i= 14 enum= [8,0,0]
i= 15 enum= [1,1,0]
i= 16 enum= [9,0,0]
i= 17 enum= [0,5,0]
i= 18 enum= [10,0,0]
i= 19 enum= [11,0,0]

"""

# %%
import openturns as ot

# %%
# [Markdown]
# In the following example, we create an hyperbolic enumerate function in 2
# dimension with a quasi-norm equal to 0.5.
# Notice, for example, that the function with multi-index [3,0] come
# before [1,1], although the sum of marginal indices is lower: this is
# the result of the hyperbolic quasi-norm.

# %%
print("HyperbolicAnisotropicEnumerateFunction, dimension 2, q=0.5")
enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(2, 0.5)
for i in range(10):
    print(enumerateFunction(i))

# %%
# In the following example, we create an hyperbolic enumerate function in
# 3 dimensions based on the weights [1,2,4].
# Notice that the first marginal index, with weight equal to 1, comes
# first in the enumeration.

# %%
print("HyperbolicAnisotropicEnumerateFunction, dimension 3, w = [1, 2, 4]")
enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction([1, 2, 4])
for i in range(20):
    print("i=", i, "enum=", enumerateFunction(i))

# %%
print("HyperbolicAnisotropicEnumerateFunction, dimension 10, q=0.7")
enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(10, 0.7)
for i in range(50):
    print(enumerateFunction(i))

# %%
