#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Évaluation du cardinal des bases polynomiales sous contrainte de degré.

Get the number of polynomials lower than a given degree.
Ce script a pour objectif de déterminer le nombre de polynômes dont le degré
total est inférieur ou égal à un seuil donné, en utilisant une règle
d'énumération hyperbolique. Il permet d'analyser la structure de la base en
identifiant précisément la strate limite et le nombre cumulé de multi-indices
nécessaires pour couvrir l'espace polynomial souhaité, facilitant ainsi le
dimensionnement des modèles de substitution.

La mise en œuvre s'appuie sur la classe HyperbolicAnisotropicEnumerateFunction
d'OpenTURNS pour lier le degré polynomial aux indices de strates. Le script
calcule les cardinaux par strate et les cardinaux cumulés, puis affiche le
détail de chaque multi-indice généré jusqu'à la strate maximale. Une fonction
Python personnalisée évaluant la quasi-norme $q$ est également incluse pour
permettre la vérification de la cohérence entre la norme analytique et le rang
d'énumération.
"""

# %%
import openturns as ot
import numpy as np


# %%
def print_up_to_maximum_strata_index(enumerateFunction, degree_strata_index):
    for strata_index in range(1 + degree_strata_index):
        print("+ strata index = ", strata_index)
        strata_cardinal = enumerateFunction.getStrataCardinal(strata_index)
        cumulated_cardinal = enumerateFunction.getStrataCumulatedCardinal(strata_index)
        number_of_indices_in_strata = cumulated_cardinal - strata_cardinal
        for i in range(number_of_indices_in_strata, cumulated_cardinal):
            multiindex = enumerateFunction(i)
            print("    ", multiindex, " sum=", sum(multiindex))
    return None


# %%

dimension = 2
quasi_norm_parameter = 0.5
polynomial_degree = 10

enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(
    dimension, quasi_norm_parameter
)
strata_index = enumerateFunction.getMaximumDegreeStrataIndex(polynomial_degree)
cardinal = enumerateFunction.getStrataCardinal(strata_index)
cumulated_cardinal = enumerateFunction.getStrataCumulatedCardinal(strata_index)
# %%

print("dimension=", dimension)
print("quasi_norm_parameter=", quasi_norm_parameter)
print("polynomial_degree=", polynomial_degree)
print("strata_index=", strata_index)
print("cardinal=", cardinal)
print("cumulated_cardinal=", cumulated_cardinal)

# %%

print_up_to_maximum_strata_index(enumerateFunction, strata_index)

# %%


class QuasiNorm(ot.OpenTURNSPythonFunction):
    def __init__(self, dimension, quasi_norm_parameter):
        super(QuasiNorm, self).__init__(dimension, 1)
        self.quasi_norm_parameter = quasi_norm_parameter

    def _exec(self, x):
        x = ot.Point(x)
        dimension = x.getDimension()
        norm = 0.0
        for i in range(dimension):
            norm += x[i] ** self.quasi_norm_parameter
        norm = np.exp(np.log(norm) / self.quasi_norm_parameter)
        return [norm]


quasi_norm = ot.Function(QuasiNorm(dimension, quasi_norm_parameter))

# %%

if False:
    polynomial_degree = 3
    strata_index = enumerateFunction.getMaximumDegreeStrataIndex(polynomial_degree)

    strata_cardinal = enumerateFunction.getStrataCardinal(strata_index)
    cumulated_cardinal = enumerateFunction.getStrataCumulatedCardinal(strata_index)
    number_of_indices_in_strata = cumulated_cardinal - strata_cardinal
    for i in range(number_of_indices_in_strata, cumulated_cardinal):
        multiindex = enumerateFunction(i)
        qnorm = quasi_norm(multiindex)
        total_degree = sum(multiindex)
        print(multiindex, "Q-Norm=", qnorm, "Total degree=", total_degree)

# %%
