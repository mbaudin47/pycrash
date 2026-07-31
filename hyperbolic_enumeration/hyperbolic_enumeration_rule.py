"""
Analyse de la structure des strates pour la règle d'énumération hyperbolique.

Ce script a pour objectif d'étudier la correspondance entre le degré total des
multi-indices et leur appartenance aux strates définies par une quasi-norme $q$.
Il vise particulièrement à identifier l'indice de strate minimal nécessaire
pour atteindre un degré total donné, tout en proposant une alternative
corrective aux méthodes natives d'OpenTURNS qui peuvent présenter des
comportements inattendus lors de la recherche du cardinal cumulé.

La mise en œuvre s'appuie sur une exploration itérative des strates de la classe
HyperbolicAnisotropicEnumerateFunction. Pour chaque strate, le code parcourt
les multi-indices associés, calcule leur somme algébrique et compare ce degré
au seuil fixé. Le script compare ensuite trois approches différentes : une
recherche exhaustive personnalisée, la méthode intégrée à la bibliothèque et
une réplication locale de l'algorithme C++ sous-jacent afin d'en vérifier la
cohérence.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment with the hyperbolic enumeration rule.
"""

import openturns as ot


def compute_strata_from_degree(
    enumerateFunction, total_degree=10, maximum_strata_index=100
):
    """
    Compute the minimum index of the strata corresponding to a given total degree.

    This is the smallest index of a strata which contains a multiindex
    having the required total degree.
    Usually, this function is used with a large maximumu strata index, so that
    a large number of stratas are explored until the required total degree
    is reached.

    The returned index is so that this strata contains a multi-index having
    a total degree equal to the required total_degree, but interaction multi-indices
    may be lower that this total_degree.

    This is a Python replicate of OpenTURNS's getStrataCumulatedCardinal(),
    but this function works properly, while OpenTURNS's has a bug.

    Parameters
    ----------
    enumerateFunction : ot.EnumerateFunction()
        The enumerate function.
    total_degree : int
        The total degree to reach.
    maximum_strata_index : int
        The maximum number of strata to try.

    Returns
    -------
    degree_strata_index : int
        The index of the strata containing a multiindex having given total
        degree.
        It is lower of equal to maximum_strata_index.
        It may be equal to maximum_strata_index if no multiindex
        has reached the given total degree.

    """
    is_degree_reached = False
    degree_strata_index = 0
    print("Hyperbolic rule, $q=%.2f$." % (quasi_norm_parameter))
    for strata_index in range(1 + maximum_strata_index):
        if is_degree_reached:
            break
        print("strata_index = ", strata_index)
        strata_cardinal = enumerateFunction.getStrataCardinal(strata_index)
        cumulated_cardinal = enumerateFunction.getStrataCumulatedCardinal(strata_index)
        number_of_indices_in_strata = cumulated_cardinal - strata_cardinal
        for i in range(number_of_indices_in_strata, cumulated_cardinal):
            multiindex = enumerateFunction(i)
            multiindex_degree = sum(multiindex)
            if multiindex_degree >= total_degree:
                print(
                    "Maximum degree reached : ",
                    multiindex_degree,
                    ", multiindex = ",
                    multiindex,
                    ", strata_index = ",
                    strata_index,
                )
                is_degree_reached = True
                degree_strata_index = strata_index
                break
    return degree_strata_index


def print_up_to_maximum_strata_index(enumerateFunction, degree_strata_index):
    for strata_index in range(1 + degree_strata_index):
        print("+ strata index = ", strata_index)
        strata_cardinal = enumerateFunction.getStrataCardinal(strata_index)
        cumulated_cardinal = enumerateFunction.getStrataCumulatedCardinal(strata_index)
        number_of_indices_in_strata = cumulated_cardinal - strata_cardinal
        for i in range(number_of_indices_in_strata, cumulated_cardinal):
            multiindex = enumerateFunction(i)
            print("    ", multiindex)
    return None


def getMaximumDegreeStrataIndex(enumerateFunction, maximumDegree):
    # From https://github.com/openturns/openturns/blob/f38378d3ea9bff8414786cdb30ff55a917f6eac5/lib/src/Base/Func/HyperbolicAnisotropicEnumerateFunction.cxx#L200
    # find indice
    index = 0
    degree = 0
    # First, a geometrical search to find an upper bound
    # Old : while (degree <= maximumDegree):
    while degree < maximumDegree:
        multiindex = enumerateFunction(index)
        degree = sum(multiindex)
        index += 1
    strataIndex = 0
    while enumerateFunction.getStrataCumulatedCardinal(strataIndex) < index:
        strataIndex += 1
    # Old : return strataIndex - 1
    return strataIndex


dimension = 2
quasi_norm_parameter = 0.5
maximum_strata_index = 100
total_degree = 10
enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(
    dimension, quasi_norm_parameter
)
degree_strata_index = compute_strata_from_degree(
    enumerateFunction, total_degree, maximum_strata_index
)
print("degree_strata_index = ", degree_strata_index)

degree_strata_index_bis = enumerateFunction.getMaximumDegreeStrataIndex(total_degree)
print("degree_strata_index_bis = ", degree_strata_index_bis)

degree_strata_index_ter = getMaximumDegreeStrataIndex(enumerateFunction, total_degree)
print("degree_strata_index_ter = ", degree_strata_index_ter)


print("Hyperbolic rule, q=%.2f." % (quasi_norm_parameter))
print_up_to_maximum_strata_index(enumerateFunction, degree_strata_index_ter)
