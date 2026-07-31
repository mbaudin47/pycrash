"""
Optimisation de la recherche de l'indice de strate pour un degré maximal.

Ce script a pour objectif de comparer deux algorithmes de détermination de
l'indice de strate minimal couvrant un degré polynomial cible. Il vise à
démontrer qu'une approche par exploration bloc par bloc (strate par strate)
est nettement plus performante qu'une recherche itérative individuelle, en
réduisant le nombre d'évaluations nécessaires pour identifier la frontière
de la base polynomiale sous une norme $q$ donnée.

La mise en œuvre repose sur une classe dérivée d'OpenTURNS qui surcharge les
méthodes de recherche d'indice. La version optimisée parcourt les strates
entières, calcule le degré maximal au sein de chaque couche et s'arrête dès
que le seuil est atteint, contrairement à la méthode classique qui incrémente
les indices un à un. Le script mesure le gain de temps substantiel entre ces
deux approches pour une configuration en dimension 10, tout en affichant la
progression du degré maximal pour chaque strate traitée.

References
----------

https://github.com/openturns/openturns/issues/2971

Benchmarking getMaximumDegreeStrataIndex
Elapsed = 2.1252 (s)
Strata is 5

Benchmarking getMaximumDegreeStrataIndexNew
In strata index 0, there are 1 multiindices and the maximum degree in this layer is 0
In strata index 1, there are 10 multiindices and the maximum degree in this layer is 1
In strata index 2, there are 55 multiindices and the maximum degree in this layer is 2
In strata index 3, there are 220 multiindices and the maximum degree in this layer is 3
In strata index 4, there are 715 multiindices and the maximum degree in this layer is 4
In strata index 5, there are 2002 multiindices and the maximum degree in this layer is 5
Target degree reached: stop!
Elapsed = 0.2957 (s)
Strata is 5

"""

# %%
import openturns as ot
import time


# %%
class SuperHyperbolicAnisotropicEnumerateFunction(
    ot.HyperbolicAnisotropicEnumerateFunction
):
    def __init__(self, dimension, qParameter):
        super().__init__(dimension, qParameter)
        self.dimension = dimension
        self.qParameter = qParameter

    def getMaximumDegreeStrataIndex(self, maximumDegree: int) -> int:
        # find indice
        index = 0
        degree = 0
        # First, a geometrical search to find an upper bound
        # print("Find upper bound")
        while True:
            multiIndex = self(index)
            degree = sum(multiIndex)
            # print(f"multiIndex = {multiIndex}, degree = {degree}")
            index += 1
            if degree > maximumDegree:
                break

        strataIndex = 0
        # Find strata index
        # print("Find strata index")
        while self.getStrataCumulatedCardinal(strataIndex) < index:
            strataIndex += 1

        return strataIndex - 1

    def getMaximumDegreeStrataIndexNew(self, maximumDegree: int) -> int:  # <- New
        number_of_coefficients = 0
        strata_index = 0
        while True:
            strata_cardinal = enumerate.getStrataCardinal(strata_index)
            multiindices_total_degree_in_layer = [
                sum(enumerate(idx + number_of_coefficients))
                for idx in range(strata_cardinal)
            ]
            maximum_degree_in_layer = max(multiindices_total_degree_in_layer)
            print(
                f"In strata index {strata_index}, "
                f"there are {strata_cardinal} multiindices and "
                f"the maximum degree in this layer is {maximum_degree_in_layer}"
            )
            number_of_coefficients += strata_cardinal
            if maximum_degree_in_layer >= maximumDegree:
                print(f"Target degree reached: stop!")
                break
            strata_index += 1
        return strata_index


# %%
print("Benchmarking getMaximumDegreeStrataIndex")
enumerate = SuperHyperbolicAnisotropicEnumerateFunction(10, 1.0)
maximumDegree = 5
t1 = time.time()
strata_index = enumerate.getMaximumDegreeStrataIndex(maximumDegree)
t2 = time.time()
print(f"Elapsed = {t2 - t1:.4f} (s)")
print(f"Strata is {strata_index}")

# %%
# From draw_stratas(enum_func)
print("Benchmarking getMaximumDegreeStrataIndexNew")
maximum_degree = 5
enumerate = SuperHyperbolicAnisotropicEnumerateFunction(10, 1.0)
t1 = time.time()
strata_index = enumerate.getMaximumDegreeStrataIndexNew(maximum_degree)
t2 = time.time()
print(f"Elapsed = {t2 - t1:.4f} (s)")
print(f"Strata is {strata_index}")
