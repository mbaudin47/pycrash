"""
Portage Python de la règle d'énumération anisotrope hyperbolique d'OpenTURNS.

Cette classe a pour objectif de fournir une bijection entre l'ensemble des
entiers naturels et l'ensemble des multi-indices, permettant ainsi d'ordonner
les fonctions d'une base de polynômes orthogonaux. En s'appuyant sur une
quasi-norme $q$ et des poids dimensionnels, elle permet de sélectionner
prioritairement les interactions de bas degré et les variables les plus
influentes, optimisant ainsi la troncature des développements en chaos
polynomial.

La mise en œuvre repose sur un algorithme d'exploration par voisinage maintenu
dans une file de priorité (implémentée ici par une liste triée et un objet
`deque`). À chaque itération, le multi-indice de norme minimale est extrait
pour alimenter un cache, puis ses successeurs directs sont générés et insérés
de manière ordonnée. Cette structure permet également la gestion dynamique des
strates (groupes d'indices de même norme) et propose des méthodes pour calculer
l'inverse de la fonction ou déterminer le cardinal cumulé des strates.
"""

import math
from collections import deque


class HyperbolicAnisotropicEnumerateFunction:
    """
    The bijective function to select polynomials in the orthogonal basis.
    Converted from C++ OpenTURNS implementation.
    """

    def __init__(self, *args):
        # Initialisation des attributs
        self.dimension = 0
        self.weight_ = []
        self.q_ = 0.0
        self.upperBound_ = []

        # Cache et structures de données (mutable en C++)
        self.candidates_ = deque()  # Équivalent std::list de paires (Indices, Scalar)
        self.cache_ = []  # Collection<Indices>
        self.strataCumulatedCardinal_ = []  # Indices

        if len(args) == 0:
            # Constructeur par défaut
            pass
        elif len(args) == 2:
            if isinstance(args[0], int):
                # Constructeur avec dimension et q
                self.dimension = args[0]
                self.weight_ = [1.0] * self.dimension
                self.upperBound_ = [float("inf")] * self.dimension
                self.setQ(args[1])
            else:
                # Constructeur avec point de poids et q
                self.weight_ = list(args[0])
                self.dimension = len(self.weight_)
                self.upperBound_ = [float("inf")] * self.dimension
                self.setQ(args[1])

        if self.dimension > 0:
            self.initialize()

    def initialize(self):
        self.cache_ = []
        self.candidates_ = deque()
        self.strataCumulatedCardinal_ = []
        # Insert indice 0, with q-norm 0.0
        zero_indices = [0] * self.dimension
        self.candidates_.append((zero_indices, 0.0))

    def qNorm(self, indices):
        result = 0.0
        if self.q_ == 1.0:
            for j in range(self.dimension):
                result += indices[j] * self.weight_[j]
            return result

        for j in range(self.dimension):
            result += math.pow(indices[j] * self.weight_[j], self.q_)
        return math.pow(result, 1.0 / self.q_)

    def computeDegree(self, indices):
        return max(indices) if indices else 0

    def __call__(self, index):
        """Équivalent de l'operator() en C++"""
        while len(self.cache_) <= index:
            if not self.candidates_:
                raise Exception(
                    f"Cannot enumerate up to index={index} because of the bounds."
                )

            # current (ValueType) est le premier candidat (trié par q-norm)
            current = self.candidates_.popleft()
            current_indices = current[0]
            current_norm = current[1]

            # Détection d'un saut de norme (strata)
            if len(self.cache_) > 0:
                prev_norm = self.qNorm(self.cache_[-1])
                if current_norm > prev_norm:
                    self.strataCumulatedCardinal_.append(len(self.cache_))
            elif len(self.cache_) == 0:
                # Initialisation pour la première strate (norme 0)
                pass

            self.cache_.append(current_indices)

            # Génération des voisins
            for j in range(self.dimension):
                next_indices = list(current_indices)
                if next_indices[j] >= self.upperBound_[j]:
                    continue

                next_indices[j] += 1
                next_norm = self.qNorm(next_indices)

                # Insertion dans la liste triée (maintien du tri par q-norm)
                inserted = False
                duplicate = False

                # Recherche de la position d'insertion
                idx = 0
                while idx < len(self.candidates_):
                    cand_norm = self.candidates_[idx][1]
                    if cand_norm > next_norm:
                        break
                    if cand_norm == next_norm:
                        if self.candidates_[idx][0] == next_indices:
                            duplicate = True
                            break
                    idx += 1

                if not duplicate:
                    self.candidates_.insert(idx, (next_indices, next_norm))

        return self.cache_[index]

    def inverse(self, indices):
        result = 0
        # Recherche dans le cache existant
        while result < len(self.cache_) and self.cache_[result] != indices:
            result += 1

        if result == len(self.cache_):
            # Génération jusqu'à trouver l'indice
            while True:
                found_indices = self.__call__(result)
                if found_indices == indices:
                    return result
                result += 1
        return result

    def getStrataCumulatedCardinal(self, strataIndex):
        while len(self.strataCumulatedCardinal_) <= strataIndex:
            self.__call__(len(self.cache_))
        return self.strataCumulatedCardinal_[strataIndex]

    def getStrataCardinal(self, strataIndex):
        result = self.getStrataCumulatedCardinal(strataIndex)
        if strataIndex > 0:
            result -= self.getStrataCumulatedCardinal(strataIndex - 1)
        return result

    def getMaximumDegreeStrataIndex(self, maximumDegree):
        index = 0
        while True:
            degree = self.computeDegree(self.__call__(index))
            if degree > maximumDegree:
                break
            index += 1

        strataIndex = 0
        while self.getStrataCumulatedCardinal(strataIndex) < index:
            strataIndex += 1
        return strataIndex - 1

    # Accessors
    def setQ(self, q):
        if not (q > 0.0):
            raise ValueError(f"q parameter should be positive, but q={q}")
        self.q_ = q
        self.initialize()

    def getQ(self):
        return self.q_

    def setWeight(self, weight):
        for w in weight:
            if not (w >= 0.0):
                raise ValueError("Anisotropic weights should not be negative")
        self.weight_ = list(weight)
        self.initialize()

    def getWeight(self):
        return self.weight_

    def setUpperBound(self, upperBound):
        self.upperBound_ = list(upperBound)
        self.initialize()

    def __repr__(self):
        return (
            f"class=HyperbolicAnisotropicEnumerateFunction "
            f"q={self.q_} weights={self.weight_}"
        )
