# Hyperbolic Enumeration

Ce dépôt est dédié à l'implémentation et à l'analyse de règles d'énumération hyperbolique et anisotrope pour les multi-indices, notamment dans le cadre de l'approximation par chaos polynomial (PCE) creux. L'objectif est de fournir des outils efficaces pour sélectionner les indices les plus significatifs en fonction d'une quasi-norme $q$ pondérée.

## Description des fichiers

* **application.py** : Interface utilisateur principale réalisée avec Streamlit pour visualiser interactivement la stratification des multi-indices.
* **benchmark_enumerate_function_performance.py** : Mesure les performances temporelles de la fonction d'énumération.
* **benchmark_enumeration_complexity_dim_scaling.py** : Analyse l'évolution de la complexité algorithmique en fonction de la dimension du problème.
* **benchmark_enumeration.py** : Script de comparaison des performances de différentes stratégies d'énumération.
* **demo_hyperbolic_rule.py** : Script de démonstration simple illustrant l'utilisation de la règle hyperbolique.
* **figures/** : Répertoire contenant les graphiques et visualisations générés.
* **HyperbolicAnisotropicEnumerateFunction.py** : Classe principale implémentant la logique d'énumération hyperbolique anisotrope.
* **hyperbolic_enumeration_cpp_specifications.md** : Document technique décrivant les spécifications pour une éventuelle implémentation en C++.
* **HyperbolicEnumerationHeap.py** : Implémentation d'une structure de données en tas pour optimiser la recherche et le tri des indices.
* **hyperbolic_enumeration_Python_specifications.md** : Spécifications détaillées de l'implémentation actuelle en Python.
* **hyperbolic_enumeration_rule.py** : Définition des règles mathématiques régissant l'énumération des multi-indices.
* **optimize_strata_index_search_algorithm.py** : Algorithme visant à optimiser la recherche d'indices au sein des strates.
* **plot_enumeratefunction.ipynb** : Notebook Jupyter pour l'exploration visuelle et le test des fonctions d'énumération.
* **plot_enumeratefunction.py** : Fonctions utilitaires pour la création de graphiques liés aux fonctions d'énumération.
* **plot_hyperbolic_anisotropic_rule_lib.py** : Bibliothèque de fonctions graphiques pour représenter les règles hyperboliques anisotropes.
* **plot_hyperbolic_anisotropic_rule.py** : Script principal pour générer les visualisations de la règle hyperbolique.
* **profile_enumerate_function_stages.py** : Analyse de profilage pour identifier les étapes les plus coûteuses en temps de calcul.
* **quasi_norm.py** : Calcul et gestion des quasi-normes $q$ utilisées pour la sélection des indices.
* **test_HyperbolicRule.py** : Suite de tests unitaires pour valider le comportement de la règle hyperbolique.

## Exemple

Pour explorer visuellement l'influence des poids et de la norme $q$ sur la sélection des indices, vous pouvez lancer l'application interactive Streamlit.

Assurez-vous d'avoir installé les dépendances nécessaires (notamment `streamlit`, `openturns`, `matplotlib` et `pandas`), puis exécutez la commande suivante à la racine du dépôt :

```bash
streamlit run application.py
```

