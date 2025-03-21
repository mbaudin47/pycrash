# Stratégie de développement de fonctionnalités graphiques dans OpenTURNS

## Introduction
Dans cette note sur OpenTURNS, j'analyse les raisons qui poussent en faveur du développement de fonctionnalités graphiques dans la librairie OpenTURNS. En effet, on peut se demander pourquoi OpenTURNS fournit des fonctionnalités graphiques alors que d'autres librairies Python comme Matplotlib fournissent déjà des fonctionnalités similaires. L'objectif de ce document est de montrer que, spécifiquement, Matplotlib ou aucune autre librairie graphique similaire ne fournissent l'ensemble des  fonctionnalités dont nous avons besoin dans le contexte de la visualisation des incertitudes.

## Motivations
Il y a plusieurs éléments qui peuvent motiver pour l'inclusion dans OpenTURNS de fonctionnalités graphiques qui ne sont pas disponibles par ailleurs.
- La librairie Matplotlib permet de représenter des données stockées dans des tableaux Numpy. Toutefois, cela nécessite de produire les tableaux requis. Par exemple, cela peut nécessiter de générer les données par **évaluation d'une fonction** par exemple. La cause profonde de cette limitation est que ni Python, ni Matplotlib, ni Numpy n'ont de concepts similaires à la `Function` ou à la `Distribution`.
- La librairie Matplotlib peut représenter une ligne représentant une courbe, mais ne peut pas connaître **le domaine** sur lequel représenter cette courbe : cette information doit être fournie à l'utilisateur. Dans certains cas, cette information peut être déduite de certaines propriétés de la `Distribution`, par exemple en utilisant les quantiles de la loi.
- Ni Matplotlib ni Seaborn ne sont des librairies de visualisation spécifiquement adaptées à la **quantification des incertitudes**. Matplotlib est une librairie Python de dessin et de données numériques tandis que Seaborn se définit comme une librairie Python s'appuyant sur Matplotlib permettant de visualiser des données statistiques. Aucune de ces deux librairies ne connaît le concept d'indice de Sobol' ou de distribution, par exemple.

## Conclusion
La librairie OpenTURNS n’a pas vocation, à priori du moins, à devenir une librairie graphique. Pour cette raison, nous recherchons un compromis entre la possibilité de produire les graphiques les plus pertinents possibles dans le contexte d’OpenTURNS et du traitement des incertitudes et la simplicité de l’implémentation.

Sur le long terme, la librairie devra veiller à conserver un périmètre restreint sur le champ graphique. Or on constate que d’autres librairies, comme Seaborn par exemple, ont des fonctionnalités qui peuvent sembler similaires. La solution à ce problème consiste plutôt à faire diffuser le moteur de calcul OpenTURNS comme dépendance de la librairie Seaborn, plutôt que de tenter de vouloir reproduire les fonctionnalités de Seaborn au sein d’OpenTURNS. Sur ce thème, le comité de pilotage du consortium OpenTURNS est certainement appelé à émettre des directions.
