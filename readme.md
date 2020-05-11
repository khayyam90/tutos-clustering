# Classification non supervisée
Ce dépôt contient quelques implémentations d'algorithmes de classification non supervisée (clustering). Elles illustrent mes tutos publiés sur www.developpez.com.
Ces implémentations n'ont pas été optimisées. L'ajout de structures spatiales comme les quad-tree améliorent grandement les explorations de voisinages.

## spectral.py
Partitionnement spectral des données, en utilisant la matrice laplacienne du graphe et ses vecteurs propres.

## hierarchical.py
Partitionnement hiérarchique des données. En regroupant itérativement les clusters les plus proches jusqu'à n'avoir plus qu'un seul cluster.