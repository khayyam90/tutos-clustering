''' Illustration didactique du partitionnement hiérarchique de données, par Pierre Schwartz. 
Illustration du tuto présent sur www.developpez.com
Usage : python3 hierarchical.py
'''

import numpy as np
import matplotlib.pyplot as plt

'''
Génération du dataset non séparable linéairement et non convexe
'''
def genererSpirale():
    amplitude = 7
    nb = 500
    result = []
    
    for i in range(nb):
        angle = i * 2 * np.pi / nb
        radius = .3 + i * amplitude / nb + np.random.rand()
        
        x = radius*np.cos(angle)
        y = radius*np.sin(angle)
        result.append( [x,y] )
        
    teta = 2*np.pi/3
    for i in range(nb):
        angle = teta + i * 2 * np.pi / nb
        radius = .3 + i * amplitude / nb + np.random.rand()
        
        x = radius*np.cos(angle)
        y = radius*np.sin(angle)
        result.append( [x,y] )    
          
    teta = 4*np.pi/3
    for i in range(nb):
        angle = teta + i * 2 * np.pi / nb
        radius = .3 + i * amplitude / nb + np.random.rand()
        
        x = radius*np.cos(angle)
        y = radius*np.sin(angle)
        result.append( [x,y] )     
                 
    index = list(range(len(result)))
    np.random.shuffle(index)
    
    randomShuffle = []
    for i in index:
        randomShuffle.append(result[i])
    
    return np.array(randomShuffle)


def calculerDissimilarite(dataset, cluster1, cluster2, method="ward"):
    toutesLesDistances = []
    
    def distance(x1, x2, y1, y2):
        # il n'est pas nécessaire de prendre la racine carrée. Les distances sont calculées pour être simplement comparées. 
        # l'ordre des distances reste inchangé sans la racine carrée. 
        return  (x1 - x2)**2 + (y1 - y2)**2 

    
    for i in cluster1:
        toutesLesDistances.append( [ distance(dataset[i][0], dataset[j][0], dataset[i][1], dataset[j][1]) for j in cluster2 if j != i])
    
    if method == "min":
        # on calcule la distance la plus petite entre toutes les paires
        distance = np.min(toutesLesDistances)
    else: 
        # on calcule la distance maximale entre deux paires
        if method == "max":
            distance = np.max(toutesLesDistances)
        else:
            # on calcule la distance moyenne entre les deux paires
            if method == "average":
                distance = np.mean(toutesLesDistances)
            else:
                # on calcule la distance de ward (distance entre les centroïdes)
                if method == "ward":
                    centroid1 = [np.mean(dataset[cluster1][0,:]), np.mean(dataset[cluster1][:,1])]
                    centroid2 = [np.mean(dataset[cluster2][0,:]), np.mean(dataset[cluster2][:,1])]
                    
                    distance = distance(centroid1[0], centroid2[0], centroid1[1], centroid2[1])

    return distance
    
    

dataset = genererSpirale()
plt.scatter(dataset[:,0], dataset[:,1], c="b", s=5)
plt.title("Dataset original")
plt.show()

# au début, on a autant de clusters que de points. Chaque cluster possède un seul point
clusters = []
for i in range(len(dataset)):
    clusters.append([i])


def genererMatriceDissimilarite(dataset, clusters, method):
    resultat = []
    for i in range(len(clusters)):
        ligne = []
        for j in range(len(clusters)):
            if i != j:
                ligne.append(calculerDissimilarite(dataset, clusters[i], clusters[j], method))
            else:
                ligne.append(0)
        resultat.append(ligne)
    return resultat
                
# méthodes autorisées : min, max, average et ward
matriceDissimilarite = genererMatriceDissimilarite(dataset, clusters, "min")


for iterations in range(len(dataset)):

    # on cherche la similarité la plus proche et non nulle
    cluster1 = cluster2 = 0
    minimum = matriceDissimilarite[0][1]
    
    # TODO, la matrice de dissimilatiré est symétrique, son parcours peut être simplifié
    for i in range(len(matriceDissimilarite)):
        for j in range(len(matriceDissimilarite)):
            if i != j and matriceDissimilarite[i][j] < minimum:
                minimum = matriceDissimilarite[i][j]
                cluster1 = i
                cluster2 = j
    
    # on regroupe cluster1 et cluster2 dans le cluster1
    clusters[cluster1] = np.concatenate( (clusters[cluster1] ,  clusters[cluster2]) )
    
    # on supprime le cluster 2
    clusters.pop(cluster2)
    
    # on supprime le cluster2 dans la matrice de matriceDissimilarite en ligne et en colonne
    matriceDissimilarite.pop(cluster2)
    for i in range(len(matriceDissimilarite)):
        matriceDissimilarite[i].pop(cluster2)
        
    # on recalcule la dissimilarité avec cluster1
    for i in range(len(matriceDissimilarite)):
        if i != cluster1:
            matriceDissimilarite[i][cluster1] = matriceDissimilarite[cluster1][i] = calculerDissimilarite(dataset, clusters[i], clusters[cluster1])
         
    # on réorganise les clusters par couleur
    couleurs = []
    for i in range(len(dataset)):
        # on cherche dans quel cluster il est
        for c in range(len(clusters)):
            if i in clusters[c]:
                couleurs.append(c)
                # TODO : possibilité d'arrêter la recherche
            
    # on dessine les clusters
    plt.clf()
    plt.scatter(dataset[:,0], dataset[:,1], c=couleurs, s=5, cmap="tab20")
    plt.savefig(str(iterations) + ".png")

    
