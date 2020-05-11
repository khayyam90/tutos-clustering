''' Illustration didactique du partitionnement spectral des données, par Pierre Schwartz. 
Illustration du tuto présent sur www.developpez.com
Usage : python3 spectral.py
'''

import numpy as np
import matplotlib.pyplot as plt
import math 
from numpy import linalg as LA

# Paramétrage de connectivité pour construire le graphe
NB_NEIGHBOURS = 5


def distance(p1, p2):   
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

'''
 Génération d'un dataset avec 3 branches de spirales
'''
def genererSpirale():
    amplitude = 7
    nb = 600
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


dataset = genererSpirale()

# Affichage du dataset
plt.scatter(dataset[:,0], dataset[:,1], c="b", s=5)
plt.title("Dataset original")
plt.show()


'''
Obtenir les ids des k plus proches voisins
'''
def trouverLesKPlusProchesVoisins(dataset, i):
    # TODO: utilisation d'un quad-tree
    pointsParDistance = [distance(ii, dataset[i]) for ii in dataset]
    
    pointsSorted = np.argsort(pointsParDistance)      
    pointsSorted = pointsSorted[1: (NB_NEIGHBOURS+1)]  
    return pointsSorted

def genererMatriceLaplacienne(dataset):
    nbPointsDuDataset = len(dataset)
    matriceLaplacienne = [];
    for i in range(nbPointsDuDataset):  
        ligneLaplacienne = np.zeros(nbPointsDuDataset)        
        
        prochesVoisins = trouverLesKPlusProchesVoisins(dataset, i)
        # la valeur diagonale est le nombre de connexions au point i
        ligneLaplacienne[i] = len(prochesVoisins)
        
        # chaque point connecté est noté -1
        for j in prochesVoisins:
            ligneLaplacienne[j] = -1
                
        matriceLaplacienne.append(ligneLaplacienne) 
    return matriceLaplacienne
    
def afficherGrapheAdjacence():
    for i in range(  len(dataset) ):
        p = dataset[i]
        # on trie les points par distance
        pointsParDistance = [distance(ii, p) for ii in dataset]
        
        pointsSorted = np.argsort(pointsParDistance)      
        pointsSorted = pointsSorted[1: (NB_NEIGHBOURS+1)]  
 
        for i in pointsSorted:
            pp = dataset[i]        
            plt.plot([p[0], pp[0]], [p[1], pp[1]], c="b", linewidth=.2)
        
    plt.title("Graphe de connectivité")
    plt.show()

matriceLaplacienne = genererMatriceLaplacienne(dataset)
afficherGrapheAdjacence()


# recherche des valeurs propres
lambdas, xlambdas = LA.eig(matriceLaplacienne)

# la matrice laplacienne est symétrique, ses xlambdas sont nécessairement réelles

indicesLambdasTries = np.argsort(lambdas)
lambdasTries = lambdas [indicesLambdasTries]
xlambdas = np.transpose(xlambdas)
xlambdasTries = xlambdas[indicesLambdasTries]
xlambdasTries = np.transpose(xlambdasTries)


plt.clf()
    
# recherche dans les 4 premières xlambdas
for indexLambda in range(4):
    # Normalement il faut regarder la première xlambda à considérer, en fonction des valeurs lambdas
    maximum = np.max(xlambdasTries[:,indexLambda])
    minimum = np.min(xlambdasTries[:,indexLambda])
    
    # découpage en 3 : 1/3, 1/3, 1/3
    # TODO : découpage plus fin, via un k-means
    limit1 = (maximum - minimum)/3
    limit2 = 2*(maximum - minimum)/3
    
    for i in range(len(dataset)):
        v = xlambdasTries[i][2];
        
        if v > limit2:
            plt.scatter([dataset[i][0]], [dataset[i][1]], c="b", s=5)
        else:
            if v > limit1:
                 plt.scatter([dataset[i][0]], [dataset[i][1]], c="r", s=5)
            else:
                 plt.scatter([dataset[i][0]], [dataset[i][1]], c="g", s=5)
        
    plt.show()
