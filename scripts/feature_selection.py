import numpy as np
from mixedRVMI import MIEstimate, CMIEstimate


def backwardFeatureSelection(delta,threshold,features,target,res,k):
    'Ritorna un insieme di features rilevanti per il target'
   
    featureScores= [] # punteggio per ogni feature
    relevantFeatures = features

    MIScore = 0
    sortedScores = []
    while MIScore < threshold and relevantFeatures.shape[1]>1: 
        featureScores = scoreFeatures(relevantFeatures, target, k)
        sortedScores = sorted(featureScores, key=lambda x:x[1]) # lista ordinata in base al punteggio di ogni feature
        MIScore += max(sortedScores[0][1],0) # se il punteggio più basso è negativo, prendo 0
        relevantFeatures = np.delete(relevantFeatures, sortedScores[0][0], axis=1) # tolgo la feature col punteggio più basso   
    res["numSelected"].append(relevantFeatures.shape[1]) 
    return relevantFeatures 

def scoreFeatures(features, target, k):
    'Ritorna una lista di features ID + punteggio CMI sul dato target'
    scores = np.zeros(features.shape[1])

    for col in range(features.shape[1]):
        scores[col] = CMIEstimate(features[:, col], target, np.delete(features,col,axis=1), k)

    return list(zip(range(len(scores)),scores))
