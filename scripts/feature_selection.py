import numpy as np
from mixedRVMI import CMIEstimate

def backwardFeatureSelection(threshold,features,target,res,k):
    'the function returns the selected features starting from the full dataset and removing features keeping the loss of information smaller than the threshold'

    featureScores= []
    relevantFeatures = features # at the beginning all features are included
    CMIScore = 0 # cumulative loss of information
    sortedScores = []

    while CMIScore < threshold and relevantFeatures.shape[1]>1: 
        featureScores = scoreFeatures(relevantFeatures, target, k) # for each feature it evaluates the I(Y,X_i|X_A), at first step I(Y,X_i|X_{-i}),...
        sortedScores = sorted(featureScores, key=lambda x:x[1]) # lista ordinata (ascending) in base al punteggio di ogni feature
        CMIScore += max(sortedScores[0][1],0) # se il punteggio più basso è negativo, prendo 0
        if CMIScore > threshold: break
        relevantFeatures = np.delete(relevantFeatures, sortedScores[0][0], axis=1) # tolgo la feature (column) con punteggio più basso   
    res["numSelected"].append(relevantFeatures.shape[1]) 
    return relevantFeatures 

def scoreFeatures(features, target, k):
    'Ritorna una lista di features ID + punteggio CMI sul dato target'
    scores = np.zeros(features.shape[1])

    for col in range(features.shape[1]):
        scores[col] = CMIEstimate(features[:, col], target, np.delete(features,col,axis=1), k)

    return list(zip(range(len(scores)),scores))
