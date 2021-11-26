from scipy.special import digamma
from scipy.spatial import KDTree
import numpy as np

def MIEstimate(X,Y,k=5):
    'MI Estimator based on Mixed Random Variable Mutual Information Estimator - Gao et al.'
    nSamples = len(X)
    if X.ndim == 1:
        X = X.reshape(-1,1)
    if Y.ndim == 1:
        Y = Y.reshape(-1,1)
    dataset = np.concatenate((X,Y), axis=1) # concatenate Y to X as a column

    # kdtree per trovare più rapidamente i k nearest neighbors
    tree_xy = KDTree(dataset) 
    tree_x = KDTree(X)
    tree_y = KDTree(Y)
    
    # rho
    Knn_dists = [tree_xy.query(sample, k+1, p=float('inf'))[0][k] for sample in dataset]
    
    res = 0
    
    for i in range(nSamples):
        k_hat, n_xi, n_yi = k, k, k
        if Knn_dists[i] == 0:
            # punti a distanza inferiore uguale a (quasi) 0 
            k_hat = len(tree_xy.query_ball_point(dataset[i], 1e-15, p=float('inf')))
            n_xi = len(tree_x.query_ball_point(X[i], 1e-15, p=float('inf')))
            n_yi = len(tree_y.query_ball_point(Y[i], 1e-15, p=float('inf')))
        else:
            # punti a distanza inferiore uguale a rho
            n_xi = len(tree_x.query_ball_point(X[i], Knn_dists[i]-1e-15, p=float('inf')))
            n_yi = len(tree_y.query_ball_point(Y[i], Knn_dists[i]-1e-15, p=float('inf')))
        res += (digamma(k_hat) + np.log(nSamples) - digamma(n_xi) - digamma(n_yi))/nSamples
        # risultato diverso se uso digamma(n_xi), digamma(n_yi)
    return res

def CMIEstimate(X,Y,Z,k=5):
    """
    I(X;Y|Z) = I(X,Z; Y) - I(Z; Y)
    """
    if X.ndim == 1:
        X = X.reshape(-1,1)
    if Y.ndim == 1:
        Y = Y.reshape(-1,1)
    if Z.ndim == 1:
        Z = Z.reshape(-1,1) 
    
    XZ = np.hstack((X,Z)) # concateno X,Z in colonne

    return (MIEstimate(XZ,Y,k)-MIEstimate(Z,Y,k))