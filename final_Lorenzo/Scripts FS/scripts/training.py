import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.multioutput import MultiOutputRegressor


def computeAccuracy(task, features, target):
    'Returns prediction accuracy of a model trained on the given dataset'
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, target)

    if task == 1: # classification
        lm = LogisticRegression()
    else:
        lm = LinearRegression()
    
    lm.fit(Xtrain, Ytrain)
    return lm.score(Xtest, Ytest)
    