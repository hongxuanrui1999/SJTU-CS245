import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def runSVM(X_train, X_test, y_train, y_test, C, kernel):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    model = SVC(C=C, kernel=kernel, gamma='auto', verbose=False)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print("score=%f"%(score))
    return score

def getBestParam(kernel):
    if kernel == 'rbf':
        C = 5.0
    elif kernel == 'linear':
        C = 0.002
    else:
        C = None
    return  C
