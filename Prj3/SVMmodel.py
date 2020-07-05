import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def runSVM(X_train, X_test, y_train, y_test, C, kernel):
    model = SVC(C=C, kernel=kernel, gamma='auto', verbose=False)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print("C=%f, score=%f, kernel:%s"%(C, score, kernel))
    return score

def getBestParam(kernel):
    if kernel == 'rbf':
        C = 5.0
    elif kernel == 'linear':
        C = 0.002
    else:
        C = None
    return  C
