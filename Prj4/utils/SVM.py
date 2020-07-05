import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import copy

def SVM(X_train, X_test, y_train, y_test):
    params = {'kernel': ('linear', 'rbf'), 'C': [0.001, 0.01, 0.1, 1, 10]}
    svc = SVC()
    clf = GridSearchCV(estimator=svc, param_grid=params, verbose=True, n_jobs=8)
    clf.fit(X_train, y_train)
    score = clf.best_estimator_.score(X_test, y_test)
    print("score: %f" % (score))
    return score
