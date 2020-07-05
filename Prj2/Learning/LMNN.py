import numpy as np
import pandas as pd
import matplotlib.pyplot as ply
from metric_learn import LMNN
import sklearn
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.preprocessing import StandardScaler
import sys

sys.path.append("..")
from processData import loadDataDivided
import KNN

def runLMNN(X_train, X_test, y_train, t_test, k):
    transformer = LMNN(k=k, learn_rate=1e-6, convergence_tol=0.1, verbose=True)
    transformer.fit(X_train, y_train)
    X_train_proj = transformer.transform(X_train)
    X_test_proj = transformer.transform(X_test)
    np.save('X_train_LMNN_'+str(k), X_train_proj)
    np.save('X_test_LMNN_'+str(k), X_test_proj)
    return X_train_proj, X_test_proj

def main():
    k_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    LMNN_k_range = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    X_train, X_test, y_train, y_test = loadDataDivided(ifSubDir=False, ifScale=True, suffix='_LDA')
    for i in LMNN_k_range:
        X_train_proj, X_test_proj = runLMNN(X_train, X_test, y_train, y_test, i)
        KNN.runKNN(X_train_proj, X_test_proj, y_train, y_test, k_range, metric='euclidean', metric_params=None,
        label='_LMNN_euclidean_k='+str(i))

if __name__=='__main__':
    main()
