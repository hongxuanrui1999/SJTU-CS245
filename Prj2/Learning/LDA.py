import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import sys

sys.path.append("..")
from processData import loadDataDivided
import KNN

def runLDA(X_train, X_test, y_train, y_test, n_comp):
    print("\nn_comp=%d\n"%(n_comp))
    transformer = LinearDiscriminantAnalysis(solver='svd', n_components=n_comp)
    transformer.fit(X_train, y_train)
    X_train_proj = transformer.transform(X_train)
    X_test_proj = transformer.transform(X_test)
    np.save('X_train_LDA', X_train_proj)
    np.save('X_test_LDA', X_test_proj)
    return X_train_proj, X_test_proj

def cosine(x, y):
    s = np.linalg.norm(x, ord=2) * np.linalg.norm(y, ord=2)
    if s == 0:
        return 0
    return 1 - np.dot(x, y) / s

def main():
    dim_range = [40]
    k_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    
    X_train, X_test, y_train, y_test = loadDataDivided(ifSubDir=True, ifScale=True, suffix='')
    for dim in dim_range:
        print("dim: %d, method: LDA, metric: %s" % (dim, "euclidean"))
        X_train_proj, X_test_proj = runLDA(X_train, X_test, y_train, y_test, dim)
        KNN.runKNN(X_train_proj, X_test_proj, y_train, y_test, k_range, metric='euclidean', metric_params=None,
                   label=str(dim) + '_LDA_euclidean')

if __name__ == '__main__':
    main()
