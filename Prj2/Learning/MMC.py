import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metric_learn import MMC_Supervised
import sklearn
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import sys

sys.path.append("..")
from processData import loadDataDivided
import KNN

def runMMC(X_train, X_test, y_train, y_test):
    transformer = MMC_Supervised(num_constraints=200, diagonal=True, verbose=True)
    transformer.fit(X_train, y_train)
    X_train_proj = transformer.transform(X_train)
    X_test_proj = transformer.transform(X_test)
    np.save('X_train_MMC', X_train_proj)
    np.save('X_test_MMC', X_test_proj)
    return X_train_proj, X_test_proj

def cosine(x, y):
    s = np.linalg.norm(x, ord=2) * np.linalg.norm(y, ord=2)
    if s == 0:
        return 0
    return 1 - np.dot(x, y) / s

def main():
    k_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    X_train, X_test, y_train, y_test = loadDataDivided(ifSubDir=False, ifScale=True, suffix='_LDA')
    X_train_proj, X_test_proj = runMMC(X_train, X_test, y_train, y_test)
    KNN.runKNN(X_train_proj, X_test_proj, y_train, y_test, k_range, metric='euclidean', metric_params=None,
                label='_MMC_euclidean')

if __name__ == '__main__':
    main()
