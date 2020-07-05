import numpy as np
import pandas as pd
import matplotlib.pyplot as ply
from metric_learn import RCA_Supervised
import sklearn
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.preprocessing import StandardScaler
import sys

sys.path.append("..")
from processData import loadDataDivided
import KNN

def runRCA(X_train, X_test, y_train, t_test):
    transformer = RCA_Supervised()
    transformer.fit(X_train, y_train)
    X_train_proj = transformer.transform(X_train)
    X_test_proj = transformer.transform(X_test)
    np.save('X_train_RCA', X_train_proj)
    np.save('X_test_RCA', X_test_proj)
    return X_train_proj, X_test_proj

def main():
    k_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    X_train, X_test, y_train, y_test = loadDataDivided(ifSubDir=False, ifScale=False, suffix='_LDA')
    X_train_proj, X_test_proj = runRCA(X_train, X_test, y_train, y_test)
    KNN.runKNN(X_train_proj, X_test_proj, y_train, y_test, k_range, metric='euclidean', metric_params=None,
    label='_RCA_euclidean')

if __name__=='__main__':
    main()
