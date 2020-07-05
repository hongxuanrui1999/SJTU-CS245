import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import sys
sys.path.append("..")
from processData import loadDataDivided
import KNN

def cosine(x, y):
    s = np.linalg.norm(x, ord=2) * np.linalg.norm(y, ord=2)
    if s == 0:
        return 0
    return 1 - np.dot(x, y) / s

def main():
    #kernel_range = ['linear', 'rbf', 'poly', 'sigmoid', 'cosine']
    #dim_range = [50, 500, 2048]
    #k_range = [9]
    #metric_range = ['euclidean', 'manhattan', 'chebyshev']
    #for dim in dim_range:
    #    for metric in metric_range:
    #        if dim != 2048:
    #            for kernel in kernel_range:
    #                print("dim: %d, kernel: %s, metric: %s" % (dim, kernel, metric))
    #                X_train, X_test, y_train, y_test = loadDataDivided(ifSubDir=True, ifScale=False, suffix='_' + str(dim) + '_' + kernel)
    #                KNN.runKNN(X_train, X_test, y_train, y_test, k_range, metric=metric, metric_params=None, label=str(dim) + '_' + kernel + '_' + metric + '_9')
    #        else:
    #            X_train, X_test, y_train, y_test = loadDataDivided(ifSubDir=True, ifScale=False, suffix='')
    #            print("dim: %d, metric: %s" % (dim, metric))
    #            KNN.runKNN(X_train, X_test, y_train, y_test, k_range, metric=metric, metric_params=None, label=str(dim) + '_' + metric + '_9')
    
    k_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    #for dim in dim_range:
        #if dim != 2048:
        #    for kernel in kernel_range:
        #        print("dim: %d, kernel: %s, metric: %s" % (dim, kernel, "cosine"))
        #        X_train, X_test, y_train, y_test = loadDataDivided(ifSubDir=True, ifScale=False, suffix='_' + str(dim) + '_' + kernel)
        #        KNN.runKNN(X_train, X_test, y_train, y_test, k_range, metric=cosine, metric_params=None, label=str(dim) + '_' + kernel + '_cosine1')
        #else:
    X_train, X_test, y_train, y_test = loadDataDivided(ifSubDir=False, ifScale=False, suffix='')
    print("dim: %d, metric: %s" % (2048, "cosine"))
    KNN.runKNN(X_train, X_test, y_train, y_test, k_range, metric=cosine, metric_params=None, label=str(2048) + '_cosine1')

    #print("Checkpoint")

    #k_range = [13, 14, 15, 16, 50, 100, 200, 500, 1000]
    #for dim in dim_range:
    #    if dim != 2048:
    #        for kernel in kernel_range:
    #            print("dim: %d, kernel: %s, metric: %s" % (dim, kernel, "cosine"))
    #            X_train, X_test, y_train, y_test = loadDataDivided(ifSubDir=True, ifScale=False, suffix='_' + str(dim) + '_' + kernel)
    #            KNN.runKNN(X_train, X_test, y_train, y_test, k_range, metric=cosine, metric_params=None, label=str(dim) + '_' + kernel + '_cosine2')
    #    else:
    #        X_train, X_test, y_train, y_test = loadDataDivided(ifSubDir=True, ifScale=False, suffix='')
    #        print("dim: %d, metric: %s" % (dim, "cosine"))
    #        KNN.runKNN(X_train, X_test, y_train, y_test, k_range, metric=cosine, metric_params=None, label=str(dim) + '_cosine2')

if __name__ == '__main__':
    main()
