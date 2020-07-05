import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.preprocessing import StandardScaler

def runKNN(X_train, X_test, y_train, y_test, k_range, metric, metric_params=None, label=""):
    scores = []
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k, metric=metric, metric_params=metric_params, n_jobs=8)
        model.fit(X_train, y_train)
        print("train finished")
        score = model.score(X_test, y_test)
        print("k=%d, score=%f" % (k, score))
        scores.append(score)

    bestIdx = np.argmax(scores)
    bestK = k_range[bestIdx]
    bestAcc = scores[bestIdx]
    with open('res_KNN_' + label + '.txt', 'w') as f:
        for i in range(len(k_range)):
            f.write(label + ": k = %d, acc = %f\n" % (k_range[i], scores[i]))
        f.write(label + ": Best k = %d\n" % (bestK))
        f.write(label + ": Best acc = %f\n" % (bestAcc))
    return scores
