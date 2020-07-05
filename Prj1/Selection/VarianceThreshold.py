import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import sys
sys.path.append("..")
from processData import loadDataDivided
import SVMmodel

def runVarianceThreshold(X_train, X_test, y_train, y_test, comp_range):
    rbf_scores = []
    linear_scores = []
    dimension = []

    for n_comp in comp_range:
        print("\nn_comp=%f\n"%(n_comp))

        selector = VarianceThreshold(threshold=n_comp)
        selector.fit(X_train)
        X_train_sel = selector.transform(X_train)
        X_test_sel = selector.transform(X_test)

        dimension.append(X_train_sel.shape[1])

        score_rbf = SVMmodel.runSVM(X_train_sel, X_test_sel, y_train, y_test, SVMmodel.getBestParam('rbf'), 'rbf')
        rbf_scores.append(score_rbf.mean())
        score_linear = SVMmodel.runSVM(X_train_sel, X_test_sel, y_train, y_test, SVMmodel.getBestParam('linear'), 'linear')
        linear_scores.append(score_linear.mean())
    return rbf_scores, linear_scores, dimension

def draw(comp_range, scores, dimension, kernel):
    bestIdx = np.argmax(scores)
    bestNComp = comp_range[bestIdx]
    bestAcc = scores[bestIdx]
    bestDimension = dimension[bestIdx]
    with open('res_VarianceThreshold_' + kernel + '.txt', 'w') as f:
        for i in range(len(comp_range)):
            f.write(kernel + ": n_comp = %f, acc = %f, dimension = %d\n"%(comp_range[i], scores[i], dimension[i]))
        f.write(kernel + ": Best n_comp = %f\n"%(bestNComp))
        f.write(kernel + ": acc = %f\n"%(bestAcc))
        f.write(kernel + ": dimension = %f\n" % (bestDimension))

    plt.figure()
    plt.plot(comp_range, scores, 'bo-', linewidth=2)
    plt.title('VarianceThreshold with SVM ' + kernel + ' kernel')
    plt.xlabel('threshold')
    plt.ylabel('Accuracy')
    plt.savefig('VarianceThreshold_' + kernel + '.jpg')

def main():
    comp_range = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    X_train, X_test, y_train, y_test = loadDataDivided(ifSubDir=True)
    rbf_scores, linear_scores, dimension = runVarianceThreshold(X_train, X_test, y_train, y_test, comp_range)
    draw(comp_range, rbf_scores, dimension, 'rbf')
    draw(comp_range, linear_scores, dimension, 'linear')

if __name__ == '__main__':
    main()
