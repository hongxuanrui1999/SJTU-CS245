import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append("..")
from processData import loadDataDivided
import SVMmodel

def runLLE(X_train, X_test, y_train, y_test, comp_range, n_neigh):
    rbf_scores = []
    linear_scores = []
    for n_comp in comp_range:
        print("\nn_comp=%d\n"%(n_comp))
        # transformer = LocallyLinearEmbedding(n_neighbors=n_neigh, n_components=n_comp, eigen_solver='dense', n_jobs=8)
        transformer = LocallyLinearEmbedding(n_neighbors=n_neigh, n_components=n_comp, n_jobs=8)
        transformer.fit(X_train)
        X_train_proj = transformer.transform(X_train)
        X_test_proj = transformer.transform(X_test)
        if n_comp == 2:
            np.save('X_train_proj_2d_LLE_' + str(n_neigh), X_train_proj)
            np.save('X_test_proj_2d_LLE_' + str(n_neigh), X_test_proj)
        score_rbf = SVMmodel.runSVM(X_train_proj, X_test_proj, y_train, y_test, SVMmodel.getBestParam('rbf'), 'rbf')
        rbf_scores.append(score_rbf.mean())
        score_linear = SVMmodel.runSVM(X_train_proj, X_test_proj, y_train, y_test, SVMmodel.getBestParam('linear'), 'linear')
        linear_scores.append(score_linear.mean())
    for i, scores in enumerate([rbf_scores, linear_scores]):
        if i == 0:
            kernel = 'rbf'
        elif i == 1:
            kernel = 'linear'
        else:
            kernel = ''
        bestIdx = np.argmax(scores)
        bestNComp = comp_range[bestIdx]
        bestAcc = scores[bestIdx]
        with open('res_LLE_' + kernel + '_' + str(n_neigh) + '.txt', 'w') as f:
            for j in range(len(comp_range)):
                f.write(kernel + ": n_comp = %f, acc = %f\n"%(comp_range[j], scores[j]))
            f.write(kernel + ": Best n_comp = %f\n"%(bestNComp))
            f.write(kernel + ": acc = %f\n"%(bestAcc))
    return rbf_scores, linear_scores

def draw(comp_range, neigh_range, scoresS, kernel):
    lines = []
    plt.figure()
    for scores in scoresS:
        l, = plt.plot(comp_range, scores, 'o-', linewidth=1)
        lines.append(l)
    plt.legend(handles=lines, labels=['n_neigh='+str(n_neigh) for n_neigh in neigh_range], loc='best')
    plt.title('LLE with SVM ' + kernel + ' kernel')
    plt.xlabel('n_components')
    plt.ylabel('Accuracy')
    plt.savefig('LLE_' + kernel + '.jpg')

def main():
    rbf_scoresS = []
    linear_scoresS = []
    comp_range = [2, 3, 50, 100, 500, 1000, 2000]
    neigh_range = [4, 8, 16]
    X_train, X_test, y_train, y_test = loadDataDivided(ifSubDir=True)
    for n_neigh in neigh_range:
        print('n_neigh=%d'%(n_neigh))
        rbf_scores, linear_scores = runLLE(X_train, X_test, y_train, y_test, comp_range, n_neigh)
        rbf_scoresS.append(rbf_scores)
        linear_scoresS.append(linear_scores)
    draw(comp_range, neigh_range, rbf_scoresS, 'rbf')
    draw(comp_range, neigh_range, linear_scoresS, 'linear')

if __name__ == '__main__':
    main()