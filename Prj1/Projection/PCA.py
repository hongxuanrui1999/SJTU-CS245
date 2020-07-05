import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append("..")
from processData import loadDataDivided
import SVMmodel

def runPCA(X_train, X_test, y_train, y_test, comp_range, Kernel):
    C = SVMmodel.getBestParam(Kernel)
    scores = []
    for n_comp in comp_range:
        print("\nn_comp=%d\n"%(n_comp))
        transformer = KernelPCA(n_components=n_comp, kernel=Kernel, copy_X=True, n_jobs=8)
        transformer.fit(X_train)
        X_train_proj = transformer.transform(X_train)
        X_test_proj = transformer.transform(X_test)
        if n_comp == 2:
            np.save('X_train_proj_2d_' + Kernel, X_train_proj)
            np.save('X_test_proj_2d_' + Kernel, X_test_proj)
        score = SVMmodel.runSVM(X_train_proj, X_test_proj, y_train, y_test, C, Kernel)
        scores.append(score.mean())
        print(scores)
    return scores

def draw(comp_range, scores, kernel):
    bestIdx = np.argmax(scores)
    bestNComp = comp_range[bestIdx]
    bestAcc = scores[bestIdx]
    with open('res_PCA_' + kernel + '.txt', 'w') as f:
        for i in range(len(comp_range)):
            f.write(kernel + ": n_comp = %f, acc = %f\n"%(comp_range[i], scores[i]))
        f.write(kernel + ": Best n_comp = %f\n"%(bestNComp))
        f.write(kernel + ": acc = %f\n"%(bestAcc))

    plt.figure()
    plt.plot(comp_range, scores, 'bo-', linewidth=2)
    plt.title('PCA with ' + kernel)
    plt.xlabel('n_components')
    plt.ylabel('Accuracy')
    plt.savefig('PCA_' + kernel + '.jpg')

def main():
    comp_range = [2, 5, 10, 20, 50, 100, 200, 500, 750, 1000, 1200, 1500, 2000]
    X_train, X_test, y_train, y_test = loadDataDivided(ifSubDir=True)
    rbf_scores = runPCA(X_train, X_test, y_train, y_test, comp_range, 'rbf')
    linear_scores = runPCA(X_train, X_test, y_train, y_test, comp_range, 'linear')
    draw(comp_range, rbf_scores, 'rbf')
    draw(comp_range, linear_scores, 'linear')

if __name__ == '__main__':
    main()
