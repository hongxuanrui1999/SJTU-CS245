import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
import sys
from processData import loadDataDivided

def runPCA(X_train, X_test, comp_range, Kernel):
    for n_comp in comp_range:
        print("\nn_comp=%d\n" % (n_comp))
        transformer = KernelPCA(n_components=n_comp, kernel=Kernel, copy_X=True, n_jobs=8)
        transformer.fit(X_train)
        X_train_proj = transformer.transform(X_train)
        X_test_proj = transformer.transform(X_test)
        np.save('X_train_' + str(n_comp) + '_' + Kernel, X_train_proj)
        np.save('X_test_' + str(n_comp) + '_' + Kernel, X_test_proj)

def main():
    comp_range = [50, 500]
    X_train, X_test, y_train, y_test = loadDataDivided(ifSubDir=False)
    for kernel in ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']:
        print("kernel: %s" % (kernel))
        runPCA(X_train, X_test, comp_range, kernel)

if __name__ == '__main__':
    main()
