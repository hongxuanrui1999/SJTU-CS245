# encoding=utf-8
import sys
sys.path.append('..')
from utils import dataloader, SVM, TSNE
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import multiprocessing

def kernel(ker, X1, X2, gamma):
    K = None
    if ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1), np.asarray(X2))
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1))
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1), np.asarray(X2), gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1), None, gamma)
    return K

class KMM:
    def __init__(self, kernel_type='linear', gamma=1.0, B=1.0, eps=None):
        '''
        Initialization function
        :param kernel_type: 'linear' | 'rbf'
        :param gamma: kernel bandwidth for rbf kernel
        :param B: bound for beta
        :param eps: bound for sigma_beta
        '''
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.B = B
        self.eps = eps

    def fit(self, Xs, Xt):
        '''
        Fit source and target using KMM (compute the coefficients)
        :param Xs: ns * dim
        :param Xt: nt * dim
        :return: Coefficients (Pt / Ps) value vector (Beta in the paper)
        '''
        ns = Xs.shape[0]
        nt = Xt.shape[0]
        if self.eps == None:
            self.eps = self.B / np.sqrt(ns)
        K = kernel(self.kernel_type, Xs, None, self.gamma)
        kappa = np.sum(kernel(self.kernel_type, Xs, Xt, self.gamma) * float(ns) / float(nt), axis=1)

        K = matrix(K)
        kappa = matrix(kappa)
        G = matrix(np.r_[np.ones((1, ns)), -np.ones((1, ns)), np.eye(ns), -np.eye(ns)])
        h = matrix(np.r_[ns * (1 + self.eps), ns * (self.eps - 1), self.B * np.ones((ns,)), np.zeros((ns,))])

        sol = solvers.qp(K, -kappa, G, h)
        beta = np.array(sol['x'])
        return beta

def runKMM():
    pairs = [('Art', 'RealWorld'), ('Clipart', 'RealWorld'), ('Product', 'RealWorld')]
    kernels = ['linear', 'rbf']
    params_dict = {'ARC': 10.0, 'ARK': 'rbf', 'CRC': 0.01, 'CRK': 'linear', 'PRC': 10.0, 'PRK': 'rbf'}
    for p in pairs:
        print("%s->%s" % (p[0], p[1]))
        X_train, y_train, X_test, y_test = dataloader.loadData(p[0], p[1])
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
        for k in kernels:
            print("kernel: %s" % (k))
            model = KMM(kernel_type=k)
            weight = model.fit(X_train, X_test)
            # plt.figure()
            # plt.plot(weight)
            # plt.title(p[0] + '->' + p[1])
            # plt.ylabel('weight')
            # plt.savefig(p[0][0] + '_' + p[1][0] + '_weight_' + k)
            clf = SVC(C=params_dict[p[0][0] + p[1][0] + 'C'], kernel=params_dict[p[0][0] + p[1][0] + 'K'])
            clf.fit(X_train, y_train, weight.reshape(-1))
            score = clf.score(X_test, y_test)
            with open('KMM.txt', 'a') as f:
               f.write('%s->%s, kernel=%s, with acc=%f\n' % (p[0], p[1], k, score))
        print()

if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    runKMM()