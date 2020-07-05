#%env LOKY_PICKLER='cloudpickle'
import sys
sys.path.append('..')
from utils import dataloader, SVM, TSNE
import numpy as np
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import multiprocessing

def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


class TCA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def fit(self, Xs, Xt):
        '''
        Transform Xs and Xt
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: Xs_new and Xt_new after TCA
        '''
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = np.dot(A.T, K)
        Z /= np.linalg.norm(Z, axis=0)
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return Xs_new, Xt_new

def runTCA():
    pairs = [('Art', 'RealWorld'), ('Clipart', 'RealWorld'), ('Product', 'RealWorld')]
    #kernels = ['primal', 'linear', 'rbf']
    kernels = ['primal']
    #dim_range = [32, 64, 128, 256, 512, 1024, 2048]
    dim_range = [128, 256]
    for p in pairs:
        print("%s->%s" % (p[0], p[1]))
        X_train, y_train, X_test, y_test = dataloader.loadData(p[0], p[1])
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
        for k in kernels:
            for n in dim_range:
                print("kernel: %s, dim: %d" % (k, n))
                if p[0][0] != 'P' and n == 128:
                    continue
                model = TCA(kernel_type=k, dim=n)
                X_train_new, X_test_new = model.fit(X_train, X_test)
                #score = SVM.SVM(X_train_new, X_test_new, y_train, y_test)
                TSNE.draw(X_train_new, X_test_new, p[0][0] + '_' + p[1][0] + '_TCA_' + k + '_' + str(n), p[0] + '->' + p[1])
                #with open('TCA.txt', 'a') as f:
                #    f.write('%s->%s, kernel=%s, dim=%d, with acc=%f\n' % (p[0], p[1], k, n, score))
        print()
        
if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    runTCA()
