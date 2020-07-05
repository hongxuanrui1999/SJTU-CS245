# encoding=utf-8
import sys
sys.path.append('..')
from utils import dataloader, SVM, TSNE
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
import sklearn.neighbors
from sklearn.preprocessing import StandardScaler
import multiprocessing

class CORAL:
    def __init__(self):
        super(CORAL, self).__init__()

    def fit(self, Xs, Xt):
        '''
        Perform CORAL on the source domain features
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: New source domain features
        '''
        cov_src = np.cov(Xs.T) + np.eye(Xs.shape[1])
        cov_tar = np.cov(Xt.T) + np.eye(Xt.shape[1])
        A_coral = np.dot(scipy.linalg.fractional_matrix_power(cov_src, -0.5),
                         scipy.linalg.fractional_matrix_power(cov_tar, 0.5))
        Xs_new = np.dot(Xs, A_coral)
        return Xs_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Perform CORAL, then predict using 1NN classifier
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted labels of target domain
        '''
        Xs_new = self.fit(Xs, Xt)
        clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt)
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)
        return acc, y_pred

def runCORAL():
    pairs = [('Art', 'RealWorld'), ('Clipart', 'RealWorld'), ('Product', 'RealWorld')]
    for p in pairs:
        print("%s->%s" % (p[0], p[1]))
        X_train, y_train, X_test, y_test = dataloader.loadData(p[0], p[1])
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
        model = CORAL()
        X_train_new = model.fit(X_train, X_test)
        X_test_new = X_test
        score = SVM.SVM(X_train_new, X_test_new, y_train, y_test)
        TSNE.draw(X_train_new, X_test_new, p[0][0] + '_' + p[1][0] + '_CORAL', p[0] + '->' + p[1])
        with open('CORAL.txt', 'a') as f:
            f.write('%s->%s with acc=%f\n' % (p[0], p[1], score))

if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    runCORAL()
