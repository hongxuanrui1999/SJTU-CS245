import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
from multiprocessing import Process, Queue
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.svm import SVC
from sklearn.decomposition import KernelPCA, IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
import sys
sys.path.append("..")
import SVMmodel

SIFT_PATH = '../AwA2-data/SIFT_LD/'
DL_PATH = '../AwA2-data/DL_LD/'
y_file_name = "../AwA2-data/AwA2-labels.txt"

f_class_dict = np.load('../f_class_dict.npy', allow_pickle=True).item()
ld_sample = np.load('../LD_for_clustering.npy', allow_pickle=True)
dict_list = np.load('../f_class_dict_mul.npy', allow_pickle=True)

class FVProcess(Process):
    def __init__(self, class_dict, k, model, q, idx):
        super(FVProcess, self).__init__()
        self.class_dict = class_dict
        self.k = k
        self.model = model
        self.q = q
        self.i = idx
    
    def run(self):
        feature = []
        for className, totalNum in self.class_dict.items():
            print("SS at %s" % (className))
            for idx in range(10001, totalNum + 1):
                ld = np.load(DL_PATH + className + '/' + className + '_' + str(idx) + '.npy', allow_pickle=True)  # 2d np array
                fv = [np.zeros((1, ld.shape[1])) for i in range(2 * self.k)]
                for i in range(self.k):
                    for des in ld:
                        gamma = self.model.predict_proba(des.reshape(1,-1))[0][i]  # gamma_des(i)
                        mu = self.model.means_[i]
                        sigma = np.diagonal(self.model.covariances_[i])
                        pi = self.model.weights_[i]
                        fv[i * 2] += gamma * (des - mu) / sigma  # of shape (d, )
                        fv[i * 2 + 1] += gamma * (np.square(des - mu) / np.square(sigma) - 1)
                    fv[i * 2] /= ld.shape[0] * math.sqrt(pi)
                    fv[i * 2 + 1] /= ld.shape[0] * math.sqrt(2 * pi)
                fv = np.hstack(fv)
                feature.append(fv)
        self.q.put((self.i, np.vstack(feature)))

def FV(k):
    print("Start clustering")
    model = GaussianMixture(n_components=k, verbose=2, max_iter=500)
    model.fit(ld_sample)
    print("Clustering Ended")

    q = Queue()
    feature = [None for i in range(8)]
    processPool = [FVProcess(dict_list[i], k, model, q, i) for i in range(8)]
    for i in range(8):
        processPool[i].start()
    for i in range(8):
        tmp = q.get()
        feature[tmp[0]] = tmp[1]
    for i in range(8):
        processPool[i].join()

    return np.vstack(feature)

def main():
    k_range = [2]#[4, 8, 16]
    C_range = [[0.001, 5], [0.005, 10]]
    #pca = KernelPCA(n_components=50, kernel='linear')
    pca = IncrementalPCA(n_components=50, batch_size=1000)
    #lda = LinearDiscriminantAnalysis(n_components=40)
    for k in k_range:
        print("FV, k:%d" % (k))
        X = FV(k)
        print(X.shape)
        X = StandardScaler().fit_transform(X)
        
        col_name = ['feature' + str(i) for i in range(X.shape[1])]
        X = pd.DataFrame(data=X, columns=col_name)
        y = pd.read_csv(y_file_name, names=['label'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

        print("PCA")
        pca.fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        for C in C_range:
            linear_score = SVMmodel.runSVM(X_train_pca, X_test_pca, y_train, y_test, C[0], 'linear')
            rbf_score = SVMmodel.runSVM(X_train_pca, X_test_pca, y_train, y_test, C[1], 'rbf')
            with open('res_FV_PCA.txt', "a") as f:
                f.write("FV with k=%d, Z-score, SVM with %s kernel, C=%f, score=%f\n"%(k, 'linear', C[0], linear_score))
                f.write("FV with k=%d, Z-score, SVM with %s kernel, C=%f, score=%f\n" % (k, 'rbf', C[1], rbf_score))

        #print("LDA")
        #lda.fit(X_train, y_train)
        #X_train_lda = lda.transform(X_train)
        #X_test_lda = lda.transform(X_test)
        #for C in C_range:
        #    linear_score = SVMmodel.runSVM(X_train_lda, X_test_lda, y_train, y_test, C[0], 'linear')
        #    rbf_score = SVMmodel.runSVM(X_train_lda, X_test_lda, y_train, y_test, C[1], 'rbf')
        #    with open('res_FV_LDA.txt', "a") as f:
        #        f.write("FV with k=%d, Z-score, SVM with %s kernel, C=%f, score=%f\n"%(k, 'linear', C[0], linear_score))
        #        f.write("FV with k=%d, Z-score, SVM with %s kernel, C=%f, score=%f\n"%(k, 'rbf', C[1], rbf_score))

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end - start)

