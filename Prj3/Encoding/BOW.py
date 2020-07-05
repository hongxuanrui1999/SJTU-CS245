import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from multiprocessing import Process, Queue
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sys
sys.path.append("..")
import SVMmodel

SIFT_PATH = '../AwA2-data/SIFT_LD/'
DL_PATH = '../AwA2-data/DL_LD/'
y_file_name = "../AwA2-data/AwA2-labels.txt"

f_class_dict = np.load('../f_class_dict.npy', allow_pickle=True).item()
ld_sample = np.load('../LD_for_clustering.npy', allow_pickle=True)
dict_list = np.load('../f_class_dict_mul.npy', allow_pickle=True)

class BOWProcess(Process):
    def __init__(self, class_dict, k, model, q, idx):
        super(BOWProcess, self).__init__()
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
                bow = np.zeros((1, self.k))
                for des in ld:
                    bow[0][self.model.predict(des.reshape(1, -1))[0]] += 1
                feature.append(bow)
        self.q.put((self.i, np.vstack(feature)))

def BOW(k):
    print("Start clustering")
    model = KMeans(n_clusters=k, copy_x=False, n_jobs=8)
    model.fit(ld_sample)
    print("Clustering Ended")

    q = Queue()
    feature = [None for i in range(8)]
    processPool = [BOWProcess(dict_list[i], k, model, q, i) for i in range(8)]
    for i in range(8):
        processPool[i].start()
    for i in range(8):
        tmp = q.get()
        feature[tmp[0]] = tmp[1]
    for i in range(8):
        processPool[i].join()

    return np.vstack(feature)

def main():
    k_range = [8, 16, 32, 64, 128, 256, 512, 1024]
    C_range = [[0.001, 5], [0.005, 10]]
    for k in k_range:
        print("BOW, k:%d" % (k))
        X = BOW(k)
        print(X.shape)

        col_name = ['feature' + str(i) for i in range(k)]
        y = pd.read_csv(y_file_name, names=['label'])

        # X_scaled = MinMaxScaler().fit_transform(X)
        X_scaled = StandardScaler().fit_transform(X)
        X_scaled = pd.DataFrame(data=X_scaled, columns=col_name)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4)
        for C in C_range:
            linear_score = SVMmodel.runSVM(X_train, X_test, y_train, y_test, C[0], 'linear')
            rbf_score = SVMmodel.runSVM(X_train, X_test, y_train, y_test, C[1], 'rbf')
            with open('res_BOW.txt', "a") as f:
                f.write("BOW with k=%d, scale=%s, SVM with %s kernel, C=%f, score=%f\n"%(k, 'Z-score', 'linear', C[0], linear_score))
                f.write("BOW with k=%d, scale=%s, SVM with %s kernel, C=%f, score=%f\n"%(k, 'Z-score', 'rbf', C[1], rbf_score))

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end - start)
