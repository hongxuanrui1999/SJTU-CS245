import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from processData import loadDataDivided

class tuningThread(threading.Thread):
    def __init__(self, X_train, X_test, y_train, y_test, C_range, Kernel, k, tag):
        threading.Thread.__init__(self)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.C_range = C_range
        self.Kernel = Kernel
        self.k = k
        self.tag = tag
    def run(self):
        cv_scores = []
        for c in self.C_range:
            print("\nCross-Validation " + self.Kernel + ": C=%f\n"%(c))
            model = SVC(C=c, kernel=self.Kernel, gamma='auto', verbose=False)
            score = cross_val_score(model, self.X_train, self.y_train, cv=self.k, scoring='accuracy')
            cv_scores.append(score.mean())

        bestC = self.C_range[cv_scores.index(max(cv_scores))]
        bestModel = SVC(C=bestC, kernel=self.Kernel, gamma='auto', verbose=False)
        bestModel.fit(self.X_train, self.y_train)
        bestAcc = bestModel.score(self.X_test, self.y_test)

        with open('res_' + self.tag + '_' + self.Kernel + '.txt', 'w') as f:
            for i in range(len(self.C_range)):
                f.write(self.Kernel + ": C = %f, acc = %f\n"%(self.C_range[i], cv_scores[i]))
            f.write(self.Kernel + ": Best C = %f\n"%(bestC))
            f.write(self.Kernel + ": acc = %f\n"%(bestAcc))

        plt.figure()
        plt.plot(self.C_range, cv_scores, 'bo-', linewidth=2)
        plt.title('SVM with ' + self.Kernel + ' kernel')
        plt.xlabel('C')
        if self.tag == 'coarse':
            plt.xscale('log')
            plt.xticks(np.logspace(-6, 6, 13))
        else:
            plt.xticks([self.C_range[0] * i for i in range(12)])
        plt.ylabel('Accuracy')
        plt.savefig(self.tag + 'TuningParam_' + self.Kernel + '.jpg')
    
def coarseTuning(k=5):
    C_range = np.logspace(-5, 5, 11)
    X_train, X_test, y_train, y_test = loadDataDivided()
    rbfT = tuningThread(X_train, X_test, y_train, y_test, C_range, 'rbf', k, 'coarse')
    linearT = tuningThread(X_train, X_test, y_train, y_test, C_range, 'linear', k, 'coarse')

    rbfT.start()
    linearT.start()
    rbfT.join()
    linearT.join()

def fineTuning(k=5):
    C_range_rbf = [5.0, 10.0, 20.0, 50.0]
    C_range_linear = [0.0005, 0.001, 0.002, 0.005]
    X_train, X_test, y_train, y_test = loadDataDivided()
    rbfT = tuningThread(X_train, X_test, y_train, y_test, C_range_rbf, 'rbf', k, 'fine')
    linearT = tuningThread(X_train, X_test, y_train, y_test, C_range_linear, 'linear', k, 'fine')

    rbfT.start()
    linearT.start()
    rbfT.join()
    linearT.join()

def main():
    # coarseTuning()
    fineTuning()

if __name__ == '__main__':
    main()
