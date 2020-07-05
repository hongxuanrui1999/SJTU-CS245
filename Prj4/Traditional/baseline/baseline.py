import sys
sys.path.append('..')
from utils import dataloader, SVM, TSNE

def baseline():
    pairs = [('Art', 'RealWorld'), ('Clipart', 'RealWorld'), ('Product', 'RealWorld')]
    for p in pairs:
        print("%s->%s" % (p[0], p[1]))
        X_train, y_train, X_test, y_test = dataloader.loadData(p[0], p[1])
        score = SVM.SVM(X_train, X_test, y_train, y_test)
        # TSNE.draw(X_train, X_test, p[0][0] + '_' + p[1][0] + '_baseline', p[0] + '->' + p[1])
        # with open('baseline.txt', 'a') as f:
        #     f.write('%s->%s with acc=%f\n' % (p[0], p[1], score))
        print()

baseline()