import numpy as np
import threading
from multiprocessing import Process, Queue

f_class_dict = np.load('f_class_dict.npy', allow_pickle=True).item()
dict_list = np.load('f_class_dict_mul.npy', allow_pickle=True)

class BOWProcess(Process):
    def __init__(self, class_dict, q, i):  # model
        super(BOWProcess, self).__init__()
        self.class_dict = class_dict
        self.q = q
        self.i = i
    
    def run(self):
        feature = []
        for className, totalNum in self.class_dict.items():
            print("SS at %s" % (className))
            for idx in range(10001, totalNum + 1):
                feature.append(className + '_' + str(idx))
        self.q.put((self.i, np.vstack(feature)))

def divideDict():
    dict_list = [{} for i in range(8)]
    i = 0
    for className, totalNum in f_class_dict.items():
        dict_list[i // 7][className] = totalNum
        i += 1
    np.save('f_class_dict_mul', dict_list)

def main():
    q = Queue()
    feature = [None for i in range(8)]
    threadPool = [BOWProcess(dict_list[i], q, i) for i in range(8)]
    for i in range(8):
        threadPool[i].start()
    for i in range(8):
        tmp = q.get()
        feature[tmp[0]] = tmp[1]
    for i in range(8):
        threadPool[i].join()
    print(q.empty())
    n = np.vstack(feature)

main()
main()
main()
