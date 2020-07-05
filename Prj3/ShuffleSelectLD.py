import numpy as np

f_class_dict = np.load('f_class_dict.npy', allow_pickle=True).item()
SIFT_PATH = 'AwA2-data/SIFT_LD/'
DL_PATH = 'AwA2-data/DL_LD/'

def shuffleSelectorClass(className, tot, prop):
    pool = np.load(DL_PATH + className + '/' + className + '_10001.npy', allow_pickle=True)
    #print(className + '_10001 ', end='')
    for i in range(10002, tot + 1):
        tmp = np.load(DL_PATH + className + '/' + className + '_' + str(i) + '.npy', allow_pickle=True) # 2d np array
        print(className + '_' + str(i) + ' ', end='')
        print(tmp.shape)
        pool = np.concatenate((pool, tmp))
    rand_array = np.arange(pool.shape[0])
    np.random.shuffle(rand_array)
    num = pool.shape[0] // prop
    return pool[rand_array[0: num]] # 2d np array

def shuffleSelector(prop):
    ld = [] # list of selected features of each class in 2d np array
    for className, totalNum in f_class_dict.items():
        print("At: " + className)
        ld.append(shuffleSelectorClass(className, totalNum, prop))
    return np.vstack(ld)

def main():
    lds = shuffleSelector(10)
    print(lds.shape)
    np.save('LD_for_clustering', lds)

if __name__ == '__main__':
    main()
