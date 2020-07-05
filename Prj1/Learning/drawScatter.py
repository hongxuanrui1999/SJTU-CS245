import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def draw(suffix):
    X_train_2d = pd.DataFrame(np.load('X_train_proj_2d_' + suffix + '.npy'), columns=['x', 'y'])
    X_test_2d = pd.DataFrame(np.load('X_test_proj_2d_' + suffix + '.npy'), columns=['x', 'y'])
    y_train = pd.DataFrame(np.load('../y_train.npy'), columns=['label'])
    y_test = pd.DataFrame(np.load('../y_test.npy'), columns=['label'])
    train_categories = np.unique(y_train)
    train_colors = [plt.cm.tab10(i/float(len(train_categories)-1)) for i in range(len(train_categories))]
    test_categories = np.unique(y_test)
    test_colors = [plt.cm.tab10(i/float(len(test_categories)-1)) for i in range(len(test_categories))]
    train_2d = pd.concat([X_train_2d, y_train], axis=1)
    test_2d = pd.concat([X_test_2d, y_test], axis=1)

    plt.figure()
    for i, label in enumerate(train_categories):
        plt.scatter(train_2d.loc[train_2d.label==label].x, train_2d.loc[train_2d.label==label].y, s=2, cmap=train_colors[i], alpha=0.5)
    plt.title('X_train')
    plt.savefig('X_train_scatter_2d_' + suffix + '.jpg')

    plt.figure()
    for i, label in enumerate(test_categories):
        plt.scatter(test_2d.loc[test_2d.label==label].x, test_2d.loc[test_2d.label==label].y, s=2, cmap=test_colors[i], alpha=0.5)
    plt.title('X_test')
    plt.savefig('X_test_scatter_2d_' + suffix + '.jpg')

def main():
    draw('LLE_4')
    draw('LLE_8')
    draw('LLE_16')
    #draw('TSNE_10.0')
    #draw('TSNE_20.0')
    #draw('TSNE_30.0')
    #draw('TSNE_40.0')
    #draw('TSNE_50.0')

if __name__ == '__main__':
    main()
