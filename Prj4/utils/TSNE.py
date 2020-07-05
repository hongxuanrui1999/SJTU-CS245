import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

colors = [plt.cm.tab10(i / 10.0) for i in range(10)]

def draw(source, target, name, title):
    print("Drawing %s" % (name))
    model = TSNE(n_jobs=8)
    source2D = np.ascontiguousarray(model.fit_transform(source).transpose())
    target2D = np.ascontiguousarray(model.fit_transform(target).transpose())
    plt.figure()
    plt.scatter(source2D[0], source2D[1], marker='o', cmap=colors[0], s=10, alpha=0.5, label='source domain')
    plt.scatter(target2D[0], target2D[1], marker='^', cmap=colors[1], s=10, alpha=0.5, label='target domain')
    plt.legend()
    plt.title(title)
    plt.savefig(name)