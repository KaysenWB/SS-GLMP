import pickle
import numpy as np
import matplotlib.pyplot as plt


# args
map_root = '/home/user/Documents/Yangkaisen/GCN_Informer_test/map/map_Aarea.png'
Preds = np.load('/output/TCNN/Preds.npy')
Reals = np.load('/output/TCNN/Reals.npy')
st = 0  # how many ship to visual
ed = 20
K = 20
observed = 32
"""
for i in range(50):
    plt.scatter(Reals[:, i, 0], Reals[:, i, 1], c='b', s=3)
    imp = plt.imread(map_root)
    plt.imshow(imp, extent=[114.099003, 114.187537, 22.265695, 22.322062])
    plt.show()
    print(';')
"""

# show
mean_true = Reals.mean(axis=(0, 1), keepdims = True)
std_true = Reals.std(axis=(0, 1), keepdims = True)
mean_true = mean_true[:, :, :2]
std_true = std_true[:, :, :2]

Preds = Preds * std_true + mean_true
Reals = Reals[observed:,:, :2]


for i in range(16):
    plt.scatter(Reals[:, i, 0], Reals[:, i, 1],c='b',s=3)
    plt.scatter(Preds[:, i, 0], Preds[:, i, 1],c='r', s=3)
    imp = plt.imread(map_root)
    plt.imshow(imp, extent=[114.099003, 114.187537, 22.265695, 22.322062])
    plt.show()
    print(';')


