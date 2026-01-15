import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Dataset 1
X = np.array([
    [1, 1],
    [4, 1],
    [1, 2],
    [3, 4],
    [5, 4]
])

labels = ['1', '2', '3', '4', '5']

# Linkage methods
methods = ['single', 'complete', 'average']

plt.figure(figsize=(15, 5))

for i, method in enumerate(methods):
    plt.subplot(1, 3, i + 1)
    Z = linkage(X, method=method, metric='euclidean')
    dendrogram(Z, labels=labels)
    plt.title(f'AHC - {method.capitalize()} Linkage')
    plt.xlabel('Data')
    plt.ylabel('Jarak')

plt.tight_layout()
plt.show()
