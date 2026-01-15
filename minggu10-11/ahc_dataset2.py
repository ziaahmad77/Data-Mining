import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Dataset 2
X = np.array([
    [1, 1],
    [4, 1],
    [6, 1],
    [1, 2],
    [2, 3],
    [5, 3],
    [2, 5],
    [3, 5],
    [2, 6],
    [3, 8]
])

labels = [str(i) for i in range(1, 11)]

# Average linkage
Z = linkage(X, method='average', metric='euclidean')

plt.figure(figsize=(8, 5))
dendrogram(Z, labels=labels)
plt.title('AHC Average Linkage (Dataset 2)')
plt.xlabel('Data')
plt.ylabel('Jarak')
plt.show()

