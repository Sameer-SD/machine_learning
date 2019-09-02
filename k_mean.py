import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
import numpy as np
from sklearn.clu
ster import KMeans

X = np.array([[2, 2], [1, 4], [3, 4], [4, 5], [6, 7]])

plt.scatter(X)
plt.show()