from sklearn.manifold import MDS
import numpy as np

mds = MDS(n_components=2, metric=False)
arr = np.random.randint(0,10, size=[2048,3])
print(mds.fit_transform(arr))