
import numpy as np
import pandas as pd

X, y = np.load('imported-data.npy', allow_pickle=True)


from tslearn.utils import to_time_series_dataset

X_formatted = to_time_series_dataset(X)

from tslearn.neighbors import KNeighborsTimeSeriesClassifier
knn = KNeighborsTimeSeriesClassifier(n_neighbors=1)
knn.fit(X_formatted, y)

print(knn.predict(X_formatted))