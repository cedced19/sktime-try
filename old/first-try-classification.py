from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.datasets import load_arrow_head
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd

X, y = np.load('imported-data.npy', allow_pickle=True)

#from sktime.datatypes import convert_to
#X=convert_to(X, to_type=["pd-multiindex"])
#for x in X:
#    print(type(x))

X_formatted = pd.DataFrame()
for i in range(len(X)):
    i_data=X[i][['i']].T.squeeze()
    w_data=X[i][['w']].T.squeeze()
    X_formated = pd.DataFrame()
    d = {'i': [i_data], 'w': [w_data]}
    df_tmp = pd.DataFrame(data=d)
    X_formatted = pd.concat([X_formatted, df_tmp], ignore_index = True)



X_train, X_test, y_train, y_test = train_test_split(X_formatted, y)
classifier = TimeSeriesForestClassifier()
print(X_train, y_train)
print(type(X_train))
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(accuracy_score(y_test, y_pred))