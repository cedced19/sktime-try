
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np

data_path = "../residuals-fault-detection/data/"
categories = listdir(data_path)
X = pd.DataFrame()
Y = []
for cat in categories:
    tmp = join(data_path, cat)
    files = [f for f in listdir(tmp) if isfile(join(tmp, f))]
    for file in files:
        serie = pd.read_csv(join(tmp, file), names=['t', 'i', 'w'])
        i_data=serie[['i']].values.flatten()
        w_data=serie[['w']].values.flatten()
        d = {'i': [pd.Series(i_data, copy=False)], 'w': [pd.Series(w_data, copy=False)]}
        df_tmp = pd.DataFrame(data=d)
        X = pd.concat([X, df_tmp], ignore_index = True)
        Y.append(cat)

Y = np.array(Y)

print("importation end")
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


X_train, X_test, y_train, y_test = train_test_split(X, Y)
print("split end")

classifier = TimeSeriesForestClassifier()
classifier.fit(X_train, y_train)
print("model fitted")

y_pred = classifier.predict(X_test)
print("model evaluated")
print(accuracy_score(y_test, y_pred))