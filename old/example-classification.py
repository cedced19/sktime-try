from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.datasets import load_arrow_head
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np

X, y = load_arrow_head()


X_train, X_test, y_train, y_test = train_test_split(X, y)
classifier = TimeSeriesForestClassifier()
classifier.fit(X_train, y_train)
print(X_train, y_train)
print(type(X_train))
y_pred = classifier.predict(X_test)
print(accuracy_score(y_test, y_pred))