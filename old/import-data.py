from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np

data_path = "../residuals-fault-detection/data/"
categories = listdir(data_path)
X = []
Y = []
for cat in categories:
    tmp = join(data_path, cat)
    files = [f for f in listdir(tmp) if isfile(join(tmp, f))]
    for file in files:
        serie = pd.read_csv(join(tmp, file), names=['t', 'i', 'w'])
        serie = serie.set_index('t') 
        X.append(serie)
        Y.append(cat)

np.save('imported-data.npy', (X, Y))