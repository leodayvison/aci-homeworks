import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

csv_filename = 'winequality-white.csv' 

try:
    df = pd.read_csv(csv_filename, sep=';')
except FileNotFoundError:
    print(f"Erro: O arquivo '{csv_filename}' não foi encontrado na pasta atual.")
    exit()

# A última coluna é o alvo (quality - Y) e as demais os preditores (X).
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train['chlorides'] = preprocessing.normalize(X_train['chlorides'], norm='l2')