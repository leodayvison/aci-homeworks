import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Suporta execução tanto a partir da raiz do repositório quanto de dentro da pasta hw2/
csv_filename = 'winequality-white.csv'
if not os.path.exists(csv_filename) and os.path.exists(os.path.join('..', csv_filename)):
    csv_filename = os.path.join('..', csv_filename)

try:
    df = pd.read_csv(csv_filename, sep=';')
except FileNotFoundError:
    print(f"Erro: O arquivo '{csv_filename}' não foi encontrado na pasta atual.")
    exit()

# A última coluna é o alvo (quality - Y) e as demais os preditores (X).
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cria cópias explícitas para evitar SettingWithCopyWarning
X_train = X_train.copy()
X_test = X_test.copy()


# Transformação logarítmica para as colunas 'residual_sugar' e 'free_sulfur_dioxide', a fim de reduzir a assimetria e melhorar a normalidade dos dados.
X_train['residual_sugar'] = np.log1p(X_train.loc[:, 'residual sugar'])
X_train['free_sulfur_dioxide'] = np.log1p(X_train.loc[:, 'free sulfur dioxide'])

X_test['residual_sugar'] = np.log1p(X_test.loc[:, 'residual sugar'])
X_test['free_sulfur_dioxide'] = np.log1p(X_test.loc[:, 'free sulfur dioxide'])

# Center & scale (Padronização)
scaler = StandardScaler().set_output(transform='pandas')
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# PCA 
pca = PCA(n_components=0.90, random_state=42).set_output(transform='pandas')

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"PCA concluído com sucesso:")
print(f"- Número original de preditores: {X_train_scaled.shape[1]}")
print(f"- Número de componentes retidos: {pca.n_components_}")
print(f"- Variância explicada acumulada: {np.sum(pca.explained_variance_ratio_):.2%}")
print(f"- Variância individual por componente: {np.round(pca.explained_variance_ratio_, 4)}")

