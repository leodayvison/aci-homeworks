import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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

'''
O mesmo pré-processamento exigido para treino e teste:
A padronização transforma os preditores para que tenham média igual a zero e variância igual a um. 
O código calcula a média e o desvio padrão exclusivamente no conjunto de treinamento (fit_transform) 
e usa esses exatos parâmetros para escalar o conjunto de teste (transform). 
Isso garante que não haja vazamento de informações (data leakage) do teste para o modelo.
'''
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#DEFINIÇÃO DAS FUNÇÕES (DO ZERO)

# Implementação do OLS do zero
def add_intercept(X_mat):
    """Adiciona uma coluna de 1s na matriz X para o intercepto (Beta 0)"""
    return np.c_[np.ones(X_mat.shape[0]), X_mat]

def ols_fit(X_mat, y_vec):
    """Calcula os coeficientes analiticamente: Beta = (X^T X)^-1 X^T y"""
    X_b = add_intercept(X_mat)
    return np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_vec)

def ols_predict(X_mat, beta):
    """Realiza predições utilizando os coeficientes calculados"""
    return add_intercept(X_mat).dot(beta)

# Métricas do Zero
'''
Avaliar a precisão do modelo de regressão, comparando o que o modelo previu (y_pred) 
com os valores reais (y_true).
O RMSE mede a distância média entre as previsões e os reais, na mesma unidade da 
variável. Quanto menor o RMSE, melhor. O R² diz que porcentagem da variação 
na qualidade o modelo consegue explicar através dos preditores.
'''
def rmse_custom_func(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def r2_custom_func(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)


#TREINAMENTO DOS MODELOS

# Treinando OLS Customizado (Do Zero)
beta_custom = ols_fit(X_train_scaled, y_train)

# Treinando com a função integrada (Sklearn)
lr_sklearn = LinearRegression()
lr_sklearn.fit(X_train_scaled, y_train)


#VALIDAÇÃO CRUZADA (CV 5-FOLD)

K = 5
indices = np.arange(X_train_scaled.shape[0])
np.random.seed(42)
np.random.shuffle(indices)
folds = np.array_split(indices, K)

cv_rmse_zero, cv_r2_zero = [], []
cv_rmse_sk, cv_r2_sk = [], []

kf = KFold(n_splits=K, shuffle=True, random_state=42)

for i, (train_idx_sk, val_idx_sk) in enumerate(kf.split(X_train_scaled)):
    # CV do zero
    val_idx = folds[i]
    train_idx = np.hstack([folds[j] for j in range(K) if j != i])
    
    X_f_train_zero, y_f_train_zero = X_train_scaled[train_idx], y_train[train_idx]
    X_f_val_zero, y_f_val_zero = X_train_scaled[val_idx], y_train[val_idx]
    
    beta_f = ols_fit(X_f_train_zero, y_f_train_zero)
    preds_f_zero = ols_predict(X_f_val_zero, beta_f)
    
    cv_rmse_zero.append(rmse_custom_func(y_f_val_zero, preds_f_zero))
    cv_r2_zero.append(r2_custom_func(y_f_val_zero, preds_f_zero))

    # CV Sklearn
    X_f_train_sk, X_f_val_sk = X_train_scaled[train_idx_sk], X_train_scaled[val_idx_sk]
    y_f_train_sk, y_f_val_sk = y_train[train_idx_sk], y_train[val_idx_sk]
    
    lr_sk_fold = LinearRegression()
    lr_sk_fold.fit(X_f_train_sk, y_f_train_sk)
    preds_f_sk = lr_sk_fold.predict(X_f_val_sk)
    
    cv_rmse_sk.append(np.sqrt(mean_squared_error(y_f_val_sk, preds_f_sk)))
    cv_r2_sk.append(r2_score(y_f_val_sk, preds_f_sk))

media_rmse_zero, media_r2_zero = np.mean(cv_rmse_zero), np.mean(cv_r2_zero)
media_rmse_sk, media_r2_sk = np.mean(cv_rmse_sk), np.mean(cv_r2_sk)


#AVALIAÇÃO NO CONJUNTO DE TESTE (FINAL)

y_pred_test_custom = ols_predict(X_test_scaled, beta_custom)
rmse_teste_final = rmse_custom_func(y_test, y_pred_test_custom)
r2_teste_final = r2_custom_func(y_test, y_pred_test_custom)


#RESULTADOS

print("======================================================")
print("1. COMPARAÇÃO DE COEFICIENTES: OLS DO ZERO VS SKLEARN")
print("======================================================")
print(f"Intercepto do Zero: {beta_custom[0]:.6f}")
print(f"Intercepto Sklearn: {lr_sklearn.intercept_:.6f}")
print(f"Diferença máxima nos coeficientes (beta_1 a beta_n): {np.max(np.abs(beta_custom[1:] - lr_sklearn.coef_)):.6e}")
print("* Nota: Se a diferença for próxima de 0, a implementação do zero está matematicamente equivalente.\n")

print("======================================================")
print("2. VALIDAÇÃO CRUZADA (5-FOLD): ESTIMATIVAS DE ERRO")
print("======================================================")
print(f"CV do Zero  -> RMSE Médio: {media_rmse_zero:.4f} | R2 Médio: {media_r2_zero:.4f}")
print(f"CV Sklearn  -> RMSE Médio: {media_rmse_sk:.4f} | R2 Médio: {media_r2_sk:.4f}\n")

print("======================================================")
print("3. DESEMPENHO NO CONJUNTO DE TESTE REAL")
print("======================================================")
print(f"Desempenho Final -> RMSE: {rmse_teste_final:.4f} | R2: {r2_teste_final:.4f}\n")

print("======================================================")
print("4. CONCLUSÃO: COMPARAÇÃO CV VS TESTE REAL")
print("======================================================")
print(f"Estimativa da CV (RMSE): {media_rmse_zero:.4f}")
print(f"Erro Real no Teste (RMSE): {rmse_teste_final:.4f}")
print("-> Os valores devem ser próximos. Se o erro no teste for significativamente maior, houve overfitting.")
print("======================================================\n")


#GRÁFICO

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test_custom, color='blue', alpha=0.6, edgecolors='k', label='Predições (Modelo do Zero)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Linha Ideal (Real = Predito)')

plt.title('Regressão Linear (OLS do Zero): Valores Reais vs. Preditos', fontsize=14)
plt.xlabel('Valores Reais (y_test)', fontsize=12)
plt.ylabel('Valores Preditos (y_pred_test)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()