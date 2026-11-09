import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Importa o pré-processamento padronizado (log1p + StandardScaler + PCA com 90% variância)
try:
    from preproc import load_and_preprocess_data
except ImportError:
    from hw2.preproc import load_and_preprocess_data

X_train_scaled, X_test_scaled, X_train_transf, X_test_transf, y_train, y_test, pca, scaler = load_and_preprocess_data(
    pca_variance=0.90, as_numpy=True
)


#FUNÇÕES DA REGRESSÃO RIDGE (L2) DO ZERO
def ridge_fit_scratch(X, y, lambda_val):
    """Calcula coeficientes: Beta = (X^T X + lambda*I)^-1 X^T y"""
    n, p = X.shape
    X_b = np.c_[np.ones(n), X]
    
    # Matriz Identidade para a penalidade
    I_b = np.eye(p + 1)
    I_b[0, 0] = 0 # O intercepto beta_0 não recebe penalização L2
    
    beta = np.linalg.inv(X_b.T.dot(X_b) + lambda_val * I_b).dot(X_b.T).dot(y)
    return beta

def predict_scratch(X, beta):
    X_b = np.c_[np.ones(X.shape[0]), X]
    return X_b.dot(beta)

def rmse_scratch(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def r2_scratch(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    if ss_tot == 0: return 0 
    return 1 - (ss_res / ss_tot)


#VALIDAÇÃO CRUZADA (5-FOLD) E BUSCA DO LAMBDA
K = 5
lambdas = np.logspace(-1, 4, 50) #(50 valores)

# Para CV do Zero
indices = np.arange(X_train_transf.shape[0])
np.random.seed(42)
np.random.shuffle(indices)
folds = np.array_split(indices, K)

cv_rmse_scratch, cv_r2_scratch = [], []
cv_rmse_sklearn, cv_r2_sklearn = [], []

kf = KFold(n_splits=K, shuffle=True, random_state=42)

for lam in lambdas:
    fold_rmse, fold_r2 = [], []
    fold_rmse_sk, fold_r2_sk = [], []
    
    # Modelo do Sklearn para comparação no mesmo loop
    ridge_sk = Ridge(alpha=lam)
    
    # A. Validação Cruzada do Zero
    for i in range(K):
        val_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(K) if j != i])
        
        X_f_train, y_f_train = X_train_transf[train_idx], y_train[train_idx]
        X_f_val, y_f_val = X_train_transf[val_idx], y_train[val_idx]
        
        beta_fold = ridge_fit_scratch(X_f_train, y_f_train, lam)
        preds_fold = predict_scratch(X_f_val, beta_fold)
        
        fold_rmse.append(rmse_scratch(y_f_val, preds_fold))
        fold_r2.append(r2_scratch(y_f_val, preds_fold))
        
    cv_rmse_scratch.append(np.mean(fold_rmse))
    cv_r2_scratch.append(np.mean(fold_r2))
    
    # B. Validação Cruzada Sklearn
    for train_idx_sk, val_idx_sk in kf.split(X_train_transf):
        X_f_train_sk, X_f_val_sk = X_train_transf[train_idx_sk], X_train_transf[val_idx_sk]
        y_f_train_sk, y_f_val_sk = y_train[train_idx_sk], y_train[val_idx_sk]
        
        ridge_sk.fit(X_f_train_sk, y_f_train_sk)
        preds_sk = ridge_sk.predict(X_f_val_sk)
        
        fold_rmse_sk.append(np.sqrt(mean_squared_error(y_f_val_sk, preds_sk)))
        fold_r2_sk.append(r2_score(y_f_val_sk, preds_sk))
        
    cv_rmse_sklearn.append(np.mean(fold_rmse_sk))
    cv_r2_sklearn.append(np.mean(fold_r2_sk))

# Encontrar o Lambda ideal (o que minimiza o RMSE do zero)
best_lambda_idx = np.argmin(cv_rmse_scratch)
best_lambda_scratch = lambdas[best_lambda_idx]


#AVALIAÇÃO FINAL NO CONJUNTO DE TESTE
# Treina o modelo final do zero com o melhor Lambda em TODOS os dados de treino
beta_final = ridge_fit_scratch(X_train_transf, y_train, best_lambda_scratch)
y_pred_test_scratch = predict_scratch(X_test_transf, beta_final)

final_rmse = rmse_scratch(y_test, y_pred_test_scratch)
final_r2 = r2_scratch(y_test, y_pred_test_scratch)

# Compara com o Sklearn treinado no melhor Lambda
ridge_final_sk = Ridge(alpha=best_lambda_scratch)
ridge_final_sk.fit(X_train_transf, y_train)


#RESULTADOS
print("======================================================")
print("1. HIPERPARÂMETRO IDEAL ENCONTRADO PELA CV")
print("======================================================")
print(f"Melhor Lambda (L2 Penalty): {best_lambda_scratch:.4f}\n")

print("======================================================")
print("2. COMPARAÇÃO DE COEFICIENTES: RIDGE DO ZERO VS SKLEARN")
print("======================================================")
print(f"Intercepto Zero:    {beta_final[0]:.6f}")
print(f"Intercepto Sklearn: {ridge_final_sk.intercept_:.6f}")
print(f"Diferença Máxima nos Coeficientes: {np.max(np.abs(beta_final[1:] - ridge_final_sk.coef_)):.6e}\n")

print("======================================================")
print("3. DESEMPENHO FINAL NO CONJUNTO DE TESTE")
print("======================================================")
print(f"RMSE (Teste): {final_rmse:.4f}")
print(f"R2 (Teste):   {final_r2:.4f}")
print("======================================================\n")


#Graficos
plt.figure(figsize=(16, 5))

# Gráfico 1: RMSE na Validação Cruzada
plt.subplot(1, 3, 1)
plt.plot(lambdas, cv_rmse_scratch, marker='o', label='RMSE do Zero', color='blue')
plt.plot(lambdas, cv_rmse_sklearn, linestyle='--', label='RMSE Sklearn', color='orange')
plt.axvline(best_lambda_scratch, color='red', linestyle=':', label=rf'Melhor $\lambda$={best_lambda_scratch:.2f}')
plt.xscale('log')
plt.xlabel(r'$\lambda$ (Penalidade L2)')
plt.ylabel('RMSE')
plt.title('Perfil da Validação Cruzada (RMSE)')
plt.legend()
plt.grid(True)

# Gráfico 2: R2 na Validação Cruzada
plt.subplot(1, 3, 2)
plt.plot(lambdas, cv_r2_scratch, marker='s', label='R2 do Zero', color='green')
plt.plot(lambdas, cv_r2_sklearn, linestyle='--', label='R2 Sklearn', color='lime')
plt.axvline(best_lambda_scratch, color='red', linestyle=':', label=rf'Melhor $\lambda$={best_lambda_scratch:.2f}')
plt.xscale('log')
plt.xlabel(r'$\lambda$ (Penalidade L2)')
plt.ylabel(r'$R^2$')
plt.title(r'Perfil da Validação Cruzada ($R^2$)')
plt.legend()
plt.grid(True)

# Gráfico 3: Predição vs Valores Reais no Teste
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_test_scratch, alpha=0.7, edgecolors='k', color='purple', label='Predições')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Linha Ideal')
plt.xlabel('Valores Reais')
plt.ylabel('Valores Preditos')
plt.title('Performance no Teste: Real vs Predito')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()  