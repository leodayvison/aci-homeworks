import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Importa o pré-processamento padronizado (log1p + StandardScaler)
try:
    from preproc import load_and_preprocess_data
except ImportError:
    from hw2.preproc import load_and_preprocess_data

X_train_scaled, X_test_scaled, _, _, y_train, y_test, _, _ = load_and_preprocess_data(pca_variance=None, as_numpy=True)

# Determinar o número máximo de componentes a testar
# Não pode ser maior que o número de features ou o número de amostras
max_components = min(X_train_scaled.shape[1], X_train_scaled.shape[0])
components_range = np.arange(1, max_components + 1)

# 4. Validação Cruzada K-Fold (5 vias) no conjunto de TREINAMENTO
K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)

cv_rmse = []
cv_r2 = []

print("Iniciando Validação Cruzada para encontrar o número ideal de componentes...")

# Loop sobre os diferentes números de componentes do PLS
for n_comp in components_range:
    pls = PLSRegression(n_components=n_comp)
    fold_rmse = []
    fold_r2 = []
    
    # K-Fold CV
    for train_idx, val_idx in kf.split(X_train_scaled):
        # Separar os dados do fold
        X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Treinar o PLS no fold atual
        pls.fit(X_fold_train, y_fold_train)
        
        # Prever no fold de validação
        preds_val = pls.predict(X_fold_val).ravel() # ravel() converte para vetor 1D
        
        # Calcular métricas
        rmse = np.sqrt(mean_squared_error(y_fold_val, preds_val))
        r2 = r2_score(y_fold_val, preds_val)
        
        fold_rmse.append(rmse)
        fold_r2.append(r2)
        
    # Salvar a média do RMSE e R2 para o n_comp atual
    cv_rmse.append(np.mean(fold_rmse))
    cv_r2.append(np.mean(fold_r2))

# 5. Determinar o número ideal de componentes (Aquele que minimiza o RMSE na validação)
best_idx = np.argmin(cv_rmse)
best_n_components = components_range[best_idx]
best_cv_rmse = cv_rmse[best_idx]

print("-" * 50)
print(f"Número ideal de componentes escolhido: {best_n_components}")
print(f"RMSE mínimo na Validação Cruzada: {best_cv_rmse:.4f}")
print("-" * 50)

# 6. Avaliação Final no Conjunto de Teste (Usando o melhor hiperparâmetro)
# Treina o modelo final com todos os dados de treino padronizados
pls_final = PLSRegression(n_components=best_n_components)
pls_final.fit(X_train_scaled, y_train)

# Previsão no conjunto de teste intocado e padronizado
y_pred_test = pls_final.predict(X_test_scaled).ravel()

test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2 = r2_score(y_test, y_pred_test)

print("PERFORMANCE NO CONJUNTO DE TESTE (GENERALIZAÇÃO):")
print(f"RMSE no Teste: {test_rmse:.4f}")
print(f"R2 no Teste:   {test_r2:.4f}")
print("-" * 50)

# 7. Geração dos Gráficos (Perfil da CV e Predição Final)
plt.figure(figsize=(18, 5))

# Gráfico 1: Perfil do RMSE na Validação Cruzada
plt.subplot(1, 3, 1)
plt.plot(components_range, cv_rmse, marker='o', color='blue', label='RMSE Validação')
plt.axvline(best_n_components, color='red', linestyle='--', label=f'Ideal = {best_n_components}')
plt.title('Perfil da Validação Cruzada (RMSE)')
plt.xlabel('Número de Componentes PLS')
plt.ylabel('RMSE (Menor é melhor)')
plt.xticks(components_range)
plt.legend()
plt.grid(True)

# Gráfico 2: Perfil do R2 na Validação Cruzada
plt.subplot(1, 3, 2)
plt.plot(components_range, cv_r2, marker='s', color='green', label=r'$R^2$ Validação')
plt.axvline(best_n_components, color='red', linestyle='--', label=f'Ideal = {best_n_components}')
plt.title(r'Perfil da Validação Cruzada ($R^2$)')
plt.xlabel('Número de Componentes PLS')
plt.ylabel(r'$R^2$ (Maior é melhor)')
plt.xticks(components_range)
plt.legend()
plt.grid(True)

# Gráfico 3: Performance Final - Real vs Predito no Teste
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_test, alpha=0.7, edgecolors='k', color='purple', label='Previsões')
# Linha ideal de 45 graus onde Previsão = Real
min_val = min(np.min(y_test), np.min(y_pred_test))
max_val = max(np.max(y_test), np.max(y_pred_test))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Acerto Perfeito')
plt.title('Teste Final: Valores Reais vs. Preditos')
plt.xlabel('Valores Reais')
plt.ylabel('Valores Preditos')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()