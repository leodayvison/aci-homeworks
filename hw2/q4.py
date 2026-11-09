import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 1. Carregar os dados de Qualidade do Vinho
# Substitua pelo nome exato do arquivo CSV na sua pasta (ex: 'winequality-red.csv')
df = pd.read_csv('winequality-white.csv')

# Separar X (preditores) e y (variável resposta: qualidade do vinho)
# O código assume que 'quality' (a qualidade) é a última coluna.
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 2. Divisão treino/teste (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Pré-processamento: O mesmo das etapas anteriores (Padronização + PCA)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Transformação com base no treino!

# Reduzindo a dimensionalidade (opcional, mantendo o padrão das questões anteriores)
# Usaremos 90% da variância explicada para reter a informação importante do vinho
pca = PCA(n_components=0.90, random_state=42)
X_train_transf = pca.fit_transform(X_train_scaled)
X_test_transf = pca.transform(X_test_scaled)

# 4. Construção e Treinamento do Modelo de Rede Neural (MLP)
# Arquitetura: 2 camadas ocultas com 64 e 32 neurônios, ativação ReLU
mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32), 
    activation='relu',           # Função não-linear
    solver='adam',               # Otimizador eficiente
    alpha=0.01,                  # Penalidade L2 (Regularização para evitar overfitting)
    max_iter=1000,               # Número máximo de épocas
    random_state=42,
    early_stopping=True          # Para o treino se o modelo parar de melhorar
)

print("Treinando a Rede Neural... Isso pode levar alguns segundos.")
mlp.fit(X_train_transf, y_train)

# 5. Avaliação do Modelo no Conjunto de Teste
y_pred_test = mlp.predict(X_test_transf)

rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)

print("-" * 50)
print("PERFORMANCE DA REDE NEURAL NO CONJUNTO DE TESTE:")
print(f"RMSE no Teste: {rmse_test:.4f}")
print(f"R2 no Teste:   {r2_test:.4f}")
print("-" * 50)

# 6. Geração de Gráficos
plt.figure(figsize=(14, 6))

# Gráfico 1: Curva de Aprendizado (Loss Curve)
# Mostra como o erro da rede diminuiu ao longo das épocas (iterações)
plt.subplot(1, 2, 1)
plt.plot(mlp.loss_curve_, color='blue', lw=2, label='Erro de Treinamento (Loss)')
plt.title('Curva de Aprendizado da Rede Neural')
plt.xlabel('Épocas (Iterações)')
plt.ylabel('Loss (Erro Quadrático)')
plt.legend()
plt.grid(True)

# Gráfico 2: Valores Reais vs Preditos no Teste
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, alpha=0.5, edgecolors='k', color='purple', label='Previsões NN')
# Linha de acerto perfeito
min_val = min(np.min(y_test), np.min(y_pred_test))
max_val = max(np.max(y_test), np.max(y_pred_test))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Acerto Perfeito (Linha Ideal)')

plt.title('Rede Neural: Real vs Predito (Conjunto de Teste)')
plt.xlabel('Qualidade Real do Vinho')
plt.ylabel('Qualidade Predita pela Rede')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()