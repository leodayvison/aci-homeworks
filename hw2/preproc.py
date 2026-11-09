import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_and_preprocess_data(test_size=0.2, random_state=42, pca_variance=0.90, as_numpy=False):
    """
    Carrega o dataset winequality-white.csv, padroniza nomes de colunas,
    divide treino/teste, trata assimetria com log1p, padroniza com StandardScaler
    e opcionalmente aplica redução de dimensionalidade via PCA.

    Parâmetros:
        test_size (float): Proporção do conjunto de teste (padrão: 0.2).
        random_state (int): Semente aleatória (padrão: 42).
        pca_variance (float ou int ou None): Variância acumulada (ex: 0.90) ou número fixo de componentes.
        as_numpy (bool): Se True, retorna matrizes numpy ao invés de pandas DataFrames/Series.

    Retorna:
        X_train_scaled, X_test_scaled, X_train_pca, X_test_pca, y_train, y_test, pca, scaler
    """
    # Suporta execução tanto a partir da raiz do repositório quanto de dentro da pasta hw2/
    csv_filename = 'winequality-white.csv'
    if not os.path.exists(csv_filename) and os.path.exists(os.path.join('..', csv_filename)):
        csv_filename = os.path.join('..', csv_filename)

    if not os.path.exists(csv_filename):
        raise FileNotFoundError(f"Erro: O arquivo '{csv_filename}' não foi encontrado.")

    df = pd.read_csv(csv_filename, sep=';')

    # Padroniza nomes das colunas substituindo espaços por underscores (snake_case)
    df.columns = df.columns.str.strip().str.replace(' ', '_')

    # A última coluna é o alvo (quality - Y) e as demais os preditores (X)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Cria cópias explícitas para evitar SettingWithCopyWarning
    X_train = X_train.copy()
    X_test = X_test.copy()

    # 1. Transformação logarítmica para as colunas com assimetria acentuada
    X_train['residual_sugar'] = np.log1p(X_train['residual_sugar'])
    X_train['free_sulfur_dioxide'] = np.log1p(X_train['free_sulfur_dioxide'])

    X_test['residual_sugar'] = np.log1p(X_test['residual_sugar'])
    X_test['free_sulfur_dioxide'] = np.log1p(X_test['free_sulfur_dioxide'])

    # 2. Padronização (Center & Scale) - fit estritamente no conjunto de treino
    scaler = StandardScaler().set_output(transform='pandas')
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Redução de Dimensionalidade via PCA - fit estritamente no conjunto de treino
    pca = None
    X_train_pca = None
    X_test_pca = None
    if pca_variance is not None:
        pca = PCA(n_components=pca_variance, random_state=random_state).set_output(transform='pandas')
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

    if as_numpy:
        return (
            X_train_scaled.to_numpy(),
            X_test_scaled.to_numpy(),
            X_train_pca.to_numpy() if X_train_pca is not None else None,
            X_test_pca.to_numpy() if X_test_pca is not None else None,
            y_train.to_numpy(),
            y_test.to_numpy(),
            pca,
            scaler,
        )

    return X_train_scaled, X_test_scaled, X_train_pca, X_test_pca, y_train, y_test, pca, scaler


if __name__ == '__main__':
    X_train_scaled, X_test_scaled, X_train_pca, X_test_pca, y_train, y_test, pca, scaler = load_and_preprocess_data()

    print("=" * 60)
    print("ETAPA 0: PRÉ-PROCESSAMENTO E ANÁLISE DE CORRELAÇÃO")
    print("=" * 60)
    print(f"Número de amostras de treino: {X_train_scaled.shape[0]}")
    print(f"Número de amostras de teste:  {X_test_scaled.shape[0]}")
    print(f"Número de preditores originais: {X_train_scaled.shape[1]}")

    if pca is not None:
        print("\n--- Redução de Dimensionalidade (PCA) ---")
        print(f"- Número de componentes retidos (90% variância): {pca.n_components_}")
        print(f"- Variância explicada acumulada: {np.sum(pca.explained_variance_ratio_):.2%}")
        print(f"- Variância individual por componente: {np.round(pca.explained_variance_ratio_, 4)}")

    # 1. Matriz de correlação entre os preditores transformados
    corr_matrix = X_train_scaled.corr()
    print("\n" + "=" * 60)
    print("1. MATRIZ DE CORRELAÇÃO ENTRE OS PREDITORES TRANSFORMADOS")
    print("=" * 60)
    print(corr_matrix.round(3))

    # Identificar pares com forte correlação (|r| >= 0.6)
    print("\nPares de preditores com correlação relevante (|r| >= 0.6):")
    pairs_found = False
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            r_val = corr_matrix.iloc[i, j]
            if abs(r_val) >= 0.6:
                pairs_found = True
                print(f"  - {col1} <-> {col2}: r = {r_val:+.3f}")
    if not pairs_found:
        print("  - Nenhum par com |r| >= 0.6 encontrado.")

    # 2. Correlações individuais entre cada preditor e a qualidade (desfecho Y)
    y_corr = X_train_scaled.apply(lambda col: col.corr(y_train))
    print("\n" + "=" * 60)
    print("2. CORRELAÇÃO PREVENTOR-DESFECHO (LINEARIDADE COM QUALITY)")
    print("=" * 60)
    for col, r_val in y_corr.sort_values(ascending=False).items():
        print(f"  {col:<25}: r = {r_val:+.4f}")

    # 3. Gráficos para o Artigo

    # --- Janela 1: Heatmap de correlação entre preditores ---
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    cax = ax1.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    fig1.colorbar(cax, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_xticks(range(len(corr_matrix.columns)))
    ax1.set_yticks(range(len(corr_matrix.columns)))
    ax1.set_xticklabels(corr_matrix.columns, rotation=90, fontsize=9)
    ax1.set_yticklabels(corr_matrix.columns, fontsize=9)
    ax1.set_title('Matriz de Correlação dos Preditores', pad=20, fontsize=12, fontweight='bold')
    fig1.tight_layout()

    # --- Janela 2: Gráfico de barras: correlação com a variável alvo ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    y_corr_sorted = y_corr.sort_values()
    colors = ['crimson' if val < 0 else 'forestgreen' for val in y_corr_sorted]
    ax2.barh(y_corr_sorted.index, y_corr_sorted.values, color=colors, edgecolor='black', alpha=0.8)
    ax2.axvline(0, color='black', linewidth=1)
    ax2.set_xlabel('Coeficiente de Correlação de Pearson com a Qualidade', fontsize=10)
    ax2.set_title('Correlação Individual Preditor-Desfecho', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6, axis='x')
    fig2.tight_layout()

    # Exibe as janelas simultaneamente
    plt.show()