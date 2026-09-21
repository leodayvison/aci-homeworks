from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

wine_data = pd.read_csv('./winequality-white.csv', sep=';')

wine_feat = wine_data.iloc[:, :-1] 
wine_target = wine_data.iloc[:, -1]

X = wine_feat.values.astype(float) #alguns valores estão em inteiro
Y = wine_target.values

"""     
        Importamos os dados usando a biblioteca Pandas, 
        e ajustamos usando a biblioteca Numpy para criar as
        matrizes: X (4898 x 11) de características; e Y (4898 x 1)
        de rótulos.
        
        Como os dados das categorias divergem fortemente entre
        si em dimensionalidade, é necessário centralizar e padronizar
        os preditores.
        Para isso usamos a formula: X = (x - media) / desvio padrao ,
        tal que as medidas podem ser conseguidas por funções do Numpy.
        """
        
meansX = np.mean(X, axis=0)
sDeviationX = np.std(X, axis=0)

Xp = (X - meansX) / sDeviationX

"""     
        A partir dos valores padronizados, podemos calcular
        a matriz de covariancia, que auxiliará na identificação
        das relações lineares e redundâncias entre as 11 características.
        
        Prosseguimos com o cálculo dos autovalores/autovetores,
        e a ordenação dos resultados.
        """

covarianceMatrix = np.cov(Xp, rowvar=False)
eValues, eVectors = np.linalg.eigh(covarianceMatrix)

indexD = np.argsort(eValues)[::-1]
eVectors = eVectors[:, indexD]

"""
        Com os autovetoes organizados, criamos a nova matriz
        n x 2 que representa os dois maiores autovetores.
        
        Então, projetamos essa matriz nos dados para obter
        a matriz do PCA
        """

Xweights = eVectors[:, :2]

PCA = np.dot(Xp, Xweights)

"""     Visualizando alguns resultados,

        Primeiro vamos ver a partir dos autovetores e autovalores
        quais características mais impactam na variação dos dados.
        
        Por fim, plotamos o scatterplot PC1 x PC2.
        """

loadings = pd.DataFrame(Xweights, index=wine_feat.columns, columns=['PC1', 'PC2'])
print(loadings)

scatter = plt.scatter(PCA[:, 0], PCA[:, 1], 
                      c=Y, cmap='viridis', alpha=0.6, edgecolors='none')

plt.colorbar(scatter, label='Quality')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA - 2D Projection')
plt.grid(True)
plt.show()
