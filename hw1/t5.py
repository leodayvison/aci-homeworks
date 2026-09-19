from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

wine_data = pd.read_csv('/mnt/hd/vit/vscode/aci-homeworks/winequality-white.csv', sep=';')

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
        si em dimensionalidade, é 
        """
        


