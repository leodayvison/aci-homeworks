from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

wine_data = pd.read_csv('/mnt/hd/vit/vscode/aci-homeworks/winequality-white.csv', sep=';')

wine_feat = wine_data.iloc[:, :-1] 
wine_target = wine_data.iloc[:, -1]

X = wine_feat.values.astype(float) #alguns valores estão em inteiro
Y = wine_target.values

N = wine_data.shape[0]
D = wine_feat.shape[1]
L = 11

class_distrib = wine_target.value_counts()
print(class_distrib)