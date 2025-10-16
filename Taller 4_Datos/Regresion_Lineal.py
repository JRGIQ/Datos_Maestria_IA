# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 19:39:00 2025

@author: jhrgu
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
from IPython.display import display
import seaborn as sns
from pandas.plotting import scatter_matrix

datos = pd.read_csv('diamonds.csv')
df=pd.DataFrame(datos)

gf=df
df.info()
# print(df.isnull().any().any()) # Devuelve true si hay dato faltante


# print(df['cut'].value_counts()) # Cuantas categorías tienen un atributo
# display(df.describe()) # Valores numéricos

# print(round(max(gf['x size']),3))


# total_ceros = (gf == 0).sum().sum()
# # print("Total de ceros:", total_ceros)
# columnas_objetivo=['price','x size','y size','z size']
# gf_filtrado = gf[~(df[columnas_objetivo] == 0).any(axis=1)]


# gf=gf.drop(columns=['cut'])
# gf=gf.drop(columns=['color'])
# gf=gf.drop(columns=['clarity'])
# gf['vol']=(gf['x size']*gf['y size']*gf['z size'])


# total_ceros = (gf_filtrado == 0).sum().sum()
# print(total_ceros)

corr_matrix = gf.corr()

display(corr_matrix["price"].sort_values(ascending=False))
print(df.head())
df.hist(bins=50, figsize=(15, 8))

# x=df['x size']
# y=df['y size']
# z=df['z size']
# vol=x*y*z

# precio=df['price']

# pt.scatter(vol,precio,color='blue')
# pt.scatter(x,precio,color='red')
# pt.scatter(y,precio,color='green')
# pt.scatter(z,precio,color='yellow')

# attributes=['price','volm']
# attributes=['price','carat','vol']
# scatter_matrix(gf[attributes], figsize=(16, 20)) # Imprime las gráficas de correlación (forma 1)
# sns.pairplot(df[attributes], kind="scatter") # Imprime las gráficas de correlación (forma 2)