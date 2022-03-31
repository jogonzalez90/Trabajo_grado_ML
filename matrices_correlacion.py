# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 19:27:17 2021

@author: Asus
"""
########## LIBRERÍAS A UTILIZAR ##########

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

########## FILTRADO BASES DE DATOS ##########

#Lectura de archivo filtrado de la NASA (atributos)
exoplanetas_2=pd.read_csv('exoplanetas_2.csv')
#Visualizamos Datagframe 
print(exoplanetas_2)
#Se imprime nombre de atributos
print(exoplanetas_2.columns)
#Lectura archivo tipos de planetas Linea de codigo para corregir el problema "encoding='latin-1'"
tipo_planeta=pd.read_csv('tipo_planeta.csv',encoding='latin-1')
#Visualizamos Datagframe 
print(tipo_planeta)
#Imprime el tipo de dato de los atributos
print(tipo_planeta.dtypes)#Imprime el tipo de dato de los atributos
#Se crea varible con atributos periodo orbital, eje-semimayor y radio del planeta
exoplanetas_3=exoplanetas_2[['pl_name','pl_orbper','pl_orbsmax','pl_rade']]
#Elimina instancias con valor unknown en atributo 'Valor_masa_multiplicador'
tipo_planeta = tipo_planeta.drop(tipo_planeta[tipo_planeta['Valor_masa_multiplicador']=='Unknown'].index)
#Elimina instancias con valor unknown en atributo 'Tipo_planeta'
tipo_planeta = tipo_planeta.drop(tipo_planeta[tipo_planeta['Tipo_planeta']=='Unknown'].index)
#Atributo  'Valor_masa_multiplicador' de tipo object a tipo float
tipo_planeta['Valor_masa_multiplicador']=tipo_planeta['Valor_masa_multiplicador'].astype('float')
#print(tipo_planeta.columns)#Se imprime nombre de atributos
#Verifiva cambio en tipo dato en dataframe
print(tipo_planeta.dtypes)#Imprime el tipo de dato de los atributos
#Se agrega al dataframe tipo_planeta la columna 'MasaPlaneta_FuncionTierra'
tipo_planeta['MasaPlaneta_FuncionTierra'] = tipo_planeta.Valor_masa_multiplicador * tipo_planeta.mj_mt
#Actualiza dataframe con los atributos de interes ('pl_name','MasaPlaneta_FuncionTierra','Tipo_planeta')
tipo_planeta=tipo_planeta[['pl_name','MasaPlaneta_FuncionTierra','Tipo_planeta']]
#Tabla definitiva
tabla_definitiva=pd.merge(exoplanetas_3, tipo_planeta, on='pl_name', how='outer')
#Sustituye nombres colunmas de ingles a español
tabla_definitiva.columns=['Nombre_planeta','Periodo_orbital','Eje_semimayor','Radio_planeta','Masa_planeta','Tipo_planeta']
#Elimina isnstancias con nan
tabla_definitiva=tabla_definitiva.dropna()
#Visualizamos Datagframe definitivo
print(tabla_definitiva)

########## PREPROCESAMIENTO ##########

#Matriz de correlacion  con datos de tabla_definitiva
matriz_corr = tabla_definitiva.corr()
sns.heatmap(matriz_corr, annot = True)
#Titulo de Matirz
plt.title("Matriz de Correlacion")
#Etiquetas de la matriz
plt.xlabel("Atributos de interes")
plt.ylabel("Atributos de interes")
plt.show()
#Convertiremos la matriz dada en una serie unidimensional de valores
uni_va = matriz_corr.unstack()
print(uni_va)
#Ahora vamos a ordenar estos valores
par_ord = uni_va.sort_values(kind="quicksort")
print(par_ord)

#Se realiza conversion datos a log
tabla_definitiva_2=tabla_definitiva
tabla_definitiva_2['Periodo_orbital']=np.log(tabla_definitiva_2['Periodo_orbital'])
tabla_definitiva_2['Eje_semimayor']=np.log(tabla_definitiva_2['Eje_semimayor'])
tabla_definitiva_2['Radio_planeta']=np.log(tabla_definitiva_2['Radio_planeta'])
tabla_definitiva_2['Masa_planeta']=np.log(tabla_definitiva_2['Masa_planeta'])

#Matriz de correlacion  con datos tipo log
matriz_corr_2 = tabla_definitiva_2.corr()
sns.heatmap(matriz_corr_2, annot = True)
#Titulo de Matirz
plt.title("Matriz de Correlacion")
#Etiquetas de la matriz
plt.xlabel("Atributos de interes")
plt.ylabel("Atributos de interes")
plt.show()
#Convertiremos la matriz dada en una serie unidimensional de valores
uni_va_2 = matriz_corr_2.unstack()
print(uni_va_2)
#Ahora vamos a ordenar estos valores
par_ord_2 = uni_va_2.sort_values(kind="quicksort")
print(par_ord_2)
