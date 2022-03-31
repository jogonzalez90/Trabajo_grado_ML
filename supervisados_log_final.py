# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 19:48:23 2021

# Seccion de codigo para coversion de datos a logaritmo natural
tabla_definitiva['Radio_planeta']=np.log(tabla_definitiva['Radio_planeta'])
tabla_definitiva['Masa_planeta']=np.log(tabla_definitiva['Masa_planeta'])
tabla_definitiva['Periodo_orbital']=np.log(tabla_definitiva['Periodo_orbital'])
tabla_definitiva['Eje_semimayor']=np.log(tabla_definitiva['Eje_semimayor'])

@author: Asus
"""
########## LIBRERÍAS A UTILIZAR ##########

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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
print(tipo_planeta.columns)#Se imprime nombre de atributos
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
#Convertir periodo orbital de dias a años
tabla_definitiva['Periodo_orbital'] = tabla_definitiva['Periodo_orbital'] * 0.002739726
#Elimina isnstancias con nan
tabla_definitiva=tabla_definitiva.dropna()
#Visualizamos Datagframe definitivo
print(tabla_definitiva)

########## ANALIZAMOS LOS DATOS ##########

#Análizamos los datos que tenemos disponibles
print('Información del dataset:')
print(tabla_definitiva.info())
#Visualizamos tados estadisticos
print('Descripción del dataset:')
print(tabla_definitiva.describe())
#Verificamos distribucion de los datos
print('Distribución de las exoplanetas:')
print(tabla_definitiva.groupby('Tipo_planeta').size())

########## APLICACIÓN DE ALGORITMOS DE MACHINE LEARNING ##########

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

################## METRICAS DE EVALUACION ########################

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score#Exactitud
from sklearn.metrics import precision_score#Presicion
#from sklearn.metrics import recall_score#Sensibilidad
from sklearn.metrics import f1_score#Puntaje F1
#Enlace estudiar
#https://aprendeia.com/metricas-de-evaluacion-clasificacion-con-scikit-learn-machine-learning/

##################################################
##### MODELOS CON DATOS RADIO-MASA DEL PLANETA ###
#### DATOS SIN COVERSION A LOGARITMO MATURAL #####
##################################################

# Seccion de codigo para coversion de datos a logaritmo natural
tabla_definitiva['Radio_planeta']=np.log(tabla_definitiva['Radio_planeta'])
tabla_definitiva['Masa_planeta']=np.log(tabla_definitiva['Masa_planeta'])
tabla_definitiva['Periodo_orbital']=np.log(tabla_definitiva['Periodo_orbital'])
tabla_definitiva['Eje_semimayor']=np.log(tabla_definitiva['Eje_semimayor'])

X_radio_masa= tabla_definitiva[['Radio_planeta','Masa_planeta']]
y_radio_masa = tabla_definitiva[['Tipo_planeta']]
#Verifico caracteristica y etiquetas
print(X_radio_masa)
print(y_radio_masa)

#Ahora procedemos a separar los datos de entrenamiento y prueba para proceder a construir los modelos.
X_train_parcial, X_test_parcial, y_train_parcial, y_test_parcial = train_test_split(X_radio_masa, y_radio_masa, test_size=0.2,random_state=0)
print('Son {} datos para entrenamiento y {} datos para prueba'.format(X_train_parcial.shape[0], X_test_parcial.shape[0]))


# El escalador de objetos (modelo)
scaler = StandardScaler ()
# Ajustar y transformar los datos 
X_train_parcial = scaler.fit_transform (X_train_parcial)
X_test_parcial = scaler.fit_transform (X_test_parcial)

#Variables para tabla de resultados
#nombreAlgoritmo=['Regresion Logistica','KVecinosMasCercanos','Arbol Decisiones']
precision_parcial=[]#Variable para radio-masa
precision_total=[]#Variable para orbital eje
exactitud_parcial=[]#Variable para radio-masa
exactitud_total=[]#Variable para orbital eje

#Modelo de Máquinas de Vectores de Soporte
algoritmo_svm = SVC(kernel = 'linear')
algoritmo_svm.fit(X_train_parcial, y_train_parcial)
Y_pred_svm = algoritmo_svm.predict(X_test_parcial)
precision_svm = precision_score(y_test_parcial, Y_pred_svm,average='micro')
precision_parcial.append(precision_svm)
print('Precisión SVM Radio-Masa= '+ str(precision_svm))
exatitud_svm = accuracy_score(y_test_parcial, Y_pred_svm)
exactitud_parcial.append(exatitud_svm)
print('Exactitud SVM Radio-Masa= '+ str(exatitud_svm))
matriz_svm = confusion_matrix(y_test_parcial, Y_pred_svm)
print('Matriz SVM Radio-Masa= ')
print(matriz_svm)

#Modelo de Vecinos más Cercanos
algoritmo_knn = KNeighborsClassifier(n_neighbors=7)
algoritmo_knn.fit(X_train_parcial, y_train_parcial)
Y_pred_knn = algoritmo_knn.predict(X_test_parcial)
precision_knn = precision_score(y_test_parcial, Y_pred_knn,average='micro')
precision_parcial.append(precision_knn)
print('Precisión k-NN Radio-Masa= '+ str(precision_knn))
exatitud_knn = accuracy_score(y_test_parcial, Y_pred_knn)
exactitud_parcial.append(exatitud_knn)
print('Exactitud k-NN Radio-Masa= '+ str(exatitud_knn))
matriz_knn = confusion_matrix(y_test_parcial, Y_pred_knn)
print('Matriz k-NN Radio-Masa= ')
print(matriz_knn)

#Modelo de Árboles de Decisión Clasificación
algoritmo_tree = DecisionTreeClassifier(criterion= 'entropy', random_state=0)
algoritmo_tree.fit(X_train_parcial, y_train_parcial)
Y_pred_tree = algoritmo_tree.predict(X_test_parcial)
precision_tree = precision_score(y_test_parcial, Y_pred_tree,average='micro')
precision_parcial.append(precision_tree)
print('Precisión Tree Radio-Masa= '+ str(precision_tree))
exatitud_tree = accuracy_score(y_test_parcial, Y_pred_tree)
exactitud_parcial.append(exatitud_tree)
print('Exactitud Tree Radio-Masa= '+ str(exatitud_tree))
matriz_tree = confusion_matrix(y_test_parcial, Y_pred_tree)
print('Matriz Tree Radio-Masa= ')
print(matriz_tree)

###############################################################
# MODELOS CON DATOS PERIODO_ORBITAL-EJE_SEMIMAYOR DEL PLANETA #
######### DATOS SIN COVERSION A LOGARITMO MATURAL #############
###############################################################

X_radio_masa_orbital= tabla_definitiva[['Periodo_orbital','Eje_semimayor']]
y_radio_masa_orbital = tabla_definitiva[['Tipo_planeta']]
#Verifico caracteristica y etiquetas
print(X_radio_masa_orbital)
print(y_radio_masa_orbital)

#Ahora procedemos a separar los datos de entrenamiento y prueba para proceder a construir los modelos.
X_train_total, X_test_total, y_train_total, y_test_total = train_test_split(X_radio_masa_orbital, y_radio_masa_orbital, test_size=0.2,random_state=0)
print('Son {} datos para entrenamiento y {} datos para prueba'.format(X_train_total.shape[0], X_test_total.shape[0]))

# El escalador de objetos (modelo)
scaler = StandardScaler ()
# Ajustar y transformar los datos 
X_train_total = scaler.fit_transform (X_train_total)
X_test_total = scaler.fit_transform (X_test_total)

#Modelo de Máquinas de Vectores de Soporte
algoritmo_svm = SVC(kernel = 'linear')
algoritmo_svm.fit(X_train_total, y_train_total)
Y_pred_svm = algoritmo_svm.predict(X_test_total)
precision_svm_total = precision_score(y_test_total, Y_pred_svm,average='micro')
precision_total.append(precision_svm_total)
print('Precisión SVM Periodo_Orbital-Eje_semimayor= '+ str(precision_svm_total))
exatitud_svm_total = accuracy_score(y_test_total, Y_pred_svm)
exactitud_total.append(exatitud_svm_total)
print('Exactitud SVM Periodo_Orbital-Eje_semimayor= '+ str(exatitud_svm_total))
matriz_svm_total = confusion_matrix(y_test_total, Y_pred_svm)
print('Matriz SVM Periodo_Orbital-Eje_semimayor= ')
print(matriz_svm_total)

#Modelo de Vecinos más Cercanos
algoritmo_knn = KNeighborsClassifier(n_neighbors=7)
algoritmo_knn.fit(X_train_total, y_train_total)
Y_pred_knn = algoritmo_knn.predict(X_test_total)
precision_knn_total = precision_score(y_test_total, Y_pred_knn,average='micro')
precision_total.append(precision_knn_total)
print('Precisión k-NN Periodo_Orbital-Eje_semimayor= '+ str(precision_knn_total))
exatitud_knn_total = accuracy_score(y_test_total, Y_pred_knn)
exactitud_total.append(exatitud_knn_total)
print('Exactitud k-NN Periodo_Orbital-Eje_semimayor= '+ str(exatitud_knn_total))
matriz_knn_total = confusion_matrix(y_test_total, Y_pred_knn)
print('Matriz k-NN Periodo_Orbital-Eje_semimayor= ')
print(matriz_knn_total)

#Modelo de Árboles de Decisión Clasificación
algoritmo_tree = DecisionTreeClassifier(criterion= 'entropy', random_state=0)
algoritmo_tree.fit(X_train_total, y_train_total)
Y_pred_tree = algoritmo_tree.predict(X_test_total)
precision_tree_total = precision_score(y_test_total, Y_pred_tree,average='micro')
precision_total.append(precision_tree_total)
print('Precisión Tree Periodo_Orbital-Eje_semimayor= '+ str(precision_tree_total))
exatitud_tree_total = accuracy_score(y_test_total, Y_pred_tree)
exactitud_total.append(exatitud_tree_total)
print('Exactitud Tree Periodo_Orbital-Eje_semimayor= '+ str(exatitud_tree_total))
matriz_tree_total = confusion_matrix(y_test_total, Y_pred_tree)
print('Matriz Tree Periodo_Orbital-Eje_semimayor= ')
print(matriz_tree_total)

######### CONSTRUCCION TABLA PARA ANALISIS DE RESULTADOS ##########

modelos=['Máquinas de Vectores','k-Vecinos más cercanos','Árbol de decisiones']

#Estructura de la tabla radio-masa del planeta
tabla_resultados_1={'Modelos ML(Datos LOG)':modelos,
                  'Precisión Masa-Radio':precision_parcial,
                  'Exactitud Masa-Radio':exactitud_parcial,}
resultados_1=pd.DataFrame(tabla_resultados_1)
print(resultados_1)

#Estructura de la tabla orbital-eje semimayor
tabla_resultados_2={'Modelos ML(Datos LOG)':modelos,
                  'Precisión Orbital-Eje':precision_total,
                  'Exactitud Orbital-Eje':exactitud_total,}
resultados_2=pd.DataFrame(tabla_resultados_2)
print(resultados_2)

#Estructura de la tabla (radio-masa)-(orbital-eje) semimayor
tabla_resultados_3={'Modelos ML(Datos LOG)':modelos,
                  'Precisión Masa-Radio':precision_parcial,'Precisión Orbital-Eje':precision_total,
                  'Exactitud Masa-Radio':exactitud_parcial,'Exactitud Orbital-Eje':exactitud_total,}
resultados_3=pd.DataFrame(tabla_resultados_3)
print(resultados_3)


###################################################################
##### MODELOS CON DATOS RADIO-MASA-PERIODO_ORBITAL DEL PLANETA ####
########### DATOS SIN COVERSION A LOGARITMO MATURAL ###############
###################################################################

X_radio_masa= tabla_definitiva[['Radio_planeta','Masa_planeta','Periodo_orbital']]
y_radio_masa = tabla_definitiva[['Tipo_planeta']]
#Verifico caracteristica y etiquetas
print(X_radio_masa)
print(y_radio_masa)

#Ahora procedemos a separar los datos de entrenamiento y prueba para proceder a construir los modelos.
X_train_parcial, X_test_parcial, y_train_parcial, y_test_parcial = train_test_split(X_radio_masa, y_radio_masa, test_size=0.2,random_state=0)
print('Son {} datos para entrenamiento y {} datos para prueba'.format(X_train_parcial.shape[0], X_test_parcial.shape[0]))


# El escalador de objetos (modelo)
scaler = StandardScaler ()
# Ajustar y transformar los datos 
X_train_parcial = scaler.fit_transform (X_train_parcial)
X_test_parcial = scaler.fit_transform (X_test_parcial)

#Variables para tabla de resultados
#nombreAlgoritmo=['Regresion Logistica','KVecinosMasCercanos','Arbol Decisiones']
precision_parcial=[]#Variable para masa-radio-orbital
precision_total=[]#Variable para masa-radio-eje
exactitud_parcial=[]#Variable para masa-radio-orbital
exactitud_total=[]#Variable para masa-radio-eje

#Modelo de Máquinas de Vectores de Soporte
algoritmo_svm = SVC(kernel = 'linear')
algoritmo_svm.fit(X_train_parcial, y_train_parcial)
Y_pred_svm = algoritmo_svm.predict(X_test_parcial)
precision_svm = precision_score(y_test_parcial, Y_pred_svm,average='micro')
precision_parcial.append(precision_svm)
print('Precisión SVM Radio-Masa-Orbital= '+ str(precision_svm))
exatitud_svm = accuracy_score(y_test_parcial, Y_pred_svm)
exactitud_parcial.append(exatitud_svm)
print('Exactitud SVM Radio-Masa-Orbital= '+ str(exatitud_svm))
matriz_svm = confusion_matrix(y_test_parcial, Y_pred_svm)
print('Matriz SVM Radio-Masa-Orbital= ')
print(matriz_svm)

#Modelo de Vecinos más Cercanos
algoritmo_knn = KNeighborsClassifier(n_neighbors=7)
algoritmo_knn.fit(X_train_parcial, y_train_parcial)
Y_pred_knn = algoritmo_knn.predict(X_test_parcial)
precision_knn = precision_score(y_test_parcial, Y_pred_knn,average='micro')
precision_parcial.append(precision_knn)
print('Precisión k-NN Radio-Masa-Orbital= '+ str(precision_knn))
exatitud_knn = accuracy_score(y_test_parcial, Y_pred_knn)
exactitud_parcial.append(exatitud_knn)
print('Exactitud k-NN Radio-Masa-Orbital= '+ str(exatitud_knn))
matriz_knn = confusion_matrix(y_test_parcial, Y_pred_knn)
print('Matriz k-NN Radio-Masa-Orbital= ')
print(matriz_knn)

#Modelo de Árboles de Decisión Clasificación
algoritmo_tree = DecisionTreeClassifier(criterion= 'entropy', random_state=0)
algoritmo_tree.fit(X_train_parcial, y_train_parcial)
Y_pred_tree = algoritmo_tree.predict(X_test_parcial)
precision_tree = precision_score(y_test_parcial, Y_pred_tree,average='micro')
precision_parcial.append(precision_tree)
print('Precisión Tree Radio-Masa-Orbital= '+ str(precision_tree))
exatitud_tree = accuracy_score(y_test_parcial, Y_pred_tree)
exactitud_parcial.append(exatitud_tree)
print('Exactitud Tree Radio-Masa-Orbital= '+ str(exatitud_tree))
matriz_tree = confusion_matrix(y_test_parcial, Y_pred_tree)
print('Matriz Tree Radio-Masa-Orbital= ')
print(matriz_tree)

##############################################################################
# MODELOS CON ATRIBUTOS RADIO-MASA-EJE_SEMIMAYOR DEL PLANETA #
############### DATOS SIN COVERSION A LOGARITMO MATURAL ######################
##############################################################################

X_radio_masa_orbital= tabla_definitiva[['Radio_planeta','Masa_planeta','Eje_semimayor']]
y_radio_masa_orbital = tabla_definitiva[['Tipo_planeta']]
#Verifico caracteristica y etiquetas
print(X_radio_masa_orbital)
print(y_radio_masa_orbital)

#Ahora procedemos a separar los datos de entrenamiento y prueba para proceder a construir los modelos.
X_train_total, X_test_total, y_train_total, y_test_total = train_test_split(X_radio_masa_orbital, y_radio_masa_orbital, test_size=0.2,random_state=0)
print('Son {} datos para entrenamiento y {} datos para prueba'.format(X_train_total.shape[0], X_test_total.shape[0]))

# El escalador de objetos (modelo)
scaler = StandardScaler ()
# Ajustar y transformar los datos 
X_train_total = scaler.fit_transform (X_train_total)
X_test_total = scaler.fit_transform (X_test_total)

#Modelo de Máquinas de Vectores de Soporte
algoritmo_svm = SVC(kernel = 'linear')
algoritmo_svm.fit(X_train_total, y_train_total)
Y_pred_svm = algoritmo_svm.predict(X_test_total)
precision_svm_total = precision_score(y_test_total, Y_pred_svm,average='micro')
precision_total.append(precision_svm_total)
print('Precisión SVM Radio-Masa-Eje_semimayor= '+ str(precision_svm_total))
exatitud_svm_total = accuracy_score(y_test_total, Y_pred_svm)
exactitud_total.append(exatitud_svm_total)
print('Exactitud SVM Radio-Masa-Eje_semimayor= '+ str(exatitud_svm_total))
matriz_svm_total = confusion_matrix(y_test_total, Y_pred_svm)
print('Matriz SVM Radio-Masa-Eje_semimayor= ')
print(matriz_svm_total)

#Modelo de Vecinos más Cercanos
algoritmo_knn = KNeighborsClassifier(n_neighbors=7)
algoritmo_knn.fit(X_train_total, y_train_total)
Y_pred_knn = algoritmo_knn.predict(X_test_total)
precision_knn_total = precision_score(y_test_total, Y_pred_knn,average='micro')
precision_total.append(precision_knn_total)
print('Precisión k-NN Radio-Masa-Eje_semimayor= '+ str(precision_knn_total))
exatitud_knn_total = accuracy_score(y_test_total, Y_pred_knn)
exactitud_total.append(exatitud_knn_total)
print('Exactitud k-NN Radio-Masa-Eje_semimayor= '+ str(exatitud_knn_total))
matriz_knn_total = confusion_matrix(y_test_total, Y_pred_knn)
print('Matriz k-NN Radio-Masa-Eje_semimayor= ')
print(matriz_knn_total)

#Modelo de Árboles de Decisión Clasificación
algoritmo_tree = DecisionTreeClassifier(criterion= 'entropy', random_state=0)
algoritmo_tree.fit(X_train_total, y_train_total)
Y_pred_tree = algoritmo_tree.predict(X_test_total)
precision_tree_total = precision_score(y_test_total, Y_pred_tree,average='micro')
precision_total.append(precision_tree_total)
print('Precisión Tree Radio-Masa-Eje_semimayor= '+ str(precision_tree_total))
exatitud_tree_total = accuracy_score(y_test_total, Y_pred_tree)
exactitud_total.append(exatitud_tree_total)
print('Exactitud Tree Radio-Masa-Eje_semimayor= '+ str(exatitud_tree_total))
matriz_tree_total = confusion_matrix(y_test_total, Y_pred_tree)
print('Matriz Tree Radio-Masa-Eje_semimayor= ')
print(matriz_tree_total)

######### CONSTRUCCION TABLA PARA ANALISIS DE RESULTADOS ##########

#Estructura de la tabla radio-masa-orbital del planeta
modelos=['Máquinas de Vectores','k-Vecinos más cercanos','Árbol de decisiones']
#Estructura de la tabla
tabla_resultados_4={'Modelos ML (Datos LOG)':modelos,
                  'Precisión radio-masa-orbital':precision_parcial,
                  'Exactitud radio-masa-orbital':exactitud_parcial,}
resultados_4=pd.DataFrame(tabla_resultados_4)
print(resultados_4)

#Estructura de la tabla radio-masa-eje del planeta
modelos=['Máquinas de Vectores','k-Vecinos más cercanos','Árbol de decisiones']
#Estructura de la tabla
tabla_resultados_5={'Modelos ML (Datos LOG)':modelos,
                  'Precisión radio-masa-eje':precision_total,
                  'Exactitud radio-masa-eje':exactitud_total,}
resultados_5=pd.DataFrame(tabla_resultados_5)
print(resultados_5)

#Estructura de la tabla comparativa (radio-masa-orbital)-(radio-masa-eje del planeta)
modelos=['Máquinas de Vectores','k-Vecinos más cercanos','Árbol de decisiones']
#Estructura de la tabla
tabla_resultados_6={'Modelos ML (Datos LOG)':modelos,
                  'Precisión radio-masa-orbital':precision_parcial,'Precisión radio-masa-eje':precision_total,
                  'Exactitud radio-masa-orbital':exactitud_parcial,'Exactitud radio-masa-eje':exactitud_total,}
resultados_6=pd.DataFrame(tabla_resultados_6)
print(resultados_6)



##############################################################################
# MODELOS CON ATRIBUTOS RADIO-MASA-PERIODO_ORBITAL-EJE_SEMIMAYOR DEL PLANETA #
############### DATOS SIN COVERSION A LOGARITMO MATURAL ######################
##############################################################################

X_radio_masa_orbital= tabla_definitiva[['Radio_planeta','Masa_planeta','Periodo_orbital','Eje_semimayor']]
y_radio_masa_orbital = tabla_definitiva[['Tipo_planeta']]
#Verifico caracteristica y etiquetas
print(X_radio_masa_orbital)
print(y_radio_masa_orbital)

#Ahora procedemos a separar los datos de entrenamiento y prueba para proceder a construir los modelos.
X_train_total, X_test_total, y_train_total, y_test_total = train_test_split(X_radio_masa_orbital, y_radio_masa_orbital, test_size=0.2,random_state=0)
print('Son {} datos para entrenamiento y {} datos para prueba'.format(X_train_total.shape[0], X_test_total.shape[0]))

# El escalador de objetos (modelo)
scaler = StandardScaler ()
# Ajustar y transformar los datos 
X_train_total = scaler.fit_transform (X_train_total)
X_test_total = scaler.fit_transform (X_test_total)

#Variables para tabla de resultados
#nombreAlgoritmo=['Regresion Logistica','KVecinosMasCercanos','Arbol Decisiones']
#precision_parcial=[]#Variable para masa-radio-orbital
precision_total=[]#Variable para todos los atributos
#exactitud_parcial=[]#Variable para masa-radio-orbital
exactitud_total=[]#Variable para todos los atributos

#Modelo de Máquinas de Vectores de Soporte
algoritmo_svm = SVC(kernel = 'linear')
algoritmo_svm.fit(X_train_total, y_train_total)
Y_pred_svm = algoritmo_svm.predict(X_test_total)
precision_svm_total = precision_score(y_test_total, Y_pred_svm,average='micro')
precision_total.append(precision_svm_total)
print('Precisión SVM Radio-Masa-Orbital-Eje_semimayor= '+ str(precision_svm_total))
exatitud_svm_total = accuracy_score(y_test_total, Y_pred_svm)
exactitud_total.append(exatitud_svm_total)
print('Exactitud SVM Radio-Masa-Orbital-Eje_semimayor= '+ str(exatitud_svm_total))
f1_svm_total=f1_score(y_test_parcial, Y_pred_svm,average='micro')
print('F1 Score SVM Radio-Masa-Orbital-Eje_semimayor= '+ str(f1_svm_total))
matriz_svm_total = confusion_matrix(y_test_total, Y_pred_svm)
print('Matriz SVM Radio-Masa-Orbital-Eje_semimayor= ')
print(matriz_svm_total)

#Modelo de Vecinos más Cercanos
algoritmo_knn = KNeighborsClassifier(n_neighbors=7)
algoritmo_knn.fit(X_train_total, y_train_total)
Y_pred_knn = algoritmo_knn.predict(X_test_total)
precision_knn_total = precision_score(y_test_total, Y_pred_knn,average='micro')
precision_total.append(precision_knn_total)
print('Precisión k-NN Radio-Masa-Orbital-Eje_semimayor= '+ str(precision_knn_total))
exatitud_knn_total = accuracy_score(y_test_total, Y_pred_knn)
exactitud_total.append(exatitud_knn_total)
print('Exactitud k-NN Radio-Masa-Orbital-Eje_semimayor= '+ str(exatitud_knn_total))
f1_knn_total=f1_score(y_test_parcial, Y_pred_knn,average='micro')
print('F1 Score k-NN Radio-Masa-Orbital-Eje_semimayor= '+ str(f1_knn_total))
matriz_knn_total = confusion_matrix(y_test_total, Y_pred_knn)
print('Matriz k-NN Radio-Masa-Orbital-Eje_semimayor= ')
print(matriz_knn_total)

#Modelo de Árboles de Decisión Clasificación
algoritmo_tree = DecisionTreeClassifier(criterion= 'entropy', random_state=0)
algoritmo_tree.fit(X_train_total, y_train_total)
Y_pred_tree = algoritmo_tree.predict(X_test_total)
precision_tree_total = precision_score(y_test_total, Y_pred_tree,average='micro')
precision_total.append(precision_tree_total)
print('Precisión Tree Radio-Masa-Orbital-Eje_semimayor= '+ str(precision_tree_total))
exatitud_tree_total = accuracy_score(y_test_total, Y_pred_tree)
exactitud_total.append(exatitud_tree_total)
print('Exactitud Tree Radio-Masa-Orbital-Eje_semimayor= '+ str(exatitud_tree_total))
f1_tree_total=f1_score(y_test_parcial, Y_pred_tree,average='micro')
print('F1 Score Tree Radio-Masa-Orbital-Eje_semimayor= '+ str(f1_tree_total))
matriz_tree_total = confusion_matrix(y_test_total, Y_pred_tree)
print('Matriz Tree Radio-Masa-Orbital-Eje_semimayor= ')
print(matriz_tree_total)

######### CONSTRUCCION TABLA PARA ANALISIS DE RESULTADOS ##########

#Estructura de la tabla todos los atributos
modelos=['Máquinas de Vectores','k-Vecinos más cercanos','Árbol de decisiones']
#Estructura de la tabla
tabla_resultados_7={'Modelos ML (Datos LOG)':modelos,
                  'Precisión total':precision_total,
                  'Exactitud total':exactitud_total,}
resultados_7=pd.DataFrame(tabla_resultados_7)
print(resultados_7)


#A traves de estas lineas de código visualizamos la
#frecuencia con que aparece cada tipo de exoplaneta
#en el test data, para corroborar con la matriz de 
#confusion de SVM 'matriz_tree_total'(que logra las mejores métricas).
#De esta manera validamos resultados

freq = y_test_total['Tipo_planeta'].value_counts()
print('Frecuencia de tipos de planeta en test data= ')
print(freq)




















