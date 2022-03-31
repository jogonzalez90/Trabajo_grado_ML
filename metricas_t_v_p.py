# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 11:35:55 2021

@author: Asus
"""
#Librerías a importar

import pandas as pd
import numpy as np
import seaborn as sb
#import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

########## FILTRADO BASES DE DATOS ##########

#Lectura de archivo filtrado de la NASA (atributos)
exoplanetas_2=pd.read_csv('exoplanetas_3.csv')
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

########## APLICACIÓN DE ALGORITMOS DE MACHINE LEARNING ##########

#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report
#from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score#Validación cruzada
from sklearn.model_selection import KFold#Número de iteraciones
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score#Puntaje F1
from sklearn import metrics

#from pylab import rcParams

#from imblearn.under_sampling import NearMiss
#from imblearn.over_sampling import RandomOverSampler
#from imblearn.combine import SMOTETomek
#from imblearn.ensemble import BalancedBaggingClassifier
 
#from collections import Counter

# Seccion de codigo para coversion de datos a logaritmo natural
tabla_definitiva['Radio_planeta']=np.log(tabla_definitiva['Radio_planeta'])
tabla_definitiva['Masa_planeta']=np.log(tabla_definitiva['Masa_planeta'])
tabla_definitiva['Periodo_orbital']=np.log(tabla_definitiva['Periodo_orbital'])
tabla_definitiva['Eje_semimayor']=np.log(tabla_definitiva['Eje_semimayor'])

##############################################################################
# MODELOS CON ATRIBUTOS RADIO-MASA-PERIODO_ORBITAL-EJE_SEMIMAYOR DEL PLANETA #
######################## A LOGARITMO MATURAL #################################
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
Training=[]#Variable para métrica en entrenamiento
Cross_val=[]#Variable para métrica validación cruzada
Testing=[]#Variable para métrica prueba
F1=[]#Variable para métricas F1 Score

#########################################
############# Modelo SVM ################
#########################################

algoritmo_svm = SVC(kernel = 'linear')
kf = KFold(n_splits=5)
algoritmo_svm.fit(X_train_total, y_train_total)
score_3 = algoritmo_svm.score(X_train_total,y_train_total)
Training.append(score_3)
print("Métricas SVM")
print('Precisión training set= '+ str(score_3))

#Métrica en validación cruzada
scores_3 = cross_val_score(algoritmo_svm, X_train_total, y_train_total, cv=kf, scoring="accuracy")
Cross_val.append(scores_3.mean())
print('Precisión media cross-validation= '+ str(scores_3.mean()))

#Se lleva a cabo una prediccion y cálculo de métrica en datos de prueba
preds = algoritmo_svm.predict(X_test_total)
score_pred_3 = metrics.accuracy_score(y_test_total, preds)
Testing.append(score_pred_3)
print('Precisión testing set= '+ str(score_pred_3))

#Cálculo F1 para datos desbalanceados
f1_score_3=f1_score(y_test_total, preds,average='micro')
F1.append(f1_score_3)
print('F1 Score= '+ str(f1_score_3))

#########################################
############ Modelo k-NN ################
#########################################

algoritmo_knn = KNeighborsClassifier(n_neighbors=7)
algoritmo_knn.fit(X_train_total, y_train_total)
score_2 = algoritmo_knn.score(X_train_total,y_train_total)
Training.append(score_2)
print("Métricas k-NN")
print('Precisión training set= '+ str(score_2))

#Métrica en validación cruzada
scores_2 = cross_val_score(algoritmo_knn, X_train_total, y_train_total, cv=kf, scoring="accuracy")
Cross_val.append(scores_2.mean())
print('Precisión media cross-validation= '+ str(scores_2.mean()))

#Se lleva a cabo una prediccion y cálculo de métrica en datos de prueba
preds = algoritmo_knn.predict(X_test_total)
score_pred_2 = metrics.accuracy_score(y_test_total, preds)
Testing.append(score_pred_2)
print('Presición testing set= '+ str(score_pred_2))

#Cálculo F1 para datos desbalanceados
f1_score_2=f1_score(y_test_total, preds,average='micro')
F1.append(f1_score_2)
print('F1 Score= '+ str(f1_score_2))

##################################################
######## Modelo de Árboles de Decisión ###########
##################################################

algoritmo_tree = DecisionTreeClassifier(criterion= 'entropy', random_state=0)
algoritmo_tree.fit(X_train_total, y_train_total)
score_1 = algoritmo_tree.score(X_train_total,y_train_total)
Training.append(score_1)
print("Métricas árbol de decisión")
print('Precisión training set= '+ str(score_1))

#Métrica en validación cruzada
scores_1 = cross_val_score(algoritmo_tree, X_train_total, y_train_total, cv=kf, scoring="accuracy")
Cross_val.append(scores_1.mean())
print('Precisión media cross-validation= '+ str(scores_1.mean()))

#Se lleva a cabo una prediccion y cálculo de métrica en datos de prueba
preds = algoritmo_tree.predict(X_test_total)
score_pred_1 = metrics.accuracy_score(y_test_total, preds)
Testing.append(score_pred_1)
print('Presición testing set= '+ str(score_pred_1))

#Cálculo F1 para datos desbalanceados
f1_score_1=f1_score(y_test_total, preds,average='micro')
F1.append(f1_score_1)
print('F1 Score= '+ str(f1_score_1))

######### CONSTRUCCION TABLA PARA ANALISIS DE RESULTADOS ##########

#Estructura de la tabla todos los atributos
modelos=['Máquinas de Vectores','k-Vecinos más cercanos','Árbol de decisiones']
#Estructura de la tabla
tabla_resultados={'Modelos Ml (Log)':modelos,
                  'Training':Training,
                  'Cross_val':Cross_val,
                  'Testing':Testing,
                  'F1 Score':F1,
                  }
resultados=pd.DataFrame(tabla_resultados)
print(resultados)

#################################
#### Tabla de resultados ########
#################################
print("###################")
print("Sintesis de métricas")
print("###################")
print("Métricas SVM")
print('Precisión training set= '+ str(score_3))
print('Precisión media cross-validation= '+ str(scores_3.mean()))
print('Presición testing set= '+ str(score_pred_3))
print('F1 Score= '+ str(f1_score_3))
print("-----------------------------")
print("Métricas k-NN")
print('Precisión training set= '+ str(score_2))
print('Precisión media cross-validation= '+ str(scores_2.mean()))
print('Presición testing set= '+ str(score_pred_2))
print('F1 Score= '+ str(f1_score_2))
print("-----------------------------")
print("Métricas árbol de decisión")
print('Precisión training set= '+ str(score_1))
print('Precisión media cross-validation= '+ str(scores_1.mean()))
print('Presición testing set= '+ str(score_pred_1))
print('F1 Score= '+ str(f1_score_1))
print("-----------------------------")

#Visulizamos frecuenica de clases para
#evaluar datos balanceados
balan = tabla_definitiva['Tipo_planeta'].value_counts()
print('Frecuencia de tipos de planeta en conjunto de datos= ')
print(balan)
print("-----------------------------")

#A traves de estas lineas de código visualizamos la
#frecuencia con que aparece cada tipo de exoplaneta
#en el test data, para corroborar con la matriz de 
#confusion de SVM 'matriz_tree_total'(que logra las mejores métricas).
#De esta manera validamos resultados

freq = y_test_total['Tipo_planeta'].value_counts()
print('Frecuencia de tipos de planeta en test data= ')
print(freq)

##########################################
#### ESTUDIO  DE ATRIBUTOS TECNICA DE ####
######## DESCUBRIMIENTO Y AÑO ############
##########################################

#Se crea varible con atributos nombre del planeta, tecnica de
#descubrimiento y año
exoplanetas_4=exoplanetas_2[['pl_name','discoverymethod','disc_year']]

#Eliminanos instancias con otros métodos de detección
exoplanetas_4 = exoplanetas_4.drop(exoplanetas_4[exoplanetas_4['discoverymethod']=='Orbital Brightness Modulation'].index)
exoplanetas_4 = exoplanetas_4.drop(exoplanetas_4[exoplanetas_4['discoverymethod']=='Disk Kinematics'].index)
exoplanetas_4 = exoplanetas_4.drop(exoplanetas_4[exoplanetas_4['discoverymethod']=='Pulsation Timing Variations'].index)
exoplanetas_4 = exoplanetas_4.drop(exoplanetas_4[exoplanetas_4['discoverymethod']=='Disk Kinematics'].index)
exoplanetas_4 = exoplanetas_4.drop(exoplanetas_4[exoplanetas_4['discoverymethod']=='Pulsar Timing'].index)
exoplanetas_4 = exoplanetas_4.drop(exoplanetas_4[exoplanetas_4['discoverymethod']=='Transit Timing Variations'].index)
exoplanetas_4 = exoplanetas_4.drop(exoplanetas_4[exoplanetas_4['discoverymethod']=='Eclipse Timing Variations'].index)
print(exoplanetas_4)

print("-------")
exoplanetas_4 = exoplanetas_4.rename(columns={'discoverymethod':'Método de descubrimiento'})#Cambio nombre de atributo
freq_1 = exoplanetas_4['Método de descubrimiento'].value_counts()
print('Frecuencia de metodos de detección= ')
print(freq_1)
sb.factorplot('Método de descubrimiento',data=exoplanetas_4,kind="count", aspect=1.5)#Gráfica

print("--------")
exoplanetas_4 = exoplanetas_4.rename(columns={'disc_year':'Año de descubrimiento'})#Cambio nombre de atributo
freq_2 = exoplanetas_4['Año de descubrimiento'].value_counts()
print('Frecuencia año de descubrimiento= ')
print(freq_2)
sb.factorplot('Año de descubrimiento',data=exoplanetas_4,kind="count", aspect=3)#Gráfica

####################################
#### No de descubrimientos por  ####
######## observatorio ##############
####################################
print("-----------------------------")
des = exoplanetas_2['disc_facility'].value_counts()
print('Frecuencia de sondas en conjunto de datos= ')
print(des)

"""
#Estrategia: Ensamble de Modelos con Balanceo
#Para esta estrategia usaremos un Clasificador de Ensamble que utiliza
#Bagging. Veamos como se comporta el SVM 

svm_balanceo = BalancedBaggingClassifier(base_estimator=SVC(kernel = 'linear'),
                                sampling_strategy='auto',
                                replacement=False,
                                random_state=0)

#Train the classifier.
svm_balanceo.fit(X_train_total, y_train_total)
pred_y = svm_balanceo.predict(X_test_total)
#mostrar_resultados(y_test, pred_y)
score_pred_1 = metrics.accuracy_score(y_test_total, pred_y)
#Testing.append(score_pred_1)
print("--------")
print("Ensamble de modelos con balanceo")
print('Precisión SVM= '+ str(score_pred_1))

#Veamos como se comporta el KNN     

knn_balanceo = BalancedBaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=7),
                                sampling_strategy='auto',
                                replacement=False,
                                random_state=0)

#Train the classifier.
knn_balanceo.fit(X_train_total, y_train_total)
pred_y = knn_balanceo.predict(X_test_total)
#mostrar_resultados(y_test, pred_y)
score_pred_1 = metrics.accuracy_score(y_test_total, pred_y)
#Testing.append(score_pred_1)
print('Precisión knn= '+ str(score_pred_1))

#Veamos como se comporta el árbol de decision 

arbol_balanceo = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(criterion= 'entropy', random_state=0),
                                sampling_strategy='auto',
                                replacement=False,
                                random_state=0)

#Train the classifier.
arbol_balanceo.fit(X_train_total, y_train_total)
pred_y = arbol_balanceo.predict(X_test_total)
#mostrar_resultados(y_test, pred_y)
score_pred_1 = metrics.accuracy_score(y_test_total, pred_y)
#Testing.append(score_pred_1)
print('Precisión árbol de decision= '+ str(score_pred_1))
print("--------")
"""












