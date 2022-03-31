# -*- coding: utf-8 -*-
"""
Recurso deteccion planetas
https://exoplanetasliada.wordpress.com/metodos-de-deteccion/
viridis
Created on Wed Aug 18 19:35:40 2021
Fecha descaga archivos de recurso Web Agosto 29 de 2021
@author: Asus-

"""
########## LIBRERÍAS A UTILIZAR ##########

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
#Convertir periodo orbital de dias a años OJO ESTA LINEA ESTABA COMENTADA
tabla_definitiva['Periodo_orbital'] = tabla_definitiva['Periodo_orbital'] * 0.002739726
#Elimina isnstancias con nan
tabla_definitiva=tabla_definitiva.dropna()
#Visualizamos Datagframe definitivo
print(tabla_definitiva)

########## PREPROCESAMIENTO ##########

#En las siguientes lineas se procede a construccion de graficas
#Se realiza la grafica Radio del planeta-Eje semimayor
#Con valores por defecto
radio_Porbi=sns.relplot(x='Radio_planeta',y='Eje_semimayor',data=tabla_definitiva)

#Se delimita el rango de visualizacion en ordenada
radio_Porbi=sns.relplot(x='Radio_planeta',y='Eje_semimayor',data=tabla_definitiva)
radio_Porbi.set(ylim=(-0.1,1.5))

#Coordenadas en escala logartimica
radio_Porbi=sns.relplot(x='Radio_planeta',y='Eje_semimayor',data=tabla_definitiva)
#Definimos limites de la escalas
radio_Porbi.set(xlim=(0.3,50))
radio_Porbi.set(ylim=(0.003,1000))
#Ejes en escala logaritmica
plt.xscale("LOG")
plt.yscale("LOG")

#Grafica de dispersion con leyenda de etiquetas
#Radio, masa del planeta y eje semimayor
#Se realiza la grafica Radio del planeta-Eje_semimayor
radio_Porbi=sns.relplot(x='Radio_planeta',y='Eje_semimayor',hue='Tipo_planeta',palette='ch:s=-.2,r=.6',data=tabla_definitiva)
#Definimos limites de la escalas
radio_Porbi.set(xlim=(0.3,50))
radio_Porbi.set(ylim=(0.003,1000))
#Nombre y etiquetas de la gráfica
#plt.title("Gráfica de dispersión Radio del Planeta-Periodo Orbital")
plt.xlabel("Radio del planeta en terminos del radio de la Tierra")
plt.ylabel("Eje semimayor (UA)")
#Ejes en escala logaritmica
plt.xscale("LOG")
plt.yscale("LOG")

#Se realiza la grafica Masa del planeta-Eje_semimayor
masa_Porbi=sns.relplot(x='Masa_planeta',y='Eje_semimayor',hue='Tipo_planeta',palette='ch:s=-.2,r=.6',data=tabla_definitiva)
#Definimos limites de la escalas
masa_Porbi.set(xlim=(0.03,40000))
masa_Porbi.set(ylim=(0.003,1000))
#Nombre y etiquetas de la gráfica
#plt.title("Gráfica de dispersión Masa del planeta-Periodo Orbital")
plt.xlabel("Masa del planeta en terminos de la masa de la Tierra")
plt.ylabel("Eje semimayor (UA)")
#Ejes en escala logaritmica
plt.xscale("LOG")
plt.yscale("LOG")

#Se realiza la grafica Eje_semimayor-Periodo Orbital
excen_Porbi=sns.relplot(x='Eje_semimayor',y='Periodo_orbital',hue='Tipo_planeta',palette='ch:s=-.2,r=.6',data=tabla_definitiva)
#Definimos limites de la escalas
excen_Porbi.set(xlim=(0.1,10))
excen_Porbi.set(ylim=(0.1,10))
#Nombre y etiquetas de la gráfica
#plt.title("Gráfica de dispersión Excentricida-Periodo Orbital")
plt.xlabel("Eje semimayor (UA)")
plt.ylabel("Periodo Orbital (días)")
#Ejes en escala logaritmica
plt.xscale("LOG")
plt.yscale("LOG")

#Se realiza la grafica masa-radio
excen_Porbi=sns.relplot(x='Masa_planeta',y='Radio_planeta',hue='Tipo_planeta',palette='ch:s=-.2,r=.6',data=tabla_definitiva)
#Definimos limites de la escalas
excen_Porbi.set(xlim=(0.01,1000000))
excen_Porbi.set(ylim=(0.2,100))
#Nombre y etiquetas de la gráfica
#plt.title("Gráfica de dispersión Excentricida-Periodo Orbital")
plt.xlabel("Masa del planeta en terminos de la masa de la Tierra")
plt.ylabel("Radio del planeta en terminos del radio de la Tierra")
#Ejes en escala logaritmica
plt.xscale("LOG")
plt.yscale("LOG")

#Se realiza la grafica Radio del planeta-Periodo orbital
radio_Porbi=sns.relplot(x='Radio_planeta',y='Periodo_orbital',hue='Tipo_planeta',palette='ch:s=-.2,r=.6',data=tabla_definitiva)
#Definimos limites de la escalas
radio_Porbi.set(xlim=(0.3,50))
radio_Porbi.set(ylim=(0.0003,1000))
#Nombre y etiquetas de la gráfica
#plt.title("Gráfica de dispersión Radio del Planeta-Periodo Orbital")
plt.xlabel("Radio del planeta en terminos del radio de la Tierra")
plt.ylabel("Periodo Orbital (días)")
#Ejes en escala logaritmica
plt.xscale("LOG")
plt.yscale("LOG")

#Se realiza la grafica Masa del planeta-Periodo orbital
masa_Porbi=sns.relplot(x='Masa_planeta',y='Periodo_orbital',hue='Tipo_planeta',palette='ch:s=-.2,r=.6',data=tabla_definitiva)
#Definimos limites de la escalas
masa_Porbi.set(xlim=(0.03,40000))
masa_Porbi.set(ylim=(0.0003,1000))
#Nombre y etiquetas de la gráfica
#plt.title("Gráfica de dispersión Masa del planeta-Periodo Orbital")
plt.xlabel("Masa del planeta en terminos de la masa de la Tierra")
plt.ylabel("Periodo Orbital (días)")
#Ejes en escala logaritmica
plt.xscale("LOG")
plt.yscale("LOG")
