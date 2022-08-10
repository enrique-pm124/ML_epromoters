#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 11:22:36 2022

@author: enriquepm124
"""
    
#%%
    
import sys 
import myFunctions
import openpyxl
#%% Interaccion por pantalla para obtener archivos 
hacerPaso = myFunctions.EjecutarPaso(1)
if (hacerPaso=="SI"):     
#   
    features = []
    answ_bed, file_pos, file_neg = myFunctions.quest_yn("bigWig")
    answ_bw, pos_ip_bed, neg_ip_bed = myFunctions.quest_yn("Bedgraph")
    ip_save = myFunctions.direc_save("archivos")
##    
#%% Parte 1 Leectura de datos y crear archivo con datos 

    file_pos = myFunctions.read_file_txt(file_pos)
    file_neg = myFunctions.read_file_txt(file_neg)
#%%
    
    
    ips_file_pos,pos_ip = myFunctions.make_fileV2(file_pos[0],file_pos[1],ip_save, features)
    ips_file_neg,neg_ip = myFunctions.make_fileV2(file_neg[0],file_neg[1],ip_save, features)
#%%
    myFunctions.descomprimir_direc(ip_save)
#%%   
    ip_data_pos = myFunctions.make_txtV2(ips_file_pos,"pos",ip_save)
    ip_data_neg = myFunctions.make_txtV2(ips_file_neg,"neg",ip_save)
    #%%
    posFeatures, posVar, posScore = myFunctions.get_scores5(ip_data_pos)
    negFeatures, negVar, negScore = myFunctions.get_scores5(ip_data_neg)
#%%
    posScore = myFunctions.imputer_score(posScore)
    negScore = myFunctions.imputer_score(negScore)
#%%
    features, posScoreBed = myFunctions.get_scoreBed(pos_ip_bed, features)
    features, negScoreBed = myFunctions.get_scoreBed(neg_ip_bed, features)
#%%
    # Introduccion del caso no y no 
    if answ_bw == "YES" and answ_bed == "YES":
        posScore = myFunctions.conc_score2(posScore, posScoreBed)
        negScore = myFunctions.conc_score2(negScore, negScoreBed)
    if answ_bw == "YES" and answ_bed == "NO":
        posScore = posScore
        negScore = negScore
    if answ_bw == "NO" and answ_bed == "YES":
        posScore = posScoreBed
        negScore = negScoreBed

#%% Guardar datos en txt
    
    myFunctions.SaveScore("Pos", ip_save, features, posVar, posScore)
    myFunctions.SaveScore("Neg", ip_save, features, negVar, negScore)
    
    ip_filepos = ip_save+"/"+"Pos_Score"
    ip_fileneg = ip_save+"/"+"Neg_Score"
    
    myFunctions.EscribirPaso(1)

if (hacerPaso=="NO"):
    ip_filepos = myFunctions.dirc_txt("de puntuacion positivas")
    if ip_filepos != "":
        ip_fileneg = myFunctions.dirc_txt("de puntuacion negativos")
    if ip_filepos == "" or ip_fileneg == "":
        sys.exit() #TErminar de golpe 
    ip_save = myFunctions.direc_save("archivos")
#%% Parte 2 Leer datos, seleccionar y normalizar escalas  

## Leectura de datos ya creados 
features, posScore, posVar = myFunctions.read_ScrTxt("Pos_Score", ip_filepos)
features, negScore, negVar = myFunctions.read_ScrTxt("Neg_Score", ip_fileneg)

#%% Analisis estadistico 

# Correlacion 
import pandas as pd

opert = ["Media",'Mediana','Desviacion tipica','Varianza','Coeficiente Varianza',"Error estandar"]
myFunctions.corr_Mtx(ip_save, posScore, features, "Pos")
myFunctions.corr_Mtx(ip_save, negScore, features, "Neg")

#%% Covarianza-   

myFunctions.mtx_cov(posScore, features, ip_save, "Pos")
myFunctions.mtx_cov(negScore, features, ip_save, "Neg")
#%%
dataP = pd.DataFrame(posScore)
dataN = pd.DataFrame(negScore)

mean_t1 = myFunctions.operations(dataP,opert[0],0) 
mean_t2 = myFunctions.operations(dataN,opert[0],0) 
median_t1 = myFunctions.operations(dataP, opert[1], 0)
median_t2 = myFunctions.operations(dataN, opert[1], 0)
std_t1 = myFunctions.operations(dataP, opert[2], 0)
std_t2 = myFunctions.operations(dataN, opert[2], 0)
var_t1 = myFunctions.operations(dataP, opert[3], 0)
var_t2 = myFunctions.operations(dataN, opert[3], 0)
cv_t1 = myFunctions.operations(dataP, opert[4], 0)
cv_t2 = myFunctions.operations(dataN, opert[4], 0)
err_tip1 = myFunctions.operations(dataP, opert[5], 0)
err_tip2 = myFunctions.operations(dataN, opert[5], 0)

result1 = [mean_t1, median_t1,std_t1,var_t1,cv_t1,err_tip1]
result2 = [mean_t2, median_t2,std_t2,var_t2,cv_t2,err_tip2]
#%% Comparaciones 

t_st, p_valor = myFunctions.t_test(dataP, dataN) 
    
#%% Ploteo de datos 

dataP = dataP.values.tolist()
dataN = dataN.values.tolist()
dataP = myFunctions.invertir_matriz(dataP)
dataN = myFunctions.invertir_matriz(dataN)

for i in range(len(features)):
    
    myFunctions.plots_datas(features[i],"",
    dataP[i],dataN[i],median_t1[i],median_t2[i],ip_save)
    
#%% Save excel  

    
wb = openpyxl.Workbook()# Crea excel
myFunctions.save_excel("Pos", features, result1, t_st, p_valor, opert, ip_save, wb)
myFunctions.save_excel("Neg", features, result2, t_st, p_valor, opert, ip_save, wb)


reject, p_valors, Sidak, Bonf = myFunctions.correct_t_test(p_valor)
