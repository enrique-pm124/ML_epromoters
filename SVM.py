#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 16:59:34 2022

@author: enriquepm124
"""

# Importar librerias de funciones 
import sys 
import myFunctions
import statistics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

#%% Interaccion por pantalla para obtener archivos 
hacerPaso = myFunctions.EjecutarPaso(1)
if (hacerPaso=="SI"):     
    features = []
    answ_bed, file_pos, file_neg = myFunctions.quest_yn("bigWig")
    answ_bw, pos_ip_bed, neg_ip_bed = myFunctions.quest_yn("Bedgraph")
    ip_save = myFunctions.direc_save("archivos")
#% Parte 1 Leectura de datos y crear archivo con datos 

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

hacerPaso = myFunctions.EjecutarPaso(2)
if (hacerPaso=="SI"):
## Leectura de datos ya creados 
    features, posScore, posVar = myFunctions.read_ScrTxt("Pos_Score", ip_filepos)
    features, negScore, negVar = myFunctions.read_ScrTxt("Neg_Score", ip_fileneg)

#%% Eleccion de caracteristicas
    inputfile = myFunctions.featuresChoose(features)
#%%    
# Selecciona los datos concretos    
    posScore = myFunctions.chooseRelevantColumns(posScore, features, inputfile)
    negScore = myFunctions.chooseRelevantColumns(negScore, features, inputfile)
    features = inputfile
#% Filtro logaritmico para el ruido 
    posScore = myFunctions.to_log(posScore)
    negScore = myFunctions.to_log(negScore)
                
#% Estandarizar y normalizar 
    Zpos,Zneg = myFunctions.calculateZscores2(posScore,negScore,negScore)
    
#% Unir scores y coordenadas 
    dataPos = myFunctions.conc_score2(posVar,Zpos)
    dataNeg = myFunctions.conc_score2(negVar,Zneg)

#% 1º Bucle remuestre

    l_prec_test = []; l_prec_train = []; l_Mconf_test = [];l_Mconf_train = [];
    for i in range(30):
        dataNeg2 = myFunctions.random_samples(dataNeg,len(Zpos))
        trainingScoresCons3 = dataPos + dataNeg2
        trainingResultsCons3 = ([1] * len(dataPos)) + ([0] * len(dataNeg2))

        # Dividimos regiones en training y test
        D_train, D_test, y_train, y_test = train_test_split(trainingScoresCons3, 
        trainingResultsCons3, test_size=0.2)

        D_train_var, X_train = myFunctions.sep_data(D_train)
        D_test_var, X_test = myFunctions.sep_data(D_test)

        SVM_model = myFunctions.performSVM(X_train, y_train)

        # Validacion cruzada 
        kf = KFold(n_splits=5)
        score = SVM_model.score(X_train,y_train) 
        scores = cross_val_score(SVM_model, X_train, y_train, cv=kf, scoring="accuracy")
        prec_mean = scores.mean()

        y_pred, pred_clss, prec, conf_matrix, exact = myFunctions.TestSVM2(X_test, y_test, SVM_model)
        y_pred_train, pred_clss_train, prec_train, conf_matrix_train, exact_train = myFunctions.TestSVM2(X_train, y_train, SVM_model)

        rmse = mean_squared_error(y_test, pred_clss)
        rmse = sqrt(rmse)

        l_prec_test.append(prec)
        l_prec_train.append(prec_mean)
        l_Mconf_train.append(conf_matrix_train)
        l_Mconf_test.append(conf_matrix)
#%% Calculo de prec, rango de error y matriz de confusion representativa
    prec_test = statistics.mean(l_prec_test)
    prec_train = statistics.mean(l_prec_train)
    edgePrec_test = [round((max(l_prec_test) - prec_test)*100,3), round((prec_test - min(l_prec_test))*100,3)]
    edgePrec_train = [round((max(l_prec_train) - prec_train)*100,3), round((prec_train - min(l_prec_train))*100,3)]

    ConfMatx_trainmean = myFunctions.Mconf_mean(l_Mconf_train)  
    ConfMatx_testmean = myFunctions.Mconf_mean(l_Mconf_test)             

    myFunctions.fig_Mconfusion(ip_save,"train_mean",ConfMatx_trainmean, prec_train,1,edgePrec_train) 
    myFunctions.fig_Mconfusion(ip_save,"test_mean",ConfMatx_testmean, prec_test,1,edgePrec_test)
#%% 2º Bucle de remuestreo 

    datatest = []; datatrain = []; testvar = []; trainVar = []; testScore = []
    trainScore = []; trainPrec = []; testPrec = []; prec_save = 1

    for i in range(30):
        dataNeg2 = myFunctions.random_samples(dataNeg,len(Zpos))
        trainingScoresCons3 = dataPos + dataNeg2
        trainingResultsCons3 = ([1] * len(dataPos)) + ([0] * len(dataNeg2))

        # Dividimos regiones en training y test
        D_train, D_test, y_train, y_test = train_test_split(trainingScoresCons3, trainingResultsCons3, test_size=0.2)

        D_train_var, X_train = myFunctions.sep_data(D_train)
        D_test_var, X_test = myFunctions.sep_data(D_test)

        SVM_model = myFunctions.performSVM(X_train, y_train)

        # Validacion cruzada 
        kf = KFold(n_splits=5)
        score = SVM_model.score(X_train,y_train) 
        scores = cross_val_score(SVM_model, X_train, y_train, cv=kf, scoring="accuracy")
        prec_mean = scores.mean()

        y_pred, pred_clss, prec, conf_matrix, exact = myFunctions.TestSVM2(X_test, y_test, SVM_model)
        y_pred_train, pred_clss_train, prec_train, conf_matrix_train, exact_train = myFunctions.TestSVM2(X_train, y_train, SVM_model)

        rmse = mean_squared_error(y_test, pred_clss)
        rmse = sqrt(rmse)
        if abs(prec_test-prec) <= prec_save:
            prec_save = abs(prec_test - prec)
            datatest = [y_pred, pred_clss, prec, conf_matrix, exact]
            datatrain = [y_pred_train, pred_clss_train, prec_mean, conf_matrix_train, exact_train]
            trainVar = D_train_var
            trainScore = X_train
            testVar = D_test_var
            testScore = X_test
            trainPrec = y_train
            testPrec = y_test
            SVM_save = SVM_model

#%% Curva ROC 
    myFunctions.ROC_curve(testPrec, datatest[0], ip_save)

#%% Curva PR
    myFunctions.PR_curve(testPrec, datatest[0] , ip_save)
#%% Matrices de confusion 
    myFunctions.fig_Mconfusion(ip_save,"test",datatest[3], datatest[4],0, edgePrec_test)
    myFunctions.fig_Mconfusion(ip_save,"train",datatrain[3], datatrain[4],0,edgePrec_train)
#%% Grafica de coeficientes 
    myFunctions.import_coef(ip_save,"",SVM_save,features)
#%% Guardar muestra 
    myFunctions.SaveTestSVM("",ip_save, testVar, datatest[2], datatest[1], datatest[0], testPrec)