#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 13:44:32 2022

@author: enriquepm124
"""

#%% 
#%% entreno original
#import training
import myFunctions
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#%%
quest_data, trainingPositives, trainingNegatives = myFunctions.quest_yn("de conjunto de datos")

quest_ndata, file_pos, file_neg = myFunctions.quest_yn("de nuevos datos bigwig")

if quest_data == "NO":
    quit()
if quest_data == "YES":
    ip_save = myFunctions.direc_save("archivos")
        
#%%

if quest_ndata == "YES":
    positiveFeatures, positivePosition, positiveScores = myFunctions.getScoresV2(trainingPositives)
    negativeFeatures, negativePosition,negativeScores = myFunctions.getScoresV2(trainingNegatives)
    
    files_pos = myFunctions.read_file_txt2(file_pos)
    files_neg = myFunctions.read_file_txt2(file_neg)
    
    posVar, posFeature, N_posScore = myFunctions.get_newScore(files_pos, ip_save)
    negVar, negFeature, N_negScore = myFunctions.get_newScore(files_neg, ip_save)
    
    positiveFeatures, posPosition, positiveScores = myFunctions.conc_reg(positivePosition,posVar,
    positiveScores,N_posScore,positiveFeatures,posFeature)
    negativeFeatures, negPosition, negativeScores = myFunctions.conc_reg(negativePosition,negVar,
    negativeScores,N_negScore,negativeFeatures,negFeature)

if quest_ndata == "NO":
    positiveFeatures, positivePosition, positiveScores = myFunctions.getScoresV2(trainingPositives)
    negativeFeatures, negativePosition,negativeScores = myFunctions.getScoresV2(trainingNegatives)
    

#%% Imputacion de reultados 
positiveScores = myFunctions.imputer_score(positiveScores)
negativeScores = myFunctions.imputer_score(negativeScores)
#%% Z_score 

Zpos, Zneg = myFunctions.calculateZscore2(positiveScores, negativeScores, negativeScores)

trainingScores = Zpos + Zneg

trainingResults = ([1] * len(Zpos)) + ([0] * len(Zneg))

X_train, X_test, y_train, y_test = train_test_split(trainingScores, trainingResults, test_size=0.2)

SVM_model = myFunctions.performSVM(trainingScores, trainingResults)

# Validacion cruzada 
kf = KFold(n_splits=5)
score = SVM_model.score(X_train,y_train) 
scores = cross_val_score(SVM_model, X_train, y_train, cv=kf, scoring="accuracy")
prec_mean = scores.mean()

y_pred, pred_clss, prec, conf_matrix, exact = myFunctions.TestSVM2(X_test, y_test, SVM_model)
y_pred_train, pred_clss_train, prec_train, conf_matrix_train, exact_train = myFunctions.TestSVM2(X_train, y_train, SVM_model)

#%%

myFunctions.fig_Mconfusion(ip_save,"test", conf_matrix, exact, 0, 0)
myFunctions.fig_Mconfusion(ip_save,"train", conf_matrix_train, prec_mean, 0, 0)

myFunctions.import_coef(ip_save,"",SVM_model, positiveFeatures)
#%% Curva ROC 
myFunctions.ROC_curve(y_test, y_pred, ip_save)
#%% Curva PR
myFunctions.PR_curve(y_test, y_pred, ip_save)