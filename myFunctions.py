#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 17:03:57 2022

@author: enriquepm124
"""

#%% Librerias

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import scipy.stats as ss
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import statistics
import math
import random
import seaborn as sns
from scipy.optimize import curve_fit 
from sklearn import svm
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
    
#%% Funciones de lectura, escritura y interaccion por pantalla


def EscribirPaso(paso):
        fConfig=open('Config.txt','w')
        fConfig.seek(0)
        fConfig.write(str(paso))
        fConfig.close()
        print "**** PASO "+str(paso)+ " realizado de forma correcta\n"
#-------------------------------------------------------------------------
def LeerPaso():
    try:
        fConfig=open('Config.txt','r')
        paso=int(fConfig.read())
        fConfig.close()
        
    except IOError:
        EscribirPaso(0)
        paso=0
    return paso
#-------------------------------------------------------------------------
def EjecutarPaso(pasoParaEjecutar): # Pregunta si queremos realizar el Paso, devuelve SI o NO
    pasoGuardado=LeerPaso()
    
    answer="SI"
    if pasoParaEjecutar<=pasoGuardado:
        answer=raw_input ("Este paso ya se realizo, Desea ejecutar nuevamente este paso ? (Si-No) [No] ")
        answer=answer.upper()
        if answer != "SI":
            answer="NO"
    if answer=="SI":
        print "El Paso "+ str(pasoParaEjecutar) + " se va a realizar.\n"

    return answer


def quest_yn(string):
    n = 0
    while n == 0:
        answ = raw_input("Introducir archivos "+string+" (Yes/No): ")
        answ = answ.upper()
        if answ == "NO":
            return answ, "", ""
        if answ == "YES":
            file_pos = dirc_txt("de carteristicas positivas")
            file_neg = dirc_txt("de carteristicas negativas")
            return answ, file_pos, file_neg
#%
           
def dirc_txt(string): 
    answer_input = " "
    while answer_input != "":
        answer_input = raw_input("Direccion del archivo "+string+": ")
        try:
            file = open(answer_input)
            print("Archivo encontrado") 
            file.close()
            return answer_input
        except:
            print("No encontrado")
            answ = repet_func()
        if answ == True:
            #break
            return ""
                  
def repet_func():
    n = 0
    while n ==0:
        answer_repet = raw_input("Desea introducir nueva direccion(Yes/No): ")
        answer_repet = answer_repet.upper()
        if answer_repet == "NO":
            return True
        if answer_repet == "YES":
            return False

#%
def read_file_txt(ip): # modificar para numero de caract o file 
    list_ip = []
    pos = []
    n = 0
    with open(ip) as f:
        for line in f:        
            if line.strip(): # Elimina huecos            
                fila = line.strip().split()
                if fila[0] == "#":
                    pos.append(n)
                    continue
                else:
                    n+=1
                    list_ip.append(fila)
    if len(pos) == 2: # Sino pones # Datos da error 
        bw = list_ip[pos[0]:pos[1]]
        bed = list_ip[pos[1]:]
        return bw,bed
    else:
        print("Error en archivo txt")

        
def direc_save(string):
    be_dirc = False
    while be_dirc == False:
        dirc_save = raw_input("Direccion del directorio para guardar "+string+": ")
        be_dirc = os.path.isdir(dirc_save)    
        if be_dirc == True:
            print("Directorio encontrado")
            return dirc_save
        if be_dirc == False:
            print("Directorio no encontrado")

def name_save():
    name_save = ""
    while name_save == "":
        name_save = raw_input("Nombre del archivo: ")
        name_save = name_save.strip()
        if name_save == "":
            print("No ha introducido ningun nombre")
    return name_save 

#%% Funciones para obtener datos de los archivos
    
def make_file(file_bw,file_bed,ip_save):
    ip_graph = []
    for i in range(len(file_bw)):
        for j in range(len(file_bed)):
            os.system("computeMatrix reference-point -S "+file_bw[i][1]+" -R "+\
            file_bed[j][4]+" -a "+ file_bed[j][1]+" -b "+file_bed[j][2]+" -bs "+\
            file_bed[j][3]+"\t"+"--nanAfterEnd"+" -o "+ip_save+"/"+file_bw[i][0]+"_"+file_bed[j][0]+".bed.tab.gz"+\
            " --outFileSortedRegions "+ip_save+"/"+file_bw[i][0]+"_"+file_bed[j][0]+".bed")
            name_file = file_bw[i][0]+"_"+file_bed[j][0]
            ip_file = ip_save+"/"+file_bw[i][0]+"_"+file_bed[j][0]+".bed.tab.gz"
            ip_graph.append([name_file,ip_file])
    return ip_graph

def make_fileV2(file_bw, file_bed, ip_save, features):
    ip_graph = []
    ips_make = []
    for i in range(len(file_bw)):
        ip_graph.append([])
        for j in range(len(file_bed)):
            os.system("computeMatrix reference-point -S "+file_bw[i][1]+" -R "+\
            file_bed[j][4]+" -a "+ file_bed[j][1]+" -b "+file_bed[j][2]+" -bs "+\
            file_bed[j][3]+"\t"+"--nanAfterEnd"+" -o "+ip_save+"/"+file_bw[i][0]+"_"+file_bed[j][0]+".bed.tab.gz"+\
            " --outFileSortedRegions "+ip_save+"/"+file_bw[i][0]+"_"+file_bed[j][0]+".bed")
            name_file = file_bw[i][0]+"_"+file_bed[j][0]
            ip_file_tab = ip_save+"/"+file_bw[i][0]+"_"+file_bed[j][0]+".bed.tab.gz"
            ip_file_bed = ip_save+"/"+file_bw[i][0]+"_"+file_bed[j][0]+".bed"
            ip_graph[-1].append([name_file,ip_file_tab])
            ips_make.append(ip_file_bed)
            ips_make.append(ip_file_tab)
            if not file_bw[i][0] in features:
                features.append(file_bw[i][0])
    return ip_graph,ips_make

def make_graph(ips_file_graph,ip_save):
    
    for i in range(len(ips_file_graph)):
        graph_name = ips_file_graph[i][0]
        os.system("plotHeatmap -m "+ips_file_graph[i][1]+" -out "+ip_save+"/"+graph_name+"_heatmap.png "+"--samplesLabel "+graph_name+" --regionsLabel 'binding site' -x 'site distance'")
#-----------------        

# Descomprimir carpetas gzip
def descomprimir_direc(ip):
    os.system("gzip -dr "+ip)
    
# archivo txt que guarda name e ips de datos creados descomprimidos 
    
def make_txt(list_data,ip):
    file = open(ip+"/Data_file","w") # Data_file
    for i in range(len(list_data)):
        list_str = list_data[i][1].split(".")
        list_str.remove('gz')
        conc_str = ".".join(list_str)
        file.write(list_data[i][0]+" ")
        file.write(conc_str +"\n")
    file.close()

def make_txtV2(list_data,name,ip):
    file = open(ip+"/"+name+"Data_file","w") # Data_file
    for i in range(len(list_data)):
        file.write("# Caracteristica " +str(i+1)+ "\n")
        for j in range(len(list_data[0])):
            list_str = list_data[i][j][1].split(".")
            list_str.remove('gz')
            conc_str = ".".join(list_str)
            file.write(list_data[i][j][0]+" ")
            file.write(conc_str +"\n")
    file.close()
    return ip+"/"+name+"Data_file"
#%%
def read_file_txt2(ip): # modificar para numero de caract o file 
    list_ip = []
    pos = []
    cart = []
    n = 0
    with open(ip) as f:
        for line in f:        
            if line.strip(): # Elimina huecos            
                fila = line.strip().split()
                if fila[0] == "#":
                    pos.append(n)
                    continue
                else:
                    n+=1
                    list_ip.append(fila)
    for i in range(len(pos)):
        if len(pos) == i+1:
            crt = list_ip[pos[i]:]
        else:
            crt = list_ip[pos[i]:pos[i+1]]
        cart.append(crt)
    return cart
   
def elim_nan(mat):
    posnan = []
    for j in range(len(mat)):
        pos = [i for i in range(len(mat[j])) if np.isnan(mat[j][i]) == True]
        posnan.extend(pos)
        mat = np.delete(mat, posnan, axis=1)
    return posnan, mat

def invertir_matriz(matriz): 
    y = []
    for i in range(len(matriz[0])):
        y.append([])
        for j in range(len(matriz)):
            y[-1].append(matriz[j][i])
    return y 

def invertir_list(lista): 
    y = []
    for j in range(len(lista)):
        y.append([])
        y[-1].append(lista[j])
    return y 

def get_scores5(ip_file):
    carts = read_file_txt2(ip_file)
    features = []
    list_var = []
    list_scores=[]
    l_gauss=[]
    for i in range(len(carts)):
        crt_one = carts[i]
        names,ips = type_data(crt_one)
        features.append(names)
        for j in range(len(ips)):
            data = data_matrix(names[j],ips[j])
            var, scores = sep_dat(data)
            d_score = operations(scores, 'Mediana', 1)#Datos por regiones 
            l_score = d_score.tolist()
            list_scores.extend(l_score)
            if i == 0:
                var = var.values.tolist()
                list_var+=var
        l_gauss.append(list_scores) # junta caracteristicas
        list_scores = []
    mat_gauss = invertir_matriz(l_gauss) # invierte matriz filas a columnas
    return features, list_var, mat_gauss

def getScoresV2(trainingScoreFile):
    features=[]
    featureScores=[]
    featurePositions=[]
    try:
		ip = open(trainingScoreFile, "r")
    except:
		sys.stderr.write("Issue: Cannot open training file " + trainingScoreFile + "\n")
		sys.exit()
    header = ip.readline()
    fields = header.strip().split("\t")
    features = fields[6:]
    for line in ip:
        fields = line.strip().split("\t")
        featureScores.append([])
        featurePositions.append([])
        for idx in range(len(fields)):
            if idx < 6:
                featurePositions[-1].append(fields[idx])
            if idx >= 6:
                featureScores[-1].append(float(fields[idx]))
    return features,featurePositions,featureScores

def conv_nan(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if np.isnan(matrix[i][j]) == True:
                matrix[i][j] = "NaN"
          
#%Impuradicon de muestras 
from sklearn.preprocessing import Imputer
def imputer_score(matrix):
    conv_nan(matrix)
    imp_median = Imputer(missing_values='NaN', strategy='median', axis=0)
    imp_median.fit(matrix)
    matrix_imp = imp_median.transform(matrix)
    matrix_imp = matrix_imp.tolist()
    return matrix_imp



#------------------------------------------ Paso 2 
def read_txt(ip_txt):
    name_ip = []
    with open(ip_txt)as f:
        for line in f:
            name_ip.append(line.strip().split())
    name, ip = type_data(name_ip)
    return name, ip
# correccion de errores con try y except
def type_data(ips):
    ip = []
    name = [] 
    for i in range(len(ips)):
        ip.append(ips[i][1].strip())
        name.append(ips[i][0].strip())
    return name, ip
    

# Conviente la ip en datos en formato str
def data_matrix(name,ip):
    content = []
    try:
        with open(ip) as f:
            for line in f:
                content.append(line.strip().split())
        file = content[1:]
        return file   
    except:
        print('No se encontro una direccion ip de '+name)

def data_matrix2(name,ip,line_op): # Vesion mejorada 
    content = []
    try:
        with open(ip) as f:
            for line in f:
                content.append(line.strip().split())
        file = content[line_op:]
        return file   
    except:
        print('No se encontro una direccion ip de '+name)
   
def read_cartbed3(ip_cart,ip_reg):
    new_cart = data_matrix2("Interactions", ip_cart, 1)
    data = data_matrix2("Regions", ip_reg, 0)
    score_cart = []
    for i in range(len(data)):
        score_cart.append([])
        #score_cart.append([data[i][0],data[i][1],data[i][2]])
        for j in range(len(new_cart)):
            if data[i][0] == new_cart[j][0]: 
                if int(new_cart[j][1]) <= int(data[i][1]) <= int(new_cart[j][2]) and int(new_cart[j][1]) <= int(data[i][2]) <= int(new_cart[j][2]):
                    score_cart[-1].append(float(new_cart[j][3]))
                    continue
                if int(data[i][1]) <= int(new_cart[j][1]) and int(new_cart[j][2]) <= int(data[i][2]):
                    score_cart[-1].append(float(new_cart[j][3]))
                    continue
                if int(new_cart[j][1]) <= int(data[i][1]) <= int(new_cart[j][2]):
                    score_cart[-1].append(float(new_cart[j][3]))
                    continue
                if int(new_cart[j][1]) <= int(data[i][2]) <= int(new_cart[j][2]):
                    score_cart[-1].append(float(new_cart[j][3]))
                    continue
            if len(new_cart) == j+1:
                if len(score_cart[i]) == 0:
                    score_cart[-1].append(0)

    for x in range(len(score_cart)):
        if not score_cart == 0:
            score_cart[x] = float(statistics.median(score_cart[x]))
    return score_cart

def get_newScore(files,ip_save):
    ips_file = make_file(files[0],files[1],ip_save) # revisar cambio --Nan
    descomprimir_direc(ip_save)
    make_txt(ips_file,ip_save)

    names,ips = read_txt(ip_save+"/"+"Data_file") # Cambiar para introduccior variable de nombre en maketxt
    datas = data_matrix(names[0],ips[0])
    var, scores = sep_dat(datas)
    d_score = operations(scores,"Mediana",1)
    l_score = d_score.tolist()
    var = var.drop([5],axis=1)
    var = var.values.tolist()
    return var, names, l_score

def get_scoreBed(ip_bed,features):
    cart = read_file_txt2(ip_bed)
    S_total = []
    Score_cart = []
    for i in range(len(cart[0])):
        if not cart[0][i][0] in features:
            features.append(cart[0][i][0])
        for j in range(len(cart[1])):
            ScoreBed = read_cartbed3(cart[0][i][1],cart[1][j][1])
            Score_cart.extend(ScoreBed)
        S_total.append(Score_cart)
        Score_cart = []
    S_total = invertir_matriz(S_total)    
    return features, S_total    

def conc_score(matrix,lista):
    matrix_conc = []
    for i in range(len(lista)):
        matrix_conc.append([])
        matrix_conc[-1].extend(matrix[i])
        matrix_conc[-1].append(lista[i])
    return matrix_conc

def conc_reg(position1,position2,score1,score2,features1,features2):
    positions=[]
    scores=[]
    n=0
    features1.append(features2[0])
    for i in range(len(position1)):
        if len(scores) < len(position2):
            if set(position1[i][:2]) == set(position2[i-n][:2]):
                positions.append(position1[i])
                scores.append(score1[i])
            else:
                n+=1
    mtx_score = conc_score(scores,score2)
    return features1, positions, mtx_score

def conc_score2(matrix,lista):
    matrix_conc = []
    for i in range(len(matrix)):
        matrix_conc.append([])
        matrix_conc[-1].extend(matrix[i])
        matrix_conc[-1].extend(lista[i])
    return matrix_conc

# Separa puntuaciones y reto de variables
def sep_dat(data):
    matrix = pd.DataFrame(data)
    var = matrix.iloc[:,:6]
    punt = matrix.iloc[:,6:]
    punt = punt.astype(float)
    return var, punt


#%% Guardar datos calculados en la lectura y lee datos de ese archivo 
def SaveScore(opPrefix,ip,features, Var, scores):
    op1 = open(ip+"/"+opPrefix + "_Score" ,"w")
    op1.write("Chrom"+"\t"+"Start"+"\t"+"End"+"\t")
    for i in range(len(features)):
        op1.write(features[i]+"\t")
    op1.write("\n")
    for j in range(len(scores)):
        op1.write(Var[j][0]+"\t"+Var[j][1]+"\t"+Var[j][2]+"\t")
        for i in range(len(scores[0])):
            op1.write(str(scores[j][i])+"\t")
        op1.write("\n")
    op1.close()
    return

def read_ScrTxt(name,ip):
    Var = []
    Score = []
    data = data_matrix2(name,ip, 0)
    features = data[0][3:]
    for i in range(1,len(data)):
        Var.append([])
        Score.append([])
        for j in range(len(data[0])):
            if j < 3:
                Var[-1].append(data[i][j])
            if j >= 3:
                Score[-1].append(float(data[i][j]))
    return features, Score, Var

#%% Elejir caracteristicas 
    
def featuresChoose(features):
    l_features = []
    n = 0
    features2 = [x.upper() for x in features]
    while n == 0:
        answ = raw_input("Desea seleccionar una caracteristica (Yes/No/All): ")
        if answ.upper() == "YES":
            answ2 = raw_input("Selecciones alguna caracteristica\n"+"("+ ", ".join(features) + "): ")
            if answ2.upper() in features2:
                if features2.index(answ2.upper()) in l_features:
                    print("Caracteristica ya introducida")
                if not features2.index(answ2.upper()) in l_features:
                    l_features.append(features2.index(answ2.upper()))
            if not answ2.upper() in features2:
                print("Caracteristica mal escrita")
        if answ.upper() == "NO":
            if len(l_features) >= 1:
                l_features
                n+=1
            if len(l_features) == 0:
                print("Debe seleccionar alguna caracteristica")
        if answ.upper() == "ALL":
            return features
    l_features = sorted(l_features)
    for i in range(len(l_features)):
        l_features[i] = features[l_features[i]]
        
        
    return l_features
    
def chooseRelevantColumns(scores, features, inputFiles):
	#This function chooses features based on input
    idx = 0
    relFeatures = []
    ChooseFeatures =[]
    for currFeature in features:
        if currFeature in inputFiles:
            ChooseFeatures.append(currFeature)
            relFeatures.append(idx)
        idx += 1
    for currFeature in inputFiles:
		if currFeature not in features:
			sys.stderr.write("The feature " + currFeature + " does not have training data"+"\n")
			sys.exit()

	#Now choosing just the relevant scores
    relScores = []
    for idx1 in range(0, len(scores)):
		relScores.append([])
		for idx2 in range(0, len(scores[0])):
			if idx2 in relFeatures:
				relScores[-1].append(scores[idx1][idx2])

    return relScores

#%% Pasar a base logaritmica 
    
def to_log(matrixScore):
    for i in range(len(matrixScore)):
        for j in range(len(matrixScore[0])):
            signal = matrixScore[i][j]
            if (signal == 0.0):
                matrixScore[i][j] == 0.0
            else:
                signal = math.log(signal+0.01,10)
                matrixScore[i][j] == signal
    return matrixScore

#%% Noramlizar y estandarizar

def gauss_function(x, a, x0, sigma):
	#Gaussian function
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def concentScore(listScore,nBin):
    y, binEdges = np.histogram(listScore, bins = nBin)
    y = y/float(len(listScore))
    x = np.zeros(nBin)
    for i in range(len(binEdges) - 1):
        x[i] = (binEdges[i] + binEdges[i+1])/2
        x = np.array(x)
    return x, y

def fitScore(function, dataX, dataY, p0):
    try:
        nopt, ncov = curve_fit(function, dataX, dataY, p0= p0)
    except:
        nopt = [1, p0[1], p0[2]]
    return nopt

def Zscore(matZ, idx, matScore, median, sigma):
    idx2 = 0
    for currScore in matScore:
        matZ[idx2].append((currScore[idx] - median)/sigma)
        idx2 +=1
    return matZ

def calculateZscore2(positiveScore,negativeScore, BackgroundScore):
    Zpos = [[] for i in range(len(positiveScore))]
    Zneg = [[] for j in range(len(negativeScore))]
    for idx in range(len(positiveScore[0])):
        subset = []
        for currX in BackgroundScore:
            if currX[idx] == 0:
                continue
            subset.append(currX[idx])
        dataX, dataY = concentScore(subset, 50)
        median = np.median(subset)
        sigma = np.std(subset)
        nopt = fitScore(gauss_function, dataX, dataY, [max(dataY),median,sigma])
        Zscore(Zpos, idx, positiveScore, nopt[1],nopt[2])
        Zscore(Zneg, idx, negativeScore, nopt[1],nopt[2])
    return Zpos,Zneg


#%%

def random_samples(lista,limt):
    n = 1
    list_rdm = []
    while n <= limt:
        x = random.choice(lista) 
        if x in list_rdm:
            continue
        list_rdm.append(x)
        n+=1
    return list_rdm

def sep_data(matrix):
    var = []
    score = []
    for i in range(len(matrix)):
        var.append(matrix[i][:3])
        score.append(matrix[i][3:])
    return var, score

def TestSVM2(X_test,y_test,SVM_model):
    y_pred= SVM_model.predict_proba(X_test) # Porcentaje de predicciones 
    pred = SVM_model.predict(X_test) # Predicion en clases
    prec = SVM_model.score(X_test,y_test) # Preccision del modelo 
    conf_matrix = confusion_matrix(y_test,pred)
    exactitud = accuracy_score(y_test, pred)
    return y_pred, pred, prec, conf_matrix, exactitud

def performSVM(trainingScores, trainingResults):
	#print "->SVM training"

	clf = svm.SVC(kernel='linear', probability=True)
	clf.fit(trainingScores, trainingResults)
	#print clf.coef_
	#print "<-SVM training"

	return clf
#%%Training y test 
    
def Mconf_mean(lista):
    Matx =[]
    for i in range(len(lista[0][0])):
        for j in range(len(lista[0])):
            Matx.append([])
            for z in range(len(lista)):
                Matx[-1].append(lista[z][i][j])        
    for x in range(len(Matx)):
        Matx[x] = int(statistics.mean(Matx[x]))
    Mconf = []
    for y in range(0,len(Matx),2):
        Mconf.append([Matx[y],Matx[y+1]])
    return Mconf

def fig_Mconfusion(ip_save,name,conf_matrix,exact,ansR, edge):
    fig, ax = plt.subplots()
    dataframe = pd.DataFrame(conf_matrix, index=["promoter","epromoter"], columns=["promoter","epromoter"])
    sns.heatmap(dataframe, annot=True, cbar=None, fmt="d",cmap="Blues")
    if ansR == 1:
        plt.title("Confusion Matrix"+"\n"+"Exactitud = "+str(round(exact*100,3))+"%,"+" Range "+"( +"+str(edge[0])+", -"+str(edge[1])+")%"+"\n"), plt.tight_layout()
    if ansR == 0:
        plt.title("Confusion Matrix"+"\n"+"Exactitud = "+str(round(exact*100,3))+"%"+"\n"), plt.tight_layout()
    plt.ylabel("True Class"), plt.xlabel("Predicted Class")
    plt.savefig(ip_save+"/"+name+"Matrix_confusion"+".png",bbox_inches = 'tight')

def import_coef(ip_save, name, SVM_model,features):
    fig, ax = plt.subplots()
    importances = pd.DataFrame(data={"Attribute":features,"Importance": SVM_model.coef_[0]})
    importances = importances.sort_values(by='Importance', ascending=False) 
    plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
    plt.title('Feature importances obtained from coefficients', size=20)
    plt.xticks(rotation='vertical')
    plt.savefig(ip_save+"/"+name+"Coef_model"+".png",bbox_inches = 'tight')  
    
def type_clss(string):
    if string == "0":
        return "promoter"
    if string == "1":
        return "epromoter"
    
def comp_clss(string1,string2):
    if string1 == "1" and string2 == "1":
        return "True-positive"
    if string1 == "0" and string2 == "0":
        return "True-negative"
    if string1 == "0" and string2 == "1":
        return "False-positivo"
    if string1 == "1" and string2 == "0":
        return "False-negativo"

def SaveTestSVM(opPrefix,ip, var, prec, pred, y_pred, y_test):
    op1 = open(ip+"/"+opPrefix + "SVMtestScores.dat" ,"w")
    prec = prec*100
    op1.write('Precision SVM '+ str(prec)+' %'+"\n")
    fields = ["Chrom", "Start","End","True_Class","Pred_Class","Prob_Class","Matriz_Conf"]
    for i in range(len(fields)):
        op1.write(fields[i]+"\t")
    op1.write("\n")
    for j in range(len(y_test)):
        op1.write(var[j][0]+"\t"+var[j][1]+"\t"+var[j][2]+"\t"+\
        type_clss(str(y_test[j]))+"\t"+ type_clss(str(pred[j]))+"\t"+\
        str(y_pred[j][1])+"\t"+comp_clss(str(y_test[j]),str(pred[j]))+"\n")
    op1.close()
    return

#%%
# Operaciones en funcion de la eje
def operations(score, string, axes):
    if string == 'Media':
        means = score.mean(axis = axes, skipna = True)
        return means
    if string == 'Mediana':
        medians = score.median(axis= axes, skipna = True)
        return medians
    if string == 'Desviacion tipica':
        std = score.std(ddof=0,axis = axes, skipna = True)
        return std
    if string == 'Varianza':
        var = score.var(ddof=0, axis = axes, skipna = True)
        return var
    if string == 'Coeficiente Varianza':
        cv = ss.variation(score, axis= axes,nan_policy= 'omit')
        return cv
    if string == "Error estandar":
        error_t = score.sem(axis = axes, skipna = True)
        return error_t



# Covarianza   
def mtx_cov(data, features, save_ip, name):
    data = pd.DataFrame(data) 
    matrix_cov = np.ma.cov(data, data, rowvar = False)[:len(features),:len(features)]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(len(features), len(features)))
    sns.heatmap(matrix_cov, annot= True, cbar= False, annot_kws = {"size": 11}, 
    vmin = -1, vmax = 1, center= 0, cmap = sns.diverging_palette(20, 220, n=200),
    square = True, ax = ax, xticklabels = features , yticklabels = features,
    linecolor = "white", linewidths = 0.01)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment = 'right',fontsize = 1)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 45, horizontalalignment = 'right',fontsize = 1)
    ax.tick_params(labelsize = 11)
    ax.set_title("Matriz de Covarianza")
    fig.savefig(save_ip +'/'+ name +"_Cov_matrix.png")
# Correlacion   
def corr_Mtx(save_ip, data_matx,features, name):
    data = pd.DataFrame(data_matx, columns = features) 
    matrix_corr = data.corr(method = "pearson")
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(len(features), len(features)))
    sns.heatmap(matrix_corr, annot     = True, cbar      = False, annot_kws = {"size": 11},
    vmin      = -1, vmax      = 1, center    = 0, cmap      = sns.diverging_palette(20, 220, n=200),
    square    = True, ax        = ax, linecolor = "white", linewidths = 0.01)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment = 'right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 45, horizontalalignment = 'right')
    ax.tick_params(labelsize = 11)
    ax.set_title("Matriz de Correlacion")
    fig.savefig(save_ip +'/'+ name +"_Corr_matrix.png")
 
# Excel 
def indice_excel(ip,op,wb):
    hoja = wb.active
    hoja.cell(row=1, column=1, value='Linea celular')
    for i in range(len(op)):
        hoja.cell(row=1, column=i+2, value=op[i])
    wb.save(ip+"/Data.xlsx")

def data_excel(ip,name,datas,n_row,wb):
    hoja = wb.active
    hoja.cell(row=n_row+1, column=1, value=name)
    for i in range(len(datas)):
        hoja.cell(row=n_row+1, column=i+2, value=datas[i])
    wb.save(ip+"/Data.xlsx")

#-----------------------------

def t_test(score1,score2):
    a, b = stats.ttest_ind(score1,score2,nan_policy='omit')
    return a, b

def t_test_excel(ip,t,p,op,n_row,wb):
    hoja = wb.active
    hoja.cell(row=1, column=len(op)+2, value='t estadistica')
    hoja.cell(row=1, column=len(op)+3, value='p-valor')
    hoja.cell(row=n_row+1, column=len(op)+2, value= t)
    hoja.cell(row=n_row+1, column=len(op)+3, value= p)
    wb.save(ip+"/Data.xlsx")

def correct_t_test(L_p_valor):
    reject, p_valors, Sidak, Bonf = sm.stats.multipletests(L_p_valor)
    return reject, p_valors, Sidak, Bonf 
    
def new_p_excel(ip,reject, p_valors, Sidak, Bonf,op,l_row,wb):
    hoja = wb.active
    hoja.cell(row=1, column=len(op)+4, value='rechazamos hipotesis nula')
    hoja.cell(row=1, column=len(op)+5, value='new p-valor')
    hoja.cell(row=1, column=len(op)+6, value='Correlación Sidak')
    hoja.cell(row=1, column=len(op)+7, value='Corrección Bonferroni')
    for i in range(len(l_row)):
        hoja.cell(row=l_row[i], column=len(op)+4, value= reject[i] )
        hoja.cell(row=l_row[i], column=len(op)+5, value= p_valors[i])
        hoja.cell(row=l_row[i], column=len(op)+6, value= Sidak)
        hoja.cell(row=l_row[i], column=len(op)+7, value= Bonf)
    wb.save(ip+"/Data.xlsx")
    
def save_excel(tipo, features ,result, t_sts, p_valors, opert, ip, wb):
    if tipo == "Pos":
        pos = 0
    if tipo == "Neg":
        pos = 1
    hoja = wb.active
    n = 0
    # Index
    hoja.cell(row=1, column=1, value='Caracteristicas')
    hoja.cell(row=1, column=len(opert)+2, value='t estadistica')
    hoja.cell(row=1, column=len(opert)+3, value='p-valor')
    for i in range(len(opert)):
        hoja.cell(row=1, column=i+2, value=opert[i])
        # Data   
    for i in range(0, 3*len(features), 3):
        hoja.cell(row=2+i+pos, column=1, value=features[n]+"_"+tipo)
        hoja.cell(row=2+i, column=len(opert)+2, value= str(round(t_sts[n],4)))
        hoja.cell(row=2+i, column=len(opert)+3, value= str(p_valors[n]))
        for j in range(len(result)):
            hoja.cell(row=2+i+pos, column=2+j, value= str(round(result[j][n],6)))
        n+=1    
    # Guarda lo excrito 
    wb.save(ip+"/Data.xlsx")
    
#----------------------
def ord_asc(series):
    serie_ord = series.sort_values(axis=0, ascending=True)
    serie_ord = serie_ord.reset_index(drop=True)
    return serie_ord  

# Plotea ambas graficas juntas
def plots_datas(file_name,title,means1,means2,score_t1,score_t2,save_path_fig):
    fig, ax = plt.subplots(2,1) 
    ax[0].plot(means1,color = 'tab:blue',label = file_name+" Pos "+str(round(score_t1,4)))
    ax[1].plot(means2,color = 'tab:red',label =file_name+" Neg "+str(round(score_t2,4)))
    ax[0].grid(axis = 'y', color = 'gray', linestyle = 'dashed')
    ax[1].grid(axis = 'y', color = 'gray', linestyle = 'dashed')
    ax[0].legend(loc = 'lower right')
    ax[1].legend(loc = 'lower right')
    ax[1].set_xlabel('Number of regions')
    ax[0].set_ylabel('Value')
    ax[0].set_title(file_name+" Pos-Neg "+title, loc = "left", fontdict = {'fontsize':14, 'fontweight':'bold', 'color':'tab:blue'})
    plt.show()
    if title == "":
        fig.savefig(save_path_fig +'/'+ file_name+"_"+"Pos-Neg"+title+".png") 
    if title != "":
        fig.savefig(save_path_fig +'/'+ file_name+"_"+"Pos-Neg_"+title+".png")  

# Plotea curvas ROC y PR 
    
def PR_curve(y_test, y_score_prob ,save_ip):   
# Predecimos las probabilidades
    lr_probs = y_score_prob
# Nos quedamos unicamente con las predicciones positicas
    lr_probs = lr_probs[:, 1]
    lr_probs.tolist()
# Sacamos los valores
    lr_precision, lr_recall, thresholds = precision_recall_curve(y_test, lr_probs)
    lr_auc = auc(lr_recall, lr_precision)
    ns_auc = 0.5

    # Pintamos la curva precision-sensibilidad
    fig, ax = plt.subplots()
    ax.plot(lr_recall, lr_precision, label = "SVM lineal (area = %0.2f)" % lr_auc)
    ax.plot([0, 1], [1, 0], linestyle = "--", label = "Sin entrenar(balanceado) (area = %0.2f)" % ns_auc)
    # Etiquetas de los ejes
    ax.legend()
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Sensibilidad')
    plt.show()
# Guardamos figura
    fig.savefig(save_ip +'/'+ "PR_curve.png")


def ROC_curve(y_test, y_score_prob, save_ip):
#Generamos un clasificador sin entrenar , que asignará 0 a todo
    ns_probs = [ 0 for x in range(len(y_test))]
# Predecimos las probabilidades
    lr_probs = y_score_prob
#Nos quedamos con las probabilidades de la clase positiva (la probabilidad de 1)
    lr_probs = lr_probs[:, 1]
# Calculamos el AUC
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)
# Calculamos las curvas ROC
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    # Pintamos las curvas ROC
    fig, ax = plt.subplots()
    ax.plot(ns_fpr, ns_tpr, linestyle="--",label = "Sin entrenar (area = %0.2f)" % ns_auc)
    ax.plot(lr_fpr, lr_tpr, label = "SVM lineal (area = %0.2f)" % lr_auc)
    # Etiquetas de los ejes
    ax.legend(loc = "lower right")
    ax.set_title("ROC Curve")
    ax.set_xlabel("Tasa de Falsos Positivos ")
    ax.set_ylabel("Tasa de Verdaderos Positivos")
    plt.show()
    # Guardamos figura
    fig.savefig(save_ip +'/'+ "ROC_curve.png")