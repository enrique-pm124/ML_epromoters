# ML_epromoters

Esta herramienta está compuesta por tres programas con diferentes funciones: 

1. Analizar un conjunto de datos creados a partir de los diferentes estudios genómicos y regiones que se proporcionen. 
2. Programa para crear modelos de aprendizaje automático para clasificar entre dos clases de regiones a partir de los diferentes estudios genómicos que se proporcionen. 
3. Modificación del modelo de Matched Filter que permiten crear un modelo con nuevas características basadas en los estudios genómicos que se proporcionen. 

Esta herramienta ha sido creada utilizando la versión 2.7 de Python, haciendo uso de varias librerias de Python, herramientas proporcionadas por UNAM como bigwigtoBedGraph y la herramientas de Deeptools. 

Hay 4 script dentro de esta herramienta: 

1. Analisis.py crea tablas y figuras que contienen los valores las diferentes pruebas realizadas a cada estudio genómico utilizado como característica predictiva, diferenciando entre los dos tipos de regiones utilizada. Para ejecutar el script:

python Analisis.py 

Una vez ejecutado, pedirá introducir estos argumentos: 

<FilePosBw.txt> archivo txt con datos y direcciones de archivos sobre estudios genómico de anotaciones en formato BigWig y regiones de estudio de clase positiva.

<FileNegBw.txt> archivo txt con datos y direcciones de archivos sobre estudios genómico de anotaciones en formato BigWig y regiones de estudio de clase negativa.

<FilePosBed.txt> archivo txt con datos y direcciones de archivos sobre estudios genómico de anotaciones en formato BedGraph y regiones de estudio de clase positiva.

<FileNegBed.txt> archivo txt con datos y direcciones de archivos sobre estudios genómico de anotaciones en formato BedGraph y regiones de estudio de clase negativa.

La forma de estructura de los archivos de datos y direcciones se encuentra en el archivo FormatoTxt.

2. SVM.py crea un modelo de aprendizaje automático de Support Vector Machine para predecir entre dos clases de regiones, utilizando como datos diferentes estudios genómico de anotaciones y de regiones que se proporcionaran al programa. Este produce como respuesta la media de las matrices de confusión del test y entrenamiento obtenidas de un remuestreo, junto con su exactitud y su rango de variación. Además, como caso concreto, se produce la matriz de confusión del entrenamiento y test del conjunto de datos más estable, la exactitud de cada matriz y las curvas ROC y PR. Para ejecutar el script:

python SVM.py 

Una vez ejecutado, pedirá introducir estos argumentos: 

<FilePosBw.txt> archivo txt con datos y direcciones de archivos sobre estudios genómico de anotaciones en formato BigWig y regiones de estudio de clase positiva.

<FileNegBw.txt> archivo txt con datos y direcciones de archivos sobre estudios genómico de anotaciones en formato BigWig y regiones de estudio de clase negativa.

<FilePosBed.txt> archivo txt con datos y direcciones de archivos sobre estudios genómico de anotaciones en formato BedGraph y regiones de estudio de clase positiva.

<FileNegBed.txt> archivo txt con datos y direcciones de archivos sobre estudios genómico de anotaciones en formato BedGraph y regiones de estudio de clase negativa.

La forma de estructura de los archivos de datos y direcciones se encuentra en el archivo FormatoTxt.

3. MF_modif.py crea un modelo, basado en el programa de Matched-Filter, con la capacidad de predecir entre regiones promotoras y de doble función a partir de los datos originales junto con nuevos estudios genómicos de anotación. Este produce las matrices de confusión de los subconjuntos de entrenamiento y test junto con su exactitud y las curvas ROC y PR. Para ejecutar el script:

python MF_modif.py 

Una vez ejecutado, pedirá introducir estos argumentos: 

<FilePos.txt> archivo txt con las regiones y datos originales de la clase positiva del programa Matched-Filted.

<FileNeg.txt> archivo txt con las regiones y datos originales de la clase negativa del programa Matched-Filted.

<FilePosBed.txt> archivo txt con datos y direcciones de los archivos sobre estudios genómico de anotaciones en formato BedGraph y regiones de estudio de clase positiva.

<FileNegBed.txt> archivo txt con datos y direcciones de los archivos sobre estudios genómico de anotaciones en formato BedGraph y regiones de estudio de clase negativa.

La forma de estructura de los archivos de datos y direcciones se encuentra en el archivo FormatoTxt.

4. myFunctions.py archivo con las funciones utilizadas en los tres programas anteriores.




