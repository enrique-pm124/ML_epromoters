# ML_epromoters

Esta herramienta esta compuesta por tres programas con diferentes funciones: 

1. Analizar un conjunto de datos creados a partir de los diferentes estudios quenomicos y regiones que se proporcionen. 
2. Programa para crear modelos de aprendizaje automático para clasificar entre dos clases de regiones a partir de los diferentes estuidos genomicos que se proporcionen. 
3. Modificacion del modelo de Matched Filter que permiter crear un modelo con nuevas caracteristicas basadas en los estuidos genomicos que se proporcionen. 

Esta herramienta ha sido creada utilizando la version 2.7 de Python.

Hay 4 script dentro de esta herramienta: 

1. Analisis.py crea tablas y figuras que contienen los valores las diferentes pruebas realizadas a cada estudio genomico utilizado como caracteristica predictiva, diferenciando entre los dos tipos de regiones utilizada. Para ejecutar el script:

python Analisis.py 

Una vez ejecutado, pedira introducir estos argumentos: 
<FilePosBw.txt> 
<FileNegBw.txt> 
<FilePosBed.txt> 
<FileNegBed.txt>



2. SVM.py crea un modelo de aprendizaje automatico de Support Vector Machine para predecir entre dos clases de regiones, utilizando como datos diferentes estudios genomico de anotaciones y de regiones que se proporcionaran al programa. Este produce como respuesta la media de las matrices de confusion de test y entrenamiento obtenidas de un remuestreo, junto con su exactitud y su rango de variacion. Ademas, como caso concreto, se produce la matriz de confusion del entrenamiento y test del conjunto de datos más estable, la exactitud de cada matriz y las curvas ROC y PR. Para ejecutar el script:

python SVM.py 

3. MF_modif.py crea un modelo, basado en el programa de Matched-Filter, con la capacidad de predecir entre regiones promotoras y de doble funcion a partir de los datos originales junto con nuevos estudios genomicos de anotación. Este producira las matrices de confusion de los subconjuntos de entrenamiento y test junto con su exactitud y las curvas ROC y PR. Para ejecutar el script:

python MF_modif.py 

4. myFunctions.py archivo con las funciones utilizadas en los tres programas anteriores.




