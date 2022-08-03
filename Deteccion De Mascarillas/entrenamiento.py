import cv2
import os
import numpy as np

ruta="./"
directorios=os.listdir(ruta)
##print("directorios", directorios)

etiqueta=[]
caras=[]
posicion=0
for dir in directorios:
    if str(dir)=="Con_mascarilla" or str(dir)=="Sin_mascarilla" : 
        ruta_data=ruta+dir

        for archivo in os.listdir(ruta_data):
            imagen_ruta=ruta_data+"/"+archivo
            #print(imagen_ruta)
            image=cv2.imread(imagen_ruta,0)
            #cv2.imshow("Image",image)
            #cv2.waitKey(10)
            caras.append(image)
            etiqueta.append(posicion)
        posicion +=1

#print("Etiqueta 0:",np.count_nonzero(np.array(etiqueta)==0))
#print("Etiqueta 1:",np.count_nonzero(np.array(etiqueta)==1))

mascarilla=cv2.face.LBPHFaceRecognizer_create()
print("empieza entrenamiento")
mascarilla.train(caras,np.array(etiqueta))

mascarilla.write("mascarilla_model.xml")
print("modelo terminado")