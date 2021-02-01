import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import *


#Cargar imagen del explorador de archivos

root = Tk()
root.withdraw()
root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("all files",".*"),("jpg files",".jpg")))
imagen = cv2.imread(root.filename)
root.destroy()

#Mostrar imagen cargada

cv2.imshow("Ventana de imagen seleccionada",imagen)

#Calcular histograma de la imagen en escala de grises

hist = cv2.calcHist([imagen], [0], None, [256], [0, 256])
plt.figure(u'Histograma global')
plt.subplot(211)
plt.plot(hist)
plt.title(u'Histograma escala de grises')
#plt.xlabel(u'Valor de pixel') 
plt.ylabel(u'Número de pixeles')

#Calcular histograma de la imagen a color

color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([imagen],[i],None,[256],[0,256])
    plt.subplot(212)
    plt.plot(histr,color = col)
    plt.xlim([0,256])
    plt.title(u'Histograma en RGB')
    plt.xlabel(u'Valor de pixel') 
    plt.ylabel(u'Número de pixeles')

#Calcular histograma horizontal y vertical

(h, w) = imagen.shape[:2] #Calcular tamaño de la imageb
sumCols = []            
sumFils=[]
plt.figure(u'Histograma horizontal y vertical')

#Histograma vertical

for j in range(w):
        col = imagen[0:h, j:j+1] # y1:y2, x1:x2
        sumCols.append(np.sum(col))
        plt.subplot(211)
        plt.plot(sumCols)
        plt.title(u'Histogramas vertical y horizontal')
        plt.xlabel(u'Número de columnas') 
        plt.ylabel(u'Nivel de intensidad')

#Histograma horizontal

for j in range(h):
        cole = imagen[j:j+1, 0:w ] # y1:y2, x1:x2
        sumFils.append(np.sum(cole))
        plt.subplot(212)
        plt.plot(sumFils)
        #plt.title(u'Histograma horizontal')
        plt.xlabel(u'Número de filas') 
        plt.ylabel(u'Nivel de intensidad')

plt.show()