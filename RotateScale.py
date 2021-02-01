import cv2
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import pyplot as plt
from tkinter import filedialog
from tkinter import *


root = Tk()
root.withdraw()
root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("all files",".*"),("jpg files",".jpg")))
img = cv2.imread(root.filename,0)
root.destroy()
""" cv2.imshow("Ventana de imagen seleccionada",img)
cv2.waitKey(0) """

######Equalizado adaptado
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl = clahe.apply(img)
cv2.imshow("Ventana de imagen ",cl)
cv2.imwrite('clk.png',cl)


#####Equalizacion
equ = cv2.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side
cv2.imshow("Ventana de imagen seleccionada",res)

####Rotation and SCALE

rows,cols = equ.shape

M = cv2.getRotationMatrix2D((cols/2,rows/2),-15,1)

Rotada = cv2.warpAffine(equ,M,(cols,rows))
cv2.imshow("Ventana de imagen rotada",Rotada)
""" dst2 = cv2.filter2D(cl,-1,kernel)
cv2.imshow("Ventana",dst2) """
##############


#Calcular histograma horizontal y vertical

(h, w) = Rotada.shape[:2] #Calcular tamaño de la imageb
sumCols = []            
sumFils=[]
plt.figure(u'Histograma horizontal y vertical')

#Histograma vertical

for j in range(w):
        col = Rotada[0:h, j:j+1] # y1:y2, x1:x2
        sumCols.append(np.sum(col))
        plt.subplot(211)
        plt.plot(sumCols)
        plt.title(u'Histogramas vertical y horizontal')
        plt.xlabel(u'Número de columnas') 
        plt.ylabel(u'Nivel de intensidad')

#Histograma horizontal

for j in range(h):
        cole = Rotada[j:j+1, 0:w ] # y1:y2, x1:x2
        sumFils.append(np.sum(cole))
        plt.subplot(212)
        plt.plot(sumFils)
        #plt.title(u'Histograma horizontal')
        plt.xlabel(u'Número de filas') 
        plt.ylabel(u'Nivel de intensidad')

#Calcular histograma horizontal y vertical

(h, w) = equ.shape[:2] #Calcular tamaño de la imageb
sumCols = []            
sumFils=[]
plt.figure(u'Histograma horizontal y vertical Norotada')

#Histograma vertical

for j in range(w):
        col = equ[0:h, j:j+1] # y1:y2, x1:x2
        sumCols.append(np.sum(col))
        plt.subplot(211)
        plt.plot(sumCols)
        plt.title(u'Histogramas vertical y horizontal')
        plt.xlabel(u'Número de columnas') 
        plt.ylabel(u'Nivel de intensidad')

#Histograma horizontal

for j in range(h):
        cole = equ[j:j+1, 0:w ] # y1:y2, x1:x2
        sumFils.append(np.sum(cole))
        plt.subplot(212)
        plt.plot(sumFils)
        #plt.title(u'Histograma horizontal')
        plt.xlabel(u'Número de filas') 
        plt.ylabel(u'Nivel de intensidad')
plt.show()