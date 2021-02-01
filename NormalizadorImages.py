import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import *
import os

#Cargar imagen del explorador de archivos

root = Tk()
root.withdraw()

#root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("all files",".*"),("jpg files",".jpg")))
#root.destroy()
path = "C:/Users/andre/Desktop/DataSetRNA/ImagenesSinProcesamiento/harry/harry."
pathjpg = ".jpg"
pathdestino ="C:/Users/andre/Desktop/DataSetRNA/ImagenesNormalizadas/Harry"

for i in range(1,21):
    pathcounter = str(i)
    pathfull = path+pathcounter+pathjpg
    imagen = cv2.imread(pathfull,0)
    ancho=64
    alto =64
    dim =(ancho,alto)
    
    equ = cv2.equalizeHist(imagen)
    equ = cv2.resize(equ,dim,interpolation = cv2.INTER_AREA)
    #cv2.imshow("Ventana de im seleccionada",equ)
    normalizada = (1/255)*equ
    full = "Harry"+pathcounter+pathjpg
    print(full)
    cv2.imwrite(os.path.join(pathdestino , full), normalizada)
    

   # print (j)
#noise = np.random.randint(5, size = (164, 278), dtype = 'uint8')

a =21

for omega in range (1,7):
    pathcounter = str(omega)
    pathfull = path+pathcounter+pathjpg
    photo = cv2.imread(pathfull,0)
    img = cv2.equalizeHist(photo)
    #cv2.imshow("Ventana de im seleccionada",equ)
    normalizada = (1/255)*equ
    (HEIGHT, WIDTH) = img.shape[:2] 
    #full = "ferbN"+pathcounter+pathjpg
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(HEIGHT,WIDTH))
    noise = gauss.reshape(HEIGHT,WIDTH)
    if omega == 1:
      flipped_img = np.fliplr(img)
    elif omega == 2:#shiftleft
        for i in range(WIDTH, 1, -1):
            for j in range(HEIGHT):
                if (i < WIDTH-10):
                  img[j][i] = img[j][i-10]
                elif (i < WIDTH-1):
                  img[j][i] = 0
    elif omega == 3:##Shiftright
        for j in range(HEIGHT):
            for i in range(WIDTH):
                if (i < WIDTH-20):
                 img[j][i] = img[j][i+20]
    elif omega == 4:
        # Shifting Up
        for j in range(HEIGHT):
            for i in range(WIDTH):
                if (j < HEIGHT - 20 and j > 20):
                 img[j][i] = img[j+20][i]
                else:
                 img[j][i] = 0

    elif omega==5:
     #Shifting Down
         M = cv2.getRotationMatrix2D((WIDTH/2,HEIGHT/2),-5,1)

         img = cv2.warpAffine(img,M,(WIDTH,HEIGHT))
    elif omega==6:
        #DEPTH = 700
        for i in range(HEIGHT):
            for j in range(WIDTH):
                
                    if (img[i][j]!= 255):
                        img[i][j]+= 3
    for w in range (1,6):
         newcount = str (a)
         completo = "Harry"+newcount+pathjpg
         
         img = cv2.resize(img,dim,interpolation = cv2.INTER_AREA)
         norm = (1/255)*img
         cv2.imwrite(os.path.join(pathdestino , completo), norm)
         a = a+1 
#print (normalizada[34,68])
#print (root.filename)



#Mostrar imagen cargada

#cv2.imshow("Ventana de imagen seleccionada",imagen)

#Calcular histograma de la imagen en escala de grises
""" equ = cv2.equalizeHist(imagen)
#cv2.imshow("Ventana de im seleccionada",equ)
normalizada = (1/255)*equ
#cv2.imshow = ("Hello",normalizada)
cv2.imshow("Ventana de im seleccionada",normalizada)
#print(normalizada[5,7])
cv2.waitKey(0) """
""" hist = cv2.calcHist([imagen], [0], None, [256], [0, 256])
plt.figure(u'Histograma global')
#plt.subplot(211)
plt.plot(hist)
plt.title(u'Histograma escala de grises')
#plt.xlabel(u'Valor de pixel') 
plt.ylabel(u'Número de pixeles') """
###Ecualizacion
#path= 
#cv2.imwrite('Proyect.png',dst)
#plt.show()
