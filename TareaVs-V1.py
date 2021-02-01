import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import *


root = Tk()
root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files",".jpg"),("all files",".*")))
imagen = cv2.imread(root.filename)
gray=cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
cv2.imshow("Ventana de imagen NORMI",imagen)
""" image = cv2.imread('image.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('OpenCV Python - Original', image)
cv2.imshow('OpenCV Python - Gray', gray) """
hist = cv2.calcHist([imagen], [0], None, [255], [0, 255])
plt.figure(1)
plt.subplot(211)
plt.plot(hist)
""" plt.hist(imagen.ravel(),254,[0,254])
plt.show() """
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([imagen],[i],None,[255],[0,255])
    plt.subplot(212)
    plt.plot(histr,color = col)
    plt.xlim([0,255])
plt.figure(2)

#Vertical
(h, w) = imagen.shape[:2]
sumCols = []
horiz=[]
for j in range(w):
        col = imagen[0:h, j:j+1] # y1:y2, x1:x2
        sumCols.append(np.sum(col))
        plt.subplot(211)
        plt.plot(sumCols)
#horizontal

for j in range(h):
        cole = imagen[j:j+1, 0:w ] # y1:y2, x1:x2
        horiz.append(np.sum(cole))
        plt.subplot(212)
        plt.plot(horiz)


plt.figure(3)
y_axe=list (range(0,len(horiz)+1))
x_axe=list (range(0,len(horiz)+1))
for j in range(len(horiz)):

        x_axe[j]=horiz[-j]
        
        plt.plot(x_axe,y_axe)
#print (y_axe)
plt.show()

#cv2.waitKey(0)
#print (horiz)