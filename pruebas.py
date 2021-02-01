import numpy as np
import cv2
from tkinter import filedialog
from tkinter import *
import os

#Cargar imagen del explorador de archivos

root = Tk()
root.withdraw()

root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("all files",".*"),("jpg files",".jpg")))
root.destroy()
img = cv2.imread(root.filename)
noise = np.random.randint(5, size = (164, 278, 4), dtype = 'uint8')

(HEIGHT,WIDTH,DEPTH)=img.shape
cv2.imshow("Venda",img)
for i in range(HEIGHT):
            for j in range(WIDTH):
                for k in range(DEPTH):
                    if (img[i][j][k] != 255):
                        img[i][j][k] += noise[i][j][k]
""" for j in range(5,1,-1):
    w = str(j)
    print (w) """
""" kernel=np.array([[-1,-1,-1],[-1,-8,-1],[-1,-1,-1]])
p= (255)*kernel
p1= "Hola"
hello ="2"
jpg = "jpg"
ex= p1+hello+jpg
print (ex) """
#WIDTH, HEIGHT,ch = img.shape
#print (img.shape)

""" for i in range(HEIGHT, 1, -1):
            for j in range(WIDTH):
                if (i < HEIGHT-10):
                  img[j][i] = img[j][i-10]
                elif (i < HEIGHT-1):
                  img[j][i] = 0
 """

cv2.imshow("Ventana de im seleccionada",img)
cv2.waitKey(0)