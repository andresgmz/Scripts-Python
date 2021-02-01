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
######Highpass
kernel=np.array([[-1,-1,-1],[-1,-8,-1],[-1,-1,-1]])
dst = cv2.filter2D(equ,-1,kernel)
####Rotation and SCALE

rows,cols = equ.shape

M = cv2.getRotationMatrix2D((cols/2,rows/2),-15,1)

Rotada = cv2.warpAffine(equ,M,(cols,rows))
cv2.imshow("Ventana de imagen rotada",dst)
cv2.waitKey(0)

""" dst2 = cv2.filter2D(cl,-1,kernel)
cv2.imshow("Ventana",dst2) """
##############
