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
cv2.imwrite('homoGray0.9b2.png',img)

root.destroy()
""" cv2.imshow("Ventana de imagen seleccionada",img)
cv2.waitKey(0) """
kernel = np.ones((3,3),np.float32)/9
#kernel=np.array([[-1,-1,-1],[-1,-25,-1],[-1,-1,-1]])
#dst = cv.fastNlMeansDenoisingColored(imagen,None,10,10,7,21)

######Equalizado adaptado
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl = clahe.apply(img)
cv2.imshow("Ventana de imagen ecualizado adaptativo ",cl)
cv2.imwrite('clk.png',cl)

#####Equalizacion
equ = cv2.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side
cv2.imshow("Imagen Entrada-Imagen Histograma Ecualizado",res)
#cv2.waitKey(0)
cv2.imwrite('res.png',res)
####Suavizado
dst = cv2.filter2D(equ,-1,kernel)
#dst =signal.convolve2d(img,kernel)
#imag=np.uint8(img)
#gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)



""" dst=cv2.integral(img)
cv2.imshow("Ventanimaa de imagen selecca",dst)
 """
dst2 = cv2.filter2D(cl,-1,kernel)
cv2.imshow("Ventana Histograma ecualizado  y suavizado",equ)
cv2.imshow("Ventana Histograma ecualizado adaptativo y suavizado",dst2)
cv2.imwrite('Proyect.png',dst)
######  Histograma Grises###First
hist = cv2.calcHist([cl], [0], None, [256], [0, 256])
plt.figure(u'Histograma global')
plt.subplot(211)
plt.plot(hist)
plt.title(u'Histograma Imagen Original')
#plt.xlabel(u'Valor de pixel') 
plt.ylabel(u'Número de pixeles')

#####  Histograma Grises###First
hist2 = cv2.calcHist([equ], [0], None, [256], [0, 256])
plt.figure(u'Histograma global')
plt.subplot(212)
plt.plot(hist2)
plt.title(u'Histograma Imagen Equalizada')
#plt.xlabel(u'Valor de pixel') 
plt.ylabel(u'Número de pixeles')

plt.show()


