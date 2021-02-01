import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import *
from PIL import Image



#Cargar imagen del explorador de archivos

root = Tk()
root.withdraw()
root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("all files",".*"),("jpg files",".jpg")))
imagen = cv2.imread(root.filename)
hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
root.destroy()

H = hsv[:,:,0]
S = hsv[:,:,1]
V = hsv[:,:,2]
print (H)
VIe = cv2.equalizeHist(V)
hsv[:,:,2]=VIe
new=cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#mghsi = imagen.convert('HSI')
hmin = 0
hmax = 30.6  #0.2
sat = 38.25##0.15
##print (len(H))
##print (len(H[]))
""" skin=(S>sat) and (H>hmin)and (H<hmax)
i2=np.multiply(H,skin)
i2[:,:,1]=np.multiply(S,skin)
i2[:,:,2]=np.multiply(VIe,skin)
i3=cv2.cvtColor(i2, cv2.COLOR_HSV2BGR)
##imghsi= BGR_TO_HSI(imagen)
SkinS= [m for i in len(S) for j in len(H[0]) if (S[i][j] > sat) and (H[i][j]>hmin) and (H[i][j]<hmax) ]
[<the_expression> if <the_condition> else <other_expression> for <the_element> in <the_iterable>]
#cv2.imshow("hsi",imghsi)
skin = () """
#skin1= [1 if ((S[i][j] > sat) and (H[i][j]>hmin) and (H[i][j]<hmax)) else 0 ]
skin = np.zeros([len(H), len(H[0])])
for i in range(0,len(S)):
     for j in range (0,len(H[0])) : 
         if ((S[i][j] > sat) and (H[i][j]>hmin) and (H[i][j]<hmax)):
             skin[i][j] = 1
         else:
             skin[i][j] = 0

#print (skin1)
#skin = np.array(skin1)
i2=imagen
h=np.multiply(H,skin)
s=np.multiply(S,skin)
v=np.multiply(VIe,skin)
i2[:,:,0]=h
i2[:,:,1]=s
i2[:,:,2]=v
segmenta=cv2.cvtColor(i2, cv2.COLOR_HSV2BGR)
#ret1,th1 = cv2.threshold(segmenta,60,155,cv2.THRESH_BINARY)
ret, otsu = cv2.threshold(segmenta,0,255,cv2.THRESH_BINARY)

kernel = np.ones((5,5), np.uint8)
kernel1 = np.ones((9,9), np.uint8)

#img_erosion = cv2.erode(img, kernel, iterations=1)
img_dilation = cv2.dilate(otsu, kernel, iterations=2)
closing = cv2.morphologyEx(img_dilation, cv2.MORPH_CLOSE, kernel1)
img_erosion = cv2.erode(closing, kernel, iterations=1)
ret,thresh = cv2.threshold(img_erosion,127,255,0)
#im2,contours= cv2.findContours(thresh, 1)
#cnt = contours[0]
#M = cv2.contourArea(cnt)
#cv2.imshow("Ventana de imagen seleccionada",M)
cv2.imshow("Ventana de segmentación",segmenta)
cv2.imshow("Ventana de binarizada",otsu)
cv2.imshow("Ventana dilatación",img_dilation)
cv2.imshow("Ventana erosión",img_erosion)
cv2.imshow("Ventana filling",closing)
cv2.imshow("Ventana de imagen hsv",hsv)

cv2.waitKey(0)