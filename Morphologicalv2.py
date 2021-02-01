import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageFilter
from skimage import io
from skimage.filters.rank import median
from skimage.morphology import disk
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

root = Tk()
root.withdraw()
root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("all files",".*"),("jpg files",".jpg")))
imagen = cv2.imread(root.filename)
""" imagen[177:200,:,0] = 0
imagen[177:200,:,1] = 0
imagen[177:200,:,2] = 0
 """

hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
root.destroy()

H = hsv[:,:,0]
S = hsv[:,:,1]
V = hsv[:,:,2]
print (H)
VIe = cv2.equalizeHist(V)
hsv[:,:,2]=VIe
new=cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

hmin = 0
hmax = 30.6  #0.2
sat = 1.25 ##0.15   38.25  0.0025

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
ret, otsu = cv2.threshold(segmenta,0,255,cv2.THRESH_BINARY)

kernelCross = cv2.getStructuringElement(cv2.MORPH_CROSS,(9,9))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
#kernel9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(,5))
kerneldiamond = np.array([[0, 0, 1, 0, 0],
                          [0, 1, 1, 1, 0],
                          [1, 1, 1, 1, 1],
                          [0, 1, 1, 1, 0],
                          [0, 0, 1, 0, 0]], np.uint8)

kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))

img_dilation = cv2.dilate(otsu, kernel, iterations=4)
img_erosion = cv2.erode(img_dilation, kernel, iterations=1)


imgconv = cv2.cvtColor(img_dilation, cv2.COLOR_HSV2BGR)
imgconvGray = cv2.cvtColor(imgconv, cv2.COLOR_BGR2GRAY);
(ret, imgconvGray) = cv2.threshold(imgconvGray, 50, 255, cv2.THRESH_BINARY)



# Find image contours
contours, hierarchy = cv2.findContours(imgconvGray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Maximum blobs size
threshold_blobs_area = 2000

# Loop over all contours and fill draw white color for area smaller than threshold.
for i in range(1, len(contours)):
    index_level = int(hierarchy[0][i][1])
    if index_level <= i:
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        print(area)
        if area <= threshold_blobs_area:
            # Draw white color for small blobs
            cv2.drawContours(imgconvGray, [cnt], -1, 255, -1, 1)

#imgconvGray = cv2.erode(imgconvGray, kernel, iterations=4)
imgconvGray[185:200,:]=0
#imgconvGray = cv2.dilate(imgconvGray, kernel, iterations=2)
""" imgconvGray = cv2.erode(imgconvGray, kernelCross, iterations=8)
imgconvGray = cv2.dilate(imgconvGray, kernel1, iterations=5)
imgconvGray = cv2.erode(imgconvGray, kerneldiamond, iterations=12)
imgconvGray = cv2.dilate(imgconvGray, kernel1, iterations=6)
imgconvGray = cv2.erode(imgconvGray, kernel, iterations=4) """
imgconvGray = cv2.erode(imgconvGray, kernelCross, iterations=8)
imgconvGray = cv2.dilate(imgconvGray, kernel1, iterations=5)
imgconvGray = cv2.erode(imgconvGray, kerneldiamond, iterations=2)

im= (1/255)*(cv2.imread(root.filename))
imgconvGraynorm = ((1/255)*imgconvGray)
a=(np.multiply(im[:,:,0],imgconvGraynorm))
b=(np.multiply(im[:,:,1],imgconvGraynorm))
c=(np.multiply(im[:,:,2],imgconvGraynorm))
imgfull = im
imgfull[:,:,0]=a
imgfull[:,:,1]=b
imgfull[:,:,2]=c



cv2.imshow("Ventana de binarizada",otsu)
cv2.imshow("Ventana dilatación",img_dilation)
cv2.imshow("Ventana erosión",img_erosion)
cv2.imshow("Resultado Completo",imgfull)
cv2.imshow("mascara",imgconvGray)
cv2.imwrite("wjalbe.1.jpg",imgfull*255)

cv2.waitKey(0)