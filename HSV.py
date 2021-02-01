import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import *
from PIL import Image

import cv2
import numpy as np
from math import pi

def BGR_TO_HSI(img):

  with np.errstate(divide='ignore', invalid='ignore'):

    bgr = np.int32(cv2.split(img))

    blue = bgr[0]
    green = bgr[1]
    red = bgr[2]

    intensity = np.divide(blue + green + red, 3)

    minimum = np.minimum(np.minimum(red, green), blue)
    saturation = 1 - 3 * np.divide(minimum, red + green + blue)

    sqrt_calc = np.sqrt(((red - green) * (red - green)) + ((red - blue) * (green - blue)))

    if (green >= blue).any():
      hue = np.arccos((1/2 * ((red-green) + (red - blue)) / sqrt_calc))
    else:
      hue = 2*pi - np.arccos((1/2 * ((red-green) + (red - blue)) / sqrt_calc))

    hue = hue*180/pi

    hsi = cv2.merge((hue, saturation, intensity))
    return hsi


#Cargar imagen del explorador de archivos

root = Tk()
root.withdraw()
root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("all files",".*"),("jpg files",".jpg")))
imagen = cv2.imread(root.filename)
hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
root.destroy()

h = np.array([])
s = np.array([])
v = np.array([])
H = hsv[:,:,0]
S = hsv[:,:,1]
V = hsv[:,:,2]
VIe = cv2.equalizeHist(V)
hsv[:,:,2]=VIe
new=cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
imghsi = imagen.convert('HSI')

##imghsi= BGR_TO_HSI(imagen)

cv2.imshow("hsi",imghsi)

cv2.imshow("Ventana de H seleccionada",new)
cv2.imshow("Ventana de S seleccionada",VIe)
cv2.imshow("Ventana de V seleccionada",imagen)
cv2.imshow("Ventana de imagen sionada",hsv)

cv2.waitKey(0)