import cv2
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import pyplot as plt
from tkinter import filedialog
from tkinter import *

root = Tk()
root.withdraw()
root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("all files",".*"),("jpg files",".jpg")))
img = cv2.imread(root.filename)
root.destroy()

# Convert to gray-scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image to reduce noise
img_blur = cv2.medianBlur(gray, 5)
# Apply hough transform on the image8 $$$img.shape[0]/16, param1=100, param2=11, minRadius=62, maxRadius=67
# Draw detected circles; circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img.shape[0]/16, param1=200, param2=25, minRadius=60, maxRadius=67)
face_cascade = cv2.CascadeClassifier('C:/Users/andre/Desktop/NovenoSemestre/VisionArtificial/Python/haarcascade_frontalface_alt.xml')


gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    center = (x + w//2, y + h//2)
    
#circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img.shape[0]/128, param1=100, param2=11, minRadius=50, maxRadius=100)
circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img.shape[0]/128, param1=100, param2=11, minRadius=(w//2-10), maxRadius=(w//2+10))


(h, w) = img_blur.shape[:2] #Calcular tamaño de la imageb
(pointRefX,pointRefY) = center
puntoMinimo =100
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:

        #Definir el circulo mas cercano de la 
         
        xCercano =np.absolute(i[0]-pointRefX) 
        yCercano =np.absolute(i[1]-pointRefY)
        puntoCercano = xCercano+yCercano
        if (puntoCercano < puntoMinimo):
            puntoMinimo = puntoCercano
            circuloCercano = i
        # Draw outer circle
#frame = cv2.ellipse(img, center, (w//2, h//2), 0, 0, 360,(100, 7, 55), 2)
cv2.ellipse(img, (circuloCercano[0], circuloCercano[1]),(circuloCercano[2],circuloCercano[2]+15),0,0,360,(0, 255, 0), 2)
# Draw inner circle
cv2.circle(img, (circuloCercano[0], circuloCercano[1]), circuloCercano[2], (0, 255, 0), 2)
cv2.circle(img, (circuloCercano[0], circuloCercano[1]), 2, (0, 0, 255), 3)
""" cv2.circle(img, (circuloCercano[0], circuloCercano[1]), circuloCercano[2], (0, 255, 0), 2)
# Draw inner circle
cv2.circle(img, (circuloCercano[0], circuloCercano[1]), 2, (0, 0, 255), 3) """

    



""" if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:

        #Definir el circulo mas cercano de la 
         
        xCercano =np.absolute(i[0]-pointRefX) 
        yCercano =np.absolute(i[1]-pointRefY)
        puntoCercano = xCercano+yCercano
        if (puntoCercano < puntoMinimo):
            puntoMinimo = puntoCercano
            circuloCercano = i
        
           
        # Draw outer circle
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw inner circle
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
 """
cv2.imshow("Mascara",img)

cv2.waitKey(0)