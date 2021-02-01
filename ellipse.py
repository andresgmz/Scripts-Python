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

face_cascade = cv2.CascadeClassifier('C:/Users/andre/Desktop/NovenoSemestre/VisionArtificial/Python/haarcascade_frontalface_alt.xml')

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
    center = (x + w//2, y + h//2)
    frame = cv2.ellipse(img, center, (w//2, h//2), 0, 0, 360,(100, 7, 55), 2)
    #frame = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

 
print (faces.shape)
print (faces)
cv2.imshow('img',img)
cv2.imshow('dimg',roi_color)
cv2.imshow('dixd',roi_gray)
cv2.imwrite('cuadrad2.jpg',frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

