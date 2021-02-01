import cv2
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import pyplot as plt
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageFilter


def cv2pil(cv_im):
    # Convert the cv image to a PIL image
    return Image.fromstring("L", cv2.GetSize(cv_im), cv_im.tostring())

def imgCrop(image, cropBox, boxScale=1):
    # Crop a PIL image with the provided box [x(left), y(upper), w(width), h(height)]

    # Calculate scale factors
    xDelta=max(cropBox[2]*(boxScale-1),0)
    yDelta=max(cropBox[3]*(boxScale-1),0)

    # Convert cv box to PIL box [left, upper, right, lower]
    PIL_box=[cropBox[0]-xDelta, cropBox[1]-yDelta, cropBox[0]+cropBox[2]+xDelta, cropBox[1]+cropBox[3]+yDelta]

    return image.crop(PIL_box)

def pil2cvGrey(pil_im):
    # Convert a PIL image to a greyscale cv image
    # from: http://pythonpath.wordpress.com/2012/05/08/pil-to-opencv-image/
    pil_im = pil_im.convert('L')
    cv_im = cv2.CreateImageHeader(pil_im.size, cv2.IPL_DEPTH_8U, 1)
    cv2.SetData(cv_im, pil_im.tostring(), pil_im.size[0]  )
    return cv_im

root = Tk()
root.withdraw()
root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("all files",".*"),("jpg files",".jpg")))
img = cv2.imread(root.filename)
original=cv2.imread(root.filename)
root.destroy()

face_cascade = cv2.CascadeClassifier('C:/Users/andre/Desktop/NovenoSemestre/VisionArtificial/Python/haarcascade_frontalface_alt.xml')


gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
new=img
for (x,y,w,h) in faces:
    center = (x + w//2, y + h//2)
    frame = cv2.ellipse(img, center, (w//2, h//2), 0, 0, 360,(100, 7, 55), 2)
    #frame = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    cropped=img[y:y+h,x:x+w]
#for i in range(img.shape[0]):
  #  for j in range(img.shape[1]):
 #       new[y:y+i+h+5-i,x:x+j+h+5-j]=0
        #new[y:y+h-i,x:x+h-i]=0
        #x=new
   #new=img-cropped
#final=original-new

#cv2.imwrite("face.jpg", final)

cv2.imshow('img',img)
cv2.imshow('dimg',cropped)
#cv2.imshow('fin',new)
#cv2.imshow('x',final)
#cv2.imwrite('segmecntc.jpg',final)
cv2.imshow('or',original)
cv2.waitKey(0)
cv2.destroyAllWindows()