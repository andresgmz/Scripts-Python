import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from tkinter import filedialog
from tkinter import *
import matplotlib.pyplot as plt
from skimage import data, util
from skimage.draw import ellipse
from skimage.measure import label, regionprops
from skimage.transform import rotate
import matplotlib.pyplot as plt
from scipy.stats import skew 
from scipy.stats import kurtosis
import math
root = Tk()
root.withdraw()
root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("all files",".*"),("jpg files",".jpg")))
#img = cv.imread(root.filename,0)
root.destroy()
# Write Python code here 
# import the necessary packages 
import cv2 
import argparse 
  
# now let's initialize the list of reference point 
ref_point = [] 
crop = False
  
def shape_selection(event, x, y, flags, param): 
    # grab references to the global variables 
    global ref_point, crop 
  
    # if the left mouse button was clicked, record the starting 
    # (x, y) coordinates and indicate that cropping is being performed 
    if event == cv2.EVENT_LBUTTONDOWN: 
        ref_point = [(x, y)] 
  
    # check to see if the left mouse button was released 
    elif event == cv2.EVENT_LBUTTONUP:
       # center = (x + w//2, y + h//2)

       # cv2.ellipse(img, center, (w//2, h//2), 0, 0, 360,(100, 7, 55), 2)
        # record the ending (x, y) coordinates and indicate that 
        # the cropping operation is finished 
        ref_point.append((x, y)) 
        #frame = cv2.ellipse(image, center, (w//2, h//2), 0, 0, 360,(100, 7, 55), 2)
        Punto_antes = ref_point[0]
        Punto_actual = ref_point[1]
        


        center = (Punto_antes[0]+(Punto_actual[0]-Punto_antes[0])//2,Punto_antes[1]+(Punto_actual[1]-Punto_antes[1])//2)
        #frame = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
        cv2.ellipse(image, center, ((Punto_actual[0]-Punto_antes[0])//2,(Punto_actual[1]-Punto_antes[1])//2), 0, 0, 360,(100, 7, 55), 2)
        # draw a rectangle around the region of interest 
        #cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2) 
        cv2.imshow("image", image) 

    
# load the image, clone it, and setup the mouse callback function 
image = cv2.imread(root.filename) 
clone = image.copy() 
cv2.namedWindow("image") 
cv2.setMouseCallback("image", shape_selection) 
  
  
# keep looping until the 'q' key is pressed 
while True: 
    # display the image and wait for a keypress 
    cv2.imshow("image", image) 
    key = cv2.waitKey(1) & 0xFF
  
    # press 'r' to reset the window 
    if key == ord("r"): 
        image = clone.copy() 
  
    # if the 'c' key is pressed, break from the loop 
    elif key == ord("c"): 
        break
  
if len(ref_point) == 2: 
    crop_img2 = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]: 
                                                           ref_point[1][0]]  
    height = image.shape[0]
    width = image.shape[1]
    Punto_antes = ref_point[0]
    Punto_actual = ref_point[1]
    center = (Punto_antes[0]+(Punto_actual[0]-Punto_antes[0])//2,Punto_antes[1]+(Punto_actual[1]-Punto_antes[1])//2)

    # create a mask image of the same shape as input image, filled with 0s (black color)
    mask = np.zeros_like(clone)
    rows, cols,_ = mask.shape
    # create a white filled ellipse
    mask = cv2.ellipse(mask, center, ((Punto_actual[0]-Punto_antes[0])//2,(Punto_actual[1]-Punto_antes[1])//2), 0, 0, 360,(255, 255, 255), -1)
    
    # Bitwise AND operation to black out regions outside the mask
    crop_img = np.bitwise_and(clone,mask)
    # Convert from BGR to RGB for displaying correctly in matplotlib
    # Note that you needn't do this for displaying using OpenCV's imshow()
    
    (h, w) = crop_img2.shape[:2] #Calcular tamaño de la imageb
    sumCols = []            
    sumFils=[]
    plt.figure(u'Histograma horizontal y vertical')

    newarray=np.array(crop_img2.flatten())
    meanArr=np.mean(newarray)
    varArr=np.var(newarray)
    skeArr=skew(newarray)
    kurArr=kurtosis(newarray)
    #Histograma vertical

    for j in range(w):
            col = crop_img2[10:h, j:j+1] # y1:y2, x1:x2
            sumCols.append(np.sum(col))
            plt.subplot(211)
            plt.plot(sumCols)
            plt.title(u'Histogramas vertical y horizontal')
            plt.xlabel(u'Número de columnas') 
            plt.ylabel(u'Nivel de intensidad')

    #Histograma horizontal

    for j in range(h):
            cole = crop_img2[j:j+1, 10:w ] # y1:y2, x1:x2
            sumFils.append(np.sum(cole))
            plt.subplot(212)
            plt.plot(sumFils)
            #plt.title(u'Histograma horizontal')
            plt.xlabel(u'Número de filas') 
            plt.ylabel(u'Nivel de intensidad')

    meanH=np.mean(sumFils)
    varH= np.var(sumFils)
    stdH = np.std(sumFils)
    skeH=skew(sumFils)
    kurH=kurtosis(sumFils)

    meanV=np.mean(sumCols)
    varV= np.var(sumCols)
    stdV = np.std(sumCols)
    skeV=skew(sumCols)
    kurV=kurtosis(sumCols)

    kernelCross = cv2.getStructuringElement(cv2.MORPH_CROSS,(9,9))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    #kernel9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(,5))
    kerneldiamond = np.array([[0, 0, 1, 0, 0],
                            [0, 1, 1, 1, 0],
                            [1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 0],
                            [0, 0, 1, 0, 0]], np.uint8)

    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))

    # close all open windows
    crop_img =cv2.cvtColor(crop_img,cv2.COLOR_RGB2GRAY) 
    ret,thresh = cv.threshold(crop_img,130,255,cv.THRESH_BINARY_INV)
    thresh = cv2.erode(thresh, kernel, iterations=6)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours2, hierarchy = cv2.findContours(255-thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]
    cnt2=contours2[0]
    M = cv.moments(cnt2)
    
    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    # put text and highlight the center
    cv.circle(crop_img, (cX, cY), 3, (0, 0, 155), -1)
    cv.putText(crop_img, "C", (cX - 1, cY - 10),cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 155), 2)
    #MAx, MAy = int(0.5 * ellipseMajorAxisx*math.sin(ellipseAngle)), int(0.5 * ellipseMajorAxisy*math.cos(ellipseAngle))
    altura=crop_img2.shape[0]
    ancho=crop_img2.shape[1]
    #area = altura*ancho
    #area = cv.contourArea(cnt)
    (x,y),(MA,ma),angle = cv.fitEllipse(cnt)
    (x2,y2),(MA2,ma2),angle2 = cv.fitEllipse(cnt2)
    area=(int(MA2)//2)*(int(ma2)//2)*(math.pi)

    xMinor=cX + int((MA/2)*np.cos(angle))
    x2Minor=cX - int((MA/2)*np.cos(angle))
    yMinor=cY + int((MA/2)*np.sin(angle))
    y2Minor=cY - int((MA/2)*np.sin(angle))

    xMajor=cX + int((ma/2)*np.sin(angle))
    x2Major=cX - int((ma/2)*np.sin(angle))
    yMajor=cY -int((ma/2)*np.cos(angle))
    y2Major=cY +int((ma/2)*np.cos(angle))
    cv.line(crop_img, (xMinor, yMinor),(x2Minor,y2Minor), (0, 0, 155), 1)
    cv.line(crop_img, (xMajor, yMajor),(x2Major,y2Major), (0, 0, 155), 1)
    #cv.line(im, (xMinor, yMinor),(cX, cY), (0, 255, 0), 1)
    #cv.line(im, (xMajor, yMajor),(cX, cY), (0, 255,0), 1)
    #cv.putText(crop_img, "A: "+str(int(area)), (cX - int(0.42*height), cY + int(0.42*width)),cv.FONT_HERSHEY_SIMPLEX, 0.42, (255, 55, 0),1)
    cv.putText(crop_img, "D.Ma: "+str(int(ma2)), (int(0.1*height), cY-int(0.4*width)),cv.FONT_HERSHEY_SIMPLEX, 0.42, (255, 55, 0),1)
    cv.putText(crop_img, "D.Me: "+str(int(MA2)), (int(0.1*height),cY - int(0.5*width)),cv.FONT_HERSHEY_SIMPLEX, 0.42, (255, 55, 0),1 )
   
    print("LOS VALORES OBTENIDOS SON :")
    print("El área es "+str(area))
    print("La media es: " + str(meanArr) +" Varianza  " + str(varArr) +" Oblicuidad  " + str(skeArr) +" Kurtosis "+ str(kurArr))
    print("MEDIA HORIZONTAL " +str(meanH) + " Vrianza horizonral "+ str(varH) + " Desviación horizontal " + str(stdH))
    print("MEDIA VERTICAL " + str(meanV) +" Vrianza VERTICAL " + str(varV) +" Desviación VERTICAL " + str(stdV))
    print("Oblicuidad HORIZONTAL " +str(skeH) + " Oblicuidad VERTICAL "+ str(skeV) )
    print("kurtosis HORIZONTAL " +str(kurH) + " kurtosis VERTICAL "+ str(kurV) )
    cv.drawContours(crop_img,contours,-1,255,2)
    #cv.drawContours(im,[cnt],0,(255,0,0),-1)
    cv2.imshow('Fitting an Ellipse  ',crop_img)
    cv2.imshow('Fitting an   ',255-thresh)
    #cv2.imshow("cropeada",crop_img2)
    plt.show()
    cv2.waitKey(0) 
   
# close all open windows 
cv2.destroyAllWindows() 