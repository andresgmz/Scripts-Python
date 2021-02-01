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

"""
    Requires:
    - Python 3
    - pip install opencv-python
"""

""" import cv2

SRC_PATH = "images/remove_small_blobs"
TEMP_PATH = SRC_PATH


def remove_small_blob(file_path):
    # Read image with OpenCv
    img = cv2.imread(file_path)
    cv2.imshow("origin", img)

    # Convert image to Gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Gray", img)

    # Apply threshold to make image black and white
    ret, img = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)
    cv2.imshow("Black-white", img)

    # Find image contours
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Maximum blobs size
    threshold_blobs_area = 30

    # Loop over all contours and fill draw white color for area smaller than threshold.
    for i in range(1, len(contours)):
        index_level = int(hierarchy[0][i][1])
        if index_level <= i:
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            print(area)
            if area <= threshold_blobs_area:
                # Draw white color for small blobs
                cv2.drawContours(img, [cnt], -1, 255, -1, 1)

    cv2.imshow("result", img)
    cv2.waitKey(0)

print("===== Start -------")
remove_small_blob(SRC_PATH + "blob1.jpg")
print("Done") """
####
#Cargar imagen del explorador de archivos
""" def cv2pil(cv_im):
    # Convert the cv image to a PIL image
    return Image.fromstring("L", cv2.GetSize(cv_im), cv_im.tostring()) """

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
sat = 5.25 ##0.15   38.25  0.0025
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
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
""" kernel = np.ones((5,5), np.uint8)
kernel1 = np.ones((9,9), np.uint8) """

#img_erosion = cv2.erode(img, kernel, iterat2ions=1)
img_dilation = cv2.dilate(otsu, kernel, iterations=4)
img_erosion = cv2.erode(img_dilation, kernel, iterations=1)



#img_erosion3 = cv2.erode(closing2, kernel, iterations=6)
#img_erosion2= cv2.erode(img_dilation2, kernel, iterations=4)

#ret,thresh = cv2.threshold(img_erosion,127,255,0)
imgconv = cv2.cvtColor(img_erosion, cv2.COLOR_HSV2BGR)
imgconvGray = cv2.cvtColor(imgconv, cv2.COLOR_BGR2GRAY);
(ret, imgconvGray) = cv2.threshold(imgconvGray, 50, 255, cv2.THRESH_BINARY)


#im_pil = Image.fromarray(closing)


#nueva_imagen = im_pil.filter(ImageFilter.MedianFilter(tamaño))
#im_np = np.asarray(nueva_imagen)
#median = cv2.medianBlur(closing,5)
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

#cv2.imshow("Ventana de imagen seleccionada",M)
im= (1/255)*(cv2.imread(root.filename))
imgconvGraynorm = ((1/255)*imgconvGray)
a=(np.multiply(im[:,:,0],imgconvGraynorm))
b=(np.multiply(im[:,:,1],imgconvGraynorm))
c=(np.multiply(im[:,:,2],imgconvGraynorm))
imgfull = im
imgfull[:,:,0]=a
imgfull[:,:,1]=b
imgfull[:,:,2]=c

# Read image as gray-scale
# Convert to gray-scale
img= imgfull
gray = cv2.cvtColor(imgfull, cv2.COLOR_BGR2GRAY)
# Blur the image to reduce noise
img_blur = cv2.medianBlur(gray, 5)
# Apply hough transform on the image
circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img.shape[0]/64, param1=200, param2=10, minRadius=5, maxRadius=30)
# Draw detected circles
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Draw outer circle
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw inner circle
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)


cv2.imshow("Ventana de binarizada",otsu)
cv2.imshow("Ventana dilatación",img_dilation)
cv2.imshow("Ventana erosión",img_erosion)
cv2.imshow("Ventana Imagen Resultado",img)
cv2.imshow("Mascara",imgconvGray)
cv2.imwrite("prueba.jpg",imgfull*255)

cv2.waitKey(0)