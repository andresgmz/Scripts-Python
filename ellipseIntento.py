import matplotlib.pyplot as plt
import cv2

from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
from skimage.color import rgb2gray
from skimage.transform import rescale, resize, downscale_local_mean
# Load picture, convert to grayscale and detect edges
from tkinter import filedialog
from tkinter import *


def find_ellipse(edgesImg,img):


    print("starting Hough ellipse detection....")
    #ximg = rescale(edgesImg, 0.3)
    ximg = edgesImg
    result = hough_ellipse(ximg, accuracy=20, threshold=250,
                       min_size=0, max_size=0)
    """ result = hough_ellipse(ximg,accuracy=45, threshold=10,
                       min_size=70, max_size=180) """
    #result.sort(order='accumulator')
    print("sorted result:", result)
    print("list size:", result.size)
    """ if result.size:
        #return result
        best = list(result[-1])
        yc, xc, a, b = [int(round(x)) for x in best[1:5]]
        orientation = best[5]
        print("orientation", orientation)
        coor = ((int(xc),int(yc)),(int(a),int(b)),orientation)
        print("coordinates:",coor)
        cv2.ellipse(ximg,coor,(255,0,0),2,cv2.LINE_AA)
        cv2.imshow("Hough",ximg)
        cv2.imshow("edges",edgesImg)
        cv2.waitKey()
        return coor
    else:
        print("Sorry nothing found!") """

root = Tk()
root.withdraw()
root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("all files",".*"),("jpg files",".jpg")))
face_cascade = cv2.CascadeClassifier('C:/Users/andre/Desktop/NovenoSemestre/VisionArtificial/Python/haarcascade_frontalface_alt.xml')
image_rgb = data.coffee()[0:220, 160:420]
#image_rgb = cv2.imread(root.filename)
root.destroy()

""" image_rgb = data.coffee()[0:220, 160:420]
image_gray = color.rgb2gray(image_rgb) """
image_gray= cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)
""" for (x,y,w,h) in faces:
    center = (x + w//2, y + h//2)
#3.5 0.55 0.8
minRadius=int((w//2-10)*0.3)
maxRadius=int((w//2+10)*0.3) """
edges = canny(image_gray, sigma=2,
              low_threshold=0.55, high_threshold=0.8)
#3.5,3.3,6
#20  50 
print("starting Hough ellipse detection....")
ximg = rescale(edges, 0.3, multichannel=True)

result = hough_ellipse(ximg, accuracy=0, threshold=250,
                    min_size=100, max_size=120)
#print(maxRadius,minRadius)
""" result = hough_ellipse(ximg,accuracy=45, threshold=10,
                    min_size=70, max_size=180) """
#result.sort(order='accumulator')
print("sorted result:", result)
print("list size:", result.size)

fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4),
                                sharex=True, sharey=True)
ax1.set_title('Original picture')
ax1.imshow(image_rgb)

ax2.set_title('Edge (white) and result (red)')
ax2.imshow(edges)
plt.show()
""" cv2.imshow('img',image_rgb)
cv2.imshow('dimg',edges)
cv2.waitKey(0) """
#Image_ellipses = find_ellipse(edges,image_gray)


