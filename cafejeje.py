from skimage import data, color, img_as_ubyte
import cv2
image_rgb = data.coffee()
cv2.imshow('dimg',image_rgb)
cv2.waitKey(0)