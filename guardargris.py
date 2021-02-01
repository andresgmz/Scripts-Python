import cv2
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import pyplot as plt
from tkinter import filedialog
from tkinter import *


root = Tk()
root.withdraw()
root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("all files",".*"),("jpg files",".jpg")))
img = cv2.imread(root.filename,0)
equ = cv2.equalizeHist(img)

cv2.imwrite('fotogrisequ.png',equ)
cv2.imshow("Que tal",img)
cv2.waitKey(0)
#cv2.imwrite(os.path.join(pathdestino , completo), norm)
