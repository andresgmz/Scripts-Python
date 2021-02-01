import numpy
import cv2
from tkinter import filedialog
from tkinter import *
""" from tkinter import *
from tkinter.filedialog import askopenfilename

root = Tk()
root.withdraw()
root.update()
pathString = askopenfilename(filetypes=[("Text files","*.jpg")])
if pathString != "":
    openFile = open(pathString, 'r')
    fileString = openFile.read()
    #print(fileString)
root.destroy() """


root = Tk()
root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))

imagen = cv2.imread(root.filename)
gray=cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
cv2.imshow("Ventana de imagen",gray)
cv2.imshow("Ventana de imagen NORMI",imagen)
cv2.waitKey(0)