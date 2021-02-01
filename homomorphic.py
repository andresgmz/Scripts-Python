import logging
import numpy as np
import cv2
    
from tkinter import filedialog
from tkinter import *

# Homomorphic filter class
class HomomorphicFilter:
    

    def _init_(self, a = 0.5, b = 1.5):
        self.a = float(a)
        self.b = float(b)

    # Filters
    def __butterworth_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)*2+(V-Q)*2)).astype(float)
        H = 1/(1+(Duv/filter_params[0]*2)*filter_params[1])
        return (1 - H)

    def __gaussian_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        H = np.zeros(I_shape)
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)*2+(V-Q)*2)).astype(float)
        H = np.exp((-Duv/(2*(filter_params[0])**2)))
        return (1 - H)

    # Methods
    def __apply_filter(self, I, H):
        H = np.fft.fftshift(H)
        I_filtered = (self.a + self.b*H)*I
        return I_filtered

   

    def filter(self, I, filter_params, filter='butterworth', H = None):
        

        #  Validating image
        if len(I.shape) is not 2:
            raise Exception('Improper image')

        # Take the image to log domain and then to frequency domain 
        I_log = np.log1p(np.array(I, dtype="float"))
        I_fft = np.fft.fft2(I_log)

        # Filters
        if filter=='butterworth':
            H = self.__butterworth_filter(I_shape = I_fft.shape, filter_params = filter_params)
        elif filter=='gaussian':
            H = self.__gaussian_filter(I_shape = I_fft.shape, filter_params = filter_params)
        elif filter=='external':
            print('external')
            if len(H.shape) is not 2:
                raise Exception('Invalid external filter')
        else:
            raise Exception('Selected filter not implemented')
        
        # Apply filter on frequency domain then take the image back to spatial domain
        I_fft_filt = self.__apply_filter(I = I_fft, H = H)
        I_filt = np.fft.ifft2(I_fft_filt)
        I = np.exp(np.real(I_filt))-1
        return np.uint8(I)

    def mediana(self,I,paramMediana ):
        Imedian = cv2.medianBlur(I,paramMediana)
        return Imedian


# End of class HomomorphicFilter
""" class FiltroSuavizado:
     def _init_(self, a = 0.5, b = 1.5):
        self.a = float(a)
        self.b = float(b)
        # Filters
    def __butterworth_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)*2+(V-Q)*2)).astype(float)
        H = 1/(1+(Duv/filter_params[0]*2)*filter_params[1])
        return (1 - H)
        
 """

if __name__== "_main_":
    

    root = Tk()
    root.withdraw()
    root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("all files",".*"),("jpg files",".jpg")))
    img = cv2.imread(root.filename,0)
    i2 = cv2.imread(root.filename)
    
    root.destroy()
    # Main code
    
    ##objeto.
    homo_filter = HomomorphicFilter(a = 0.9, b =1.4)
    IMediana = homo_filter.mediana(I=img,paramMediana = 3)
    img_filtered = homo_filter.filter(I=IMediana, filter_params=[45,1])
    img_stacked = np.hstack((img,img_filtered)) #stacking images side-by-side 0.6l
    new=img_filtered+15
    # Ecualizacion
   # equ = cv2.equalizeHist(img_filtered)

    #mediana
    #cv2.imshow("Ventana de imagen Ecualizada",equ)
    cv2.imshow("Ventana de imagen salida",img_stacked)
    cv2.imshow("Ventana sin nada",i2)
    cv2.imshow("ImagenMediana",IMediana)
    cv2.imshow("final",new)
    cv2.imwrite('image.png',new)
    print("HOMADALDALD....")
    cv2.waitKey(0)