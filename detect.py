class Caja:
    def __init__(self,largo,ancho, costo =10):
        self.largo=largo 
        self.ancho= ancho
        self.costo = costo
    
    def area(self):
        return self.largo * self.ancho 
    
    def precio (self):
        
        return self.area()*self.costo
r = Caja(160,120,200)
print("El area", (r.precio()))
