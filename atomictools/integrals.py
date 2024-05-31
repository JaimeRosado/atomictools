import numpy as np
import atomictools as at

def sph_to_cart(r, theta, phi):
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return x, y, z

def cart_to_sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)
    return r, theta, phi

#JR: Crear que funcion para que el usuario no tenga que crear un objeto integral

class integral():
    
    def __init__(self, orbital1, orbital2=None):
        
        if orbital2==None:
            self.orb1 = orbital1
            self.orb2 = orbital1
            rmax = self.orb1.rmax
            self.rmax = rmax
        else:     # Para orbitales moleculares, obtener xmin, xmax, ...
            self.orb1 = orbital1
            self.orb2 = orbital2
            rmax1 = self.orb1.rmax
            rmax2 = self.orb2.rmax
            
            if rmax1 > rmax2:
                self.rmax = rmax1
            else:
                self.rmax = rmax2
        
        #Para el caso orbital2=orbital1, no hace falta redefinir nada de esto
        #Usa lo que ya está definido en orbital1 (p.ej. self.r = orbital1.r)
        self.x, self.y, self.z = np.mgrid[-rmax:rmax:40j, -rmax:rmax:40j, -rmax:rmax:40j]
        r, theta, phi = cart_to_sph(self.x, self.y, self.z)
        self.r = r
        self.theta = theta
        self.phi = phi
        d3 = (2.*rmax/39.)**3 
        self.d3 = d3
    
    #Calcular las derivadas en esféricas en términos de las derivadas de x, y, z
    def derivative_r(self, matrix, kr):
        dr = self.r[1:,1:,1:]-self.r[:-1,:-1,:-1]
        deriv_amp = np.zeros([40,40,40])    
        for n in range(kr):
            dmatrix = matrix[1:,1:,1:]-matrix[:-1,:-1,:-1]
            #for theta in range(39.):
            #    for phi in range(39.):
            #        for r in range(39.): 
            #            dket[r,theta,phi] = self.ket[r+1,theta,phi]-self.ket[r,theta,phi]
            deriv = dmatrix/dr #shape[39,39,39]
            deriv_amp[:-1,:-1,:-1] = deriv #shape[40,40,40]
            matrix = deriv_amp    
        return deriv_amp
        
    def derivative_theta(self, matrix, ktheta):
        dtheta = self.theta[1:,1:,1:]-self.theta[:-1,:-1,:-1]
        deriv_amp = np.zeros([40,40,40])    
        for n in range(ktheta):
            dmatrix = matrix[:-1,1:,:-1]-matrix[:-1,:-1,:-1]
            deriv = dmatrix/dtheta #shape[39,39,39]
            deriv_amp[:-1,:-1,:-1] = deriv #shape[40,40,40]
            matrix = deriv_amp    
        return deriv_amp
    
    def derivative_phi(self, matrix, kphi):
        dtheta = self.phi[1:,1:,1:]-self.phi[:-1,:-1,:-1]
        deriv_amp = np.zeros([40,40,40])    
        for n in range(kphi):
            dmatrix = matrix[:-1,1:,:-1]-matrix[:-1,:-1,:-1]
            deriv = dmatrix/dphi #shape[39,39,39]
            deriv_amp[:-1,:-1,:-1] = deriv #shape[40,40,40]
            matrix = deriv_amp    
        return deriv_amp
    
    #Usar diff de numpy con las opciones prepend/append
    def derivative_x(self, matrix, kx):
        dx = self.x[1:,:,:]-self.x[:-1,:,:]
        deriv_amp = np.zeros([40,40,40])    
        for n in range(kx):
            dmatrix = matrix[1:,:,:]-matrix[:-1,:,:]
            deriv = dmatrix/dx #shape[39,39,39]
            deriv_amp[:-1,:,:] = deriv #shape[40,40,40]
            matrix = deriv_amp
        return deriv_amp
    
    def derivative_y(self, matrix, ky):
        dy = self.y[1:,1:,1:]-self.y[:-1,:-1,:-1]
        deriv_amp = np.zeros([40,40,40])    
        for n in range(ky):
            dmatrix = matrix[:-1,1:,:-1]-matrix[:-1,:-1,:-1]
            deriv = dmatrix/dy #shape[39,39,39]
            deriv_amp[:-1,:-1,:-1] = deriv #shape[40,40,40]
            matrix = deriv_amp    
        return deriv_amp
    
    def derivative_z(self, matrix, kz):
        dz = self.z[1:,1:,1:]-self.z[:-1,:-1,:-1]
        deriv_amp = np.zeros([40,40,40])    
        for n in range(kz):
            dmatrix = matrix[:-1,1:,:-1]-matrix[:-1,:-1,:-1]
            deriv = dmatrix/dz #shape[39,39,39]
            deriv_amp[:-1,:-1,:-1] = deriv #shape[40,40,40]
            matrix = deriv_amp    
        return deriv_amp
        
    def integral_sph(self,f=None, kr=None, ktheta=None, kphi=None):
        
        orb1 = self.orb1.evaluate(self.r, self.theta, self.phi)
        bra = np.conjugate(orb1) #debería hacer la traspuesta?
        orb2 = self.orb2.evaluate(self.r, self.theta, self.phi)
        ket = orb2
        
        if kr==None and ktheta==None and kphi==None:
            if f is not None:
                F = bra*ket*f(self.r,self.theta,self.phi) #Quedarse con parte Re si la parte imaginaria es nula
                return F.sum() * self.d3 
            else: 
                raise ValueError("A variable is necessary")
        else:
            if kr!=None and ktheta==None and kphi==None:
                if isinstance(kr, int) and kr >= 1:
                    deriv_ket = self.derivative_r(ket, kr)         
                else:
                    raise ValueError("kr must be a positive integer")
                F = np.abs(bra*deriv_ket) #*self.r**2*np.sin(self.theta) (si integro en cartisanas el final de la matriz no es aprox 0 no?)
                return F.sum() * self.d3
            
            if kr==None and ktheta!=None and kphi==None:
                if isinstance(ktheta, int) and ktheta >= 1:
                    deriv_ket = self.derivative_theta(ket, ktheta)
                else:
                    raise ValueError("ktheta must be a positive integer")
                F = np.abs(bra*deriv_ket) #*self.r**2*np.sin(self.theta)
                return F.sum() * self.d3
            
            if kr==None and ktheta==None and kphi!=None:
                if isinstance(kphi, int) and kphi >= 1:
                    deriv_ket = self.derivative_phi(ket, kphi)     
                else:
                    raise ValueError("kphi must be a positive integer")
                F = np.abs(bra*deriv_ket) #*self.r**2*np.sin(self.theta)
                return F.sum()*self.d3
                    
    #Generalizar para cualquier operador, redifiniendo ket en cada paso con condiciones en f, kx...
    def integral_cart(self, f=None, kx=None, ky=None, kz=None): 
        
        #orb1 = at.hybrid_orbital(orbitals=[self.orb1], coefficients=[1.]).evaluate(self.x,self.y,self.z)
        #bra = np.conjugate(orb1) 
        #orb2 = at.hybrid_orbital(orbitals=[self.orb2], coefficients=[1.]).evaluate(self.x,self.y,self.z)
        #ket = orb2
        orb1 = self.orb1.evaluate(self.r, self.theta, self.phi)
        bra = np.conjugate(orb1) #debería hacer la traspuesta?
        orb2 = self.orb2.evaluate(self.r, self.theta, self.phi)
        ket = orb2
        
        h = self.d3
        
        if kx==None and ky==None and kz==None:
            if f!=None:
                F = np.abs(bra*ket)*f(self.x,self.y,self.z)
                return F.sum() * self.d3 
            else: 
                raise ValueError("A variable is necessary")
        else:
            if kx!=None and ky==None and kz==None:
                if isinstance(kx, int) and kx >= 1:
                    deriv_ket = self.derivative_x(ket, kx)         
                else:
                    raise ValueError("kr must be a positive integer")
                F = np.abs(bra*deriv_ket)
                return F.sum() * self.d3
            
            if kx==None and ky!=None and kz==None:
                if isinstance(ky, int) and ky >= 1:
                    deriv_ket = self.derivative_y(ket, ky)
                else:
                    raise ValueError("ktheta must be a positive integer")
                F = np.abs(bra*deriv_ket)
                return F.sum() * self.d3
            
            if kx==None and ky==None and kz!=None:
                if isinstance(kz, int) and kz >= 1:
                    deriv_ket = self.derivative_z(ket, kz)     
                else:
                    raise ValueError("kphi must be a positive integer")
                F = np.abs(bra*deriv_ket)
                return F.sum()*self.d3
            
        # 1) Si no se da orbital2, asumir orbital2=orbital1. Se mantiene la discretización del orbital.
        #    Si se da orbital2, comparar xmin, xmax, ymin... de ambos orbitales
        #    y tomar los rangos extremos una discretización común para ambos orbitales.
        #    Reevaluarlos con esta nueva discretización
        # 2) Si alguno o varios de kr, ktheta o kphi son enteros >=1, entonces derivar orbital2.
        #    En caso contrario, tomar orbital2 (función compleja).
        # 3) Hacer el conjugado complejo de orbital1 y multiplicar por el resultado del punto 2)
        # 4) Si se da f (se espera que f sea una función de r, theta, phi en ese orden),
        #    evaluar la función en los puntos de discretización y multiplicar por el resultado de 3)
        # 5) Integrar (sumar)
        #
        # Hacer lo mismo para una función integral_cart en el que se espera un operador de x,y,z
        # Si sólo se da orbital1 y f, debería dar el mismo resultado que el valor esperado.
        # El método de la clase orbital para el valor esperado con operadores diferenciales
        # puede usar esta función para no repetir código.
        # El cálculo de derivadas puede estar en una función auxiliar.    