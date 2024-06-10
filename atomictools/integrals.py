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

class integral():
    
    def __init__(self, orbital1, orbital2=None):
        
        if orbital2==None:
            self.orb1 = orbital1
            self.orb2 = orbital1
            if hasattr(self.orb1, 'xmin'):
                self.xmin = self.orb1.xmin
                self.xmax = self.orb1.xmax
                self.ymin = self.orb1.ymin
                self.ymax = self.orb1.ymax
                self.zmin = self.orb1.zmin
                self.zmax = self.orb1.zmax 
                
                self.delta_x = (self.xmax-self.xmin) / 39.
                self.delta_y = (self.ymax-self.ymin) / 39.
                self.delta_z = (self.zmax-self.zmin) / 39.
                self.d3 = self.delta_x * self.delta_y * self.delta_z
            else:            
                rmax = self.orb1.rmax
                self.rmax = rmax
                self.d3 = (2.*rmax/39.)**3 
                
            self.x, self.y, self.z = self.orb1.x, self.orb1.y, self.orb1.z
            self.r, self.theta, self.phi = self.orb1.r, self.orb1.theta, self.orb1.phi
            
        else:     
            self.orb1 = orbital1
            self.orb2 = orbital2
            if hasattr(self.orb1, 'xmin') and hasattr(self.orb2, 'xmin'):
                xmin1 = self.orb1.xmin
                xmax1 = self.orb1.xmax
                ymin1 = self.orb1.ymin
                ymax1 = self.orb1.ymax
                zmin1 = self.orb1.zmin
                zmax1 = self.orb1.zmax  
           
                xmin2 = self.orb1.xmin
                xmax2 = self.orb1.xmax
                ymin2 = self.orb1.ymin
                ymax2 = self.orb1.ymax
                zmin2 = self.orb1.zmin
                zmax2 = self.orb1.zmax 
           
                self.xmin = max([xmin1,xmin2])
                self.xmax = max([xmax1,xmax2])
                self.ymin = max([ymin1,ymin2])
                self.ymax = max([ymax1,ymax2])
                self.zmin = max([zmin1,zmin2])
                self.zmax = max([zmax1,zmax2])
                self.x, self.y, self.z = np.mgrid[self.xmin:self.xmax:40j, self.ymin:self.ymax:40j, self.zmin:self.zmax:40j]
                
                self.delta_x = (self.xmax-self.xmin) / 39.
                self.delta_y = (self.ymax-self.ymin) / 39.
                self.delta_z = (self.zmax-self.zmin) / 39.
                self.d3 = self.delta_x * self.delta_y * self.delta_z
            else:
                rmax1 = self.orb1.rmax
                rmax2 = self.orb2.rmax
                rmax = max([rmax1,rmax2])
                self.rmax = rmax
                self.x, self.y, self.z = np.mgrid[-rmax:rmax:40j, -rmax:rmax:40j, -rmax:rmax:40j] 
                self.d3 = (2.*rmax/39.)**3 
        
            r, theta, phi = cart_to_sph(self.x, self.y, self.z)
            self.r = r
            self.theta = theta
            self.phi = phi
            
    def derivative_r(self, matrix, kr):
        dr = (self.x/self.r)*np.diff(self.r, n=1, axis=0, append=0) + (self.y/self.r)*np.diff(self.r , n=1, axis=1, append=0) +      (self.z/self.r)*np.diff(self.r, n=1, axis=-1, append=0)  
        for n in range(kr):
            dmatrix = (self.x/self.r)*np.diff(matrix, n=1, axis=0, append=0) + (self.y/self.r)*np.diff(matrix , n=1, axis=1, append=0) + (self.z/self.r)*np.diff(matrix, n=1, axis=-1, append=0) 
            deriv = dmatrix/dr #shape[40,40,40]
            matrix = deriv  
        return deriv
        
    def derivative_theta(self, matrix, ktheta):
        dtheta = self.r*np.cos(self.theta)*(np.diff(self.theta, n=1, axis=0, append=0)*np.cos(self.phi) + np.diff(self.theta, n=1, axis=1, append=0)*np.sin(self.phi))-self.r*np.sin(self.theta)*np.diff(self.theta, n=1, axis=-1, append=0)
        for n in range(ktheta):
            dmatrix = self.r*np.cos(self.theta)*(np.diff(matrix, n=1, axis=0, append=0)*np.cos(self.phi) + np.diff(matrix, n=1, axis=1, append=0)*np.sin(self.phi))-self.r*np.sin(self.theta)*np.diff(matrix, n=1, axis=-1, append=0)
            deriv = dmatrix/dtheta #shape[40,40,40]
            matrix = deriv 
        return deriv
    
    def derivative_phi(self, matrix, kphi):
        dphi = self.r*np.sin(self.theta)*(-np.diff(self.phi, n=1, axis=0, append=0)*np.sin(self.phi) + np.diff(self.phi, n=1, axis=1, append=0)*np.cos(self.phi))
        for n in range(kphi):
            dmatrix = self.r*np.sin(self.theta)*(-np.diff(matrix, n=1, axis=0, append=0)*np.sin(self.phi) + np.diff(matrix, n=1, axis=1, append=0)*np.cos(self.phi))
            deriv = dmatrix/dphi #shape[40,40,40]
            matrix = deriv
        return deriv
    
    def derivative_x(self, matrix, kx):
        dx = np.diff(self.x, n=1, axis=0, append=0)   
        for n in range(kx):
            dmatrix = np.diff(matrix, n=1, axis=0, append=0)
            deriv = dmatrix/dx #shape[40,40,40]
            matrix = deriv
        return deriv
    
    def derivative_y(self, matrix, ky):
        dy = np.diff(self.y, n=1, axis=1, append=0)
        for n in range(ky):
            dmatrix = np.diff(matrix, n=1, axis=1, append=0)
            deriv = dmatrix/dy #shape[40,40,40]
            matrix = deriv   
        return deriv
    
    def derivative_z(self, matrix, kz):
        dz = np.diff(self.z, n=1, axis=-1, append=0)   
        for n in range(kz):
            dmatrix = np.diff(matrix, n=1, axis=-1, append=0)
            deriv = dmatrix/dz #shape[40,40,40]
            matrix = deriv  
        return deriv
        
    def integral_sph(self,f=None, kr=None, ktheta=None, kphi=None):
        
        orb1 = self.orb1.evaluate(self.r, self.theta, self.phi)
        bra = np.conjugate(orb1) 
        orb2 = self.orb2.evaluate(self.r, self.theta, self.phi)
        ket = orb2
        
        h = self.d3
        
        if kr is not None:
                if isinstance(kr, int) and kr >= 1:
                    deriv_ket = self.derivative_r(ket, kr)         
                else:
                    raise ValueError("kr must be a positive integer")
                ket = deriv_ket
        
        if ktheta is not None:
                if isinstance(ktheta, int) and ktheta >= 1:
                    deriv_ket = self.derivative_y(ket, ktheta)
                else:
                    raise ValueError("ktheta must be a positive integer")
                ket = deriv_ket
            
        if kphi is not None:
                if isinstance(kphi, int) and kphi >= 1:
                    deriv_ket = self.derivative_z(ket, kphi)     
                else:
                    raise ValueError("kphi must be a positive integer")
                ket = deriv_ket
                
        if f is not None:
            F = bra*f(self.r,self.theta,self.phi)*ket
            I = F.sum() * self.h
            if I.imag == 0.0:
                return I.real
            else: 
                return I
        if f is None:
            F = bra*ket
            I = F.sum() * self.h
            if I.imag == 0.0:
                return I.real
            else: 
                return I
    def integral_cart(self, f=None, kx=None, ky=None, kz=None): 
        
        orb1 = self.orb1.evaluate(self.r, self.theta, self.phi)
        bra = np.conjugate(orb1) 
        orb2 = self.orb2.evaluate(self.r, self.theta, self.phi)
        ket = orb2
        
        h = self.d3
        
        if kx is not None:
                if isinstance(kx, int) and kx >= 1:
                    deriv_ket = self.derivative_x(ket, kx)         
                else:
                    raise ValueError("kr must be a positive integer")
                ket = deriv_ket
        
        if ky is not None:
                if isinstance(ky, int) and ky >= 1:
                    deriv_ket = self.derivative_y(ket, ky)
                else:
                    raise ValueError("ktheta must be a positive integer")
                ket = deriv_ket
            
        if kz is not None:
                if isinstance(kz, int) and kz >= 1:
                    deriv_ket = self.derivative_z(ket, kz)     
                else:
                    raise ValueError("kphi must be a positive integer")
                ket = deriv_ket
                
        if f is not None:
            F = bra*f(self.x,self.y,self.z)*ket
            I = F.sum() * self.h
            if I.imag == 0.0:
                return I.real
            else: 
                return I
        if f is None:
            F = bra*ket
            I = F.sum() * self.h
            if I.imag == 0.0:
                return I.real
            else: 
                return I
        
class integral_aux(integral):
    
    def __init__(self, x, y, z, r, theta, phi, orbital1, d3, orbital2=None): #these orbitals must be matrix
        if orbital2==None:
            self.orb1 = orbital1
            self.orb2 = orbital1
        else:     
            self.orb1 = orbital1
            self.orb2 = orbital2
        
        h = d3
        self.h = h
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.theta = theta
        self.phi = phi
                                                 
    def integral_cart(self, f=None, kx=None, ky=None, kz=None): 
        bra = np.conjugate(self.orb1) 
        ket = self.orb2
        
        if kx is not None:
                if isinstance(kx, int) and kx >= 1:
                    deriv_ket = self.derivative_x(ket, kx)         
                else:
                    raise ValueError("kx must be a positive integer")
                ket = deriv_ket
        
        if ky is not None:
                if isinstance(ky, int) and ky >= 1:
                    deriv_ket = self.derivative_y(ket, ky)
                else:
                    raise ValueError("ky must be a positive integer")
                ket = deriv_ket
            
        if kz is not None:
                if isinstance(kz, int) and kz >= 1:
                    deriv_ket = self.derivative_z(ket, kz)     
                else:
                    raise ValueError("kz must be a positive integer")
                ket = deriv_ket
                
        if f is not None:
            F = bra*f(self.x,self.y,self.z)*ket
            I = F.sum() * self.h
            if I.imag == 0.0:
                return I.real
            else: 
                return I
        if f is None:
            F = bra*ket
            I = F.sum() * self.h
            if I.imag == 0.0:
                return I.real
            else: 
                return I

    def integral_sph(self, f=None, kr=None, ktheta=None, kphi=None): 
        bra = np.conjugate(self.orb1) 
        ket = self.orb2
        
        if kr is not None:
                if isinstance(kr, int) and kr >= 1:
                    deriv_ket = self.derivative_r(ket, kr)         
                else:
                    raise ValueError("kr must be a positive integer")
                ket = deriv_ket
        
        if ktheta is not None:
                if isinstance(ktheta, int) and ktheta >= 1:
                    deriv_ket = self.derivative_theta(ket, ktheta)
                else:
                    raise ValueError("ktheta must be a positive integer")
                ket = deriv_ket
            
        if kphi is not None:
                if isinstance(kphi, int) and kphi >= 1:
                    deriv_ket = self.derivative_z(ket, kphi)     
                else:
                    raise ValueError("kphi must be a positive integer")
                ket = deriv_ket
                
        if f is not None:
            F = bra*f(self.r,self.theta,self.phi)*ket
            I = F.sum() * self.h
            if I.imag == 0.0:
                return I.real
            else: 
                return I
        if f is None:
            F = bra*ket
            I = F.sum() * self.h
            if I.imag == 0.0:
                return I.real
            else: 
                return I