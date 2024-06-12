import numpy as np
import atomictools as at
from .orbitals import cart_to_sph, sph_to_cart

class integral():
    
    def __init__(self, orbital1, orbital2=None):
        
        self.orb1 = orbital1
        if orbital2 is None:
            self.orb2 = orbital1
            self.xmin = orbital1.xmin
            self.xmax = orbital1.xmax
            self.ymin = orbital1.ymin
            self.ymax = orbital1.ymax
            self.zmin = orbital1.zmin
            self.zmax = orbital1.zmax

            self.delta_x = orbital1.delta_x
            self.delta_y = orbital1.delta_y
            self.delta_z = orbital1.delta_z
            self.d3 = orbital1.d3
                
            self.x, self.y, self.z = orbital1.x, orbital1.y, orbital1.z
            self.r, self.theta, self.phi = orbital1.r, orbital1.theta, orbital1.phi
            
        else:
            self.orb2 = orbital2
            
            self.xmin = min([orbital1.xmin,orbital2.xmin])
            self.xmax = max([orbital1.xmax,orbital2.xmax])
            self.ymin = min([orbital1.ymin,orbital2.ymin])
            self.ymax = max([orbital1.ymax,orbital2.ymax])
            self.zmin = min([orbital1.zmin,orbital2.zmin])
            self.zmax = max([orbital1.zmax,orbital2.zmax])
            self.x, self.y, self.z = np.mgrid[self.xmin:self.xmax:40j, self.ymin:self.ymax:40j, self.zmin:self.zmax:40j]
                
            self.delta_x = (self.xmax-self.xmin) / 39.
            self.delta_y = (self.ymax-self.ymin) / 39.
            self.delta_z = (self.zmax-self.zmin) / 39.
            self.d3 = self.delta_x * self.delta_y * self.delta_z
        
            r, theta, phi = cart_to_sph(self.x, self.y, self.z)
            self.r = r
            self.theta = theta
            self.phi = phi
            
    def derivative_r(self, matrix, kr):
        for n in range(kr):
            deriv  = (self.x/self.r) * np.diff(matrix, axis= 0, append=0.) / self.delta_x
            deriv += (self.y/self.r) * np.diff(matrix, axis= 1, append=0.) / self.delta_y
            deriv += (self.z/self.r) * np.diff(matrix, axis=-1, append=0.) / self.delta_z
            matrix = deriv * 1.
        return deriv
        
    def derivative_theta(self, matrix, ktheta):
        for n in range(ktheta):
            deriv  = self.z * np.diff(matrix, axis=0, append=0.) * np.cos(self.phi) / self.delta_x
            deriv += self.z * np.diff(matrix, axis=1, append=0.) * np.sin(self.phi) / self.delta_y
            deriv -= self.r * np.sin(self.theta) * np.diff(matrix, axis=-1, append=0.) / self.delta_z
            matrix = deriv * 1.
        return deriv
    
    def derivative_phi(self, matrix, kphi):
        for n in range(kphi):
            deriv  = - self.y * np.diff(matrix, axis=0, append=0.) / self.delta_x
            deriv +=   self.x * np.diff(matrix, axis=1, append=0.) / self.delta_y
            matrix = deriv * 1.
        return deriv
    
    def derivative_x(self, matrix, kx): 
        for n in range(kx):
            deriv = np.diff(matrix, n=1, axis=0, append=0.) / self.delta_x #shape[40,40,40]
            matrix = deriv * 1.
        return deriv
    
    def derivative_y(self, matrix, ky):
        for n in range(ky):
            deriv = np.diff(matrix, n=1, axis=1, append=0.) / self.delta_y #shape[40,40,40]
            matrix = deriv * 1.
        return deriv
    
    def derivative_z(self, matrix, kz): 
        for n in range(kz):
            deriv = np.diff(matrix, n=1, axis=-1, append=0.) / self.delta_z #shape[40,40,40]
            matrix = deriv * 1.
        return deriv
    
    def bra(self):
        if isinstance(self.orb1, at.hybrid_orbital):
            bra = self.orb1.evaluate(self.x, self.y, self.z)
        else:
            bra = self.orb1.evaluate(self.r, self.theta, self.phi)
        return np.conjugate(bra)

    def ket(self):
        if isinstance(self.orb2, at.hybrid_orbital):
            ket = self.orb2.evaluate(self.x, self.y, self.z)
        else:
            ket = self.orb2.evaluate(self.r, self.theta, self.phi)
        return ket
            
    def integral_sph(self, f=None, kr=None, ktheta=None, kphi=None):       
        bra = self.bra()
        ket = self.ket()
        h = self.d3
        if kr is not None:
                if isinstance(kr, int) and kr >= 1:
                    ket = self.derivative_r(ket, kr)         
                else:
                    raise ValueError("kr must be a positive integer")
        
        if ktheta is not None:
                if isinstance(ktheta, int) and ktheta >= 1:
                    ket = self.derivative_theta(ket, ktheta)
                else:
                    raise ValueError("ktheta must be a positive integer")
            
        if kphi is not None:
                if isinstance(kphi, int) and kphi >= 1:
                    ket = self.derivative_phi(ket, kphi)     
                else:
                    raise ValueError("kphi must be a positive integer")
                
        if f is not None:
            F = bra * f(self.r,self.theta,self.phi) * ket
        else:
            F = bra * ket
        I = F.sum() * h
        if I.imag == 0.0:
            return I.real
        else: 
            return I

    def integral_cart(self, f=None, kx=None, ky=None, kz=None): 
        bra = self.bra()
        ket = self.ket()
        h = self.d3
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
            F = bra * f(self.x,self.y,self.z) * ket
        else:
            F = bra * ket
        I = F.sum() * h
        if I.imag == 0.0:
            return I.real
        else: 
            return I

def matrix_element(orbital1, orbital2=None,
                   f=None, kx=None, ky=None, kz=None, kr=None, ktheta=None, kphi=None, coord='cart'):
    integ = integral(orbital1, orbital2=None)
    if coord=='sph':
        return integ.integral_sph(f, kr, ktheta, kphi)
    else:
        return integ.integral_cart(f, kx, ky, kz)
