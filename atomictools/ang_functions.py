#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#import plotly.express as px
import plotly.graph_objects as go
#import plotly.io as pio
#from plotly.subplots import make_subplots
#from cmath import phase
import scipy.special as spe
import numpy as np
#import sys
#from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
#pio.renderers.default='iframe'
#np.seterr(invalid='ignore')

def ftheta(l, m, theta):
    C = np.sqrt((2.*l+1.) * spe.factorial(l-m) / 2. / spe.factorial(l+m))
    legendre = spe.lpmv(m, l, np.cos(theta))
    return C * legendre

def fphi(m, phi, part=None):
    if part is None:
        return np.sqrt(1./(2.*np.pi)) * np.exp(1j * m * phi)
    elif part=="Re" or m==0:
        return (-1)**m * np.sqrt(1./np.pi) * np.cos(m * phi)
    elif part=="Im":
        return (-1)**m * np.sqrt(1./np.pi) * np.sin(m * phi)
    else:
        raise ValueError("The parameter part should be Re or Im.")

class spherical_harmonic:
    """
    Class for the angular function calculated by orbitals.py
    Parameters are in radians.
    
    Parameters
    ----------
    l : int
        Orbital angular momentum quantum number.
    m : int
        Magnetic quantum number.

    Attributes
    ----------
    theta : array
            Polar angle in radians.
    phi : array
          Azimutal angle in radians.
        
    Methods
    -------
    Plot_prob()
        Plot the spherical harmonic as a probability distribution per solid angle.
    Plot_phase()
        Plot the spherical harmonic as a complex function.
    """
    
    def __init__(self, l, m):
        # definition of class atributes
        self.l = l
        self.m = m
        theta = np.linspace(0., np.pi, 50)
        self.theta = theta
        dtheta = theta[1]-theta[0]
        self.dtheta = dtheta
        sin_theta = np.sin(theta)
        self.sin_theta = sin_theta
        sin_theta[-1] = 0. # the last theta is pi, but sin_theta is not exactly 0
        phi = np.linspace(0., 2.*np.pi, 100)
        self.phi = phi
        dphi = phi[1] - phi[0]
        self.dphi = dphi
        self.part = None
        
        # Computed results of ftheta() and fphi()
        self.ftheta_lm = ftheta(l, m, theta)
        self.fphi_m = fphi(m, phi)
        
        # calculation of probability distributions
        self.theta_dist = (self.ftheta_lm**2 * sin_theta).cumsum() #* np.pi/49.
        self.theta_dist /= self.theta_dist[-1]
        
        # Y values, their absolute values, the square of them and phase
        Y = np.outer(self.fphi_m, self.ftheta_lm)
        self.Y = Y
        module = np.abs(Y)
        self.module = module
        prob = module**2
        self.prob = prob
        #prob_sin_theta = prob * sin_theta
        #self.prob_sin_theta = prob_sin_theta
        phase = np.angle(Y)
        self.phase = phase
        
        # Cartesian coordinates for r=1
        x = np.outer(np.cos(phi), sin_theta)
        y = np.outer(np.sin(phi), sin_theta)
        z = np.outer(np.ones_like(phi), np.cos(theta))
        
        # Starting zoom of the graph. Relies on the l variable to adjust the graph
        #side = 1.25 + l - np.abs(m)
        #self.camera = dict(eye=dict(x=side, y=side, z=1.25))

        # Cartesian projection of prob
        self.prob_x = prob * x
        self.prob_y = prob * y
        self.prob_z = prob * z
        
        # Cartesian projection of module
        self.module_x = module * x
        self.module_y = module * y
        self.module_z = module * z
    
    
    def plot_prob(self):
        fig = go.Figure(data=[go.Surface(x=self.prob_x, y=self.prob_y, z=self.prob_z, 
                                         surfacecolor=self.prob, colorscale='Oranges')])
  
        fig.update_traces(contours_z=dict(
               show=True, usecolormap=True,
               highlightcolor="limegreen",
               project_z=True))  
    
        fig.update_layout(scene_aspectmode='data')#, scene_camera=self.camera)
  
        fig.show()
    
            
    def plot_phase(self):    
        fig = go.Figure(data=[go.Surface(x=self.module_x, y=self.module_y, z=self.module_z,
            surfacecolor=self.phase, colorscale='HSV', cmin=-np.pi, cmax=np.pi)])

        fig.update_traces(contours_z=dict(
              show=True, usecolormap=True,
              highlightcolor="limegreen",
              project_z=True))
        
        fig.update_layout(scene_aspectmode='data')#, scene_camera=self.camera)
  
        fig.show()

    def evaluate(self, theta, phi):
        theta = np.array(theta)
        phi = np.array(phi)
        ftheta_lm = ftheta(self.l, self.m, theta)       
        fphi_m = fphi(self.m, phi)
        return ftheta_lm * fphi_m

    def get_theta(self, points=1):
        p = np.random.random(points)
        theta = np.interp(p, self.theta_dist, self.theta)
        return theta

    def get_phi(self, points=1):
        phi = np.random.random(points) * 2. * np.pi
        return phi

    def get_angles(self, points=1):
        theta = self.get_theta(points)
        phi = self.get_phi(points)
        return theta, phi
    
    #Defining integrands for expected values
    
    def integrand_theta(self,theta_value,phi_p,theta_p,k):
        
        return self.module[phi_p,theta_p]**2*theta_value**k*self.sin_theta[theta_p]
    
    def integrand_phi(self,phi_value,phi_p,theta_p,k):
        
        return self.module[phi_p,theta_p]**2*phi_value**k*self.sin_theta[theta_p]
    
    def integrand_ftheta(self,theta_value,phi_p,theta_p,f_given):
        
        return self.module[phi_p,theta_p]**2*f_given(theta_value)*self.sin_theta[theta_p]
    
    def integrand_fphi(self,phi_value,phi_p,theta_p,f_given):
        
        return self.module[phi_p,theta_p]**2*f_given(phi_value)*self.sin_theta[theta_p]
    
    def integrand_fthetaphi(self,phi_value,theta_value,phi_p,theta_p,f_given):
        
        return self.module[phi_p,theta_p]**2*f_given(theta_value,phi_value)*self.sin_theta[theta_p]
    
    #Obtaining the expected value of theta**k
    def expected_thetapower(self,k): #using the trapezoid method
        
        h_theta = self.dtheta
        h_phi = self.dphi
        I = (self.integrand_theta(0.,0,0,k)+self.integrand_theta(np.pi,99,49,k))/2 #se divide entre 4?
        for i in range(1,50): #range of theta
            for j in range(1,100): #range of phi
                I+=self.integrand_theta(i*h_theta,j,i,k)
        return I*h_theta*h_phi
    
    #Obtaining the expected value of f(theta)
    def expected_ftheta(self,f_given): #using the trapezoid method
        
        h_theta = self.dtheta
        h_phi = self.dphi
        I = (self.integrand_ftheta(0.,0,0,f_given)+self.integrand_ftheta(np.pi,99,49,f_given))/2 #se divide entre 4?
        for i in range(1,50): #range of theta
            for j in range(1,100): #range of phi
                I+=self.integrand_ftheta(i*h_theta,j,i,f_given)
        return I*h_theta*h_phi
    
    #Obtaining the expected value of phi**k
    def expected_phipower(self,k): #using the trapezoid method
        
        h_theta = self.dtheta
        h_phi = self.dphi
        I = (self.integrand_phi(0.,0,0,k)+self.integrand_phi(np.pi*2,99,49,k))/2 #se divide entre 4?
        for i in range(1,50): #range of theta
            for j in range(1,100): #range of phi
                I+=self.integrand_theta(j*h_phi,j,i,k)
        return I*h_theta*h_phi
    
    #Obtaining the expected value of f(phi)
    def expected_fphi(self,f_given): #using the trapezoid method
        
        h_theta = self.dtheta
        h_phi = self.dphi
        # F = np.outer(np.ones("self.theta"), f_given(self.phi))
        # integrand = self.prob * F * self.sin_theta * h_theta * h_phi
        # integrand.sum() - ...
        I = (self.integrand_fphi(0.,0,0,f_given)+self.integrand_fphi(np.pi*2,99,49,f_given))/2 #se divide entre 4?
        for i in range(1,50): #range of theta
            for j in range(1,100): #range of phi
                I+=self.integrand_fphi(j*h_phi,j,i,f_given)
        return I*h_theta*h_phi
    
    #Obtaining the expected value of f(theta,phi)
    def expected_f_theta_phi(self,f_given): #using the trapezoid method
        
        h_theta = self.dtheta
        h_phi = self.dphi
        # f(theta, phi) bucle -> matriz F
        # integrand = self.prob * F * self.sin_theta * h_theta * h_phi
        # integrand.sum() - ...
        I = (self.integrand_fthetaphi(0.,0.,0,0,f_given)+self.integrand_fthetaphi(np.pi*2,np.pi,99,49,f_given))/2 #se divide entre 4?
        for i in range(1,50): #range of theta
            for j in range(1,100): #range of phi
                I+=self.integrand_fthetaphi(j*h_phi,i*h_theta,j,i,f_given)
        return I*h_theta*h_phi



class real_ang_function(spherical_harmonic):
    """
    Class for the real angular function calculated by orbitals.py
    Parameters are in radians.
    
    Parameters
    ----------
    l : int
        Orbital angular momentum quantum number.
    m : int
        Magnetic quantum number.

    Attributes
    ----------
    theta : array
        Polar angle in radians.
    phi : array
        Azimutal angle in radians.
        
    Methods
    -------
    Plot_prob()
        Plot the spherical harmonic as a probability distribution per solid angle.
    Plot_phase()
        Plot the spherical harmonic as a complex function.
    """
    def __init__(self, l, m, part="Re"):
        # definition of class atributes
        self.l = l
        m = abs(m)
        self.m = m
        theta = np.linspace(0., np.pi, 50)
        self.theta = theta
        sin_theta = np.sin(theta)
        sin_theta[-1] = 0. # the last theta is pi, but sin_theta is not exactly 0
        phi = np.linspace(0., 2.*np.pi, 100)
        self.phi = phi
        
        if m==0 and part!="Re":
            print("For m=0, there is only real part.")
            part = "Re"
            self.part = part
        elif part=="Re" or part=="Im":
            self.part = part
        else:
            raise ValueError("The parameter part should be Re or Im.")
        
        # Computed results of ftheta() and fphi()
        self.ftheta_lm = ftheta(l, m, theta)       
        self.fphi_m = fphi(m, phi, part)
        
        #calculation of theta's probability distribution
        self.theta_dist = (self.ftheta_lm**2 * sin_theta).cumsum() #* np.pi / 49.
        self.theta_dist /= self.theta_dist[-1]
        
        #calculation of phi's probability distribution
        self.phi_dist = (self.fphi_m**2).cumsum() #* 2. * np.pi / 99.
        self.phi_dist /= self.phi_dist[-1]
        
        # Y values, their absolute values, the square of them and phase
        Y = np.outer(self.fphi_m, self.ftheta_lm)
        self.Y = Y
        module = abs(Y)
        self.module = module
        prob = Y**2
        self.prob = prob
        #prob_sin_theta = prob * sin_theta
        #self.prob_sin_theta = prob_sin_theta
        phase = np.angle(Y)
        self.phase = phase
        
        # Cartesian coordinates for r=1
        x = np.outer(np.cos(phi), sin_theta)
        y = np.outer(np.sin(phi), sin_theta)
        z = np.outer(np.ones_like(phi), np.cos(theta))
        
        # Starting zoom of the graph. Relies on the l variable to adjust the graph
        #side = 1.25 + l - np.abs(m)
        #self.camera = dict(eye=dict(x=side, y=side, z=1.25))

        # Cartesian projection of prob
        self.prob_x = prob * x
        self.prob_y = prob * y
        self.prob_z = prob * z
        
        # Cartesian projection of module
        self.module_x = module * x
        self.module_y = module * y
        self.module_z = module * z
        
    def evaluate(self, theta, phi):
        theta = np.array(theta)
        phi = np.array(phi)
        ftheta_lm = ftheta(self.l, self.m, theta)       
        fphi_m = fphi(self.m, phi, self.part)
        return ftheta_lm * fphi_m
    
    def get_phi(self, points=1):
        p = np.random.random(points)
        phi = np.interp(p, self.phi_dist, self.phi)
        return phi


class comb_ang_function(real_ang_function):
    """
    Class for linear combination of angular functions calculated by orbitals.py
    Parameters are in radians.
    
    Parameters
    ----------
    functions: list
        Parameters for the angular functions.
    coefficients: list
        Coefficients for the angular functions.

    Attributes
    ----------
    theta : array
            Polar angle in radians.
    phi : array
          Azimutal angle in radians.
        
    Methods
    -------
    Plot_prob()
        Plot the spherical harmonic as a probability distribution per solid angle.
    Plot_phase()
        Plot the spherical harmonic as a complex function.
    """
    
    def __init__(self, functions, coefficients):
        if len(functions)!=len(coefficients):
            raise ValueError(
                "The lists of functions and coefficients must have the same length.")

        # Normalization
        norm = np.array([abs(c)**2 for c in coefficients]).sum()
        norm = np.sqrt(norm)
        coefficients /= norm
        
        self.l = None
        self.m = None
        theta = np.linspace(0., np.pi, 50)
        self.theta = theta
        sin_theta = np.sin(theta)
        sin_theta[-1] = 0. # the last theta is pi, but sin_theta is not exactly 0
        phi = np.linspace(0., 2.*np.pi, 100)
        self.phi = phi
        Y = np.zeros((100, 50), dtype='complex128')
        check = []
        # Combination of spherical harmonics, i.e., list of (l, m)
        if all([len(f)==2 for f in functions]):
            # Calculate Y
            for f, c in zip(functions, coefficients):
                if f in check:
                    raise ValueError("The spherical harmonics must be different.")
                check.append(f)
                sh = spherical_harmonic(*f)
                Y += c * sh.Y

        # Combination of real angular functions, i.e., list of (l, m, part)
        elif all([len(f)==3 for f in functions]):
            # Check if part="Re" for m=0
            for i, (l, m, part) in enumerate(functions):
                if m==0 and part!="Re":
                    part = "Re"
                    functions[i] = (l, m, part)
            # Calculate Y
            for f, c in zip(functions, coefficients):
                if f in check:
                    raise ValueError("The angular functions must be different.")
                check.append(f)
                sh = real_ang_function(*f)
                Y += c * sh.Y

        else:
            raise ValueError(
                "All the elements of functions must in the format either (l, m) or (l, m, part).")

        self.Y = Y
        module = np.abs(Y)
        self.module = module
        prob = module**2
        self.prob = prob
        prob_sin_theta = prob * sin_theta
        self.prob_sin_theta = prob_sin_theta
        phase = np.angle(Y)
        self.phase = phase
        
        # auxiliary
        phi_dist_t = np.cumsum(prob_sin_theta, axis=0) # cumsum over phi
        norm_t = phi_dist_t[-1].copy() # sum over phi
        # probability distribution of theta, averaged over phi
        theta_dist = norm_t.cumsum() # sum over phi and cumsum over theta
        norm = theta_dist[-1] # sum over theta and phi
        theta_dist /= norm
        self.theta_dist = theta_dist
        # probability distribution of phi, averaged over theta
        phi_dist = np.sum(phi_dist_t, axis=1) # cumsumm over phi and sum over theta
        phi_dist /= norm
        self.phi_dist = phi_dist
        # probability distribution of phi for each theta value
        norm_t[norm_t==0.] = 1. # prob_sin_theta=0 for some theta values
        phi_dist_t /= norm_t
        self.phi_dist_t = phi_dist_t

        # Cartesian coordinates for r=1
        x = np.outer(np.cos(phi), sin_theta)
        y = np.outer(np.sin(phi), sin_theta)
        z = np.outer(np.ones_like(phi), np.cos(theta))
        
        # Starting zoom of the graph. Relies on the l variable to adjust the graph
        #side = 1.25 + self.l - self.m
        #self.camera = dict(eye=dict(x=side, y=side, z=1.25))

        # Cartesian projection of prob
        self.prob_x = prob * x
        self.prob_y = prob * y
        self.prob_z = prob * z
        
        # Cartesian projection of module
        self.module_x = module * x
        self.module_y = module * y
        self.module_z = module * z
        
#    def check_norm(self):
#        d_th = np.pi/49
#        d_phi = 2*np.pi/100
#        
#        dist_aux = self.prob_sin_theta*d_th*d_phi
#        suma = dist_aux.sum()
#        print(suma)
        
    def evaluate(self, theta, phi):
        interp = RegularGridInterpolator((self.phi, self.theta), self.Y,
                                     bounds_error=False, fill_value=None, method="quintic")
        Y = interp((phi, theta))
        return 1.*Y
    
    def get_angles(self, points=1):
        theta = self.get_theta(points) # 0 < theta < pi
        d_th = np.pi / 49.
        #index = (theta/d_th).astype(int) # 0 <= index <= 48 # rounded down
        index = (np.rint(theta/d_th)).astype(int) # nearest int
        phi_dist_t = self.phi_dist_t
        phi = np.zeros(points)
        p = np.random.random(points)

        for point, (i, th_point, p_point) in enumerate(zip(index, theta, p)):
            # distribution taken for the nearest theta
            phi_dist = phi_dist_t[:, i]
            phi[point] = np.interp(p_point, phi_dist, self.phi)
            # interpolated distribution from the lower and higher theta values
            #phi_dist = phi_dist_t[:, i].copy
            #delta_dist = phi_dist_t[:, i+1] - phi_dist # i <=48
            #phi_dist += delta_dist * (th_point - self.theta[i]) / d_th
            #phi_dist /= phi_dist[-1]
            #phi[point] = np.interp(p_point, phi_dist, self.phi)
        return theta, phi


