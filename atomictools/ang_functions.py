#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#import plotly.express as px
import plotly.graph_objects as go
#import plotly.io as pio
#from plotly.subplots import make_subplots
from cmath import phase
import scipy.special as spe
import numpy as np
import sys
#from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
#pio.renderers.default='iframe'
np.seterr(invalid='ignore')

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
    
    def __init__(self, l, m):
        # definition of class atributes
        self.l = l
        self.m = m
        theta = np.linspace(0., np.pi, 50)
        self.theta = theta
        phi = np.linspace(0., 2.*np.pi, 100)
        self.phi = phi
        self.part = None
        
        # Computed results of ftheta() and fphi()
        self.ftheta_lm = ftheta(l, m, theta)
        self.fphi_m = fphi(m, phi)
        
        # calculation of probability distributions
        self.theta_dist = (self.ftheta_lm**2 * np.sin(self.theta)).cumsum() * np.pi/49.
        self.theta_dist /= self.theta_dist[-1]
        
        # Y values, their absolute values, the square of them and phase
        Y = np.outer(self.fphi_m, self.ftheta_lm)
        self.Y = Y
        module = np.abs(Y)
        self.module = module
        prob = module**2
        self.prob = prob
        phase = np.angle(Y)
        self.phase = phase
        
        # Cartesian coordinates for r=1
        sin_theta = np.sin(theta)
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
        phi = np.random.random(points)*2.*np.pi
        return phi
    
    def get_angles(self, points):
        theta = self.get_theta(points)
        phi = self.get_phi(points)
        return theta, phi


class real_ang_function(spherical_harmonic):

    def __init__(self, l, m, part="Re"):
        # definition of class atributes
        self.l = l
        m = abs(m)
        self.m = m
        theta = np.linspace(0., np.pi, 50)
        self.theta = theta
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
        self.theta_dist = (self.ftheta_lm**2 * np.sin(self.theta)).cumsum() * np.pi/49.
        self.theta_dist /= self.theta_dist[-1]
        
        #calculation of phi's probability distribution
        self.phi_dist = (self.fphi_m**2).cumsum() * 2. * np.pi / 99.
        self.phi_dist /= self.phi_dist[-1]
        
        # Y values, their absolute values, the square of them and phase
        Y = np.outer(self.fphi_m, self.ftheta_lm)
        self.Y = Y
        module = abs(Y)
        self.module = module
        prob = Y**2
        self.prob = prob
        phase = np.angle(Y)
        self.phase = phase
        
        # Cartesian coordinates for r=1
        sin_theta = np.sin(theta)
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


class comb_ang_function(spherical_harmonic):
    
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
        phase = np.angle(Y)
        self.phase = phase
        
        #calculation of probability distributions
        theta_dist_aux = []
        phi_dist_aux = []

        d_phi = self.phi[1] - self.phi[0]
        d_theta = self.theta[1] - self.theta[0]

        for i in range(len(self.theta)):
            theta_dist_aux.append(sum(self.prob[:,i])*d_phi)

        for i in range(len(self.phi)):
            phi_dist_aux.append(sum(self.prob[i,:])*d_theta)

        theta_dist = np.array(theta_dist_aux).cumsum()
        phi_dist = np.array(phi_dist_aux).cumsum()

        theta_dist /= theta_dist[-1]
        phi_dist /= phi_dist[-1]
        
        self.theta_dist = theta_dist
        self.phi_dist = phi_dist

        # Cartesian coordinates for r=1
        sin_theta = np.sin(theta)
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

    def evaluate(self, theta, phi):
        interp = RegularGridInterpolator((self.phi, self.theta), self.Y,
                                     bounds_error=False, fill_value=None, method="quintic")
        Y = interp((phi, theta))
        return 1.*Y
    
    def get_phi(self, points=1):
        p = np.random.random(points)
        phi = np.interp(p, self.phi_dist, self.phi)
        return phi
    
    def get_angles(self, points):
        phis = []
        thetas = []

        for i in range(points):
            theta_index = np.array([])
            
            while np.size(theta_index)==0:

                theta_aux = self.get_theta()
                theta_index = np.where(abs(self.theta-theta_aux)<0.032)
                
            prob_dist = self.prob*np.sin(self.theta[theta_index])

            dist_phi_i = prob_dist[:,theta_index].cumsum()*2*np.pi/100
            dist_phi_i /= dist_phi_i[-1]

            point = 1
            p = np.random.random(point)
            phi_fin = float(np.interp(p, dist_phi_i, self.phi))
            phis.append(phi_fin)

            thetas.append(float(theta_aux))

        return thetas, phis
        


