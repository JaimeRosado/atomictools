#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
#from plotly.subplots import make_subplots
from cmath import phase
import scipy.special as spe
import numpy as np
#pio.renderers.default='iframe'

def ftheta(l, m, theta):
    C = np.sqrt((2.*l+1.) * spe.factorial(l-m) / 2. / spe.factorial(l+m))
    legendre = spe.lpmv(m, l, np.cos(theta))
    return C * legendre

def fphi(m, phi, part=None):
    if part is None:
        return np.sqrt(1./(2.*np.pi)) * np.exp(1j * m * phi)
    elif part=="Re" or m==0:
        return np.sqrt(1./np.pi) * np.cos(m * phi)
    elif part=="Im":
        return np.sqrt(1./np.pi) * np.sin(m * phi)
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
        
        # Computed results of ftheta() and fphi()
        self.ftheta_lm = ftheta(l, m, theta)
        self.fphi_m = fphi(m, phi)
        
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
        side = 1.25 + l - np.abs(m)
        self.camera = dict(eye=dict(x=side, y=side, z=1.25))

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
    
        fig.update_layout(scene_aspectmode='data', scene_camera=self.camera)
  
        fig.show()
    
            
    def plot_phase(self):    
        fig = go.Figure(data=[go.Surface(x=self.module_x, y=self.module_y, z=self.module_z,
            surfacecolor=self.phase, colorscale='HSV', cmin=-np.pi, cmax=np.pi)])

        fig.update_traces(contours_z=dict(
              show=True, usecolormap=True,
              highlightcolor="limegreen",
              project_z=True))
        
        fig.update_layout(scene_aspectmode='data', scene_camera=self.camera)
  
        fig.show()

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
        side = 1.25 + l - np.abs(m)
        self.camera = dict(eye=dict(x=side, y=side, z=1.25))

        # Cartesian projection of prob
        self.prob_x = prob * x
        self.prob_y = prob * y
        self.prob_z = prob * z
        
        # Cartesian projection of module
        self.module_x = module * x
        self.module_y = module * y
        self.module_z = module * z
