#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from cmath import phase
import scipy.special as spe
import numpy as np
pio.renderers.default='iframe'

def ftheta(l, m, theta):
    C = np.sqrt((2.*l+1.) * spe.factorial(l-m) / (4.*np.pi) / spe.factorial(l+m))
    legendre = spe.lpmv(m, l, np.cos(theta))
    return C * legendre

def fphi(m, phi):
    return np.exp(1j * m * phi)

class spherical_harmonics:
    
    def __init__(self, l, m):
        
        # definition of class atributes
        self.l = l
        self.m = m
        theta = np.linspace(0., np.pi, 50)
        self.theta = theta
        phi = np.linspace(0., 2.*np.pi, 100)
        self.phi = phi
        
        # Computed results of ftheta() and fphi()
        ftheta_lm = ftheta(l, m, theta)
        fphi_m = fphi(m, phi)
        
        # Y values, their absolute values, the square of them and phase
        Y = np.outer(fphi_m, ftheta_lm)
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

