#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import scipy.special as spe
import numpy as np
pio.renderers.default='iframe'

def radial(r, n, l, Z, mu):
    C = np.sqrt((2.*mu/n)**3 * spe.factorial(n-l-1) / (2.*n*spe.factorial(n+l)))
    rho = 2. * mu * Z * r / n
    laguerre = spe.assoc_laguerre(rho, n-l-1, 2*l+1)
    return C * np.exp(-rho/2.) * rho**l * laguerre

class R_hydrog:
    def __init__(self, n, l, Z=1, mu=1):
        self.n = n         
        self.l = l
        self.npt = 500
        self.Z = Z
        self.mu = mu

        # El límte de r se calcula iterativamente empezando por este valor inicial
        rm = 3. * self.n**2 / self.Z / self.mu
        rmax = np.array([])
        integral = np.array([])
        loop = True
        while loop:
            integ, r, R, P2 = self.calculate(rm)
            rmax = np.append(rmax, rm)
            integral = np.append(integral, integ)
            if integ<0.99:
                rm *= 1.1
            elif integ>0.999:
                rm *= 0.9
            else:
                loop = False
        self.r = r
        self.R = R
        self.R2 = R**2
        self.P = r * R
        self.P2 = P2
        
        self.rmax = rmax
        self.integral = integral

    def calculate(self, rm):
        Dr = rm / (self.npt-1)
        r = np.linspace(0., rm, self.npt)
        R = radial(r, self.n, self.l, self.Z, self.mu)
        P2 = (r * R)**2
        integ =  P2.sum() * Dr
        return integ, r, R, P2

    def plot_R(self):
        fig = go.Figure(data = go.Scatter(x = self.r, y = self.R))
        
        fig.update_layout(
           title={
             'text': "$$R\ \textit{vs}\ r$$",
             'y':0.9,
             'x':0.5,
             'xanchor': 'center',
             'yanchor': 'top'},
           xaxis_title="$$r (u.a)$$",
           yaxis_title="$$R$$")
        
        fig.show()
        
    def plot_P(self):
        fig = go.Figure(data = go.Scatter(x = self.r, y = self.P))
        
        fig.update_layout(
           title={
             'text': "$$P\ \textit{vs}\ r$$",
             'y':0.9,
             'x':0.5,
             'xanchor': 'center',
             'yanchor': 'top'},
           xaxis_title="$$r (u.a)$$",
           yaxis_title="$$P$$")
        
        fig.show()
        
    def plot_R2(self):
        fig = go.Figure(data = go.Scatter(x = self.r, y = self.R2))
        
        fig.update_layout(
           title={
             'text': "$$R^2\ \textit{vs}\ r$$",
             'y':0.9,
             'x':0.5,
             'xanchor': 'center',
             'yanchor': 'top'},
           xaxis_title="$$r (u.a)$$",
           yaxis_title="$$R^2$$")
        
        fig.show()
    
    def plot_P2(self):
        fig = go.Figure(data = go.Scatter(x = self.r, y = self.P2))
        
        fig.update_layout(
           title={
             'text': "$$P^2\ \textit{vs}\ r$$",
             'y':0.9,
             'x':0.5,
             'xanchor': 'center',
             'yanchor': 'top'},
           xaxis_title="$$r (u.a)$$",
           yaxis_title="$$P^2$$")
        
        fig.show()

