#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import scipy.special as spe
import numpy as np
pio.renderers.default='iframe'

def _R_hydrog(r, n, l, Z, mu):
    # Evaluate the radial function R of a hydrogenic atom at r
    # r is in atomic units and can be an array
    # Z is the nuclear charge and mu is the reduced mass in atomic units
    C = np.sqrt((2.*mu/n)**3 * spe.factorial(n-l-1) / (2.*n*spe.factorial(n+l)))
    rho = 2. * mu * Z * r / n
    laguerre = spe.assoc_laguerre(rho, n-l-1, 2*l+1)
    return C * np.exp(-rho/2.) * rho**l * laguerre

class R_hydrog:
    """
    Class for the radial function of a hydrogenic atom.
    Parameters are in atomic units.
    
    Parameters
    ----------
    n : int
        Principal quantum number.
    l : int
        Orbital angular momentum quantum number.
    Z : int, default 1
        Nuclear charge in a.u.
    mu : float, default 1.0
        Reduced mass in a.u.
    
    Attributes
    ----------
    npt : int
        Number of points.
    r : array
        Radial distance in a.u.
    rmax : 
    R : array
        Radial function.
    R2 : array
        R^2.
    P : array
        r * R.
    P2 : arra
        P^2.
        
    Methods
    -------
    Plot_R()
        Plot the radial function R.
    Plot_R2()
        Plot R^2.
    Plot_P()
        Plot P = r * R.
    Plot_P2()
        Plot P^2.
    """
    def __init__(self, n, l, Z=1, mu=1.):
        self.n = n         
        self.l = l
        self.npt = 500
        self.Z = Z
        self.mu = mu

        # Calculate iteratively the upper limit of r
        # such that the integration of P^2 is between 0.99 and 0.999
        rm = 3. * self.n**2 / self.Z / self.mu
        rmax = np.array([])
        integral = np.array([])
        loop = True
        while loop:
            integ, r, R, P2 = self._calculate(rm)
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

        # only for checking
        self._rmax = rmax
        self._integral = integral

    def _calculate(self, rm):
        Dr = rm / (self.npt-1)
        r = np.linspace(0., rm, self.npt)
        R = _R_hydrog(r, self.n, self.l, self.Z, self.mu)
        P2 = (r * R)**2
        integ =  P2.sum() * Dr
        return integ, r, R, P2

    def plot_R(self):
        fig = go.Figure(data = go.Scatter(x = self.r, y = self.R))
        
        fig.update_layout(
           title={
             'text': "$$R\ vs\ r$$",
             'y':0.9,
             'x':0.5,
             'xanchor': 'center',
             'yanchor': 'top'},
           xaxis_title="$$r (a.u.)$$",
           yaxis_title="$$R$$")
        
        fig.show()
        
    def plot_P(self):
        fig = go.Figure(data = go.Scatter(x = self.r, y = self.P))
        
        fig.update_layout(
           title={
             'text': "$$P\ vs\ r$$",
             'y':0.9,
             'x':0.5,
             'xanchor': 'center',
             'yanchor': 'top'},
           xaxis_title="$$r (a.u.)$$",
           yaxis_title="$$P$$")
        
        fig.show()
        
    def plot_R2(self):
        fig = go.Figure(data = go.Scatter(x = self.r, y = self.R2))
        
        fig.update_layout(
           title={
             'text': "$$R^2\ vs\ r$$",
             'y':0.9,
             'x':0.5,
             'xanchor': 'center',
             'yanchor': 'top'},
           xaxis_title="$$r (a.u.)$$",
           yaxis_title="$$R^2$$")
        
        fig.show()
    
    def plot_P2(self):
        fig = go.Figure(data = go.Scatter(x = self.r, y = self.P2))
        
        fig.update_layout(
           title={
             'text': "$$P^2\ vs\ r$$",
             'y':0.9,
             'x':0.5,
             'xanchor': 'center',
             'yanchor': 'top'},
           xaxis_title="$$r (a.u.)$$",
           yaxis_title="$$P^2$$")
        
        fig.show()


##### Para implementar en este módulo
# Una clase hija que tome como parámetro de entrada un fichero
# con un orbital numérico (resultado del programa de las prácticas de cálculo).
# Esta clase tiene que generar los mismos atributos npt, R, P, etc.
# Con eso, los métodos gráficos herededos de R_hidrog deben funcionar perfectamente.
# La clase tiene que comprobar que la integral de P^2 es superior a 0.99.
# Si no, el rango en r debe alargarse hasta que se cumpla y dar un aviso de esto.
# Para ello, se hace una extrepolación exponencial de la función a partir de los últimos dos puntos.

# Dejo este código por si es útil
def import_radius(self, route):
    data_info=open(route)
    i=0
    print("Imported data info: ")
    print()
    for linea in data_info:
        print(linea)
        i+=1
        if (i>=7):
            break
    data_set=np.loadtxt(route, skiprows=8)
    return data_set
