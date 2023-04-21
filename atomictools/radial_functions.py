#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#import plotly.express as px
import plotly.graph_objects as go
#import plotly.io as pio
#from plotly.subplots import make_subplots
import scipy.special as spe
import numpy as np
from scipy.interpolate import interp1d
#pio.renderers.default='iframe'

def radial(r, n, l, Z, mu):
    # Evaluate the radial function R of a hydrogenic atom at r
    # r is in atomic units and can be an array
    # Z is the nuclear charge and mu is the reduced mass in atomic units
    C = np.sqrt((2.*mu*Z/n)**3 * spe.factorial(n-l-1) / (2.*n*spe.factorial(n+l)))
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

        # Calculate iteratively the upper limit of r such that
        # the integration of P^2 is between 0.9999 and 0.99999
        rm = 3. * self.n**2 / self.Z / self.mu
        rmax = np.array([])
        integral = np.array([])
        loop = True
        while loop:
            integ, r, R, P2 = self._calculate(rm)
            rmax = np.append(rmax, rm)
            integral = np.append(integral, integ)
            if integ<0.9999:
                rm *= 1.1
            elif integ>0.99999:
                rm *= 0.9
            else:
                loop = False
        self.rmax = rm
        self.r = r
        self.R = R
        self.R2 = R**2
        self.P = r * R
        self.P2 = P2

        # only for checking
        #self._rmax = rmax
        #self._integral = integral
        
        #calculation of probability distribution
        self.r_dist = self.P2.cumsum() * self.rmax / (self.npt-1)
        self.r_dist /= self.r_dist[-1]
        
    def get_r(self, points=1):
        p = np.random.random(points)
        r = np.interp(p, self.r_dist, self.r)
        return r    
    
    def _calculate(self, rm):
        Dr = rm / (self.npt-1)
        r = np.linspace(0., rm, self.npt)
        R = radial(r, self.n, self.l, self.Z, self.mu)
        P2 = (r * R)**2
        integ =  P2.sum() * Dr
        return integ, r, R, P2

    def evaluate(self, r):
        r = np.array(r)
        R = radial(r, self.n, self.l, self.Z, self.mu)
        return R

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


class R_num(R_hydrog):
    """
    Class for the radial function calculated by orbitals.py
    Parameters are in atomic units.
    
    Parameters
    ----------
    file : str
        Path of the ASCII file that stores the solution.

    Attributes
    ----------
    npt : int
        Number of points.
    r : array
        Radial distance in a.u. 
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

    def __init__(self, file):
        data_info = open(file)
        print("Imported data info: ")
        print()
        for i, line in enumerate(data_info):
            print(line)
            if i==3:
                model = line[0]
            if i>6:
                break
        data_info.close

        if model=='H':
            Z, L, E, rmc, npt = np.loadtxt(file, skiprows=5, unpack=True, max_rows=1)
        elif model=='S':
            Z, A, a1, a2, L, E, rmc, npt = np.loadtxt(file, skiprows=5, unpack=True, max_rows=1)
        elif model=='G':
            Z, N, H, D, L, E, rmc, npt = np.loadtxt(file, skiprows=5, unpack=True, max_rows=1)
        self.Z = Z
        self.l = L
        self.npt = npt
        self.rmax = rmc

        data_set = np.loadtxt(file, skiprows=8)
        r = data_set[:,0]
        P = data_set[:,1]
        R = P / r
        r = np.append(0., r)
        P = np.append(0., P)
        R = np.append(0., R)
        # Only s functions
        if L==0:
            R[0] = R[1]
        self.r = r
        self.P = P
        self.R = R
        self.R2 = R**2
        self.P2 = P**2
        
        #calculation of probability distribution
        d_r = self.rmax/len(r)
        self.r_linear = np.arange(0, self.rmax, d_r)
        interp = interp1d(self.r, self.P2, kind = 'cubic', bounds_error='false')
        self.P2_linear = interp(self.r_linear)
        self.r_dist = self.P2_linear.cumsum() * self.rmax / (self.npt-1)
        self.r_dist /= self.r_dist[-1]

    def evaluate(self, r):
        r = np.array(r)
        lnR1 = np.log(self.R[-1])
        lnR2 = np.log(self.R[-2])
        r1 = self.r[-1]
        r2 = self.r[-2]
        m = (lnR1 - lnR2) / (r1 - r2)
        interp = interp1d(self.r, self.R, fill_value="extrapolate")
        R = interp(r)
        r_out = r>self.rmax
        R[r_out] = np.exp(lnR1 + m*(r[r_out]-r1))
        return R




