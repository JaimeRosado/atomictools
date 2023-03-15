#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#import plotly.express as px
import plotly.graph_objects as go
#import plotly.io as pio
#from plotly.subplots import make_subplots
import numpy as np
import atomictools as at
#pio.renderers.default='iframe'

def sph_to_cart(r, theta, phi):
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return x, y, z

def cart_to_sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan(y/x)
    return r, theta, phi

class orbital_hydrog():
    def __init__(self, n, l, m, part=None, Z=1, mu=1.):
        self.n = n
        self.l = l
        self.m = m
        self.Z = Z
        self.mu = mu
        
        self.R = at.R_hydrog(n, l, Z, mu)
        if part is None:
            self.part = part
            self.Y = at.spherical_harmonic(l, m)
        else:
            if m==0 and part!="Re":
                print("For m=0, there is only real part.")
                part = "Re"
                self.part = part
            elif part=="Re" or part=="Im":
                self.part = part
            else:
                raise ValueError("The parameter part should be Re or Im.")
            self.Y = at.real_ang_function(l, m, part)
        
        rmax = self.R.rmax
        self.rmax = rmax
        self.r = self.R.r
        self.r_dist = self.R.P2.cumsum() * rmax / (self.R.npt-1)
        self.r_dist /= self.r_dist[-1]
        
        self.theta = self.Y.theta
        self.theta_dist = (self.Y.ftheta_lm**2 * np.sin(self.theta)).cumsum() * np.pi/49.
        self.theta_dist /= self.theta_dist[-1]
        
        self.phi = self.Y.phi
        if part is not None:
            self.phi_dist = (self.Y.fphi_m**2).cumsum() * 2. * np.pi / 99.
            self.phi_dist /= self.phi_dist[-1]
        
        self.x, self.y, self.z = np.mgrid[-rmax:rmax:40j, -rmax:rmax:40j, -rmax:rmax:40j]
        r, theta, phi = cart_to_sph(self.x, self.y, self.z)

        self.prob = np.abs(self.evaluate(r, theta, phi))**2

    def evaluate(self, r, theta, phi):
        R = self.R.evaluate(r)
        Y = self.Y.evaluate(theta, phi)
        return R * Y

    def get_r(self, points=1):
        p = np.random.random(points)
        r = np.interp(p, self.r_dist, self.r)
        return r
    
    def get_theta(self, points=1):
        p = np.random.random(points)
        theta = np.interp(p, self.theta_dist, self.theta)
        return theta
    
    def get_phi(self, points=1):
        if self.part is None:
            phi = np.random.random(points)*2.*np.pi
            return phi
        else:
            p = np.random.random(points)
            phi = np.interp(p, self.phi_dist, self.phi)
            return phi
        
    def plot_volume(self):
        
        min_val = np.min(self.prob)
        max_val = np.max(self.prob)
        
        fig = go.Figure(data=go.Volume(
            x=self.x.flatten(),
            y=self.y.flatten(),
            z=self.z.flatten(),
            value=self.prob.flatten(),
            opacity=0.1,
            isomin=0.002*max_val,
            isomax=0.99*max_val,
            surface_count=100,
            colorscale='viridis'
            ))
        fig.show()
        
    def plot_scatter(self, points=100000, op=0.01):
        if points>6.7e5:
            print("The maximum number of points is 6.7e5.")
            points = 6.7e5
        points = int(points)
        r = self.get_r(points)
        theta = self.get_theta(points)
        phi = self.get_phi(points)
        rmax = self.rmax
        
        x, y, z = sph_to_cart(r, theta, phi)
        fig=go.Figure(data=go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(size=1., color = 'red'),
            opacity=op
            ))
        fig.update_layout(
            #showlegend=False,
            scene = dict(
                xaxis = dict(nticks=4, range=[-rmax, rmax]),
                yaxis = dict(nticks=4, range=[-rmax, rmax]),
                zaxis = dict(nticks=4, range=[-rmax, rmax]),
                aspectmode = 'cube'
            )
        )
        fig.show()
        
        

class orbital(orbital_hydrog):
    def __init__(self, f_rad, f_ang):
        
        l=f_rad.l
        m=f_ang.m
        Z=f_rad.Z
        
        
        self.R = f_rad
        rmax = self.R.rmax
        self.rmax = rmax
        self.r = self.R.r
        self.r_dist = self.R.P2.cumsum() * rmax / (self.R.npt-1)
        self.r_dist /= self.r_dist[-1]
        
        self.Y=f_ang
        self.theta = self.Y.theta
        
        if f_ang.switcher==True:
            self.theta_dist = (self.Y.ftheta_lm**2 * np.sin(self.theta)).cumsum() * np.pi/49.
            self.theta_dist /= self.theta_dist[-1]
        
        self.phi = self.Y.phi
        
        if f_ang.switcher==True:
            part=self.Y.part
            self.part=part
        
            if part is not None:
                self.phi_dist = (self.Y.fphi_m**2).cumsum() * 2. * np.pi / 99.
                self.phi_dist /= self.phi_dist[-1]
            
        self.x, self.y, self.z = np.mgrid[-rmax:rmax:40j, -rmax:rmax:40j, -rmax:rmax:40j]
        r, theta, phi = cart_to_sph(self.x, self.y, self.z)
        
        self.prob = np.abs(self.evaluate(r, theta, phi))**2
        
    def get_r(self, points=1):
            p = np.random.random(points)
            r = np.interp(p, self.r_dist, self.r)
            return r
    
    def get_theta(self, points=1):
            p = np.random.random(points)
            
            if self.Y.switcher == True:
                theta = np.interp(p, self.theta_dist, self.theta)
            elif self.Y.switcher == False:
                theta = np.interp(p, np.abs(self.theta_dist), self.theta)
            return theta
    
    def get_phi(self, points=1):
            if self.part is None:
                phi = np.random.random(points)*2.*np.pi
                return phi
            else:
                p = np.random.random(points)
                phi = np.interp(p, self.phi_dist, self.phi)
                return phi
        
        


