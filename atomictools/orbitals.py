#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#import plotly.express as px
import plotly.graph_objects as go
#import plotly.io as pio
#from plotly.subplots import make_subplots
import numpy as np
import atomictools as at
from scipy.interpolate import interpn
#pio.renderers.default='iframe'

def sph_to_cart(r, theta, phi):
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return x, y, z

def cart_to_sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(x, y)
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
        self.xmin = -rmax
        self.xmax = rmax
        self.ymin = -rmax
        self.ymax = rmax
        self.zmin = -rmax
        self.zmax = rmax
        self.axis = np.mgrid[-rmax:rmax:40j]
        #self.r = self.R.r
        #self.theta = self.Y.theta
        #self.phi = self.Y.phi
        
        self.x, self.y, self.z = np.mgrid[-rmax:rmax:40j, -rmax:rmax:40j, -rmax:rmax:40j]
        r, theta, phi = cart_to_sph(self.x, self.y, self.z)
        
        self.orbital = self.evaluate(r, theta, phi)
        self.prob = np.abs(self.orbital)**2

    def evaluate(self, r, theta, phi):
        R = self.R.evaluate(r)
        Y = self.Y.evaluate(theta, phi)
        return R * Y
        
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

    def get_points(self, points=1):
        r = self.R.get_r(points)
        theta, phi = self.Y.get_angles(points)
        return sph_to_cart(r, theta, phi)

    def plot_scatter(self, points=10000, op=0.05):
        if points>6.7e5:
            print("The maximum number of points is 6.7e5.")
            points = 6.7e5
        points = int(points)
        
        x, y, z = self.get_points(points)
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
                xaxis = dict(nticks=4, range=[self.xmin, self.xmax]),
                yaxis = dict(nticks=4, range=[self.ymin, self.ymax]),
                zaxis = dict(nticks=4, range=[self.zmin, self.zmax]),
                aspectmode = 'data'
            )
        )
        fig.show()
        
        

class orbital(orbital_hydrog):
    def __init__(self, f_rad, f_ang):
        #l = f_rad.l
        #m = f_ang.m
        #Z = f_rad.Z      
        
        self.R = f_rad
        rmax = self.R.rmax
        self.rmax = rmax
        self.xmin = -rmax
        self.xmax = rmax
        self.ymin = -rmax
        self.ymax = rmax
        self.zmin = -rmax
        self.zmax = rmax
        #self.r = self.R.r

        self.Y = f_ang
        #self.theta = self.Y.theta
            
        self.x, self.y, self.z = np.mgrid[-rmax:rmax:40j, -rmax:rmax:40j, -rmax:rmax:40j]
        r, theta, phi = cart_to_sph(self.x, self.y, self.z)
        
        self.orbital = self.evaluate(r, theta, phi)
        self.prob = np.abs(self.orbital)**2



class hybrid_orbital(orbital_hydrog):
    def __init__(self, orbitals, coefficients):
        
        if len(orbitals)!=len(coefficients):
            raise ValueError(
                "The lists of functions and coefficients must have the same length.")
    
        rmax = max([orb.rmax for orb in orbitals])
        self.rmax = rmax
        self.xmin = -rmax
        self.xmax = rmax
        self.ymin = -rmax
        self.ymax = rmax
        self.zmin = -rmax
        self.zmax = rmax
        axis = np.mgrid[-rmax:rmax:40j]
        self.x_axis = axis
        self.y_axis = axis
        self.z_axis = axis
        delta = 2. * rmax / 39.
        self.delta_x = delta
        self.delta_y = delta
        self.delta_z = delta
        self.x, self.y, self.z = np.mgrid[-rmax:rmax:40j, -rmax:rmax:40j, -rmax:rmax:40j]
        
        orbital = np.zeros([40,40,40], dtype = 'complex128')
        
        for orb, c in zip(orbitals, coefficients):
            orbital += c * orb.orbital
            
        prob = np.abs(orbital)**2
        norm = prob.sum() * (delta)**3
        self.norm = norm
        prob /= norm
        self.prob = prob
        orbital /= np.sqrt(norm)
        self.orbital = orbital

        # probability distribution of z, averaged over x and y
        z_dist = prob.sum(axis=0).sum(axis=0).cumsum() # sum over x and y and cumsum over z
        norm = z_dist[-1].copy() # sum over x, y, z
        z_dist /= norm
        self.z_dist = z_dist
        # probability distribution of y for a given z, averaged over x
        y_dist_z = prob.sum(axis=0).cumsum(axis=0) # sum over x and cumsum over y
        norm = y_dist_z[-1].copy() # sum over x and y
        y_dist_z /= norm
        self.y_dist_z = y_dist_z
        # probability distribution of x for given y and z
        x_dist_yz = prob.cumsum(axis=0) # cumsum over x
        norm = x_dist_yz[-1].copy() # sum over x
        x_dist_yz /= norm
        self.x_dist_yz = x_dist_yz

    def evaluate(self, x, y, z):
        orb_val = interpn((self.x_axis, self.y_axis, self.z_axis), self.orbital, (x, y, z))
        print("The wave function value in [",x,",",y,",",z,"] is ", orb_val)
        prob_val = interpn((self.x_axis, self.y_axis, self.z_axis), self.prob, (x, y, z))
        print("The probability value in [",x,",",y,",",z,"] is ", prob_val)
        return 1.*orb_val, 1.*prob_val 

    def get_points(self, points=1):
        pz = np.random.random(points)
        z = np.interp(pz, self.z_dist, self.z_axis)

        index = (np.rint((z-self.zmax)/self.delta_z)).astype(int) # nearest int
        y_dist_z = self.y_dist_z
        y = np.zeros(points)
        py = np.random.random(points)
        x_dist_yz = self.x_dist_yz
        x = np.zeros(points)
        px = np.random.random(points)

        for point, (i, z_point, py_point, px_point) in enumerate(zip(index, z, py, px)):
            # distribution of y taken for the nearest z
            y_dist = y_dist_z[:,i].copy()
            y_point = np.interp(py_point, y_dist, self.y_axis)
            y[point] = y_point
            
            j = (np.rint((y_point-self.ymax)/self.delta_y)).astype(int) # nearest int
            # distribution of x taken for the nearest y, z
            x_dist = x_dist_yz[:,j,i].copy()
            x_point = np.interp(px_point, x_dist, self.x_axis)
            x[point] = x_point

        return x, y, z
    
    
class molecular_orbital(hybrid_orbital):
    def __init__(self, orbitals, coefficients, centers):

        if len(orbitals)!=len(coefficients):
            raise ValueError(
                "The lists of functions and coefficients must have the same length.")

        if len(orbitals)!=len(centers):
            raise ValueError(
                "The lists of functions and centers must have the same length.")
        
        rmax = [orb.rmax for orb in orbitals]
        self.rmax = max(rmax)
        xmin = min([-r+c[0] for r, c in zip(rmax, centers)])
        xmax = max([ r+c[0] for r, c in zip(rmax, centers)])
        ymin = min([-r+c[1] for r, c in zip(rmax, centers)])
        ymax = max([ r+c[1] for r, c in zip(rmax, centers)])
        zmin = min([-r+c[2] for r, c in zip(rmax, centers)])
        zmax = max([ r+c[2] for r, c in zip(rmax, centers)])

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        
        self.x_axis = np.mgrid[xmin:xmax:40j]
        self.y_axis = np.mgrid[ymin:ymax:40j]
        self.z_axis = np.mgrid[zmin:zmax:40j]
        
        self.delta_x = (xmax-xmin) / 39.
        self.delta_y = (ymax-ymin) / 39.
        self.delta_z = (zmax-zmin) / 39.
        
        x, y, z = np.mgrid[xmin:xmax:40j, ymin:ymax:40j, zmin:zmax:40j]
        self.x, self.y, self.z = x, y, z
        
        orbital = np.zeros([40,40,40], dtype = 'complex128')
                
        for orb, c, cents in zip(orbitals, coefficients, centers):
            x_aux = x - cents[0]
            y_aux = y - cents[1]
            z_aux = z - cents[2]
            r, theta, phi = cart_to_sph(x_aux, y_aux, z_aux)
            orbital += c * orb.evaluate(r, theta, phi)
            
        prob = np.abs(orbital)**2
        norm = prob.sum() * self.delta_x * self.delta_y * self.delta_z
        self.norm = norm
        prob /= norm
        self.prob = prob
        orbital /= np.sqrt(norm)
        self.orbital = orbital
        
        # probability distribution of z, averaged over x and y
        z_dist = prob.sum(axis=0).sum(axis=0).cumsum() # sum over x and y and cumsum over z
        norm = z_dist[-1].copy() # sum over x, y, z
        z_dist /= norm
        self.z_dist = z_dist
        # probability distribution of y for a given z, averaged over x
        y_dist_z = prob.sum(axis=0).cumsum(axis=0) # sum over x and cumsum over y
        norm = y_dist_z[-1].copy() # sum over x and y
        y_dist_z /= norm
        self.y_dist_z = y_dist_z
        # probability distribution of x for given y and z
        x_dist_yz = prob.cumsum(axis=0) # cumsum over x
        norm = x_dist_yz[-1].copy() # sum over x
        x_dist_yz /= norm
        self.x_dist_yz = x_dist_yz 
