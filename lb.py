# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 01:17:11 2017

@author: martin
"""
import numpy as np
from numpy import linspace, arange, zeros, ones

#%% Definimos las dimesiones espaciales y temporales

N = 10 
t_max = 5
q = 9
tau = 1

f_old = zeros((N, N, q)) 
f_new = zeros((N, N, q)) 
f_eq = zeros((N, N, q)) 

rho = ones((N, N))
u_x = zeros((N, N))
u_y = zeros((N, N))

#%% Generación de variables macroscópicas (moment update)

def generar_rho(f):
    rho = np.sum(f, axis = 2)
    return rho

def generar_vel(f, rho):
    u_x = ( (f[:,:,1] + f[:,:,5] + f[:,:,8]) - (f[:,:,3] + f[:,:,6] + f[:,:,7]) )/rho
    u_y = ( (f[:,:,2] + f[:,:,5] + f[:,:,6]) - (f[:,:,4] + f[:,:,7] + f[:,:,8]) )/rho
    return u_x, u_y

#%% Distribución de equilibrio a partir de rho y u

def generar_eq(rho, u_x, u_y, f_eq):
    """
    A partir de rho y u_x, u_y genera valores la distribución de equilibrio 
    f_eq según la página 93
    """
    u2 = u_x**2 + u_y**2
    f_eq[:,:,0] = (2*rho/9)*(2 - 3*u2)
    f_eq[:,:,1] = (rho/18)*(2 + 6*u_x + 9*u_x**2 - 3*u2)
    f_eq[:,:,2] = (rho/18)*(2 + 6*u_y + 9*u_y**2 - 3*u2)
    f_eq[:,:,3] = (rho/18)*(2 - 6*u_x + 9*u_x**2 - 3*u2)
    f_eq[:,:,4] = (rho/18)*(2 - 6*u_y + 9*u_y**2 - 3*u2)
    f_eq[:,:,5] = (rho/36)*(1 + 3*(u_x + u_y) + 9*u_x*u_y + 3*u2)    
    f_eq[:,:,6] = (rho/36)*(1 - 3*(u_x - u_y) - 9*u_x*u_y + 3*u2)    
    f_eq[:,:,7] = (rho/36)*(1 - 3*(u_x - u_y) + 9*u_x*u_y + 3*u2)    
    f_eq[:,:,8] = (rho/36)*(1 + 3*(u_x - u_y) - 9*u_x*u_y + 3*u2)    

#%% Relajación mediante colisiones

def coll(f, f_eq, tau):
    w = 1/tau; 
    f = (1-w)*f + w*f_eq     

#%% Advección


  
    
    
    
    
    

