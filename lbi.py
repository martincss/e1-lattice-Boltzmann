# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 04:11:42 2017

@author: martin
"""

import numpy as np
from numpy import linspace, arange, zeros, ones, pi
import matplotlib.pyplot as plt

#%% Definimos las dimesiones espaciales y temporales

N_x = 5; X = np.arange(N_x)
N_y = 5; Y = np.arange(N_y)
t_max = 5
q = 9
tau = 1.1
delta_x = 1
delta_t = 1
c = delta_x/delta_t 

params_slg = [5/12, 1/3, 1/12] # los parámetros sigma, lambda y gamma
w = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36] # los pesos

g_old = zeros((N_x, N_y, q)) 
g_new = zeros((N_x, N_y, q)) 
g_eq = zeros((N_x, N_y, q)) 

p = ones((N_x, N_y))
u_x = zeros((N_x, N_y))
u_y = zeros((N_x, N_y))
u_y[1,:] = ones(N_y)

e = zeros((9,2))
for i in range(1,5):
    e[i,0] = np.cos((i-1)*pi/2)
    e[i,1] = np.sin((i-1)*pi/2)

for i in range(5,9):
    e[i,0] = np.sqrt(2)*(np.cos((i-5)*pi/2 + pi/4))
    e[i,1] = np.sqrt(2)*(np.sin((i-5)*pi/2 + pi/4))
    

#%% Generación de variables macroscópicas (moment update)

def generar_p(g, sigma, u_x, u_y, w, e, c):
    pres = c**2/(4*sigma)*( np.sum(g[:,:,1:], axis = 2) + s(0, u_x, u_y, w, e, c) )
    return pres

def generar_vel(g, c):
    u_x = c*( (g[:,:,1] + g[:,:,5] + g[:,:,8]) - (g[:,:,3] + g[:,:,6] + g[:,:,7]) )
    u_y = c*( (g[:,:,2] + g[:,:,5] + g[:,:,6]) - (g[:,:,4] + g[:,:,7] + g[:,:,8]) )
    return u_x, u_y

#%% Distribución de equilibrio a partir de rho y u

def s(i, u_x, u_y, w, e, c):
    si = w[i]*( 3*(e[i,0]*u_x + e[i,1]*u_y)/c + 4.5*((e[i,0]*u_x + e[i,1]*u_y)**2)/(c**2) - 1.5*(u_x**2 + u_y**2)/(c**2) )
    return si

def generar_eq(p, u_x, u_y, e, slg, c, g_eq):
    """
    A partir de rho y u_x, u_y genera valores la distribución de equilibrio 
    g_eq según la página 93
    """
    g_eq[:,:,0] = -4*slg[0]*p/(c**2) + s(0, u_x, u_y, w, e, c)
    g_eq[:,:,1] = slg[1]*p/(c**2) + s(1, u_x, u_y, w, e, c)
    g_eq[:,:,2] = slg[1]*p/(c**2) + s(2, u_x, u_y, w, e, c)
    g_eq[:,:,3] = slg[1]*p/(c**2) + s(3, u_x, u_y, w, e, c)
    g_eq[:,:,4] = slg[1]*p/(c**2) + s(4, u_x, u_y, w, e, c)
    g_eq[:,:,5] = slg[2]*p/(c**2) + s(5, u_x, u_y, w, e, c)
    g_eq[:,:,6] = slg[2]*p/(c**2) + s(6, u_x, u_y, w, e, c)
    g_eq[:,:,7] = slg[2]*p/(c**2) + s(7, u_x, u_y, w, e, c)
    g_eq[:,:,8] = slg[2]*p/(c**2) + s(8, u_x, u_y, w, e, c)
    

#%% Relajación mediante colisiones

def coll(g, g_eq, tau):
    """
    Actualiza la g por relajación con el término de colisiones
    """
    omega = 1/tau; 
    g[:,:,:] = (1-omega)*g[:,:,:] + omega*g_eq[:,:,:]     

#%% Advección

def advect(g_old, g_new):
    """
    Realiza la advección para los puntos en el interior del recinto
    """
    N_x = g_old.shape[0]
    N_y = g_old.shape[1]
    
    for i in range(1,N_x-1):
        for j in range(1,N_y-1):
            g_new[i,j,0] = g_old[i,j,0]
            g_new[i,j,1] = g_old[i-1,j,1]
            g_new[i,j,2] = g_old[i,j-1,2]
            g_new[i,j,3] = g_old[i+1,j,3]
            g_new[i,j,4] = g_old[i,j+1,4]
            g_new[i,j,5] = g_old[i-1,j-1,5]
            g_new[i,j,6] = g_old[i+1,j-1,6]
            g_new[i,j,7] = g_old[i+1,j+1,7]
            g_new[i,j,8] = g_old[i-1,j+1,8]
    

#%% Condiciones de contorno

def bounce_sup(g_old, g_new):
    """
    Realiza la advección para los puntos en el borde superior del recinto,
    incluyendo la condición de contorno de rebote
    """
    N_y = g_old.shape[1]
    
    for j in range(1,N_y-1):
        g_new[0,j,0] = g_old[0,j,0]
        g_new[0,j,1] = g_old[0,j-1,1]
        g_new[0,j,2] = g_old[1,j,2]
        g_new[0,j,3] = g_old[0,j+1,3]
        g_new[0,j,4] = g_old[0,j,2]
        g_new[0,j,5] = g_old[1,j-1,5]
        g_new[0,j,6] = g_old[1,j-1,6]
        g_new[0,j,7] = g_old[0,j,5]
        g_new[0,j,8] = g_old[0,j,6]
    
    
def bounce_inf(g_old, g_new):
    """
    Realiza la advección para los puntos en el borde inferior del recinto,
    incluyendo la condición de contorno de rebote
    """
    N_x = g_old.shape[0]
    N_y = g_old.shape[1]
    
    piso = N_x - 1
    for j in range(1,N_y-1):
        g_new[piso,j,0] = g_old[piso,j,0]
        g_new[piso,j,1] = g_old[piso,j-1,1]
        g_new[piso,j,2] = g_old[piso,j,4]
        g_new[piso,j,3] = g_old[piso,j+1,3]
        g_new[piso,j,4] = g_old[piso-1,j,4]
        g_new[piso,j,5] = g_old[piso,j,7]
        g_new[piso,j,6] = g_old[piso,j,8]
        g_new[piso,j,7] = g_old[piso-1,j+1,7]
        g_new[piso,j,8] = g_old[piso-1,j-1,8]
        
def pressure_left(g_old, g_new):
    """
    Realiza la advección para los puntos del borde izquierdo del recinto,
    con la condición de contorno de presión fija, según el método 'anti-bounce
    back' (libro LB ss 5.3.5.2)
    """
    
    N_x = g_old.shape[0]
    N_y = g_old.shape[1]
    
    # Primero las direcciones que advectan desde la derecha hacia el borde izquierdo
    for i in range(1,N_x-1):
        g_new[i,0,0] = g_old[i,0,0]
        g_new[i,0,2] = g_old[i+1,0,2]
        g_new[i,0,3] = g_old[i,1,3]
        g_new[i,0,4] = g_old[i-1,0,4]
        g_new[i,0,6] = g_old[i+1,1,6]
        g_new[i,0,7] = g_old[i-1,1,0]
    
    # Ahora los que "rebotan" contra el borde
    u_w = 
    
    for i in range(1,N_x-1):
        g_new[i,0,1] = -g_old[i,0,3] + 6*p_inlet*w[1]*(1 + )



def pressure_period_inlet():
    
    N_x = g_old.shape[0]
    N_y = g_old.shape[1]
    
    g_ext = np.zeros((N_y, 9))
    pass





    
#%% Evolución temporal

def time_evol(g_eq, g_old, g_new, p, u_x, u_y, params_slg, w, e, c):
    """
    """
    # A partir de p y u, genero la g de equilibrio
    generar_eq(p, u_x, u_y, e, params_slg, c, g_eq)
    
    # Hago la relajación por colisiones
    coll(g_old, g_eq, tau)
    
    # Hago la advección en el interior
    advect(g_old, g_new)
    
    # Hago la advección en los bordes con las condiciones de contorno
    bounce_sup(g_old, g_new)
    bounce_inf(g_old, g_new)
    #acá vienen los contornos para presion/vel
    
    # Actualizo la presión y velocidad macro
    p = generar_p(g_new, params_slg[0], u_x, u_y, w, e, c)
    u_x, u_y = generar_vel(g_new, c)
    g_old = g_new
#
    
#%% Acá el main

generar_eq(p, u_x, u_y, e, params_slg, c, g_old)
time_evol(g_eq, g_old, g_new, p, u_x, u_y, params_slg, w, e, c)

            
#%% Ploteo

plt.figure()
xx, yy = np.meshgrid(X,Y)
plt.quiver(xx, yy, u_x, u_y)
            
            
            
            
            
    
    

