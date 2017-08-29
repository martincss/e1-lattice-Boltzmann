# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 04:11:42 2017

@author: martin
"""

import numpy as np
from numpy import linspace, zeros, ones, pi, sum
import matplotlib.pyplot as plt

#%% Definimos las dimesiones espaciales y temporales

N_x = 16; X = linspace(0,1,N_x)
N_y = 32; Y = linspace(0,2,N_y)
t_max = 300
q = 9
omega = 0.9
tau = 1/omega
nu = 1
delta_x = 2/N_y
delta_t = (2*tau-1)/6 *(delta_x)**2/nu
c = delta_x/delta_t 

sigma = 5/12
lambdaa = 1/3
gamma = 1/12
params_slg = [-4*sigma, lambdaa, lambdaa, lambdaa, lambdaa, gamma, gamma, gamma, gamma ] # los parámetros sigma, lambda y gamma
w = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36] # los pesos

g_old = zeros((N_x, N_y, q)) 
g_new = zeros((N_x, N_y, q)) 
g_eq = zeros((N_x, N_y, q)) 

p_inlet = 1.1
p_outlet = 1.0
pres = ones((N_x, N_y))*1.05
pres[:,0] = 1.1
pres[:,N_y-1] = 1
u_x = zeros((N_x, N_y))
u_y = zeros((N_x, N_y))
#u_y[1,:] = ones(N_y)

def generar_d2q9():
    e = zeros((9,2))
    for i in range(1,5):
        e[i,0] = np.cos((i-1)*pi/2)
        e[i,1] = np.sin((i-1)*pi/2)
    
    for i in range(5,9):
        e[i,0] = np.sqrt(2)*(np.cos((i-5)*pi/2 + pi/4))
        e[i,1] = np.sqrt(2)*(np.sin((i-5)*pi/2 + pi/4))
        
    return e
    

#%% Generación de variables macroscópicas (moment update)

def generar_p(g, sigma, u_x, u_y, w, c, p_inlet, p_outlet):
    N_y = g.shape[1]
    # en la sum venia el axis = 2
    pres = c**2/(4*sigma)*( sum(g[:,:,1:], axis = 2) + s(0, u_x, u_y, w, c) )
    pres[:,0] = p_inlet
    pres[:,N_y-1] = p_outlet
    return pres

def generar_vel(g, c):
    u_x = c*( (g[:,:,1] + g[:,:,5] + g[:,:,8]) - (g[:,:,3] + g[:,:,6] + g[:,:,7]) )
    u_y = c*( (g[:,:,2] + g[:,:,5] + g[:,:,6]) - (g[:,:,4] + g[:,:,7] + g[:,:,8]) )
    return u_x, u_y

#%% Distribución de equilibrio a partir de rho y u

def s(i, u_x, u_y, w, c):
    e = generar_d2q9()
    si = w[i]*( 3*(e[i,0]*u_x + e[i,1]*u_y)/c + 4.5*((e[i,0]*u_x + e[i,1]*u_y)**2)/(c**2) - 1.5*(u_x**2 + u_y**2)/(c**2) )
    return si

def generar_eq(p, u_x, u_y, slg, c, g_eq):
    """
    A partir de rho y u_x, u_y genera valores la distribución de equilibrio 
    g_eq según la página 93
    """
    for i in range(9):
        g_eq[:,:,i] = slg[i]*p/(c**2) + s(i, u_x, u_y, w, c)
    

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
            g_new[i,j,1] = g_old[i,j-1,1]
            g_new[i,j,2] = g_old[i+1,j,2]
            g_new[i,j,3] = g_old[i,j+1,3]
            g_new[i,j,4] = g_old[i-1,j,4]
            g_new[i,j,5] = g_old[i+1,j-1,5]
            g_new[i,j,6] = g_old[i+1,j+1,6]
            g_new[i,j,7] = g_old[i-1,j+1,7]
            g_new[i,j,8] = g_old[i-1,j-1,8]
    

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

# usar la old en estas? (en el termino de 1-w)
def pressure_left(g_old, g_new, slg, p_inlet, u_x, u_y, w, c, omega):

    N_x = g_old.shape[0]
   
    g_bar = zeros((N_x,9))
    for k in range(N_x):
        for i in range(9):
            g_bar[k,i] = slg[i]*p_inlet/(c**2) + s(i, u_x[k,1], u_y[k,1], w, c)
        g_new[k,0,:] = g_bar[k,:] + (1-omega)*(g_old[k,1,:] - g_eq[k,1,:])


def pressure_right(g_old, g_new, slg, p_outlet, u_x, u_y, w, c, omega):
    
    N_x = g_old.shape[0]
    N_y = g_old.shape[1]
    
    g_bar = zeros((N_x,9))
    for k in range(N_x):
        for i in range(9):
            g_bar[k,i] = slg[i]*p_outlet/(c**2) + s(i, u_x[k,N_y-2], u_y[k,N_y-2], w, c)      
        g_new[k,N_y-1,:] = g_bar[k,:] + (1-omega)*(g_old[k,N_y-2,:] - g_eq[k,N_y-2,:])
         

#%% Evolución temporal

def time_evol(g_eq, g_old, g_new, pres, u_x, u_y, params_slg, w, c, omega, sigma, p_inlet, p_outlet):
    """
    """
    # A partir de p y u, genero la g de equilibrio
    generar_eq(pres, u_x, u_y, params_slg, c, g_eq)
    
    # Hago la relajación por colisiones
    coll(g_old, g_eq, tau)
    
    #acá vienen los contornos para presion/vel
    pressure_left(g_old, g_new, params_slg, p_inlet, u_x, u_y, w, c, omega)
    pressure_right(g_old, g_new, params_slg, p_outlet, u_x, u_y, w, c, omega)
    
    # Hago la advección en el interior
    advect(g_old, g_new)
    
    # Hago la advección en los bordes con las condiciones de contorno
    bounce_sup(g_old, g_new)
    bounce_inf(g_old, g_new)
    
    
    # Actualizo la presión y velocidad macro
    u_xnew, u_ynew = generar_vel(g_new, c)
    pres_new = generar_p(g_new, sigma, u_xnew, u_ynew, w, c, p_inlet, p_outlet)
    g_old = g_new
    return pres_new, u_xnew, u_ynew
#
    
#%% Acá el main

generar_eq(pres, u_x, u_y, params_slg, c, g_old)
for i in range(t_max):
    pres, u_x, u_y = time_evol(g_eq, g_old, g_new, pres, u_x, u_y, params_slg, w, c, omega, sigma, p_inlet, p_outlet)

            
#%% Ploteo

plt.figure()
xx, yy = np.meshgrid(X,Y)
plt.quiver(yy, xx, u_x, u_y)

plt.figure()
for i in range(N_y):
    plt.plot(X, u_x[:,i])

plt.figure()            
for i in range(N_x):
    plt.plot(Y, pres[i,:])
            
            
    
    

