# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 04:11:42 2017

@author: martin
"""

import numpy as np
from numpy import linspace, zeros, ones, pi, sum
import matplotlib.pyplot as plt

#%% Definimos las dimesiones espaciales y temporales

N = 64
X = linspace(0,1,N)
Y = linspace(0,1,N)
t_max = 50
q = 9
#omega = 0.9
#tau = 1/omega
nu = 1
delta_x = 1/N
#delta_t = (2*tau-1)/6 *(delta_x)**2/nu
#c = delta_x/delta_t 

sigma = 5/12
lambdaa = 1/3
gamma = 1/12
params_slg = [-4*sigma, lambdaa, lambdaa, lambdaa, lambdaa, gamma, gamma, gamma, gamma ] # los parámetros sigma, lambda y gamma
w = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36] # los pesos

Re = 400
U_wall = 0.1

tau=3*N*U_wall/Re + 0.5
omega = 1/tau

delta_t = (2*tau-1)/6 *(delta_x)**2/nu
c = 1

g_old = zeros((N, N, q)) 
g_new = zeros((N, N, q)) 
g_eq = zeros((N, N, q)) 

pres = ones((N, N))

u_x = zeros((N, N))
#u_x[0,:] = U_wall
u_y = zeros((N, N))

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

def generar_p(g, sigma, u_x, u_y, w, c):
    # en la sum venia el axis = 2
    pres = c**2/(4*sigma)*( sum(g[:,:,1:], axis = 2) + s(0, u_x, u_y, w, c) )
    return pres

def generar_vel(g, c):
    u_x = c*( (g[:,:,1] + g[:,:,5] + g[:,:,8]) - (g[:,:,3] + g[:,:,6] + g[:,:,7]) )
    u_y = c*( (g[:,:,2] + g[:,:,5] + g[:,:,6]) - (g[:,:,4] + g[:,:,7] + g[:,:,8]) )
#    u_x[0,:] = U_wall
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
    N = g_old.shape[0]
    N = g_old.shape[1]
    
    for i in range(1,N-1):
        for j in range(1,N-1):
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

def bounce_sup_vel(g_old, g_new, U_wall, slg, pres, w, c, omega):
    """
    Realiza la advección para los puntos en el borde superior del recinto,
    incluyendo la condición de contorno de rebote, con velocidad en la pared
    """
    N = g_old.shape[1]
    
    g_dbar = zeros((N,9))
    for k in range(N):
        for i in range(9):
            g_dbar[k,i] = slg[i]*pres[1,k]/(c**2) + s(i, U_wall, 0, w, c) # ver si es s_i ó s_0
        g_new[0,k,:] = g_dbar[k,:] + (1-omega)*(g_old[1,k,:] - g_eq[1,k,:])
        
        
def bounce_back_mod_sup(g_old, g_new, U_wall, pres, w):
    """
    Realiza la advección para los puntos en el borde superior del recinto,
    incluyendo la condición de contorno de rebote, con velocidad en la pared
    """
    #sacado del guo p.126
    N = g_old.shape[1]
    for k in range(N):
        g_new[0,k,4] = g_old[0,k,2]
        g_new[0,k,7] = g_old[0,k,5] + 6*pres[0,k]/3*w[7]*(-1)*U_wall
        g_new[0,k,8] = g_old[0,k,6] + 6*pres[0,k]/3*w[8]*(1)*U_wall
    
    
def bounce_inf(g_old, g_new):
    """
    Realiza la advección para los puntos en el borde inferior del recinto,
    incluyendo la condición de contorno de rebote
    """
    N = g_old.shape[0]
    N = g_old.shape[1]
    
    piso = N - 1
    for j in range(1,N-1):
        g_new[piso,j,0] = g_old[piso,j,0]
        g_new[piso,j,1] = g_old[piso,j-1,1]
        g_new[piso,j,2] = g_old[piso,j,4]
        g_new[piso,j,3] = g_old[piso,j+1,3]
        g_new[piso,j,4] = g_old[piso-1,j,4]
        g_new[piso,j,5] = g_old[piso,j,7]
        g_new[piso,j,6] = g_old[piso,j,8]
        g_new[piso,j,7] = g_old[piso-1,j+1,7]
        g_new[piso,j,8] = g_old[piso-1,j-1,8]
        
    #en las esquinas izquierda inferior y derecha inferior (ss. 5.3.4.1)
    
    g_new[piso,0,1] = g_old[piso,0,3]
    g_new[piso,0,2] = g_old[piso,0,4]
    g_new[piso,0,5] = g_old[piso,0,7]
    
    g_new[piso,N-1,2] = g_old[piso,N-1,4]
    g_new[piso,N-1,3] = g_old[piso,N-1,1]
    g_new[piso,N-1,6] = g_old[piso,N-1,8]
    


def bounce_left(g_old, g_new):
    """
    Realiza la advección para los puntos en el borde izquierdo del recinto,
    incluyendo la condición de contorno de rebote
    """
    N = g_old.shape[0]
    N = g_old.shape[1]
    
    for i in range(1,N-1):
        g_new[i,0,0] = g_old[i,0,0]
        g_new[i,0,1] = g_old[i,0,3]
        g_new[i,0,2] = g_old[i+1,0,2]
        g_new[i,0,3] = g_old[i,1,3]
        g_new[i,0,4] = g_old[i-1,0,4]
        g_new[i,0,5] = g_old[i,0,7]
        g_new[i,0,6] = g_old[i+1,1,6]
        g_new[i,0,7] = g_old[i-1,1,7]
        g_new[i,0,8] = g_old[i,0,6]
        
        
def bounce_right(g_old, g_new):
    """
    Realiza la advección para los puntos en el borde izquierdo del recinto,
    incluyendo la condición de contorno de rebote
    """
    N = g_old.shape[0]
    N = g_old.shape[1]
    pared = N-1
    for i in range(1,N-1):
        g_new[i,pared,0] = g_old[i,pared,0]
        g_new[i,pared,1] = g_old[i,pared-1,1]
        g_new[i,pared,2] = g_old[i+1,pared,2]
        g_new[i,pared,3] = g_old[i,pared,1]
        g_new[i,pared,4] = g_old[i-1,pared,4]
        g_new[i,pared,5] = g_old[i+1,pared-1,5]
        g_new[i,pared,6] = g_old[i,pared,8]
        g_new[i,pared,7] = g_old[i,pared,5]
        g_new[i,pared,8] = g_old[i-1,pared-1,8]

         

#%% Evolución temporal

def time_evol(g_eq, g_old, g_new, pres, u_x, u_y, params_slg, w, c, omega, sigma, U_wall):
    """
    """
    # A partir de p y u, genero la g de equilibrio
    generar_eq(pres, u_x, u_y, params_slg, c, g_eq)
    
    # Hago la relajación por colisiones
    coll(g_old, g_eq, tau)
    
    # Hago la advección en el interior
    advect(g_old, g_new)
    
    # Hago la advección en los bordes con las condiciones de contorno
    #bounce_sup_vel(g_old, g_new, U_wall, params_slg, pres, w, c, omega)
    bounce_back_mod_sup(g_old, g_new, U_wall, pres, w)
    bounce_inf(g_old, g_new)
    bounce_left(g_old, g_new)
    bounce_right(g_old, g_new)
    
    # Actualizo la presión y velocidad macro
    u_xnew, u_ynew = generar_vel(g_new, c)
    pres_new = generar_p(g_new, sigma, u_xnew, u_ynew, w, c)
    g_old = g_new
    return pres_new, u_xnew, u_ynew
#
    
#%% Acá el main

generar_eq(pres, u_x, u_y, params_slg, c, g_old)
for i in range(t_max):
    pres, u_x, u_y = time_evol(g_eq, g_old, g_new, pres, u_x, u_y, params_slg, w, c, omega, sigma, U_wall)

            
#%% Ploteo

plt.figure()
xx, yy = np.meshgrid(X,Y)
plt.streamplot(xx, yy, u_x, u_y)

#plt.figure()
#plt.plot(X, u_x[:,N//2])
#
#plt.figure()            
#plt.plot(Y, u_y[N//2,:])
            
plt.figure()
plt.contour(xx, yy, (pres-1)/(U_wall**2))
            
    
    

