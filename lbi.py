# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 04:11:42 2017

@author: martin
"""

import numpy as np
from numpy import linspace, arange, zeros, ones, pi
import matplotlib.pyplot as plt

#%% Definimos las dimesiones espaciales y temporales

N_x = 10; X = np.arange(N_x)
N_y = 10; Y = np.arange(N_y)
t_max = 5
q = 9
tau = 3.5
omega = 1/tau
delta_x = 1
delta_t = 1
c = delta_x/delta_t 

params_slg = [5/12, 1/3, 1/12] # los parámetros sigma, lambda y gamma
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

e = zeros((9,2))
for i in range(1,5):
    e[i,0] = np.cos((i-1)*pi/2)
    e[i,1] = np.sin((i-1)*pi/2)

for i in range(5,9):
    e[i,0] = np.sqrt(2)*(np.cos((i-5)*pi/2 + pi/4))
    e[i,1] = np.sqrt(2)*(np.sin((i-5)*pi/2 + pi/4))
    

#%% Generación de variables macroscópicas (moment update)

def generar_p(g, sigma, u_x, u_y, w, e, c):
    # en la sum venia el axis = 2
    pres = c**2/(4*sigma)*( np.sum(g[:,:,1:]) + s(0, u_x, u_y, w, e, c) )
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

def pressure_left(g_old, g_new, slg, p_inlet, u_x, u_y, w, e, c, omega):
    
    N_x = g_old.shape[0]
    N_y = g_old.shape[1]
    
    g_bar = zeros((N_x,9))
    for k in range(N_x):
         g_bar[k,0] = 4*slg[0]*p_inlet/(c**2) + s(0, u_x, u_y, w, e, c)[k,1]
         g_bar[k,1] = slg[1]*p_inlet/(c**2) + s(1, u_x, u_y, w, e, c)[k,1]
         g_bar[k,2] = slg[1]*p_inlet/(c**2) + s(2, u_x, u_y, w, e, c)[k,1]
         g_bar[k,3] = slg[1]*p_inlet/(c**2) + s(3, u_x, u_y, w, e, c)[k,1]
         g_bar[k,4] = slg[1]*p_inlet/(c**2) + s(4, u_x, u_y, w, e, c)[k,1]
         g_bar[k,5] = slg[2]*p_inlet/(c**2) + s(5, u_x, u_y, w, e, c)[k,1]
         g_bar[k,6] = slg[2]*p_inlet/(c**2) + s(6, u_x, u_y, w, e, c)[k,1]
         g_bar[k,7] = slg[2]*p_inlet/(c**2) + s(7, u_x, u_y, w, e, c)[k,1]
         g_bar[k,8] = slg[2]*p_inlet/(c**2) + s(8, u_x, u_y, w, e, c)[k,1]
         
         g_new[k,0,:] = g_bar[k,:] + (1-omega)*(g_new[k,1,:] - g_eq[k,1,:])


def pressure_right(g_old, g_new, slg, p_outlet, u_x, u_y, w, e, c, omega):
    
    N_x = g_old.shape[0]
    N_y = g_old.shape[1]
    
    g_bar = zeros((N_x,9))
    for k in range(N_x):
         g_bar[k,0] = 4*slg[0]*p_outlet/(c**2) + s(0, u_x, u_y, w, e, c)[k,N_y-2]
         g_bar[k,1] = slg[1]*p_outlet/(c**2) + s(1, u_x, u_y, w, e, c)[k,N_y-2]
         g_bar[k,2] = slg[1]*p_outlet/(c**2) + s(2, u_x, u_y, w, e, c)[k,N_y-2]
         g_bar[k,3] = slg[1]*p_outlet/(c**2) + s(3, u_x, u_y, w, e, c)[k,N_y-2]
         g_bar[k,4] = slg[1]*p_outlet/(c**2) + s(4, u_x, u_y, w, e, c)[k,N_y-2]
         g_bar[k,5] = slg[2]*p_outlet/(c**2) + s(5, u_x, u_y, w, e, c)[k,N_y-2]
         g_bar[k,6] = slg[2]*p_outlet/(c**2) + s(6, u_x, u_y, w, e, c)[k,N_y-2]
         g_bar[k,7] = slg[2]*p_outlet/(c**2) + s(7, u_x, u_y, w, e, c)[k,N_y-2]
         g_bar[k,8] = slg[2]*p_outlet/(c**2) + s(8, u_x, u_y, w, e, c)[k,N_y-2]
         
         g_new[k,N_y-1,:] = g_bar[k,:] + (1-omega)*(g_new[k,N_y-2,:] - g_eq[k,N_y-2,:])
         
    
def pressure_left2(g_old, g_new, e, w, u_x, u_y, p_inlet):
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
        g_new[i,0,7] = g_old[i-1,1,7]
    
    # Ahora los que "rebotan" contra el borde   
    for i in range(1,N_x-1):
        # La velocidad en la pared
        u_wx = u_x[i,0] + 0.5*(u_x[i,0] - u_x[i,1])
        u_wy = u_y[i,0] + 0.5*(u_y[i,0] - u_y[i,1])
        g_new[i,0,1] = -g_old[i,0,1] + 6*p_inlet*w[1]*(1 + (e[1,0]*u_wx + e[1,1]*u_wy)**2/(2/9) - (u_wx**2 + u_wy**2)/(2/3) )
        g_new[i,0,5] = -g_old[i,0,5] + 6*p_inlet*w[5]*(1 + (e[5,0]*u_wx + e[5,1]*u_wy)**2/(2/9) - (u_wx**2 + u_wy**2)/(2/3) )
        g_new[i,0,1] = -g_old[i,0,8] + 6*p_inlet*w[8]*(1 + (e[8,0]*u_wx + e[8,1]*u_wy)**2/(2/9) - (u_wx**2 + u_wy**2)/(2/3) )


def pressure_right2(g_old, g_new, e, w, u_x, u_y, p_outlet):
    """
    Realiza la advección para los puntos del borde derecho del recinto,
    con la condición de contorno de presión fija, según el método 'anti-bounce
    back' (libro LB ss 5.3.5.2)
    """
    
    N_x = g_old.shape[0]
    N_y = g_old.shape[1]
    
    # Primero las direcciones que advectan desde la izquerda hacia el borde derecho
    for i in range(1,N_x-1):
        g_new[i,N_y-1,0] = g_old[i,N_y-1,0]
        g_new[i,N_y-1,1] = g_old[i,N_y-2,1]
        g_new[i,N_y-1,2] = g_old[i+1,N_y-1,2]
        g_new[i,N_y-1,4] = g_old[i-1,N_y-1,4]
        g_new[i,N_y-1,5] = g_old[i+1,N_y-2,5]
        g_new[i,N_y-1,8] = g_old[i-1,N_y-2,8]
    
    # Ahora los que "rebotan" contra el borde   
    for i in range(1,N_x-1):
        # La velocidad en la pared
        u_wx = u_x[i,N_y-1] + 0.5*(u_x[i,N_y-1] - u_x[i,N_y-2])
        u_wy = u_y[i,N_y-1] + 0.5*(u_y[i,N_y-1] - u_y[i,N_y-2])
        g_new[i,N_y-1,3] = -g_old[i,N_y-1,3] + 6*p_outlet*w[3]*(1 + (e[3,0]*u_wx + e[3,1]*u_wy)**2/(2/9) - (u_wx**2 + u_wy**2)/(2/3) )
        g_new[i,N_y-1,6] = -g_old[i,N_y-1,6] + 6*p_outlet*w[6]*(1 + (e[6,0]*u_wx + e[6,1]*u_wy)**2/(2/9) - (u_wx**2 + u_wy**2)/(2/3) )
        g_new[i,N_y-1,7] = -g_old[i,N_y-1,7] + 6*p_outlet*w[7]*(1 + (e[7,0]*u_wx + e[7,1]*u_wy)**2/(2/9) - (u_wx**2 + u_wy**2)/(2/3) )


#%% Evolución temporal

def time_evol(g_eq, g_old, g_new, pres, u_x, u_y, params_slg, w, e, c, omega, p_inlet, p_outlet):
    """
    """
    # A partir de p y u, genero la g de equilibrio
    generar_eq(pres, u_x, u_y, e, params_slg, c, g_eq)
    
    # Hago la relajación por colisiones
    coll(g_old, g_eq, tau)
    
    # Hago la advección en el interior
    advect(g_old, g_new)
    
    # Hago la advección en los bordes con las condiciones de contorno
    bounce_sup(g_old, g_new)
    bounce_inf(g_old, g_new)
    #acá vienen los contornos para presion/vel
    pressure_left(g_old, g_new, params_slg, p_inlet, u_x, u_y, w, e, c, omega)
    pressure_right(g_old, g_new, params_slg, p_outlet, u_x, u_y, w, e, c, omega)
    
    # Actualizo la presión y velocidad macro
    u_xnew, u_ynew = generar_vel(g_new, c)
    pres_new = generar_p(g_new, params_slg[0], u_xnew, u_ynew, w, e, c)
    g_old = g_new
    
    return pres_new, u_xnew, u_ynew
#
    
#%% Acá el main

generar_eq(pres, u_x, u_y, e, params_slg, c, g_old)
for i in range(12):
    pres, u_x, u_y = time_evol(g_eq, g_old, g_new, pres, u_x, u_y, params_slg, w, e, c, omega, p_inlet, p_outlet)

            
#%% Ploteo

plt.figure()
xx, yy = np.meshgrid(X,Y)
plt.quiver(xx, yy, u_x, u_y)
            
            
            
            
            
    
    

