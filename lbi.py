# -*- coding: utf-8 -*-
"""
Final de Estructura de la materia 1
Método de Lattice-Boltzmann incompresible para el problema de Poiseuille plano

Martín Carusso
2017
"""

import numpy as np
from numpy import linspace, zeros, ones, pi, sum
import matplotlib.pyplot as plt

#%% Definimos las dimesiones espaciales y temporales

N_x = 16; X = linspace(0,1,N_x)
N_y = 32; Y = linspace(0,2,N_y)
t_max = 300
q = 9
# frecuencia y tiempo de relajación
omega = 0.9
tau = 1/omega
nu = 1
# tomamos el paso temporal acorde al tiempo de relajación y la viscosidad
delta_x = 2/N_y
delta_t = (2*tau-1)/6 *(delta_x)**2/nu
c = delta_x/delta_t 

sigma = 5/12
lambdaa = 1/3
gamma = 1/12
# los parámetros sigma, lambda y gamma, que corresponden a cada i
params_slg = [-4*sigma, lambdaa, lambdaa, lambdaa, lambdaa, gamma, gamma, gamma, gamma ] 
w = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36] # los pesos

# inicializamos las funciones de distribución como arrays
g_old = zeros((N_x, N_y, q)) 
g_new = zeros((N_x, N_y, q)) 
g_eq = zeros((N_x, N_y, q)) 

# condiciones iniciales para la presion y velocidad
p_inlet = 1.1
p_outlet = 1.0
pres = ones((N_x, N_y))*1.05
pres[:,0] = p_inlet
pres[:,N_y-1] = p_outlet
u_x = zeros((N_x, N_y))
u_y = zeros((N_x, N_y))


def generar_d2q9():
	"""
	Genera un array con los vectores del conjunto d2q9
	como filas.
	"""
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
	"""
	Genera la un array con la presión en cada punto
	a partir de la distribución g, los parámetros sigma, c y w, 
	las condiciones de contorno p_inlet y p_outlet y 
	las componentes de velocidad u_x u_y
	"""
    N_y = g.shape[1]

    pres = c**2/(4*sigma)*( sum(g[:,:,1:], axis = 2) + s(0, u_x, u_y, w, c) )
    pres[:,0] = p_inlet
    pres[:,N_y-1] = p_outlet
    return pres

def generar_vel(g, c):
	"""
	Genera las componentes de la velocidad u_x, u_y en cada punto
	a partir de la distribución g y el parámetro c.
	"""
    u_x = c*( (g[:,:,1] + g[:,:,5] + g[:,:,8]) - (g[:,:,3] + g[:,:,6] + g[:,:,7]) )
    u_y = c*( (g[:,:,2] + g[:,:,5] + g[:,:,6]) - (g[:,:,4] + g[:,:,7] + g[:,:,8]) )
    return u_x, u_y

#%% Distribución de equilibrio a partir de rho y u

def s(i, u_x, u_y, w, c):
	"""
	Genera la función s_i usada para calcular la distribución de 
	equilibrio en Guo et. al, ec (2.4)
	"""
    e = generar_d2q9()
    si = w[i]*( 3*(e[i,0]*u_x + e[i,1]*u_y)/c + 4.5*((e[i,0]*u_x + e[i,1]*u_y)**2)/(c**2) - 1.5*(u_x**2 + u_y**2)/(c**2) )
    return si

def generar_eq(p, u_x, u_y, slg, c, g_eq):
    """
    A partir de la presion y u_x, u_y genera valores 
	para la distribución de equilibrio g_eq según la ec. (2.9)
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


def pressure_left(g_old, g_new, slg, p_inlet, u_x, u_y, w, c, omega):
	"""
	Actualiza las distribuciones g_i en el borde izquierdo del recinto
	imponiendo la presión dada por p_inlet, según la extrapolación de Guo.
	"""

    N_x = g_old.shape[0]
   
    g_bar = zeros((N_x,9))
    for k in range(N_x):
        for i in range(9):
            g_bar[k,i] = slg[i]*p_inlet/(c**2) + s(i, u_x[k,1], u_y[k,1], w, c)
        g_new[k,0,:] = g_bar[k,:] + (1-omega)*(g_old[k,1,:] - g_eq[k,1,:])


def pressure_right(g_old, g_new, slg, p_outlet, u_x, u_y, w, c, omega):
	"""
	Actualiza las distribuciones g_i en el borde derecho del recinto
	imponiendo la presión dada por p_outlet, según la extrapolación de Guo.
	"""
    
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
	Genera la evolución temporal del fluido, realizando todos los pasos del algoritmo.
	Devuelve los arrays de presión y velocidades u_x, u_y actualizados.
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
    
#%% Acá la iteración temporal (main del programa)

generar_eq(pres, u_x, u_y, params_slg, c, g_old)
for i in range(t_max):
    pres, u_x, u_y = time_evol(g_eq, g_old, g_new, pres, u_x, u_y, params_slg, w, c, omega, sigma, p_inlet, p_outlet)

            
#%% Ploteo

#%% Ploteo de los campos de velocidad y presión 
plt.figure()
yy, xx = np.meshgrid(Y,X)
cont = plt.contourf(yy, xx, pres, cmap = 'cool')
cbar = plt.colorbar(cont)
cbar.ax.set_ylabel('Presión', fontsize = 15)
plt.quiver(yy[::2, ::4], xx[::2, ::4], u_x[::2, ::4], u_y[::2, ::4], color = 'k')
plt.xlabel('Coordenada $x$', fontsize = 15)
plt.ylabel('Coordenada $y$', fontsize = 15)
plt.title('Campo de velocidades y presiones', fontsize = 20)
plt.show()

#%% Ploteo de los perfiles de velocidad y presión

plt.figure()
f = lambda x, delta_p, nu: delta_p/(4*nu)*x*(1-x) # solución teórica
for i in range(N_y):
    puntos = plt.scatter(X, u_x[:,i])
    plt.xlabel('Coordenada $y$', fontsize = 15)
    plt.ylabel('Velocidad $u_x$', fontsize = 15)
    plt.title('Perfil vertical de velocidades', fontsize = 20)
    plt.grid(True)
puntos = plt.scatter(X, u_x[:,i], label = 'Simulaciones')
plt.plot(X, f(X, p_inlet-p_outlet, nu), label = 'Teórico')
plt.legend()
plt.show()

plt.figure()
h = lambda x, p_in, delta_p: p_in - delta_p/2*x  # solución teórica         
for i in range(N_x):
    plt.scatter(Y, pres[i,:])
    plt.xlabel('Coordenada $x$', fontsize = 15)
    plt.ylabel('Presión $p$', fontsize = 15)
    plt.title('Perfil horizontal de presiones', fontsize = 20)
    plt.grid(True)
plt.scatter(Y, pres[i,:], label = 'Simulaciones')
plt.plot(Y, h(Y, p_inlet, p_inlet-p_outlet), label = 'Teórico')
plt.legend()
plt.show()
            
    
    

