# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Paramètres
r = 0.01  # Rayon de l'oeuf en mètres
T0 = 50.0  # Température initiale de l'oeuf en degrés Celsius
Tf = -50.0 # Température de l'eau en degrés Celsius
D = 1.40E-7  # Diffusivité thermique en mètres carrés par seconde
N = 100  # Nombre de points de discrétisation
dx = 2 * r / (N - 1)  # Pas d'espace
dt = dx**2 / (4 * D)  # Pas de temps
t_final = 200  # Durée totale de la simulation en secondes

# Initialisation de la solution
T = np.ones((N, N)) * T0

# Conditions aux limites
for j in range(N):
    for i in range(N):
        x = i * dx - r
        y = j * dx - r
        if x**2 + y**2 > r**2:
            T[j, i] = Tf

# Fonction pour mettre à jour la température
def step(T):
    T_new = np.copy(T)
    for j in range(1, N-1):
        for i in range(1, N-1):
            laplacian = (T[j, i+1] + T[j, i-1] + T[j+1, i] + T[j-1, i] - 4*T[j, i]) / dx**2
            T_new[j, i] = T[j, i] + D * dt * laplacian
            x = i * dx - r
            y = j * dx - r
            dist = np.sqrt(x**2 + y**2)
            if dist > r:
                T_new[j, i] = Tf
    return T_new

# Animation
fig, ax = plt.subplots()
im = ax.imshow(T, cmap='jet', vmin=Tf, vmax=T0, extent=[-r, r, -r, r])
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Température (°C)')
ax.set_title('Refroidissement d\'un œuf dur')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')

def animate(frame):
    global T
    T = step(T)
    im.set_array(T)
    return [im]

ani = FuncAnimation(fig, animate, frames=int(t_final/dt), interval=50, blit=True)
plt.show()
