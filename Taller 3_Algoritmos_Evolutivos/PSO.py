# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 20:13:59 2025

@author: jhrgu
"""

import pandas as pd
import numpy as np
import random
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class PSO:
    
    def __init__(self, particulas):
        self.particulas = particulas
        self.Mejor_Fit = None
        self.Mejor_Pos_x = None
        self.Mejor_Pos_y = None
        self.mejor_x_enjanbre = None
        self.mejor_y_enjambre = None
        self.mejor_f_enjambre = None
        self.vel_x = None
        self.vel_y = None
        self.fit_guardado = float('inf')

    def Particulas(self, x0, xn, y0, yn):
        x1 = np.random.uniform(x0, xn, self.particulas)
        y1 = np.random.uniform(y0, yn, self.particulas)
        return x1, y1  # sin redondeo

    def Fitness(self, f, X, Y):
        x, y = sp.symbols('x y')
        fitness = sp.lambdify((x, y), f, 'numpy')
        return X, Y, fitness(X, Y)  # sin redondeo

    def mejores_part(self, x, y, fitness):
        if self.Mejor_Fit is None:
            self.Mejor_Pos_x = np.array([x.copy()])
            self.Mejor_Pos_y = np.array([y.copy()])
            self.Mejor_Fit = np.array([fitness.copy()])
        for i in range(self.particulas):
            if fitness[i] < self.Mejor_Fit[0][i]:
                self.Mejor_Pos_x[0][i] = x[i]
                self.Mejor_Pos_y[0][i] = y[i]
                self.Mejor_Fit[0][i] = fitness[i]
        return self.Mejor_Pos_x, self.Mejor_Pos_y, self.Mejor_Fit

    def mejores_enjambre(self):
        idx = np.argmin(self.Mejor_Fit[0])
        x_enjambre = self.Mejor_Pos_x[0][idx]
        y_enjambre = self.Mejor_Pos_y[0][idx]
        fit_enjambre = self.Mejor_Fit[0][idx]

        # Actualización directa sin condicional
        self.mejor_x_enjanbre = x_enjambre
        self.mejor_y_enjambre = y_enjambre
        self.mejor_f_enjambre = fit_enjambre
        self.fit_guardado = fit_enjambre

        return self.mejor_x_enjanbre, self.mejor_y_enjambre, self.fit_guardado

    def velocidad(self, x_mejor, x_actual, y_mejor, y_actual, x_mejor_enj, y_mejor_enj, K_iner, K_cogn, K_soc):
        if self.vel_x is None or self.vel_y is None:
            self.vel_x = np.zeros(self.particulas)
            self.vel_y = np.zeros(self.particulas)

        r1 = np.random.uniform(0, 1, self.particulas)
        r2 = np.random.uniform(0, 1, self.particulas)

        cognitive_x = K_cogn * r1 * (x_mejor - x_actual)
        social_x = K_soc * r2 * (x_mejor_enj - x_actual)
        self.vel_x = K_iner * self.vel_x + cognitive_x + social_x

        cognitive_y = K_cogn * r1 * (y_mejor - y_actual)
        social_y = K_soc * r2 * (y_mejor_enj - y_actual)
        self.vel_y = K_iner * self.vel_y + cognitive_y + social_y

        # Limitar velocidad
        vel_max = 1.0
        self.vel_x = np.clip(self.vel_x, -vel_max, vel_max)
        self.vel_y = np.clip(self.vel_y, -vel_max, vel_max)

        return self.vel_x, self.vel_y

# ---------------------------------------------------------------------------------------- #

def correr_PSO(f, x0, xn, y0, yn, N_particulas, K_iner, K_cogn, K_soc, Iteraciones):
    enjambre = PSO(N_particulas)
    x1, y1 = enjambre.Particulas(x0, xn, y0, yn)

    trayectoria_x = []
    trayectoria_y = []

    for i in range(Iteraciones):
        a, b, c = enjambre.Fitness(f, x1, y1)
        X, Y, F = enjambre.mejores_part(a, b, c)
        max_x, max_y, max_f = enjambre.mejores_enjambre()
        vel_x, vel_y = enjambre.velocidad(X[0], x1, Y[0], y1, max_x, max_y, K_iner, K_cogn, K_soc)

        x1 = np.clip(x1 + vel_x, x0, xn)
        y1 = np.clip(y1 + vel_y, y0, yn)

        trayectoria_x.append(x1.copy())
        trayectoria_y.append(y1.copy())

        print(f"Iteración {i+1}: Mejor X = {max_x:.3f}, Mejor Y = {max_y:.3f}, Fitness = {max_f:.3f}")

    visualizar_trayectorias(trayectoria_x, trayectoria_y, x0, xn, y0, yn)

# ---------------------------------------------------------------------------------------- #

def visualizar_trayectorias(trayectoria_x, trayectoria_y, x0, xn, y0, yn):
    fig, ax = plt.subplots()
    ax.set_xlim(x0, xn)
    ax.set_ylim(y0, yn)
    ax.set_title("Trayectorias de partículas PSO")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    puntos = ax.scatter([], [], c='red', s=40)

    def actualizar(frame):
        x = trayectoria_x[frame]
        y = trayectoria_y[frame]
        puntos.set_offsets(np.c_[x, y])
        ax.set_title(f"Iteración {frame+1}")
        return puntos,

    anim = FuncAnimation(fig, actualizar, frames=len(trayectoria_x), interval=100, blit=True)
    plt.show()

# ---------------------------------------------------------------------------------------- #

# Parámetros del problema
x, y = sp.symbols('x y')
f = sp.sin(x) * sp.cos(y) + 0.1 * x + 0.1 * y
x0, xn = -5, 5
y0, yn = -5, 5
N_particulas = 20
K_iner = 0.2
K_cogn = 0.35
K_soc = 0.45
Iteraciones = 100

# Ejecutar PSO
correr_PSO(f, x0, xn, y0, yn, N_particulas, K_iner, K_cogn, K_soc, Iteraciones)