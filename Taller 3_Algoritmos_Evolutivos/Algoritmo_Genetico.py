# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 21:54:10 2025

@author: jhrgu
"""
import pandas as pd
import numpy as np
import random

class AG:
    
    def __init__(self, pob, df, Peso_Max, prob_mutacion=0.1):
        self.pob = pob
        self.df = df
        self.Peso_Max = Peso_Max
        self.prob_mutacion = prob_mutacion
        self.elite = None

    def Poblacion(self):
        self.crom = np.random.randint(0, 2, size=(self.pob, len(self.df)))
        self.precio = np.array(self.df['Precio [$]'])
        self.peso = np.array(self.df['Peso [kg]'])
        return self.crom, self.precio, self.peso

    def fitness(self, crom):
        fit = np.zeros(crom.shape[0])
        fit2 = np.zeros(crom.shape[0])
        for j in range(crom.shape[0]):
            for i in range(crom.shape[1]):
                if crom[j][i] == 1:
                    fit[j] += self.precio[i]
                    fit2[j] += self.peso[i]
        return fit, fit2

    def torneo(self, crom):
        indices = np.random.permutation(crom.shape[0])
        matriz_reordenada = crom[indices]
        fit1, fit2 = self.fitness(matriz_reordenada)

        Matriz_torneo = []
        Matriz_restriccion = []
        Matriz_Fitness = []

        i = 0
        while i < len(matriz_reordenada):
            idx1 = i
            idx2 = i+1 if i+1 < len(matriz_reordenada) else i

            fit1_1 = fit1[idx1]
            fit1_2 = fit1[idx2]
            fit2_1 = fit2[idx1]
            fit2_2 = fit2[idx2]
            fila1 = matriz_reordenada[idx1]
            fila2 = matriz_reordenada[idx2]

            Matriz_torneo.append(fila1 if fit1_1 > fit1_2 and fit2_1 < fit2_2 else fila2)
            Matriz_restriccion.append(fit2_1 if fit1_1 > fit1_2 and fit2_1 < fit2_2 else fit2_2)
            Matriz_Fitness.append(fit1_1 if fit1_1 > fit1_2 and 0 < fit2_1 < fit2_2 else fit1_2)
            i += 2

        Matriz_Ganadores = np.array(Matriz_torneo)
        Valor_Rest_crom = np.array(Matriz_restriccion)
        Valor_Fitness = np.array(Matriz_Fitness)

        ganadores, padres_n, restriccion, fitness = self.restriccion(Matriz_Ganadores, Valor_Rest_crom, Valor_Fitness, self.Peso_Max)
        return ganadores, padres_n, fitness, restriccion

    def restriccion(self, Matriz_Ganadores, matriz_restriccion, matriz_objetivo, rest):
        for i in range(len(Matriz_Ganadores)):
            if rest < matriz_restriccion[i]:
                continue
            else:
                peso_valido = False
                intentos = 0
                while not peso_valido and intentos < 100:
                    nuevo = np.random.randint(0, 2, size=len(self.df))
                    peso = np.sum(nuevo * self.peso)
                    if peso <= rest:
                        Matriz_Ganadores[i] = nuevo
                        peso_valido = True
                    intentos += 1

        matriz_padres_n = Matriz_Ganadores
        fit, rest = self.fitness(matriz_padres_n)
        return Matriz_Ganadores, matriz_padres_n, rest, fit

    def mutacion_bitflip(self, cromosomas):
        for i in range(cromosomas.shape[0]):
            for j in range(cromosomas.shape[1]):
                if random.random() < self.prob_mutacion:
                    cromosomas[i][j] = 1 - cromosomas[i][j]
        return cromosomas

    def actualizar_elite(self, matriz_padres, fit, rest):
        for i in range(len(fit)):
            if rest[i] <= self.Peso_Max:
                if self.elite is None or fit[i] > self.elite[0]:
                    self.elite = (fit[i], rest[i], matriz_padres[i])

    def mostrar_generacion(self, generacion):
        if self.elite:
            fit_val, peso_val, crom = self.elite
            objetos = [self.df['Objeto'][i] for i in range(len(crom)) if crom[i] == 1]
            print(f'🧬 Generación {generacion}:')
            print(f'   - Mejor Fitness hasta ahora: ${fit_val:.2f}')
            print(f'   - Peso: {peso_val} kg')
            print(f'   - Cromosoma: {crom}')
            print(f'   - Objetos: {objetos}\n')
        else:
            print(f'⚠️ Generación {generacion}: No se encontró solución válida.\n')

    def ejecutar(self, generaciones):
        self.Poblacion()
        poblacion_actual = self.crom
        for gen in range(1, generaciones + 1):
            ganadores, padres_n, fitness, restriccion = self.torneo(poblacion_actual)
            padres_mutados = self.mutacion_bitflip(padres_n)  # 🔁 Aplicar mutación
            self.actualizar_elite(padres_mutados, fitness, restriccion)
            self.mostrar_generacion(gen)
            poblacion_actual = padres_mutados

        if not self.elite:
            return "❌ No se encontró solución válida al finalizar las generaciones."

        fit_val, peso_val, crom = self.elite
        objetos = [self.df['Objeto'][i] for i in range(len(crom)) if crom[i] == 1]
        return f'✅ Solución final:\n- Objetos seleccionados: {objetos}\n- Valor total: ${fit_val:.2f}\n- Peso total: {peso_val} kg'
    
 
datos = {
    'Objeto': ['Sensor IR', 'Sensor sonido', 'Sensor GPS', 'Sensor T','Sensor humedad','Sensor presión','Sensor gases','Cámara','Computadora','Power banks','Tienda campaña','Telefono satelital'],
    'Peso [kg]': [5, 11, 15, 10, 8, 9, 7, 10, 5, 10, 40, 18],
    'Precio [$]': [500, 230, 100, 95.2, 112.5, 220, 450, 400, 1000, 500, 1000, 400]
}

df = pd.DataFrame(datos)
generaciones=5000
poblacion = 20
Peso_Maximo=80
modelo = AG(poblacion, df,Peso_Maximo)
print(modelo.ejecutar(generaciones))