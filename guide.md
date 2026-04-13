# [cite_start]UNIVERSIDAD SAN FRANCISCO DE QUITO [cite: 1]
[cite_start]**COLEGIO:** CIENCIAS E INGENIERÍAS [cite: 1]
[cite_start]**Horario:** * S 08:00 - 11:50 [cite: 2]
* [cite_start]LMIJ 18:30 - 20:20 [cite: 3]

## [cite_start]Taller 01 [cite: 5]
[cite_start]Consideren los siguientes ejercicios: [cite: 4]

### [cite_start]1. Optimización de Funciones (30%) [cite: 6]
[cite_start]Utilizando **algoritmos genéticos**, el algoritmo de **simmulated annealing** y el algoritmo de **particle swarm optimization** encuentre la solución para las siguientes funciones: [cite: 6, 7]
* [cite_start]Ackley Function [cite: 8]
* [cite_start]Goldstein-Price Function [cite: 9]
* [cite_start]Easom Function [cite: 10]

**Requerimientos:**
* [cite_start]Utilice diferentes hiperparámetros de cada algoritmo (al menos dos configuraciones distintas) y discuta brevemente los resultados obtenidos. [cite: 11]
* [cite_start]Referencia: [Test functions for optimization](https://en.wikipedia.org/wiki/Test_functions_for_optimization). [cite: 12]

---

### [cite_start]2. Travelling Salesman Problem - Grafos Grandes (50%) [cite: 13]
[cite_start]Utilizando **algoritmos genéticos** y el algoritmo de **ant colony optimization** (ant system) resuelva las siguientes instancias del Travelling Salesman Problem para grafos completamente conectados de 25, 100 y 225 nodos: [cite: 13]

* [cite_start]**Grafos Aleatorios:** En un área de $100 \times 100$ unidades. [cite: 14]
* [cite_start]**Grafos en Grid NxN:** Así para el $25=5 \times 5$, $100=10 \times 10$ y $225=15 \times 15$. [cite: 15] [cite_start]Los nodos tienen que estar separados 10 unidades unos de otros en línea recta. [cite: 15]

**Especificaciones técnicas:**
* [cite_start]**Visualización:** Para los dos algoritmos, grafique los grafos y muestre la mejor solución encontrada. [cite: 16]
* [cite_start]**Algoritmo Genético:** Implemente dos métodos de crossover y compare sus mejores resultados. [cite: 17]
* **Ant Colony Optimization:** Implemente límites inferior y superior para el valor de feromona, y al final de cada época, implemente elitismo. [cite_start]Compare sus mejores resultados para la versión sin estas mejoras y la versión con mejoras. [cite: 18]

---

### [cite_start]3. Travelling Salesman Problem - Instancia Específica (20%) [cite: 19]
[cite_start]Utilizando algoritmos genéticos y el algoritmo de ant colony optimization (ant system) encuentre la solución a la siguiente instancia del problema del viajero: [cite: 19]

[cite_start]**Número de ciudades = 10** [cite: 20]
[cite_start]**Matriz de distancias:** [cite: 20]
```python
import numpy as np

# Matriz de distancias (simétrica)
distances = np.array([
    [0, 29, 20, 21, 16, 31, 100, 12, 4, 31],
    [29, 0, 15, 29, 28, 40, 72, 21, 29, 41],
    [20, 15, 0, 15, 14, 25, 81, 9, 23, 27],
    [21, 29, 15, 0, 4, 12, 92, 12, 25, 13],
    [16, 28, 14, 4, 0, 16, 94, 9, 20, 16],
    [31, 40, 25, 12, 16, 0, 98, 24, 36, 3],
    [100, 72, 81, 92, 94, 98, 0, 90, 101, 99],
    [12, 21, 9, 12, 9, 24, 90, 0, 15, 25],
    [4, 29, 23, 25, 20, 36, 101, 15, 0, 35],
    [31, 41, 27, 13, 16, 3, 99, 25, 35, 0]
])
```
[cite_start]*Reporte la distancia mínima y el camino asumiendo que las ciudades están indexadas del 0 al 9.* [cite: 35]

---

### [cite_start]4. Entrenamiento de Neurona (30%) [cite: 36]
[cite_start]Utilizando el algoritmo de descenso de gradiante y algoritmos genéticos entrene el siguiente modelo de neurona: [cite: 36]

[cite_start]**Modelo:** [cite: 37]
[cite_start]$$z = w_0X_0 + w_1X_1 + w_2X_2 + w_3X_1^2 + w_4X_2^2 + w_5X_1X_2$$ [cite: 39]

[cite_start]**Donde:** [cite: 38]
* [cite_start]$X_0 = 1$ representa el término de sesgo (bias). [cite: 40]
* [cite_start]$w_i$ son los pesos correspondientes. [cite: 41]
* [cite_start]Los términos $X_1^2$, $X_2^2$ y $X_1X_2$ capturan la no linealidad polinómica. [cite: 42]

[cite_start]**La salida de la neurona es:** [cite: 43]
[cite_start]$\hat{y} = f(z)$ [cite: 44]
[cite_start]donde f es la función de activación, $f(z) = \frac{1}{1+e^{-z}}$ [cite: 45]

[cite_start]**Utilizando la función de pérdida:** [cite: 46]
[cite_start]$L = \frac{1}{2}(\hat{y} - y)^2$ [cite: 47]
[cite_start]donde y es la salida real (label). [cite: 48]
[cite_start]Para los datos no lineales adjuntos en el ejercicio. [cite: 49]

---

[cite_start]**Instrucciones de entrega:** Entreguen los resultados de todos los ejercicios en un único documento, de preferencia un pdf que incluya el código y capturas de la resolución de los ejercicios. [cite: 50]
