import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cons
from scipy import integrate


def kometen_function(vector, t):
    G = 6.6743e-11
    M = 1.9884 * 10 ** 30
    x_strich = vector[1]
    x = vector[0]
    y_strich = vector[3]
    y = vector[2]
    r = np.sqrt(x ** 2 + y ** 2)
    f1 = x_strich
    f2 = -G * M * x / (np.sqrt(x ** 2 + y ** 2) ** 3)
    f3 = y_strich
    f4 = -G * M * y / (np.sqrt(x ** 2 + y ** 2) ** 3)
    return np.array([f1, f2, f3, f4], float)


def kometen_solve(N):
    vector_work = np.array([4e12, 0, 0, 500], float)
    start = 0
    end = 3e9
    h = float((end - start) / N)
    tpoints = np.arange(start, end, h)
    vector_solution_x = []
    vector_solution_y = []
    # vector_solution_x_strich = []
    # vector_solution_y_strich = []
    for t in tpoints:
        # vector_solution_x_strich.append(vector_work[0])
        vector_solution_x.append(vector_work[0])
        # vector_solution_y_strich.append(vector_work[2])
        vector_solution_y.append(vector_work[2])
        k1 = h * kometen_function(vector_work, t)
        k2 = h * kometen_function(vector_work + 0.5 * k1, t + 0.5 * h)
        k3 = h * kometen_function(vector_work + 0.5 * k2, t + 0.5 * h)
        k4 = h * kometen_function(vector_work + k3, t + h)
        vector_work += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    plt.plot(vector_solution_x, vector_solution_y)
    plt.show()


kometen_solve(10000)