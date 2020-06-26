import numpy as np
import matplotlib.pyplot as plt

sigma = 10
r = 28
b = 8/3

def Lorenz_function(vector,t):
    f_1 = sigma * (vector[1] - vector[0])
    f_2 = r * vector[0] - vector[1] - vector[0] * vector[2]
    f_3 = vector[0] * vector[1] - b * vector[2]
    return np.array([f_1,f_2,f_3],float)



def Lorenz_solve(N):
    vector_work = np.array([0,1,0], float)
    start = 0
    end = 50
    h = (end - start)/N
    tpoints = np.arange(start, end, h)
    vector_solution_x = []
    vector_solution_y = []
    vector_solution_z = []
    for t in tpoints:
        vector_solution_x.append(vector_work[0])
        vector_solution_y.append(vector_work[1])
        vector_solution_z.append(vector_work[2])
        k1 = h * Lorenz_function(vector_work, t)
        k2 = h * Lorenz_function(vector_work + 0.5 * k1, t + 0.5 * h)
        k3 = h * Lorenz_function(vector_work + 0.5 * k2, t + 0.5 * h)
        k4 = h * Lorenz_function(vector_work + k3, t + h)
        vector_work += (k1 + 2 * k2 + 2 * k3 + k4) / 6

    plt.plot(vector_solution_x,vector_solution_z)
    plt.show()

Lorenz_solve(100000)