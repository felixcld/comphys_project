import numpy as np
import matplotlib.pyplot as plt


l = 0.1
#omega = 5
C = 2
g = 9.81
print(np.sin(np.pi/2))

def pendel_function(vector,t, omega):
    y = vector[1] #y = d theta/dt
    theta = vector[0]
    f_1 = y
    f_2 = -g/l * np.sin(theta) + C * np.cos(theta) * np.sin(omega * t)
    return np.array([f_1,f_2])

def pendel_solve(N, omega):
    vector_work = np.array([0,0], float)
    start = 0
    end = 100
    h = (end - start) / N
    tpoints = np.arange(start, end, h)
    vector_solution_theta = []
    vector_solution_y = []
    for t in tpoints:
        vector_solution_theta.append(vector_work[0])
        vector_solution_y.append(vector_work[1])
        k1 = h * pendel_function(vector_work, t, omega)
        k2 = h * pendel_function(vector_work + 0.5 * k1, t + 0.5 * h, omega)
        k3 = h * pendel_function(vector_work + 0.5 * k2, t + 0.5 * h, omega)
        k4 = h * pendel_function(vector_work + k3, t + h, omega)
        vector_work += (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return vector_solution_y
    plt.plot(tpoints, vector_solution_theta)
    plt.show()
#pendel_solve(10000,10)



#pendel_solve(10000,np.sqrt(98.1))
#plt.plot(np.arange(0,100,0.01), np.sin((70 + 158*0.1)*np.arange(0,100,0.01)))
#plt.show()

def pendel_resonanz():
    test = []
    for i in np.arange(9.4,9.6,0.01):
        theta_max = np.amax(abs(np.array(pendel_solve(1000,i))))
        test.append(theta_max)
    resonanz_grob = np.where(np.array(test) == np.amax(np.array(test)))
    print(resonanz_grob)


pendel_resonanz()