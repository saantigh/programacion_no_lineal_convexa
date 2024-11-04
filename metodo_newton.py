import numpy as np
from scipy.optimize import minimize

# Definir la función objetivo
def objective(x):
    x1, x2 = x
    return x1**2 - 5*x1 + 2*x2**2 - 8*x2

# Definir el gradiente de la función objetivo
def gradient(x):
    x1, x2 = x
    gradiente_x1 = 2*x1 - 5
    gradiente_x2 = 4*x2 - 8
    return np.array([gradiente_x1, gradiente_x2])

# Definir la hessiana de la función objetivo
def hessian(x):
    return np.array([[2, 0], [0, 4]])

# Definir las restricciones (3x1 + 2x2 <= 6)
def constraint1(x):
    return 6 - (3*x[0] + 2*x[1])

# Definir las restricciones en formato scipy
constr = {'type': 'ineq', 'fun': constraint1}

# Restricciones de no negatividad (x1 >= 0, x2 >= 0)
bnds = [(0, None), (0, None)]

# Inicialización del punto de partida (X)^0 = (0, 0)
punto_partida = [0, 0]



# Resolver utilizando el método Newton-CG
result = minimize(objective, punto_partida, jac=gradient, hess=hessian, 
                  method='trust-constr', bounds=bnds, constraints=constr)

# Imprimir los resultados
print("Valores optimizados: x1 =", result.x[0], ", x2 =", result.x[1])
print("Valor mínimo de la función objetivo:", result.fun)

# Verificar que las restricciones se cumplen
constraint_value = 3*result.x[0] + 2*result.x[1]
print("Comprobación de la restricción 3x1 + 2x2 <= 6:", constraint_value <= 6)

# Evaluar la función objetivo en los valores obtenidos
objective_value = objective(result.x)
print("Valor de la función objetivo evaluada:", objective_value)

