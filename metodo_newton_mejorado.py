import numpy as np
from scipy.optimize import minimize
from sympy import symbols, diff, Matrix, lambdify

# Función para verificar convexidad mediante valores propios de la hessiana
def is_convex(hessian_func, x):
    hessian_eval = hessian_func(x[0], x[1])  # Pasar x[0] y x[1] como argumentos separados
    eigenvalues = np.linalg.eigvals(hessian_eval)
    return np.all(eigenvalues >= 0)  # Verdadero si todos los valores propios son no negativos

# Solicitar al usuario la función objetivo y las restricciones
def get_user_function():
    x1, x2 = symbols('x1 x2')
    
    # Función objetivo
    objective_expr = x1**2 - 5*x1 + 2*x2**2 - 8*x2  # o pedir al usuario que la ingrese
    
    # Derivadas parciales para el gradiente
    gradient_expr = [diff(objective_expr, var) for var in (x1, x2)]
    
    # Hessiana
    hessian_expr = Matrix([[diff(g, var) for var in (x1, x2)] for g in gradient_expr])
    
    # Convertir expresiones simbólicas a funciones numéricas evaluables
    objective_func = lambdify((x1, x2), objective_expr, 'numpy')
    gradient_func = lambdify((x1, x2), gradient_expr, 'numpy')
    hessian_func = lambdify((x1, x2), hessian_expr, 'numpy')
    
    return objective_func, gradient_func, hessian_func

# Restricciones de ejemplo
def constraint1(x):
    return 6 - (3*x[0] + 2*x[1])

# Definir restricciones y límites en formato scipy
constr = {'type': 'ineq', 'fun': constraint1}
bnds = [(0, None), (0, None)]

# Obtener las funciones definidas por el usuario
objective_func, gradient_func, hessian_func = get_user_function()

# Inicialización del punto de partida
punto_partida = [0, 0]

# Resolver usando trust-constr con hessiana proporcionada
result = minimize(
    lambda x: objective_func(*x),  # Función objetivo
    punto_partida,
    jac=lambda x: np.array(gradient_func(*x)),  # Gradiente
    hess=lambda x: np.array(hessian_func(*x)),  # Hessiana
    method='trust-constr',
    bounds=bnds,
    constraints=constr
)

# Resultados
print("Valores optimizados: x1 =", result.x[0], ", x2 =", result.x[1])
print("Valor mínimo de la función objetivo:", result.fun)

# Comprobación de la convexidad en el punto óptimo
print("La función es convexa en el punto óptimo:", is_convex(hessian_func, result.x))


