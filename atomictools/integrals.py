import numpy as np
import atomictools as at

def integral_sph(orbital1, orbital2=None, f=None, kr=None, ktheta=None, kphi=None):
    # 1) Si no se da orbital2, asumir orbital2=orbital1. Se mantiene la discretización del orbital.
    #    Si se da orbital2, comparar xmin, xmax, ymin... de ambos orbitales
    #    y tomar los rangos extremos una discretización común para ambos orbitales.
    #    Reevaluarlos con esta nueva discretización
    # 2) Si alguno o varios de kr, ktheta o kphi son enteros >=1, entonces derivar orbital2.
    #    En caso contrario, tomar orbital2 (función compleja).
    # 3) Hacer el conjugado complejo de orbital1 y multiplicar por el resultado del punto 2)
    # 4) Si se da f (se espera que f sea una función de r, theta, phi en ese orden),
    #    evaluar la función en los puntos de discretización y multiplicar por el resultado de 3)
    # 5) Integrar (sumar)
    #
    # Hacer lo mismo para una función integral_cart en el que se espera un operador de x,y,z
    # Si sólo se da orbital1 y f, debería dar el mismo resultado que el valor esperado.
    # El método de la clase orbital para el valor esperado con operadores diferenciales
    # puede usar esta función para no repetir código.
    # El cálculo de derivadas puede estar en una función auxiliar.
    
    #at.integral_sph(orbital1 = orb1, f = f, orb2, kphi=1)
    
def derivative_r(matrix, dr):
def derivative_theta(matrix, dtheta):