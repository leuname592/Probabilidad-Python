'''
Tarea 1
Estudiante: Emmanuel Chavarría Solís
Carnet: B51977
Correo: leuname592@hotmail.com
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

# Se leen los datos del csv
matriz = np.genfromtxt('xy.csv', skip_header=1 ,delimiter=',')[:,1:]

# Se calcula la probabilidades marginales
filas=np.sum(matriz, axis=1) # fx
columnas=np.sum(matriz, axis=0) # fy
valor_fila=[5,6,7,8,9,10,11,12,13,14,15]
valor_columna=[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]


# Se imprimen las funciones de densidad marginal para resolver parte de la 4ta pregunta
fig1 = plt.figure()
plt.plot(valor_fila,filas)
plt.xlabel('Valor de x')
plt.ylabel('Probabilidad')
plt.title('Función de densidad marginal de X')
plt.savefig('4.2D_fx.png')

fig2 = plt.figure()
plt.plot(valor_columna,columnas)
plt.xlabel('Valor de y')
plt.ylabel('Probabilidad')
plt.title('Función de densidad marginal de Y')
plt.savefig('4.2D_fy.png')

print('\nTarea 3')
print('Estudiante: Emmanuel Chavarría Solís')
print('Carnet: B51977\n')


# Se calcula los parámetros de ajuste para fy tomando en cuenta una curva gaussiana
def gaussiana(y,mu,sigma):
    return 1/(np.sqrt(2*np.pi*sigma**2))*np.exp(-(y-mu)**2/(2*sigma**2))
# Parametros de mejor ajuste
param,_ = curve_fit(gaussiana,valor_columna,columnas)
mu,sigma=param
print("Sigma para fy: ",sigma)
print("Mu para fy: ", mu)
# Generacion de curva de mejor ajuste de fy
n = [0]*21
N = stats.norm(mu,sigma)
n = N.pdf(valor_columna)



# Se imprime la curva de ajuste para fy 
fig3 = plt.figure()
plt.plot(valor_columna,n,'r')
plt.xlabel('Valor de y')
plt.ylabel('Probabilidad')
plt.title('Modelo de ajuste para fy')
plt.savefig('1.ajustey.png')


# Se calcula los parámetros de ajuste para fx
def gaussiana2(x,mu2,sigma2):
    return 1/(np.sqrt(2*np.pi*sigma2**2))*np.exp(-(x-mu2)**2/(2*sigma2**2))
param2,_=curve_fit(gaussiana2,valor_fila,filas)
mu2,sigma2=param2
print("\nSigma para fx: ",sigma2)
print("Mu para fx: ", mu2)
# Generacion del mejor ajuste para fx
n2 = [0]*11
N2 = stats.norm(mu2,sigma2)
n2 = N2.pdf(valor_fila)


# Se imprime la curva de ajuste para fx
fig3 = plt.figure()
plt.plot(valor_fila,n2,'r')
plt.xlabel('Valor de x')
plt.ylabel('Probabilidad')
plt.title('Modelo de ajuste para fx')
plt.savefig('1.ajustex.png')

#Lectura de datos para calcular la correlación
matriz2 = np.genfromtxt('xyp.csv', skip_header=1 ,delimiter=',')
altura=matriz2.shape[0]
correlacion=0

#Calculo de correlación multiplicando columnas y sumando filas
for i in range(0,altura):
    correlacion += matriz2[i][0]*matriz2[i][1]*matriz2[i][2]
print("\nLa correlación es: ", correlacion)

#Calculo de covarianza y del coeficiente de correlación
covarianza=correlacion-mu*mu2 
print("\nLa covarianza es:",covarianza)
    
pearson=covarianza/(sigma*sigma2)    
print("\nEl coeficiente de correlación es:", pearson)
    
    
# Se genera la figura en 3d para la función de probabilidad marginal conjunta
fig4 = plt.figure()
ax = plt.axes(projection='3d')
X, Y = np.meshgrid(range(5,16), range(5,26))
conjunta= 1/(np.sqrt(2*np.pi*sigma**2))*np.exp(-(Y-mu)**2/(2*sigma**2))*1/(np.sqrt(2*np.pi*sigma2**2))*np.exp(-(X-mu2)**2/(2*sigma2**2))
ax.plot_wireframe(X, Y, conjunta, color='r')
ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
ax.set_zlabel('Probabilidad')
plt.savefig('4.3D.png')
    
    
    
    