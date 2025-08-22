# -*- coding: utf-8 -*-


import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import inspect

H = np.arange(3) # Posibles valores de las hipótesis
# Como estamos trabajando en python vamos empezar con 0
# Es decir, la posición del regalo r \in {0,1,2} y de
# la misma forma con el resto de las variables.

def pr(r): # P(r)
    if r not in H:
      raise ValueError("El regalo puede estar en la caja 0, 1 o 2")
    return 1/3

def pc(c): # P(c)
    if c not in H:
      raise ValueError("La caja a abrir debe ser 0, 1 o 2")
    return 1/3

######

def ps_rM0(s,r): # P(s|r,M=0)
    if s not in H or r not in H:
      raise ValueError("La caja a abrir debe ser 0, 1 o 2")
    if s == r: 
      return 0 
    else:
      return 1/2
    
def ps_rcM1(s,r,c): # P(s|r,c,M=1)
    if s not in H or r not in H or c not in H:
      raise ValueError("s,r y c deben ser 0, 1 o 2")
    if s == c or s == r:
      return 0
    cajas_disponibles = [x for x in H if x != c and x != r]
    return 1 / len(cajas_disponibles)

#####

def prcs_M(r,c,s,m): # P(r,c,s|M)
    # producto de las condicionales
    if s not in H or r not in H or c not in H:
      raise ValueError("s,r y c deben ser 0, 1 o 2")
    if m not in [0,1]:
      raise ValueError("m debe ser 0 o 1")
    return pr(r) * pc(c) * (ps_rM0(s,r) if m == 0 else ps_rcM1(s,r,c))

## Aca empiezo a modificar yo.

def ps_cM(s,c,m):
    # Predicción del segundo dato dado el primero
    if s not in H or c not in H:
      raise ValueError("s y c deben ser 0, 1 o 2")
    num = sum(prcs_M(r,c,s,m) for r in H) # P(s,c|M) = sum_r P(r,c,s|M)
    den = sum(prcs_M(r,c,o,m) for r in H for o in H) # P(c|M) = sum_{rs} P(r,c,s|M)
    if den == 0:
      raise ZeroDivisionError("P(c|M) es 0")
    res = num / den # P(s|c,M) = P(s,c|M)/P(c|M)
    return res

def pr_csM(r,c,s,m):
    num = prcs_M(r,c,s,m) # p(r,c,s|M)
    den = sum(prcs_M(r,c,s,m) for r in H) # p(c,s|M) = sum_r P(r,c,s|M)
    if den == 0:
      raise ZeroDivisionError("p(c,s|M) es 0")
    res = num / den # P(r|c,s,M) = P(r,c,s|M)/p(c,s|M)
    return res

### Tamos hasta aca

def pEpisodio_M(c,s,r,m): # P(Datos = (c,s,r) | M)
    # Predicción del conjunto de datos P(c,s,r|M)
    if m not in [0,1]:
        raise ValueError("m debe ser 0 o 1")
    if s not in H or r not in H or c not in H:
        raise ValueError("s,r,c deben ser 0,1,2")
    return prcs_M(r,c,s,m)

# 1.2 Simular datos con Monty Hall
def simular(T=16,seed=0):
    import pandas as pd
    np.random.seed(seed)
    Datos = []
    for t in range(T):
        r = np.random.choice(3, p=[pr(hr) for hr in H]) # regalo
        c = np.random.choice(3, p=[pc(hc) for hc in H]) # elegida
        s = np.random.choice(3, p=[ps_rcM1(s_,r,c) for s_ in H]) # señalada
        Datos.append((c,s,r))
    return Datos
T = 1000
Datos = simular(T, 16)

# 1.3 Predicción P(Datos = {(c0,s0,r0),(c1,s1,r1),...} | M )

def _secuencia_de_predicciones(Datos, m):
    # Si se guarda la lista de predicciones de cada uno
    # de los episodios [P(Episodio0|M),P(Episodio1|M),... ]
    # esto va a servir tanto para calcular la predicción
    # P(Datos = {(c0,s0,r0),(c1,s1,r1),...} | M ),
    # pero también va a servir después para graficar como
    # va cambiando el posterior de los modelos en el tiempo
    return [pEpisodio_M(c, s, r, m) for (c, s, r) in Datos]

def pDatos_M(Datos, m):
    # P(Datos | M) = ∏_t P(c_t, s_t, r_t | M)
    if m not in [0,1]:
      raise ValueError("m debe ser 0 o 1")
    probs = [pEpisodio_M(c, s, r, m) for (c, s, r) in Datos]
    prod = 1.0
    for p in probs:
        prod *= p
    return prod

def pM(m):
  if m not in [0,1]:
    raise ValueError("m debe ser 0 o 1")
    # Prior de los modelos
    # A priori no sé nada, por lo que pongo un prior uniforme
  return 1/2

def pDatos(Datos):
    evidencia = 0
    # sum_m P(Datos|M=m)P(M=m)
    for m in [0,1]:
      prediccion = pDatos_M(Datos, m)
      creencia = pM(m)
      evidencia += prediccion * creencia
    return evidencia

# 1.5 Posterior de los modelos

def pM_Datos(m,Datos):
    # P(M | Datos = {(c0,s0,r0),(c1,s1,r1),...})
    likelihood = pDatos_M(Datos, m)
    prior = pM(m)
    evidencia = pDatos(Datos)
    if evidencia == 0:
      raise ZeroDivisionError("La evidencia no puede ser nula")
    return likelihood*prior/evidencia


def lista_pM_Datos(m, Datos):
    lista = [pM(m)]  # prior en t=0
    datos_parciales = []
    for (c, s, r) in Datos:
        datos_parciales.append((c, s, r))
        like = pDatos_M(datos_parciales, m)      
        evid = pDatos(datos_parciales)           
        post = like * pM(m) / evid               
        lista.append(post)
    return lista

plt.plot(lista_pM_Datos(0, Datos), label="M0: Base")
plt.plot(lista_pM_Datos(1, Datos), label="M1: Monty Hall")
plt.legend()
plt.show()


# 2.1

def pM(m): #Redefino el prior porque ahora consideramos 3 modelos
    if m not in [0, 1, 2]:
        raise ValueError("Modelo debe ser 0 o 1")
    return 1/3

grid = np.linspace(0, 1, 21)
def pp_Datos(p,Datos):
    prior_p = 1 / grid.size  # Asumimos una distribución uniforme sobre p
    likelihood_p = 1.0 #inicializamos la verosimilitud

    for (c, s, r) in Datos:
        term = (1 - p) * ps_rM0(s, r) + p * ps_rcM1(s, r, c) #(1-p) probabilidad de que se olvide, y p de que se acuerde, dependiendo el caso usa cada modelo y lo ponderamos
        likelihood_p *= term * pr(r) * pc(c)
    num = likelihood_p * prior_p  # P(Datos | p) * P(p)
    den = 0
    for p_grid in grid:
        likelihood_p_grid = 1.0
        prior_p_grid = 1 / len(grid)
        for (c, s, r) in Datos:
            term2 = (1 - p_grid) * ps_rM0(s, r) + p_grid * ps_rcM1(s, r, c) # Repetimos el mismo cálculo que antes, pero ahora para cada valor de p_grid
            likelihood_p_grid *= term2 * pr(r) * pc(c) 
        den += likelihood_p_grid * (1 / grid.size)  # P(Datos | p_grid) * P(p_grid)
    if den == 0:
        return 0
    return num / den


# 2.2

def pEpisodio_DatosMa(Episodio, Datos):
    cT, sT, rT = Episodio
    posterior_p = []
    for pv in grid:
        likelihood = 1.0
        for (c,s,r) in Datos:
            term = (1.0 - pv) * ps_rM0(s,r) + pv * ps_rcM1(s,r,c)
            likelihood *= pr(r) * pc(c) * term
        posterior_p.append(likelihood)
    posterior_p = np.array(posterior_p)
    if posterior_p.sum() == 0:
        posterior_p = np.ones_like(posterior_p) / posterior_p.size #Normalizacion
    else:
        posterior_p = posterior_p / posterior_p.sum()
    total = 0.0
    for pv, post in zip(grid, posterior_p):
        term = (1.0 - pv) * ps_rM0(sT, rT) + pv * ps_rcM1(sT, rT, cT) #Al igual que como expliqué en la función anterior, adjudicamos cierto peso a cada modelo
        total += term * pr(rT) * pc(cT) * post
    return total

# 2.3

def pDatos_M(Datos, m, log = False):
    if m in (0,1):
        if log:
            suma_log = 0
            for (c, s_, r) in Datos:
                p = pEpisodio_M(c, s_, r, m) 
                if p <= 0:
                    return -np.inf
                suma_log += np.log(p) 
            return suma_log
        else:
            prod = 1.0
            for (c, s_, r) in Datos:
                prod *= pEpisodio_M(c, s_, r, m)
            return prod
    elif m == 2: 
        if log:
            suma_log = 0
            for i in range(len(Datos)):
                p = pEpisodio_DatosMa(Datos[i], Datos[:i])
                if p <= 0:
                    return -np.inf
                suma_log += np.log(p)
            return suma_log
        else:
            prod = 1.0
            for i in range(len(Datos)):
                prod *= pEpisodio_DatosMa(Datos[i], Datos[:i])
            return prod

# 2.4

def log_Bayes_factor(log_pDatos_Mi,log_pDatos_Mj):
    return log_pDatos_Mi - log_pDatos_Mj

# 2.5

def geometric_mean(Datos, m, log = False):
    N = len(Datos)
    if log:
        sum_logs = 0
        for i in range(N):
            if m in (0, 1):
                p = pEpisodio_M(*Datos[i], m)
            elif m == 2:
                p = pEpisodio_DatosMa(Datos[i], Datos[:i])
            suma_logs += np.log(p)
        return sum_logs / N
    else:
        prod = 1.0
        for i in range(N):
            if m in (0, 1):
                p = pEpisodio_M(*Datos[i], m)
            elif m == 2:
                p = pEpisodio_DatosMa(Datos[i], Datos[:i])
            prod *= p
    return prod ** (1 / N)  # P(Datos|M)^(1/N) = (P(Datos|M1)*P(Datos|M2)*...*P(Datos|MN))^(1/N)

# 2.6

def pM_Datos(m, Datos, log=False):
    """
    P(M=m | Datos) = P(Datos|M=m) P(M=m) / P(Datos)
    Si log=True, devuelve el logaritmo del posterior.
    """
    if m not in [0, 1, 2]:
        raise ValueError("Modelo debe ser 0, 1 o 2")

    # Numerador
    if log:
        num = pDatos_M(Datos, m, log=True) + np.log(pM(m))
        den = np.log(sum(np.exp(pDatos_M(Datos, mm, log=True) + np.log(pM(mm)))
                         for mm in [0, 1, 2]))
        return num - den  # log posterior
    else:
        num = pDatos_M(Datos, m) * pM(m)
        den = pDatos(Datos)
        return num / den if den > 0 else 0


df = pd.read_csv("C:/Users/WildFi/Desktop/NoMontyHall.csv")
Datos = list(df.iloc[:60].itertuples(index=False, name=None))

pm0 = lista_pM_Datos(0, Datos)
pm1 = lista_pM_Datos(1, Datos)
pm2 = lista_pM_Datos(2, Datos)

plt.plot(pm0, label="M0: Base")
plt.plot(pm1, label="M1: Monty Hall")
plt.plot(pm2, label="M2: Alternativo")
plt.ylim(-0.01, 1.01)
plt.xlabel("Episodio")
plt.ylabel("P(Modelo | Datos)")
plt.legend()
plt.tight_layout()
plt.show()