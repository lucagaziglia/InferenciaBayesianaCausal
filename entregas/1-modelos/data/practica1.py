import inspect
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import math
import numpy as np
<< << << < HEAD
== == == =
# -*- coding: utf-8 -*-


>>>>>> > 1fcb8ddc847d43da139dc2858bf4713120940f0c

<< << << < HEAD
H = np.arange(3)  # Posibles valores de las hipótesis
== == == =
H = np.arange(3)  # Posibles valores de las hipótesis
>>>>>> > 1fcb8ddc847d43da139dc2858bf4713120940f0c
# Como estamos trabajando en python vamos empezar con 0
# Es decir, la posición del regalo r \in {0,1,2} y de
# la misma forma con el resto de las variables.

<< << << < HEAD


def pr(r):  # P(r)
    if r not in H:
        raise ValueError("El regalo puede estar en la caja 0, 1 o 2")
    return 1/3


def pc(c):  # P(c)
    if c not in H:
        raise ValueError("La caja a abrir debe ser 0, 1 o 2")
    return 1/3


def ps_rM0(s, r):  # P(s|r,M=0)
    if s not in H or r not in H:
        raise ValueError("La caja a abrir debe ser 0, 1 o 2")
    return 0 if s == r else 1/2


def ps_rcM1(s, r, c):  # P(s|r,c,M=1)
    if s not in H or r not in H or c not in H:
        raise ValueError("s,r y c deben ser 0, 1 o 2")
    if s == c or s == r:
        return 0
    cajas_disponibles = [x for x in H if x != c and x != r]
    return 1 / len(cajas_disponibles)


def prcs_M(r, c, s, m):  # P(r,c,s|M)
    # producto de las condicionales
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")


def ps_cM(s, c, m):
    # Predicción del segundo dato dado el primero
    num = 0  # P(s,c|M) = sum_r P(r,c,s|M)
    den = 0  # P(c|M) = sum_{rs} P(r,c,s|M)
    res = num/den  # P(s|c,M) = P(s,c|M)/P(c|M)
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")


def pr_csM(r, c, s, m):
    # Predicción del segundo dato dado el primero
    num = 0  # p(r,c,s|M)
    den = 0  # p(c,s|M) = sum_r P(r,c,s|M)
    res = num/den  # P(r|c,s,M) = P(r,c,s|M)/p(c,s|M)
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")


def pEpisodio_M(c, s, r, m):  # P(Datos = (c,s,r) | M)
    # Predicción del conjunto de datos P(c,s,r|M)
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

# 1.2 Simular datos con Monty Hall


def simular(T=16, seed=0):
    np.random.seed(seed)
    Datos = []
    for t in range(T):
        r = np.random.choice(3, p=[pr(hr) for hr in H])
        c = None
        s = None
        Datos.append((c, s, r))
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")


T = 16
Datos = simular()

# 1.3 Predicción P(Datos = {(c0,s0,r0),(c1,s1,r1),...} | M )


def _secuencia_de_predicciones(Datos, m):
    # Si se guarda la lista de predicciones de cada uno
    # de los episodios [P(Episodio0|M),P(Episodio1|M),... ]
    # esto va a servir tanto para calcular la predicción
    # P(Datos = {(c0,s0,r0),(c1,s1,r1),...} | M ),
    # pero también va a servir después para graficar como
    # va cambiando el posterior de los modelos en el tiempo
    return None


def pDatos_M(Datos, m):
    # P(Datos = {(c0,s0,r0),(c1,s1,r1),...} | M )
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")


pDatos_M(Datos, m=0)  # 8.234550899283273e-21
pDatos_M(Datos, m=1)  # 3.372872048346429e-17

# 1.4 Calcular predicción de los datos P(Datos)


def pM(m):
    # Prior de los modelos
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")


def pDatos(Datos):
    # sum_m P(Datos,M=m)
    # sum_m P(Datos|M=m)P(M=m)
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

# 1.5 Posterior de los modelos


def pM_Datos(m, Datos)


# P(M|Datos = {(c0,s0,r0),(c1,s1,r1),...})
return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")


def lista_pM_Datos(m, Datos):
    # [P(M | (c0,s0,r0) ), P(M | (c0,s0,r0),(c1,s1,r1) ), ... ]
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")


plt.plot(lista_pM_Datos(m=0, Datos), label="M0: Base")
plt.plot(lista_pM_Datos(m=1, Datos), label="M1: Monty Hall")
plt.legend()
plt.show()


# 2.1

def pp_Datos(p, Datos):
    # P(p | Datos = {(c0,s0,r0),(c1,s1,r1), ... })
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")


# 2.2

def pEpisodio_DatosMa(Episodio, Datos):
    # P(EpisodioT = (cT, sT, rT) | Datos = {(c0,s0,r0),(c1,s1,r1), ... })
    cT, sT, rT = Episodio
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

# 2.3

# Actualizar pDatos_M(Datos, m, log = False) agregándole un parámetro log

# 2.4


def log_Bayes_factor(log_pDatos_Mi, log_pDatos_Mj):
    # Recibe la predicción de los datos en ordenes de magnitud
    # y devuelve el logaritmo del Bayes factor, es decir,
    # la diferencia de predicciones en órdenes de magnitud
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

# 2.5


def geometric_mean(Datos, m, log=False)


# Dado los datos y el modelo devuelve la media geométrica
return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

# 2.6

# actualizar pM_Datos(m,Datos) para que soporte al modelo alternativo


def pr(r):  # P(r)
    if r not in H:
        raise ValueError("El regalo puede estar en la caja 0, 1 o 2")
    return 1/3


def pc(c):  # P(c)
    if c not in H:
        raise ValueError("La caja a abrir debe ser 0, 1 o 2")
    return 1/3

######


def ps_rM0(s, r):  # P(s|r,M=0)
    if s not in H or r not in H:
        raise ValueError("La caja a abrir debe ser 0, 1 o 2")
    if s == r:
        return 0
    else:
        return 1/2


def ps_rcM1(s, r, c):  # P(s|r,c,M=1)
    if s not in H or r not in H or c not in H:
        raise ValueError("s,r y c deben ser 0, 1 o 2")
    if s == c or s == r:
        return 0
    cajas_disponibles = [x for x in H if x != c and x != r]
    return 1 / len(cajas_disponibles)

#####


def prcs_M(r, c, s, m):  # P(r,c,s|M)
    # producto de las condicionales
    if s not in H or r not in H or c not in H:
        raise ValueError("s,r y c deben ser 0, 1 o 2")
    if m not in [0, 1]:
        raise ValueError("m debe ser 0 o 1")
    return pr(r) * pc(c) * (ps_rM0(s, r) if m == 0 else ps_rcM1(s, r, c))


def ps_cM(s, c, m):
    # Predicción del segundo dato dado el primero
    if s not in H or c not in H:
        raise ValueError("s y c deben ser 0, 1 o 2")
    num = sum(prcs_M(r, c, s, m) for r in H)  # P(s,c|M) = sum_r P(r,c,s|M)
    den = sum(prcs_M(r, c, o, m)
              for r in H for o in H)  # P(c|M) = sum_{rs} P(r,c,s|M)
    if den == 0:
        raise ZeroDivisionError("P(c|M) es 0")
    res = num / den  # P(s|c,M) = P(s,c|M)/P(c|M)
    return res


def pr_csM(r, c, s, m):
    num = prcs_M(r, c, s, m)  # p(r,c,s|M)
    den = sum(prcs_M(r, c, s, m) for r in H)  # p(c,s|M) = sum_r P(r,c,s|M)
    if den == 0:
        raise ZeroDivisionError("p(c,s|M) es 0")
    res = num / den  # P(r|c,s,M) = P(r,c,s|M)/p(c,s|M)
    return res


def pEpisodio_M(c, s, r, m):  # P(Datos = (c,s,r) | M)
    # Predicción del conjunto de datos P(c,s,r|M)
    if m not in [0, 1]:
        raise ValueError("m debe ser 0 o 1")
    if s not in H or r not in H or c not in H:
        raise ValueError("s,r,c deben ser 0,1,2")
    return prcs_M(r, c, s, m)


def simular(T=16, seed=0):
    import pandas as pd
    np.random.seed(seed)
    Datos = []
    for t in range(T):
        r = np.random.choice(3, p=[pr(hr) for hr in H])  # regalo
        c = np.random.choice(3, p=[pc(hc) for hc in H])  # elegida
        s = np.random.choice(3, p=[ps_rcM1(s_, r, c) for s_ in H])  # señalada
        Datos.append((c, s, r))
    return Datos


Datos = simular()


def _secuencia_de_predicciones(Datos, m):

    return [pEpisodio_M(c, s, r, m) for (c, s, r) in Datos]


def pDatos_M(Datos, m):
    preds = _secuencia_de_predicciones(Datos, m)
    prod = 1.0
    for p in preds:
        prod *= p
    return prod  # P(Datos|M) = producto de las predicciones de cada episodio


# 1.4 Calcular predicción de los datos P(Datos)

def pM(m):
    if m not in [0, 1]:
        raise ValueError("Modelo debe ser 0 o 1")
    return 1/2


def pDatos(Datos):
    pD = 0.0
    for m in [0, 1, 2]:  # Regla de la suma
        try:
            pD += pDatos_M(Datos, m) * pM(m)  # Regla del producto
        except ValueError:
            # pDatos_M todavía no soporta el modelo alternativo (el modelo correspondiente a m=2 antes de 2.x)
            continue
    return pD

# 1.5 Posterior de los modelos


def pM_Datos(m, Datos):
    denom = pDatos(Datos)
    if denom == 0:
        return 0.0
    num = pDatos_M(Datos, m) * pM(m)
    return num / denom


def lista_pM_Datos(m, Datos):
    posts = [pM(m)]  # prior en t=0
    for t in range(1, len(Datos) + 1):
        posts.append(pM_Datos(m, Datos[:t]))
    return posts


plt.plot(lista_pM_Datos(0, Datos), label="M0: Base")
plt.plot(lista_pM_Datos(1, Datos), label="M1: Monty Hall")
plt.legend()
plt.show()

# 2.1


def pM(m):  # Redefino el prior porque ahora consideramos 3 modelos
    if m not in [0, 1, 2]:
        raise ValueError("Modelo debe ser 0 o 1")
    return 1/3


def pcsr_p(c, s, r, p):
    # P(c,s,r | p) del modelo alternativo
    mezcla = (1 - p) * ps_rM0(s, r) + p * ps_rcM1(s, r, c)
    return pr(r) * pc(c) * mezcla


def pp_Datos(Datos):
    p_grid = np.linspace(0, 1, 101)
    prior = np.ones_like(p_grid) / 101  # prior uniforme en p
    likelihood = np.ones_like(p_grid, dtype=float)

    for (c, s, r) in Datos:
        # verosimilitud episodio para cada p de la grilla
        likelihood *= np.array([pcsr_p(c, s, r, p) for p in p_grid])

    num = prior * likelihood
    den = np.sum(prior * likelihood)
    return num / den

# 2.2


def pa_p(a, p):
    # Bernoulli(p): a=1 "recuerda", a=0 "olvida"
    return p if a == 1 else (1 - p)


def ps_rca(s, r, c, a):
    # compuerta: si a=0 usa Base, si a=1 usa Monty
    return ps_rM0(s, r) if a == 0 else ps_rcM1(s, r, c)


def pEpisodio_DatosMa(Episodio, Datos):
    cT, sT, rT = Episodio
    p_grid = np.linspace(0.0, 1.0, 101)

    if len(Datos) == 0:
        post_p = np.ones_like(p_grid) / len(p_grid)
    else:
        post_p = pp_Datos(Datos)

    total = 0.0
    for p, w in zip(p_grid, post_p):  # w = P(p | Datos previos)
        for a in (0, 1):
            total += pr(rT) * pc(cT) * ps_rca(sT, rT, cT, a) * pa_p(a, p) * w

    return total

# 2.3


def pDatos_M(Datos, m, log=False):
    if m in (0, 1):
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


def log_Bayes_factor(log_pDatos_Mi, log_pDatos_Mj):
    return log_pDatos_Mi - log_pDatos_Mj

# 2.5


def geometric_mean(Datos, m, log=False):
    N = len(Datos)
    if N == 0:
        raise ValueError("Datos está vacío")

    sum_log10 = 0.0

    if m in (0, 1):
        for (c, s, r) in Datos:
            p = prcs_M(r, c, s, m)
            if p <= 0.0:
                return (-np.inf if log else 0.0)
            sum_log10 += np.log10(p)
    elif m == 2:
        prefijo = []
        for epi in Datos:
            p = pEpisodio_DatosMa(epi, prefijo)
            if p <= 0.0:
                return (-np.inf if log else 0.0)
            sum_log10 += np.log10(p)
            prefijo.append(epi)
    else:
        raise ValueError("m debe ser 0, 1 o 'MA'")

    avg_log10 = sum_log10 / N
    return (avg_log10 if log else 10**avg_log10)

# 2.6


def pM_Datos(m, Datos, log=False):
    if m not in [0, 1, 2]:
        raise ValueError("Modelo debe ser 0, 1 o 2")

    if log:
        num = pDatos_M(Datos, m, log=True) + np.log(pM(m))
        den = np.log(sum(np.exp(pDatos_M(Datos, mm, log=True) + np.log(pM(mm)))
                         for mm in [0, 1, 2]))
        return num - den  # log posterior
    else:
        num = pDatos_M(Datos, m) * pM(m)
        den = pDatos(Datos)
        return num / den if den > 0 else 0


df = pd.read_csv("entregas/1-modelos/data/NoMontyHall.csv")
DatosM2 = list(df.iloc[:60].itertuples(index=False, name=None))

pm0 = lista_pM_Datos(0, DatosM2)
pm1 = lista_pM_Datos(1, DatosM2)
pm2 = lista_pM_Datos(2, DatosM2)

plt.plot(pm0, label="M0: Base")
plt.plot(pm1, label="M1: Monty Hall")
plt.plot(pm2, label="M2: Alternativo")
plt.ylim(-0.01, 1.01)
plt.xlabel("Episodio")
plt.ylabel("P(Modelo | Datos)")
plt.legend()
plt.tight_layout()
plt.show()
>>>>>> > 1fcb8ddc847d43da139dc2858bf4713120940f0c
