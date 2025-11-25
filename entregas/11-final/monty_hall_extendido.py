# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


H = [0, 1, 2]  # cajas posibles


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


def prcs_M(r, c, s, m):  # P(r,c,s|M)
    # producto de las condicionales
    if s not in H or r not in H or c not in H:
        raise ValueError("s,r y c deben ser 0, 1 o 2")
    if m not in [0, 1]:
        raise ValueError("m debe ser 0 o 1")
    return pr(r) * pc(c) * (ps_rM0(s, r) if m == 0 else ps_rcM1(s, r, c))


def pEpisodio_M(c, s, r, m):  # P(Datos = (c,s,r) | M)
    # Predicción del conjunto de datos P(c,s,r|M)
    if m not in [0, 1]:
        raise ValueError("m debe ser 0 o 1")
    if s not in H or r not in H or c not in H:
        raise ValueError("s,r,c deben ser 0,1,2")
    return prcs_M(r, c, s, m)

# Modelo alternativo


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


def pa_p(a, p):
    # Bernoulli(p): a=1 "recuerda", a=0 "olvida"
    return p if a == 1 else (1 - p)


def ps_rca(s, r, c, a):
    # compuerta: si a=0 usa Base, si a=1 usa Monty
    return ps_rM0(s, r) if a == 0 else ps_rcM1(s, r, c)


def pEpisodio_DatosMa(Episodio, Datos):
    cT, sT, rT = Episodio
    p_grid = np.linspace(0.0, 1.0, 101)

    # Posterior sobre p dado Datos previos
    if len(Datos) == 0:
        post_p = np.ones_like(p_grid) / len(p_grid)  # prior uniforme
    else:
        post_p = pp_Datos(Datos)

    total = 0.0
    for p, w in zip(p_grid, post_p):   # w = P(p | Datos previos)
        for a in (0, 1):               # a = 0 (Base), 1 (Monty)
            total += pr(rT) * pc(cT) * ps_rca(sT, rT, cT, a) * pa_p(a, p) * w

    return total


# %%
df_monty = pd.read_csv(
    '../../materiales_del_curso/11-final/datos/NoMontyHall.csv')
Datos = list(df_monty.iloc[:2000].itertuples(index=False, name=None))
# %%
evidencias = []
datos_previos = []
for d in Datos:
    evidencia_base = pEpisodio_M(d[0], d[1], d[2], 0)
    evidencia_monty = pEpisodio_M(d[0], d[1], d[2], 1)
    evidencia_alternativo = pEpisodio_DatosMa(d, datos_previos)

    predicciones = [evidencia_base, evidencia_monty, evidencia_alternativo]
    datos_previos.append(d)

    evidencias.append(predicciones)

df_evidencia = pd.DataFrame(
    evidencias,
    columns=['Base', 'MontyHall', 'Alternativo']
)

df_evidencia.to_csv('../../entregas/11-final/EvidenciasNoMontyHall.csv')

# %%

for col in df_evidencia.columns:
    plt.plot(df_evidencia[col], label=f"Modelo: {col}")
    plt.xlabel("Episodio")
    plt.ylabel("Evidencia P(c,s,r | M)")
    plt.title("Evidencia por episodio")
    plt.legend()
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %%
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(df_evidencia['Base'], bins=30)
plt.title("Base (M0)")
plt.yscale("log")

plt.subplot(1, 3, 2)
plt.hist(df_evidencia['MontyHall'], bins=30)
plt.title("Monty Hall (M1)")
plt.yscale("log")

plt.subplot(1, 3, 3)
plt.hist(df_evidencia['Alternativo'], bins=30)
plt.title("Alternativo (M2)")
plt.yscale("log")

plt.tight_layout()
plt.show()

# %% [markdown]
# # Justificación de los resultados.
# ##
# Tuve bastantes complicaciones para traer las funciones de la practica 1 que hicimos al principio de la cursada. Por lo cual, definí todas las funciones necesarias nuevamente para poder calcular las evidencias de los modelos.
#
# El codigo para calcular las evidencias y armar el dataframe es bastante facil de entender, si es que se conocen esas funciones que definimos al principio del archivo. Lo que me gustaría comentar es un poco el porque los gráficos son de esa forma.
#
# ## Evidencia por episodio (plots de líneas)
# En este gráfico separe en tres gráficos distintos los 3 modelos con el objetivo de poder entenderlos, ya que se hacia complicado entenderlos cuando se superponían mucho.
#
# ### Modelo Base (M0):
# Este modelo simula la situación en la cual el host elige una caja al azar (siempre que no sea el premio). Donde la evidencia se mantiene constante a lo largo de todos los episodios, alrededor del valor de 0,055.
#
# $$ P(c,s,r \mid M_0) = \underbrace{\frac{1}{3}}_{\text{regalo}} \times \underbrace{\frac{1}{3}}_{\text{elección}} \times \underbrace{\frac{1}{2}}_{\text{abrir}} = \frac{1}{18} \approx 0.055 $$

# ### Modelo Monty Hall (M1):
# Cuando el host sigue las reglas, este modelo predice MUCHO mejor que el base (el doble de probabilidad, $\approx 0.11$). El bloque denso pareciendo que el modelo ocupa distintas evidencias es una confusión visual por lo mucho que se sorprende el modelo en cuanto el presentador 'se olvida' y muestra la misma caja que recibimos. Esto sucede, ya que el modelo le asigna probabilidad 0 a ese evento, y en este gráfico (al estar en escala logaritmica), se va a menos infinito.

# ### Modelo Alternativo (M2):
# Este modelo es el que mejor predice los datos, ya que llega a los mismos valores que el modelo Monty Hall (M1), pero cuando ocurre este 'error' del presentador, el modelo Alternativo (M2) no se sorprende tanto, es decir que,  la evidencia baja, pero no colapsa. El modelo "sabe" que existe una pequeña posibilidad ($1-p$) de que el host sea torpe, por lo que asigna una probabilidad baja pero no cero.

# ## Histograma de evidencias por modelo (plots de barras)
#
# Para el modelo base (M0) se observa una unica barra centrada en 0,055. Lo cual es algo que ya sabiamos solamente viendo el gráfico previo. Con respecto al Monty Hall, sucede algo mas que interesante. Cuando se siguen las reglas del juego el modelo M1 tiene una gran capacidad predictiva, sin embargo, el histograma muestra como hay bastantes situaciones (tantas como veces el presentador se olvide de las reglas del juego) donde la evidencia calculada es cero, esto ocurre exactamente 187 veces, de los 2000 episodios.
#
# Finalmente para el modelo alternativo, este gráfico quizas confunde un poco los rendimientos y si lo vemos rapidamente podemos llegar a pensar que quizas el MontyHall de alguna forma predice mejor que el Alternativo. Esto no es así! Si nos fijamos en el eje Y y vemos las escalas, podemos entender que el modelo Alternativo funciona mejor que el MontyHall. Sin embargo, esto no es así principalmente porque el modelo Alternativo nunca se sorprende demasiado. Es decir que no tiene ninguna combinación de (c, s, r) a la cual le asigne probabilidad 0. Esto en el grafico parece que es así pero en realidad son predicciones con muy baja probabilidad y no exactamente 0. Esto sucede cuando el presentador se olvida de las reglas en los primeros episodioos, luego el modelo aprende que puede pasar eso y la probabilidad nunca es 0.
