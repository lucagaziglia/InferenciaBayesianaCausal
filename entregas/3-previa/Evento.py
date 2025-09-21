# -*- coding: utf-8 -*-
import math
import numpy as np
from Gaussiana import *

# El desvío estándar del desempeño: N(desempeño | mu=habilidad, sd=BETA).
BETA = 1.0


class Evento(object):
    #
    # Constructor
    def __init__(self, equipos):
        # Ejemplo:
        # equipos = [ [priorA, priorB], [priorC]  ]
        # donde el orden indica qué equipo ganó.
        self.equipos = equipos
    #
    # Representación

    def __repr__(self):
        return f'{self.equipos}'
    #

    @property
    def desempeño_individuos(self):
        # Genera todos los desempeños sumando ruido a las habilidades
        res = []
        ruido = Gaussian(0, BETA)
        for equipo in self.equipos:
            res.append([])  # Contenedor de equipo
            for habilidad in equipo:
                res[-1].append(habilidad + ruido)
        return (res)
    #

    @property
    def desempeño_equipos(self):
        # Suma de los desempeños individuales
        res = []
        for equipo in self.desempeño_individuos:
            ta = suma(equipo)
            res.append(ta)
        return res
    #

    @property
    def diferencia_equipos(self):
        # Diferencia de desempeños de los equipos
        ta, tb = self.desempeño_equipos
        d = ta - tb
        return d

    @property
    def marginal_diferencia(self):
        # Marginal de la diferencia: distribución truncada (d > 0)
        d = self.diferencia_equipos
        d_aprox = d > 0   # usa __gt__ de Gaussian → truncnorm
        return d_aprox

    @property
    def likelihood_diferencia(self):
        # Likelihood = truncada / original
        d = self.diferencia_equipos
        d_aprox = self.marginal_diferencia
        likelihood = d_aprox / d   # usa __truediv__ de Gaussian
        return likelihood

    @property
    def likelihood_equipos(self):
        # I(d = ta - tb) Diferencia entre los desempeños de los equipos
        ta, tb = self.desempeño_equipos
        likelihood_d = self.likelihood_diferencia
        likelihood_ta = tb + likelihood_d
        likelihood_tb = ta - likelihood_d
        res = [likelihood_ta, likelihood_tb]
        return res
    #

    @property
    def likelihood_desempeño(self):  # p(desempeño | resultado)
        evento = self
        desempeño_individuos = evento.desempeño_individuos
        likelihood_equipos = evento.likelihood_equipos

        res = []
        for e, equipo in enumerate(desempeño_individuos):
            res.append([])
            lk_te = likelihood_equipos[e]
            for i in range(len(equipo)):
                # suma de los desempeños de los otros integrantes
                others = [equipo[j] for j in range(len(equipo)) if j != i]
                # <-- usar suma(...) o construir con Gaussian(0,0)+...
                te_sin_i = suma(others)
                # mensaje al desempeño individual i: lk_te - te_sin_i
                lh_p_i = lk_te - te_sin_i
                res[-1].append(lh_p_i)
        return res

    @property
    def likelihood_habilidad(self):
        res = []
        ruido = Gaussian(0, BETA)
        for equipo in self.likelihood_desempeño:
            res.append([])
            for lh_p in equipo:
                lh_s = lh_p - ruido
                res[-1].append(lh_s)
        return res
    #

    @property
    def posterior(self):
        evento = self
        likelihood = evento.likelihood_habilidad
        prior = evento.equipos
        res = []
        for e in range(len(prior)):
            res.append([])
            for i in range(len(prior[e])):
                prior_i = prior[e][i]
                lh_i = likelihood[e][i]
                posterior_i = prior_i * lh_i
                res[-1].append(posterior_i)
        return res


priorA = Gaussian(3, 1)
priorB = Gaussian(2, 1)
priorC = Gaussian(6, 1)
Equipo1, Equipo2 = Evento([[priorA, priorB], [priorC]]).posterior
print(Equipo1)
# [N(mu=3.439, sigma=0.938), N(mu=2.439, sigma=0.938)]
print(Equipo2)
# [N(mu=5.561, sigma=0.938)]
