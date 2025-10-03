# %%
import scipy
import ModeloLineal as ml

from statsmodels.api import OLS  # Para selección de hipótesis
from scipy.stats import norm    # La distribución gaussiana
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({'font.size': 14})
np.random.seed(1)  # Para reproducir los datos

cmap = plt.get_cmap("tab10")  # Colores para los modelos
# ###########


# Ejercicio 1
# %% 1.1 Generar 20 datos alrededor de una per ́ıodo de una sinoidal

N = 20  # Cantidad de datos
D = 10  # Cantidad de modelos (de 0 al 9)

BETA = (1/0.04)  # Precisión de los datos, el inverso de su varianza
ALPHA = (10e-6)  # Precisión de la creencia a prior, el inverso de su varianza

# Realidad causal subyacente


def realidad_causal_subyacente(X, beta=BETA):
    return np.sin(2 * np.pi * X) + np.random.normal(0, np.sqrt(1/beta), X.shape)

# Las transformaciones de X que hace el modelo de grado D


def phi(X, complejidad=D):
    return (pd.DataFrame({f'X{d}': X[:, 0]**d for d in range(complejidad+1)}))


X = np.random.rand(N, 1)-0.5
Y = realidad_causal_subyacente(X)

x_grid = np.linspace(-0.5, 0.5, 200).reshape(-1, 1)
y_true = np.sin(2*np.pi*x_grid)

plt.figure(figsize=(7, 5))
plt.plot(x_grid, y_true, "k--", label="Función objetivo: sin(2πx)")
plt.scatter(X, Y, c="red", label="Datos con ruido")
plt.title("Ejercicio 1.1: Datos simulados alrededor de una sinusoide")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# %% 1.2 Graficar el valor de m ́axima verosimilitud obtenido por los modelos polinomiales de grado 0 a 9

D = 10  # Cantidad de modelos (de 0 al 9)

# Itero por modelos Md
modelos_OLS = []
for d in range(0, D):
    # Ajusto el modelo de compeljidad d
    modelos_OLS.append(OLS(Y, phi(X, d)).fit())

loglikes = [m.llf for m in modelos_OLS]

plt.figure(figsize=(6, 4))
plt.bar(range(len(loglikes)), np.exp(
    loglikes - np.max(loglikes)), color=plt.cm.tab10.colors)
plt.xlabel("Modelos (grado del polinomio)")
plt.ylabel("Máxima verosimilitud (escala relativa)")
plt.title("Máxima verosimilitud por modelo")
plt.show()

# %% 1.3 Graficar la función ajustada por los modelos de grado 0 a 9
print("a) Ajuste de modelos polinomiales de grado 0 a 9")

plt.figure(figsize=(8, 5))

# Datos originales
plt.scatter(X, Y, c="red", label="Datos con ruido")

# Función objetivo
plt.plot(x_grid, y_true, "k--", label="Función objetivo: sin(2πx)")

# Ajustes polinomiales de grado 0 a 9
for d in range(D):
    y_pred = modelos_OLS[d].predict(phi(x_grid, d))
    plt.plot(x_grid, y_pred, label=f"Grado {d}", alpha=0.8)

plt.ylim(-1.5, 1.5)
plt.title("Ajuste de modelos polinomiales (grados 0–9)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

print("b) Selección de hipótesis por máxima verosimilitud")
mod_max_verosimilitud = np.array(loglikes).argmax()
y_pred = modelos_OLS[mod_max_verosimilitud].predict(
    phi(x_grid, mod_max_verosimilitud))

plt.figure(figsize=(8, 5))
plt.scatter(X, Y, c="red", label="Datos con ruido")
plt.plot(x_grid, y_true, "k--", label="Función objetivo: sin(2πx)")
plt.plot(x_grid, y_pred,
         label="modelo de grado {mod_max_verosimilitud}", alpha=0.8)

plt.ylim(-1.5, 1.5)
plt.title(
    "Ajuste del modelo seleccionado por máxima verosimilitud (grado {mod_max_verosimilitud})")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()

# %% 1.4

all_likelihoods = []


def log_likelihood(X, Y, d):
    phi_d = np.array(phi(X, d))
    loglikelihood = []
    for n in range(1, N):
        phi_n = phi_d[0:n]
        Y_n = Y[0:n]
        model_d = OLS(Y_n, phi_n).fit()
        x = phi_d[n]
        y = Y[n]

        y_hat = float(model_d.predict(x))
        V = float(model_d.mse_resid) * (1.0 + x @
                                        model_d.normalized_cov_params @ x.T)
        loglike = -0.5*np.log(2*np.pi*V) - 0.5*((y - y_hat)**2 / V)
        loglikelihood.append(loglike)
    return loglikelihood


for d in range(D):
    log_like = log_likelihood(X, Y, d)
    all_likelihoods.append(np.array(log_like))

rendimiento_modelos = {}

for i, like in enumerate(all_likelihoods):
    pDatos_Modelo = np.nansum(like)
    rendimiento_modelos[i] = pDatos_Modelo
rendimiento_df = pd.DataFrame.from_dict(
    rendimiento_modelos, orient='index', columns=['loglike'])
rendimiento_df

plt.figure(figsize=(6, 4))
plt.bar(rendimiento_df.index, np.exp(rendimiento_df.loglike -
        np.max(rendimiento_df.loglike)), color=plt.cm.tab10.colors)
plt.xlabel("Modelos (grado del polinomio)")
plt.ylabel("Producto predictivo (escala relativa)")
plt.title("Producto predictivo por modelo")
plt.show()


# %% 1.5
p_M = 0.1
posteriors = {}

denominador = np.nansum([np.exp(ml.log_evidence(Y, phi(X, d))[0][0]) * p_M for d in range(D)])

for d in range(D):
    evidencia = np.exp(ml.log_evidence(Y, phi(X, d))[0][0])
    posterior = evidencia * p_M / denominador
    posteriors[d] = posterior

posteriors_df = pd.DataFrame.from_dict(
    posteriors, orient='index', columns=['posterior']
)

plt.figure(figsize=(6, 4))
plt.bar(posteriors_df.index, posteriors_df.posterior, color=plt.cm.tab10.colors)
plt.xlabel("Modelos (grado del polinomio)")
plt.ylabel("Posterior (escala relativa)")
plt.title("Posterior por modelo")
plt.show()

#%% 1.6

modelos_OLS[1].params  # Las hipótesis seleccionadas
# El likelihood de la hipótesis seleccionada (en escala log).
modelos_OLS[1].llf

modelos_BAY = []
for d in range(D):
    MU_d, COV_d = ml.posterior(Y, phi(X, d))
    log_evidence_d = ml.log_evidence(Y, phi(X, d))[0][0]
    modelos_BAY.append({"mean": MU_d.reshape(
        1, d+1)[0], "cov": COV_d, "log_evidence": log_evidence_d})


print("6.a) ")

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, label='datos', zorder=5)

for d, m in enumerate(modelos_BAY):
    phi_d = phi(x_grid, d)
    y_hat = phi_d.values.dot(m['mean'])
    plt.plot(x_grid, y_hat, label=f'Modelo Bayesiano d={d}', alpha=0.7, linewidth=1)
plt.title('Media del posterior con priori no informativo del 0 al 9')
plt.ylim(-1.5, 1.5)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()

print("6.b)")

x_val = -0.23
distributions = {}

for d in [0, 3, 9]:
    phi_x = phi(np.array([[x_val]]), d).values      
    mu = float(phi_x @ modelos_BAY[d]['mean'])      
    var = (1.0 / BETA) + float(phi_x @ modelos_BAY[d]['cov'] @ phi_x.T)  # Var[y* | ...]
    sigma = np.sqrt(var)
    distributions[d] = (mu, sigma)

# Graficar densidades normales p(y | x=-0.23, M_d)
plt.figure(figsize=(7, 5))
labels = {0: "Rígido (grado 0)", 3: "Simple (grado 3)", 9: "Complejo (grado 9)"}

for d, (mu, sigma) in distributions.items():
    grid = np.linspace(mu - 4*sigma, mu + 4*sigma, 400)
    pdf = (1.0 / (np.sqrt(2*np.pi) * sigma)) * np.exp(-0.5 * ((grid - mu)/sigma)**2)
    plt.plot(grid, pdf, linewidth=2, label=labels[d])

plt.xlabel(r"$y \mid x=-0.23$")
plt.ylabel(r"$P(\text{Dato}\mid \text{Modelo } d)$")
plt.title("Predicción en x = -0.23")
plt.legend()
plt.tight_layout()
plt.show()
#%%
print("7)")

x_val = 0.1
y_true = np.sin(2*np.pi*x_val)

distributions = {}

for d in [0, 3, 9]:
    phi_x = phi(np.array([[x_val]]), d).values
    mu = float(phi_x @ modelos_BAY[d]['mean'])
    # usar solo la parte de modelo: var = φ(x)^T Σ_d φ(x)
    var = float((phi_x @ modelos_BAY[d]['cov'] @ phi_x.T).squeeze())
    sigma = np.sqrt(var)
    distributions[d] = (mu, sigma)

# Graficar densidades normales p(y | x=0.1, M_d)
plt.figure(figsize=(7, 5))
labels = {0: "Rígido (grado 0)", 3: "Simple (grado 3)", 9: "Complejo (grado 9)"}
for d, (mu, sigma) in distributions.items():
    grid = np.linspace(mu - 4*sigma, mu + 4*sigma, 400)
    pdf = (1.0 / (np.sqrt(2*np.pi) * sigma)) * np.exp(-0.5 * ((grid - mu)/sigma)**2)
    plt.plot(grid, pdf, linewidth=2, label=labels[d])
plt.axvline(x=y_true, color='k', linestyle='--', label='Valor real')
plt.xlabel(r"$y \mid x=0.1$")
plt.ylabel(r"$P(\text{Dato}\mid \text{Modelo } d)$")
plt.title("Predicción en x = 0.1")
plt.legend()
plt.tight_layout()
plt.show()

#%%
print("7)")

x_val = 0.1
y_true = np.sin(2*np.pi*x_val)

distributions = {}

for d in [0, 3, 9]:
    phi_x = phi(np.array([[x_val]]), d).values      
    mu = float(phi_x @ modelos_BAY[d]['mean'])      
    var = (1.0 / BETA) + float(phi_x @ modelos_BAY[d]['cov'] @ phi_x.T)  # Var[y* | ...]
    sigma = np.sqrt(var)
    distributions[d] = (mu, sigma)

# Graficar densidades normales p(y | x=0.1, M_d)
plt.figure(figsize=(7, 5))
labels = {0: "Rígido (grado 0)", 3: "Simple (grado 3)", 9: "Complejo (grado 9)"}
for d, (mu, sigma) in distributions.items():
    grid = np.linspace(mu - 4*sigma, mu + 4*sigma, 400)
    pdf = (1.0 / (np.sqrt(2*np.pi) * sigma)) * np.exp(-0.5 * ((grid - mu)/sigma)**2)
    plt.plot(grid, pdf, linewidth=2, label=labels[d])
plt.axvline(x=y_true, color='k', linestyle='--', label='Valor real')
plt.xlabel(r"$y \mid x=0.1$")
plt.ylabel(r"$P(\text{Dato}\mid \text{Modelo } d)$")
plt.title("Predicción en x = 0.1")
plt.legend()
plt.tight_layout()
plt.show()


# %%
#
# 3.1 Data
#

X = np.random.rand(N, 1)-0.5
Y = realidad_causal_subyacente(X)
# Grilla
X_grilla = np.linspace(0, 1, 100).reshape(-1, 1)-0.5
Y_grilla = realidad_causal_subyacente(X_grilla, np.inf)


# Ejercicio 2

Alturas = pd.read_csv("datos/alturas.csv")
Alturas.head()


N, _ = Alturas.shape
Y_alturas = Alturas.altura
X_base = pd.DataFrame({"Base": [1 for _ in range(N)],    # Origen
                       "Altura": Alturas.altura_madre,  # Pendiente
                       })

ml.log_evidence(Y_alturas, X_base)

# Ejercicio 3

M = 1000
z1 = np.random.uniform(-3, 3, size=M)
w1 = 3*z1 + np.random.normal(size=M, scale=1)
z2 = np.random.uniform(-3, 3, size=M)
w2 = 2*z2 + np.random.normal(size=M, scale=1)
z3 = -2*z1 + 2*z2 + np.random.normal(size=M, scale=1)
x = -1*w1 + 2*z3 + np.random.normal(size=M, scale=1)
w3 = 2*x + np.random.normal(size=M, scale=1)
y = 2 - 1*w3 - z3 + w2 + np.random.normal(size=M, scale=1)


X_3_1 = pd.DataFrame({
    "w_0": [1 for _ in range(M)],    # Origen
    "w_x": x,
    "w_z3": z3,
    # "w_w2": w2, # Hay que controlar por esta variable, no?
})

MU_3_1, COV_3_1 = ml.posterior(y, X_3_1)
MU_3_1


model_ols_3_1 = OLS(y, X_3_1).fit()
model_ols_3_1.summary()
MU_3_1_ols = model_ols_3_1.params
