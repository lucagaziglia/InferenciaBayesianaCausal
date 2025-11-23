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

plt.figure(figsize=(12, 5))

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
         label=f"modelo de grado {mod_max_verosimilitud}", alpha=0.8)

plt.ylim(-1.5, 1.5)
plt.title(
    f"Ajuste del modelo seleccionado por máxima verosimilitud (grado {mod_max_verosimilitud})")
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

denominador = np.nansum(
    [np.exp(ml.log_evidence(Y, phi(X, d))[0][0]) * p_M for d in range(D)])

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

# %% 1.6

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
    plt.plot(x_grid, y_hat,
             label=f'Modelo Bayesiano d={d}', alpha=0.7, linewidth=1)
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
    var = (1.0 / BETA) + float(phi_x @
                               # Var[y* | ...]
                               modelos_BAY[d]['cov'] @ phi_x.T)
    sigma = np.sqrt(var)
    distributions[d] = (mu, sigma)

# Graficar densidades normales p(y | x=-0.23, M_d)
plt.figure(figsize=(7, 5))
labels = {0: "Rígido (grado 0)", 3: "Simple (grado 3)",
          9: "Complejo (grado 9)"}

for d, (mu, sigma) in distributions.items():
    grid = np.linspace(mu - 4*sigma, mu + 4*sigma, 400)
    pdf = (1.0 / (np.sqrt(2*np.pi) * sigma)) * \
        np.exp(-0.5 * ((grid - mu)/sigma)**2)
    plt.plot(grid, pdf, linewidth=2, label=labels[d])

plt.xlabel(r"$y \mid x=-0.23$")
plt.ylabel(r"$P(\text{Dato}\mid \text{Modelo } d)$")
plt.title("Predicción en x = -0.23")
plt.legend()
plt.tight_layout()
plt.show()
# %%
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
labels = {0: "Rígido (grado 0)", 3: "Simple (grado 3)",
          9: "Complejo (grado 9)"}
for d, (mu, sigma) in distributions.items():
    grid = np.linspace(mu - 4*sigma, mu + 4*sigma, 400)
    pdf = (1.0 / (np.sqrt(2*np.pi) * sigma)) * \
        np.exp(-0.5 * ((grid - mu)/sigma)**2)
    plt.plot(grid, pdf, linewidth=2, label=labels[d])
plt.axvline(x=y_true, color='k', linestyle='--', label='Valor real')
plt.xlabel(r"$y \mid x=0.1$")
plt.ylabel(r"$P(\text{Dato}\mid \text{Modelo } d)$")
plt.title("Predicción en x = 0.1")
plt.legend()
plt.tight_layout()
plt.show()

<<<<<<< HEAD
# %%
print("7)")
=======
#%% 2.1) 
>>>>>>> d9815ad4f87517116fb4f8fb6917f747c3a56894

Alturas = pd.read_csv("datos/alturas.csv")
Alturas.head()

hombres = Alturas[Alturas.sexo == 'M']
mujeres = Alturas[Alturas.sexo == 'F']
plt.figure(figsize=(8, 5))
plt.scatter(hombres.altura_madre, hombres.altura, c='blue', label='Hombres', alpha=0.6)
plt.scatter(mujeres.altura_madre, mujeres.altura, c='red', label='Mujeres', alpha=0.6)
plt.xlabel('Altura de la madre (cm)') 
plt.ylabel('Altura del hijo/a (cm)')
plt.title('Alturas de madres e hijos/as')
plt.legend()
plt.show()

<<<<<<< HEAD
for d in [0, 3, 9]:
    phi_x = phi(np.array([[x_val]]), d).values
    mu = float(phi_x @ modelos_BAY[d]['mean'])
    var = (1.0 / BETA) + float(phi_x @
                               # Var[y* | ...]
                               modelos_BAY[d]['cov'] @ phi_x.T)
    sigma = np.sqrt(var)
    distributions[d] = (mu, sigma)

# Graficar densidades normales p(y | x=0.1, M_d)
plt.figure(figsize=(7, 5))
labels = {0: "Rígido (grado 0)", 3: "Simple (grado 3)",
          9: "Complejo (grado 9)"}
for d, (mu, sigma) in distributions.items():
    grid = np.linspace(mu - 4*sigma, mu + 4*sigma, 400)
    pdf = (1.0 / (np.sqrt(2*np.pi) * sigma)) * \
        np.exp(-0.5 * ((grid - mu)/sigma)**2)
    plt.plot(grid, pdf, linewidth=2, label=labels[d])
plt.axvline(x=y_true, color='k', linestyle='--', label='Valor real')
plt.xlabel(r"$y \mid x=0.1$")
plt.ylabel(r"$P(\text{Dato}\mid \text{Modelo } d)$")
plt.title("Predicción en x = 0.1")
=======
#%% 2.2)

# Modelo base

N, _ = Alturas.shape
Y_alturas = Alturas.altura
X_base = pd.DataFrame({"Base": [1 for _ in range(N)],    # Origen
                       "Altura": Alturas.altura_madre,  # Pendiente
                       })

log_evidence_base = ml.log_evidence(Y_alturas, X_base)
MU_base, COV_base = ml.posterior(Y_alturas, X_base)

df_m = Alturas[Alturas.sexo == 'M']
df_f = Alturas[Alturas.sexo == 'F']

x_grid = np.linspace(Alturas.altura_madre.min(), Alturas.altura_madre.max(), 200)
X_grid = pd.DataFrame({"Base": np.ones_like(x_grid), "Altura": x_grid})

mu_pred_base = (X_grid.values @ MU_base).astype(float)

# --- plot coloreado por sexo + modelo ---
plt.figure(figsize=(7,5))
plt.scatter(df_m.altura_madre, df_m.altura, s=16, alpha=0.7, label="Hombres")
plt.scatter(df_f.altura_madre, df_f.altura, s=16, alpha=0.7, label="Mujeres")
plt.plot(x_grid, mu_pred_base, lw=2, label="Modelo base (media posterior)")
plt.xlabel("Altura madre")
plt.ylabel("Altura descendencia")
>>>>>>> d9815ad4f87517116fb4f8fb6917f747c3a56894
plt.legend()
plt.tight_layout()
plt.show()

<<<<<<< HEAD
# %% [markdown]
# # Ejercicio 2
# ### Efecto causal del sexo biol ́ogico sobre la altura.

Alturas = pd.read_csv("alturas.csv")
Alturas.head()
hombres = Alturas[Alturas["sexo"] == "M"]
mujeres = Alturas[Alturas["sexo"] == "F"]

plt.figure(figsize=(7, 5))
plt.scatter(hombres.altura_madre, hombres.altura,
            alpha=0.5, c="blue", label="Hombres")
plt.scatter(mujeres.altura_madre, mujeres.altura,
            alpha=0.5, c="red", label="Mujeres")

plt.xlabel("Altura madre (cm)")
plt.ylabel("Altura hijo (cm)")
plt.title("Altura de hijos vs altura de madres")
plt.legend()
plt.show()
# %%
N, _ = Alturas.shape
Y_alturas = Alturas.altura
X_base = pd.DataFrame({"Base": [1 for _ in range(N)],    # Origen
                       "Altura": Alturas.altura_madre,  # Pendiente
                       })


log_evidence_base = ml.log_evidence(Y_alturas, X_base)


# %%
modelos_BAY = []
for d in range(D):
    MU_d, COV_d = ml.posterior(Y, phi(X, d))
    log_evidence_d = ml.log_evidence(Y, phi(X, d))[0][0]
    modelos_BAY.append({"mean": MU_d.reshape(
        1, d+1)[0], "cov": COV_d, "log_evidence": log_evidence_d})

=======
#%%

print("Modelo biológico")

X_bio = pd.DataFrame({
    "Base": np.ones(N),
    "Altura": Alturas.altura_madre,
    "Indicadora": Alturas.sexo.apply(lambda x: 1 if x == 'F' else 0)
})

MU_bio, COV_bio = ml.posterior(Y_alturas, X_bio)
log_evidence_bio = ml.log_evidence(Y_alturas, X_bio)

# grids separados para H (0) y F (1)
x_grid = np.linspace(Alturas.altura_madre.min(), Alturas.altura_madre.max(), 200)
X_grid_bio_H = pd.DataFrame({"Base": np.ones_like(x_grid), "Altura": x_grid, "Indicadora": np.zeros_like(x_grid)})
X_grid_bio_F = pd.DataFrame({"Base": np.ones_like(x_grid), "Altura": x_grid, "Indicadora": np.ones_like(x_grid)})

mu_pred_bio_H = (X_grid_bio_H.values @ MU_bio).astype(float)
mu_pred_bio_F = (X_grid_bio_F.values @ MU_bio).astype(float)

# --- plot coloreado por sexo + líneas del modelo biológico ---
plt.figure(figsize=(7,5))
plt.scatter(df_m.altura_madre, df_m.altura, s=16, alpha=0.7, label="Hombres")
plt.scatter(df_f.altura_madre, df_f.altura, s=16, alpha=0.7, label="Mujeres")
plt.plot(x_grid, mu_pred_bio_H, lw=2.5, label="Modelo biológico (H)")
plt.plot(x_grid, mu_pred_bio_F, lw=2.5, label="Modelo biológico (F)")
plt.xlabel("Altura madre")
plt.ylabel("Altura descendencia")
plt.legend()
plt.tight_layout()
plt.show()

#%%
print("Modelo identitario")

X_id = pd.DataFrame({
    "Base": np.ones(N),
    "Altura": Alturas.altura_madre,
})

indices = np.arange(N)
np.random.shuffle(indices)  
# Dividimos en grupos de 2
grupos = np.array_split(indices, 25)

# Creamos columnas dummy
for i, grupo in enumerate(grupos):
    col = np.zeros(N)
    col[grupo] = 1
    X_id[f"G{i+1}"] = col


MU_id, COV_id = ml.posterior(Y_alturas, X_id)
log_evidence_id = ml.log_evidence(Y_alturas, X_id)

x_grid = np.linspace(Alturas.altura_madre.min(), Alturas.altura_madre.max(), 200)

n_grupos = 25
cols = ["Base", "Altura"] + [f"G{i+1}" for i in range(n_grupos)]

X_grids = []
for g in range(n_grupos):
    df = pd.DataFrame(0.0, index=np.arange(len(x_grid)), columns=cols)
    df["Base"] = 1.0
    df["Altura"] = x_grid
    df[f"G{g+1}"] = 1.0
    X_grids.append(df.values)

X_stack = np.vstack(X_grids)
mu_pred_por_grupo = (X_stack @ MU_id).astype(float).reshape(n_grupos, -1)

# --- plot coloreado por sexo + líneas del modelo identitario ---
plt.figure(figsize=(7,5))
plt.scatter(df_m.altura_madre, df_m.altura, s=16, alpha=0.7, label="Hombres")
plt.scatter(df_f.altura_madre, df_f.altura, s=16, alpha=0.7, label="Mujeres")
for g in range(n_grupos):
    plt.plot(x_grid, mu_pred_por_grupo[g], lw=1, alpha=0.4, label=f"Grupo {g+1}" if g < 1 else "")
plt.xlabel("Altura madre")
plt.ylabel("Altura descendencia")
plt.legend()
plt.tight_layout()
plt.show()

#%% 2.3)
print("Comparación de modelos (log-evidencia)")

df_log_evidences = pd.DataFrame({
    "Base": [log_evidence_base],
    "Biológico": [log_evidence_bio],
    "Identitario": [log_evidence_id]})

plt.figure(figsize=(6,4))
plt.bar(df_log_evidences.columns, df_log_evidences.iloc[0], color=plt.cm.tab10.colors)
plt.ylabel("Log-evidencia")
plt.title("Comparación de modelos")
plt.tight_layout()
plt.show()


#%% 2.5)

print("Posterior de los modelos")

import numpy as np

# log evidences (en log natural)
log_evidences = df_log_evidences.iloc[0].values
# prior uniforme
# prior uniforme: 1/3  --> en log:
log_prior = np.log(1/3)

# numerador en log para cada modelo: log P(D|M_i) + log P(M_i)
z = log_evidences + log_prior

# denominador en log: log P(D) con log-sum-exp
max_z = np.max(z)
log_PDatos = max_z + np.log(np.sum(np.exp(z - max_z)))

# posterior en log y en probas
log_posteriors = z - log_PDatos
posteriors = np.exp(log_posteriors)

print("Posteriores (Base, Bio, ID):", posteriors)
#%%
log_evidence_base
>>>>>>> d9815ad4f87517116fb4f8fb6917f747c3a56894

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

Alturas = pd.read_csv("alturas.csv")
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

# %%
