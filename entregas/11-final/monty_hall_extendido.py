#%%
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

## Modelo alternativo


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


#%%
df_monty = pd.read_csv('../../materiales_del_curso/11-final/datos/NoMontyHall.csv')
Datos = list(df_monty.iloc[:2000].itertuples(index=False, name=None))
Datos[0]
#%%
evidencias =[]
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

print(df_evidencia.head(10))
print(df_evidencia.shape)
# %%
plt.figure(figsize=(12,5))

#plt.plot(df_evidencia['Base'], label="M0 Base")
#plt.plot(df_evidencia['MontyHall'], label="M1 Monty")
plt.plot(df_evidencia['Alternativo'], label="M2 Alternativo")

plt.xlabel("Episodio")
plt.ylabel("Evidencia P(c,s,r | M)")
plt.title("Evidencia por episodio según cada modelo")
plt.legend()
plt.yscale("log")  # MUY recomendable: las probabilidades son muy pequeñas
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.hist(df_evidencia['Base'], bins=30)
plt.title("Base (M0)")
plt.yscale("log")

plt.subplot(1,3,2)
plt.hist(df_evidencia['MontyHall'], bins=30)
plt.title("Monty Hall (M1)")
plt.yscale("log")

plt.subplot(1,3,3)
plt.hist(df_evidencia['Alternativo'], bins=30)
plt.title("Alternativo (M2)")
plt.yscale("log")

plt.tight_layout()
plt.show()

#%%

MARKDOWN

#%% Ejercicio 2


'''La empresa quiere conocer el efecto causal que el monto credito tiene sobre el volumen de
ventas del mes (ventas). Sin embargo, hasta ahora no se ha realizado ning ́un experimento aleato-
rizado (de tipo A/B testing) para evaluar el efecto causal, por lo que nos piden hacer un an ́alisis
de efecto causal analizando  ́unicamente los datos observados. El problema es que no conocemos
la estructura causal subyacente, ni las variables forman parte de esa estructura causal. Pero en la
empresa nos insisten que quieren conocer el efecto causal. Y como empleados nos vemos obligados
a reportar un n ́umero que sea lo m ́as razonable posible.'''
# %% 1er paso de la inferencia causal: Definir el DAG
import graphviz

# Configuración del gráfico
dag = graphviz.Digraph(comment='DAG Créditos Final (Un Mes)', format='png')
dag.attr(rankdir='TB') # De arriba a abajo

# --- 1. Nodos Observados ---
dag.node('Ventas', 'Ventas (Outcome)', shape='ellipse', style='filled', color='lightblue')
dag.node('Monto', 'Monto Crédito (Treatment)', shape='ellipse', style='filled', color='lightgrey')

# Variables Intermedias / Negocio
dag.node('Plazo', 'Plazo')
dag.node('Riesgo', 'Nivel Riesgo')
dag.node('Compras', 'Compras Pasadas')

# Covariables (Sin Tiempo/Mes)
dag.node('Sector', 'Sector')
dag.node('Antiguedad', 'Antigüedad')

# --- 2. Variables NO Observadas (U) ---
dag.attr('node', style='dashed', color='gray')
dag.node('U_Riesgo', 'U_Riesgo')
dag.node('U_Plazo', 'U_Plazo')
dag.node('U_Compras', 'U_Compras')
dag.node('U_Ventas', 'U_Ventas')

# --- 3. Relaciones (Aristas) ---

# TUS RELACIONES PERSONALIZADAS:
# "Antigüedad determina Sector" (Tu hipótesis)
dag.edge('Antiguedad', 'Sector') 

# "Sector y Antigüedad determinan Compras Pasadas"
dag.edge('Sector', 'Compras')
dag.edge('Antiguedad', 'Compras')

# RELACIONES DEL MODELO DE CRÉDITO (Inputs de Monto):
dag.edge('Plazo', 'Monto')
dag.edge('Riesgo', 'Monto')
dag.edge('Compras', 'Monto')
dag.edge('Antiguedad', 'Monto') # Regla de negocio directa

# ESTRUCTURA CAUSAL (Hacia Ventas):
dag.edge('Monto', 'Ventas')       # <--- Lo que queremos medir
dag.edge('Plazo', 'Ventas')       # El plazo afecta ventas independientemente del monto
dag.edge('Sector', 'Ventas')      # El sector afecta ventas
dag.edge('Antiguedad', 'Ventas')  # La antigüedad afecta ventas

# RELACIONES INTERNAS:
dag.edge('Riesgo', 'Plazo')       # El riesgo define el plazo

# FACTORES NO OBSERVADOS (RUIDO):
dag.edge('U_Riesgo', 'Riesgo')
dag.edge('U_Plazo', 'Plazo')
dag.edge('U_Compras', 'Compras')
dag.edge('U_Ventas', 'Ventas')

# Renderizar
dag.render('dag_creditos_final_sin_tiempo', view=True)
# %%
import pandas as pd
df_creditos = pd.read_csv("C:/Users/WildFi/Desktop/InferenciaBayesianaCausal/materiales_del_curso/11-final/datos/data_credito.csv")
df_creditos.columns

# %% 2do paso de la inferencia causal: Identificación del estimando
import dowhy
from dowhy import CausalModel
import pandas as pd

# 1. Cargar tus datos (Asumiendo que ya tienes el DF cargado como 'df')
# Si no, descomenta la linea de abajo:
# df = pd.read_csv('data_credito.csv')

# 2. Definir el DAG en formato texto (DOT format)
# ¡OJO! Aquí uso EXATAMENTE los nombres de tus columnas:
# 'compras_pasadas', 'nivel_riesgo', 'plazo', 'monto_credito', 'sector', 'antiguedad', 'ventas'

dag_str = """digraph {
    /* Relaciones de Negocio / Demográficas */
    antiguedad -> sector;
    antiguedad -> compras_pasadas;
    sector -> compras_pasadas;
    
    /* Inputs del Modelo de Crédito (Determinan el Monto) */
    plazo -> monto_credito;
    nivel_riesgo -> monto_credito;
    compras_pasadas -> monto_credito;
    antiguedad -> monto_credito;
    
    /* Relación interna del banco */
    nivel_riesgo -> plazo;

    /* Relaciones Causa-Efecto hacia VENTAS (Outcome) */
    monto_credito -> ventas;  /* <--- Nuestra hipótesis causal */
    
    /* Confusores (Variables que afectan ventas independientemente del monto) */
    plazo -> ventas;
    sector -> ventas;
    antiguedad -> ventas;
}"""

# 3. Inicializar el Modelo Causal
model = CausalModel(
    data=df_creditos,
    treatment='monto_credito',
    outcome='ventas',
    graph=dag_str
)

# 4. Visualizar el modelo (Opcional, para verificar que DoWhy entendió el gráfico)
model.view_model() 

# 5. IDENTIFICACIÓN DEL ESTIMANDO
print("--- Buscando caminos causales... ---")
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

print(identified_estimand)
#%%
'''
Markdown explicando porque esta bien que haya solo por backdor.

'''

# %% 3er paso de la inferencia causal: Estimación del efecto causal

import pandas as pd
import numpy as np
import dowhy
from dowhy import CausalModel
import statsmodels.api as sm # Lo usaremos para extraer los coeficientes limpiamente

# 1. Carga y Preprocesamiento
# ---------------------------------------------------------
# Asumimos que df ya está cargado. Si no: df = pd.read_csv('data_credito.csv')
# ¡IMPORTANTE! Convertimos 'sector' a números (One-Hot Encoding) antes del modelo
df_model = pd.get_dummies(df_creditos, columns=['sector'], drop_first=True, dtype=int)
df_model
#%%
# 2. Redefinir el Modelo Causal con los datos numéricos
# ---------------------------------------------------------
# Dowhy necesita saber que 'plazo' modifica el efecto (Effect Modifier)
model = CausalModel(
    data=df_model,
    treatment='monto_credito',
    outcome='ventas',
    # Usamos common_causes para listar los confusores numéricos
    # (Incluimos todas las columnas de sector_X que se crearon)
    common_causes=['antiguedad', 'plazo', 'compras_pasadas'] + [c for c in df_model.columns if 'sector_' in c],
    effect_modifiers=['plazo'] # <--- ¡ESTO ES CLAVE PARA EL CATE!
)

# 3. Identificación (Repetimos brevemente con el nuevo setup)
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

# 4. Estimación (Usando Linear Regression)
# ---------------------------------------------------------
# Le pedimos un modelo lineal customizado que incluya la interacción
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression",
    test_significance=True
)

print("\n--- Resultado de la Estimación ---")
print(f"Efecto Promedio (ATE): {estimate.value}")

# --- PARTE CLAVE: Extraer coeficientes para las funciones del examen ---
# DoWhy usa statsmodels internamente. Accedemos al modelo entrenado:
model_results = estimate.estimator.model
params = model_results.params
# Imprimimos el resumen completo de la regresión
# Esto mostrará una tabla con "x1", "x2" y (esperemos) su descripción
# --- EXTRACCIÓN ROBUSTA POR POSICIÓN ---

# 1. Obtenemos los valores numéricos de los coeficientes
# params es una Serie de pandas. .values nos da el array numpy puro.
coefs_values = estimate.estimator.model.params.values

print("Valores de los coeficientes encontrados:", coefs_values)

# 2. Asignamos por posición estándar
# Posición 0: const
# Posición 1: x1 -> Tratamiento (Monto)
# Última Posición: x9 -> Interacción (Monto * Plazo)

beta_monto = coefs_values[1]       # x1 (-0.0839 según tu foto)
beta_interaccion = coefs_values[-1] # El último de la lista (0.0100 según tu foto anterior)

print(f"✅ Beta Monto (x1): {beta_monto}")
print(f"✅ Beta Interacción (último): {beta_interaccion}")

# 3. Definimos las funciones con estos valores
def CATE(plazo: int) -> float:
    # Ecuación: Efecto = Beta_Monto + (Beta_Interaccion * Plazo)
    return beta_monto + (beta_interaccion * plazo)

def ATE(mean_plazo: float = None) -> float:
    if mean_plazo is None:
        # Si no hay data cargada, usaremos un valor dummy o el del df si existe
        try:
            mean_plazo = df['plazo'].mean()
        except:
            mean_plazo = 30 # Valor por defecto si no hay dataframe
    return CATE(mean_plazo)


# --- Verificación ---
print("\n--- Respuestas para el Examen ---")
print(f"Si el plazo es 10 días, por cada $1 extra de crédito, las ventas suben: ${CATE(10):.4f}")
print(f"Si el plazo es 60 días, por cada $1 extra de crédito, las ventas suben: ${CATE(60):.4f}")
print(f"Efecto Promedio Global (ATE): ${ATE():.4f}")

# %% 4to paso de la inferencia causal: Refutación
# --- PASO 4: REFUTACIÓN (Validación de Robustez) ---

print("\n--- Iniciando Refutación del Modelo ---")

# Prueba 1: Placebo Refuter
# Reemplazamos el 'monto_credito' real por uno aleatorio.
# Si el modelo es bueno, el nuevo efecto estimado debería ser CERO (o muy cercano).
ref_placebo = model.refute_estimate(
    identified_estimand, 
    estimate,
    method_name="placebo_treatment_refuter"
)

print("\n1. Prueba Placebo (Esperamos 'New Effect' cercano a 0):")
print(ref_placebo)

# Prueba 2: Random Common Cause
# Agregamos una variable aleatoria al dataset que actúe como causa común.
# El 'New Effect' debería mantenerse muy parecido al 'Original Effect' (-0.046).
ref_random = model.refute_estimate(
    identified_estimand, 
    estimate,
    method_name="random_common_cause"
)

print("\n2. Prueba Causa Común Aleatoria (Esperamos que el efecto no cambie mucho):")
print(ref_random)

# %%
