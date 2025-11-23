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

