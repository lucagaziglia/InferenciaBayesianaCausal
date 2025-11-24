# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: global
#     language: python
#     name: python3
# ---


# %% Ejercicio 2


'''La empresa quiere conocer el efecto causal que el monto credito tiene sobre el volumen de
ventas del mes (ventas). Sin embargo, hasta ahora no se ha realizado ning ́un experimento aleato-
rizado (de tipo A/B testing) para evaluar el efecto causal, por lo que nos piden hacer un an ́alisis
de efecto causal analizando  ́unicamente los datos observados. El problema es que no conocemos
la estructura causal subyacente, ni las variables forman parte de esa estructura causal. Pero en la
empresa nos insisten que quieren conocer el efecto causal. Y como empleados nos vemos obligados
a reportar un n ́umero que sea lo m ́as razonable posible.'''

import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from dowhy import CausalModel
import dowhy
import graphviz
import pandas as pd


df_creditos = pd.read_csv(
    "../../../InferenciaBayesianaCausal/materiales_del_curso/11-final/datos/data_credito.csv")

# %% 1er paso de la inferencia causal: Definir el DAG

dag = graphviz.Digraph(comment='DAG Créditos Final Modificado', format='png')
dag.attr(rankdir='TB')

dag.node('Ventas', 'Ventas (Outcome)', shape='ellipse',
         style='filled', color='lightblue')
dag.node('Monto', 'Monto Crédito (Treatment)',
         shape='ellipse', style='filled', color='lightgrey')

dag.node('Plazo', 'Plazo')
dag.node('Riesgo', 'Nivel Riesgo')
dag.node('Compras', 'Compras Pasadas')

dag.node('Sector', 'Sector')
dag.node('Antiguedad', 'Antigüedad')

dag.attr('node', style='dashed', color='gray')
dag.node('U_Riesgo', 'U_Riesgo')
dag.node('U_Plazo', 'U_Plazo')
dag.node('U_Compras', 'U_Compras')
dag.node('U_Ventas', 'U_Ventas')

dag.edge('Antiguedad', 'Sector')

dag.edge('Sector', 'Compras')
dag.edge('Antiguedad', 'Compras')

dag.edge('Plazo', 'Monto')
dag.edge('Riesgo', 'Monto')
dag.edge('Compras', 'Monto')

dag.edge('Monto', 'Ventas')
dag.edge('Plazo', 'Ventas')
dag.edge('Sector', 'Ventas')
dag.edge('Antiguedad', 'Ventas')

dag.edge('Compras', 'Ventas')
dag.edge('Riesgo', 'Ventas')

dag.edge('Riesgo', 'Plazo')

dag.edge('U_Riesgo', 'Riesgo')
dag.edge('U_Plazo', 'Plazo')
dag.edge('U_Compras', 'Compras')
dag.edge('U_Ventas', 'Ventas')

dag.render('dag_creditos_final_v4', view=True)

# %% [markdown]
# # Por qué modelo asi el problema?
# ##
#
# Tenemos las siguientes variables: [compras_pasadas, nivel_riesgo, plazo, monto_credito, sector, antiguedad, ventas]. Donde entiendo que las ventas depende de todas ellas, es por eso que todas las variables apuntan a ventas. Por otro lado, entiendo que la antigüedad influye en el sector y en las compras pasadas. Por último, como dice el enunciado, el monto del crédito esta calculado en función de un modelo de caja negra que depende del plazo, nivel de riesgo y compras pasadas. Es por eso que cada una de estas 3 variables tienen su propio ruido (U_plazo, U_riesgo, U_compras). Finalmente, U_ventas representa ese ruido inevitable en las ventas que desconocemos.

# %% 2do paso de la inferencia causal: Identificación del estimando


dag_str = """digraph {
    /* --- 1. Relaciones Demográficas y de Negocio --- */
    antiguedad -> sector;
    antiguedad -> compras_pasadas;
    sector -> compras_pasadas;
    
    /* --- 2. Inputs del Modelo de Crédito (Determinan el Monto) --- */
    plazo -> monto_credito;
    nivel_riesgo -> monto_credito;
    compras_pasadas -> monto_credito;
    
    /* CAMBIO 1: ELIMINADO antiguedad -> monto_credito */
    /* Ya no creemos que la antigüedad defina el monto directamente, 
       sino a través del sector o compras */
    
    /* --- 3. Relación interna del banco --- */
    nivel_riesgo -> plazo;

    /* --- 4. Relaciones hacia VENTAS (Outcome) --- */
    monto_credito -> ventas;  /* Hipótesis Causal */
    
    /* Confusores clásicos */
    plazo -> ventas;
    sector -> ventas;
    antiguedad -> ventas;
    
    /* CAMBIO 2: NUEVAS RELACIONES AGREGADAS */
    /* Ahora asumimos que estas variables afectan directamente cuánto vende el cliente */
    compras_pasadas -> ventas;
    nivel_riesgo -> ventas;
}"""

model = CausalModel(
    data=df_creditos,
    treatment='monto_credito',
    outcome='ventas',
    graph=dag_str
)

model.view_model()

print("--- Buscando caminos causales... ---")
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

print(identified_estimand)

# %% [markdown]
# # Uso Backdoor. Por qué?
# ##
#
# Para este caso, era esperable que se identifique el criterio backdoor, pero también aparecio el estimando: General Adjustment. Como en la materia el criterio que mas vimos fue el backdoor, voy a usar este criterio para estimar el efecto causal. Esta decisión también se basa en que ambos criterios llegan a la misma expresión matemática, que es la siguiente:
#
# $\frac{d}{d[\text{monto\_credito}]} E[\text{ventas} \mid \text{plazo}, \text{nivel\_riesgo}, \text{compras\_pasadas}, \text{sector}, \text{antiguedad}]$

# %% 3er paso de la inferencia causal: Estimación del efecto causal

df_model = pd.get_dummies(df_creditos, columns=[
                          'sector'], drop_first=True, dtype=int)

model = CausalModel(
    data=df_model,
    treatment='monto_credito',
    outcome='ventas',
    common_causes=['antiguedad', 'plazo', 'compras_pasadas', 'nivel_riesgo'] +
    [c for c in df_model.columns if 'sector_' in c],
    effect_modifiers=['plazo']
)

identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)


estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression",
    test_significance=True
)

estimate.get_confidence_intervals()

print("\n--- Resultado de la Estimación ---")
print(f"Efecto Promedio (ATE): {estimate.value:.4f}")

model_results = estimate.estimator.model
params = model_results.params
coefs_values = estimate.estimator.model.params.values

print("Valores de los coeficientes encontrados:", coefs_values)


beta_monto = coefs_values[1]
beta_interaccion = coefs_values[-1]

print(f"✅ Beta Monto (x1): {beta_monto}")
print(f"✅ Beta Interacción (último): {beta_interaccion}")


def CATE(plazo: int) -> float:
    return beta_monto + (beta_interaccion * plazo)


def ATE(mean_plazo: float = None) -> float:
    if mean_plazo is None:
        mean_plazo = df_creditos['plazo'].mean()

    return beta_monto + beta_interaccion * mean_plazo


print(
    f"Si el plazo es 1 días, por cada $1 extra de crédito, las ventas suben: ${CATE(1):.4f}")
print(
    f"Si el plazo es 5 días, por cada $1 extra de crédito, las ventas suben: ${CATE(5):.4f}")
print(
    f"Si el plazo es 10 días, por cada $1 extra de crédito, las ventas suben: ${CATE(10):.4f}")
print(
    f"Si el plazo es 20 días, por cada $1 extra de crédito, las ventas suben: ${CATE(20):.4f}")
print(
    f"Si el plazo es 50 días, por cada $1 extra de crédito, las ventas suben: ${CATE(50):.4f}")
print(
    f"Si el plazo es 100 días, por cada $1 extra de crédito, las ventas suben: ${CATE(100):.4f}")
print(
    f"Si el plazo es 200 días, por cada $1 extra de crédito, las ventas suben: ${CATE(200):.4f}")
print(
    f"Si el plazo es 500 días, por cada $1 extra de crédito, las ventas suben: ${CATE(500):.4f}")
print(
    f"Si el plazo es 1000 días, por cada $1 extra de crédito, las ventas suben: ${CATE(1000):.4f}")
print(
    f"Si el plazo es 2000 días, por cada $1 extra de crédito, las ventas suben: ${CATE(2000):.4f}")

print(f"Efecto Promedio Global (ATE): ${ATE():.4f}")

# %%

# Rango de valores de plazo
plazo_min = df_creditos['plazo'].min()
plazo_max = df_creditos['plazo'].max()

plazos = np.linspace(plazo_min, plazo_max, 300)
effects = CATE(plazos)

# Punto donde el efecto se vuelve positivo

plazo_cero_teorico = -beta_monto / beta_interaccion
plazo_min_positivo = int(np.floor(plazo_cero_teorico)) + 1

print(f"Plazo donde CATE(plazo) = 0 (teórico): {plazo_cero_teorico:.2f} días")
print(f"Primer plazo entero con efecto positivo: {plazo_min_positivo} días")
print(f"CATE({plazo_min_positivo}) = {CATE(plazo_min_positivo):.4f}")


plt.figure(figsize=(8, 5))
plt.plot(plazos, effects, label='CATE(plazo)')
plt.axhline(0, linestyle='--', linewidth=1, c='r', label='Efecto nulo')
plt.axvline(plazo_min_positivo, linestyle=':', linewidth=1, c='black',
            label=f'Umbral ≈ {plazo_min_positivo} días')
plt.axvline(plazo_cero_teorico, linestyle='-.', linewidth=1, c='r',
            label=f'Plazo CATE=0 (teórico) ≈ {plazo_cero_teorico:.2f} días')

plt.xlabel("Plazo (días)")
plt.ylabel("Efecto marginal por $1 extra de crédito")
plt.title("CATE(plazo): efecto causal del monto del crédito según el plazo")
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# # Aclaraciones
# ##
#
# A través del DAG realizado, previamente hemos identificado que las variables que hay que controlar para que no haya correlación espuria entre el monto del crédito y las ventas son: [antigüedad, plazo, compras pasadas, nivel de riesgo y sector]. Una vez controladas, decidi usar una regresión lineal ya que al ser mi primer acercamiento con un ejercicio completo de inferencia bayesiana causal, quería que el modelo sea lo más interpretable posible y asi poder entender bien el nucleo de la materia y no cuestiones algoritmicas de predicción.
#
# La regresión, si nos fijamos, le incorporo una interacción entre el monto y el plazo. Esto implica que el impacto del crédito no es constante, sino que cambia segun el plazo. A partir de est se desprende nuestra formula del CATE, que nos permite calcular el efecto marginal del crédito en las ventas para cualquier valor del plazo.
#
#   $CATE(plazo)= \beta_1 + \beta_3 ⋅ plazo$

# # Interpretación de los resultados

# **Interpretación del ATE**:
#
# El ATE queda levemente negativo porque la mayoría de los clientes del dataset trabajan con plazos muy cortos, y en ese segmento el crédito extra prácticamente no cambia su comportamiento de compra. Los casos donde el crédito sí impulsa ventas —los que tienen plazos más largos— existen, pero son menos frecuentes, por lo que su aporte positivo pesa menos en el promedio general. En otras palabras: cuando el cliente tiene poco tiempo para devolver el crédito, ese monto adicional no le da margen real para gastar más, y eso empuja el promedio hacia abajo. Solo en operaciones con plazos más amplios el crédito tiene un impacto comercial fuerte.

#
# **Interpretación del CATE**:
#
#  El CATE nos muestra cómo cambia el efecto del crédito según el plazo. Lo que encontramos es que el crédito no impacta igual en todos los clientes: depende fuertemente del tiempo que tienen para devolverlo. En plazos muy cortos, el crédito adicional prácticamente no mueve las ventas e incluso puede parecer levemente negativo. Esto tiene lógica: si el cliente tiene que devolver el dinero enseguida, ese crédito no le mejora la liquidez ni le da margen para comprar más. En cambio, a medida que el plazo se estira, el efecto del crédito empieza a volverse positivo y crece de forma clara. Con más tiempo para pagar, el cliente puede usar ese crédito de manera más holgada, lo que sí se traduce en mayores ventas. En resumen, el CATE revela que el crédito funciona, pero solo cuando el plazo acompaña. El crédito adicional genera ventas cuando el cliente tiene suficiente espacio financiero para aprovecharlo. Cuando no lo tiene, el impacto es prácticamente nulo.
#
# Por otro lado, podemos ver el grafico donde vemos que el CATE es una función lineal creciente con el plazo (Esto tiene sentido por la construcción del modelo, siendo una regresión lineal a través de backdoor con ciertas variables de control). Donde lo que buscamos explicar es cual es ese número de plazo crítico en donde el efecto marginal por cada extra de crédito empieza a ser positivo. Por lo cual graficamos el plazo en el eje X, el CATE(plazo) en el eje Y, y vemos que CATE (6,72) = 0. Por lo cual, a partir de los 7 días de plazo, el crédito empieza a tener un efecto positivo en las ventas.

# %% 4to paso de la inferencia causal: Refutación

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

# %% [markdown]
# # Refutación y validamos los modelos
# ##
#
# Para validar que nuestro modelo funciona, realizamos dos pruebas de refutación. La primera se llama Placebo Treatment Refuter, que consta en reemplazar el tratamiento real (monto_credito) por una variable completamente aleatoria. Donde, si el modelo está capturando una relación causal real, el nuevo efecto estimado debería ser cero, porque ahora no hay ninguna relación significativa entre ese “tratamiento falso” y las ventas.
#
# ## **Placebo Treatment Refuter**:
#
# Consta en reemplazar el tratamiento real (monto_credito) por una variable completamente aleatoria. Donde, si el modelo está capturando una relación causal real, el nuevo efecto estimado debería ser cero, porque ahora no hay ninguna relación significativa entre ese “tratamiento falso” y las ventas.
#
# ###  *Resultados*
# - Efecto original: –0.031
# - Nuevo efecto con placebo: –3.4e-10 (es decir, 0.00000000034, básicamente cero)
# - p-value: 0.0


# ## **Random Common Cause Refuter**:
#
# Agrega una variable aleatoria al dataset simulando una “causa común adicional” ficticia. La idea es ver si la estimación cambia drásticamente cuando incorporamos una variable irrelevante. Si la estimación es robusta, el nuevo efecto debería quedar prácticamente igual al original.
#
# ###  *Resultados*

# - Efecto original: –0.0310286
# - Nuevo efecto: –0.0310362
# - Diferencia: 0.0000076 (prácticamente cero)
# - p-value: 0.96

# ## ***Conclusiones***:

# En conjunto, las pruebas de refutación muestran que el modelo está capturando relaciones reales del negocio y no inventando patrones. Cuando reemplazamos el monto del crédito por un valor totalmente aleatorio, el efecto desaparece por completo, lo cual confirma que no estamos frente a un modelo sobreajustado ni sensible a coincidencias raras. Y cuando agregamos una variable aleatoria para “molestarlo”, el efecto prácticamente no cambia, señal de que el resultado es estable y no depende de factores sin sentido. En otras palabras, el modelo está reaccionando a información real de los clientes —como su plazo, riesgo o historial— y no a ruido. Esto nos da confianza en que las conclusiones obtenidas son consistentes y reflejan comportamientos genuinos dentro del dataset.
