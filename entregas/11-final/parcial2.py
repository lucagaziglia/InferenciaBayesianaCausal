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

# %%
import math
import random
import inspect
import warnings

respuestas = {}

# %% [markdown]
# # Parcial
# ##
#
# Este notebook contiene una lista de preguntas junto con una lista exhaustiva de respuestas mutuamente contradictorias. A diferencia de los enunciados de tipo "multiple choise" en los que se pide seleccionar una única opción, aquí se pide que distribuyan creencias entre las diferentes opciones, asegurándose que el valor asignado sea positivo y la suma sea 1. La evaluación será el producto de las creencias asignadas a las respuestas correctas. En caso de que la respuesta sea una variable aleatoria, se considerará la predicción típica a largo plazo, es decir, su media geométrica. Notar que asignar cero a una posible respuesta correcta hace que el producto sea cero. Por ello, en caso de duda, no conviene que concentren toda su creencia en una sola opción, sino distribuir algo de creencia en todas las opciones que consideran posibles. Noten también que conviene asignar más a la opción en la que más creen, porque distribuir creencias en partes iguales entre todas las opciones no es mucho mejor que el azar (baseline).


# %% [markdown]
# ### 1. Flujo
#
# La alarma (a) de una casa se activa cuando alguien entra (e) sin apagarla. Si la alarma se activa, la dueña recibe una llamada (l) de la central de seguridad. La alarma se puede activar por otros motivos, como los terremotos (t) que no son infrecuentes en la ciudad. Siempre que se produce un terremoto, en las redes (r) se habla casi exclusivamente de eso. El producto de los mecanismos causales probabilísticos definen la distribución conjunta, P(letra) = P(e)P(t)P(a|t,e)P(r|t)P(l|a). ¿Está cerrado el flujo de inferencia entre de la entradera (e) y el terremoto (t) dado la llamada (l)?. ¿Más precisamente, P(e,t|l) = P(e|l)P(t|l)?
#
# 0. No
# 1. Sí


# %%
respuestas[(1, "Flujo")] = [
    0.99,  # 0. No
    0.01,  # 1. Sí
    "En la estructura de esta situación que ya hemos analizado bastantes veces durante la materia, esta igualdad no se cumple. En otras palabras, dada la llamada, la entradera y el terremoto no son independientes. Esto se debe principalmente a que a es un collider y l es un descendiente directo de a (l depende unicamente de a). Y cuando nosotros observamos un collider, las variables que 'van hacia' el collider dejan de ser independientes. Lo mismo pasa con la llamada, ya que si te llaman es porque se activo la alarma, entonces analizar esto sabiendo la información de si te llaman o si se activo la alarma, de alguna forma es lo mismo."
]


# %% [markdown]
# ### Backdoor
#
# Una empresas ofrece créditos a sus clientes con el objetivo de aumentar las ventas. Los créditos tienen dos características: el monto disponible a usar como crédito (`monto_credito`) y los días de plazo que tienen para cancelar el crédito (`plazo`). La empresa quiere conocer el efecto causal que el `monto_credito` tiene sobre el volumen de ventas del mes (`ventas`). El problema es que no hay disponible ningún experimento aleatorizado ni conocemos la estructura causal subyacente completa. Solo sabemos lo siguiente.
# El `monto_credito` lo asigna un modelo de caja negra en función del promedio de compras del año pasado (`compras_pasadas`), un nivel de riesgo creado por una empresa externa (`nivel_riesgo`) y la cantidad de días otorgados para hacer el pago del crédito (`plazo`), el cuál está calculado en función del nivel de riesgo y otras variables que desconocemos. Además conocemos algunas variables relevantes de los clientes como son su sector comercial (`sector`) y los años que lleva como cliente de la empresa (`antiguedad`). Todas éstas variables, realtivas a un único mes del año, están en un archivo csv, donde cada fila representa un cliente distinto.
# ¿Es posible estimar el efecto causal que el `monto_credito` tuvo sobre las `ventas` en ese mes? Explique por qué. En caso de asignar creencia mayor a 0 a la respuesta "Sí" indique además cuáles serían las variables de control.
#
# 0. No
# 1. Sí

# %%
respuestas[(2, "Backdoor")] = [
    0.3,  # 0. No
    0.7,  # 1. Sí
    "Aunque no sabemos la realidad causal subyacente, hice un DAG con el fin de aproximar esta situación (Dejo fotos en la carpeta que subo dentro de 11-final). En ninguna parte del enunciado se menciona que es lo que produce las ventas, pero entiendo que las ventas dependen, tanto del año, como del sector, como de todas las variables. Por otro lado, el enunciado dice que contamos con un .csv con toda esta información, lo que convierte a compras_pasadas, nivel_riesgo, plazo, sector y antigüedad en variables observadas y así poder condicionar explicitamente por ellas. Estas variables observadas permiten bloquear todos los caminos de backdoor entre monto_credito y ventas. por lo que al condicionarlas eliminamos la correlación espuria que generan. Además, cualquier variable no observada que afecte al plazo queda controlada al ajustar por el propio plazo. Bajo estos supuestos, el criterio de backdoor se satisface y el efecto causal de monto_credito sobre ventas puede ser identificado. Variables de control: ['compras_pasadas', 'nivel_riesgo', 'plazo', 'sector', 'antiguedad', 'año_mes']"
]


# %% [markdown]
# ### 3. do-operator
#
# En la empresa nos pidieron un análisis de efectos causales. Estuvimos trabajando arduamente y tenemos bastante certeza que el efecto causal de interés es P(y|do(x)) \approx 0. Cuando entregamos estos resultados el cliente nos afirma que hay un error, que es imposible porque ellos saben que la variable x afecta muy fuertemente a todas las personas sin excepción. ¿Si la afirmación del cliente fuera cierta implica que en nuestro análisis cometimos un error? ¿Por qué? ¿Qué pasos a seguir le propondría al cliente?
#
# 0. No
# 1. Sí


# %%
respuestas[(3, "do-operator")] = [
    0.8,  # 0. No
    0.2,  # 1. Sí
    "Si estuvimos trabajando arduamente para estimar el efecto causal y al calcular P(y|do(x)) nos da aproximadamente 0, no necesariamente cometimos un error. Quizas el cliente tiene razón en que P(Y|do(x)) es un valor considerablemente mas alto que cero, pero que a nosotos nos de proximo a cero no implica un error en el análisis, probablemente radique en la especificación del problema a la hora de armar el DAG, y un malentendido de esa estructura, alguna variable o dependencia que no estemos teniendo en cuenta. Lo que le propondría al cliente es que tengamos algunas instancias para entender mejor el problema y generar un DAG mas robusto revisando todas las variables posibles y sus conexiones."
]


# %% [markdown]
# ### 4. Ignorability
#
# Nos piden participar como jurado de una tesis de licenciatura en ciencia de datos sobre inferencia causal. El objetivo de la tesis es analizar el desempeño de distintos algoritmos de aprendizaje automático para la estimación de efectos causales heterogéneos. La tesis tiene varios errores de redacción. En la introducción de la tesis además encuentran, en contra de toda la literatura de referencia, la afirmación de que el criterio para determinar variables de control del enfoque de potential outcomes (el criterio de ignorabilidad condicional) no implica el cumplimiento del criterio backdoor de forma general en cualquier modelo causal generativo. ¿Este es un error grave de la tesis? Resuelva la contradicción entre la afirmación que se realiza en la introducción y toda la literatura de referencia explicando en profundidad por qué hay un error o por que no lo hay.
#
# 0. No
# 1. Sí


# %%
respuestas[(4, "Ignorability")] = [
    0.7,  # 0. No
    0.3,  # 1. Sí
    "Si la tesis esta mal redactada y no contiene la información/demostración de porque estos enfoques son distintos, es un error. No considero que sea un error grave ya que lo que se plantea en la tésis es correcto, ya que para alguna situación donde no se presente un Structural Causal Model, estos enfoques no son equivalentes. Al ser un concepto muy poco conocido y que dentro de todas las fuentes de referencia no se valida este concepto, creop que vale la pena aclararlo y demostrarlo contundentemente para que no haya confusiones. Es más, si todos los jurados de la tesis asumen que esto es así y luego les presentas una demostración formal y rigurosa de porque esto no es así, la percepción del jurado pasa a ser sumamente positiva. Esta demostración sería apartir de las dos caracteristicas que tienen los SCM, que son: 1) Todas las variables end ́ogenas son deterministas (para cumplir con consistencia) y 2) Cada variable end ́ogena tiene una ex ́ogena aleatoria (para cumplir con ignorability)."
]


# %% [markdown]
# ### 5. Decisiones
#
# Leer la consigna en el pdf.
#
# ¿El posterior de la variable factual $P(b|\mathcal{O}=1, \omega_0=1)$ garantiza la maximización de las ganancias factuales a largo plazo bajo el juego de apuestas descrito? Explique por qué, indicando cuál es el mensaje que envía el factor de la variable optimalidad a la decisión factual $b$.
#
# #
# 0. No
# 1. Sí


# %%
respuestas[(5, "Decisiones")] = [
    0.1,  # 0. No
    0.9,  # 1. Sí
    "El factor de optimalidad envía a b un mensaje que asigna probabilidad únicamente a la decisión que maximiza la tasa geométrica de crecimiento. Formalmente mensaje(f_optimalidad -> b) = I(b = argmax_b',r_T'(b')). Entonces si imponemos que la decisión fue óptima (O = 1), el factor de optimalidad obliga a que la única decisión posible sea la que maximiza esa tasa, descartando todos los demás valores de b. Ese valor óptimo coincide con la probabilidad real del evento (b = p), lo cual se debe al Criterio de Kelly visto en clase."
]


# %%
