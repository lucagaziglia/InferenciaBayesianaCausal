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
# Este notebook contiene una lista de preguntas junto con una lista exhaustiva de respuestas mutuamente contradictorias. A diferencia de los enunciados de tipo "multiple choise" en los que se pide seleccionar una única opción, aquí se pide que distribuyan creencias entre las diferentes opciones, asegurándose que el valor asignado sea positivo y la suma sea 1. La evaluación será el producto de las creencias asiganadas a las respuestas correctas. En caso de que la respuesta sea una variable aleatoria, se considerará la predicción típica a largo plazo, es decir, su media gométrica. Notar que asignar cero a una posible respuesta correcta hace que el producto sea cero. Por ello, en caso de duda, no conviene que concentren toda su creencia en una sola opción, sino distribuir algo de creencia en todas las opciones que consideran posibles. Noten también que conviene asignar más a la opción en la que más creen, porque distribuir creencias en partes iguales entre todas las opciones no es mucho mejor que el azar (baseline).


# %% [markdown]
# ### 1 Variable
#
# En probabilidad una variable es un conjunto de hipótesis mutuamente contradictorias.
#
# 0. No
# 1. Sí

# %%
respuestas[(1, "Variable")] = [
    0.1,  # 0. No
    0.9,  # 1. Sí
    "Según lo que busque esa es la definición de una variable. Sin embargo asigno un 0.1 a la probabilidad de que esto sea falso ya que no termino de estar seguro sobre el concepto de mutuamente excluyentes... Ya que una variable podría tomar mas de un valor, como puede ser un conjunto dentro de la variable de interes. También mi decisión de 0.1 a esta opción es para no descartar un caso y que se me anule la respuesta."
]


# %% [markdown]
# ### 2 Conjunta
#
# La distribución de probabilidad conjunta es creencia a priori.
#
# 0. No
# 1. Sí

# %%
respuestas[(2, "Conjunta")] = [
    0.9,  # 0. No
    0.1,  # 1. Sí
    "Cuando hablamos de conjunta entre hipotesis y datos, la creencia a priori es una parte de la conjunta. Esto se debe ya que P(A,B) = P(A)P(A|B) donde P(A) es la creencia a priori y P(A|B) es la verosimilitud. Entonces la conjunta no es igual a la creencia a priori. No termino de descartar debido a que hay casos donde la likelihood es 1, y en ese caso la conjunta seria igual a la creencia a priori y estimo que eso debe pasar poco, aproxiamadamente un 10 porciento de las veces"
]


# %% [markdown]
# ### 3 Universos
#
# Hay tres cajas idénticas. Detrás de una de ellas hay un regalo. El resto están vacías. Nos permiten reservar una caja. Luego, una persona elige una de las cajas que no contenga el regalo y no haya sido reservada. Supongamos que reservamos la caja 1. ¿Cuál de todos los universos paralelos va a ocurrir? ¿El regalo está en la caja 1 y nos muestran la caja 1? ¿El regalo está en la caja 1 y nos muestran la caja 2? ... ¿El regalo está en la caja 3 y nos muestran la caja 2? ¿El regalo está en la caja 3 y nos muestran la caja 3?
#
# 0. Regalo = 1, Abren = 1
# 1. Regalo = 1, Abren = 2
# 2. Regalo = 1, Abren = 3
# 3. Regalo = 2, Abren = 1
# 4. Regalo = 2, Abren = 2
# 5. Regalo = 2, Abren = 3
# 6. Regalo = 3, Abren = 1
# 7. Regalo = 3, Abren = 2
# 8. Regalo = 3, Abren = 3
#

# %%
respuestas[(3, "Universos")] = [
    0.0,  # 0. Regalo = 1, Abren = 1
    1/6,  # 1. Regalo = 1, Abren = 2
    1/6,  # 2. Regalo = 1, Abren = 3
    0.0,  # 3. Regalo = 2, Abren = 1
    0.0,  # 4. Regalo = 2, Abren = 2
    1/3,  # 5. Regalo = 2, Abren = 3
    0.0,  # 6. Regalo = 3, Abren = 1
    1/3,  # 7. Regalo = 3, Abren = 2
    0.0,  # 8. Regalo = 3, Abren = 3
    "Si asumimos como verdad el enunciado, los casos donde Regalo = Abren va a tener probabilidad 0. Por otro lado, los universos que tendran probabilidad 0 son los que comentan que abren la caja 1 ya que esta fue reservada. Por último los casos posibles son esos 4, los universos: [1, 2, 5, 7]. Cómo se distribuyen las probabilidades ahi? Es cuestión de aplicar el teorema de bayes y calcular esos casos. ",
]


# %% [markdown]
# ### 4 Overfitting
#
# En el área de aprendizaje automático e inteligencia artificial se ha descubierto un problema que se conoce con el nombre de overfitting. ¿El overfitting es/era un problema propio del sistema de razonamiento para contextos de incertidumbre?
#
# 0. No
# 1. Sí

# %%
respuestas[(4, "Overfitting")] = [
    0.9,  # 0. No
    0.1,  # 1. Sí
    " El overfitting es un problema que se puede dar en situaciones ajenas al sustena de razonamiento en contextos de incertidumbre. Por ejemplo un caso donde no se aplica el sistema de razonamiento en contextos de incertidumbre es el caso de una regresión donde tenemos 15 datos que siguen una recta y = x - 1. Si ajustamos un polinomio de grado 14 a esos datos, vamos a tener un caso de overfitting en un caso donde no hay representación explícita de incertidumbre ni sobre parámetros. Simplemente resolvemos un sistema de ecuaciones para encontrar los coeficientes del polinomio.",
]

# %% [markdown]
# ### 5 Evaluación
#
# En el área de aprendizaje automático e inteligencia artificial existe una gran cantidad de métricas distintas para evaluar los modelos alternativos. ¿En principio, existe una forma correcta, universal, de evaluar los modelos?
#
# 0. No
# 1. Sí

# %%
respuestas[(5, "Evaluación")] = [
    0.95,  # 0. No
    0.05,  # 1. Si
    "Si bien en el area de ML e IA hay muchas métricas para evaluar modelos, existe una forma universal de evaluar los modelos, aplicando las reglas de la probabilidad. En la cual lo que se suele hacer es calcular la predicción del modelo para un dato -> P(Dato = d | H1, H2, ... , Hn, Modelo). Pero a partir de esto podemos llegar a través de cuentas a que P(Modelo | Dato = d, H1, H2, ... , Hn) = P(D|M,H)*P(M|H)/P(D|H). Con esto podemos llegar finalmente a una probabilidad, que refleja la credibilidad de un modelo una vez vistos los datos y bajo ciertas hipotesis.",
]

# %% [markdown]
# ### 6 Predicción
#
# Históricamente todas las ciencias con datos, desde la física hasta las ciencias sociales, explicaron el mundo a través de teorías causales. Los recientes avances en el área de aprendizaje automático e inteligencia artificial, sin embargo, se produjeron por el desarrollo de algoritmos altamente predictivos sin ninguna interpretación causal. ¿Por qué?
#
# 0. El modelo causal correcto nunca puede ser mejor prediciendo que los complejos algoritmos de AI/ML.
# 1. El modelo causal correcto a veces puede ser mejor, y a veces peor, que los complejos algoritmos de AI/ML.
# 2. El modelo causal correcto nunca puede ser peor prediciendo que los complejos algoritmos de AI/ML.
# 3. Los modelos causales solo explican, no predicen.
# 4. Ninguna de las anteriores

# %%
respuestas[(6, "Predicción")] = [
    # 0. El modelo causal correcto nunca puede ser mejor prediciendo que los complejos algoritmos de AI/ML.
    0.01,
    # 1. El modelo causal correcto a veces puede ser mejor, y a veces peor, que los complejos algoritmos de AI/ML.
    0.30,
    # 2. El modelo causal correcto nunca puede ser peor prediciendo que los complejos algoritmos de AI/ML.
    0.67,
    0.01,  # 3. Los modelos causales solo explican, no predicen.
    0.01,  # 4. Ninguna de las anteriores
    "Si se hace una buena selección de modelo causal, siendo este el que genera la realidad causal subyacente del problema en cuestión, siempre o casi siempre va a predecir mejor que cualqueir algoritmo complejo de IA o DeepLearning.",
]


# %% [markdown]
# ### Factor graph
#
# Un factor graph es una grafo bipartito entre dos tipo de nodos: variables y funciones (distribuciones de probabilidad condicional). Los ejes representan ``la variable $v$ es parámetro de la función $f$''.
#
# 0. Falso
# 1. Verdadero

# %%
respuestas[(7, "Factor graph")] = [
    0.05,  # 0. False
    0.95,  # 1. Verdadero
    "Esta afirmación es verdadera. Las variables se conectan unicamente con las funciones y las funciones unicamente con los nodos, esto hace que sea un grafo bipartito.",
]

# %% [markdown]
# ### do-operator
#
# Cuando se aplica un do-operator a una variable se reemplaza la distribución de probabilidad condicional que naturalmente tiene esa variable por una distribución de probabilidad indicadora (determinista). ¿Es posible especificar do-operators usando la notación de factor graphs? ¿Cómo?
#
# 0. No se puede
# 1. Sí se puede

# %%
respuestas[(8, "do-operator")] = [
    0.05,  # 0. No se puede
    0.95,  # 1. Sí se puede
    "Se puede. Justamente una de las grandes ventajas de los factor-graphs vs las redes bayesianas es este concepto. Cuando se aplica un do-operator a una variable, se cambia su distribución de probabilidad por una determinista, que funciona como una compuerta lógica.",
]


# %% [markdown]
# ### Sum-product marginal
#
# El sum-product algorithm descompone las reglas de probabilidad como pasaje de mensajes entre los nodos de un factor graph. La distribución marginal de una variable es el producto de los mensajes que recibe esa variable.
#
# 0. Falso
# 1. Verdadero

# %%
respuestas[(9, "Sum-product marginal")] = [
    0.05,  # 0. False
    0.95,  # 1. Verdadero
    "El algoritmo de pasaje de mensajes entre nodos funciona exactamente asi. Se apoya en el algoritmo sum-product, que a su vez son las dos reglas claves en la probabilidad. La marginal de una variable es el producto de los mensajes recibidos que a su vez son marginales, ya que el algoritmo de pasaje de mensajes, en cada pasaje va marginalizando sobre todas las variables previas.",
]


# %% [markdown]
# ### Estructura básica
#
# Dada la siguiente estructura causal $P(X,Y,M,W) = P(X)P(Y)P(M|X,Y)P(W|M)$. ¿$X$ es independiente de $Y$ dado $W$?
#
# 0. No son independientes
# 1. Sí son independientes


# %%
respuestas[(10, "Estructura básica")] = [
    0.95,  # 0. No son independientes
    0.05,  # 1. Sí son independientes
    "X e Y no son independientes cuando condicionamos por W. Ya que cuando hacemos esto se activa el collider y se abre el flujo de inferencia. Lo resolví planteando la red bayesiana y simulando el ejemplo de clase, con terremoto, entradera, etc. Esto seria similar a calcular P(Entradera|Llamada) ¿=? P(Entradera|Terremoto, Llamada) lo cual obviamente son distintos ya que en el segundo termino aporta información el terremoto y se puede hacer inferencia para determinar la entradera o X en este caso.",
]


# %% [markdown]
# ### Predicción causal
#
# Es posible predecir el impacto causal que una variable $X$ tiene sobre una variable $Y$ usando datos observados sin intervenciones sin conocer la estructura causal subyacente.
#
# 0. Falso
# 1. Verdadero

# %%
respuestas[(11, "Predicción causal")] = [
    0.95,  # 0. Falso
    0.05,  # 1. Verdadero
    "Esto es falso ya que sin intervenciones no se puede despejar esa confusión entre correlación y causalidad. A su vez sin conocer la realidad causal subyacente no se puede hacer intervenciones, ya que si no conocemos la realidad causal e intervenimos podemos estar abriendo flujos de inferencia no deseados o correlaciones espurias.",
]


# %% [markdown]
# ### d-separation
#
# Hay flujo de inferencia entre los extremos de una cadena si y solo si se condiciona únicamente en todas las consecuencias comunes.
#
# 0. Falso
# 1. Verdadero

# %%
respuestas[(12, "d-separation")] = [
    0.95,  # 0. Falso
    0.05,  # 1. Verdadero
    "Esta afirmación no refleja correctamente las reglas de d-separation, es justamente lo contrario a la hora de hablar de pipes. Si condicionamos por una de las variables del medio de la cadena, como por ejemplo A -> B -> C, si condicioinamos por B, se bloquea el flujo de inferencia.",
]

# %% [markdown]
# ### Backdoor
#
# Si un conjunto de variable $Q$ cierra el flujo de asociación en todos los caminos ascendentes de $X$ a $Y$ necesariamente cumple con el criterio backdoor.
#
# 0. Falso
# 1. Verdadero

# %%
respuestas[(13, "Adjustment formula")] = [
    0.95,  # 0. Falso
    0.05,  # 1. Verdadero
    "Esto es falso ya que no esta del todo completo. Si el conjunto de variables Q bloquea todos los caminos ascendentes de X a Y se esta cumpliendo con uno de dos de los criterios necesarios para que se cumpla BACKDOOR. Falta que también el conjunto Q no contenga descendientes de X.",
]


# %% [markdown]
# ### Adjustment formula
#
# Sea $Q$ variables de control que cumplen con el criterio backdoor de $X$ a $Y$. Y sea $M_x$ el modelo intervenido en el que se le asigna aleatoriamente un valor a la variable $X$. ¿Es cierta la siguiente igualdad?
#
# \begin{equation}
# P(Y|\text{do}(X)) = P_{M_x}(Y|X) = \sum_{Q} P(Q|X)P(Y|X,Q)
# \end{equation}
#
# 0. No es cierta
# 1. Sí es cierta

# %%
respuestas[(14, "Adjustment formula")] = [
    0.95,  # 0. No es cierta
    0.05,  # 1. Sí es cierta
    "Si las variables Q cumplen con el criterio backdoor, esta afirmación no es cierta. La primera igualdad si se cumple, P(Y|do(X)) = P_M_x(Y|X). Lo que no termina de ser verdadero es la segunda igualdad, ya que P(Y|do(X)) = SUMA_Q[ P(Q)P(Y|X,Q) ] y no SUMA_Q[ P(Q|X)P(Y|X,Q) ]. Esto se debe a que si intervenimos en X, esta variable se desconecta de sus causas y en el modelo intervenido Q ya no depende de X, solo de si misma, es por eso que va P(Q) y no P(Q|X)"
]


# %% [markdown]
# ### Ignorar intervención
# Sea $Q$ variables de control que cumplen con el criterio backdoor de $X$ a $Y$. Y sea $M_x$ el modelo intervenido en el que se le asigna aleatoriamente un valor a la variable $X$. ¿Puede ocurrir que $P_{M_x}(Y|X,Q) \neq P(Y|X,Q)$?
#
# 0. No puede ocurrir
# 1. Sí puede ocurrir

# %%
respuestas[(15, "Ignorar intervención")] = [
    0.95,  # 0. No es cierta
    0.05,  # 1. Sí es cierta
    "Cuando hacemos do(x) se elimina la influencia de las causas de X. Y como se cumple el criterio de Backdoor, la probabilidad de Y dado X es la misma en el modelo real (sin intervención) Como en elintervenido, por lo cual PMx(Y|X,Q) = P(Y|X,Q), lo cual hace que sea falso  PMx(Y|X,Q) != P(Y|X,Q)."
]


# %% [markdown]
# ### Independencia
# Sea $Q$ variables de control que cumplen con el criterio backdoor de $X$ a $Y$. Y sea $M_x$ el modelo intervenido en el que se le asigna aleatoriamente un valor a la variable $X$. ¿Es cierta la siguiente igualdad, $P_{M_x}(Q|X) = P_{M_x}(Q)$? ¿Por qué?
#
# 0. No es cierta
# 1. Sí es cierta

# %%
respuestas[(16, "Independencia")] = [
    0.05,  # 0. No es cierta
    0.95,  # 1. Sí es cierta
    "Si Q no incluye descendientes de X, esto es cierto ya que Q no se ve afectado por la intervención. En el modelo intervenido, X y Q son independientes. Esto implica que la distribución condicional se reduce a la marginal, por lo cual PMx(Q|X) = PMx(Q).",
]


# %% [markdown]
# ### Variables de control
#
# Supongamos que tenemos un conjunto de datos observados sin intervenciones. Nos interesa conocer el efecto causal que una variable $T_i$ tiene sobre una variable objetivo $Y_i$. Sabemos que existe una variable oculta $U_i$ que es causa de las variables $T_i$ e $Y_i$. Sabemos además que el efecto causal de $T_i$ sobre $Y_i$ está mediado por $M_i$. En resumen la estructura causal está determinada por los mecanismos causales, $P(U_i)$, $P(T_i|U_i)$, $P(M_i|T_i)$ y $P(Y_i|M_i,U_i)$. ¿Podemos estimar el efecto causal únicamente con la información de $T_i$, $M_i$ e $Y_i$? Explique cómo.
#
# 0. No cumple backdoor
# 1. Sí cumple backdoor

# %%
respuestas[(17, "Variables de control")] = [
    0.95,  # 0. No cumple backdoor
    0.05,  # 1. Sí cumple backdoor
    "En este caso no se cumple backdoor debido a que este criterio exige que todos los caminos que vayan de T a Y , estén bloqueados cuando condicionamos en las variables de control. Pero en este caso existe el camino T <- U -> Y el cual mete correlación espuria entre T e Y.",
]


# %% [markdown]
# ### Causa común oculta
#
# Supongamos que tenemos un conjunto de datos observados sin intervenciones. Nos interesa conocer el efecto causal que una variable $T_i$ tiene sobre una variable objetivo $Y_i$. Sabemos que existe una variable oculta $U_i$ que es causa de las variables $T_i$ e $Y_i$. Sabemos además que el efecto causal de $T_i$ sobre $Y_i$ está mediado por $M_i$. En resumen la estructura causal está determinada por los mecanismos causales, $P(U_i)$, $P(T_i|U_i)$, $P(M_i|T_i)$ y $P(Y_i|M_i,U_i)$. ¿Podemos estimar el efecto causal únicamente con la información de $T_i$, $M_i$ e $Y_i$? Explique cómo.
#
# 0. No es posible
# 1. Sí es posible

# %%
respuestas[(18, "Causa común oculta")] = [
    0.95,  # 0. No es posible
    0.05,  # 1. Sí es posible
    "No podemos estimar el efecto causal unicamente con la información de T, M e Y ya que U es la causa común oculta de ambas varaibles. Es la que en parte genera a estas variables de interés.",
]

# %% [markdown]
# ### Experimento sin cumplimiento
#
# Supongamos que diseñamos un experimento aleatorizados, asignando el grupo al que pertenece cada persona mediante una variable aleatoria $Z_i$ tal que $P(Z_i) = \text{Bernoulli}(Z_i| 0.5)$. Supongamos que la aplicación efectiva del tratamiento $T_i$ no se cumple estrictamente sino que varía en función de las características ocultas de las personas $C_i$, $P(T_i|Z_i,C_i)$. Finalmente supongamos que la variable objetivo depende tanto del tratamiento aplicado $T_i$ y de las características ocultas, $P(Y_i|T_i, C_i)$. ¿Es posible estimar el efecto causal? Explique cómo.
#
# 0. No es posible
# 1. Sí es posible

# %%
respuestas[(19, "Experimento sin cumplimiento")] = [
    0.05,  # 0. No es posible
    0.95,  # 1. Sí es posible
    "Es posible. Aun que el tratamiento efectivo T no siga la dist. de Z (la cual es aleatoria), es posible estimar el efecto causal porque la asignación se generó de manera exógena e independiente de las características ocultas (C). La correlación de Z y el resultado Y solo puede explicarse por la inglencia de Z sobre T, lo que habilita identificar el efecto de T sobre Y",
]

# %% [markdown]
# ### Diversificación
#
# Una casa de apuestas paga $3$ por Cara y $1.2$ por Sello con cada lanzamiento de una moneda que tiene $0.5$ de probabilidad de que salga Cara y Sello. Supongamos que nos ofrecen jugar 1000 veces, apostando en cada paso temporal todos nuestros recursos. ¿Qué proporción apostaría a Cara? Notar que el resto se asigna a Sello. Notar además que si apostamos todo a Cara y sale Sello perdemos todos los recursos y no podemos volver a jugar.
#
# 0. Recursos asignados a Cara: 0.0
# 1. Recursos asignados a Cara: 0.1
# 2. Recursos asignados a Cara: 0.2
# 3. Recursos asignados a Cara: 0.3
# 4. Recursos asignados a Cara: 0.4
# 5. Recursos asignados a Cara: 0.5
# 6. Recursos asignados a Cara: 0.6
# 7. Recursos asignados a Cara: 0.7
# 8. Recursos asignados a Cara: 0.8
# 9. Recursos asignados a Cara: 0.9
# 10. Recursos asignados a Cara: 1.0

# %%
respuestas[(20, "Diversificación")] = [
    0.025,  # 0. Recursos asignados a Cara: 0.0
    0.10,  # 1. Recursos asignados a Cara: 0.1
    0.19,  # 2. Recursos asignados a Cara: 0.2
    0.19,  # 3. Recursos asignados a Cara: 0.3
    0.15,  # 4. Recursos asignados a Cara: 0.4
    0.15,  # 5. Recursos asignados a Cara: 0.5
    0.10,  # 6. Recursos asignados a Cara: 0.6
    0.05,  # 7. Recursos asignados a Cara: 0.7
    0.025,  # 8. Recursos asignados a Cara: 0.8
    0.01,  # 9. Recursos asignados a Cara: 0.9
    0.01,  # 10. Recursos asignados a Cara: 1.0
    "La verdad no tengo tiempo de hacer la cuenta a mano para entender que es lo mas efectivo, pero seguramente si asignamos un poco menos de los mitad de los recursos a cara, 'seguiremos con vida' mas tiempo y creciendo mas ya que cara paga 3 y sello 1.2. Lo que significa que en una mala racha para una P(cara) = 0.5 si tiramos n veces y pocas caras, las cveces que toco cara habremos recuperado bastante, sin perder mucho en las que salio sello.  ",
]

# %% [markdown]
# ### Apuesta individual
#
# Una casa de apuestas paga $3$ por Cara y $1.2$ por Sello con cada lanzamiento de una moneda que tiene $0.5$ de probabilidad de que salga Cara y Sello. Supongamos que nos ofrecen jugar 1000 veces apostando en cada paso temporal 50\% de los recursos a Cara y 50\% de los recursos a Sello. Notar que cuando sale Cara los recursos aumentan 50\% ($3\times50\% + 0\times50\% =150\%$), y si sale Sello los recursos se reducen 40\% ($0\times50\% + 1.2\times50\% =60\%$). Es decir, crecemos más de lo que caemos. Y efectivamente, si calculamos la riqueza promedio de una población muy muy grande veremos que crece a una tasa de 5\% por paso temporal. ¿Nos conviene jugar?
#
# 0. No conviene
# 1. Sí conviene
# 2. Es indistinto

# %%
respuestas[(21, "Apuesta individual")] = [
    0.7,  # 0. No conviene
    0.15,  # 1. Sí conviene
    0.15,  # 2. Es indistinto
    "Lo que importa es si nos conviene a nosotros como individuos jugar, independientemente del hecho de que en conjunto se crece un 5% por jugada en promedio. Considero que no es conveniente jugar... Ya que el capital se reduce en una proporción mucho mas grande de lo que crece cuando ganamos. Como ambos eventos tienen la misma probablidad, este comportamiento hace que tu capital a largo plazo tiende a 0.",
]

# %% [markdown]
# ### Fondo común
#
# Una casa de apuestas paga $3$ por Cara y $1.2$ por Sello con cada lanzamiento de una moneda que tiene $0.5$ de probabilidad de que salga Cara y Sello. Supongamos que estamos apostando 50\% de los recursos a Cara y 50\% de los recursos a Sello en cada paso temporal. Supongamos que nos proponen participar de un fondo común, en el que en cada paso temporal los recursos de todas las personas que lo integran se redistribuyen en partes iguales. Es decir, en cada paso temporal cada persona tira su propia moneda, actualiza sus propios recursos individuales, los pone en el fondo común, se dividen en partes iguales y volvemos a empezar. ¿Nos conviene participar del fondo común?
#
# 0. No conviene
# 1. Sí conviene
# 2. Es indistinto


# %%
respuestas[(22, "Fondo común")] = [
    0.15,  # 0. No conviene
    0.7,  # 1. Sí conviene
    0.15,  # 2. Es indistinto
    "En este caso es donde se ve esa riqueza promedio de que crece a una tasa de 5%, ya que al jugar en equipo a esto se mezclan los resultados de muchas personas y se va redistribuyendo en cada paso. De esta forma uno esperaría llegar a ese 5 porciento de crecimiento en cada jugada.",
]

# %% [markdown]
# ### Tragedia de los comunes
#
# Una casa de apuestas paga $3$ por Cara y $1.2$ por Sello con cada lanzamiento de una moneda que tiene $0.5$ de probabilidad de que salga Cara y Sello. Supongamos que estamos apostando 50\% de los recursos a Cara y 50\% de los recursos a Sello en cada paso temporal y que participamos de un fondo común, en el que en cada paso temporal los recursos de todas las personas que lo integran se redistribuyen en partes iguales. ¿En términos estrictamente monetarios, nos convendría dejar de aportar al fondo común en caso de que sigamos recibiendo la cuota en partes iguales del fondo común?
#
# 0. No conviene
# 1. Sí conviene
# 2. Es indistinto

# %%
respuestas[(23, "Tragedia de los comunes")] = [
    0.1,  # 0. No conviene
    0.8,  # 1. Sí conviene
    0.1,  # 2. Es indistinto
    " Para este caso se pueden aplicar muchos casos similares. Uno de ellos es por ejemplo que todo un barrio pague para la limpieza del barrio, si uno deja de aportar, seguimos disfrutando del beneficio ya que hay otros que siguen pagando. A su vez me estoy ahorrando el dinero por no pagar el servicio. Este comportamiento hace que si todos toman esta decisión el sistema colectivo deja de funcionar. Pero en terminos monetarios unicamente, claramente conviene mas dejar de aportar al fondo común y seguir recibiendo la cuota en partes iguales del fondo común y seguir recibiendo la parte correspondiente.",
]
