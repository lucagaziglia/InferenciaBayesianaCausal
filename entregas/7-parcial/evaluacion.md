
# Pregunta: 1 Variable
Creencia de referencia: [0.001, 0.999]
Creencia de respuesta: [0.1, 0.9]
Cross entropy pregunta: -0.14376526070903112

Justificación: Según lo que busque esa es la definición de una variable. Sin embargo asigno un 0.1 a la probabilidad de que esto sea falso ya que no termino de estar seguro sobre el concepto de mutuamente excluyentes... Ya que una variable podría tomar mas de un valor, como puede ser un conjunto dentro de la variable de interes. También mi decisión de 0.1 a esta opción es para no descartar un caso y que se me anule la respuesta.

Justificación de referencia: La redacción de esta pregunta puede resultar confusa para las personas que hayan interpretado la palabra 'variable' con el concepto 'variable aleatoria', y hayan interpretado el concepto 'variable aleatoria' específicamente como la 'función' que genera valores basados en su distribución de probabilidad. En `scipy` esa interpretación de 'variable aleatoria' coincide con el método asociado a las distribuciones de probabilidad, llamado `rvs()`. Sin embargo, desde un perspectiva lógica es correcto definir una variable aleatoria como un conjunto de hipótesis mutuamente contradictorias, pues una variable aleatoria representa una pregunta sobre el estado del mundo (p. ej., '¿cuál es el resultado de lanzar un dado?'), y sus posibles valores o estados ({1, 2, 3, 4, 5, 6}) son las posibles respuestas a esa pregunta. Cada una de estas respuestas puede ser vista como una hipótesis. Las hipótesis son mutuamente contradictorias (o excluyentes), ya que si el resultado es 3, no puede ser ningún otro valor simultáneamente. Además, el conjunto de valores se considera exhaustivo, lo que significa que la variable debe tomar uno de esos valores.

# Pregunta: 2 Conjunta
Creencia de referencia: [0.001, 0.099]
Creencia de respuesta: [0.9, 0.1]
Cross entropy pregunta: 0.011249238591033495

Justificación: Cuando hablamos de conjunta entre hipotesis y datos, la creencia a priori es una parte de la conjunta. Esto se debe ya que P(A,B) = P(A)P(A|B) donde P(A) es la creencia a priori y P(A|B) es la verosimilitud. Entonces la conjunta no es igual a la creencia a priori. No termino de descartar debido a que hay casos donde la likelihood es 1, y en ese caso la conjunta seria igual a la creencia a priori y estimo que eso debe pasar poco, aproxiamadamente un 10 porciento de las veces

Justificación de referencia: La distribución de probabilidad conjunta sobre todas las variables representa el estado de conocimiento que se desprende del modelo (y todo tipo de restricciones y supuestos iniciales). Esto es justamente lo que se conoce como una creencias a priori, pues está definida antes de observar cualquier dato o evidencia. De hecho, las creencias a posteriori se obtienen a través de la distribución conjunta que se encuentra en el denominador, la cual simplemente preserva la creencia previa (conjunta) que sigue siendo compatible con el dato fijado el valor de la hipótesis que pasó a ser observada, P(H,D).

# Pregunta: 3 Universos
Creencia de referencia: [0, 0.16666666666666666, 0.16666666666666666, 0, 0, 0.3333333333333333, 0, 0.3333333333333333, 0]
Creencia de respuesta: [0.0, 0.16666666666666666, 0.16666666666666666, 0.0, 0.0, 0.3333333333333333, 0.0, 0.3333333333333333, 0.0]
Cross entropy pregunta: 0.0

Justificación: Si asumimos como verdad el enunciado, los casos donde Regalo = Abren va a tener probabilidad 0. Por otro lado, los universos que tendran probabilidad 0 son los que comentan que abren la caja 1 ya que esta fue reservada. Por último los casos posibles son esos 4, los universos: [1, 2, 5, 7]. Cómo se distribuyen las probabilidades ahi? Es cuestión de aplicar el teorema de bayes y calcular esos casos. 

Justificación de referencia: Se asume que la probabilidad a priori de que el regalo esté en cualquier caja es 1/3. Dado que la caja 1 está reservada, la persona que abre una caja está restringida a no abrir la caja 1 y a no abrir la que contiene el regalo.
Si Regalo = 1 (Prob. 1/3) la persona puede abrir la caja 2 o la 3. Asumiendo que elige al azar entre ellas, hay 1/2 de probabilidad para cada una. Por lo tanto, P(Regalo=1, Abren=2) = (1/3) * (1/2) = 1/6, y P(Regalo=1, Abren=3) = (1/3) * (1/2) = 1/6. Si Regalo = 2 (Prob. 1/3): La persona no puede abrir la 1 (reservada) ni la 2 (regalo). Debe abrir la 3. Por lo tanto, P(Regalo=2, Abren=3) = (1/3) * 1 = 1/3. Si Regalo = 3 (Prob. 1/3): La persona no puede abrir la 1 (reservada) ni la 3 (regalo). Debe abrir la 2. Por lo tanto, P(Regalo=3, Abren=2) = (1/3) * 1 = 1/3. El resto de los universos son imposibles bajo las reglas del problema.


# Pregunta: 4 Overfitting
Creencia de referencia: [0.999, 0.001]
Creencia de respuesta: [0.9, 0.1]
Cross entropy pregunta: -0.14376526070903112

Justificación:  El overfitting es un problema que se puede dar en situaciones ajenas al sustena de razonamiento en contextos de incertidumbre. Por ejemplo un caso donde no se aplica el sistema de razonamiento en contextos de incertidumbre es el caso de una regresión donde tenemos 15 datos que siguen una recta y = x - 1. Si ajustamos un polinomio de grado 14 a esos datos, vamos a tener un caso de overfitting en un caso donde no hay representación explícita de incertidumbre ni sobre parámetros. Simplemente resolvemos un sistema de ecuaciones para encontrar los coeficientes del polinomio.

Justificación de referencia: Asignar más creencia a una hipótesis de la que 'merece' es la definición más precisa de sobreajuste (overfitting). El sobreajuste, sin embargo, se lo conoce más por sus consecuencias. La aplicación estricta de las reglas de la probabilidad garantiza que las distribuciones de creencias se asignen correctamente en función del 'mérito' que tiene cada hipótesis, maximizando la incertidumbre dada la información disponible (modelo y datos). Si bien hasta ahora no se ha propuesto un sistema de razonamiento para contextos de incertidumbre mejor en términos prácticos, la aplicación estricta de las reglas de la probabilidad (o enfoque bayesiano) se ha visto limitada históricamente debido al costo computacional que se requiere para la evaluación de todo el espacio de hipótesis. Por ello, durante el siglo 20 comienzan a proponerse criterios que, para evitar el costo computacional, seleccionan arbitrariamente una única hipótesis del espacio. Esto es lo que se realiza en los procesos de ML y AI que se conocen como 'entrenamiento'. Por ejemplo, el llamado 'entrenamiento' en el contexto de una regresión lineal se selecciona arbitrariamente un valor específico para la pendiente y la ordenada al origen. Al quedarse con una única hipótesis, las predicciones que el modelo hace del siguiente dato depende exclusivamente de la hipótesis seleccionada. Si esa hipótesis falla, todo el modelo falla. Por el contrario, cuando se distribuye creencias de forma óptima dada la información disponible, el modelo predice el siguiente dato con la contribución de todas las hipótesis (mediante la distribución marginal). Con que haya una única hipótesis que no falle, el modelo no falla, se adapta y aprende.

# Pregunta: 5 Evaluación
Creencia de referencia: [0.0, 1.0]
Creencia de respuesta: [0.95, 0.05]
Cross entropy pregunta: -4.321928094887363

Justificación: Si bien en el area de ML e IA hay muchas métricas para evaluar modelos, existe una forma universal de evaluar los modelos, aplicando las reglas de la probabilidad. En la cual lo que se suele hacer es calcular la predicción del modelo para un dato -> P(Dato = d | H1, H2, ... , Hn, Modelo). Pero a partir de esto podemos llegar a través de cuentas a que P(Modelo | Dato = d, H1, H2, ... , Hn) = P(D|M,H)*P(M|H)/P(D|H). Con esto podemos llegar finalmente a una probabilidad, que refleja la credibilidad de un modelo una vez vistos los datos y bajo ciertas hipotesis.

Justificación de referencia: Desde el descubrimiento de las reglas de la probabilidad propuestas a finales del siglo 18 hasta ahora no se ha propuesto un sistema de razonamiento para contextos de incertidumbre alternativo, mejor en términos prácticos. En probabilidad las hipótesis se evalúan a través de la distribución condicional de la hipótesis dado el dato, P(H|D). Esta es la única forma de evaluar hipótesis, incluyendo modelos alternativos. Si descomponemos esa distribución conjunta veremos que el único factor que actualiza nuestras creencias es la predicción que la hipótesis hace del conjunto de datos, P(D|H). La predicción a priori sobre todo el conjunto de datos se puede descomponer como una secuencia de predicciones, donde la creencia inicial es filtrada mediante la sorpresa, única fuente de información. La hipótesis que predice con 1 (sorpresa nula), preserva toda la creencia previa. La hipótesis que predice con 0 (sorpresa total), se hace falsa para siempre. Esta es la única regla correcta si consideramos que se basa en la aplicación estricta de las reglas de la probabilidad. Si restringimos la discusión a ciertas propiedades específicas, podemos ampliar levemente la gama de métricas. Una propiedad fundamental contenida naturalmente en éstas reglas de la probabilidad se obtiene la mejor puntuación promedio posible sólo si se anuncia exactamente lo que cree (honestidad). La gran mayoría de las métricas de ML, como accuracy, precision, recall, F-score, no cumplen con esta propiedad. Las métricas que cumplen con esta propiedades se conocen como 'strict proper scoring rules'.

# Pregunta: 6 Predicción
Creencia de referencia: [0, 0, 1.0, 0, 0]
Creencia de respuesta: [0.01, 0.3, 0.67, 0.01, 0.01]
Cross entropy pregunta: -0.5777669993169522

Justificación: Si se hace una buena selección de modelo causal, siendo este el que genera la realidad causal subyacente del problema en cuestión, siempre o casi siempre va a predecir mejor que cualqueir algoritmo complejo de IA o DeepLearning.

Justificación de referencia: Los argumentos causales que se corresponden con la realidad causal subyacente necesariamente son las hipótesis de nivel superior (teorías) que producen la menor cantidad de sorpresa, y ningún modelo de inteligencia artificial, por más complejo que sea, puede mejorar su desempeño. Esto se hace evidente cuando calculamos qué pasa con la predicción de los modelos. Hemos visto que la media geométrica de las predicciones, en el límite temporal, en escala logarítmica no es más que el negativo de la entropía cruzada entre la realidad causal subyacente y las predicciones del modelo. La ventaja de los modelos causales es justamente su capacidad para adaptar sus predicciones a diferentes contextos. Las redes causales permiten representar y adaptarse fácilmente a cambios en el entorno. Si ocurre una modificación local en uno de los mecanismos del sistema, como por ejemplo cuando intervenimos una variable asignándole una valor determinado, esa alteración puede expresarse directamente en la estructura de la red mediante una mínima cantidad de ajustes en las conexiones. Este es el caso del do-operator. En cambio, si la red no estuviera organizada causalmente, dichos cambios requerirían una reestructuración mucho más compleja. La fuente de esta flexibilidad radica en el hecho de que las relaciones entre las causas y sus efectos están relacionados por mecanismos causales estables y autónomos. Por este motivo, las teorías causales son una especie de oráculo que puede predecir las consecuencias de una gran cantidad de acciones y combinaciones posibles.

# Pregunta: 7 Factor graph
Creencia de referencia: [0.0, 1.0]
Creencia de respuesta: [0.05, 0.95]
Cross entropy pregunta: -0.07400058144377693

Justificación: Esta afirmación es verdadera. Las variables se conectan unicamente con las funciones y las funciones unicamente con los nodos, esto hace que sea un grafo bipartito.

Justificación de referencia: Esta es la definición precisa de un grafo de factores (factor graph): un grafo bipartito con nodos variables y nodos factores (distribuciones) en el que los ejes conectan las variables con los factores de los que son argumento. Es un tipo de modelo gráfico que se utiliza en probabilidad para representar la factorización de la distribución de probabilidad conjunta.

# Pregunta: 8 do-operator
Creencia de referencia: [0.0, 1.0]
Creencia de respuesta: [0.05, 0.95]
Cross entropy pregunta: -0.07400058144377693

Justificación: Se puede. Justamente una de las grandes ventajas de los factor-graphs vs las redes bayesianas es este concepto. Cuando se aplica un do-operator a una variable, se cambia su distribución de probabilidad por una determinista, que funciona como una compuerta lógica.

Justificación de referencia: En los modelos causales las distribuciones de probabilidad condicional representan mecaismos causales. Los mecanismos causales que pueden ser modificados por nuestra intervención pueden ser representados mediante una función partida. Si no hay intervención entonces el mecanismo causal que opera es la distribución de probabilidad condicional que vincula a la variable con sus causas naturales, P(v|causas). Si hay intervención entonces el mecanismo causal que opera es la distribución de probabilidad indicadora, I(v = intervención). Luego, se puede definir una única distribución condicional. P(v|causas,intervencion) = P(v|causas)^{1-def(intervencion)}I(v=intervencion)^{def(intervencion)}, donde la función def() indica si la variable intervencion está definida. En una factor graph este tipo de distribuciones se puede representar mediante la notación de compuertas (gates).

# Pregunta: 9 Sum-product marginal
Creencia de referencia: [0.0, 1.0]
Creencia de respuesta: [0.05, 0.95]
Cross entropy pregunta: -0.07400058144377693

Justificación: El algoritmo de pasaje de mensajes entre nodos funciona exactamente asi. Se apoya en el algoritmo sum-product, que a su vez son las dos reglas claves en la probabilidad. La marginal de una variable es el producto de los mensajes recibidos que a su vez son marginales, ya que el algoritmo de pasaje de mensajes, en cada pasaje va marginalizando sobre todas las variables previas.

Justificación de referencia: Esta es la esencia del algoritmo de suma-producto. En el sum-product algorithm, la distribución marginal de una variable es exactamente el producto de todos los mensajes que llegan a ese nodo. El algoritmo descompone las reglas de la probabilidad como mensajes que se envían los nodos de del factor graph, lo que permite calcular de manera eficiente las distribuciones marginales de las variables.

# Pregunta: 10 Estructura básica
Creencia de referencia: [1.0, 0.0]
Creencia de respuesta: [0.95, 0.05]
Cross entropy pregunta: -0.07400058144377693

Justificación: X e Y no son independientes cuando condicionamos por W. Ya que cuando hacemos esto se activa el collider y se abre el flujo de inferencia. Lo resolví planteando la red bayesiana y simulando el ejemplo de clase, con terremoto, entradera, etc. Esto seria similar a calcular P(Entradera|Llamada) ¿=? P(Entradera|Terremoto, Llamada) lo cual obviamente son distintos ya que en el segundo termino aporta información el terremoto y se puede hacer inferencia para determinar la entradera o X en este caso.

Justificación de referencia: M es un collider entre X e Y, y W es un descendente de M. Condicionar por W tiene consecuencias similares a condicionar por M, pues W es una 'proxy' de M y por lo tanto es como observar M de forma indirecta. Luego, si M es observado (indirectamente a través de W), al ser un collider sabemos que X no es independiente de Y dado W. Al condicionar en W, se abre el camino de inferencia entre X e Y, haciéndolos dependientes

# Pregunta: 11 Predicción causal
Creencia de referencia: [1.0, 0.0]
Creencia de respuesta: [0.95, 0.05]
Cross entropy pregunta: -0.07400058144377693

Justificación: Esto es falso ya que sin intervenciones no se puede despejar esa confusión entre correlación y causalidad. A su vez sin conocer la realidad causal subyacente no se puede hacer intervenciones, ya que si no conocemos la realidad causal e intervenimos podemos estar abriendo flujos de inferencia no deseados o correlaciones espurias.

Justificación de referencia: Con datos puramente observacionales, podemos medir asociaciones estadísticas, pero no podemos, sin supuestos adicionales, distinguir si X causa Y, si Y causa X, o si una tercera variable causa a ambas. Para estimar un efecto causal a partir de datos observacionales, es indispensable hacer supuestos sobre la estructura causal subyacente (por ejemplo, en la forma de un grafo causal). Estos supuestos son los que nos permiten identificar por qué caminos transita el la asociación espuria y por que caminos transita la asociación causal. Solo si se conoce la estructura podemos identificar el conjunto de variables de control que permiten eliminar las asociaciones espurias de las prediciones, haciendo que estas se correspondan con sus efectos causales.

# Pregunta: 12 d-separation
Creencia de referencia: [1.0, 0.0]
Creencia de respuesta: [0.95, 0.05]
Cross entropy pregunta: -0.07400058144377693

Justificación: Esta afirmación no refleja correctamente las reglas de d-separation, es justamente lo contrario a la hora de hablar de pipes. Si condicionamos por una de las variables del medio de la cadena, como por ejemplo A -> B -> C, si condicioinamos por B, se bloquea el flujo de inferencia.

Justificación de referencia: Hay flujo de inferencia entre los extremos de una cadena si y solo si se condiciona en todas las consecuencias comunes (o sus descendentes) y en ninguna otra variable. La consigna es falsa debido a que afirma la necesidad de que se condicione en los colliders, cuando en realidad puede haber flujo de inferencia sin condicionar sobre los colliders (condicionando sobre las consecuencias de los colliders).

# Pregunta: 13 Adjustment formula
Creencia de referencia: [1.0, 0.0]
Creencia de respuesta: [0.95, 0.05]
Cross entropy pregunta: -0.07400058144377693

Justificación: Esto es falso ya que no esta del todo completo. Si el conjunto de variables Q bloquea todos los caminos ascendentes de X a Y se esta cumpliendo con uno de dos de los criterios necesarios para que se cumpla BACKDOOR. Falta que también el conjunto Q no contenga descendientes de X.

Justificación de referencia: El criterio backdoor tiene dos condiciones. Una de las condiciones es la que se menciona en el enunciado. Cerrar el flujo de asociación en todos los caminos ascendentes de $X$ a $Y$. Sin embargo, es importante no cerrar el flujo de inferencia eligiendo variables descendentes a X, porque en ese caso se estaría cerrando flujo de asociación causal que queremos preservar. Además, es importante no condicionar sobre variables descendentes a X para no abrir flujo de asociación espurias que se encuentras cerrados a través de un collider no observado (como una variable causada por X e Y simultáneamente).

# Pregunta: 14 Adjustment formula
Creencia de referencia: [1.0, 0.0]
Creencia de respuesta: [0.95, 0.05]
Cross entropy pregunta: -0.07400058144377693

Justificación: Si las variables Q cumplen con el criterio backdoor, esta afirmación no es cierta. La primera igualdad si se cumple, P(Y|do(X)) = P_M_x(Y|X). Lo que no termina de ser verdadero es la segunda igualdad, ya que P(Y|do(X)) = SUMA_Q[ P(Q)P(Y|X,Q) ] y no SUMA_Q[ P(Q|X)P(Y|X,Q) ]. Esto se debe a que si intervenimos en X, esta variable se desconecta de sus causas y en el modelo intervenido Q ya no depende de X, solo de si misma, es por eso que va P(Q) y no P(Q|X)

Justificación de referencia: La igualdad no es cierta. La adjustment formula correcta es P_{M_x}(Y|X) = \sum_Q P(Y|X,Q)P(Q). Esto contrasta con la ecuación de la derecha, que en vez integrar pesando por P(Q) integra pesando por P(Q|X). Ambas ecuaciones serían iguales si P(Q|X) = P(Q). Sin embargo esto no es cierto, pues en el modelo no intervenido X no es independiente de las variables de control Q. Q son variables de control que cumplen con el criterio backdoor, y por lo tanto son variables ascendentes a X. Si el modelo estuviera intervenido, X no recibiría flechas de ninguna variable pues no tendría causas y sería independiente de todas las variables ascendentes. Sin embargo, en el modelo no intervenido X está conectado con las variables ascendentes a través de las flechas que recibe de sus causas.

# Pregunta: 15 Ignorar intervención
Creencia de referencia: [1.0, 0.0]
Creencia de respuesta: [0.95, 0.05]
Cross entropy pregunta: -0.07400058144377693

Justificación: Cuando hacemos do(x) se elimina la influencia de las causas de X. Y como se cumple el criterio de Backdoor, la probabilidad de Y dado X es la misma en el modelo real (sin intervención) Como en elintervenido, por lo cual PMx(Y|X,Q) = P(Y|X,Q), lo cual hace que sea falso  PMx(Y|X,Q) != P(Y|X,Q).

Justificación de referencia: En P_{M_x}(Y|X,Q) se está realizando una intervención que corta todas las flechas entrantes a X. Esto elimina forzosamente cualquier posible flujo de asociación espurio que vaya por los caminos traseros entre X e Y. Además, como Q cumple con el criterio backdoor, ninguna de las variables Q se encuentra en los caminos descendentes por lo cual, no se bloquea ninguna asociación causal entre X e Y ni se abren asociaciones espurias descendentes entre X e Y. Toda la asociación fluye desde X hacia Y exclusivamente a través de los caminos causales directos que conectan a X con Y. Por otro lado, P(Y|X,Q) no se está interviniendo, y por lo tanto la variable X recibe todas las flechas que la vinculan con sus causas naturales. A pesar de que ahora existan caminos backdoor, la asociación espuria circula únicamente por los caminos causales que conectan a X con Y. La asociación espuria está bloqueada, tanto en los caminos traseros como en los delanteros por gracias a que las variables de control Q cumplen el criterio backdoor (cierran flujo trasero sin bloquear asociación causal ni generar nuevas asociaciones espurias).

# Pregunta: 16 Independencia
Creencia de referencia: [0.0, 1.0]
Creencia de respuesta: [0.05, 0.95]
Cross entropy pregunta: -0.07400058144377693

Justificación: Si Q no incluye descendientes de X, esto es cierto ya que Q no se ve afectado por la intervención. En el modelo intervenido, X y Q son independientes. Esto implica que la distribución condicional se reduce a la marginal, por lo cual PMx(Q|X) = PMx(Q).

Justificación de referencia: En ambos lados de la igualdad la intervención elimina todas las flechas entrantes a X haciendo que la variable X se convierta en un nodo raı́z. La clave está en que la variable X sólo afecta a sus descendientes pero no a su ancestros, porque todos los caminos que conectan a X con Q en el grafo intervenido necesariamente tienen un collider no observado que la hacen independiente de Q. Esta independencia permite eliminar a la variable X del condicional sin que se produzca ningún cambio en la probabilidad de Q en el grafo intervenido. Luego, P_Mz(Q|X) = P_Mz(Q)

# Pregunta: 17 Variables de control
Creencia de referencia: [1.0, 1.0]
Creencia de respuesta: [0.95, 0.05]
Cross entropy pregunta: -4.395928676331139

Justificación: En este caso no se cumple backdoor debido a que este criterio exige que todos los caminos que vayan de T a Y , estén bloqueados cuando condicionamos en las variables de control. Pero en este caso existe el camino T <- U -> Y el cual mete correlación espuria entre T e Y.

Justificación de referencia: Esta pregunta no va a ser tenida en cuenta en la evaluación. La consigna es exactamente igual que a la siguiente. Por este motivo, las opciones de respuesta (No/Sí cumple backdoor) no se corresponde con la pregunta ¿Podemos estimar el efecto causal únicamente con la información de T, M e Y?. La pregunta debería haber sido: ¿Existe un conjunto de variables que cumpla con el criterio backdoor?. Ante este error en la redacción del enunciado, ambas respuestas pueden ser válidas. Por un lado es cierto que ninguna variable cumple con el criterio backdoor entre T e Y. Pero al mismo tiempo, es posible estimar el efecto causal únicamente con la información de T, M e Y mediante el procedimiento frontdoor.

# Pregunta: 18 Causa común oculta
Creencia de referencia: [0.01, 0.99]
Creencia de respuesta: [0.95, 0.05]
Cross entropy pregunta: -4.198655683857015

Justificación: No podemos estimar el efecto causal unicamente con la información de T, M e Y ya que U es la causa común oculta de ambas varaibles. Es la que en parte genera a estas variables de interés.

Justificación de referencia: Sí, es posible estimar el efecto causal. Aunque no hay ningún conjunto de variables que cumplan el criterio backdoor entre T e Y, la estructura del problema cumple las condiciones para aplicar una estimación del efecto causal front-door. La variable mediadora M satisface las tres condiciones necesarias: 1. M intercepta todos los caminos dirigidos de T a Y; 2. No hay caminos traseros no bloqueados entre T y M; 3. Todos los caminos traseros entre M e Y son bloqueados por T. Esto permite usar la fórmula de ajuste frontdoor, que integra el efecto causal de T sobre M con el efecto causal (controlado por T) de M sobre Y.

# Pregunta: 19 Experimento sin cumplimiento
Creencia de referencia: [0.1, 0.9]
Creencia de respuesta: [0.05, 0.95]
Cross entropy pregunta: -0.02979773919885431

Justificación: Es posible. Aun que el tratamiento efectivo T no siga la dist. de Z (la cual es aleatoria), es posible estimar el efecto causal porque la asignación se generó de manera exógena e independiente de las características ocultas (C). La correlación de Z y el resultado Y solo puede explicarse por la inglencia de Z sobre T, lo que habilita identificar el efecto de T sobre Y

Justificación de referencia: Sí, es posible estimar un efecto causal. Esta situación se conoce como un problema de variables instrumentales. La asignación aleatoria Z funciona como una variable instrumental válida porque: Z es causa directa del tratamiento T, Z solo afecta al resultado Y exclusivamente a través de T, y en ninguno de los dos efectos causales (Z-T y Z-Y) hay asociación espuria. En clase hemos visto que en caso de que la relación entre las variables sea lineal, el efecto causal se puede estimar como el cociente de las regresiones lineales. Por lo tanto, cuando suponemos una relación paramétrica entre las variables, sabemos que sí es posible estimar el efecto causal. No hemos visto aún qué podemos hacer en términos generales, de forma no paramétrica. En esos casos también podemos estimar efectos causales, aunque solo hay garantías de que sea válido para un subconjunto de la población. Por eso en esta pregunta asignamos solo 0.9 de creencia a la respuesta 'Sí es posible'.

# Pregunta: 20 Diversificación
Creencia de referencia: [0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0]
Creencia de respuesta: [0.025, 0.1, 0.19, 0.19, 0.15, 0.15, 0.1, 0.05, 0.025, 0.01, 0.01]
Cross entropy pregunta: -2.736965594166206

Justificación: La verdad no tengo tiempo de hacer la cuenta a mano para entender que es lo mas efectivo, pero seguramente si asignamos un poco menos de los mitad de los recursos a cara, 'seguiremos con vida' mas tiempo y creciendo mas ya que cara paga 3 y sello 1.2. Lo que significa que en una mala racha para una P(cara) = 0.5 si tiramos n veces y pocas caras, las cveces que toco cara habremos recuperado bastante, sin perder mucho en las que salio sello.  

Justificación de referencia: La riqueza en este juego de apuestas se actualiza siguiendo un proceso 'multiplicativo'. Supongamos que la riqueza inicial es w, y que b son los recursos asignados a Cara. Si en el primer paso temporal sale Cara, los recursos será el producto w*b*3, porque la casa de apuesta paga 3 por wb de los recursos apostados a Cara (los recursos apostados a Sello se perdieron). Si en el segundo paso temporal sale Sello, los recursos serán (w*b*3)*(1-b)*1.2, porque la casa de apuesta paga 1.2 por una proporción (1-b) de los recursos previos (w*b*3). Debido a que la moneda tiene una probabilidad 0.5 de salir Cara, en tiempo infinito van a salir Nc Caras y Ns Sellos, con Nc = Ns y T = Nc + Ns. La riqueza en el límite va a ser w*(b*3)^Nc*((1-b)*1.2)^Ns. Y su media geométrica (la raiz 1/T) va a ser w*(b*3)^0.5*((1-b)*1.2)^0.5. Para determinar cuál es la apuesta óptima necesitamos comparar dos apuestas distintas, digamos b y d. Si hacemos el cociente de la media geométrica de sus riquezas los pagos de la casa de apuestas se cancelan!! También se cancela la riqueza inicial. El cociente queda determinado por b^0.5*(1-b)^0.5/d^0.5*(1-d)^0.5. Maximizar b^0.5*(1-b)^0.5 es lo mismo que minimizar su logaritmo  0.5 * log b + 0.5 * log (1-b). Esto no es más que el negativo de entropía cruzada! Y la entropía cruzada se maximiza cuendo es entropía, cuando b = 0.5

# Pregunta: 21 Apuesta individual
Creencia de referencia: [1.0, 0, 0]
Creencia de respuesta: [0.7, 0.15, 0.15]
Cross entropy pregunta: -0.5145731728297583

Justificación: Lo que importa es si nos conviene a nosotros como individuos jugar, independientemente del hecho de que en conjunto se crece un 5% por jugada en promedio. Considero que no es conveniente jugar... Ya que el capital se reduce en una proporción mucho mas grande de lo que crece cuando ganamos. Como ambos eventos tienen la misma probablidad, este comportamiento hace que tu capital a largo plazo tiende a 0.

Justificación de referencia: Supongamos que la riqueza inicial es 1. Supogamos que en el primer paso sale Sello. La riqueza va a pasar de 1 a 0.6, caemos 40%. Y si en el siguiente paso sale Cara crecemos 50%, de 0.6 a 0.9. Si esto se repite sucesivamente, como debería ocurrir con una moneda normal, vamos a estar cayendo 10% por cada dos pasos temporales. Por lo tanto no nos conviene jugar. Si revisamos la media geométrica de los recursos (0.5*3)^0.5 * (0.5*1.2)^0.5 = 0.95. Es decir, la tasa de crecimiento de los recursos en el tiempo es -5% por paso temporal.

# Pregunta: 22 Fondo común
Creencia de referencia: [0.0, 1.0, 0.0]
Creencia de respuesta: [0.15, 0.7, 0.15]
Cross entropy pregunta: -0.5145731728297583

Justificación: En este caso es donde se ve esa riqueza promedio de que crece a una tasa de 5%, ya que al jugar en equipo a esto se mezclan los resultados de muchas personas y se va redistribuyendo en cada paso. De esta forma uno esperaría llegar a ese 5 porciento de crecimiento en cada jugada.

Justificación de referencia: Supongamos que el fondo común está compuesto por solo dos personas. Supongamos los recursos iniciales son 1 para ambos. Si a amabas personas le sale Cara en el primer paso temporal, los recursos van a ser 1.5 para ambos. Si a una persona le sale Cara y a la otra Sello, los recursos van a ser (1.5+0.6)/2 = 1.05. Y a la inversa lo mismo. En estos 3 casos nuestra riqueza aumenta. Solo en el caso de que a ambas personas le salga Sello es cuando la riqueza de ambas personas cae 40%. Antes teniamos probabilidad 0.5 de caer 40%, ahora solo probabilidad 0.25 de caer 40%. Si calculamos la media geomtrica van a ver que la tasa de crecimiento va a ser mayor. En el extremo, cuando el tamaño del grupo tiene a infinito, a la mitad de las personas les va a salir Cara y a la mitad les va a salir Sello. Luego, la tasa de crecimiento en todos los pasos temporales va a ser de 5%, (1.5*(N/2)+0.6*(N/2))/N = 1.05.

# Pregunta: 23 Tragedia de los comunes
Creencia de referencia: [1.0, 0.0, 0.0]
Creencia de respuesta: [0.1, 0.8, 0.1]
Cross entropy pregunta: -3.321928094887362

Justificación:  Para este caso se pueden aplicar muchos casos similares. Uno de ellos es por ejemplo que todo un barrio pague para la limpieza del barrio, si uno deja de aportar, seguimos disfrutando del beneficio ya que hay otros que siguen pagando. A su vez me estoy ahorrando el dinero por no pagar el servicio. Este comportamiento hace que si todos toman esta decisión el sistema colectivo deja de funcionar. Pero en terminos monetarios unicamente, claramente conviene mas dejar de aportar al fondo común y seguir recibiendo la cuota en partes iguales del fondo común y seguir recibiendo la parte correspondiente.

Justificación de referencia: A primera vista puede parecer buena idea evitar el costo (de aportar al fondo común) mientra se sigue recibiendo los beneficios (la cuota en partes iguales del fondo común). Sin embargo, en un grupo de tamaño 2 dejar de aportar al fondo común significaría volver a jugar individualmente, porque la persona que sigue aportando va a quedar rápidamente en bancarrota. Lo que ya hemos visto que es una mala idea. Si la otra persona queda en bancarrota y me veo obligado a jugar solo, no importa cuál sea los recursos iniciales que tenga, a tasa de crecimiento individual es negativa. Algo similar ocurre en cualquier grupo de tamaño finito. La demostración matemática no es compleja, aunque un tanto larga de desarrollar aquí. También se verifica cuando realizamos simulaciones. Ahí se puede ver que la tasa de crecimiento de la primera persona que deja de aportar mientras sigue recibiendo del fondo común es menor a la que tenía cuando aportaba al fondo común. Intuitivamente, esto ocurre porque al dejar de aportar afecta la tasa de crecimiento de las personas de las cuales depende y por lo tanto reduce su propia tasa de crecimiento.

- - -Puntaje global: -20.063647851330995