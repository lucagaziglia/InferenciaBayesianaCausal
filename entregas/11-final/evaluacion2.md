
# Pregunta: 1 Flujo
Creencia de referencia: [1.0, 0.0]
Creencia de respuesta: [0.99, 0.01]
Cross entropy pregunta: -0.014499569695115089

Justificación: En la estructura de esta situación que ya hemos analizado bastantes veces durante la materia, esta igualdad no se cumple. En otras palabras, dada la llamada, la entradera y el terremoto no son independientes. Esto se debe principalmente a que a es un collider y l es un descendiente directo de a (l depende unicamente de a). Y cuando nosotros observamos un collider, las variables que 'van hacia' el collider dejan de ser independientes. Lo mismo pasa con la llamada, ya que si te llaman es porque se activo la alarma, entonces analizar esto sabiendo la información de si te llaman o si se activo la alarma, de alguna forma es lo mismo.

Justificación de referencia: La estructura e->a<-t es un collider, y l es un descendiente de a, por lo tanto el flujo entre e y t se abre cuando se observa la variable a (o cualquiera de sus descendientes). Es decir, NO está cerrado el flujo entre la entradera (e) y el terremoto (t) dado la llamada (l). Está abierto. Luego e no es independiente de t dado l. La descomposición P(e,t|l) = P(e|l)P(t|l) no vale de forma general (podría valer en casos bordes muy extremos).

# Pregunta: 2 Backdoor
Creencia de referencia: [0.0, 1.0]
Creencia de respuesta: [0.3, 0.7]
Cross entropy pregunta: -0.5145731728297583

Justificación: Aunque no sabemos la realidad causal subyacente, hice un DAG con el fin de aproximar esta situación (Dejo fotos en la carpeta que subo dentro de 11-final). En ninguna parte del enunciado se menciona que es lo que produce las ventas, pero entiendo que las ventas dependen, tanto del año, como del sector, como de todas las variables. Por otro lado, el enunciado dice que contamos con un .csv con toda esta información, lo que convierte a compras_pasadas, nivel_riesgo, plazo, sector y antigüedad en variables observadas y así poder condicionar explicitamente por ellas. Estas variables observadas permiten bloquear todos los caminos de backdoor entre monto_credito y ventas. por lo que al condicionarlas eliminamos la correlación espuria que generan. Además, cualquier variable no observada que afecte al plazo queda controlada al ajustar por el propio plazo. Bajo estos supuestos, el criterio de backdoor se satisface y el efecto causal de monto_credito sobre ventas puede ser identificado. Variables de control: ['compras_pasadas', 'nivel_riesgo', 'plazo', 'sector', 'antiguedad', 'año_mes']

Justificación de referencia: Si bien no conocemos la estructura causal completa, conocemos TODAS las causas directas del monto_credito. Si condicionamos en todas ellas (compras_pasadas, nivel_riesgo y plazo) vamos a estar cortando COMPLETAMENTE el flujo de inferencia trasero, eliminando por completo la asociación espuria entre monto_credito y ventas. Sea T monto_credito y sea C alguna de sus causa. Luego puede pasar que la causa C sea un fork T <- C -> X o sea pipe T <- C <- X. En ambos casos garantizamos que al condicionar en C se cierre el flujo de inferencia. 

# Pregunta: 3 do-operator
Creencia de referencia: [0.99, 0.01]
Creencia de respuesta: [0.8, 0.2]
Cross entropy pregunta: -0.26113495899145106

Justificación: Si estuvimos trabajando arduamente para estimar el efecto causal y al calcular P(y|do(x)) nos da aproximadamente 0, no necesariamente cometimos un error. Quizas el cliente tiene razón en que P(Y|do(x)) es un valor considerablemente mas alto que cero, pero que a nosotos nos de proximo a cero no implica un error en el análisis, probablemente radique en la especificación del problema a la hora de armar el DAG, y un malentendido de esa estructura, alguna variable o dependencia que no estemos teniendo en cuenta. Lo que le propondría al cliente es que tengamos algunas instancias para entender mejor el problema y generar un DAG mas robusto revisando todas las variables posibles y sus conexiones.

Justificación de referencia: La afirmación no implica necesariamente que en nuestro análisis hayamos cometido un error. El análisis que hicimos es un efecto causal 'general', P(y|do(x)), que está integrando el efecto específico de cada tipo de persona. Si a uno de los grupos la intervención en x afecta muy positivamente y en otros afecta muy negativamente, esos efectos podrían estar cancelándose. Por lo tanto, lo primero que le preguntaría al cliente es si esos efectos pueden ir en sentido contrario, algunos positivos y otro negativos. En caso de que la respuesta sea afirmativa (o no sepa), le propondría hacer una análisis de efectos causales heterogéneos en los que se consideren las características de las personas. De esa forma podríamos ver para qué personas el efecto es positivo y para cuales el efecto es negativo.

# Pregunta: 4 Ignorability
Creencia de referencia: [1.0, 0.0]
Creencia de respuesta: [0.7, 0.3]
Cross entropy pregunta: -0.5145731728297583

Justificación: Si la tesis esta mal redactada y no contiene la información/demostración de porque estos enfoques son distintos, es un error. No considero que sea un error grave ya que lo que se plantea en la tésis es correcto, ya que para alguna situación donde no se presente un Structural Causal Model, estos enfoques no son equivalentes. Al ser un concepto muy poco conocido y que dentro de todas las fuentes de referencia no se valida este concepto, creop que vale la pena aclararlo y demostrarlo contundentemente para que no haya confusiones. Es más, si todos los jurados de la tesis asumen que esto es así y luego les presentas una demostración formal y rigurosa de porque esto no es así, la percepción del jurado pasa a ser sumamente positiva. Esta demostración sería apartir de las dos caracteristicas que tienen los SCM, que son: 1) Todas las variables end ́ogenas son deterministas (para cumplir con consistencia) y 2) Cada variable end ́ogena tiene una ex ́ogena aleatoria (para cumplir con ignorability).

Justificación de referencia: La afirmación que se hace en la tesis respecto de que el cumplimiento de ignorabilidad condicional no implica el cumplimiento de backdoor en cualquier modelo causal generativo es una afirmación correcta, por lo que no hay ningún error grave en ese punto. La equivalencia entre ignorabilidad condicional y backdoor vale solamente sobre un subconjunto de modelos causales generativos definidos por Judea Pearl como Structural Causal Model (SCM). Para que un modelo generativo sea considerado como un SCM es necesario que cada variable endógena sea determinista y tenga una variable exógena aleatoria. La equivalencia se rompe en los casos en los que alguna de las variables endógenas descendientes del tratamiento no tiene una variable exógena asociada (sea porque queda totalmente determinada por el resto de las variables endógenas o sea porque tiene una aleatoriedad intrínseca). En esos casos vamos a poder condicionar sobre la variable descendiente del tratamiento sin variable exógena para cumplir ignorabilidad, lo que rompe con el criterio backdoor. La demuestración rrequiere especificar la distribución conjunta entre el tratamiento factual y los resultados potenciales contrafactuales (a través de las Twin Networks) y mostrando que se cierra el flujo entre ellos a pesar de estar contolando por una variable descendiente.

# Pregunta: 5 Decisiones
Creencia de referencia: [0.0, 1.0]
Creencia de respuesta: [0.1, 0.9]
Cross entropy pregunta: -0.15200309344504995

Justificación: El factor de optimalidad envía a b un mensaje que asigna probabilidad únicamente a la decisión que maximiza la tasa geométrica de crecimiento. Formalmente mensaje(f_optimalidad -> b) = I(b = argmax_b',r_T'(b')). Entonces si imponemos que la decisión fue óptima (O = 1), el factor de optimalidad obliga a que la única decisión posible sea la que maximiza esa tasa, descartando todos los demás valores de b. Ese valor óptimo coincide con la probabilidad real del evento (b = p), lo cual se debe al Criterio de Kelly visto en clase.

Justificación de referencia: El posterior P(b|\mathcal{O}=1, \omega_0=1) garantiza la maximización de las ganancias factuales a largo plazo. El mensaje que envía el factor de la variable optimalidad a la variable de decisión factual es proporcional al producto entre el mensaje descrito en la ecuación y la distribución de probabilidad de la la variable optimialidad (que es una indiciadora).

\sum_{r_T}  I(r_T = (b^{\prime} Q_c)^p ((1-b^{\prime})Q_s)^{1-p}) I(True = (b = argmax_{b^{\prime}} r_T ))

Hay que integrar por r_T, que es una variable oculta. Pero podemos hacer un reemplazo de variables (como las que hicimos en estimación de habilidad), y evitamos hacer la integración.

I(True (b = argmax_{b^{\prime}} (b^{\prime} Q_c)^p ((1-b^{\prime})Q_s)^{1-p}) )

En clase vimos cómo maximiar $ (b^{\prime} Q_c)^p ((1-b^{\prime})Q_s)^{1-p} $, que eso ocurre cuando $b^{\prime} = p$. Luego,

I(True = (b = p)) =  I(b = p)

Si ese es el mensaje que le enviamos a la variable de decisión b, entonces la conjunta

P(b,\mathcal{O}=1, \omega_0=1) \propto P(b)  I(b = p).

Luego el posterior es

P(b|\mathcal{O}=1, \omega_0=1) = I(b = p)

Que es la decisión que garantiza la maximización de las ganancias factuales a largo plazo BAJO ESTE JUEGO de apuestas.


- - -Puntaje global: -1.537577103687044