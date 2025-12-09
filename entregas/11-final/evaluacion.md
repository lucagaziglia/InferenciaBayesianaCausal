# Evaluación trabajo final.

## Ejercicio 1.

Perfecta las evidencias de los modelos. En un momento se rompe la del modelo alternativo, dan todos NANs, pero hasta ese momento vienen perfectas las evidencias. No encontré el error. Muy buen plantado el ejercicio. Mirando los primero 100 datos vemos que:

```
        Base  MontyHall  Alternativo
0   0.055556   0.111111     0.083333
1   0.055556   0.055556     0.055556
2   0.055556   0.111111     0.086481
3   0.055556   0.111111     0.089400
4   0.055556   0.111111     0.092009
..       ...        ...          ...
95  0.055556   0.111111     0.102293
96  0.055556   0.055556     0.055556
97  0.055556   0.055556     0.055556
98  0.055556   0.000000     0.008681
99  0.055556   0.055556     0.055556
```

## Ejercicio 2.

Muy buena visualización del Dag incluyendo variables exógenas. Controla correctamente en términos de backdoor criterion, y se incluyen covariables para efectos causales heterogéneos como `sector`. El método de estimación es una regresión lineal a través del ecosistema de inferencia causal de Python, DoWhy.

```
model = CausalModel(
    data=df_model,
    treatment='monto_credito',
    outcome='ventas',
    common_causes=['antiguedad', 'plazo', 'compras_pasadas', 'nivel_riesgo'] +
    [c for c in df_model.columns if 'sector_' in c],
    effect_modifiers=['plazo']
)
```

El modelo saca 11 coeficientes. No me queda claro cómo determinar qué coeficientes se corresponde con qué variables. Acá se toma

```
beta_monto = coefs_values[1]
beta_interaccion = coefs_values[-1]
```

El enfoque está bien planteado. La estimación de los CATE son
```
Si el plazo es 1 días, por cada $1 extra de crédito, las ventas suben: $-0.0605
Si el plazo es 2 días, por cada $1 extra de crédito, las ventas suben: $-0.0499
Si el plazo es 3 días, por cada $1 extra de crédito, las ventas suben: $-0.0393
Si el plazo es 4 días, por cada $1 extra de crédito, las ventas suben: $-0.0288
Si el plazo es 5 días, por cada $1 extra de crédito, las ventas suben: $-0.0182
Si el plazo es 6 días, por cada $1 extra de crédito, las ventas suben: $-0.0076
Si el plazo es 7 días, por cada $1 extra de crédito, las ventas suben: $0.0030
Si el plazo es 8 días, por cada $1 extra de crédito, las ventas suben: $0.0136
Si el plazo es 9 días, por cada $1 extra de crédito, las ventas suben: $0.0242
Si el plazo es 10 días, por cada $1 extra de crédito, las ventas suben: $0.0348
```

Y del ATE es
```
Efecto Promedio Global (ATE): $-0.0310
```

### Explicación del dataset simulado.

Este ejercicio es extremadamente difícil de resolver debido a una muy baja aleatoriedad con la que se generó la variable tratamiento. En el siguiente código se reproduce la simulación de la variable tratamiento.

```
monto_credito = (
    f(compras_pasadas, nivel_riesgo,plazo) *
    np.random.uniform(0.98, 1.02, n) # 2% de ruido
)
```

Si la simulación se hubiera hecho con una función totalmente determinista, sería imposible evaluar los efectos causales porque dadas las variables de control todos reciben siempre el mismo tratamiento, haciendo que sea imposible conocer el efecto de las variaciones en el tratamiento sobre la variable objetivo (no tenemos con qué comparar). Es necesario que existe un mínimo de solapamiento (overlap) entre los tratamientos. Este requisito también se conoce como positividad, porque se puede traducir a que el propensity score, P(T|Q), no sea 0.

Los datos simulados no se generaron de forma totalmente determinista, se le agregó un ruido aleatorio del 2% sobre el cual los modelos van a aprovechar para evaluar el impacto que las variaciones en el tratamiento tiene sobre la variable objetivo para cada uno de los contextos de control. En este caso la positividad solo es local, y el overlap es parcial, lo que hace que los efectos causales simulados sean extremadamente difícil de estimar.

Además, hay que tener en cuenta el balance entre preservar el overlap y el tamaño del conjunto de variables de control. Aunque condicionar en más covariables podría aumentar las chances de eliminación de la asociación espuria, también puede aumentar la chances de perder la positividad existente. Esto ocurre porque al incrementar la dimensionalidad de las covariables Q, hacemos que los subgrupos para cualquier nivel Q sean más pequeños, aumentando así la probabilidad de que todo el subgrupo tenga un único tratamiento asociado.

En particular, los modelos de Double Machine Learning se pueden ver afectados severamente por la falta de overlap entre los tratamientos. Para que tengan una noción del problema, acá se muestra el efecto causal simulado.

```
venta = (
    compras_pasadas * u2_estacionalidad * u3_salud + # Sin efecto
    monto_credito * 0.04 + # Efecto del crédito
    monto_credito*plazo* (0.08/15) + # Interacción con plazo
    np.random.normal(0, 1) # Ruido
)
```






