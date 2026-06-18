# Metodología técnica del proyecto

## 1. Datos

Se integran tres bloques:

1. Base Colombia.
2. Enfermedad holandesa.
3. Contexto global de mercados y commodities.

## 2. Tratamiento

- Homogeneización de fechas.
- Conversión de variables numéricas.
- Imputación por mediana dentro del pipeline.
- Construcción de rezagos, retornos y ventanas móviles.
- Diagnóstico de frecuencia efectiva del oro.

## 3. Regímenes

Se construyen dos vistas:

- `ABS`: niveles de variables, interpretación estructural.
- `CHG`: cambios de variables, interpretación dinámica.

Flujo:

```text
variables macrofinancieras → escalamiento → UMAP → KMeans → regímenes
```

## 4. Target

La variable objetivo principal es:

```text
target_up_h21 = 1 si el retorno logarítmico futuro a 21 observaciones es positivo
```

## 5. Modelos

Se comparan:

- HistGradientBoostingClassifier.
- XGBClassifier.
- RandomForestClassifier.
- LogisticRegression.
- Reglas determinísticas de momentum y medias móviles.

## 6. Validación

Se usa validación walk-forward:

```text
pasado → entrenamiento
pasado reciente → calibración de umbral
futuro → prueba
```

## 7. Selección

La métrica principal es `balanced_accuracy`, porque el problema es binario y puede tener desbalance de clases.

## 8. Dashboard y simulador

El dashboard consolida:

- resultados predictivos,
- regímenes,
- errores,
- simulador,
- explicabilidad,
- auditoría anti-fuga.
