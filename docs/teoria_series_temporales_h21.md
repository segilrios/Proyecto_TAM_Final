# Teoría usada: series temporales, componentes y horizonte h21

## Descomposición clásica

Una serie temporal puede escribirse como una suma de componentes:

```text
y_t = T_t + S_t + C_t + ε_t
```

Donde:

- `T_t`: tendencia de largo plazo.
- `S_t`: estacionalidad.
- `C_t`: ciclo económico o financiero.
- `ε_t`: ruido o componente no explicada.

## Adaptación al proyecto

Para el oro colombiano se usó una lectura más económica:

```text
y_t = T_t + ABS_t + CHG_t + G_t + L_t + ε_t
```

- `ABS_t`: estado macrofinanciero en niveles; régimen estructural.
- `CHG_t`: componente de cambios o shocks.
- `G_t`: contexto global.
- `L_t`: contexto local colombiano.
- `ε_t`: ruido.

## Horizonte h21

El objetivo no es explicar solo `y_t`, sino el cambio acumulado entre `t` y `t+21`:

```text
r_{t,21} = log(y_{t+21}) - log(y_t)
```

La clase supervisada es:

```text
target_up_h21 = 1 si r_{t,21} > 0
target_up_h21 = 0 si r_{t,21} <= 0
```

## Por qué h21

La base tiene frecuencia nominal diaria, pero el precio del oro presenta pocos valores únicos frente al número total de registros. Esto indica una frecuencia efectiva cercana a mensual. En finanzas, 21 observaciones se usan como aproximación de un mes operativo.
