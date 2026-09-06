# Sistema de tracking de pelota — indice de comandos

Todo lo que se puede correr, en un solo lugar. El detalle de cada modulo esta
en su docstring; el estado del proyecto y los numeros medidos, en `ESTADO.md`.

Cada archivo acepta `--help`.

---

## Antes que nada

```
python inferencia.py            # que backend eligio y como es el modelo
python fuente.py --carpeta imagenes/Soleado --n 5
```

Si `inferencia.py` describe el modelo con `IN (640, 1152, 3)` y
`CLASES 1 -> 0:ball`, el entorno esta bien.

`FileNotFoundError` = la ruta del `.pt`. Esta en `config.MODELO_PT`, y se
puede pisar sin editar nada: `MODELO_PT=modelos/otro.pt python replay.py ...`

---

## replay.py — el banco de pruebas

La herramienta principal. Corre el pipeline sobre material grabado, sin motor,
sin encoder y sin GoPro. Escribe CSV + mp4 anotado + resumen.

```
python replay.py --carpeta imagenes/Soleado
python replay.py --video grabacion.mkv --salida-csv salidas/r.csv --salida-mp4 salidas/r.mp4
python replay.py --carpeta imagenes --limite 300 --sin-mp4
```

### Los dos modos, y cual usar

| | cuando | que mide |
|---|---|---|
| normal | **secuencia continua** (video o rafaga) | tracking: % en TRACK, caidas |
| `--solo-search` | **imagenes sueltas** | recall puro, comparable entre carpetas |

Con imagenes sueltas el modo normal MIENTE: despues de la primera deteccion
buena entra a TRACK y recorta 1152x640 alrededor de donde estaba la pelota en
OTRA foto, asi que pierde detecciones por un motivo que no tiene nada que ver
con el modelo. Y el efecto depende del orden de los archivos, o sea que
contamina la comparacion entre carpetas.

```
python replay.py --carpeta imagenes/Soleado --solo-search
```

### Comparar A/B: cuanto aporta cada mecanismo

El tracker agrega DOS cosas sobre la maquina de estados vieja. Los
interruptores estan separados a proposito: con uno solo no se sabe a cual
atribuir la diferencia.

```
python replay.py --video v.mp4 --sin-csv --sin-mp4                          # completo
python replay.py --video v.mp4 --sin-csv --sin-mp4 --sin-kalman             # sin prediccion
python replay.py --video v.mp4 --sin-csv --sin-mp4 --sin-kalman --sin-gate  # = state_machine.py
```

- `--sin-kalman`: la prediccion deja de usarse para elegir; se toma el
  candidato mas confiado. El filtro igual corre y `err_pred` se sigue
  reportando, asi que las corridas son comparables.
- `--sin-gate`: no se rechaza ningun salto por implausible.

El titulo del resumen dice la variante (`[completo]`, `[sin kalman]`...) para
no mezclar corridas.

**Mira el reparto de motivos, no solo los porcentajes.** Ahi se ve directo
cuantos frames rescato el Kalman (`kalman`) y cuantos falsos positivos freno
el gate (`delta imposible`). Esos dos numeros son el aporte real de cada
mecanismo y son mejor evidencia para la tesis que la diferencia agregada.

### OJO con `--fps` si el material no es de 40 fps

El reloj sintetico avanza 1/fps por imagen, y el gate de plausibilidad escala
con ese tiempo. Con la tasa mal, el gate rechaza detecciones buenas en masa:
sobre `imagenes/test_kalman` (capturada mas espaciada) el default de 40 fps
tiraba el 38% de los frames y la deteccion aceptada caia de 88% a 42%. Con
`--fps 4` volvio a 87.5% y los rechazos a cero.

```
python replay.py --carpeta imagenes/test_kalman --fps 4
```

Si no sabes la tasa, mirala en el resumen: `desplazamiento entre frames`. Con
cajas de ~20 px, un p50 de ~13 px es material continuo a 40 fps; 143 px es
diez veces mas espaciado.

### Otras opciones

`--camara 0|1` (para `pixel_a_angulo`), `--fps` (pisa el reloj sintetico),
`--limite N`, `--modelo ruta.pt`, `--escala`, `--sin-csv`, `--sin-mp4`.

### Columnas del CSV

```
n, ts, modo, camara, tile, conf, x, y, w, h, angulo, sector, flost,
aceptado_por, pred_x, pred_y, err_pred, vx, vy, motivo, dist_ult
```

Las mismas que va a escribir `main_final.py`: el analisis de la tesis es un
solo script para los dos.

- `err_pred` — residuo del Kalman, para tunear `KALMAN_SIGMA_*` (ver
  `config.py`)
- `motivo` — se llena SIEMPRE, tambien al rechazar. Sin esto no se distingue
  "el modelo no vio nada" de "el gate lo rechazo"
- `dist_ult` — desplazamiento contra la ultima aceptada; comparalo con el gate
- `sector` — vacia hasta P4

---

## Tests

```
python test_preproceso.py               # 10, grilla y remapeo de coordenadas
python test_tracker.py                  # 23, Kalman, gate y patron de SEARCH
python test_inferencia_ultralytics.py   # contrato del backend, sin .pt real
```

Ninguno necesita modelo, camara ni NPU.

---

## Modulos sueltos

```
python preproceso.py                    # describe la grilla de SEARCH
python preproceso.py --ancho 2304 --alto 1296 --modelo 1152x640
```

Deteccion sobre UNA imagen, con la caja dibujada y sin maquina de estados:

```
python inferencia_ultralytics.py --imagen foto.jpg --tiles --salida det.jpg
python inferencia_ultralytics.py --info
```

---

## Hardware (solo en la Raspberry Pi)

```
python motor_lib.py --test              # ida y vuelta de 90 grados
python motor_lib.py                     # terminal interactiva
python encoder_lib.py                   # config y monitor del encoder
python gopro_lib.py --estado
python camera_source.py --n 300 --modos # banco de captura
```

**`encoder_lib.py` interactivo expone `poner_cero_aca()`, que reescribe el
registro ZPOS del chip y invalida `ENCODER_GRADOS_EN_MOTOR_CERO = 79.0`.** Con
eso mal, el homing queda corrido en todos los arranques siguientes. No usarla.

---

## Que corre donde

| | PC | Raspberry Pi |
|---|---|---|
| backend | Ultralytics (`.pt`) | Hailo-8 (`.hef`) |
| entrada | carpeta de imagenes / video | Picamera2 |

Los dos se eligen solos (`inferencia.py`, `fuente.py`) y se pueden forzar:

```
BACKEND=ultralytics FUENTE=carpeta python replay.py --carpeta imagenes/Soleado
```

**Las latencias medidas en la PC no significan nada** para el presupuesto de
25 ms de P0: Ultralytics en CPU esta uno o dos ordenes de magnitud por encima
de la Hailo. Ese numero se mide en la Pi con el `.hef`.

---

## Archivos que se van a reemplazar

`main.py` (loop viejo sobre video, lo reemplaza `replay.py`),
`state_machine.py` (lo reemplaza `tracker.py`) y `main_partido.py` (queda como
debug manual). Siguen ahi porque `main_partido.py` importa `main`, y
`gpio_timer.py` importa `Mode` de `state_machine`. No los borres todavia.

---

## Carpetas

```
modelos/    los .pt, versionados. Ver modelos/README.md
imagenes/   ignorada por git
videos/     ignorada por git
salidas/    ignorada por git; aca escribe replay.py por defecto
```