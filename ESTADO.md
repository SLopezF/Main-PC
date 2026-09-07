# ESTADO

Memoria entre sesiones. Se lee junto con el briefing. Los comandos estan en
`README.md`; aca van el estado y los numeros medidos. Se actualiza al cerrar
cada paso: que se cerro, que archivos se tocaron, **que numeros dio la
medicion**, y que quedo pendiente.

Ultima actualizacion: 2026-09-06

---

## Donde estamos

| Paso | Estado | Nota |
|---|---|---|
| P0 `bench.py` | **no empezado** | bloqueado: no hay `.hef` ni Pi a mano |
| P1 dos camaras + `grabar_dataset.py` | **no empezado** | necesita las dos camaras fisicas |
| P2 `preproceso.py` + `replay.py` | **CERRADO** | linea de base medida: 84.8% TRACK, 79.7% aceptadas |
| P3 `tracker.py` | **CERRADO** | 92.4% TRACK (base 84.8%), 1 caida (base 3) |
| P4 `sectores.py` | **CERRADO** | 20/20 tests; 9 sectores sobre 0..180 |
| P5 `init_sistema.py` | **escrito, sin validar en fierro** | 11/11 tests con hw_falsos |
| P6 a P8 | no empezado | |

### Trabajo fuera de los pasos numerados (hecho, y era prerrequisito)

Capa de abstraccion para poder correr todo en una PC sin Raspberry Pi, sin
Hailo-8 y sin las camaras. No es un paso del briefing, pero sin esto P2 y P3
no se podian iterar.

- `inferencia.py` — elige el backend: Hailo real o Ultralytics. Por
  `BACKEND=hailo|ultralytics`, o `config.BACKEND_INFERENCIA`, o auto segun
  exista `hailo_platform`.
- `inferencia_ultralytics.py` — backend de PC con el contrato de
  `hailo_inference.HailoInference`. Devuelve la salida en el formato "NMS en
  el chip" (formato 3 de `postprocess.py`).
- `fuente.py` — misma idea para la entrada: `FuenteCamara` (Picamera2) o
  `FuenteCarpeta` (imagenes), via `abrir_fuente()`. Reloj sintetico
  (reproducible, para replay) o de pared (para debug manual).
- `test_inferencia_ultralytics.py` — pasa.

---

## P2 — parte 1: `preproceso.py` CERRADA

**Archivos creados:** `preproceso.py`, `test_preproceso.py`
**Archivos tocados:** `config.py` (ver abajo)

Mudanza de las funciones de armado de tensor que vivian en `main.py`, que se
reemplaza. Se copiaron `_repartir`, `search_tiles`, `tile_para_punto`,
`build_search_input` y `get_model_hw` sin cambiar la logica.

### Cambios deliberados respecto de `main.py`

1. **`build_track_input` cambio de firma.** Era `(frame, state, model_hw)` y
   llamaba a `state.crop_center()`, lo que ataba el modulo a
   `state_machine.TrackerState`. Ahora es `(frame, centro_xy, model_hw)` con
   una tupla. Sin esto `preproceso.py` dependia de un modulo condenado y no
   se podia testear solo (regla 4).
2. **No se copio `build_search_full`.** Se descarto junto con el modelo
   entrenado en downscale.
3. **No se copiaron `modelo_hw()` ni `inferir()`**, que existian solo para
   despachar entre dos modelos.
4. **La conversion de canales salio a `config.FUENTE_ENTREGA_BGR`.** Antes
   habia dos `cvtColor(BGR2RGB)` clavados en funciones distintas.

### Numeros medidos

Grilla real con frame nativo 2304x1296 y modelo 1152x640:

```
TRACK   recorte de 1152x640 centrado, escala 1.000 (nativo), 1 inferencia
SEARCH  grilla 2x2 de 1166x648, escala 0.9880, 4 inferencias
        solape  x=28 px  y=0 px

tiles: 0:(0,0,1166,648)     1:(1138,0,1166,648)
       2:(0,648,1166,648)   3:(1138,648,1166,648)
```

Confirma la seccion 4.1 del briefing: las dos escalas son practicamente 1.0,
que es por lo que un solo modelo entrenado en recortes nativos sirve para los
dos modos.

### Verificacion

`python3 test_preproceso.py` — 10/10 en verde. Lo que cubre:

- grilla y tensores **identicos byte a byte** a los de `main.py` en los 4
  tiles: la mudanza no cambio comportamiento;
- BGR rojo puro entra, `(255,0,0)` sale: la conversion a RGB es correcta;
- shapes exactos `(640,1152,3)` uint8 en los dos modos;
- `to_global` cierra el circulo en TRACK (centro del tensor -> centro pedido)
  y en los 4 tiles de SEARCH;
- round-trip punto -> `tile_para_punto` -> `build_search_input` -> `to_global`
  vuelve al mismo punto, en 4 puntos repartidos;
- clampeo en la esquina sin cambiar el shape;
- `tile_idx` ciclico (un contador que crece sirve de barrido);
- `build_search_full` ausente.

### Comando

```
python3 preproceso.py          # describe la grilla, no necesita modelo ni cv2
python3 test_preproceso.py     # los 10 tests
```

---

### LINEA DE BASE DE TRACKING (la que compara P3)

Misma carpeta `Soleado` (315 imagenes, secuencia continua), `replay.py` SIN
`--solo-search`, o sea con la maquina de estados de `state_machine.py`:
histeresis de confianza a secas, sin Kalman y sin gate de plausibilidad.

| metrica | valor |
|---|---|
| **tiempo en TRACK** | **267/315 (84.8%)** |
| **con deteccion aceptada** | **251/315 (79.7%)** |
| deteccion estando en TRACK | 243/267 (91.0%) |
| deteccion estando en SEARCH | 8/48 (16.7%) |
| transiciones SEARCH->TRACK | 4 |
| **transiciones TRACK->SEARCH** | **3** |
| confianza p50 / p95 / min | 0.810 / 0.929 / 0.134 |
| latencia p50 / p95 (PC) | 12.6 / 16.8 ms |

**Los tres numeros que P3 tiene que mejorar: 84.8%, 79.7% y 3 caidas.**

Que dice la comparacion con la corrida `--solo-search` de la misma carpeta:

- **El tracking pierde contra la deteccion pura: 79.7% contra 85.1%.** El ROI
  de TRACK se centra en la ultima posicion conocida sin extrapolar velocidad,
  asi que a veces mira donde la pelota ya no esta y pierde detecciones que el
  barrido completo si encontraba. Eso es lo que ataca el Kalman.
- **Recuperarse cuesta ~16 frames.** 48 frames en SEARCH en 3 caidas. La
  deteccion en SEARCH es 16.7%, o sea aproximadamente 1 de cada 4: coherente
  con barrer UN cuadrante por frame. El patron de P3 (los 4 cuadrantes por
  barrido, arrancando por `tile_para_punto`) ataca esto.
- **TRACK cuesta 12.6 ms y SEARCH 63 ms** en la misma maquina. Una inferencia
  contra cuatro. Ese es el argumento entero de por que TRACK existe.

---

## LINEA DE BASE MEDIDA (2026-09-06)

Primera corrida de `replay.py --solo-search` con el .pt real, sobre las tres
carpetas de `imagenes/`. Recall puro: 4 cuadrantes por imagen, sin maquina de
estados, mismo tratamiento para las tres.

| carpeta | imagenes | aceptadas | conf p50 | conf p95 | conf min | cuadrante ganador | rango angulo |
|---|---|---|---|---|---|---|---|
| Soleado | 315 | 268 (85.1%) | 0.797 | 0.927 | 0.107 | t0:7 t1:41 t2:119 t3:101 | 40.3 - 104.8 |
| Sombra | 50 | 50 (100%) | 0.847 | 0.926 | 0.161 | t0:8 t2:32 t3:10 | 28.7 - 60.8 |
| Sombra_Soleado | 50 | 50 (100%) | 0.849 | 0.939 | 0.473 | t0:1 t1:44 t2:3 t3:2 | 41.6 - 86.9 |

Latencia en la PC con Ultralytics: p50 ~59-63 ms por imagen, o sea ~15 ms por
inferencia (son 4). **No dice nada del presupuesto de 25 ms de P0**, que se
mide en la Pi con el .hef.

### Como leer esto

- **La confianza p50 de 0.80-0.85 es buena** y esta cerca del 0.879 medido a
  resolucion nativa. La luz no la mueve mucho: 0.797 / 0.847 / 0.849.
- **El 85.1% es un TECHO, no el recall.** `CONF_MIN_DETECTION = 0.1` y la
  confianza minima medida es 0.107: cualquier cosa por encima de ese piso
  cuenta como aceptada, incluidos falsos positivos. Falta revisar la cola baja
  contra el mp4 anotado.
- **Los dos 100% son de sets chicos y poco representativos.** 50 imagenes, y
  en Sombra_Soleado 44 de 50 detecciones caen en el MISMO cuadrante con 45
  grados de rango angular: es la pelota casi en el mismo lugar. Soleado (315
  imagenes, los 4 cuadrantes usados, 64 grados de rango) es el unico set con
  peso estadistico.
- **La columna `angulo` todavia no significa nada**: sale de
  `CAM_CENTRO_ANGULO = {0: 55.0}`, que esta inventado y se mide en P8.

### Lo que falta para que esta sea la linea de base de P3

Esta corrida mide DETECCION, no TRACKING. El criterio de P3 es que suba el
"% de tiempo en TRACK", y ese numero solo sale de `replay.py` SIN
`--solo-search` sobre una **secuencia continua** (video o rafaga). Con
imagenes sueltas no se puede medir. Falta esa corrida.

---

## P2 — parte 2: `replay.py` CERRADA (falta la corrida real)

**Archivos creados:** `replay.py`

Banco de pruebas offline: lee un .mp4 o una carpeta, corre deteccion, y
escribe CSV + mp4 anotado + resumen. Sin motor, sin encoder, sin GoPro.

### Decisiones

1. **`FuenteVideo` vive en `replay.py`, no en `fuente.py`.** `fuente.py`
   modela las dos fuentes del sistema REAL (camara y carpeta). Un archivo de
   video es una comodidad del banco de pruebas, no una entrada del producto.
   Con `--carpeta` se usa `fuente.FuenteCarpeta` tal como esta.
2. **Reloj sintetico en los dos caminos.** El `ts` avanza 1/fps por frame, no
   con el reloj de pared. Sin esto, comparar dos versiones del algoritmo mide
   la maquina y no el algoritmo.
3. **Un cuadrante por frame en SEARCH**, rotando, igual que hacia `main.py`.
   Es la linea de base honesta. P3 lo cambia a los 4 cuadrantes por barrido.
4. **Sin `gpio_timer`.** El pin solo sirve midiendo en la Pi con osciloscopio;
   en la PC llenaria la salida de prints por frame.
5. **`procesar_frame()` recibe el `info` entero** de la fuente aunque en P2
   solo use `ts_ns`: cuando entre el Kalman va a necesitar el `dt` real y la
   firma no tiene que cambiar. Es la funcion que despues comparte
   `main_final.py`.

### Columnas del CSV (regla 8: identicas a las de main_final.py)

```
n, ts, modo, camara, tile, conf, x, y, w, h, angulo, sector, flost, aceptado_por
```

`sector` va vacia hasta P4. `aceptado_por` es "confianza" o vacio hasta P3.

### Verificacion

Corrido de punta a punta con un backend stub (deteccion por umbral, sin .pt
ni ultralytics) sobre videos sinteticos de 2304x1296:

- pelota cruzando el frame: 120 frames, 100% con deteccion, 99.2% en TRACK,
  1 transicion SEARCH->TRACK. El CSV tiene 120 filas y sus columnas coinciden
  exactamente con `replay.COLUMNAS`.
- pelota que desaparece 60 frames: 150 frames, 61.3% en TRACK, 38.7% en
  SEARCH, 2 transiciones a TRACK y 1 a SEARCH, `flost` llega a 4 y los 4
  tiles rotan. O sea: el camino de perdida y recuperacion se ejercita.
- camino de carpeta con `fuente.FuenteCarpeta` y `--camara 1`: 30 imagenes,
  reloj sintetico, ok.
- `ts` sintetico correcto: 120 frames a 40 fps -> 0.0 a 2.975 s.

**PENDIENTE: correr sobre material REAL con el .pt.** Los numeros de arriba
salen de un stub geometrico y no dicen nada del modelo. La linea de base de
P3 es la corrida real, y todavia no esta.

### Comando

Ver `README.md`. Los flags `--solo-search`, `--sin-kalman` y `--sin-gate`
estan explicados ahi, con cuando usar cada uno.

---

## Cambios en config

`config.py`:

- **`FUENTE_ENTREGA_BGR = True`** (nueva). Orden de canales que entrega la
  fuente. True = BGR, que es lo que devuelven `cv2.imread()` y Picamera2 con
  formato `"RGB888"`. La usa `preproceso._a_rgb()`.
- Se limpio el comentario `# alto, ancho. Hoy dice (640, 640)` de
  `MODEL_INPUT_SIZE`, que contradecia a la propia linea.

---

## PENDIENTES

### Bloqueantes

- **Orden de canales sin verificar (P1).** `FUENTE_ENTREGA_BGR = True` sale de
  lo que documenta `fuente.py`, NO de una medicion.
  `CameraSource.verificar_canales()` todavia no se corrio contra la camara
  real. Si dice que el array ya viene en RGB, se pone `False` en `config.py` y
  no se toca ningun `cvtColor`. Costo de equivocarse: factor 8 de confianza en
  el caso peor (0.476 contra 0.061 en frame reducido), invisible a ojo.
- **Latencia sin medir (P0).** Todo asume que TRACK entra en 25 ms. No hay
  `.hef` ni Pi disponible. Los tiempos de Ultralytics en PC **no sirven** para
  esto.
- **Las dos camaras a 40 fps simultaneas (P1).** No probado.

### Menores

- `main.py` sigue importando `state_machine` y su `abrir_inferencia()` intenta
  `from dual_inference import DualHailoInference`, que no existe. Se resuelve
  cuando `main.py` se reemplace.
- `inferencia_ultralytics.py` todavia importa `main` en su CLI para
  `search_tiles` / `build_search_input` / `build_search_full`. **Ya se puede
  apuntar a `preproceso`**, salvo `build_search_full`, que hay que sacar junto
  con la opcion `--full`.
- `config.MODELO_PT` dice `run_yolo26n_sesiones_1152x640px_300ep.pt`, que no
  es ninguno de los dos `.pt` que se discutieron (CROP / DOWNSCALE, 200ep).
  **Confirmar cual es el modelo bueno.**
- `config.py` conserva `N_ZONAS = 9` sobre 20..160 (zonas de 15.6 grados) y
  `MS_PERMANENCIA = 1500.0`. El briefing pide **7 sectores de 20 grados** y
  `MS_PERMANENCIA` de arranque 600 ms. Se resuelve en P4.
- `encoder_lib.poner_cero_aca()` escribe el registro ZPOS y esta expuesta en
  el CLI interactivo. Invalida `ENCODER_GRADOS_EN_MOTOR_CERO = 79.0`. El
  briefing dice que no se use. Archivo congelado: **no se toco**, pendiente de
  confirmacion.
- `CAM_CENTRO_ANGULO = {0: 55.0, 1: 125.0}` siguen **inventados**. Se miden en
  P8.

---

## P3 — `tracker.py` CERRADO

**Archivos creados:** `tracker.py`, `test_tracker.py`
**Tocados:** `config.py` (constantes nuevas), `replay.py` (usa el tracker)

### Piezas

- `KalmanCV` — velocidad constante en (x, y, vx, vy), px del frame nativo, dt
  REAL de `ts_ns`. Escrito a mano con numpy: son 4 estados y 2 mediciones,
  filterpy seria una dependencia mas para 16 lineas de algebra.
- `radio_gate(det) = max(60, 6*max(w,h))` — independiente de la distancia.
- `PatronBusqueda` — 3 barridos completos en la camara activa, 1 en la otra,
  arrancando por el cuadrante donde se la vio (`fn_tile` inyectada).
- `Tracker.actualizar(candidatos, ts_ns, camara) -> ResultadoFrame`.
- `Mode` compatible con el de `state_machine.py` (mismos nombres y valores):
  lo verifica un test, porque `gpio_timer.py` compara contra `Mode.TRACK`.

### Columnas nuevas del CSV: pred_x, pred_y, err_pred, vx, vy

`err_pred` es el RESIDUO del Kalman: la distancia entre lo que el filtro
predijo antes de ver el frame y donde aparecio la deteccion aceptada. Es la
señal con la que se tunean `KALMAN_SIGMA_ACEL` y `KALMAN_SIGMA_MEDICION`:

- residuo grande y sistematico -> el modelo de movimiento no alcanza, o
  `KALMAN_SIGMA_MEDICION` esta muy chico (el filtro le cree de mas a su propia
  prediccion y reacciona tarde a los cambios de direccion)
- residuo chico pero la trayectoria tiembla -> `KALMAN_SIGMA_MEDICION` muy
  grande: el filtro esta copiando el ruido de la deteccion

El resumen de `replay.py` ahora imprime p50/p95/max del residuo y el reparto
de `aceptado_por` (confianza / kalman), que dice cuanto trabajo esta haciendo
el filtro de verdad.

### Constantes agregadas a config.py

`CONF_ALTA = 0.50`, `CONF_BAJA = 0.10` (los del briefing; los que estaban,
0.60/0.25, se borraron del bloque de zonas), `FLOST_A_SEARCH = 10`,
`GATE_DIAMETROS = 6.0`, `GATE_PX_MIN = 60.0`, `KALMAN_SIGMA_ACEL = 2000.0`,
`KALMAN_SIGMA_MEDICION = 8.0`, `SEARCH_BARRIDOS_ACTIVA = 3`,
`SEARCH_BARRIDOS_OTRA = 1`. Los dos KALMAN_ son **tentativos**.

### Verificacion

`python3 test_tracker.py` — 19/19. Cubre: Mode compatible con gpio_timer,
gate con piso y escalado, deteccion buena continua (y el filtro aprende la
velocidad), baja confianza cerca (se acepta por "kalman"), baja confianza
lejos (se rechaza), **el mas cercano gana al mas confiado** (zapatilla 0.35
lejos contra pelota 0.15 cerca), salto imposible con confianza 0.95, 10
frames perdidos, no volver antes de tiempo, ROI congelado en la ultima
ACEPTADA, vuelta a TRACK, orden del patron de SEARCH, dt real, handover con y
sin semilla.

### Resultado contra la linea de base (imagenes/Soleado, 315 frames)

| metrica | base (sin Kalman) | P3 | |
|---|---|---|---|
| tiempo en TRACK | 84.8% | **92.4%** | mejor |
| deteccion aceptada | 79.7% | **81.6%** | mejor |
| caidas TRACK->SEARCH | 3 | **1** | mejor |
| deteccion estando en TRACK | 91.0% | 87.6% | ver nota |

Reparto de motivos: confianza 233 (74.0%), sin candidato 53 (16.8%),
**kalman 24 (7.6%)**, delta imposible 4 (1.3%), fuera del gate 1 (0.3%).

- **El Kalman rescata el 7.6% de los frames**: detecciones flojas que la
  version sin filtro tiraba. Aporte modesto porque el modelo detecta bien
  (p50 0.807); en la cancha, con la pelota a 20 m y mas borrosa, deberia pesar
  mas.
- **Los 4 `delta imposible` que quedan son el gate trabajando de verdad**:
  saltos que no se explican por movimiento de la pelota, o sea falsos
  positivos rechazados.
- La deteccion estando en TRACK baja de 91.0% a 87.6%, pero el sistema pasa
  mucho mas TIEMPO en TRACK (92.4% contra 84.8%), asi que en numero absoluto
  detecta mas: 255 frames contra 243.

### BUG ENCONTRADO Y CORREGIDO: bloqueo del gate

La primera version comparaba el gate contra la ultima posicion aceptada SIN
tener en cuenta su antiguedad. Como al rechazar esa posicion se congela y la
pelota sigue viaje, cada rechazo hacia mas probable el siguiente:

    frame  63    64    65    66    67    68    69
    dist  123   216   300   377   453   516   571 px   (gate = 120)

Un rechazo marginal (123 px contra un gate de 120: 3 px de diferencia) se
convertia en 10 frames perdidos garantizados, hasta que `FLOST_A_SEARCH`
forzaba la vuelta a SEARCH. Eran 44 frames, el 14% del total.

La correccion es `radio_gate_acumulado(det, frames) = radio_gate(det) *
(1 + flost)`: si pasaron k frames desde la ultima aceptada, la pelota pudo
moverse k veces la distancia de un frame. Misma fisica, aplicada al tiempo que
realmente paso. Lo cubren dos tests de regresion.

### Desplazamiento real medido

Con el bug corregido, las distancias ya no se miden contra un punto congelado:

    desplazamiento entre frames aceptados   p50 13 px   p95 102 px   max 238 px

Con el gate en 120 px (caja de ~20 px), el p95 entra holgado: **el gate esta
bien calibrado para este material.**

### BUG ENCONTRADO Y CORREGIDO (2): el gate estaba clavado a 40 fps

`radio_gate()` devolvia pixeles por FRAME, asumiendo el periodo de
`config.CAM_FPS`. Pero el gate es fisica por unidad de TIEMPO: si entre dos
imagenes pasan 250 ms en vez de 25, la pelota puede moverse diez veces mas.

Lo destapo `imagenes/test_kalman`, capturada mas espaciada que `Soleado`:

| variante | aceptada | delta imposible |
|---|---|---|
| sin kalman y sin gate | 88.0% | - |
| completo, gate clavado a 40 fps | 41.7% | 73 (38.0%) |
| completo, gate temporal + `--fps 4` | **87.5%** | **0** |

La correccion: `radio_gate(det, segundos)` escala con el tiempo REAL desde la
ultima deteccion aceptada, con piso en el periodo nominal. Dos tests de
regresion. Consecuencia practica: **con material que no es de 40 fps hay que
pasarle `--fps`**, o el gate rechaza detecciones buenas en masa.

---

## RESULTADO: el Kalman necesita frecuencia de muestreo alta

Comparando las dos corridas, el residuo del filtro sale asi:

| material | desplazamiento p50 | residuo Kalman p50 | rescates por kalman |
|---|---|---|---|
| `Soleado` (~40 fps) | 13 px | **17 px** | 24 (7.6%) |
| `test_kalman` (~4 fps) | 143 px | **391 px** | 12 (6.2%) |

En `test_kalman` el filtro predice PEOR que no predecir: si dijera "la pelota
sigue donde estaba" su error mediano seria 143 px, y predice con 391. Divide
143 px entre 250 ms, concluye ~570 px/s, extrapola, y se va lejos: entre
imagen e imagen la pelota ya cambio de direccion varias veces. **El intervalo
de muestreo es mas largo que el tiempo en que la trayectoria se mantiene
recta**, asi que la velocidad medida no predice nada.

No es un bug: es el resultado del experimento, y es material para la tesis.
El Kalman sirve a 40 fps y es contraproducente a 4.

Vale notar que **aun con el filtro prediciendo mal, el resultado global no se
degrado** (87.5% contra 88.0% sin filtro): la prediccion solo desempata
candidatos flojos, asi que el diseño tolera un filtro malo sin romperse.

### Que material vale para que

- **`imagenes/Soleado`** — material de REFERENCIA de P3. Es el que se parece a
  la cancha (continuo, ~40 fps, 13 px entre frames).
- **`imagenes/test_kalman`** — caso de ESTRES. Util para verificar que el
  sistema no se rompe con muestreo grueso, NO para predecir el comportamiento
  en cancha ni para tunear el filtro.
- **`Sombra` y `Sombra_Soleado`** — 50 imagenes cada una, sueltas. Solo con
  `--solo-search`, y solo para comparar recall entre condiciones de luz.

---

### Lo que queda abierto: el residuo del Kalman

    residuo   p50 17.2 px   p95 129.1 px   max 471.1 px

El p50 esta bien: el filtro predice bien la mayoria del tiempo. Los picos son
lo esperable de un modelo de velocidad constante cuando la pelota rebota o la
patean: el cambio de velocidad no se ve venir.

**Antes de tocar `KALMAN_SIGMA_ACEL`, verificar si esos picos coinciden con
cambios de direccion.** Si es asi, no hay nada que tunear: es la limitacion
del modelo y esta aceptada. Tunear contra un pico que el modelo no puede
predecir solo empeora el resto.

---

## P4 — `sectores.py` CERRADO

**Archivos creados:** `sectores.py`, `test_sectores.py`
**Tocados:** `config_hw.py` (constantes nuevas), `config.py` (se saco el bloque
viejo de 9 zonas)

**9 sectores de 20 grados sobre 0..180**, o sea el semiplano completo.
Bordes [0,20,...,180], centros [10,30,50,70,90,110,130,150,170]. **El motor va
SIEMPRE al centro del sector**, nunca al angulo exacto de la pelota: hay un
test que lo fija.

### Cambios respecto del briefing (pedidos y probados)

1. **9 sectores sobre 0..180, no 7 sobre 20..160.** El diseño original perdia
   los dos extremos, y con las camaras cubriendo 0..102 y 78..180 esos
   extremos SI se ven: una jugada contra el lateral cae ahi. Verificado que el
   motor los alcanza: el centro mas extremo (10 o 170 del mundo) pide 80
   grados de motor contra un limite mecanico de 95, o sea 15 de margen.
2. **`MS_PERMANENCIA = 0`: la permanencia se desactivo.** La espera era
   contraproducente justo en el pelotazo, que es el caso que importa, y la
   excepcion por `OMEGA_RAPIDA` solo lo cubria si la estimacion de velocidad
   angular (ruidosa) superaba el umbral. **Los tres criterios del briefing se
   siguen cumpliendo sin permanencia**: la histeresis sola alcanza. El codigo
   quedo, con 0 se saltea; para reactivarla se cambia la constante y los tests
   correspondientes se auto-activan.
3. **Cobertura de camaras actualizada** a 102 grados de FOV con centros en 51
   y 129 (antes 100 con 55 y 125, inventados). Sigue siendo ESTIMADO: se mide
   en P8.

### Los tres frenos

1. **Margen** `HISTERESIS_SECTOR_DEG = 5` — el sector actual se ensancha 5
   grados a cada lado.
2. **Permanencia** `MS_PERMANENCIA = 600 ms` votando al MISMO sector.
3. **Piso** `MS_MINIMO_ENTRE_MOVIMIENTOS = 800 ms` entre movimientos.

Excepcion por regimen: con `|omega| >= OMEGA_RAPIDA` se saltea la permanencia
(no el margen: el ruido de la deteccion no desaparece porque la pelota vaya
rapido). Hay un test para eso.

### Verificacion — los tres criterios del briefing

`python3 test_sectores.py` — 14/14.

| criterio | pedido | medido |
|---|---|---|
| rampa lenta 0->180 | un cambio por borde interno | **8** (sectores 1..8) |
| senoidal +-4 sobre un borde, 30 s | **cero** cambios | **0** |
| salto de 60 con omega alta | cambio en < 100 ms | **< 100 ms** |

Mas: el semiplano completo cubierto (ningun angulo de 0 a 180 sin sector), los
frenos por separado, un control de la senoidal (con +-12 grados SI cambia,
para que el test de los cero cambios no pase por estar todo roto), la
cobertura por camara, y que los bordes en pixeles crezcan con el angulo (si
falla, hay una camara espejada y `CAM_ESPEJO` no lo refleja).

### Ver los sectores sobre la imagen

```
python replay.py --carpeta imagenes/Soleado --camara 0 --lineas-sectores --salida-mp4 salidas/cam0.mp4
```

Amarillo los bordes con su angulo, cian los centros (adonde va el motor).
Cobertura calculada con el modelo nominal:

    camara 0:   0..102 grados   sectores [0, 1, 2, 3, 4, 5]
    camara 1:  78..180 grados   sectores [3, 4, 5, 6, 7, 8]
    en las DOS:                 sectores [3, 4, 5]

**Esas lineas son una HIPOTESIS, no una verdad**: salen del FOV estimado y del
modelo de lente lineal, que ignora la distorsion de barril. Sirven para poner
la pelota en un punto conocido y ver cuanto hay que corregir; esa discrepancia
es lo que P8 mide bien.

### Costo computacional de pixel -> angulo -> sector

Medido con la tabla de calibracion de 7 puntos cargada:

    pixel_a_angulo    2.2 us
    + sectorizador    3.5 us

Contra los ~13 ms de una inferencia, es el 0.03% del presupuesto del frame.
**No es una preocupacion de latencia.**

### Constantes: se mudaron a config_hw.py

`SECTOR_DESDE/HASTA`, `N_SECTORES = 7`, `HISTERESIS_SECTOR_DEG = 5.0`,
`MS_PERMANENCIA = 600.0`, `MS_MINIMO_ENTRE_MOVIMIENTOS = 800.0`,
`OMEGA_RAPIDA/LENTA`, `S_SEARCH_A_CENTRO = 25.0`.

Se BORRO de `config.py` el bloque de 9 zonas de 15.6 grados con permanencia de
1500 ms, que era del diseño viejo y contradecia a este. Queda un comentario
apuntando a config_hw.

`HISTERESIS_SECTOR_DEG`, `MS_PERMANENCIA` y `S_SEARCH_A_CENTRO` son
**tentativos**: se ajustan mirando el video de cancha. Si el motor se ve
nervioso, subir; si llega tarde, bajar.

### Comando

```
python3 sectores.py          # describe la particion y el sector de cada angulo
python3 test_sectores.py
```

---

## P5 — `init_sistema.py` ESCRITO (falta correrlo en la Pi)

**Archivos creados:** `init_sistema.py`, `test_init_sistema.py`

`inicializar(con_motor=True, con_gopro=True, simular=False) -> Sistema`, mas
un context manager `arranque()` que cierra pase lo que pase.

### Secuencia

1. Encoder: `aplicar_config()` (soft write, se pierde al cortar la
   alimentacion), verificar PPR contra `ENC_PPR_ESPERADO`, diagnostico del
   iman, leer el angulo absoluto X.
2. Motor: verificar que la ESP32 responde, deshabilitar, corriente,
   micropasos, velocidad, aceleracion, habilitar.
3. Homing: mover `HOMING_SENTIDO * (X - ENCODER_GRADOS_EN_MOTOR_CERO) *
   RELACION_TRANSMISION`, verificar contra el encoder, `motor.zero()`, guardar
   el offset EN SOFTWARE.
4. Ir a **180** del mundo (+90 de motor) y verificar.
5. Ir a **0** del mundo (-90 de motor) y verificar.
6. Volver a 90 y quedarse ahi.

### Cambios respecto del briefing

- **Los extremos son 0 y 180, no 180 y 90** (confirmado: el eje gira libre,
  no hay tope mecanico). Se verifican los dos porque un error de ESCALA
  (relacion de transmision o micropasos mal) no se ve en el homing, que es un
  movimiento corto, y si en uno de 90 grados. El mensaje de error compara los
  signos de los dos: mismo signo = offset, signos opuestos = escala.
- Se agrego una cuarta verificacion en 90 (el reposo).

### Verificacion

`python3 test_init_sistema.py` — 11/11 con `hw_falsos`.

Prueba el FLUJO y sobre todo que **aborte con un mensaje accionable** ante
cada modo de fallo: PPR distinto del esperado, iman debil o muy cerca, ESP32
muda, homing fuera de tolerancia, timeout de movimiento. Los tests exigen que
el mensaje mencione que revisar (`HOMING_SENTIDO`, "cable de solo carga",
`AS5047D`), no solo que falle.

Tambien verifica que **no se escriba el registro ZPOS** del encoder.

**NO prueba la mecanica**: el motor falso mueve exacto y el encoder falso
repite lo que dice el motor, asi que en el caso feliz todos los errores dan
0.00 y eso no significa nada.

### PENDIENTE: correrlo en la Pi

```
python3 init_sistema.py --sin-gopro
```

Criterio: los errores por debajo de `TOLERANCIA_HOMING_DEG = 1.5`. **Anotar
aca los cuatro numeros medidos.**

Nota: `MOTOR_GRADOS_MIN/MAX = +-95` es un limite por SOFTWARE. Con 0 y 180
pidiendo -/+90 entra, pero con 5 grados de margen. Si el eje gira libre,
conviene ampliarlo.

---

## Proximo paso

**Correr `init_sistema.py` en la Pi** y anotar los cuatro errores. Es lo unico
que valida P5, y bloquea P7.

Despues, **P6: `control_motor.py`** — el hilo no bloqueante. `motor_lib.
esperar_fin()` hace polling serie cada 50 ms con timeout de 12 s: si eso vive
en el loop principal, el sistema deja de ver la pelota justo cuando el motor
se mueve. Va en un `threading.Thread` daemon con una `queue.Queue(maxsize=1)`
que se pisa. Criterio: `apuntar()` llamado a 20 Hz, **ninguna llamada por
encima de 1 ms**. Se puede probar sin fierro con `hw_falsos.Motor`, al que hay
que completarle `mover_pasos`, `ir_a_pasos`, `info_movimiento` y
`en_movimiento`.

**Ojo con `omega`**: el tracker entrega velocidad en px/s (`vx`, `vy`). La
conversion a grados por segundo del mundo la hace quien llama, con
`geometria.pixel_a_angulo` sobre dos posiciones consecutivas. Hasta P8 los
angulos salen de `CAM_CENTRO_ANGULO` inventado, asi que la columna `sector`
del CSV no va a significar nada real hasta calibrar.

## P0 — presupuesto de tiempo (2026-09-07 15:36)

- hef: `yolo26n_sincalib_opt1.hef`
- fuente: `carpeta imagenes/cam0`  (300 frames medidos, 10 de calentamiento descartados)
- entrada del modelo: 300 muestras por etapa

| etapa | p50 (ms) | p95 (ms) |
|---|---|---|
| pre_track | 1.04 | 3.87 |
| infer_track | 15.86 | 16.06 |
| post_track | 0.23 | 0.26 |
| pre_search | 4.85 | 6.94 |
| infer_search | 15.70 | 15.97 |
| post_search | 0.18 | 0.24 |
| total_track | 17.14 | 19.95 |
| total_search | 83.02 | 92.88 |

**Resultado:** TRACK p95 = 19.95 ms contra un presupuesto de 25.0 ms -> CUMPLE. Techo sostenible 50.1 fps.

## P1 — dos cámaras simultáneas (2026-09-07 16:02)

- pedido: 2304x1296 @ 40 fps

| pasada | fps (timestamps) | dt p50 | dt p95 | perdidos | pisados |
|---|---|---|---|---|---|
| cam0 sola | 40.01 | 24.99 | 25.00 | 0 (0.00%) | 0 |
| cam1 sola | 40.01 | 24.99 | 25.00 | 0 (0.00%) | 0 |
| cam0 + cam1 -> cam0 | 40.01 | 24.99 | 24.99 | 0 (0.00%) | 1825 |
| cam0 + cam1 -> cam1 | 40.01 | 24.99 | 25.00 | 0 (0.00%) | 1744 |

**Resultado:** CUMPLE el criterio de menos de 2% de frames perdidos.

**Orden de canales (4.1):** cam0 entrega `BGR`, cam1 entrega `BGR`. Verificado con `verificar_canales()` contra un objeto rojo.
