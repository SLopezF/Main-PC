# Briefing y plan de construcción — sistema de tracking de pelota

Documento único de trabajo. Se le entrega a un agente junto con todo el
código del proyecto. Contiene el contexto, las decisiones ya tomadas, las
reglas de trabajo y los pasos en orden.

---

## 1. Qué es esto

Tesis de ingeniería. Sistema de grabación automática de partidos de fútbol 5
para una cancha fija.

**El problema.** Grabar un partido de fútbol 5 con una cámara fija de gran
angular da un video donde la pelota son diez píxeles y no se entiende nada.
Grabarlo con un camarógrafo cuesta plata y no escala a una cancha que alquila
turnos todo el día.

**La solución.** Un poste al medio del lateral de la cancha, a 3 metros de
altura, con:

- Dos cámaras Arducam IMX708 fijas, separadas 20 cm entre sí, apuntando cada
  una a una mitad de la cancha, con 102° de FOV cada una y solapamiento en el
  centro. Son los *ojos*: resolución 2304x1296, 40 fps, nunca se mueven, no
  graban el video final.
- Una Raspberry Pi 5 con un acelerador Hailo-8 de 26 TOPS. Es el *cerebro*:
  corre un YOLO26n custom de una sola clase (pelota) sobre los frames de las
  cámaras y devuelve dónde está la pelota.
- Una GoPro Hero 7 montada sobre un motor paso a paso que solo hace *pan*
  (giro horizontal). Es la *cámara de verdad*: graba el partido entero en 4K
  en un solo archivo de una hora, y el sistema la va apuntando hacia donde
  está la jugada.
- Un motor NEMA con driver TMC2209 comandado por una ESP32 por USB, y un
  encoder absoluto AS5047D por SPI que verifica que el motor efectivamente
  fue a donde se le dijo.

**El objetivo.** Que el video de la GoPro se parezca a una transmisión
profesional: la pelota siempre en cuadro, y la cámara quieta la mayor parte
del tiempo, con movimientos poco frecuentes y bien justificados. Un video que
sigue la pelota permanentemente marea y se ve amateur.

**La consecuencia de diseño más importante.** El sistema no tiene que apuntar
*a la pelota*, tiene que apuntar *a la zona donde está la jugada*. Por eso el
ángulo continuo se cuantiza a 7 sectores y el motor solo salta entre sectores
cuando hay evidencia sostenida de que la jugada se mudó. Con un FOV de más de
100° en la GoPro, un sector de 20° de error no saca la pelota del cuadro.

**Estado actual.** Todos los drivers de hardware están escritos, probados y
funcionando: encoder, motor, GoPro, cámaras e inferencia en la Hailo. El
modelo custom detecta bien. Lo que falta es la lógica de decisión y el
programa que ata todo. Hoy existe `main_partido.py`, que hace un ciclo por
cada ENTER que se aprieta: foto, inferencia, ángulo, mover motor. Sirvió para
validar el hardware; no es el sistema.

---

## 2. Inventario de archivos

### Congelados — NO SE TOCAN

Estos están probados contra el hardware real. Si algo parece un bug acá, se
avisa y se espera confirmación; no se edita.

| Archivo | Qué hace |
|---|---|
| `encoder_lib.py` | AS5047D por SPI: ángulo absoluto, config, diagnóstico del imán |
| `motor_lib.py` | ESP32 + TMC2209 por USB: movimiento en grados y en pasos |
| `gopro_lib.py` | GoPro por Wi-Fi: init, foto, grabar, descargar |
| `hailo_inference.py` | Wrapper síncrono de HailoRT sobre el .hef |
| `camera_source.py` | Picamera2 con último-frame-gana, timestamps y frames perdidos |
| `postprocess.py` | Decodifica las 6 salidas del .hef a detecciones |

### Se editan

| Archivo | Qué se le hace |
|---|---|
| `config.py` | Se agregan/ajustan constantes de detección y tracking |
| `config_hw.py` | Se agregan constantes de sectores, se corrigen los ángulos inventados |
| `geometria.py` | Ya tiene lo necesario; se revisa el `SelectorCamara` |
| `hw_falsos.py` | Se completan los métodos que falten en el motor falso |

### Se reemplazan

| Archivo | Por qué |
|---|---|
| `main.py` | Loop viejo sobre archivo de video. Sus funciones de preprocesado (`search_tiles`, `build_search_input`, `build_track_input`, `tile_para_punto`) se rescatan a `preproceso.py`; el resto se descarta |
| `state_machine.py` | La histéresis simple de confianza no alcanza. Lo reemplaza `tracker.py`. Cuidado: `gpio_timer.py` importa `Mode` de acá, así que `tracker.py` tiene que exportar un `Mode` compatible |
| `main_partido.py` | Queda como herramienta de debug manual, no evoluciona al main final |

### Se crean

`bench.py`, `grabar_dataset.py`, `preproceso.py`, `replay.py`, `tracker.py`,
`sectores.py`, `init_sistema.py`, `control_motor.py`, `calibrar.py`,
`main_final.py`, y los tests correspondientes.

---

## 3. Reglas de trabajo para el agente

1. **Un paso por vez.** No se empieza el paso N+1 hasta que el N cumplió su
   criterio de aceptación y el humano lo confirmó. Nada de escribir seis
   archivos de una.
2. **Se mantiene `ESTADO.md`.** Al terminar cada paso el agente lo actualiza
   con: qué paso se cerró, qué archivos se crearon o tocaron, **qué números
   dio la medición**, y qué quedó pendiente. Ese archivo es la memoria entre
   sesiones: si el agente arranca de cero, lee `ESTADO.md` y este briefing y
   sabe dónde está parado.
3. **Nada de números mágicos.** Toda constante nueva va a `config.py` o
   `config_hw.py`, con un comentario que explique de dónde salió el número.
   Si el número es tentativo, el comentario lo dice.
4. **Todo módulo nuevo de lógica tiene que importarse sin hardware.**
   `tracker.py`, `sectores.py`, `preproceso.py` y `geometria.py` no pueden
   importar `picamera2`, `hailo_platform`, `spidev` ni `serial` a nivel de
   módulo. Se testean en cualquier máquina. Los imports de `cv2` van adentro
   de las funciones, como ya hace `main.py`.
5. **Cada paso termina con un comando que se puede correr.** No "esto debería
   andar": un `python3 archivo.py` que imprime el resultado del criterio de
   aceptación.
6. **Se preguntan las ambigüedades, no se inventan.** Si el briefing no dice
   algo, se pregunta antes de decidir. En particular: nunca inventar valores
   de calibración ni de FOV.
7. **Los mensajes y comentarios en español**, sin tildes obligatorias en el
   código pero sí en los docstrings, siguiendo el estilo del código que ya
   existe: explicar *por qué*, no *qué*.
8. **El CSV de `replay.py` y el de `main_final.py` tienen las mismas
   columnas.** El análisis de resultados de la tesis es un solo script para
   los dos.

---

## 4. Decisiones ya cerradas

No se rediscuten. Si un paso las contradice, el paso está mal.

### 4.1 Entrada al modelo

El `.hef` es de **1152x640** y el frame nativo es 2304x1296. Es casi
exactamente 2x2, y de ahí sale toda la estrategia:

| Modo | Entrada | Escala | Inferencias por frame |
|---|---|---|---|
| TRACK | recorte de 1152x640 centrado en la predicción | 1.000, nativo | 1 |
| SEARCH | cuadrante de 1152x648 reescalado | 0.988 | 4 |

`SEARCH_FULL = False`. El frame entero reescalado deja la pelota en 4 px y no
se usa nunca. `SEARCH_TILE_GRID = (2, 2)` es correcto.

TRACK tiene que correr a 40 fps con una sola inferencia. SEARCH corre a lo que
dé con cuatro, y no importa.

**Canales:** el modelo rinde bastante mejor en RGB (0.879 contra 0.859 en
crop nativo, 0.476 contra 0.061 en frame reducido). Pero `camera_source.py`
pide formato `"RGB888"` a Picamera2, cuyo nombre viene del empaquetado de
bytes y **suele entregar el array en orden BGR**, mientras que `main.py` hace
`cvtColor(..., COLOR_BGR2RGB)` asumiendo BGR. Hay que correr
`CameraSource.verificar_canales()` una sola vez, dejar el resultado escrito en
`ESTADO.md` y clavar la conversión correcta. Es un factor 8 de confianza en el
caso peor y es invisible si no se verifica.

### 4.2 Gate de plausibilidad

A 50 m/s (el techo físico de un pelotazo) y 40 fps, la pelota se desplaza
1.25 m por frame. Una pelota nro 4 mide 0.21 m de diámetro. El cociente es
5.95 y es **independiente de la distancia**, porque desplazamiento aparente y
diámetro aparente se escalan los dos con 1/d.

    radio_gate_px = max(60.0, 6.0 * max(det.w, det.h))

No hace falta calibrar nada para esto: ni FOV, ni distancia, ni altura. El
piso de 60 px cubre la pelota muy lejos, donde la caja mide 8 px y 6 diámetros
serían 48 px, menos que el ruido de la propia detección.

Caso límite conocido y aceptado: a menos de 3 m de la cámara, un pelotazo
puede salirse del ROI de 1152x640 en un solo frame. Se pierde el track un
momento y SEARCH lo recupera.

### 4.3 Kalman

Estado `(x, y, vx, vy)` en **píxeles del frame nativo de la cámara activa**,
modelo de velocidad constante, `dt` real tomado de `info["ts_ns"]` (no 1/fps
fijo: `camera_source.py` descarta frames y el dt es variable).

Se elige píxeles y no ángulos a propósito: en píxeles el tamaño de la caja da
información de distancia y la precisión no se degrada en los bordes del FOV.

En el handover de cámara el filtro se reinicia. El ROI de la cámara nueva se
siembra con `geometria.angulo_a_pixel(angulo_actual, ancho, camara_nueva)` y
la misma `y`. El ROI es un cuarto del frame, así que un error de decenas de
píxeles no cambia nada. Perder el filtro por un frame en cada handover es
aceptable y está decidido.

### 4.4 Sectores del motor

7 sectores sobre 20°..160°, de 20° cada uno. Centros en 30, 50, 70, 90, 110,
130, 150. **El motor va siempre al centro del sector, nunca al ángulo exacto
de la pelota.**

Schmitt trigger sobre los bordes: para pasar del sector `i` al `i+1`, el
ángulo tiene que superar el borde por `HISTERESIS_SECTOR_DEG` (arranque: 5°) y
sostenerlo `MS_PERMANENCIA` (arranque: 600 ms). Además tiene que haber pasado
`MS_MINIMO_ENTRE_MOVIMIENTOS` desde el último salto.

Excepción por régimen: si la velocidad angular supera `OMEGA_RAPIDA`, se
saltea la permanencia y se mueve ya. Un pelotazo cruza un sector de 20° en
unos 100 ms y esperar 600 ms lo perdería. La histéresis en grados se sigue
exigiendo igual.

El motor da la vuelta a la cancha en menos de medio segundo y es silencioso,
así que la velocidad del motor no es una restricción de diseño en ningún lado.

### 4.5 Geometría

La coordenada `y` **no participa** del cálculo del sector. El motor solo hace
pan y la GoPro tiene más de 100° de FOV desde 3 m de altura: una pelota en el
aire cerca y una en el piso lejos con el mismo ángulo van al mismo sector y
las dos entran en cuadro. La `y` sirve solo para centrar el ROI de TRACK.

El paralaje entre las dos cámaras (20 cm de separación) vale `11.5/d` grados:
2.3° a 5 m, 1.15° a 10 m, 0.6° a 20 m. Con sectores de 20° e histéresis de 5°
está por debajo del ruido. **Las dos cámaras se tratan como si estuvieran en
el mismo punto.**

Las dos cámaras tampoco están sincronizadas frame a frame; puede haber hasta
25 ms de delta entre ellas. Con el ROI de un cuarto de frame, no importa.

Los valores `CAM_CENTRO_ANGULO = {0: 55.0, 1: 125.0}` de `config_hw.py`
**están inventados** y hay que reemplazarlos por los medidos en P8.

### 4.6 Motor y encoder

El motor nunca bloquea el loop principal. `motor_lib.esperar_fin()` hace
polling serie cada 50 ms con timeout de 12 s: si eso vive en el loop, el
sistema deja de ver la pelota justo cuando el motor se mueve. Va en un hilo
con un buzón de un solo elemento.

En SEARCH el motor **no se mueve**. Recién si pasan `S_SEARCH_A_CENTRO`
segundos (arranque: 25) sin ninguna detección, va a 90° y se queda ahí.

El cero del encoder **no se escribe en el chip**. `encoder_lib.poner_cero_aca()`
reescribe el registro ZPOS y invalida el `ENCODER_GRADOS_EN_MOTOR_CERO = 79.0`
que ya está medido, dejando mal el próximo arranque. El cero vive como offset
en software.

Los micropasos ya están arreglados; `MOTOR_PASOS_POR_GRADO` no bloquea nada.

### 4.7 Estados del sistema

    IDLE  --iniciar()-->  INIT  -->  BUSCANDO / SIGUIENDO  --detener()-->  IDLE

`iniciar()` y `detener()` son funciones Python. Que las llame una página web,
un botón o una tecla es un detalle posterior y no se implementa ahora.

Si la GoPro falla a mitad de partido, el sistema **sigue trackeando** y emite
una alerta. No aborta.

---

## 5. Los pasos

### P0 — Medir el presupuesto de tiempo

**Produce:** `bench.py`

Sin esto todo lo demás son suposiciones. Hay que saber, con el `.hef` custom y
en la Pi, cuánto cuesta cada parte. `camera_source.py` ya tiene su propio
banco para la captura sola (`python3 camera_source.py --n 300 --modos`); esto
mide la parte de cómputo.

Mide, con p50 y p95 sobre 200 repeticiones:

- preprocesado de un ROI de TRACK (recorte + conversión de canales, sin resize)
- preprocesado de un cuadrante de SEARCH (recorte + resize + conversión)
- `hailo.infer()` sola
- `postprocess.process_candidates()` sobre una salida real

**Criterio de aceptación:** la suma de la fila TRACK está por debajo de 25 ms
en p95. Si no, el objetivo de fps se ajusta acá y no en P7.

> **Prompt.** Escribime `bench.py`. Carga el .hef con
> `hailo_inference.HailoInference`, abre un video con OpenCV y toma 200
> frames. Por cada frame mide con `time.perf_counter()`: (a) armar un recorte
> de TRACK de 1152x640 centrado, (b) armar un cuadrante de SEARCH con
> `main.build_search_input`, (c) `hailo.infer()` sobre cada uno, (d)
> `postprocess.process_candidates()`. Imprimí p50 y p95 de cada etapa en ms, y
> el total de un frame TRACK y de un frame SEARCH de 4 cuadrantes. No toques
> motor ni GoPro. Actualizá `ESTADO.md` con los números medidos.

### P1 — Dos cámaras a la vez y grabador de dataset

**Produce:** `grabar_dataset.py`, más una verificación de `camera_source.py`

`camera_source.py` ya cumple el contrato: `read()` devuelve `(frame, info)`
con `ts_ns` del sensor, `dt_ms`, `perdidos` y `seq`, y acepta `indice` para
elegir el conector CSI. No hay que reescribirlo.

Lo que falta verificar y hacer:

1. Que las dos cámaras funcionen abiertas simultáneamente a 2304x1296 @ 40 fps
   con el IMX708 en binning. **No está probado.** Si el CSI no lo aguanta,
   avisar: el plan B es tener abierta solo la activa, lo que encarece mucho el
   patrón de SEARCH que alterna cámaras.
2. Correr `verificar_canales()` y dejar clavado si el array viene en RGB o en
   BGR (ver 4.1).
3. `grabar_dataset.py`: graba las dos cámaras N segundos a dos .mp4 más un CSV
   con `(camara, seq, ts_ns, perdidos)`.

**Criterio de aceptación:** 60 s de las dos cámaras a 40 fps con menos del 2%
de frames perdidos en cada una, y el orden de canales escrito en `ESTADO.md`.

> **Prompt.** Primero: escribime un script corto que abra las DOS cámaras con
> `camera_source.ThreadedCameraSource(indice=0)` e `(indice=1)`
> simultáneamente a 2304x1296 @ 40 fps, lea 400 frames de cada una alternando,
> y reporte fps efectivo y frames perdidos por cámara. Corré también
> `verificar_canales()` en cada una. Después, si eso funciona, escribime
> `grabar_dataset.py` que grabe N segundos de las dos a .mp4 más un CSV de
> timestamps.

### P2 — `preproceso.py` y `replay.py`, el banco de pruebas

**Produce:** `preproceso.py`, `replay.py`

`preproceso.py` es el rescate de las funciones útiles de `main.py`:
`search_tiles`, `build_search_input`, `build_track_input`, `tile_para_punto`,
`get_model_hw`. Se copian tal cual, sin la lógica de dos modelos
(`dual_inference` no existe y no se va a usar).

`replay.py` es la pieza que hace que todo lo demás se pueda iterar sin ir a la
cancha. Corre **exactamente el mismo código de decisión** que el main final,
pero leyendo frames de un .mp4 en vez de la cámara, y sin motor ni GoPro.

Salidas:

- CSV por frame: `n, ts, modo, camara, tile, conf, x, y, w, h, angulo,
  sector, flost, aceptado_por`
- video anotado a 0.5x: ROI dibujado, predicción del Kalman, sector actual
- resumen impreso: % de frames con detección aceptada, % de tiempo en TRACK,
  transiciones TRACK→SEARCH, cambios de sector, latencia p50/p95

Ese resumen es la métrica que se compara entre versiones del algoritmo, y es
también la tabla de resultados de la tesis.

**Criterio de aceptación:** corre de punta a punta sobre el video que el
modelo nunca vio y escupe el resumen.

> **Prompt.** Primero movete las funciones de preprocesado de `main.py` a
> `preproceso.py`, sin la lógica de `dual_inference` ni de `SEARCH_FULL`, y
> sin importar `cv2` a nivel de módulo. Después escribime `replay.py` con
> `--video`, `--salida-csv`, `--salida-mp4`. Levanta el .hef, lee frame a
> frame, y por cada uno llama a un `procesar_frame()` que después va a ser
> compartido con el main final. Por ahora ese `procesar_frame()` hace solo la
> detección con la lógica de tiles. Escribí el CSV con estas columnas [pegar
> lista], el mp4 anotado, y el resumen final. Nada de motor, encoder ni GoPro.

### P3 — `tracker.py`, el algoritmo de decisión

**Produce:** `tracker.py`, `test_tracker.py`

Este es el corazón y el paso que más iteraciones va a llevar. Python puro: no
importa cv2, ni hailo, ni nada de hardware. Se testea con listas de
detecciones inventadas.

Lógica por frame, dada una lista de candidatos de `process_candidates()`
ordenada por confianza:

```
si modo == TRACK:
    pred = kalman.predecir(dt)
    c = mejor candidato por confianza

    si c.conf >= CONF_ALTA (0.50):
        aceptar = True; motivo = "confianza"
    elif c.conf >= CONF_BAJA (0.10):
        # el mas CERCANO a la prediccion, no el mas confiado
        c = argmin(dist(cand, pred)) entre los que superan CONF_BAJA
        aceptar = dist(c, pred) < radio_gate(c); motivo = "kalman"
    else:
        aceptar = False; motivo = "sin candidato"

    si aceptar y dist(c, ultima_pos_aceptada) > radio_gate(c):
        aceptar = False; motivo = "delta imposible"

    si aceptar:
        flost = 0
        kalman.actualizar(c)
        roi_centro = c
    si no:
        flost += 1
        roi_centro = CONGELADO en la ultima posicion ACEPTADA
        motor = congelado

    si flost >= FLOST_A_SEARCH (10):
        modo = SEARCH

si modo == SEARCH:
    barrer segun el patron de abajo
    si aparece un candidato con conf >= CONF_ALTA:
        modo = TRACK; kalman.reiniciar(c); flost = 0
```

Dos detalles que no son obvios y que son el punto entero del diseño:

- Con confianza baja **no se toma el candidato más confiado**, se toma el más
  cercano a la predicción entre los que pasan el piso. Ese es todo el motivo
  por el que `postprocess.process_candidates()` devuelve `topk` y no uno solo:
  el pico más fuerte del frame puede ser una zapatilla blanca mientras la
  pelota real está en el segundo candidato.
- Cuando no se acepta, el ROI se congela en la última posición **aceptada**,
  no en la predicción del Kalman. Si el filtro está siguiendo un fantasma, un
  ROI que persigue la predicción se va del frame y no vuelve nunca.

**Patrón de SEARCH:**

1. Cámara donde se vio la pelota por última vez: 3 barridos completos. Cada
   barrido es un frame nuevo con sus 4 cuadrantes, o sea 4 inferencias.
2. La otra cámara: 1 barrido completo.
3. Volver a 1, indefinidamente.

Cada barrido arranca por el cuadrante donde se la vio por última vez
(`preproceso.tile_para_punto` ya hace eso).

**Criterio de aceptación:** pasan los tests, y en `replay.py` sobre el video
real el porcentaje de tiempo en TRACK sube respecto de la versión sin Kalman.
Se comparan los dos resúmenes de P2.

> **Prompt.** Escribime `tracker.py` con la clase `Tracker` y el método
> `actualizar(candidatos, ts_ns, camara) -> ResultadoFrame`. Sin imports de
> hardware ni de cv2. Implementá exactamente esta lógica [pegar el
> pseudocódigo y el patrón de SEARCH]. El Kalman de velocidad constante en
> `(x, y, vx, vy)` con dt real escribilo a mano con numpy, no traigas
> filterpy. `radio_gate(det) = max(60.0, 6.0 * max(det.w, det.h))`. Todas las
> constantes salen de `config.py`. Exportá también un enum `Mode` compatible
> con el que `gpio_timer.py` importa hoy de `state_machine.py`. Después
> escribime `test_tracker.py` con pytest cubriendo: detección buena continua,
> baja confianza cerca de la predicción (se acepta), baja confianza lejos (se
> rechaza), salto imposible, 10 frames perdidos seguidos, vuelta a TRACK, y el
> orden del patrón de SEARCH.

### P4 — `sectores.py`, el Schmitt trigger

**Produce:** `sectores.py`, `test_sectores.py`

Python puro también. Entra `(angulo, omega, t)`, sale `(sector, cambio,
angulo_objetivo, motivo)`.

```
bordes  = [20, 40, 60, 80, 100, 120, 140, 160]
centros = [30, 50, 70, 90, 110, 130, 150]

candidato = sector cuyo tramo contiene el angulo, PERO exigiendo que el
            angulo haya cruzado el borde por mas de HISTERESIS_SECTOR_DEG

si candidato == actual:  no pasa nada
si candidato != actual:
    si |omega| >= OMEGA_RAPIDA:  cambiar ya
    si no: exigir MS_PERMANENCIA sostenidos votando al mismo candidato
si cambio: exigir ademas MS_MINIMO_ENTRE_MOVIMIENTOS desde el ultimo
```

**Criterio de aceptación:** tres tests. Una rampa lenta de 20° a 160° produce
exactamente 6 cambios. Una senoidal de ±4° centrada justo sobre un borde
durante 30 s produce **cero** cambios. Un salto de 60° con omega alta produce
un cambio en menos de 100 ms.

> **Prompt.** Escribime `sectores.py` con la clase `Sectorizador` según [pegar
> el pseudocódigo]. Sin dependencias de hardware. Que exponga `sector_actual`,
> `angulo_objetivo` (el centro del sector) y `motivo` (string para el CSV).
> Las constantes nuevas van a `config_hw.py` con comentario. Después
> `test_sectores.py` con los tres casos.

### P5 — `init_sistema.py`, el arranque completo

**Produce:** `init_sistema.py`

Secuencia:

1. Encoder: `aplicar_config()` (es soft write, se pierde al cortar la
   alimentación), verificar PPR contra `ENC_PPR_ESPERADO`, leer el diagnóstico
   del imán, leer el ángulo absoluto X.
2. Motor: verificar USB, `deshabilitar()`, fijar corriente, micropasos,
   velocidad y aceleración, `habilitar()`.
3. Homing al cero mecánico: mover `-(X - ENCODER_GRADOS_EN_MOTOR_CERO)`,
   verificar contra el encoder, `motor.zero()`, y guardar el offset del
   encoder en software.
4. Ir a 180° del mundo, verificar contra el encoder.
5. Ir a 90° del mundo, verificar contra el encoder, quedarse ahí.
6. Devolver un informe con los tres errores medidos.

**Criterio de aceptación:** los tres errores por debajo de
`TOLERANCIA_HOMING_DEG`. Si alguno se pasa, el init falla ruidosamente con un
mensaje que diga qué revisar, y no deja arrancar la grabación.

> **Prompt.** Escribime `init_sistema.py` con `inicializar(con_motor=True,
> con_gopro=True) -> Sistema`. Seguí exactamente los 6 pasos de [pegar la
> lista]. Usá `encoder_lib`, `motor_lib` y `geometria` tal como están, sin
> modificarlos. El cero del encoder se guarda como offset en software, NO se
> escribe el registro ZPOS. Cada verificación imprime encoder medido, esperado
> y error. Si un error supera la tolerancia, excepción con mensaje accionable.

### P6 — `control_motor.py`, el hilo no bloqueante

**Produce:** `control_motor.py`, y los métodos que falten en `hw_falsos.Motor`

```python
ControlMotor(motor, encoder)
    .apuntar(angulo_mundo)   # no bloquea nunca, pisa el objetivo anterior
    .estado()  -> {objetivo, ultimo_error_encoder, moviendo, movimientos}
    .parar()
```

Un `threading.Thread` daemon con una `queue.Queue(maxsize=1)` que se pisa: si
está llena, se descarta el objetivo viejo y se mete el nuevo. El hilo
convierte con `geometria.angulo_a_grados_motor()`, manda el movimiento, espera
el fin con `esperar_fin()`, lee el encoder y guarda el error. Si mientras se
movía llegó un objetivo nuevo, lo atiende enseguida.

Si el error contra el encoder supera la tolerancia dos veces seguidas, alerta:
eso son pasos perdidos.

`hw_falsos.Motor` hoy no tiene `mover_pasos`, `ir_a_pasos`, `info_movimiento`
ni `en_movimiento`. Se completan para que el hilo se pueda probar sin fierro.

**Criterio de aceptación:** un script que llame `apuntar()` a 20 Hz con
ángulos aleatorios y verifique que **ninguna llamada tarda más de 1 ms**.

> **Prompt.** Escribime `control_motor.py` según [pegar el contrato].
> Completá también los métodos que falten en `hw_falsos.Motor` para que se
> pueda probar sin hardware. Incluí el script de prueba de las 20 Hz que
> reporta el p95 del tiempo de `apuntar()`.

### P7 — `main_final.py`

**Produce:** `main_final.py`

Un solo hilo de decisión. El motor y la GoPro en los suyos.

```
IDLE
  iniciar():
      sistema = init_sistema.inicializar()
      gopro.iniciar_grabacion()
      loop:
          frame, info = camaras.leer(activa)
          candidatos = inferir segun modo (ROI de TRACK o cuadrante de SEARCH)
          r = tracker.actualizar(candidatos, info["ts_ns"], activa)
          si r.aceptado:
              angulo = geometria.pixel_a_angulo(r.x, ancho, activa)
              activa = selector.actualizar(angulo, r.conf, t)
              s = sectorizador.actualizar(angulo, r.omega, t)
              si s.cambio: control_motor.apuntar(s.angulo_objetivo)
          si tracker.segundos_sin_deteccion > S_SEARCH_A_CENTRO:
              control_motor.apuntar(90.0)
          log_csv(...)
  detener():
      gopro.detener_grabacion(); motor a 90; deshabilitar; cerrar todo
```

**Criterio de aceptación:** corre 10 minutos en la mesa con la pelota en la
mano sin excepciones, con el CSV completo, y `detener()` deja el sistema
limpio (motor deshabilitado, cámaras cerradas, GoPro detenida).

> **Prompt.** Escribime `main_final.py` que ate `init_sistema`,
> `camera_source`, `hailo_inference`, `postprocess`, `preproceso`, `tracker`,
> `geometria`, `sectores`, `control_motor` y `gopro_lib` según [pegar el
> pseudocódigo]. Estados IDLE / INIT / GRABANDO expuestos como `iniciar()` y
> `detener()`, más un `__main__` que los llame por teclado mientras no exista
> la web. El CSV con las mismas columnas que `replay.py`. Nada de `input()`
> dentro del loop. Si la GoPro falla, alerta y seguir.

### P8 — Calibración en la cancha

**Produce:** `calibrar.py` y los números de `CAL_PIXEL_ANGULO`

Es lo único que hay que medir físicamente. Los 55° y 125° de `config_hw.py`
están inventados.

**El truco: el encoder es el transportador.** No hace falta transportador ni
cinta métrica. El motor tiene un encoder absoluto de 2048 pasos por vuelta, o
sea 0.18° de resolución: es el instrumento más preciso que hay en el poste.

1. Poné la pelota quieta en un punto de la cancha.
2. Movés el motor a mano hasta que la pelota queda centrada en la mira de la
   GoPro.
3. Leés el encoder. Ese es el ángulo del mundo verdadero de ese punto.
4. Al mismo tiempo, las dos cámaras sacan una foto y el modelo detecta la
   pelota. Se guarda `(x_cam0, angulo)` y `(x_cam1, angulo)`.
5. Repetir en 7 puntos repartidos, incluyendo los dos extremos y al menos tres
   en la zona de solape.

Con eso `CAL_PIXEL_ANGULO[0]` y `[1]` quedan llenos con pares medidos, y
`geometria._interpolar_tabla()` (que ya está escrito) ignora el FOV nominal.
Los dos ejes quedan atados al mismo cero **por construcción**, porque los dos
se midieron contra el mismo encoder.

**Criterio de aceptación:** para los puntos del solape, el ángulo que reporta
la cámara 0 y el que reporta la cámara 1 difieren en menos de 3°. Si difieren
más, hay error de montaje o una cámara está espejada.

> **Prompt.** Escribime `calibrar.py`. Loop interactivo: comandos para mover
> el motor (`+5`, `-5`, `+1`, `-1`), un comando `medir` que saca foto con las
> dos cámaras, corre el modelo, lee el encoder y guarda `(camara, x_px,
> angulo_encoder, conf)` en un JSON, y `fin` que imprime las dos listas listas
> para pegar en `config_hw.CAL_PIXEL_ANGULO`, más el ajuste por mínimos
> cuadrados de centro y FOV de cada cámara para comparar contra lo nominal.

---

## 6. Orden del día de cancha

1. `init_sistema.py` solo. Verificar los tres errores.
2. `calibrar.py`, 7 puntos. Pegar los números en `config_hw.py`.
3. `grabar_dataset.py`, 10 minutos de partido real con las dos cámaras. Este
   material es el que después alimenta el replay.
4. `main_final.py` en vivo, 10 minutos, mirando el video de la GoPro.
5. Volver y correr `replay.py` sobre lo grabado en el punto 3 para ajustar
   umbrales sin volver a la cancha.

---

## 7. Lo que sigue abierto

- **Latencia real.** Todo asume que TRACK entra en 25 ms. P0 lo confirma o lo
  desmiente.
- **Las dos cámaras a 40 fps simultáneas.** No probado. Si el CSI no lo
  aguanta, cambia el patrón de SEARCH.
- **Orden de canales del array de Picamera2.** Hasta que P1 lo verifique, no
  se sabe si el modelo está recibiendo RGB o BGR.
- **`MS_PERMANENCIA` (600 ms) y `S_SEARCH_A_CENTRO` (25 s).** Valores
  tentativos. Se ajustan mirando el video del punto 4 del día de cancha: si el
  motor se ve nervioso, subir; si llega tarde, bajar.
- **La zapatilla blanca.** El único falso positivo conocido. El gate de
  plausibilidad lo mata mientras haya track, pero puede enganchar el arranque
  de SEARCH. Si aparece en el replay, la defensa es exigir dos frames
  consecutivos con confianza alta para entrar a TRACK.
