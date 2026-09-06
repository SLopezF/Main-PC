"""
config_hw.py

Constantes del hardware que config.py todavia no cubre: encoder, motor,
geometria de las dos camaras y la histeresis de cambio entre ellas.

Va aparte de config.py a proposito, para que puedas seguir editando
config.py (camara / modelo / umbrales de deteccion) sin mezclarlo con la
mecanica.
"""

# =============================================================================
# ENCODER  (AS5047D por SPI, ver as5047_config.py / test3.py)
# =============================================================================

ENC_BUS = 0
ENC_DEVICE = 0                     # chip select: /dev/spidev0.0

# Lo que esperamos que reporte el encoder despues de aplicar la config.
# El arranque compara contra esto y avisa si no coincide: si el chip quedo
# con la config de fabrica, la resolucion del ABI no es la que creemos.
# OJO: es un AS5047D, no el P. En el D el maximo del ABI binario es 512 ppr
# = 2048 pasos/vuelta (el P llega a 1024/4096). Pedir 1024 aca hace que
# encoder_lib rechace la config con un ValueError antes de escribir nada.
ENC_PPR_ESPERADO = 512             # pulsos por vuelta en modo binario
ENC_PASOS_VUELTA_ESPERADO = 2048   # = ppr * 4 (cuadratura)

# La config del AS5047 es VOLATIL (soft write): se pierde al cortar la
# alimentacion. Por eso se re-aplica en cada arranque.
ENC_APLICAR_CONFIG_AL_ARRANCAR = True


# =============================================================================
# MOTOR  (ESP32 + TMC2209 + AccelStepper por USB, ver rpi5_motor_control_usb.py)
# =============================================================================

MOTOR_PUERTO = None                # None = autodeteccion /dev/ttyUSB* o ttyACM*
MOTOR_BAUD = 115200
MOTOR_CORRIENTE_MA = 600
MOTOR_MICROPASOS = 16
MOTOR_VELOCIDAD = 2000             # pasos/s
MOTOR_ACELERACION = 4000           # pasos/s^2

# --- Homing -----------------------------------------------------------------
# Dato medido: con el motor en su cero mecanico, el encoder absoluto marca
# 79 grados. Al arrancar se lee X y se corrige la diferencia.
ENCODER_GRADOS_EN_MOTOR_CERO = 79.0

# El motor esta corrido (X - 79) grados. Para volver al home hay que moverlo
# -(X - 79). Si al probarlo se va para el lado contrario, poné +1 aca en vez
# de tocar la formula.
HOMING_SENTIDO = -1

# Grados de MOTOR por grado de EJE (el que mira el encoder). 1.0 = acople
# directo. Con reduccion esto es != 1, y sin este factor el homing queda mal
# por un factor constante.
RELACION_TRANSMISION = 1.0

# Tolerancia al verificar el homing contra el encoder. Por encima de esto el
# arranque avisa (no aborta: puede ser backlash).
TOLERANCIA_HOMING_DEG = 1.5

# --- Apuntado ---------------------------------------------------------------
# Angulo del MUNDO (0..180, el que devuelve geometria.py) al que apunta el
# motor cuando esta en su cero.
MOTOR_ANGULO_MUNDO_EN_CERO = 90.0

# +1 si aumentar el angulo del mundo requiere grados de motor positivos.
MOTOR_SENTIDO = +1

# Limites mecanicos en grados de motor respecto del cero. Todo comando se
# clampea contra esto ANTES de mandarlo.
MOTOR_GRADOS_MIN = -95.0
MOTOR_GRADOS_MAX = +95.0

MOTOR_TIMEOUT_MOV_S = 12.0

# --- Conversion grados -> pasos ---------------------------------------------
# El firmware reporta mal los micropasos en 'status' (dice la mitad) y calcula
# 'deg'/'degto' con esa misma variable, asi que los movimientos en grados
# salen con error. Con este numero seteado, motor_lib convierte los grados a
# pasos DEL LADO DE LA PI y manda 'move'/'moveto', sin pasar por esa cuenta.
#
# Se mide una sola vez, contra el encoder absoluto:
#     python3 motor_lib.py --calibrar
#
# Valor teorico de referencia (motor de 200 pasos/vuelta, acople directo):
#     16 micropasos  ->  200*16/360  =  8.889 pasos/grado
#     32 micropasos  ->  200*32/360  = 17.778 pasos/grado
# Compará lo medido contra esa tabla: te dice en cuantos micropasos esta
# REALMENTE el driver, que es lo que el 'status' y la consulta no se ponen
# de acuerdo en decir.
#
# None = volver a delegar los grados al firmware (no recomendado hasta que el
# bug este arreglado).
MOTOR_PASOS_POR_GRADO = None

# Pasos por vuelta del motor SIN micropasos (1.8 deg/paso = 200). Solo lo usa
# el diagnostico para deducir en cuantos micropasos esta el driver de verdad.
MOTOR_PASOS_VUELTA = 200


# =============================================================================
# MODELO: ESTRATEGIA DE ENTRADA (mosaico vs frame entero)
# =============================================================================
# El .hef oficial de COCO es CUADRADO: 640x640, no 640x1152 como el propio.
# El codigo lee el tamano de hailo.input_shape en tiempo de ejecucion, asi
# que no hay que tocar nada mas; pero cambia cual conviene de las dos formas
# de armar la entrada.
#
# Tamano de una pelota nro 4 (21 cm de diametro) con FOV 100 deg sobre
# 2304 px, segun distancia y segun como se arme la entrada:
#
#     distancia   nativo    frame entero (0.278x)   mosaico 2x2 (0.556x)
#        2.0 m    138 px          38 px                   77 px
#        2.5 m    111 px          31 px                   62 px
#        3.0 m     92 px          26 px                   51 px
#       20.0 m     14 px           4 px                    8 px
#
# El mosaico existe por la fila de 20 m: ahi el frame entero deja la pelota
# en 4 px y el modelo no ve nada. Pero a 2-3 m el frame entero da 26-38 px,
# que a un modelo COCO le sobra, y cuesta UNA inferencia en vez de cuatro,
# sin riesgo de que la pelota caiga partida entre dos tiles.
#
# True  = frame entero con letterbox. Para la prueba de cerca.
# False = mosaico de SEARCH_TILE_GRID tiles. Para la cancha de verdad.
SEARCH_FULL = False

# Para el aviso de arranque: con esto se estima a cuantos pixeles va a quedar
# la pelota y se avisa si queda demasiado chica.
DIAMETRO_PELOTA_M = 0.21
DISTANCIA_PRUEBA_M = 2.5


# =============================================================================
# MODELO: CLASE OBJETIVO
# =============================================================================
# Mientras no este el .hef propio entrenado con TUS pelotas, se usa el .hef
# oficial de Hailo para YOLO26n con COCO y se filtra una sola clase.
#
#     32 = sports ball   (COCO, indexado desde 0: person=0, ... )
#
# None solo sirve para el modelo propio, que tiene una unica clase. Con el de
# COCO, dejarlo en None hace que postprocess tome el maximo sobre las 80
# clases, o sea la persona mas confiada del frame: en una cancha de futbol 5
# eso gana SIEMPRE y el sistema apunta a un jugador en vez de a la pelota.
CLASE_OBJETIVO = None            # config_hw.py: tu modelo tiene una sola clase
NOMBRE_CLASE = "ball"

# Que esperar del modelo generico, para no perder tiempo buscando un bug
# donde no lo hay:
#   - COCO tiene 'sports ball' entrenada mayormente con pelotas grandes y
#     cercanas (basquet, tenis en primer plano). Una pelota de futbol a 20 m
#     de una camara gran angular es un caso raro para ese modelo.
#   - Vas a ver confianzas bastante mas bajas que las 0.879 medidas con el
#     .hef propio, y falsos positivos en cabezas y en carteles redondos.
#   - Por eso los umbrales de deteccion bajan mientras dure esta etapa. No
#     dejes estos valores cuando vuelvas a tu modelo.


# =============================================================================
# GEOMETRIA DE LAS CAMARAS
# =============================================================================

CAMARAS = (0, 1)

# Campo de vision horizontal de cada camara, en grados.
CAM_FOV = {0: 100.0, 1: 100.0}

# A que angulo del mundo (0..180) mira el CENTRO de cada camara. Medilo con
# la pelota en el centro de cada imagen y un transportador.
CAM_CENTRO_ANGULO = {0: 55.0, 1: 125.0}

# True si la camara esta montada dada vuelta (x creciente -> angulo decreciente).
CAM_ESPEJO = {0: False, 1: False}

# Calibracion opcional por camara: pares (pixel_x, angulo_mundo) medidos
# poniendo la pelota en puntos conocidos. Con 3 o mas puntos se interpola y se
# ignora el FOV nominal, que tiene error de barril en los bordes.
CAL_PIXEL_ANGULO = {0: [], 1: []}


# =============================================================================
# HISTERESIS DE CAMBIO DE CAMARA
# =============================================================================
# Con dos camaras que se solapan, la pelota cerca del limite hace que la
# "mejor" camara alterne frame a frame. Tres frenos:
#   1. margen angular: la otra camara tiene que estar mejor por al menos
#      HIST_CAMARA_GRADOS, no apenas mejor;
#   2. permanencia: esa condicion se sostiene MS_CONFIRMAR_CAMBIO ms;
#   3. piso duro de tiempo entre cambios.

HIST_CAMARA_GRADOS = 8.0
MS_CONFIRMAR_CAMBIO = 400.0

# Confianza minima para que una deteccion cuente como voto de cambio: un
# falso positivo de 0.3 no deberia mover la camara.
CONF_CAMBIO_CAMARA = 0.50

# Si la camara activa no ve nada por este tiempo, se pasa a la otra a barrer,
# sin esperar voto angular.
MS_PERDIDA_CAMARA = 1500.0

# Cambiar de camara cuesta (el AE de la otra tarda en converger).
MS_MINIMO_ENTRE_CAMBIOS = 1000.0


# =============================================================================
# SECTORES DEL MOTOR  (P4)
# =============================================================================
# LA DECISION DE DISEÑO CENTRAL DEL TRABAJO:
# el motor NO apunta a la pelota, apunta al CENTRO DEL SECTOR donde esta la
# jugada. Un video que sigue la pelota permanentemente marea y se ve amateur;
# lo que se busca es una transmision, o sea la camara quieta la mayor parte
# del tiempo y movimientos poco frecuentes y bien justificados.
#
# Con un FOV de mas de 100 grados en la GoPro desde 3 m de altura, un error de
# medio sector (10 grados) no saca la pelota del cuadro.

# 7 sectores de 20 grados sobre 20..160.
SECTOR_DESDE = 20.0
SECTOR_HASTA = 160.0
N_SECTORES = 7
# bordes  = [20, 40, 60, 80, 100, 120, 140, 160]
# centros = [30, 50, 70, 90, 110, 130, 150]

# --- Schmitt trigger ---------------------------------------------------------
# Para pasar del sector i al i+1 no alcanza con cruzar el borde: hay que
# superarlo por este margen. Sin esto, una pelota parada justo sobre un borde
# hace saltar el motor de ida y vuelta indefinidamente.
#
# TENTATIVO (arranque del briefing). Se ajusta mirando el video de cancha: si
# el motor se ve nervioso, subir.
HISTERESIS_SECTOR_DEG = 5.0

# Ademas del margen, la condicion tiene que SOSTENERSE este tiempo. El margen
# solo no alcanza porque el ruido de la deteccion (la pelota "salta" varios
# grados entre frames) puede superarlo; la permanencia sola tampoco, porque
# una pelota que se queda sobre el limite la cumple todo el tiempo.
#
# TENTATIVO. El valor viejo del diseño de 9 zonas era 1500 ms.
MS_PERMANENCIA = 600.0

# Piso duro entre movimientos del motor, pase lo que pase.
MS_MINIMO_ENTRE_MOVIMIENTOS = 800.0

# --- Excepcion por regimen ---------------------------------------------------
# Si la velocidad angular supera esto, se saltea la PERMANENCIA y se mueve ya.
# Un pelotazo cruza un sector de 20 grados en unos 100 ms: esperar 600 ms lo
# perderia. La histeresis en grados se sigue exigiendo igual.
#
# En grados por segundo. Se ajusta mirando la columna omega del CSV: ahi se ve
# el corte real entre jugada trabada y pelotazo.
OMEGA_RAPIDA = 28.0
OMEGA_LENTA = 13.0

# El motor da la vuelta a la cancha en menos de medio segundo y es silencioso,
# asi que su velocidad no es una restriccion de diseño en ningun lado.

# Si no se ve la pelota por este tiempo, el motor va al centro (90 grados) y se
# queda ahi. En SEARCH el motor NO se mueve hasta que pasa esto.
# TENTATIVO.
S_SEARCH_A_CENTRO = 25.0


# =============================================================================
# GOPRO
# =============================================================================

GOPRO_SSID = "HERO7 Silver"
GOPRO_PASSWORD = "ZBn-6K2-BY9"
GOPRO_IP = "10.5.5.9"
GOPRO_HABILITADA = True            # False = saltear la GoPro en el debug


# =============================================================================
# SALIDAS DEL MODO DEBUG
# =============================================================================

# /dev/shm es RAM: no toca la SD ni frena el loop. Se borra al reiniciar.
DIR_DEBUG = "/dev/shm/partido"
DEBUG_ESCALA = 0.5                 # escala de la imagen anotada que se guarda
DEBUG_GUARDAR_RECORTE = True       # recorte alrededor de la deteccion

# Mantener las dos Picamera2 abiertas a la vez: cambiar de camara pasa a ser
# instantaneo, a costa de ancho de banda CSI y RAM. Con False se cierra la
# anterior al cambiar y cada cambio cuesta ~2 s de convergencia del AE.
CAM_MANTENER_ABIERTAS = True