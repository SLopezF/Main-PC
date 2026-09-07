"""
config.py

Configuración centralizada del sistema de detección/tracking de pelota
sobre Raspberry Pi 5 + Hailo-8.

Todas las constantes ajustables del proyecto viven acá para no tener
que buscarlas dispersas en el resto de los módulos.
"""

# =============================================================================
# ENTRADA: CÁMARA EN VIVO
# =============================================================================

# Modo de sensor del IMX708. Los tres que expone libcamera son:
#     4608x2592 -> 14.35 FPS   (pelota ~34 px, confianza medida 0.879)
#     2304x1296 -> 56.03 FPS   (pelota ~17 px, confianza medida ~0.72)
#     1536x864  -> 120.13 FPS  (RECORTE de 3072x1728: perdés campo de visión)
CAM_ANCHO = 2304
CAM_ALTO = 1296

# FPS pedidos a la cámara. OJO: 56.03 es el techo del sensor a esta
# resolución. Pedir 60 hace que libcamera baje a lo que puede.
CAM_FPS = 40.0

# None = AUTOMATICO. La cámara se comporta igual que `rpicam-hello`:
# auto-exposición, auto-balance de blancos y autofoco continuo. Es el
# modo en el que la imagen se ve mejor y el modelo detecta mejor, así
# que es el default.
#
# Poné un número solo si sabés por qué. Referencia (f/2.2, ganancia 1.0):
#     sol directo   ~150 us      nublado claro  ~600 us
#     sol con nube  ~300 us      sombra        ~1200 us
# Con exposición fija hay que fijar TAMBIÉN la ganancia (mínimo 1.0),
# porque mandar 0 hace que libcamera rechace todo el set_controls.
CAM_EXPOSICION_US = None
CAM_GANANCIA = None

# None = no tocar el foco (queda el autofoco continuo por defecto).
# "continuo", "auto" (un ciclo y fija), o un número = LensPosition manual
# en dioptrías (0.0 = infinito). Para una cancha, 0.0 suele ser correcto
# y elimina el vaivén del autofoco.
CAM_ENFOQUE = None

# Cantidad de buffers de la cámara. Con menos de 3 se frena si el
# consumidor tarda.
CAM_BUFFERS = 4


# =============================================================================
# ENTRADA: ARCHIVO DE VIDEO (camino viejo, para pruebas offline)
# =============================================================================

VIDEO_PATH = "grabacion.mkv"
NATIVE_WIDTH = 2304
NATIVE_HEIGHT = 1296
NATIVE_FPS = 60


# =============================================================================
# MODELO / ENTRADA A LA NPU
# =============================================================================

HEF_PATH = "yolo26n_sincalib_opt1.hef"      # lo usa el backend Hailo, en la Pi
MODELO_PT = "run_yolo26n_sesiones_1152x640px_300ep.pt"     # lo usa el backend Ultralytics, en la PC

# CORREGIDO. Antes decía 1152 (escalar, cuadrado). El modelo NO es
# cuadrado: `hailortcli parse-hef` reporta NHWC(640x1152x3), o sea
# 640 de alto por 1152 de ancho. El escalar cuadrado era el origen del
# error "shape (1152,1152,3), se esperaba (640,1152,3)".
#
# El código lo lee de hailo.input_shape en tiempo de ejecución, así que
# esto es solo respaldo. Formato: (alto, ancho).
MODEL_INPUT_SIZE = (640, 1152)   # alto, ancho

MODEL_INPUT_CHANNELS = 3

# Grilla del mosaico de SEARCH. Con (2,2) sobre 2304x1296 la escala
# efectiva es 0.988, o sea casi resolución nativa: la pelota se mantiene
# en ~17 px. Meter el frame entero en el modelo daría 8.5 px, donde la
# confianza medida cae a 0.0025.
# Orden de canales que entrega la fuente de frames. True = BGR, que es lo que
# devuelven las DOS fuentes de fuente.py: cv2.imread() y Picamera2 con formato
# "RGB888" (el nombre viene del empaquetado de bytes, no del orden en numpy).
# preproceso._a_rgb() convierte a RGB, que es lo que espera el modelo.
#
# PENDIENTE DE P1: esto NO se confirmo todavia contra la camara real con
# CameraSource.verificar_canales(). Si aquello dice que el array ya viene en
# RGB, se pone False ACA y no se toca ningun cvtColor. El modelo rinde 0.879
# contra 0.859 en recorte nativo y 0.476 contra 0.061 en frame reducido, o sea
# que equivocarse cuesta un factor 8 en el caso peor y es invisible a ojo.
FUENTE_ENTREGA_BGR = True

SEARCH_TILE_GRID = (2, 2)


# =============================================================================
# UMBRALES DE DETECCIÓN
# =============================================================================

# Umbral de aceptación de una detección.
CONF_MIN_DETECTION = 0.1

# Piso para que postprocess.process_candidates() devuelva un candidato.
# Debe ser MÁS BAJO que CONF_MIN_DETECTION: es "qué vale la pena mirar",
# no "qué se acepta". Solo lo usa el módulo de seguimiento.
CONF_CANDIDATO = 0.05


# =============================================================================
# HISTÉRESIS DE ESTADOS (SEARCH <-> TRACK)
# =============================================================================

# BAJADO de 0.80 a 0.60.
# Medición real a resolución nativa: 0.879. Pero a 2304x1296 la pelota
# mide ~17 px en vez de 34, y la confianza esperada ronda 0.72. Con el
# umbral en 0.80 el sistema no entraría NUNCA en TRACK y se quedaría
# barriendo tiles para siempre.
CONF_ENTER_TRACK = 0.60

CONF_EXIT_TRACK = 0.40

# Umbral por CANTIDAD de frames. Sirve con archivo de video, donde no se
# descarta ninguno.
MAX_FRAMES_LOST = 5

# Umbral TEMPORAL, equivalente y preferible con cámara en vivo: si se
# descartan frames, contar frames deja de significar nada (5 frames
# pueden ser 90 ms o 250 ms). A 40 FPS, 300 ms son ~12 frames.
MS_PERDIDA_TRACK = 300.0


# =============================================================================
# TRACKER (P3)
# =============================================================================
# Umbrales de aceptacion de un candidato. Los dos son del briefing.
#
# Medicion real sobre imagenes/Soleado con el .pt propio: confianza p50 0.810,
# p95 0.929, minima aceptada 0.107. O sea que una deteccion buena queda MUY
# por encima de CONF_ALTA y se acepta sola.
#
# CONF_BAJA no es "que se acepta": es "que vale la pena mirar". Un candidato
# entre CONF_BAJA y CONF_ALTA se acepta SOLO si cae cerca de donde el Kalman
# predijo. Bajarlo ofrece mas candidatos al filtro; subirlo lo deja ciego
# cuando la pelota se ve mal (borrosa, tapada, contra el sol).
CONF_ALTA = 0.50
CONF_BAJA = 0.10

# Cuantos frames seguidos sin aceptar nada antes de volver a SEARCH.
# A 40 fps, 10 frames son 250 ms.
FLOST_A_SEARCH = 10

# --- Gate de plausibilidad ---------------------------------------------------
# radio_gate_px = max(GATE_PX_MIN, GATE_DIAMETROS * max(w, h))
#
# De donde sale el 6: a 50 m/s (el techo fisico de un pelotazo) y 40 fps, la
# pelota se desplaza 1.25 m por frame. Una pelota nro 4 mide 0.21 m. El
# cociente es 5.95 y es INDEPENDIENTE DE LA DISTANCIA, porque el
# desplazamiento aparente y el diametro aparente se escalan los dos con 1/d.
# No hay que calibrar nada: ni FOV, ni distancia, ni altura.
#
# El piso de 60 px cubre la pelota muy lejos, donde la caja mide 8 px y 6
# diametros serian 48 px, menos que el ruido de la propia deteccion.
GATE_DIAMETROS = 6.0
GATE_PX_MIN = 60.0

# --- Kalman ------------------------------------------------------------------
# Velocidad constante en (x, y, vx, vy), en pixeles del frame NATIVO.
#
# ESTOS DOS NUMEROS SON TENTATIVOS y son el ajuste fino del filtro. Se tunean
# mirando la columna err_pred del CSV de replay.py, que es el residuo
# (distancia entre lo que el filtro predijo y donde aparecio la deteccion):
#
#   err_pred grande y sistematico  -> el modelo de movimiento no alcanza, o
#       KALMAN_SIGMA_MEDICION esta muy chico (el filtro le cree de mas a su
#       propia prediccion y reacciona tarde a los cambios de direccion)
#   err_pred chico pero la trayectoria tiembla -> KALMAN_SIGMA_MEDICION muy
#       grande: el filtro esta copiando el ruido de la deteccion
#
# Ruido de proceso: cuanta aceleracion NO modelada se admite, en px/s^2. Una
# pelota que rebota o la patean cambia de velocidad de golpe, y eso el modelo
# de velocidad constante no lo ve venir. Arranque: 2000 px/s^2.
KALMAN_SIGMA_ACEL = 2000.0

# Ruido de medicion: cuanto se desvia el centro que reporta el modelo respecto
# del centro real, en px. Arranque: 8 px, del orden de medio diametro de la
# pelota a distancia media.
KALMAN_SIGMA_MEDICION = 8.0

# --- Patron de SEARCH --------------------------------------------------------
# Barridos COMPLETOS (los 4 cuadrantes) en la camara donde se vio la pelota
# por ultima vez, antes de probar con la otra. La asimetria es a proposito: lo
# mas probable es que siga donde estaba.
#
# Medicion de la linea de base (un cuadrante por frame, sin este patron):
# recuperarse de una perdida costaba ~16 frames y la deteccion en SEARCH era
# 16.7%, o sea aproximadamente 1 de cada 4.
SEARCH_BARRIDOS_ACTIVA = 3
SEARCH_BARRIDOS_OTRA = 1


# =============================================================================
# GPIO
# =============================================================================

# Pin de timing: alto durante preprocesado + inferencia + postproceso.
# NO cubre la captura ni el dibujado, a propósito.
GPIO_PIN_TIMING = 17

# Pin de modo: 0 = SEARCH, 1 = TRACK. Con un solo pin no se pueden
# distinguir los dos modos en el osciloscopio; con este, CH2 te separa
# las dos poblaciones de pulsos.
GPIO_PIN_MODE = 27


# =============================================================================
# SALIDA DE DEBUG
# =============================================================================

# /dev/shm es un sistema de archivos EN RAM: no toca la tarjeta SD, no
# frena el loop y no la desgasta. Se borra al reiniciar, así que copiá
# lo que quieras conservar.
DEBUG_VIDEO_PATH = "/dev/shm/debug_out.mp4"
DEBUG_VIDEO_SCALE = 0.25

LOG_TO_CSV = True
LOG_CSV_PATH = "/dev/shm/log_frames.csv"


# =============================================================================
# SEGUIMIENTO Y ZONAS (todavía no se usa en run_camara.py)
# =============================================================================

# Pares (pixel, angulo) medidos poniendo la pelota en cada borde de zona.
# Se llena con `python3 run_seguimiento.py --calibrar`. Vacía significa
# usar el modelo de lente nominal, que tiene error.
TABLA_CALIBRACION = []

FOV_HORIZONTAL = 100.0
ANGULO_CENTRO = 90.0

# LAS ZONAS SE MUDARON A config_hw.py. Este bloque definia 9 zonas de 15.6
# grados con permanencia de 1500 ms; el diseño actual son 7 SECTORES de 20
# grados con permanencia de 600 ms, y sus constantes viven en config_hw.py
# (SECTOR_DESDE, N_SECTORES, HISTERESIS_SECTOR_DEG, MS_PERMANENCIA,
# MS_MINIMO_ENTRE_MOVIMIENTOS, OMEGA_RAPIDA, OMEGA_LENTA).

FILTRO_ALFA = 0.4
