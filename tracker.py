"""
tracker.py

El algoritmo de decision. Reemplaza a state_machine.py, que solo miraba la
confianza del frame actual y no tenia memoria de por donde venia la pelota.

QUE AGREGA RESPECTO DE state_machine.py
Memoria, y con ella tres decisiones nuevas:

  1. PREDICE. Un Kalman de velocidad constante dice donde deberia estar la
     pelota en este frame. El ROI de TRACK se centra ahi en vez de en la
     ultima posicion vista.
  2. ELIGE POR CERCANIA CUANDO LA CONFIANZA ES BAJA. Si el mejor candidato
     no llega a CONF_ALTA, se toma el MAS CERCANO A LA PREDICCION entre los
     que pasan CONF_BAJA, no el mas confiado. Ese es el motivo entero por el
     que postprocess.process_candidates() devuelve topk: el pico mas fuerte
     del frame puede ser una zapatilla blanca mientras la pelota real esta en
     el segundo candidato.
  3. RECHAZA LO IMPOSIBLE. Un candidato que aparece mas lejos que
     radio_gate() de la ultima posicion aceptada no puede ser la misma
     pelota, por mas confianza que traiga.

SIN HARDWARE Y SIN cv2
Python puro. No importa cv2, ni hailo, ni postprocess. Los candidatos se
reciben como objetos con .x, .y, .w, .h y .confidence (duck typing), asi que
los tests usan detecciones inventadas y no hace falta ni video ni NPU.

EL ROI SE CONGELA EN LA ULTIMA POSICION ACEPTADA, NO EN LA PREDICCION
Parece un detalle y no lo es. Si el filtro esta siguiendo un fantasma, un ROI
que persigue la prediccion se va del frame y no vuelve nunca: cada frame sin
correccion lo empuja mas lejos en la direccion equivocada. Congelado en la
ultima posicion REAL, el ROI se queda donde la pelota estuvo de verdad, que
es el mejor lugar para volver a encontrarla.

COORDENADAS
Todo en pixeles del frame NATIVO de la camara activa. Se eligen pixeles y no
angulos a proposito: en pixeles el tamano de la caja da informacion de
distancia (y de ahi sale el gate) y la precision no se degrada en los bordes
del FOV, donde la lente tiene distorsion de barril.

EL HANDOVER DE CAMARA REINICIA EL FILTRO
Las dos camaras miran a distinto lado: un (x, y) de la camara 0 no significa
nada en la camara 1. Al cambiar, el filtro se reinicia y quien llama tiene
que sembrar el ROI nuevo con geometria.angulo_a_pixel(). Perder el filtro por
un frame en cada handover esta decidido y es aceptable: el ROI es un cuarto
del frame, un error de decenas de pixeles no cambia nada.

USO
    tr = Tracker(n_tiles=4, fn_tile=lambda x, y: ...)
    r = tr.actualizar(candidatos, info["ts_ns"], camara)
    if r.aceptado:
        ...
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

import config


class Mode(Enum):
    """
    Compatible con el Mode que gpio_timer.py importa hoy de state_machine.py:
    mismos nombres y mismos valores. gpio_timer compara contra Mode.TRACK para
    fijar el nivel del pin, asi que los nombres NO se pueden cambiar.
    """
    SEARCH = 0
    TRACK = 1


@dataclass
class ResultadoFrame:
    """
    Todo lo que el tracker decidio para un frame. Es lo que se escribe en el
    CSV y lo que main_final.py usa para apuntar el motor.
    """
    modo: Mode                       # el modo con el que se PROCESO este frame
    aceptado: bool
    motivo: str                      # "confianza", "kalman", "sin candidato",
                                     # "delta imposible", "fuera del gate"

    # Deteccion aceptada, en pixeles del frame nativo. None si no se acepto.
    x: float | None = None
    y: float | None = None
    w: float | None = None
    h: float | None = None
    conf: float | None = None

    # Prediccion del Kalman ANTES de ver este frame, y el residuo respecto de
    # la deteccion que finalmente se acepto. Es la señal con la que se tunean
    # KALMAN_SIGMA_ACEL y KALMAN_SIGMA_MEDICION: ver el comentario en
    # config.py. None cuando el filtro todavia no arranco.
    pred_x: float | None = None
    pred_y: float | None = None
    err_pred: float | None = None

    # Velocidad estimada, px/s. main_final.py la convierte a omega (grados por
    # segundo del mundo) para el regimen rapido/lento de sectores.
    vx: float | None = None
    vy: float | None = None

    # Donde centrar el ROI del proximo frame de TRACK.
    roi_x: float | None = None
    roi_y: float | None = None

    # Distancia entre el candidato mirado y la ultima posicion ACEPTADA. Es
    # contra esto que se aplica el gate, asi que sirve para saber si el gate
    # esta rechazando por buenos motivos o porque el material no es continuo.
    dist_ult: float | None = None

    flost: int = 0
    camara: int = 0
    tile: int = -1                   # cuadrante barrido; -1 en TRACK
    camara_sugerida: int = 0         # con cual mirar el proximo frame


# =============================================================================
# Kalman de velocidad constante
# =============================================================================

class KalmanCV:
    """
    Estado (x, y, vx, vy), modelo de velocidad constante, dt REAL.

    Escrito a mano con numpy y no con filterpy: son cuatro estados y dos
    mediciones, la libreria seria una dependencia mas para dieciseis lineas de
    algebra, y aca conviene poder leer exactamente que hace.

    El dt se pasa en cada predict() y no se asume 1/fps: camera_source.py
    descarta frames con politica de ultimo-frame-gana, asi que el dt entre dos
    frames ENTREGADOS es variable. Con dt fijo, el filtro subestima el
    desplazamiento justo cuando el sistema va lento, que es cuando mas
    necesita acertar.
    """

    def __init__(self, sigma_acel: float | None = None,
                 sigma_medicion: float | None = None):
        self.sigma_acel = float(
            sigma_acel if sigma_acel is not None else config.KALMAN_SIGMA_ACEL)
        self.sigma_medicion = float(
            sigma_medicion if sigma_medicion is not None
            else config.KALMAN_SIGMA_MEDICION)

        self.x = None            # vector de estado (4,)
        self.P = None            # covarianza (4, 4)

        # Matriz de medicion: se observan posiciones, no velocidades.
        self.H = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]])
        self.R = np.eye(2) * (self.sigma_medicion ** 2)

    @property
    def iniciado(self) -> bool:
        return self.x is not None

    def reiniciar(self, x: float, y: float,
                  vx: float = 0.0, vy: float = 0.0) -> None:
        """
        Arranca (o rearranca) el filtro en una posicion conocida. La
        incertidumbre inicial de la velocidad es alta a proposito: no se sabe
        nada de hacia donde va, y conviene que las primeras mediciones pesen
        mucho en vez de que el filtro insista con vx=0.
        """
        self.x = np.array([float(x), float(y), float(vx), float(vy)])
        self.P = np.diag([
            self.sigma_medicion ** 2,
            self.sigma_medicion ** 2,
            (self.sigma_acel * 0.1) ** 2,
            (self.sigma_acel * 0.1) ** 2,
        ])

    def olvidar(self) -> None:
        self.x = None
        self.P = None

    def predecir(self, dt: float) -> tuple[float, float] | None:
        """
        Avanza el estado dt segundos y devuelve (x, y) predicho. None si el
        filtro no arranco todavia.

        El ruido de proceso Q modela una aceleracion desconocida constante
        durante dt: una pelota que rebota o que patean cambia de velocidad de
        golpe, y el modelo de velocidad constante no lo ve venir. Sin este
        termino el filtro se vuelve cada vez mas confiado en una velocidad que
        ya no es cierta.
        """
        if self.x is None:
            return None
        dt = max(0.0, float(dt))

        F = np.array([[1.0, 0.0, dt, 0.0],
                      [0.0, 1.0, 0.0, dt],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])

        s2 = self.sigma_acel ** 2
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        q_pp, q_pv, q_vv = dt4 / 4.0, dt3 / 2.0, dt2
        Q = np.array([[q_pp, 0.0, q_pv, 0.0],
                      [0.0, q_pp, 0.0, q_pv],
                      [q_pv, 0.0, q_vv, 0.0],
                      [0.0, q_pv, 0.0, q_vv]]) * s2

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return float(self.x[0]), float(self.x[1])

    def actualizar(self, x: float, y: float) -> None:
        """Corrige el estado con una medicion. Si no arranco, arranca ahi."""
        if self.x is None:
            self.reiniciar(x, y)
            return

        z = np.array([float(x), float(y)])
        y_res = z - self.H @ self.x                  # residuo
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)     # ganancia

        self.x = self.x + K @ y_res
        self.P = (np.eye(4) - K @ self.H) @ self.P

    @property
    def velocidad(self) -> tuple[float, float]:
        if self.x is None:
            return 0.0, 0.0
        return float(self.x[2]), float(self.x[3])


# =============================================================================
# Gate de plausibilidad
# =============================================================================

def _periodo_nominal() -> float:
    """Segundos entre frames a los que esta referido GATE_DIAMETROS."""
    fps = float(getattr(config, "CAM_FPS", 40.0)) or 40.0
    return 1.0 / fps


def radio_gate(det, segundos: float | None = None) -> float:
    """
    Cuanto se puede haber movido la pelota en `segundos`, en pixeles.

    max(GATE_PX_MIN, GATE_DIAMETROS * max(w, h)) escalado por el tiempo REAL
    transcurrido. El 6 sale de que a 50 m/s la pelota se desplaza 1.25 m en
    25 ms (un frame a 40 fps) y una pelota nro 4 mide 0.21 m: el cociente es
    5.95 y NO depende de la distancia, porque desplazamiento aparente y
    diametro aparente escalan los dos con 1/d.

    POR QUE EL TIEMPO Y NO EL NUMERO DE FRAMES. El gate es fisica por unidad
    de TIEMPO, no por frame. Con `segundos=None` se asume el periodo nominal
    de config.CAM_FPS, que es lo correcto en vivo. Pero con material capturado
    a otra tasa, un gate clavado a 40 fps rechaza detecciones buenas en masa.

    Medido: sobre imagenes/Soleado (continuas a ~40 fps) el desplazamiento
    mediano entre frames es 13 px y el gate de 120 px sobra. Sobre
    imagenes/test_kalman el mediano es 142-210 px: el MISMO gate rechazaba el
    38-45% de los frames y la deteccion aceptada caia de 88% a 42%. No era la
    pelota moviendose a 60 m/s: eran imagenes mas espaciadas en el tiempo.
    """
    lado = max(float(getattr(det, "w", 0.0)), float(getattr(det, "h", 0.0)))
    base = max(float(config.GATE_PX_MIN), float(config.GATE_DIAMETROS) * lado)
    if segundos is None:
        return base
    return base * max(1.0, float(segundos) / _periodo_nominal())


def radio_gate_acumulado(det, frames: int, segundos: float | None = None) -> float:
    """
    El gate contra una posicion vieja.

    Se prefiere `segundos` (tiempo REAL desde la ultima aceptada) y se cae a
    `frames` solo si no hay timestamps. Las dos formas dicen lo mismo cuando
    el material corre a config.CAM_FPS; con otra tasa, solo el tiempo acierta.

    POR QUE ESTO NO ES UN DETALLE. Sin el factor acumulado, un rechazo
    marginal se vuelve permanente: al rechazar, la ultima posicion se congela,
    la pelota sigue viaje, y el frame siguiente esta MAS lejos todavia. Cada
    rechazo hace mas probable el proximo. Medido sobre imagenes/Soleado: un
    primer rechazo de 123 px contra un gate de 120 encadenaba
    123 -> 216 -> 300 -> 377 -> 453 -> 516 -> 571 px, diez frames perdidos,
    hasta que FLOST_A_SEARCH forzaba la vuelta a SEARCH. Eran 44 frames
    (14% del total) tirados por 3 px de diferencia en el primero.
    """
    if segundos is not None:
        return radio_gate(det, segundos)
    return radio_gate(det) * (1 + max(0, int(frames)))


def _dist(ax, ay, bx, by) -> float:
    return float(np.hypot(ax - bx, ay - by))


# =============================================================================
# Patron de SEARCH
# =============================================================================

class PatronBusqueda:
    """
    Orden en el que se barren los cuadrantes mientras se busca.

    La regla: SEARCH_BARRIDOS_ACTIVA barridos COMPLETOS (los n_tiles
    cuadrantes) en la camara donde se vio la pelota por ultima vez, despues
    SEARCH_BARRIDOS_OTRA en la otra, y vuelta a empezar.

    La asimetria (3 contra 1) es a proposito: lo mas probable es que la pelota
    siga por donde estaba. Y cada barrido arranca por el cuadrante donde se la
    vio, no por el 0, porque la pelota estaba ahi hace milisegundos: eso pasa
    el caso comun de hasta 4 inferencias de barrido a 1.
    """

    def __init__(self, n_tiles: int = 4, camaras=(0, 1)):
        self.n_tiles = int(n_tiles)
        self.camaras = tuple(camaras)
        self._camara_base = self.camaras[0]
        self._tile_inicial = 0
        self._i = 0                  # cuantos cuadrantes se barrieron ya

    def sembrar(self, camara: int, tile_inicial: int = 0) -> None:
        """Arranca el patron en la camara y el cuadrante donde se la vio."""
        self._camara_base = int(camara)
        self._tile_inicial = int(tile_inicial) % self.n_tiles
        self._i = 0

    def _otra(self, camara: int) -> int:
        for c in self.camaras:
            if c != camara:
                return c
        return camara

    def actual(self) -> tuple[int, int]:
        """(camara, tile) que toca mirar ahora."""
        por_activa = config.SEARCH_BARRIDOS_ACTIVA * self.n_tiles
        por_otra = config.SEARCH_BARRIDOS_OTRA * self.n_tiles
        ciclo = por_activa + por_otra

        pos = self._i % ciclo
        if pos < por_activa:
            camara = self._camara_base
        else:
            camara = self._otra(self._camara_base)

        tile = (self._tile_inicial + pos) % self.n_tiles
        return camara, tile

    def avanzar(self) -> None:
        self._i += 1


# =============================================================================
# Tracker
# =============================================================================

class Tracker:
    """
    Maquina de estados SEARCH / TRACK con Kalman y gate de plausibilidad.

    `fn_tile(x, y) -> int` mapea una posicion en pixeles nativos al indice del
    cuadrante que la contiene. Se INYECTA en vez de importarse para que este
    modulo no dependa de preproceso (que necesita conocer el tamano del frame
    y del modelo). En replay.py y en main_final.py se pasa
    `lambda x, y: preproceso.tile_para_punto(frame_hw, model_hw, x, y)`.
    """

    def __init__(self, n_tiles: int = 4, camaras=(0, 1),
                 camara_inicial: int = 0, fn_tile=None,
                 usar_kalman: bool = True, usar_gate: bool = True):
        # Los dos interruptores existen para poder medir por separado cuanto
        # aporta cada mecanismo, no para apagarlos en produccion. Apagar los
        # dos deja una maquina de estados equivalente a state_machine.py, que
        # es la linea de base contra la que se compara.
        self.usar_kalman = bool(usar_kalman)
        self.usar_gate = bool(usar_gate)

        self.modo = Mode.SEARCH
        self.camara = int(camara_inicial)
        self.flost = 0

        self.kalman = KalmanCV()
        self.patron = PatronBusqueda(n_tiles=n_tiles, camaras=camaras)
        self._fn_tile = fn_tile

        # Ultima posicion ACEPTADA. Es la que congela el ROI, y contra la que
        # se mide el salto imposible. No es la prediccion del filtro.
        self.ultima_x = None
        self.ultima_y = None
        self.ultima_camara = int(camara_inicial)

        self._ts_ns_previo = None
        self.ts_ns_ultima_deteccion = None
        self._ts_ns_actual = None

    # ------------------------------------------------------------- consultas
    def crop_center(self) -> tuple[float, float]:
        """
        Donde centrar el ROI del proximo frame de TRACK: SIEMPRE la ultima
        posicion ACEPTADA.

        NO la prediccion del Kalman, y esto se probo con datos. Al centrar el
        ROI en la prediccion, sobre imagenes/Soleado la deteccion estando en
        TRACK cayo de 91.0% a 73.4% y las caidas a SEARCH pasaron de 3 a 6: con
        un residuo p95 de 93 px (max 199), la prediccion se equivoca lo
        suficiente como para dejar la pelota fuera del recorte, y cada frame
        sin correccion empuja el filtro mas lejos en la direccion equivocada.

        La prediccion sirve para ELEGIR Y FILTRAR candidatos, que es donde
        importan los pixeles. El ROI mide 1152x640: no necesita precision,
        necesita no equivocarse.
        """
        if self.ultima_x is None:
            raise RuntimeError("crop_center() sin una posicion conocida previa")
        return float(self.ultima_x), float(self.ultima_y)

    def segundos_sin_deteccion(self, ts_ns: int) -> float:
        if self.ts_ns_ultima_deteccion is None:
            return 0.0
        return (ts_ns - self.ts_ns_ultima_deteccion) / 1e9

    def siguiente_entrada(self) -> tuple[int, int]:
        """
        (camara, tile) con los que armar el proximo frame. tile == -1 en
        TRACK, donde el recorte lo decide crop_center().
        """
        if self.modo == Mode.TRACK:
            return self.camara, -1
        return self.patron.actual()

    def cambiar_camara(self, camara: int, x=None, y=None) -> None:
        """
        Handover. El filtro se reinicia porque un (x, y) de una camara no
        significa nada en la otra. Si se pasa una posicion sembrada (la que
        sale de geometria.angulo_a_pixel para la camara nueva), el ROI arranca
        ahi; si no, se cae a SEARCH.
        """
        self.camara = int(camara)
        self.kalman.olvidar()
        if x is None or y is None:
            self.modo = Mode.SEARCH
            self.patron.sembrar(self.camara, 0)
        else:
            self.ultima_x, self.ultima_y = float(x), float(y)
            self.ultima_camara = int(camara)
            self.kalman.reiniciar(x, y)
        self.flost = 0

    # ---------------------------------------------------------- actualizacion
    def actualizar(self, candidatos, ts_ns: int,
                   camara: int | None = None) -> ResultadoFrame:
        """
        Un frame. `candidatos` es lo que devuelve
        postprocess.process_candidates(): objetos con .x, .y, .w, .h y
        .confidence, en pixeles del frame NATIVO, ordenados por confianza.
        """
        if camara is not None:
            self.camara = int(camara)

        dt = 0.0
        if self._ts_ns_previo is not None:
            dt = max(0.0, (ts_ns - self._ts_ns_previo) / 1e9)
        self._ts_ns_previo = ts_ns
        self._ts_ns_actual = ts_ns

        modo_previo = self.modo
        _, tile = self.siguiente_entrada()

        if modo_previo == Mode.TRACK:
            r = self._paso_track(candidatos, dt, ts_ns)
        else:
            r = self._paso_search(candidatos, ts_ns, tile)

        r.modo = modo_previo
        r.camara = self.camara
        r.flost = self.flost
        r.camara_sugerida, _ = self.siguiente_entrada()
        if self.ultima_x is not None:
            try:
                r.roi_x, r.roi_y = self.crop_center()
            except RuntimeError:
                pass
        return r

    # ------------------------------------------------------------- internos
    def _seg_desde_ultima(self) -> float | None:
        """Tiempo real desde la ultima deteccion aceptada, en segundos."""
        if self.ts_ns_ultima_deteccion is None or self._ts_ns_actual is None:
            return None
        return max(0.0, (self._ts_ns_actual - self.ts_ns_ultima_deteccion) / 1e9)

    def _paso_track(self, candidatos, dt: float, ts_ns: int) -> ResultadoFrame:
        # Con usar_kalman=False el filtro se sigue actualizando (para poder
        # reportar el residuo y comparar), pero su prediccion NO se usa para
        # elegir: la eleccion cae al candidato mas confiado, como hacia
        # state_machine.py.
        pred_real = self.kalman.predecir(dt)
        pred = pred_real if self.usar_kalman else None
        px, py = pred if pred is not None else (None, None)

        elegido, motivo = None, "sin candidato"

        if candidatos:
            mejor = max(candidatos, key=lambda c: c.confidence)

            if mejor.confidence >= config.CONF_ALTA:
                elegido, motivo = mejor, "confianza"

            elif mejor.confidence >= config.CONF_BAJA and pred is not None:
                # El MAS CERCANO a la prediccion entre los que pasan el piso,
                # no el mas confiado: el pico mas fuerte del frame puede ser
                # una zapatilla blanca mientras la pelota esta en el segundo.
                flojos = [c for c in candidatos
                          if c.confidence >= config.CONF_BAJA]
                cerca = min(flojos, key=lambda c: _dist(c.x, c.y, px, py))
                # Contra la PREDICCION el radio tambien se afloja con los
                # frames sin corregir: el filtro extrapola, pero su error
                # crece cuanto mas hace que no ve una medicion.
                if _dist(cerca.x, cerca.y, px, py) < radio_gate_acumulado(
                        cerca, self.flost, self._seg_desde_ultima()):
                    elegido, motivo = cerca, "kalman"
                else:
                    motivo = "fuera del gate"

            elif mejor.confidence >= config.CONF_BAJA:
                # Sin prediccion (primer frame tras reiniciar el filtro) no hay
                # con que decidir cercania: se acepta el mas confiado.
                elegido, motivo = mejor, "kalman"

        # Salto imposible contra la ultima posicion ACEPTADA. Se mira aunque
        # la confianza sea alta: la fisica no negocia con el clasificador.
        dist_ult = None
        if elegido is not None and self.ultima_x is not None:
            dist_ult = _dist(elegido.x, elegido.y,
                             self.ultima_x, self.ultima_y)
            # El gate crece con la antiguedad de la ultima posicion aceptada.
            if (self.usar_gate and dist_ult > radio_gate_acumulado(
                    elegido, self.flost, self._seg_desde_ultima())):
                elegido, motivo = None, "delta imposible"

        err = None
        if elegido is not None:
            # El residuo se reporta SIEMPRE, aunque la prediccion no se use
            # para elegir: asi las dos corridas son comparables.
            if pred_real is not None:
                err = _dist(elegido.x, elegido.y, pred_real[0], pred_real[1])
            self.kalman.actualizar(elegido.x, elegido.y)
            self.ultima_x, self.ultima_y = float(elegido.x), float(elegido.y)
            self.ultima_camara = self.camara
            self.ts_ns_ultima_deteccion = ts_ns
            self.flost = 0
        else:
            self.flost += 1
            if self.flost >= config.FLOST_A_SEARCH:
                self.modo = Mode.SEARCH
                self.kalman.olvidar()
                tile0 = (self._fn_tile(self.ultima_x, self.ultima_y)
                         if self._fn_tile and self.ultima_x is not None else 0)
                self.patron.sembrar(self.ultima_camara, tile0)

        vx, vy = self.kalman.velocidad
        return ResultadoFrame(
            modo=Mode.TRACK,
            aceptado=elegido is not None,
            motivo=motivo,
            x=elegido.x if elegido else None,
            y=elegido.y if elegido else None,
            w=elegido.w if elegido else None,
            h=elegido.h if elegido else None,
            conf=elegido.confidence if elegido else None,
            pred_x=px, pred_y=py, err_pred=err, dist_ult=dist_ult,
            vx=vx, vy=vy, tile=-1,
        )

    def _paso_search(self, candidatos, ts_ns: int, tile: int) -> ResultadoFrame:
        self.patron.avanzar()

        elegido = None
        if candidatos:
            mejor = max(candidatos, key=lambda c: c.confidence)
            if mejor.confidence >= config.CONF_ALTA:
                elegido = mejor

        if elegido is None:
            return ResultadoFrame(
                modo=Mode.SEARCH, aceptado=False,
                motivo="sin candidato", tile=tile,
            )

        # Para entrar a TRACK se exige CONF_ALTA y nada mas: el gate no aplica
        # porque no hay posicion previa contra la cual comparar (y si la hay,
        # es vieja y no dice nada).
        self.modo = Mode.TRACK
        self.kalman.reiniciar(elegido.x, elegido.y)
        self.ultima_x, self.ultima_y = float(elegido.x), float(elegido.y)
        self.ultima_camara = self.camara
        self.ts_ns_ultima_deteccion = ts_ns
        self.flost = 0

        return ResultadoFrame(
            modo=Mode.SEARCH, aceptado=True, motivo="confianza",
            x=elegido.x, y=elegido.y, w=elegido.w, h=elegido.h,
            conf=elegido.confidence, tile=tile, vx=0.0, vy=0.0,
        )