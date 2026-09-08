"""
geometria.py

Tres cosas, todas sin hardware (se pueden testear en cualquier maquina):

  1. pixel -> angulo del mundo (0..180), por camara.
  2. angulo del mundo -> grados de motor, con limites y sentido.
  3. SelectorCamara: decide con que camara mirar, con histeresis, para que
     la pelota parada sobre el limite entre las dos no haga alternar.

CONVENCION DE ANGULOS
El "angulo del mundo" es el mismo eje para las dos camaras y para el motor:
0 a 180 grados, 90 al frente. Cada camara cubre un tramo centrado en
CAM_CENTRO_ANGULO[i] y de ancho CAM_FOV[i]; los tramos se solapan.

EL MODELO DE LENTE
pixel_a_angulo tiene tres caminos, por orden de prioridad:

  1. INTRINSECOS (ChArUco). Se desdistorsiona el pixel con K y los
     coeficientes, y el angulo respecto del eje optico es atan de la
     coordenada normalizada. Es la proyeccion REAL de la lente. Necesita
     ademas el yaw de la camara (a que angulo del mundo mira su eje optico),
     que se mide con calibrar.py.
  2. TABLA de pares (pixel, angulo) medidos a mano, si hay 3 o mas.
  3. FOV NOMINAL lineal. Es el respaldo, y asume proyeccion equidistante:
     medido contra los intrinsecos reales, se equivoca hasta 4.9 grados en el
     medio del cuadro, casi un cuarto de sector.

EL LUT
El camino 1 se precalcula UNA vez como una tabla de angulo por columna de
pixeles, y despues cada consulta es una interpolacion lineal. Asi cv2 no se
toca por frame (y este modulo se puede importar sin cv2 instalado).

El LUT se arma con y = cy, o sea la fila central. El angulo horizontal de un
punto depende un poco de su y por la distorsion, pero el motor solo hace pan y
los sectores son de 20 grados: esa dependencia esta muy por debajo del ruido.
"""

import bisect
import json
import os

import config_hw as chw


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# =============================================================================
# 0. Intrinsecos de las lentes
# =============================================================================

_INTRINSECOS = None          # {camara: dict} o {} si no hay archivo
_LUT: dict = {}              # {(camara, ancho): (xs, angulos)}


def intrinsecos(camara: int) -> dict | None:
    """
    fx, fy, cx, cy y coeficientes de distorsion de una camara, del archivo
    CAL_INTRINSECOS. None si no hay archivo o no esta esa camara.
    """
    global _INTRINSECOS
    if _INTRINSECOS is None:
        _INTRINSECOS = {}
        ruta = getattr(chw, "CAL_INTRINSECOS", None)
        if ruta and os.path.exists(ruta):
            try:
                with open(ruta, encoding="utf-8") as fh:
                    datos = json.load(fh)
                letras = getattr(chw, "CAM_LETRA_CALIBRACION", {})
                for cam, letra in letras.items():
                    c = datos.get("camaras", {}).get(letra)
                    if c:
                        _INTRINSECOS[int(cam)] = c
            except Exception as exc:
                print(f"[geometria] no pude leer '{ruta}': {exc}")
    return _INTRINSECOS.get(int(camara))


def yaw(camara: int) -> float:
    """
    Angulo del mundo al que mira el eje optico. Del CAM_YAW medido con
    calibrar.py; si no esta, se cae a CAM_CENTRO_ANGULO (estimado).
    """
    v = getattr(chw, "CAM_YAW", {}).get(int(camara))
    if v is None:
        return float(chw.CAM_CENTRO_ANGULO[int(camara)])
    return float(v)


def _lut(camara: int, ancho_img: int):
    """
    (xs, angulos) con el angulo respecto del EJE OPTICO de cada columna, ya
    desdistorsionado. Se calcula una vez por camara y ancho.
    """
    clave = (int(camara), int(ancho_img))
    if clave in _LUT:
        return _LUT[clave]

    c = intrinsecos(camara)
    if c is None:
        _LUT[clave] = None
        return None

    try:
        import math

        import cv2
        import numpy as np
    except Exception:
        _LUT[clave] = None
        return None

    # Los intrinsecos valen para la resolucion con la que se calibro. Si el
    # frame viene a otra, hay que escalarlos: fx, cx y compania son
    # proporcionales al ancho. Es un error silencioso clasico.
    escala = float(ancho_img) / float(c.get("ancho_px", ancho_img))
    K = np.array([[c["fx"] * escala, 0.0, c["cx"] * escala],
                  [0.0, c["fy"] * escala, c["cy"] * escala],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array(c["dist_coeffs"], dtype=np.float64).reshape(1, -1)

    xs = np.linspace(0.0, float(ancho_img - 1), 256)
    cy = c["cy"] * escala
    pts = np.array([[[x, cy]] for x in xs], dtype=np.float64)
    normalizados = cv2.undistortPoints(pts, K, dist)
    angulos = np.array([math.degrees(math.atan(p[0][0]))
                        for p in normalizados], dtype=np.float64)

    # np.interp exige xs creciente; angulos tiene que ser monotono para que la
    # inversa tambien funcione. Si no lo es, la calibracion esta rota.
    if not np.all(np.diff(angulos) > 0):
        print(f"[geometria] el mapa de la camara {camara} no es monotono: "
              f"la calibracion esta mal, uso el modelo lineal")
        _LUT[clave] = None
        return None

    _LUT[clave] = (xs, angulos)
    return _LUT[clave]


def usa_intrinsecos(camara: int, ancho_img: int) -> bool:
    return _lut(camara, ancho_img) is not None


# =============================================================================
# 1. Pixel -> angulo del mundo
# =============================================================================

def pixel_a_angulo(px: float, ancho_img: int, camara: int) -> float:
    """
    Convierte una coordenada X en pixeles del frame NATIVO al angulo del
    mundo (0..180) al que corresponde, para la camara dada.
    """
    lut = _lut(camara, ancho_img)
    if lut is not None:
        import numpy as np
        xs, angulos = lut
        off = float(np.interp(float(px), xs, angulos))
        if chw.CAM_ESPEJO.get(camara, False):
            off = -off
        return _clamp(yaw(camara) + off, 0.0, 180.0)

    tabla = chw.CAL_PIXEL_ANGULO.get(camara) or []
    if len(tabla) >= 3:
        return _interpolar_tabla(px, tabla)

    fov = float(chw.CAM_FOV[camara])
    centro = float(chw.CAM_CENTRO_ANGULO[camara])

    # -0.5 .. +0.5 respecto del centro de la imagen.
    u = (float(px) / float(ancho_img)) - 0.5
    if chw.CAM_ESPEJO.get(camara, False):
        u = -u

    return _clamp(centro + u * fov, 0.0, 180.0)


def angulo_a_pixel(angulo: float, ancho_img: int, camara: int) -> float:
    """
    Inversa de pixel_a_angulo. Sirve para dibujar y para sembrar el ROI en el
    handover de camara. Puede devolver un valor fuera de [0, ancho): eso
    significa que ese angulo NO se ve desde esta camara, y quien llama tiene
    que mirarlo.
    """
    lut = _lut(camara, ancho_img)
    if lut is not None:
        import numpy as np
        xs, angulos = lut
        off = float(angulo) - yaw(camara)
        if chw.CAM_ESPEJO.get(camara, False):
            off = -off
        if off <= angulos[0]:
            return -1.0
        if off >= angulos[-1]:
            return float(ancho_img) + 1.0
        return float(np.interp(off, angulos, xs))

    fov = float(chw.CAM_FOV[camara])
    centro = float(chw.CAM_CENTRO_ANGULO[camara])
    u = (float(angulo) - centro) / fov
    if chw.CAM_ESPEJO.get(camara, False):
        u = -u
    return (u + 0.5) * float(ancho_img)


def _interpolar_tabla(px: float, tabla) -> float:
    """Interpolacion lineal por tramos sobre pares (pixel, angulo) medidos."""
    pares = sorted((float(p), float(a)) for p, a in tabla)
    xs = [p for p, _ in pares]
    ys = [a for _, a in pares]

    if px <= xs[0]:
        i = 0
    elif px >= xs[-1]:
        i = len(xs) - 2
    else:
        i = bisect.bisect_right(xs, px) - 1

    x0, x1 = xs[i], xs[i + 1]
    y0, y1 = ys[i], ys[i + 1]
    if x1 == x0:
        return _clamp(y0, 0.0, 180.0)
    return _clamp(y0 + (y1 - y0) * (px - x0) / (x1 - x0), 0.0, 180.0)


def cobertura(camara: int, ancho_img: int | None = None) -> tuple[float, float]:
    """(angulo_min, angulo_max) que ve esta camara."""
    if ancho_img is None:
        import config
        ancho_img = int(getattr(config, "CAM_ANCHO", 2304))

    lut = _lut(camara, ancho_img)
    if lut is not None:
        _, angulos = lut
        a, b = float(angulos[0]), float(angulos[-1])
        if chw.CAM_ESPEJO.get(camara, False):
            a, b = -b, -a
        return (yaw(camara) + a, yaw(camara) + b)

    fov = float(chw.CAM_FOV[camara])
    c = float(chw.CAM_CENTRO_ANGULO[camara])
    return (c - fov / 2.0, c + fov / 2.0)


# =============================================================================
# 2. Angulo del mundo -> motor
# =============================================================================

def angulo_a_grados_motor(angulo: float) -> tuple[float, bool]:
    """
    Grados de motor (respecto de su cero) para apuntar a `angulo`.

    Devuelve (grados, clampeado). `clampeado` en True significa que el
    objetivo cae fuera del recorrido mecanico y se recorto: el motor va a
    quedar mirando a otro lado del que pide el detector, y eso hay que verlo
    en el log en vez de descubrirlo mirando el fierro.
    """
    crudo = (float(angulo) - chw.MOTOR_ANGULO_MUNDO_EN_CERO)
    crudo *= chw.MOTOR_SENTIDO * chw.RELACION_TRANSMISION

    limitado = _clamp(crudo, chw.MOTOR_GRADOS_MIN, chw.MOTOR_GRADOS_MAX)
    return limitado, (abs(limitado - crudo) > 1e-6)


def grados_motor_a_angulo(grados: float) -> float:
    """Inversa: a que angulo del mundo esta mirando el motor."""
    d = float(grados) / (chw.MOTOR_SENTIDO * chw.RELACION_TRANSMISION)
    return chw.MOTOR_ANGULO_MUNDO_EN_CERO + d


def encoder_esperado(grados_motor: float) -> float:
    """
    Que deberia marcar el encoder absoluto (0..360) con el motor en esos
    grados. Con el motor en 0 el encoder marca ENCODER_GRADOS_EN_MOTOR_CERO.
    Se usa para verificar el homing y cada movimiento.
    """
    eje = float(grados_motor) / chw.RELACION_TRANSMISION
    return (chw.ENCODER_GRADOS_EN_MOTOR_CERO + eje) % 360.0


def diferencia_angular(a: float, b: float) -> float:
    """Diferencia a-b en grados, llevada al rango -180..180."""
    d = (float(a) - float(b) + 180.0) % 360.0 - 180.0
    return d


# =============================================================================
# 3. Seleccion de camara con histeresis
# =============================================================================

class SelectorCamara:
    """
    Decide con que camara mirar. La regla:

      - la camara "natural" para un angulo es la de centro mas cercano;
      - solo se cambia si la otra es mejor por mas de HIST_CAMARA_GRADOS
        (margen angular) y esa condicion se sostiene MS_CONFIRMAR_CAMBIO ms
        (permanencia), y si paso MS_MINIMO_ENTRE_CAMBIOS desde el ultimo;
      - si la camara activa no ve nada por MS_PERDIDA_CAMARA, se cambia a la
        otra a barrer, sin esperar los votos.

    Por que las tres condiciones y no una: el margen solo no alcanza porque
    el ruido de la deteccion (la pelota "salta" varios grados entre frames)
    puede superarlo; la permanencia sola tampoco, porque una pelota que se
    queda justo sobre el limite cumple la condicion todo el tiempo.

    No toca hardware ni sabe de picamera2: se testea con timestamps a mano.
    """

    def __init__(self, camara_inicial: int = 0):
        self.activa = int(camara_inicial)
        self._t_voto = None          # cuando arranco el voto sostenido
        self._voto_a = None          # a que camara vota
        self._t_ultimo_cambio = -1e9
        self._t_ultima_deteccion = None
        self.motivo = "inicial"

    # -- consultas ------------------------------------------------------------
    @staticmethod
    def mejor_camara(angulo: float) -> int:
        """Camara cuyo centro esta mas cerca del angulo, sin histeresis."""
        return min(
            chw.CAMARAS,
            key=lambda c: abs(diferencia_angular(angulo, chw.CAM_CENTRO_ANGULO[c])),
        )

    def _ventaja(self, angulo: float, candidata: int) -> float:
        """Cuantos grados mejor centrada queda `candidata` que la activa."""
        d_act = abs(diferencia_angular(angulo, chw.CAM_CENTRO_ANGULO[self.activa]))
        d_can = abs(diferencia_angular(angulo, chw.CAM_CENTRO_ANGULO[candidata]))
        return d_act - d_can

    # -- actualizacion --------------------------------------------------------
    def actualizar(
        self,
        angulo: float | None,
        confianza: float | None,
        t_s: float,
    ) -> int:
        """
        Se llama una vez por deteccion (o por frame sin deteccion, con
        angulo=None). Devuelve el indice de la camara que hay que usar para
        el proximo frame.
        """
        # --- sin deteccion: solo corre el reloj de perdida
        if angulo is None or confianza is None:
            self._voto_a, self._t_voto = None, None
            if self._t_ultima_deteccion is None:
                self._t_ultima_deteccion = t_s
            perdida_ms = (t_s - self._t_ultima_deteccion) * 1000.0
            if (perdida_ms >= chw.MS_PERDIDA_CAMARA
                    and self._puede_cambiar(t_s)):
                self._cambiar_a(self._otra(), t_s, f"perdida {perdida_ms:.0f} ms")
            else:
                self.motivo = f"sin deteccion hace {perdida_ms:.0f} ms"
            return self.activa

        self._t_ultima_deteccion = t_s

        # --- deteccion floja: cuenta como "la veo", pero no vota cambio
        if confianza < chw.CONF_CAMBIO_CAMARA:
            self._voto_a, self._t_voto = None, None
            return self.activa

        candidata = self.mejor_camara(angulo)
        if candidata == self.activa:
            self._voto_a, self._t_voto = None, None
            self.motivo = "centrada"
            return self.activa

        # --- la otra esta mejor: exigir margen
        if self._ventaja(angulo, candidata) < chw.HIST_CAMARA_GRADOS:
            self._voto_a, self._t_voto = None, None
            self.motivo = "en el limite, sin margen"
            return self.activa

        # --- exigir permanencia
        if self._voto_a != candidata:
            self._voto_a, self._t_voto = candidata, t_s
            self.motivo = "votando cambio"
            return self.activa

        sostenido_ms = (t_s - self._t_voto) * 1000.0
        if sostenido_ms < chw.MS_CONFIRMAR_CAMBIO:
            self.motivo = f"votando cambio ({sostenido_ms:.0f} ms)"
            return self.activa

        if not self._puede_cambiar(t_s):
            self.motivo = "cambio reciente, se espera"
            return self.activa

        self._cambiar_a(candidata, t_s, f"angulo {angulo:.1f} sostenido")
        return self.activa

    # -- internos -------------------------------------------------------------
    def _otra(self) -> int:
        for c in chw.CAMARAS:
            if c != self.activa:
                return c
        return self.activa

    def _puede_cambiar(self, t_s: float) -> bool:
        return ((t_s - self._t_ultimo_cambio) * 1000.0
                >= chw.MS_MINIMO_ENTRE_CAMBIOS)

    def _cambiar_a(self, camara: int, t_s: float, motivo: str) -> None:
        self.activa = int(camara)
        self._t_ultimo_cambio = t_s
        self._voto_a, self._t_voto = None, None
        self._t_ultima_deteccion = t_s
        self.motivo = motivo

    def forzar(self, camara: int, t_s: float) -> int:
        """Cambio manual (tecla 0/1 en el debug). Saltea la histeresis."""
        self._cambiar_a(camara, t_s, "manual")
        return self.activa
