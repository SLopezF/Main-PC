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

EL MODELO DE LENTE ES APROXIMADO
pixel -> angulo lineal asume proyeccion equidistante. Una lente real tiene
distorsion de barril: cerca del borde el error puede ser de varios grados.
Por eso existe CAL_PIXEL_ANGULO: con 3 o mas pares medidos, se interpola y
el FOV nominal se ignora.
"""

import bisect

import config_hw as chw


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# =============================================================================
# 1. Pixel -> angulo del mundo
# =============================================================================

def pixel_a_angulo(px: float, ancho_img: int, camara: int) -> float:
    """
    Convierte una coordenada X en pixeles del frame NATIVO al angulo del
    mundo (0..180) al que corresponde, para la camara dada.
    """
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
    """Inversa de pixel_a_angulo, con el modelo nominal. Sirve para dibujar."""
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


def cobertura(camara: int) -> tuple[float, float]:
    """(angulo_min, angulo_max) que ve esta camara, segun el modelo nominal."""
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
