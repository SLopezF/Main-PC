"""
pipeline.py

El procesamiento de UN frame, compartido por `replay.py` (offline, sobre
material grabado) y `main_final.py` (en vivo, en la Pi).

POR QUE ESTA COMPARTIDO Y NO DUPLICADO
Todo el valor de `replay.py` es que corre EXACTAMENTE el mismo codigo de
decision que el sistema real. Si fueran dos implementaciones parecidas,
ajustar un umbral mirando el replay no diria nada sobre lo que va a pasar en
la cancha, y el banco de pruebas seria decorativo.

Por eso tambien las columnas del CSV viven aca: el analisis de resultados de
la tesis es un solo script para los dos.

QUE HACE Y QUE NO
Arma el tensor segun el modo que pide el tracker, infiere, decodifica los
candidatos, los remapea a pixeles del frame nativo y deja que el tracker
decida. Despues convierte a angulo del mundo y consulta al sectorizador.

NO mueve el motor ni toca la GoPro: devuelve una decision. Que hacer con ella
es del que llama, que en el replay no hace nada y en el main manda el
objetivo al hilo del motor.

OMEGA
El tracker trabaja en pixeles y el sectorizador en grados por segundo del
mundo. La conversion se hace aca, con dos angulos consecutivos y el dt real:
llevar la velocidad en pixeles a grados por segundo pasando por la derivada
de pixel_a_angulo seria mas elegante y mucho mas fragil, porque esa derivada
cambia con la calibracion y en los bordes de la lente.
"""

import time

import config
import geometria
import postprocess
import preproceso
from tracker import Mode

# Las mismas para replay.py y main_final.py (regla 8 del briefing).
COLUMNAS = ["n", "ts", "modo", "camara", "tile", "conf", "x", "y", "w", "h",
            "angulo", "sector", "flost", "aceptado_por",
            "pred_x", "pred_y", "err_pred", "vx", "vy", "motivo", "dist_ult",
            "omega", "objetivo_motor", "cambio_sector"]


class Candidato:
    """
    Deteccion en pixeles del frame NATIVO. El tracker solo pide .x .y .w .h y
    .confidence; se guarda .original para poder dibujar la caja despues, que
    sigue en coordenadas del tensor.
    """
    __slots__ = ("x", "y", "w", "h", "confidence", "original")

    def __init__(self, x, y, w, h, confidence, original):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.confidence, self.original = confidence, original


class CalculadorOmega:
    """
    Velocidad angular en grados por segundo, de dos angulos consecutivos.

    Se suaviza con un filtro exponencial porque omega es una derivada
    numerica sobre una posicion ruidosa: sin suavizar, un salto de deteccion
    de 3 grados en un frame de 25 ms se lee como 120 deg/s y dispara el
    regimen rapido del sectorizador sin que la pelota se haya movido.
    """

    def __init__(self, alfa: float | None = None):
        self.alfa = float(alfa if alfa is not None
                          else getattr(config, "FILTRO_ALFA", 0.4))
        self._angulo = None
        self._ts_ns = None
        self.omega = 0.0

    def actualizar(self, angulo: float | None, ts_ns: int) -> float:
        if angulo is None:
            return self.omega

        if self._angulo is not None and self._ts_ns is not None:
            dt = (ts_ns - self._ts_ns) / 1e9
            if dt > 1e-6:
                cruda = geometria.diferencia_angular(angulo, self._angulo) / dt
                self.omega = self.alfa * cruda + (1 - self.alfa) * self.omega

        self._angulo, self._ts_ns = angulo, ts_ns
        return self.omega

    def reiniciar(self) -> None:
        """En el handover de camara: los angulos son comparables, pero el
        salto de paralaje y de calibracion ensucia una derivada."""
        self._angulo = None
        self._ts_ns = None
        self.omega = 0.0


def procesar_frame(frame, info, hailo, model_hw, tr, sect=None, omega_calc=None):
    """
    Un frame de punta a punta.

    Devuelve (fila, deteccion_para_dibujar, to_global, resultado_tracker,
    resultado_sector, ms). `sect` y `omega_calc` son opcionales: sin ellos se
    hace solo deteccion y tracking, y las columnas de sector quedan vacias.

    Se usa process_candidates() y no process(): el tracker necesita VARIOS
    candidatos para elegir el mas cercano a la prediccion cuando la confianza
    es baja. Con uno solo, esa mitad del algoritmo no existe.
    """
    t0 = time.perf_counter()

    # Antes que nada: tapar la zona alta si hay mascara. Ver config.MASCARA_Y.
    preproceso.aplicar_mascara(frame)

    _, tile = tr.siguiente_entrada()

    if tr.modo == Mode.SEARCH:
        tensor, to_global = preproceso.build_search_input(frame, model_hw, tile)
    else:
        tensor, to_global = preproceso.build_track_input(
            frame, tr.crop_center(), model_hw)

    crudos = postprocess.process_candidates(
        hailo.infer(tensor), model_hw,
        umbral=getattr(config, "CONF_CANDIDATO", 0.05), topk=8)

    candidatos = []
    for d in crudos:
        gx, gy = to_global(d.x, d.y)
        candidatos.append(Candidato(gx, gy, d.w, d.h, d.confidence, d))

    r = tr.actualizar(candidatos, info["ts_ns"], info.get("camara", 0))
    ms = (time.perf_counter() - t0) * 1000.0

    ancho = frame.shape[1]
    angulo = (geometria.pixel_a_angulo(r.x, ancho, r.camara)
              if r.x is not None else None)

    omega = 0.0
    if omega_calc is not None:
        omega = omega_calc.actualizar(angulo, info["ts_ns"])

    rs = None
    if sect is not None and angulo is not None:
        rs = sect.actualizar(angulo, omega, info["ts_ns"] / 1e9)

    det_dibujo = None
    if r.aceptado:
        for c in candidatos:
            if c.x == r.x and c.y == r.y:
                det_dibujo = c.original
                break

    fila = {
        "n": info["seq"] - 1,
        "ts": info["ts_ns"] / 1e9,
        "modo": r.modo.name,
        "camara": r.camara,
        "tile": r.tile,
        "conf": f"{r.conf:.4f}" if r.conf is not None else "",
        "x": f"{r.x:.1f}" if r.x is not None else "",
        "y": f"{r.y:.1f}" if r.y is not None else "",
        "w": f"{r.w:.1f}" if r.w is not None else "",
        "h": f"{r.h:.1f}" if r.h is not None else "",
        "angulo": f"{angulo:.2f}" if angulo is not None else "",
        "sector": rs.sector if rs is not None else (
            sect.sector if sect is not None else ""),
        "flost": r.flost,
        "aceptado_por": r.motivo if r.aceptado else "",
        "pred_x": f"{r.pred_x:.1f}" if r.pred_x is not None else "",
        "pred_y": f"{r.pred_y:.1f}" if r.pred_y is not None else "",
        "err_pred": f"{r.err_pred:.1f}" if r.err_pred is not None else "",
        "vx": f"{r.vx:.0f}" if r.vx is not None else "",
        "vy": f"{r.vy:.0f}" if r.vy is not None else "",
        "motivo": r.motivo,
        "dist_ult": f"{r.dist_ult:.1f}" if r.dist_ult is not None else "",
        "omega": f"{omega:.1f}" if omega_calc is not None else "",
        "objetivo_motor": (f"{sect.angulo_objetivo:.1f}"
                           if sect is not None else ""),
        "cambio_sector": int(rs.cambio) if rs is not None else "",
    }
    return fila, det_dibujo, to_global, r, rs, ms
