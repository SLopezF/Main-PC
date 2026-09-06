"""
state_machine.py

Máquina de estados SEARCH / TRACK con histéresis de confianza.

No conoce nada de Hailo, GStreamer, GPIO ni geometría de imágenes: solo
recibe una confianza de detección (y opcionalmente una coordenada) y
decide en qué estado hay que procesar el próximo frame.

Se prueba de forma completamente aislada, sin necesidad de video ni NPU.
"""

from dataclasses import dataclass, field
from enum import Enum

import config


class Mode(Enum):
    SEARCH = 0
    TRACK = 1


@dataclass
class TrackerState:
    """
    Estado mutable de la máquina. Se crea una instancia al arrancar el
    programa y se va actualizando frame a frame con update().
    """
    mode: Mode = Mode.SEARCH

    # Última coordenada válida conocida (en coordenadas del frame nativo).
    # None mientras nunca se detectó la pelota con confianza suficiente.
    last_x: float | None = None
    last_y: float | None = None

    # Contador de frames consecutivos "perdidos" mientras se está en TRACK
    # (confianza por debajo de CONF_EXIT_TRACK, o sin detección).
    frames_lost: int = field(default=0)

    def crop_center(self) -> tuple[float, float]:
        """
        Punto en el que hay que centrar el recorte para el próximo frame
        en modo TRACK. Solo tiene sentido llamarlo si mode == TRACK
        (en ese caso last_x/last_y siempre están seteados, porque no se
        puede entrar a TRACK sin haber tenido antes una detección válida).
        """
        assert self.last_x is not None and self.last_y is not None, (
            "crop_center() llamado sin una posición conocida previa"
        )
        return self.last_x, self.last_y


def update(
    state: TrackerState,
    confidence: float | None,
    global_x: float | None = None,
    global_y: float | None = None,
) -> TrackerState:
    """
    Actualiza la máquina de estados en base al resultado de un frame
    ya procesado (confianza de la detección, y su coordenada global si
    la hubo).

    `confidence` puede ser None si no hubo ninguna detección por encima
    de config.CONF_MIN_DETECTION (se trata igual que confianza 0.0 a
    los efectos de la histéresis).

    Devuelve la misma instancia de `state`, ya actualizada (se muta y
    se retorna por comodidad de encadenar llamadas / dejar el flujo
    explícito en quien la usa).
    """
    conf = confidence if confidence is not None else 0.0

    if state.mode == Mode.SEARCH:
        _update_from_search(state, conf, global_x, global_y)
    else:
        _update_from_track(state, conf, global_x, global_y)

    return state


def _update_from_search(
    state: TrackerState,
    conf: float,
    global_x: float | None,
    global_y: float | None,
) -> None:
    if conf >= config.CONF_ENTER_TRACK:
        # Detección lo suficientemente confiable: pasamos a TRACK.
        state.mode = Mode.TRACK
        state.last_x = global_x
        state.last_y = global_y
        state.frames_lost = 0
    # Si no alcanza el umbral, nos quedamos en SEARCH sin más cambios.
    # (last_x/last_y no se tocan: en SEARCH no se usan para nada, pero
    # se preservan por si hubo una detección previa de baja confianza
    # que en el futuro se quiera loguear o inspeccionar.)


def _update_from_track(
    state: TrackerState,
    conf: float,
    global_x: float | None,
    global_y: float | None,
) -> None:
    if conf >= config.CONF_EXIT_TRACK:
        # Detección aceptable: seguimos en TRACK, actualizamos posición
        # y reseteamos el contador de frames perdidos.
        state.last_x = global_x
        state.last_y = global_y
        state.frames_lost = 0
        return

    # Frame "perdido": confianza por debajo del umbral de salida.
    state.frames_lost += 1

    if state.frames_lost >= config.MAX_FRAMES_LOST:
        # Se superó la tolerancia: volvemos a SEARCH.
        state.mode = Mode.SEARCH
        state.frames_lost = 0
        # last_x/last_y se preservan (por si sirve como pista para
        # priorizar zonas de búsqueda en el futuro), pero no se usan
        # mientras estemos en SEARCH.
    # Si todavía no se superó la tolerancia, seguimos en TRACK, con el
    # crop centrado en la última posición conocida (no se actualiza
    # last_x/last_y con este frame perdido, a propósito).
