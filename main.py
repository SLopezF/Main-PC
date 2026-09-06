"""
main.py

Loop principal del sistema. Por cada frame del video de entrada:

  1. Arranca la medición de tiempo (pin GPIO de timing).
  2. Según el estado actual (SEARCH/TRACK), arma la imagen que se le
     manda a la NPU (un tile del mosaico, o crop centrado).
  3. Corre inferencia síncrona en la Hailo.
  4. Post-procesa las 6 salidas -> detección (o None).
  5. Reproyecta la coordenada local a coordenadas del frame nativo.
  6. Apaga el pin GPIO de timing (fin de la medición del frame).
  7. Actualiza la máquina de estados con el resultado del frame.
  8. (Opcional) loguea el frame en un CSV.

DOS DECISIONES DE DISEÑO QUE VIENEN DE LA CAMPAÑA DE DEBUG:

1) Los canales van en RGB, no en BGR.
   OpenCV entrega BGR. El modelo responde consistentemente mejor con
   RGB: en el mismo crop, 0.879 contra 0.859; en frame reducido, 0.476
   contra 0.061. Es un factor 8 en el caso peor.

2) SEARCH es un mosaico, no un downscale del frame entero.
   Se midió cómo cae la confianza al reducir el objeto:
       1.00x -> 0.879     0.50x -> 0.720     0.25x -> 0.0025
   Reducir el frame nativo (4608x2592) a la entrada del modelo
   (1152x640) implica 0.25x, donde el modelo directamente no ve la
   pelota. Y como la máquina de estados arranca en SEARCH y solo pasa
   a TRACK con una detección, el modo que sí funciona era inalcanzable.
   La solución es recorrer el frame con una grilla de tiles a ~0.5x.
   Con SEARCH_TILE_GRID = (2, 2): escala efectiva 0.494, cuatro tiles.

   Los tiles se rotan de a uno por frame en vez de correr los cuatro
   juntos: así el costo por frame en SEARCH es una sola inferencia,
   igual que en TRACK, y la medición del pin de GPIO sigue siendo
   comparable entre los dos modos. El precio es tardar hasta N frames
   en enganchar la pelota en vez de uno.

Import tardío de cv2 y de hailo_inference dentro de main() para que el
resto del código (geometry, postprocess, state_machine, gpio_timer) se
pueda importar y testear en cualquier máquina, sin necesitar OpenCV ni
el SDK de HailoRT instalados.
"""

import csv
import time
from typing import Callable

import numpy as np

import config
import gpio_timer
import postprocess
import state_machine
from state_machine import Mode, TrackerState

# Una función que mapea (x, y) en coordenadas de la imagen que vio la NPU
# a (x, y) en coordenadas del frame nativo.
ToGlobal = Callable[[float, float], tuple[float, float]]

_tiles_cache: dict[tuple, list[tuple[int, int, int, int]]] = {}


def get_model_hw(hailo) -> tuple[int, int]:
    """
    Devuelve (alto, ancho) esperado por el HEF. Se lee del objeto de
    inferencia para que el código no dependa de que config esté
    sincronizado con el modelo compilado.
    """
    shape = getattr(hailo, "input_shape", None)
    if shape is None:
        shape = getattr(config, "MODEL_INPUT_SIZE", None)

    if shape is None:
        raise RuntimeError(
            "No pude determinar el input shape del modelo: ni hailo.input_shape "
            "ni config.MODEL_INPUT_SIZE están disponibles."
        )
    if isinstance(shape, int):
        raise RuntimeError(
            f"El input del modelo es rectangular, pero se configuró como un "
            f"escalar ({shape}). Tiene que ser (alto, ancho)."
        )
    return int(shape[0]), int(shape[1])


def _repartir(largo: int, tile: int, n: int) -> list[int]:
    """
    Offsets de n tiles de tamaño `tile` repartidos sobre `largo`, con el
    solape distribuido de forma pareja. El primero arranca en 0 y el
    último termina exactamente en el borde.
    """
    if n <= 1 or largo <= tile:
        return [0]
    return [round(i * (largo - tile) / (n - 1)) for i in range(n)]


def search_tiles(
    frame_hw: tuple[int, int], model_hw: tuple[int, int]
) -> list[tuple[int, int, int, int]]:
    """
    Grilla de regiones EN COORDENADAS NATIVAS que cubren el frame entero.
    Cada región se reduce después al tamaño de entrada del modelo.

    Se recorta primero y se reduce después (no al revés) para no pagar
    el resize del frame completo de 4608x2592 en cada frame de SEARCH.

    Con frame 4608x2592, modelo 640x1152 y grilla (2, 2):
        escala efectiva 0.4938, regiones de 2333x1296
        offsets x = [0, 2275], y = [0, 1296]
    """
    cache_key = (frame_hw, model_hw, tuple(getattr(config, "SEARCH_TILE_GRID", (2, 2))))
    if cache_key in _tiles_cache:
        return _tiles_cache[cache_key]

    cols, rows = getattr(config, "SEARCH_TILE_GRID", (2, 2))
    H, W = frame_hw
    th, tw = model_hw

    # La escala más grande con la que la grilla todavía cubre el frame.
    s = min(cols * tw / W, rows * th / H)
    rw = min(W, int(round(tw / s)))
    rh = min(H, int(round(th / s)))

    tiles = [
        (x, y, rw, rh)
        for y in _repartir(H, rh, rows)
        for x in _repartir(W, rw, cols)
    ]
    _tiles_cache[cache_key] = tiles
    return tiles


def tile_para_punto(
    frame_hw: tuple[int, int], model_hw: tuple[int, int], x: float, y: float
) -> int:
    """
    Indice del tile de SEARCH que contiene el punto (x, y) en coordenadas
    nativas. Si el punto cae en la zona de solape, devuelve el primero.

    Para que sirve: cuando TRACK pierde la pelota y hay que volver a
    SEARCH, arrancar la rotacion por este tile en vez de por donde haya
    quedado el contador. La pelota estaba ahi hace milisegundos, asi que
    el caso comun pasa de hasta 4 frames de barrido a 1.

    Con umbral de perdida de 150 ms y una pelota a 72 km/h, el objeto se
    movio unos 230 px: sigue dentro del mismo tile de 1166 px o, como
    mucho, en el de al lado.
    """
    tiles = search_tiles(frame_hw, model_hw)
    for idx, (tx, ty, tw, th) in enumerate(tiles):
        if tx <= x < tx + tw and ty <= y < ty + th:
            return idx

    # Fuera de todos los tiles (no deberia pasar): el mas cercano.
    mejor, mejor_d = 0, float("inf")
    for idx, (tx, ty, tw, th) in enumerate(tiles):
        cx, cy = tx + tw / 2.0, ty + th / 2.0
        d = (cx - x) ** 2 + (cy - y) ** 2
        if d < mejor_d:
            mejor, mejor_d = idx, d
    return mejor


def build_search_input(
    frame: np.ndarray, model_hw: tuple[int, int], tile_idx: int
) -> tuple[np.ndarray, ToGlobal]:
    """
    Modo SEARCH: toma un tile de la grilla, lo reduce al tamaño de
    entrada del modelo y lo convierte a RGB.
    """
    import cv2

    th, tw = model_hw
    tiles = search_tiles(frame.shape[:2], model_hw)
    x0, y0, rw, rh = tiles[tile_idx % len(tiles)]

    region = frame[y0 : y0 + rh, x0 : x0 + rw]

    # INTER_AREA y no INTER_LINEAR: a factor ~0.5 el bilineal submuestrea
    # y un objeto chico puede desaparecer de forma intermitente.
    chico = cv2.resize(region, (tw, th), interpolation=cv2.INTER_AREA)
    tensor = np.ascontiguousarray(cv2.cvtColor(chico, cv2.COLOR_BGR2RGB))

    fx = rw / tw
    fy = rh / th

    def to_global(local_x: float, local_y: float) -> tuple[float, float]:
        return x0 + local_x * fx, y0 + local_y * fy

    return tensor, to_global


def build_search_full(
    frame: np.ndarray, model_hw: tuple[int, int]
) -> tuple[np.ndarray, ToGlobal]:
    """
    Modo SEARCH con modelo dedicado: el frame ENTERO reducido al tamano
    de entrada del modelo, con letterbox si hace falta. Una sola
    inferencia por frame y campo de vision completo, sin mosaico.

    Esto solo sirve si el .hef de SEARCH fue entrenado para detectar la
    pelota a este tamano reducido. Con el modelo general no funciona: a
    0.25x la confianza medida cae a 0.0025, que fue justamente el motivo
    por el que existe el mosaico.
    """
    import cv2

    model_h, model_w = model_hw
    h, w = frame.shape[:2]

    s = min(model_w / w, model_h / h)
    new_w = max(1, min(model_w, int(round(w * s))))
    new_h = max(1, min(model_h, int(round(h * s))))

    chico = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_x = (model_w - new_w) // 2
    pad_y = (model_h - new_h) // 2

    canvas = np.zeros((model_h, model_w, 3), dtype=frame.dtype)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = chico
    tensor = np.ascontiguousarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    def to_global(local_x: float, local_y: float) -> tuple[float, float]:
        return (local_x - pad_x) / s, (local_y - pad_y) / s

    return tensor, to_global


def modelo_hw(hailo, modo) -> tuple[int, int]:
    """
    (alto, ancho) del modelo que corresponde a este modo. Con dos .hef
    cargados cada uno puede tener su propio tamano de entrada.
    """
    if hasattr(hailo, "input_shape_por_modo"):
        nombre = modo.name if hasattr(modo, "name") else str(modo)
        shape = hailo.input_shape_por_modo(nombre)
        return int(shape[0]), int(shape[1])
    return get_model_hw(hailo)


def inferir(hailo, tensor: np.ndarray, modo):
    """Adapta la llamada segun haya uno o dos modelos cargados."""
    if hasattr(hailo, "input_shape_por_modo"):
        nombre = modo.name if hasattr(modo, "name") else str(modo)
        return hailo.infer(tensor, nombre)
    return hailo.infer(tensor)


def build_track_input(
    frame: np.ndarray, state, model_hw: tuple[int, int]
) -> tuple[np.ndarray, ToGlobal]:
    """
    Modo TRACK: recorte centrado en la última posición conocida de la
    pelota, del mismo tamaño que espera el modelo (sin resize, o sea
    a resolución nativa, que es donde el modelo rinde mejor: 0.879).

    El recorte se clampea contra los bordes del frame: cerca del borde
    la pelota deja de estar centrada, pero el tamaño del tensor se
    mantiene constante, que es lo que exige la NPU.
    """
    import cv2

    crop_h, crop_w = model_hw
    h, w = frame.shape[:2]

    center_x, center_y = state.crop_center()

    x0 = int(round(center_x - crop_w / 2.0))
    y0 = int(round(center_y - crop_h / 2.0))
    x0 = max(0, min(x0, max(0, w - crop_w)))
    y0 = max(0, min(y0, max(0, h - crop_h)))

    crop = frame[y0 : y0 + crop_h, x0 : x0 + crop_w]

    if crop.shape[0] != crop_h or crop.shape[1] != crop_w:
        canvas = np.zeros((crop_h, crop_w, 3), dtype=frame.dtype)
        canvas[: crop.shape[0], : crop.shape[1]] = crop
        crop = canvas

    tensor = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    def to_global(local_x: float, local_y: float) -> tuple[float, float]:
        return local_x + x0, local_y + y0

    return tensor, to_global


def annotate(
    frame: np.ndarray,
    mode_name: str,
    detection,
    global_x: float | None,
    global_y: float | None,
    scale: float,
    to_global: ToGlobal | None = None,
) -> np.ndarray:
    """
    Dibuja la detección sobre una copia reducida del frame nativo, para
    el video de debug. Se llama FUERA del bloque de medición GPIO, así
    que no contamina los tiempos.
    """
    import cv2

    h, w = frame.shape[:2]
    vis = cv2.resize(
        frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )

    color = (0, 255, 0) if mode_name == "TRACK" else (0, 200, 255)
    label = mode_name

    if global_x is not None and global_y is not None:
        px = int(round(global_x * scale))
        py = int(round(global_y * scale))

        # La caja viene en coords del tensor: se reproyecta igual que el centro.
        if detection is not None and to_global is not None and detection.w > 0:
            gx1, gy1 = to_global(detection.x1, detection.y1)
            gx2, gy2 = to_global(detection.x2, detection.y2)
            cv2.rectangle(
                vis,
                (int(gx1 * scale), int(gy1 * scale)),
                (int(gx2 * scale), int(gy2 * scale)),
                color,
                2,
            )
        cv2.drawMarker(vis, (px, py), color, cv2.MARKER_CROSS, 18, 1)

        if detection is not None:
            label += f"  conf={detection.confidence:.2f}"
            label += f"  s{detection.stride}"
            label += f"  ({int(global_x)}, {int(global_y)})"
    else:
        label += "  sin deteccion"

    cv2.putText(
        vis, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA
    )
    cv2.putText(
        vis, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA
    )
    return vis


def process_frame(
    frame: np.ndarray,
    state: TrackerState,
    hailo,
    model_hw: tuple[int, int] | None = None,
    tile_idx: int = 0,
    search_completo: bool = False,
):
    """
    Procesa un único frame de punta a punta, incluyendo la medición
    con el pin de GPIO. Devuelve el estado actualizado, la detección
    (si hubo), la coordenada global (x, y), y la función de
    reproyección que se usó (para dibujar la caja después).

    Con dos .hef cargados, el tamaño de entrada se resuelve por modo y
    `model_hw` se ignora.
    """
    with gpio_timer.FrameTimer(mode=state.mode):
        hw = modelo_hw(hailo, state.mode) if model_hw is None else model_hw
        if hasattr(hailo, "input_shape_por_modo"):
            hw = modelo_hw(hailo, state.mode)

        if state.mode == Mode.SEARCH:
            if search_completo:
                model_input, to_global = build_search_full(frame, hw)
            else:
                model_input, to_global = build_search_input(frame, hw, tile_idx)
        else:
            model_input, to_global = build_track_input(frame, state, hw)

        raw_output = inferir(hailo, model_input, state.mode)
        detection = postprocess.process(raw_output, hw)

        if detection is not None:
            global_x, global_y = to_global(detection.x, detection.y)
        else:
            global_x, global_y = None, None

    confidence = detection.confidence if detection is not None else None
    state = state_machine.update(state, confidence, global_x, global_y)

    return state, detection, global_x, global_y, to_global


def abrir_inferencia():
    """
    Devuelve (objeto_de_inferencia, search_completo).

    Si config.SEARCH_HEF_PATH esta definido, carga DOS modelos y SEARCH
    pasa a ser una sola inferencia sobre el frame completo. Si no, sigue
    el camino de un solo modelo con mosaico de tiles.
    """
    search_hef = getattr(config, "SEARCH_HEF_PATH", None)
    if search_hef:
        from dual_inference import DualHailoInference

        track_hef = getattr(config, "TRACK_HEF_PATH", None) or config.HEF_PATH
        return DualHailoInference(search_hef, track_hef), True

    from inferencia import HailoInference     # antes: from hailo_inference import ...

    return HailoInference(config.HEF_PATH), False


def main() -> None:
    import cv2

    state = TrackerState()

    csv_writer = None
    csv_file = None
    if config.LOG_TO_CSV:
        csv_file = open(config.LOG_CSV_PATH, mode="w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "frame_idx",
                "mode",
                "tile",
                "confidence",
                "stride",
                "global_x",
                "global_y",
                "box_w",
                "box_h",
                "timestamp",
            ]
        )

    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {config.VIDEO_PATH}")

    debug_path = getattr(config, "DEBUG_VIDEO_PATH", None)
    debug_scale = getattr(config, "DEBUG_VIDEO_SCALE", 0.25)
    writer = None

    frame_idx = 0
    frames_leidos = 0
    frames_con_deteccion = 0
    tile_idx = 0

    inferencia, search_completo = abrir_inferencia()

    try:
        with inferencia as hailo:
            print(hailo.describe() if hasattr(hailo, "describe") else "")
            model_hw = None if search_completo else get_model_hw(hailo)
            if search_completo:
                print("[init] SEARCH: modelo dedicado, frame completo, 1 inferencia")
                print(f"[init] SEARCH entrada: {modelo_hw(hailo, Mode.SEARCH)}")
                print(f"[init] TRACK  entrada: {modelo_hw(hailo, Mode.TRACK)}")
            else:
                print(f"[init] modelo unico, entrada {model_hw}")

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames_leidos += 1

                if frame is None or frame.size == 0:
                    print(f"[warn] frame {frames_leidos} vacio, se saltea")
                    continue

                if frame_idx == 0:
                    h, w = frame.shape[:2]
                    print(f"[init] video {w}x{h}")
                    if not search_completo:
                        tiles = search_tiles((h, w), model_hw)
                        esc = model_hw[1] / tiles[0][2]
                        print(
                            f"[init] SEARCH: {len(tiles)} tiles de "
                            f"{tiles[0][2]}x{tiles[0][3]} nativos -> escala {esc:.4f}"
                        )

                mode_before = state.mode  # el modo con el que se procesó ESTE frame
                tile_usado = (
                    tile_idx if (mode_before == Mode.SEARCH and not search_completo) else -1
                )

                state, detection, global_x, global_y, to_global = process_frame(
                    frame, state, hailo, model_hw, tile_idx, search_completo
                )

                # Rotar el tile solo si hay mosaico y este frame fue SEARCH.
                if mode_before == Mode.SEARCH and not search_completo:
                    tile_idx += 1

                if global_x is not None:
                    frames_con_deteccion += 1

                if csv_writer is not None:
                    csv_writer.writerow(
                        [
                            frame_idx,
                            mode_before.name,
                            tile_usado,
                            f"{detection.confidence:.4f}" if detection else "",
                            detection.stride if detection else "",
                            f"{global_x:.1f}" if global_x is not None else "",
                            f"{global_y:.1f}" if global_y is not None else "",
                            f"{detection.w:.1f}" if detection else "",
                            f"{detection.h:.1f}" if detection else "",
                            time.time(),
                        ]
                    )

                if debug_path:
                    vis = annotate(
                        frame,
                        mode_before.name,
                        detection,
                        global_x,
                        global_y,
                        debug_scale,
                        to_global,
                    )
                    if writer is None:
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        if not fps or fps <= 0 or fps > 1000:
                            fps = 30.0
                        vh, vw = vis.shape[:2]
                        writer = cv2.VideoWriter(
                            debug_path,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (vw, vh),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(
                                f"No se pudo abrir el VideoWriter en {debug_path}"
                            )
                        print(f"[init] video de debug: {debug_path} @ {vw}x{vh}")
                    writer.write(vis)

                frame_idx += 1

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        gpio_timer.close()
        if csv_file is not None:
            csv_file.close()
        print(
            f"[fin] frames leidos: {frames_leidos} | "
            f"frames procesados: {frame_idx} | "
            f"con deteccion: {frames_con_deteccion}"
        )


if __name__ == "__main__":
    main()
