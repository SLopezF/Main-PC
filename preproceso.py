"""
preproceso.py

Armado del tensor que entra al modelo. Rescatado de main.py, que se
reemplaza: estas funciones ya estaban probadas y no cambian de logica, solo
de casa.

QUE HACE Y QUE NO
Aca se decide QUE PEDAZO del frame nativo mira el modelo y como se lleva ese
pedazo al tamano exacto que la NPU exige. No se decide nada de tracking: no
sabe que es un modo, ni un estado, ni una prediccion. Recibe un frame y un
punto, devuelve un tensor y la funcion para volver de coordenadas del tensor
a coordenadas del frame nativo.

SIN HARDWARE Y SIN cv2 AL IMPORTAR
No importa picamera2, hailo_platform, spidev ni serial. `cv2` se importa
adentro de las funciones que lo usan, no a nivel de modulo, para que este
archivo se pueda importar y testear en cualquier maquina (y para que el
__main__ de abajo corra sin OpenCV instalado).

LOS DOS MODOS
Con frame nativo 2304x1296 y modelo 1152x640, que es casi exactamente 2x2:

    TRACK   recorte de 1152x640 centrado en la posicion conocida.
            Escala 1.000: resolucion NATIVA, sin resize. Una inferencia.

    SEARCH  mosaico de 2x2 cuadrantes de 1166x648, cada uno reducido a
            1152x640. Escala 0.9877, o sea 1.2% de reduccion: tambien
            practicamente nativo. Cuatro inferencias.

    grilla real: (0,0,1166,648)      (1138,0,1166,648)
                 (0,648,1166,648)    (1138,648,1166,648)
    solape horizontal 28 px, vertical 0 px

Que las dos escalas sean casi 1.0 es el motivo por el que un solo modelo,
entrenado sobre recortes nativos, sirve para los dos modos: en los dos ve la
pelota del mismo tamano aparente.

NO EXISTE build_search_full()
El frame entero reducido dejaba la pelota en 4 px a 20 m y se descarto junto
con el modelo entrenado en downscale. Si vuelve a hacer falta, esta en el
historial de main.py; no se mantiene aca codigo que no se usa.
"""

from typing import Callable

import numpy as np

import config

# Mapea (x, y) en coordenadas de la imagen que vio el modelo a (x, y) en
# coordenadas del frame nativo. Toda funcion que arma un tensor devuelve
# tambien una de estas: sin ella, una deteccion es un punto sin lugar.
ToGlobal = Callable[[float, float], tuple[float, float]]

_tiles_cache: dict[tuple, list[tuple[int, int, int, int]]] = {}


# =============================================================================
# Orden de canales
# =============================================================================

def _a_rgb(imagen: np.ndarray) -> np.ndarray:
    """
    Lleva la imagen al orden de canales que espera el modelo (RGB).

    Por que es una funcion y no un cvtColor clavado en cada sitio: si el orden
    que entrega la fuente resulta ser el contrario, se cambia UNA constante en
    config y no dos llamadas dispersas. El modelo rinde bastante mejor en RGB
    (0.879 contra 0.859 en recorte nativo, 0.476 contra 0.061 en frame
    reducido), asi que equivocarse aca cuesta un factor 8 en el caso peor y no
    se nota mirando la imagen.

    PENDIENTE DE P1: config.FUENTE_ENTREGA_BGR vale True porque fuente.py
    documenta que las dos fuentes entregan BGR (Picamera2 con formato
    "RGB888" devuelve el array en orden B,G,R, igual que cv2.imread). Eso
    todavia NO se confirmo contra la camara real con
    CameraSource.verificar_canales().
    """
    if not getattr(config, "FUENTE_ENTREGA_BGR", True):
        return np.ascontiguousarray(imagen)

    import cv2
    return np.ascontiguousarray(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))


# =============================================================================
# Tamano de entrada del modelo
# =============================================================================

def get_model_hw(hailo) -> tuple[int, int]:
    """
    Devuelve (alto, ancho) esperado por el modelo. Se lee del objeto de
    inferencia y no de config para que el codigo no dependa de que config
    este sincronizado con el modelo que efectivamente se cargo.
    """
    shape = getattr(hailo, "input_shape", None)
    if shape is None:
        shape = getattr(config, "MODEL_INPUT_SIZE", None)

    if shape is None:
        raise RuntimeError(
            "No pude determinar el input shape del modelo: ni hailo.input_shape "
            "ni config.MODEL_INPUT_SIZE estan disponibles."
        )
    if isinstance(shape, int):
        raise RuntimeError(
            f"El input del modelo es rectangular, pero se configuro como un "
            f"escalar ({shape}). Tiene que ser (alto, ancho)."
        )
    return int(shape[0]), int(shape[1])


# =============================================================================
# SEARCH: mosaico de cuadrantes
# =============================================================================

def _repartir(largo: int, tile: int, n: int) -> list[int]:
    """
    Offsets de n tiles de tamano `tile` repartidos sobre `largo`, con el
    solape distribuido de forma pareja. El primero arranca en 0 y el ultimo
    termina exactamente en el borde.
    """
    if n <= 1 or largo <= tile:
        return [0]
    return [round(i * (largo - tile) / (n - 1)) for i in range(n)]


def search_tiles(
    frame_hw: tuple[int, int], model_hw: tuple[int, int]
) -> list[tuple[int, int, int, int]]:
    """
    Grilla de regiones EN COORDENADAS NATIVAS que cubren el frame entero.
    Cada region se reduce despues al tamano de entrada del modelo.

    Se recorta primero y se reduce despues (no al reves) para no pagar el
    resize del frame completo en cada frame de SEARCH.

    Con frame 2304x1296, modelo 1152x640 y grilla (2, 2):
        regiones de 1166x648, escala efectiva 0.9877
        offsets x = [0, 1138], y = [0, 648]
        solape horizontal 28 px, vertical 0 px
    """
    cache_key = (frame_hw, model_hw,
                 tuple(getattr(config, "SEARCH_TILE_GRID", (2, 2))))
    if cache_key in _tiles_cache:
        return _tiles_cache[cache_key]

    cols, rows = getattr(config, "SEARCH_TILE_GRID", (2, 2))
    H, W = frame_hw
    th, tw = model_hw

    # La escala mas grande con la que la grilla todavia cubre el frame.
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

    Para que sirve: cuando TRACK pierde la pelota y hay que volver a SEARCH,
    arrancar el barrido por este tile en vez de por donde haya quedado el
    contador. La pelota estaba ahi hace milisegundos, asi que el caso comun
    pasa de hasta 4 inferencias de barrido a 1.

    Con un umbral de perdida de 300 ms y una pelota a 72 km/h, el objeto se
    movio unos 460 px: sigue dentro del mismo tile de 1166 px de ancho o,
    como mucho, en el de al lado.
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
    Modo SEARCH: toma un tile de la grilla, lo reduce al tamano de entrada
    del modelo y lo deja en el orden de canales correcto.

    `tile_idx` se toma modulo la cantidad de tiles, asi que un contador que
    crece indefinidamente sirve como barrido ciclico sin tener que resetearlo.
    """
    import cv2

    th, tw = model_hw
    tiles = search_tiles(frame.shape[:2], model_hw)
    x0, y0, rw, rh = tiles[tile_idx % len(tiles)]

    region = frame[y0:y0 + rh, x0:x0 + rw]

    # INTER_AREA y no INTER_LINEAR: al reducir, el bilineal submuestrea y un
    # objeto chico puede desaparecer de forma intermitente. A escala 0.99 la
    # diferencia es minima, pero no cuesta nada y protege si algun dia la
    # grilla se hace mas gruesa.
    chico = cv2.resize(region, (tw, th), interpolation=cv2.INTER_AREA)
    tensor = _a_rgb(chico)

    fx = rw / tw
    fy = rh / th

    def to_global(local_x: float, local_y: float) -> tuple[float, float]:
        return x0 + local_x * fx, y0 + local_y * fy

    return tensor, to_global


# =============================================================================
# TRACK: recorte centrado, sin resize
# =============================================================================

def build_track_input(
    frame: np.ndarray, centro: tuple[float, float], model_hw: tuple[int, int]
) -> tuple[np.ndarray, ToGlobal]:
    """
    Modo TRACK: recorte del tamano exacto que espera el modelo, centrado en
    `centro` = (x, y) en coordenadas del frame NATIVO. Sin resize, o sea a
    resolucion nativa, que es donde el modelo rinde mejor.

    `centro` es una tupla y no un objeto de estado a proposito: la version de
    main.py recibia el TrackerState y llamaba a state.crop_center(), lo que
    ataba este modulo a la maquina de estados. Con la tupla, preproceso.py no
    depende de nada y se testea solo.

    El recorte se clampea contra los bordes del frame: cerca del borde la
    pelota deja de estar centrada, pero el tamano del tensor se mantiene
    constante, que es lo que la NPU exige.
    """
    import cv2  # noqa: F401  (lo usa _a_rgb; se importa aca por simetria)

    crop_h, crop_w = model_hw
    h, w = frame.shape[:2]

    center_x, center_y = centro

    x0 = int(round(center_x - crop_w / 2.0))
    y0 = int(round(center_y - crop_h / 2.0))
    x0 = max(0, min(x0, max(0, w - crop_w)))
    y0 = max(0, min(y0, max(0, h - crop_h)))

    crop = frame[y0:y0 + crop_h, x0:x0 + crop_w]

    # Si el frame es MAS CHICO que la entrada del modelo, se rellena con
    # negro en vez de fallar. Pasa con material de prueba de baja resolucion.
    if crop.shape[0] != crop_h or crop.shape[1] != crop_w:
        canvas = np.zeros((crop_h, crop_w, 3), dtype=frame.dtype)
        canvas[:crop.shape[0], :crop.shape[1]] = crop
        crop = canvas

    tensor = _a_rgb(crop)

    def to_global(local_x: float, local_y: float) -> tuple[float, float]:
        return local_x + x0, local_y + y0

    return tensor, to_global


# --------------------------------------------------------------------------- #
# CLI: describe la grilla sin necesitar modelo, imagenes ni OpenCV
# --------------------------------------------------------------------------- #

def describir(frame_hw: tuple[int, int], model_hw: tuple[int, int]) -> str:
    H, W = frame_hw
    th, tw = model_hw
    tiles = search_tiles(frame_hw, model_hw)
    rw, rh = tiles[0][2], tiles[0][3]

    cols, rows = getattr(config, "SEARCH_TILE_GRID", (2, 2))
    xs = sorted({t[0] for t in tiles})
    ys = sorted({t[1] for t in tiles})
    solape_x = (xs[0] + rw - xs[1]) if len(xs) > 1 else 0
    solape_y = (ys[0] + rh - ys[1]) if len(ys) > 1 else 0

    lineas = [
        f"frame nativo   {W}x{H}",
        f"entrada modelo {tw}x{th}",
        "",
        f"TRACK   recorte de {tw}x{th} centrado, escala 1.000 (nativo), "
        f"1 inferencia",
        f"SEARCH  grilla {cols}x{rows} de {rw}x{rh}, escala "
        f"{tw / rw:.4f}, {len(tiles)} inferencias",
        f"        solape  x={solape_x} px  y={solape_y} px",
        "",
        "tiles (x, y, ancho, alto):",
    ]
    for i, t in enumerate(tiles):
        lineas.append(f"  {i}: {t}")

    # Un punto por cuadrante, para ver que el ruteo del barrido sea el que uno
    # espera mirando la grilla de arriba.
    lineas.append("")
    lineas.append("tile_para_punto en el centro de cada cuadrante:")
    for py in (H * 0.25, H * 0.75):
        fila = [f"{tile_para_punto(frame_hw, model_hw, px, py)}"
                for px in (W * 0.25, W * 0.75)]
        lineas.append("  " + "  ".join(fila))

    return "\n".join(lineas)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ancho", type=int, default=getattr(config, "CAM_ANCHO", 2304))
    ap.add_argument("--alto", type=int, default=getattr(config, "CAM_ALTO", 1296))
    ap.add_argument("--modelo", type=str, default=None,
                    help="ANCHOxALTO de la entrada; por defecto, config")
    args = ap.parse_args()

    if args.modelo:
        mw, mh = (int(v) for v in args.modelo.lower().split("x"))
        model_hw = (mh, mw)
    else:
        model_hw = tuple(config.MODEL_INPUT_SIZE)

    print(describir((args.alto, args.ancho), model_hw))
    print()
    print(f"canales: la fuente entrega "
          f"{'BGR' if getattr(config, 'FUENTE_ENTREGA_BGR', True) else 'RGB'}"
          f" -> se entrega al modelo en RGB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())