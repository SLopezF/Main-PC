"""
postprocess.py

Decodifica la salida cruda del HEF a una unica deteccion de la clase que
interesa.

Soporta TRES formatos de salida, porque el .hef oficial de Hailo y el propio
no se parecen en nada:

  1. CABEZA CRUDA, 1 CLASE  -- el .hef propio (una sola clase: la pelota).
        conv61 (80,144,4) caja  |  conv64 (80,144,1) clase   stride 8
        conv77 (40, 72,4) caja  |  conv80 (40, 72,1) clase   stride 16
        conv91 (20, 36,4) caja  |  conv94 (20, 36,1) clase   stride 32

  2. CABEZA CRUDA, 80 CLASES -- el .hef oficial de COCO sin NMS. Identico,
     pero la rama de clase trae 80 canales y hay que quedarse con el de
     'sports ball' (indice 32), NO con el maximo sobre canales: el maximo
     seguiria a una persona, que es lo que mas hay en una cancha.

  3. NMS EN EL CHIP -- lo mas comun en los .hef del Model Zoo de Hailo. La
     salida ya viene decodificada: una lista/array por clase con filas
     (ymin, xmin, ymax, xmax, score) en coordenadas NORMALIZADAS 0..1. Aca no
     hay strides ni anclas: se filtra por clase y se escala.

El formato se detecta solo, por la forma de los arrays. No hay que
configurar cual es.

LA CLASE OBJETIVO
Se lee de config.CLASE_OBJETIVO (o config_hw.CLASE_OBJETIVO). None, o un
modelo con un solo canal de clase, significa "hay una sola clase, usala".
Para el .hef de COCO, 32 = sports ball.

CAJA CON DFL
YOLO26 elimino el DFL: la rama de caja son las distancias l,t,r,b directas,
en unidades de celda. Pero si el .hef que consigas resulta ser de la familia
YOLOv8, esa rama trae 64 canales (4 lados x 16 bins) y hay que hacer softmax
e integral. Esta soportado para que un .hef de v8 no sea un callejon sin
salida, pero si aparece se avisa por consola: no es lo esperado.

RENDIMIENTO
Se evita aplicar sigmoide sobre las ~15.000 celdas de las tres escalas. Como
la sigmoide es monotona, el umbral se convierte UNA vez a su logit
equivalente y la comparacion se hace sobre los logits crudos. Solo la celda
ganadora pasa por sigmoide.
"""

import math
from dataclasses import dataclass

import numpy as np

import config

try:
    import config_hw as _chw
except Exception:  # config_hw puede no estar en un entorno pelado
    _chw = None


@dataclass
class Detection:
    """Resultado final del post-procesamiento para un frame."""

    x: float           # centro X, en coords del espacio de entrada del modelo
    y: float           # centro Y, en coords del espacio de entrada del modelo
    confidence: float  # probabilidad, ya pasada por sigmoide
    w: float = 0.0     # ancho de la caja, en px del espacio de entrada
    h: float = 0.0     # alto de la caja
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    stride: int = 0    # escala de la que salio (0 si vino del NMS del chip)


def _logit(p: float) -> float:
    """Inversa de la sigmoide, para comparar contra logits crudos."""
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _sigmoide(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


_UMBRAL_LOGIT = _logit(config.CONF_MIN_DETECTION)

# Cache del emparejamiento caja/clase por escala. Las shapes no cambian entre
# frames, asi que el agrupamiento se calcula una sola vez.
_layout_cache: dict[frozenset, list] = {}
_aviso_dado: set[str] = set()


def clase_objetivo() -> int | None:
    """Indice de clase a buscar. None = el modelo tiene una sola clase."""
    c = getattr(config, "CLASE_OBJETIVO", None)
    if c is None and _chw is not None:
        c = getattr(_chw, "CLASE_OBJETIVO", None)
    return None if c is None else int(c)


def _avisar_una_vez(clave: str, mensaje: str) -> None:
    if clave not in _aviso_dado:
        _aviso_dado.add(clave)
        print(f"[postprocess] {mensaje}")


def _input_hw(input_hw) -> tuple[int, int]:
    if input_hw is None:
        input_hw = getattr(config, "MODEL_INPUT_SIZE", (640, 1152))
    return int(input_hw[0]), int(input_hw[1])


def _como_hwc(arr) -> np.ndarray:
    """Saca la dimension de batch si viene, y garantiza 3 dimensiones."""
    a = np.asarray(arr)
    if a.ndim == 4:
        a = a[0]
    if a.ndim == 2:
        a = a[:, :, None]
    return a


# =============================================================================
# Formato 3: NMS resuelto en el chip
# =============================================================================

def _es_salida_nms(raw_output: dict) -> bool:
    """
    El NMS del chip entrega, por clase, una tabla de 5 columnas
    (ymin, xmin, ymax, xmax, score). Se reconoce por esas 5 columnas, o por
    venir como lista de arrays (una por clase) en vez de un tensor.
    """
    for valor in raw_output.values():
        if isinstance(valor, (list, tuple)):
            return True
        a = np.asarray(valor)
        if a.dtype == object:
            return True
        if a.ndim >= 2 and a.shape[-1] == 5:
            return True
    return False


def _filas_de_clase(valor, clase: int | None) -> list:
    """
    Filas de deteccion de la clase pedida, sea cual sea el empaquetado que
    use HailoRT: lista por clase, array 3D (clases, max_det, 5), o tabla
    plana de una sola clase.
    """
    if isinstance(valor, (list, tuple)) or (
            isinstance(valor, np.ndarray) and valor.dtype == object):
        por_clase = list(valor)
        if clase is None:
            return [(i, np.asarray(f)) for i, f in enumerate(por_clase)
                    if np.asarray(f).size]
        if clase >= len(por_clase):
            return []
        f = np.asarray(por_clase[clase])
        return [(clase, f)] if f.size else []

    a = np.asarray(valor)
    if a.ndim == 4:            # (batch, clases, max_det, 5)
        a = a[0]
    if a.ndim == 3:            # (clases, max_det, 5)
        if clase is None:
            return [(i, a[i]) for i in range(a.shape[0])]
        if clase >= a.shape[0]:
            return []
        return [(clase, a[clase])]
    if a.ndim == 2:            # tabla plana: una sola clase
        return [(0, a)]
    return []


def _detecciones_nms(raw_output: dict, input_hw, umbral: float,
                     topk: int) -> list[Detection]:
    alto, ancho = _input_hw(input_hw)
    clase = clase_objetivo()
    _avisar_una_vez(
        "nms",
        f"el HEF trae NMS en el chip; se filtra la clase "
        f"{clase if clase is not None else '(unica)'}")

    salida = []
    for valor in raw_output.values():
        for _, filas in _filas_de_clase(valor, clase):
            filas = np.asarray(filas, dtype=np.float32)
            if filas.ndim == 1:
                filas = filas[None, :]
            for fila in filas:
                if fila.shape[0] < 5:
                    continue
                ymin, xmin, ymax, xmax, score = (float(v) for v in fila[:5])
                if score < umbral:
                    continue
                # Normalizadas 0..1 -> pixeles del espacio de entrada.
                x1, x2 = xmin * ancho, xmax * ancho
                y1, y2 = ymin * alto, ymax * alto
                if x2 <= x1 or y2 <= y1:
                    continue
                salida.append(Detection(
                    x=(x1 + x2) / 2.0, y=(y1 + y2) / 2.0,
                    confidence=score, w=x2 - x1, h=y2 - y1,
                    x1=x1, y1=y1, x2=x2, y2=y2, stride=0,
                ))

    salida.sort(key=lambda d: -d.confidence)
    return salida[:topk]


# =============================================================================
# Formatos 1 y 2: cabeza cruda
# =============================================================================

def _armar_layout(raw_output: dict, input_hw: tuple[int, int]) -> list:
    """
    Empareja las salidas por tamano espacial. En cada escala, la rama de caja
    es la de 4 canales (o 64 si el modelo usa DFL) y la de clase es el resto.
    Se agrupa por SHAPE y no por nombre, asi que sigue funcionando si se
    recompila el HEF y cambian los nombres de las capas.
    """
    in_h, _ = input_hw
    por_escala: dict[tuple[int, int], dict] = {}

    for nombre, arr in raw_output.items():
        a = _como_hwc(arr)
        if a.ndim != 3:
            continue
        hw = (a.shape[0], a.shape[1])
        slot = por_escala.setdefault(hw, {})
        canales = a.shape[2]
        if canales == 4:
            slot["caja"], slot["dfl"] = nombre, False
        elif canales == 64:
            slot["caja"], slot["dfl"] = nombre, True
        else:
            slot["clase"] = nombre
            slot["n_clases"] = canales

    layout = []
    for hw, slot in sorted(por_escala.items(), key=lambda kv: -kv[0][0]):
        if "caja" not in slot or "clase" not in slot:
            continue
        if slot.get("dfl"):
            _avisar_una_vez(
                "dfl",
                "la rama de caja tiene 64 canales: el modelo usa DFL (familia "
                "YOLOv8). Se decodifica igual, pero no es lo que se espera de "
                "un YOLO26.")
        layout.append({
            "hw": hw,
            "stride": in_h / hw[0],
            "caja": slot["caja"],
            "clase": slot["clase"],
            "dfl": slot.get("dfl", False),
            "n_clases": slot.get("n_clases", 1),
        })

    if not layout:
        raise ValueError(
            f"No pude emparejar caja/clase en las salidas: "
            f"{[(k, np.asarray(v).shape) for k, v in raw_output.items()]}"
        )
    return layout


def _mapa_de_clase(cls: np.ndarray) -> np.ndarray:
    """
    Mapa de scores de la clase que interesa.

    Con 80 canales NO se toma el maximo sobre clases: eso devolveria a la
    persona mas confiada del frame, que en una cancha gana siempre. Se toma
    el canal de la clase objetivo y nada mas.
    """
    if cls.shape[2] == 1:
        return cls[:, :, 0]

    clase = clase_objetivo()
    if clase is None:
        _avisar_una_vez(
            "sin_clase",
            f"el modelo tiene {cls.shape[2]} clases y CLASE_OBJETIVO no esta "
            f"definida: se toma el maximo sobre clases, que casi seguro va a "
            f"ser una persona. Defini CLASE_OBJETIVO = 32 (sports ball).")
        return cls.max(axis=2)

    if clase >= cls.shape[2]:
        raise ValueError(
            f"CLASE_OBJETIVO={clase} pero el modelo tiene {cls.shape[2]} "
            f"canales de clase.")
    return cls[:, :, clase]


def _lados_caja(caja: np.ndarray, gy: int, gx: int, esc: dict) -> tuple:
    """Distancias l,t,r,b del ancla a los bordes, en unidades de celda."""
    v = caja[gy, gx]
    if not esc["dfl"]:
        return tuple(float(x) for x in v[:4])

    # DFL: 4 lados x 16 bins. Softmax y esperanza sobre los bins.
    bins = v[:64].reshape(4, 16).astype(np.float64)
    bins = bins - bins.max(axis=1, keepdims=True)
    e = np.exp(bins)
    p = e / e.sum(axis=1, keepdims=True)
    idx = np.arange(16, dtype=np.float64)
    return tuple(float(x) for x in (p * idx).sum(axis=1))


def _armar_deteccion(esc: dict, caja: np.ndarray, gx: int, gy: int,
                     logit: float) -> Detection:
    stride = esc["stride"]
    ancla_x = (gx + 0.5) * stride
    ancla_y = (gy + 0.5) * stride
    l, t, r, b = (v * stride for v in _lados_caja(caja, gy, gx, esc))

    x1, y1 = ancla_x - l, ancla_y - t
    x2, y2 = ancla_x + r, ancla_y + b
    return Detection(
        x=(x1 + x2) / 2.0,
        y=(y1 + y2) / 2.0,
        confidence=_sigmoide(logit),
        w=x2 - x1, h=y2 - y1,
        x1=x1, y1=y1, x2=x2, y2=y2,
        stride=int(round(stride)),
    )


def _layout_de(raw_output: dict, input_hw) -> list:
    # La clave incluye las SHAPES, no solo los nombres: dos modelos distintos
    # pueden usar los mismos nombres de capa con formatos distintos (por
    # ejemplo 4 canales de caja contra 64 con DFL), y cachear por nombre solo
    # haria que el segundo se decodifique con el layout del primero.
    clave = frozenset((n, tuple(np.asarray(v).shape))
                      for n, v in raw_output.items())
    layout = _layout_cache.get(clave)
    if layout is None:
        layout = _armar_layout(raw_output, input_hw)
        _layout_cache[clave] = layout
    return layout


# =============================================================================
# API publica
# =============================================================================

def process(raw_output, input_hw: tuple[int, int] | None = None) -> Detection | None:
    """
    Procesa la salida cruda de la NPU para un frame. Devuelve la deteccion de
    mayor confianza de la clase objetivo, o None si ninguna supera
    config.CONF_MIN_DETECTION.
    """
    if not isinstance(raw_output, dict):
        raise TypeError(
            "process() espera el dict con las salidas del HEF. Si llego un "
            "solo ndarray, hailo_inference.py esta descartando salidas."
        )

    input_hw = _input_hw(input_hw)

    if _es_salida_nms(raw_output):
        dets = _detecciones_nms(raw_output, input_hw,
                                config.CONF_MIN_DETECTION, topk=1)
        return dets[0] if dets else None

    layout = _layout_de(raw_output, input_hw)

    mejor_logit = -np.inf
    mejor = None
    for esc in layout:
        mapa = _mapa_de_clase(_como_hwc(raw_output[esc["clase"]]))
        plano = int(np.argmax(mapa))
        gy, gx = divmod(plano, mapa.shape[1])
        logit = float(mapa[gy, gx])
        if logit > mejor_logit:
            mejor_logit, mejor = logit, (esc, gx, gy)

    if mejor is None or mejor_logit < _UMBRAL_LOGIT:
        return None

    esc, gx, gy = mejor
    return _armar_deteccion(esc, _como_hwc(raw_output[esc["caja"]]),
                            gx, gy, mejor_logit)


def process_candidates(
    raw_output,
    input_hw: tuple[int, int] | None = None,
    umbral: float | None = None,
    topk: int = 8,
) -> list[Detection]:
    """
    Hasta `topk` candidatos por encima de `umbral`, ordenados por confianza.

    Para que sirve: con un filtro predictor se puede aceptar una deteccion de
    confianza baja SI cae cerca de donde el filtro predijo. Eso exige ver
    varios candidatos, no solo el maximo global: el pico mas fuerte del frame
    puede ser un falso positivo mientras la pelota real esta en el segundo.

    `umbral` es una probabilidad (post-sigmoide) y deberia ser MAS BAJO que
    config.CONF_MIN_DETECTION: es el piso de lo que vale la pena mirar, no el
    criterio de aceptacion.
    """
    if not isinstance(raw_output, dict):
        raise TypeError("process_candidates() espera el dict con las salidas del HEF.")

    input_hw = _input_hw(input_hw)
    if umbral is None:
        umbral = getattr(config, "CONF_CANDIDATO", 0.20)

    if _es_salida_nms(raw_output):
        return _detecciones_nms(raw_output, input_hw, umbral, topk)

    umbral_logit = _logit(umbral)
    layout = _layout_de(raw_output, input_hw)

    bruto = []
    for esc in layout:
        mapa = _mapa_de_clase(_como_hwc(raw_output[esc["clase"]]))
        # Comparar sobre logits crudos: la sigmoide es monotona, el resultado
        # es identico y se evitan miles de exp() por frame.
        ys, xs = np.where(mapa >= umbral_logit)
        if ys.size == 0:
            continue
        for gy, gx in zip(ys, xs):
            bruto.append((float(mapa[gy, gx]), esc, int(gx), int(gy)))

    if not bruto:
        return []

    bruto.sort(key=lambda t: -t[0])
    return [
        _armar_deteccion(esc, _como_hwc(raw_output[esc["caja"]]), gx, gy, logit)
        for logit, esc, gx, gy in bruto[:topk]
    ]
