"""
inferencia_ultralytics.py

Backend de inferencia por Ultralytics con la MISMA API que
hailo_inference.HailoInference, para poder correr todo el pipeline en una
PC sin Hailo-8 ni HailoRT.

QUE TIENE QUE CUMPLIR (contrato de hailo_inference.HailoInference)
    .input_shape        -> (alto, ancho, canales), tal como lo pide el modelo
    .output_shapes      -> {nombre: shape}
    .describe()         -> str para el log de arranque
    .infer(frame)       -> dict {nombre: ndarray} con TODAS las salidas
    .close(), __enter__, __exit__

EN QUE FORMATO SE DEVUELVE LA SALIDA
postprocess.py soporta tres formatos y los detecta solo por la forma de los
arrays. El unico que se puede reproducir honestamente desde Ultralytics es el
tercero: "NMS resuelto en el chip", o sea una tabla por clase con filas
(ymin, xmin, ymax, xmax, score) en coordenadas NORMALIZADAS 0..1.

No se intenta reconstruir las seis cabezas crudas (conv61/conv64/...): eso
seria inventar tensores que Ultralytics ya decodifico, y cualquier diferencia
de convencion en el DFL o en los strides aparecerian como un bug del
postproceso en vez de como lo que serian, un artefacto del emulador. Con la
tabla de NMS el camino que corre en la PC es el mismo que correria en la Pi
si el .hef se compilara con NMS adentro, y `postprocess.process()` devuelve
un `Detection` identico.

CONSECUENCIA A TENER EN CUENTA
El `Detection.stride` va a venir en 0 (la tabla de NMS no dice de que escala
salio la caja) y las cajas ya pasaron por el NMS de Ultralytics, que no es
bit a bit el mismo que el del chip. O sea: sirve para desarrollar la logica
de tracking, de sectores y del motor, NO para medir el rendimiento del
modelo cuantizado. Los numeros de confianza de la tesis se miden en la Pi.

RGB vs BGR
El pipeline (main.build_search_full / build_search_input) entrega el tensor
en RGB, porque el .hef espera RGB. Ultralytics, en cambio, asume BGR cuando
le pasas un ndarray (internamente hace im[..., ::-1]). Por eso aca se
invierte el orden de canales antes de llamarlo: sin esto el modelo ve la
imagen con los colores cambiados y la confianza se cae sin motivo aparente.

USO
    python3 inferencia_ultralytics.py --info
    python3 inferencia_ultralytics.py --imagen /ruta/foto.jpg
"""

import os

import numpy as np

import config
import postprocess

try:
    import config_hw as _chw
except Exception:                      # config_hw puede no estar
    _chw = None


# Nombre unico de la "salida" que se expone. No corresponde a ninguna capa
# real: es la etiqueta con la que viaja la tabla de NMS dentro del dict.
NOMBRE_SALIDA = "ultralytics_nms"


def _cfg(nombre: str, defecto=None):
    """Constante de config.py, o de config_hw.py, o el default."""
    v = getattr(config, nombre, None)
    if v is None and _chw is not None:
        v = getattr(_chw, nombre, None)
    return defecto if v is None else v


def _ruta_por_defecto() -> str:
    """
    Ruta del .pt. Prioridad: variable de entorno MODELO_PT, luego
    config.MODELO_PT, y si no hay ninguna, el nombre del modelo propio.
    """
    return (os.environ.get("MODELO_PT")
            or _cfg("MODELO_PT", "yolo26n_custom.pt"))


class UltralyticsInference:
    """
    Reemplazo de HailoInference para correr en la PC.

    `ruta_modelo` acepta un .pt directamente, o un .hef: si le llega un .hef
    (porque el main le pasa config.HEF_PATH sin saber que backend hay detras)
    lo ignora y usa el .pt configurado, avisando una vez.
    """

    def __init__(
        self,
        ruta_modelo: str | None = None,
        input_hw: tuple[int, int] | None = None,
        device: str | None = None,
        conf_piso: float | None = None,
        iou: float | None = None,
        max_det: int | None = None,
        verbose: bool = True,
    ):
        ruta = ruta_modelo
        if ruta is None or str(ruta).lower().endswith(".hef"):
            reemplazo = _ruta_por_defecto()
            if verbose and ruta is not None:
                print(f"[ultralytics] me pasaron '{ruta}' (un .hef): en esta "
                      f"maquina no hay Hailo, uso '{reemplazo}'")
            ruta = reemplazo

        if not os.path.exists(ruta):
            raise FileNotFoundError(
                f"No encuentro el modelo '{ruta}'. Poné el .pt al lado del "
                f"codigo, o definí MODELO_PT en config.py o en el entorno "
                f"(MODELO_PT=/ruta/al/modelo.pt python3 ...)."
            )

        from ultralytics import YOLO      # import tardio: solo si se usa

        self._ruta = ruta
        self._modelo = YOLO(ruta)

        # --- tamano de entrada -------------------------------------------
        # Se toma de config (no del .pt): el .pt guarda el imgsz de
        # entrenamiento como escalar y eso no distingue 1152x640 de 1152x1152,
        # que fue exactamente el bug que ya aparecio una vez con el .hef.
        if input_hw is None:
            input_hw = _cfg("MODEL_INPUT_SIZE", (640, 640))
        if isinstance(input_hw, int):
            raise ValueError(
                f"MODEL_INPUT_SIZE es un escalar ({input_hw}). El modelo no es "
                f"cuadrado: tiene que ser (alto, ancho), por ejemplo (640, 1152)."
            )
        alto, ancho = int(input_hw[0]), int(input_hw[1])
        canales = int(_cfg("MODEL_INPUT_CHANNELS", 3))
        self._input_shape = (alto, ancho, canales)

        if alto % 32 or ancho % 32:
            print(f"[ultralytics] !! {ancho}x{alto} no es multiplo de 32: "
                  f"Ultralytics va a agregar padding propio. Las cajas vuelven "
                  f"bien igual, pero no es el tamano con el que se entreno.")

        # --- clases -------------------------------------------------------
        nombres = getattr(self._modelo, "names", None) or {}
        self._nombres = dict(nombres) if nombres else {0: "clase0"}
        self._n_clases = max(1, len(self._nombres))

        clase = postprocess.clase_objetivo()
        if clase is not None and clase >= self._n_clases:
            print(f"[ultralytics] !! CLASE_OBJETIVO = {clase} pero este modelo "
                  f"tiene {self._n_clases} clase(s) ({self._nombres}). "
                  f"postprocess va a filtrar por una clase que no existe y no "
                  f"vas a ver NUNCA una deteccion. Poné CLASE_OBJETIVO = None "
                  f"en config_hw.py para el modelo propio.")

        # --- parametros de inferencia -------------------------------------
        # El piso de confianza tiene que ser el MAS BAJO de los dos umbrales:
        # process() filtra por CONF_MIN_DETECTION y process_candidates() por
        # CONF_CANDIDATO, que es mas bajo a proposito. Si el backend cortara
        # en el umbral alto, los candidatos flojos nunca llegarian.
        if conf_piso is None:
            conf_piso = min(
                float(_cfg("CONF_MIN_DETECTION", 0.1)),
                float(_cfg("CONF_CANDIDATO", 0.05)),
            )
        self._conf = max(1e-3, float(conf_piso))
        self._iou = float(iou if iou is not None else _cfg("ULTRA_IOU", 0.45))
        self._max_det = int(max_det if max_det is not None
                            else _cfg("ULTRA_MAX_DET", 32))
        self._device = device if device is not None else _cfg("ULTRA_DEVICE", None)

        self._output_shapes = {
            NOMBRE_SALIDA: (self._n_clases, self._max_det, 5)
        }

    # ---------------------------------------------------------------- API
    @property
    def input_shape(self) -> tuple[int, int, int]:
        """(alto, ancho, canales), igual que lo reporta el .hef real."""
        return self._input_shape

    @property
    def output_shapes(self) -> dict[str, tuple]:
        return dict(self._output_shapes)

    @property
    def nombres_clases(self) -> dict:
        return dict(self._nombres)

    def describe(self) -> str:
        alto, ancho, canales = self._input_shape
        clases = ", ".join(f"{i}:{n}" for i, n in sorted(self._nombres.items()))
        return "\n".join([
            "BACKEND ultralytics (emulando hailo_inference)",
            f"MODELO  {self._ruta}",
            f"IN      {self._input_shape}  ({ancho}x{alto} ancho x alto, "
            f"{canales} canales, RGB)",
            f"CLASES  {self._n_clases} -> {clases}",
            f"OUT     {NOMBRE_SALIDA}: {self._output_shapes[NOMBRE_SALIDA]}  "
            f"(NMS ya resuelto; stride va a venir en 0)",
            f"PARAMS  conf>={self._conf:.3f}  iou={self._iou}  "
            f"max_det={self._max_det}  device={self._device or 'auto'}",
        ])

    def infer(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        """
        Mismo contrato que HailoInference.infer(): recibe un frame ya
        preprocesado (exactamente input_shape, uint8, canales en RGB) y
        devuelve el dict de salidas, listo para postprocess.process().
        """
        if tuple(frame.shape) != self._input_shape:
            raise ValueError(
                f"Frame de entrada con shape {frame.shape}, "
                f"se esperaba {self._input_shape}"
            )

        # El pipeline entrega RGB; Ultralytics asume BGR para los ndarray.
        bgr = np.ascontiguousarray(frame[:, :, ::-1])

        alto, ancho, _ = self._input_shape
        r = self._modelo.predict(
            bgr,
            imgsz=(alto, ancho),
            conf=self._conf,
            iou=self._iou,
            max_det=self._max_det,
            device=self._device,
            verbose=False,
        )[0]

        return {NOMBRE_SALIDA: self._tabla_nms(r)}

    # ----------------------------------------------------------- internos
    def _tabla_nms(self, resultado) -> np.ndarray:
        """
        (n_clases, max_det, 5) con filas (ymin, xmin, ymax, xmax, score) en
        0..1, que es el formato 3 de postprocess. Las filas sobrantes quedan
        en cero: score 0 cae por debajo de cualquier umbral y ademas tienen
        x2 == x1, asi que el postproceso las descarta por las dos vias.
        """
        tabla = np.zeros((self._n_clases, self._max_det, 5), dtype=np.float32)

        cajas = getattr(resultado, "boxes", None)
        if cajas is None or len(cajas) == 0:
            return tabla

        xyxyn = np.asarray(cajas.xyxyn.cpu().numpy(), dtype=np.float32)
        conf = np.asarray(cajas.conf.cpu().numpy(), dtype=np.float32)
        cls = np.asarray(cajas.cls.cpu().numpy()).astype(int)

        usados = [0] * self._n_clases
        for (x1, y1, x2, y2), score, c in zip(xyxyn, conf, cls):
            if c < 0 or c >= self._n_clases:
                continue
            k = usados[c]
            if k >= self._max_det:
                continue
            tabla[c, k] = (y1, x1, y2, x2, score)
            usados[c] = k + 1

        return tabla

    # ------------------------------------------------------------- cierre
    def close(self) -> None:
        self._modelo = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# Alias para que quien haga `from inferencia_ultralytics import HailoInference`
# no note la diferencia.
HailoInference = UltralyticsInference


# --------------------------------------------------------------------------- #
# CLI de prueba: un frame de punta a punta, sin camara ni Pi
# --------------------------------------------------------------------------- #

def _probar_imagen(hailo, ruta_imagen: str, completo: bool, salida: str) -> int:
    import cv2

    import main as pipeline

    frame = cv2.imread(ruta_imagen)
    if frame is None:
        print(f"no pude leer '{ruta_imagen}'")
        return 1

    h, w = frame.shape[:2]
    model_hw = pipeline.get_model_hw(hailo)
    print(f"\nimagen {w}x{h}  ->  entrada del modelo "
          f"{model_hw[1]}x{model_hw[0]}")

    mejor = None
    if completo:
        tensor, to_global = pipeline.build_search_full(frame, model_hw)
        entradas = [(tensor, to_global, -1)]
    else:
        tiles = pipeline.search_tiles((h, w), model_hw)
        entradas = []
        for i in range(len(tiles)):
            t, g = pipeline.build_search_input(frame, model_hw, i)
            entradas.append((t, g, i))
        print(f"mosaico de {len(tiles)} tiles de {tiles[0][2]}x{tiles[0][3]}")

    import time
    for tensor, to_global, tile in entradas:
        t0 = time.perf_counter()
        cands = postprocess.process_candidates(
            hailo.infer(tensor), model_hw,
            umbral=float(_cfg("CONF_CANDIDATO", 0.05)), topk=3)
        ms = (time.perf_counter() - t0) * 1000.0
        etiqueta = "frame entero" if tile < 0 else f"tile {tile}"
        if not cands:
            print(f"  {etiqueta}: sin candidatos  ({ms:.0f} ms)")
            continue
        for d in cands:
            gx, gy = to_global(d.x, d.y)
            print(f"  {etiqueta}: conf {d.confidence:.3f}  px ({gx:.0f}, {gy:.0f})"
                  f"  caja {d.w:.0f}x{d.h:.0f}  ({ms:.0f} ms)")
        d = cands[0]
        if mejor is None or d.confidence > mejor[0].confidence:
            mejor = (d, to_global)

    if mejor is None:
        print("\nsin detecciones por encima del umbral.")
        return 0

    d, to_global = mejor
    gx, gy = to_global(d.x, d.y)
    x1, y1 = to_global(d.x1, d.y1)
    x2, y2 = to_global(d.x2, d.y2)
    print(f"\nMEJOR: conf {d.confidence:.3f}  centro ({gx:.0f}, {gy:.0f})")

    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(frame, f"{d.confidence:.2f}", (int(x1), max(20, int(y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    if not cv2.imwrite(salida, frame):
        print(f"!! no pude escribir '{salida}'. En Windows '/tmp/...' se "
              f"interpreta como una carpeta 'tmp' del disco actual, que no "
              f"existe: pasá una ruta valida con --salida.")
        return 1
    print(f"anotada -> {os.path.abspath(salida)}")
    return 0


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", type=str, default=None, help="ruta al .pt")
    ap.add_argument("--imagen", type=str, default=None)
    ap.add_argument("--salida", type=str, default="deteccion.jpg")
    ap.add_argument("--full", dest="full", action="store_true", default=None,
                    help="frame entero con letterbox")
    ap.add_argument("--tiles", dest="full", action="store_false",
                    help="mosaico de tiles")
    ap.add_argument("--info", action="store_true", help="solo describir")
    args = ap.parse_args()

    with UltralyticsInference(args.modelo) as hailo:
        print(hailo.describe())
        if args.info or not args.imagen:
            return 0
        completo = (args.full if args.full is not None
                    else bool(_cfg("SEARCH_FULL", True)))
        return _probar_imagen(hailo, args.imagen, completo, args.salida)


if __name__ == "__main__":
    raise SystemExit(main())
