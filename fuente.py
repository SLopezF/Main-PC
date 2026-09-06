"""
fuente.py

De donde salen los frames. Dos implementaciones con la MISMA API, y un
selector que elige una u otra igual que hace inferencia.py con el backend
del modelo:

    FuenteCamara    envuelve camera_source.CameraSource (Picamera2, solo Pi)
    FuenteCarpeta   lee imagenes de un directorio (Windows, Pi, lo que sea)

CONTRATO (el que ya tenia CameraSource, para no romper nada)

    read() -> (frame, info)
        frame  ndarray HxWx3 uint8, canales en BGR
        info   dict con al menos:
               ts_ns     timestamp en nanosegundos
               dt_ms     ms desde el frame anterior ENTREGADO
               perdidos  frames del sensor que se saltearon
               seq       contador de frames entregados
        La fuente de carpeta agrega 'archivo' y 'agotada'.

    close(), __enter__/__exit__, total_perdidos, modos_sensor()

POR QUE BGR Y NO RGB
En la Pi, Picamera2 con formato "RGB888" entrega el array en orden B,G,R
(el nombre viene del empaquetado de bytes, no del orden en numpy). O sea:
lo que llega es lo mismo que devuelve cv2.imread(). Todo el pipeline cuenta
con eso -- build_search_full() hace cvtColor(BGR2RGB) al final -- asi que la
fuente de carpeta usa cv2.imread() sin convertir nada y las dos quedan
iguales. Si alguna vez cambias el formato de Picamera2, verificalo con
CameraSource.verificar_canales() antes de tocar el pipeline.

EL RELOJ, QUE NO ES UN DETALLE
Los umbrales del tracker son TEMPORALES (MS_PERDIDA_TRACK, MS_PERMANENCIA,
MS_CONFIRMAR_CAMBIO). Con una carpeta de imagenes hay dos relojes posibles y
dan resultados distintos:

    reloj="sintetico"  (default) el tiempo avanza 1/fps por imagen, sin
        importar cuanto tardo el modelo. Es el que sirve para replay: la
        misma carpeta da SIEMPRE el mismo resultado, en la PC y en la Pi,
        con GPU o sin ella. Es lo que hace falta para comparar dos versiones
        del tracker.

    reloj="pared"  el tiempo real que pasa entre lecturas. Sirve para el
        debug manual con ENTER, donde queres que 30 segundos pensando
        cuenten como 30 segundos.

USO
    python3 fuente.py --carpeta imagenes/cam0 --n 5
    python3 fuente.py --camara 0 --n 30          # solo en la Pi
"""

import os
import re
import time

import numpy as np

import config

try:
    import config_hw as _chw
except Exception:
    _chw = None


EXTENSIONES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _cfg(nombre: str, defecto=None):
    v = getattr(config, nombre, None)
    if v is None and _chw is not None:
        v = getattr(_chw, nombre, None)
    return defecto if v is None else v


def _orden_natural(nombre: str):
    """
    Clave de orden que trata los numeros como numeros: con el orden de texto
    'foto10.jpg' va antes que 'foto2.jpg', y una secuencia grabada a mano
    queda desordenada sin que se note hasta que el tracker da cualquier cosa.
    """
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", nombre)]


def listar_imagenes(ruta: str) -> list[str]:
    """Rutas de las imagenes de un directorio, en orden natural. Un archivo
    suelto se acepta como una lista de uno."""
    if os.path.isfile(ruta):
        return [ruta]
    if not os.path.isdir(ruta):
        raise FileNotFoundError(f"No existe la carpeta de imagenes '{ruta}'")
    nombres = [n for n in os.listdir(ruta)
               if n.lower().endswith(EXTENSIONES)]
    nombres.sort(key=_orden_natural)
    return [os.path.join(ruta, n) for n in nombres]


# =============================================================================
# Carpeta de imagenes
# =============================================================================

class FuenteCarpeta:
    """
    Entrega las imagenes de una carpeta, una por read(), en orden natural.

    `bucle=True` vuelve al principio al terminar (para el debug manual, que
    no tiene por que terminarse). Con `bucle=False` la ultima lectura marca
    info['agotada'] y la siguiente levanta StopIteration, que es lo que
    necesita un replay para saber cuando parar.
    """

    def __init__(
        self,
        ruta: str | None = None,
        indice: int = 0,
        fps: float | None = None,
        bucle: bool = True,
        reloj: str = "sintetico",
        forzar_tamano: tuple[int, int] | None = None,
        verbose: bool = True,
    ):
        if ruta is None:
            ruta = ruta_de_camara(indice)

        self.ruta = ruta
        self.indice = int(indice)
        self.archivos = listar_imagenes(ruta)
        if not self.archivos:
            raise FileNotFoundError(
                f"'{ruta}' no tiene imagenes con extension {EXTENSIONES}")

        self.fps = float(fps if fps is not None else _cfg("CAM_FPS", 40.0))
        self.periodo_ms = 1000.0 / self.fps
        self.bucle = bool(bucle)
        if reloj not in ("sintetico", "pared"):
            raise ValueError("reloj tiene que ser 'sintetico' o 'pared'")
        self.reloj = reloj
        self.forzar_tamano = forzar_tamano

        self._i = 0
        self._seq = 0
        self._ts_ns = 0
        self._t_pared = None
        self._agotada = False
        self._aviso_tamano = False
        self.total_perdidos = 0
        self.ms_espera = 0.0
        self.ms_copia = 0.0

        if verbose:
            print(f"[fuente] carpeta '{ruta}': {len(self.archivos)} imagen(es), "
                  f"reloj {self.reloj} a {self.fps:g} fps, "
                  f"bucle {'si' if self.bucle else 'no'}")

    # ------------------------------------------------------------------ API
    def read(self) -> tuple[np.ndarray, dict]:
        import cv2

        if self._agotada and not self.bucle:
            raise StopIteration(
                f"se acabaron las {len(self.archivos)} imagenes de "
                f"'{self.ruta}'")

        ruta = self.archivos[self._i]

        t0 = time.perf_counter()
        frame = cv2.imread(ruta, cv2.IMREAD_COLOR)
        t1 = time.perf_counter()
        if frame is None:
            raise RuntimeError(
                f"cv2.imread no pudo leer '{ruta}'. Si la ruta tiene acentos o "
                f"caracteres no ASCII, OpenCV falla en Windows sin avisar.")

        if self.forzar_tamano is not None:
            ancho, alto = self.forzar_tamano
            if (frame.shape[1], frame.shape[0]) != (ancho, alto):
                frame = cv2.resize(frame, (ancho, alto),
                                   interpolation=cv2.INTER_AREA)
        else:
            self._avisar_tamano(frame)

        # --- reloj
        if self.reloj == "sintetico":
            dt_ms = 0.0 if self._seq == 0 else self.periodo_ms
            self._ts_ns += int(dt_ms * 1e6)
        else:
            ahora = time.perf_counter()
            dt_ms = 0.0 if self._t_pared is None else (ahora - self._t_pared) * 1000.0
            self._t_pared = ahora
            self._ts_ns = int(ahora * 1e9)

        indice_archivo = self._i
        self._seq += 1
        self._i += 1
        if self._i >= len(self.archivos):
            self._agotada = True
            if self.bucle:
                self._i = 0

        return frame, {
            "ts_ns": self._ts_ns,
            "dt_ms": dt_ms,
            "perdidos": 0,          # de una carpeta no se pierde nada
            "seq": self._seq,
            "ms_espera": 0.0,
            "ms_copia": (t1 - t0) * 1000.0,   # aca es el decode del jpg
            "exposicion_us": None,
            "ganancia": None,
            "lux": None,
            "archivo": ruta,
            "indice_archivo": indice_archivo,
            "agotada": self._agotada,
            "camara": self.indice,
        }

    def reiniciar(self) -> None:
        """Vuelve a la primera imagen sin recrear la fuente."""
        self._i = 0
        self._agotada = False

    def __len__(self) -> int:
        return len(self.archivos)

    # -- compatibilidad con CameraSource, para que el codigo de debug no
    #    tenga que preguntar de que tipo es la fuente
    def modos_sensor(self) -> list:
        return []

    def verificar_canales(self, muestras: int = 5) -> str:
        return "BGR"     # cv2.imread siempre devuelve BGR

    def reportar_exposicion(self) -> dict:
        return {}

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ------------------------------------------------------------ internos
    def _avisar_tamano(self, frame) -> None:
        """
        Una imagen de un tamano distinto al de la camara no rompe nada (el
        preprocesado y pixel_a_angulo trabajan en proporciones), pero cambia
        cuantos pixeles mide la pelota y por lo tanto la confianza. Vale la
        pena enterarse una vez, no descubrirlo cuando el modelo "empeora".
        """
        if self._aviso_tamano:
            return
        self._aviso_tamano = True
        esperado = (int(_cfg("CAM_ANCHO", 2304)), int(_cfg("CAM_ALTO", 1296)))
        real = (frame.shape[1], frame.shape[0])
        if real != esperado:
            print(f"[fuente] !! las imagenes son {real[0]}x{real[1]} y la camara "
                  f"es {esperado[0]}x{esperado[1]}. Funciona igual, pero la "
                  f"pelota va a medir otra cantidad de pixeles: no compares "
                  f"confianzas contra las medidas en cancha.")


# =============================================================================
# Camara real (Picamera2)
# =============================================================================

class FuenteCamara:
    """
    camera_source.CameraSource envuelto para que su info() traiga las mismas
    claves que la fuente de carpeta. No modifica camera_source.py, que esta
    congelado: solo delega y agrega 'archivo': None y 'camara'.
    """

    def __init__(self, indice: int = 0, size=None, fps=None, hilo: bool = False,
                 **kwargs):
        import camera_source

        self.indice = int(indice)
        size = size or (int(_cfg("CAM_ANCHO", 2304)), int(_cfg("CAM_ALTO", 1296)))
        fps = float(fps if fps is not None else _cfg("CAM_FPS", 40.0))

        Clase = (camera_source.ThreadedCameraSource if hilo
                 else camera_source.CameraSource)
        self._cam = Clase(
            size=size,
            fps=fps,
            buffer_count=int(_cfg("CAM_BUFFERS", 4)),
            exposicion_us=_cfg("CAM_EXPOSICION_US", None),
            ganancia=_cfg("CAM_GANANCIA", None),
            enfoque=_cfg("CAM_ENFOQUE", None),
            indice=self.indice,
            **kwargs,
        )

    def read(self):
        frame, info = self._cam.read()
        info = dict(info)
        info["archivo"] = None
        info["agotada"] = False
        info["camara"] = self.indice
        return frame, info

    @property
    def total_perdidos(self) -> int:
        return self._cam.total_perdidos

    def modos_sensor(self) -> list:
        return self._cam.modos_sensor()

    def verificar_canales(self, muestras: int = 5) -> str:
        return self._cam.verificar_canales(muestras)

    def reportar_exposicion(self) -> dict:
        return self._cam.reportar_exposicion()

    def close(self) -> None:
        self._cam.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# =============================================================================
# Seleccion de fuente
# =============================================================================

def ruta_de_camara(indice: int) -> str:
    """
    Carpeta que le corresponde a la camara `indice`. Si existe una subcarpeta
    'cam0' / 'cam1' dentro de CARPETA_IMAGENES se usa esa; si no, la carpeta
    raiz para las dos (util cuando todavia tenes fotos de una sola camara).
    """
    base = (os.environ.get("CARPETA_IMAGENES")
            or _cfg("CARPETA_IMAGENES", "imagenes"))
    sub = os.path.join(base, f"cam{int(indice)}")
    return sub if os.path.isdir(sub) else base


def fuente_pedida() -> str:
    """'camara', 'carpeta' o 'auto'."""
    v = (os.environ.get("FUENTE")
         or getattr(config, "FUENTE_ENTRADA", None)
         or "auto")
    v = str(v).strip().lower()
    if v not in ("camara", "carpeta", "auto"):
        print(f"[fuente] fuente '{v}' desconocida, uso 'auto'")
        return "auto"
    return v


def _hay_picamera2() -> bool:
    try:
        import picamera2  # noqa: F401
        return True
    except Exception:
        return False


def abrir_fuente(indice: int = 0, **kwargs):
    """
    Fuente de frames para la camara `indice`. Igual que inferencia.abrir():
    manda la variable de entorno FUENTE, despues config.FUENTE_ENTRADA, y si
    no hay ninguna se decide sola segun exista picamera2.

    kwargs conocidos por las dos: size, fps.
    Solo carpeta: ruta, bucle, reloj, forzar_tamano.
    Solo camara: hilo.
    """
    elegida = fuente_pedida()
    if elegida == "auto":
        elegida = "camara" if _hay_picamera2() else "carpeta"
        print(f"[fuente] auto -> {elegida}")

    if elegida == "camara":
        for k in ("ruta", "bucle", "reloj", "forzar_tamano"):
            kwargs.pop(k, None)
        return FuenteCamara(indice=indice, **kwargs)

    kwargs.pop("hilo", None)
    kwargs.pop("size", None)
    return FuenteCarpeta(indice=indice, **kwargs)


# --------------------------------------------------------------------------- #
# CLI de prueba
# --------------------------------------------------------------------------- #

def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--carpeta", type=str, default=None,
                    help="fuerza una carpeta concreta")
    ap.add_argument("--camara", type=int, default=0, choices=[0, 1])
    ap.add_argument("--n", type=int, default=10, help="cuantos frames leer")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--reloj", type=str, default="sintetico",
                    choices=["sintetico", "pared"])
    ap.add_argument("--una-vez", action="store_true",
                    help="sin bucle: termina al agotar la carpeta")
    args = ap.parse_args()

    kw = {"fps": args.fps, "reloj": args.reloj, "bucle": not args.una_vez}
    if args.carpeta:
        os.environ["FUENTE"] = "carpeta"
        kw["ruta"] = args.carpeta

    with abrir_fuente(args.camara, **kw) as f:
        print(f"{'seq':>4} {'dt_ms':>8} {'perd':>5} {'shape':>16}  archivo")
        print("-" * 66)
        for _ in range(args.n):
            try:
                frame, info = f.read()
            except StopIteration as e:
                print(f"fin: {e}")
                break
            nombre = os.path.basename(info.get("archivo") or "(camara)")
            print(f"{info['seq']:>4} {info['dt_ms']:8.2f} {info['perdidos']:>5} "
                  f"{str(frame.shape):>16}  {nombre}")
        print(f"\nperdidos en total: {f.total_perdidos}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
