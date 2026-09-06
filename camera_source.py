"""
camera_source.py

Captura desde la camara de la Raspberry Pi con politica de ULTIMO FRAME
GANA: cada llamada a read() devuelve el frame mas reciente disponible y
descarta los intermedios, en vez de acumularlos en una cola.

Por que importa: si el computo tarda mas que el periodo de la camara, una
cola hace crecer la latencia sin limite y terminas rastreando donde estaba
la pelota hace cientos de milisegundos. Para un tracker, un dato viejo es
peor que ningun dato.

Ademas lleva la cuenta de cuantos frames se perdieron entre lecturas, que
es un dato necesario para dos cosas:
  - reproyectar/extrapolar el centro del crop de TRACK
  - que los umbrales de la maquina de estados sean temporales y no por
    cantidad de frames

OJO CON EL ORDEN DE CANALES:
Picamera2 usa nombres de formato que vienen del empaquetado de bytes, no
del orden en el array de numpy. El formato llamado "RGB888" suele entregar
un array en orden B,G,R (lo que OpenCV considera BGR), y "BGR888" entrega
R,G,B. Es una fuente clasica de confusion. Como ya sabemos que el modelo
rinde bastante mejor en RGB (0.879 contra 0.859 en crop nativo, 0.476
contra 0.061 en frame reducido), conviene NO confiar en el nombre y
verificarlo empiricamente con verificar_canales().
"""

import time

import numpy as np


class CameraSource:
    """
    Envuelve Picamera2 con politica de ultimo-frame-gana.

    read() devuelve (frame, info) donde info trae:
        ts_ns      timestamp del sensor en nanosegundos
        dt_ms      milisegundos desde el frame anterior ENTREGADO
        perdidos   cuantos frames del sensor se saltearon
        seq        contador de frames entregados
    """

    def __init__(
        self,
        size: tuple[int, int] = (2304, 1296),
        fps: float = 56.0,
        formato: str = "RGB888",
        buffer_count: int = 4,
        exposicion_us: int | None = None,
        ganancia: float | None = None,
        indice: int = 0,
        enfoque: str | float | None = None,
    ):
        from picamera2 import Picamera2

        self.size = size
        self.fps = float(fps)
        self.periodo_us = int(round(1_000_000.0 / self.fps))
        self.periodo_ms = 1000.0 / self.fps

        # indice elige el conector CSI: 0 o 1. Sin argumento, Picamera2
        # siempre agarra la camara 0. Verifica cuales ve la Pi con
        # `rpicam-hello --list-cameras`.
        self.indice = int(indice)
        self._picam = Picamera2(self.indice)

        # sensor_modes reconfigura la camara por dentro para inspeccionarla,
        # asi que hay que leerlo ANTES de configure()/start(). Se cachea.
        try:
            self._modos = list(self._picam.sensor_modes)
        except Exception as exc:
            print(f"[warn] no pude leer sensor_modes: {exc}")
            self._modos = []

        cfg = self._picam.create_video_configuration(
            main={"size": size, "format": formato},
            buffer_count=buffer_count,
            controls={
                # Fijar el periodo arriba y abajo obliga a la camara a
                # correr al framerate pedido y no a bajarlo por exposicion.
                "FrameDurationLimits": (self.periodo_us, self.periodo_us),
            },
        )
        self._picam.configure(cfg)

        self._picam.start()

        # POLITICA: no se toca NINGUN control salvo que se pida explicito.
        # Sin argumentos, la camara se comporta igual que `rpicam-hello`:
        # auto-exposicion, auto-balance de blancos y autofoco continuo. Es
        # el modo en el que la imagen se ve mejor y el modelo detecta mejor.
        #
        # Antes esto forzaba AfMode=0 (foco MANUAL), que dejaba la lente
        # donde hubiera quedado. Una pelota desenfocada es una mancha y el
        # modelo la detecta mal: por eso la preview se veia mejor.
        auto = exposicion_us is None and ganancia is None
        time.sleep(2.0 if auto else 0.5)  # el AE tarda en converger

        if enfoque is not None:
            try:
                if enfoque == "continuo":
                    self._picam.set_controls({"AfMode": 2})
                    print("[cam] autofoco continuo")
                elif enfoque == "auto":
                    self._picam.set_controls({"AfMode": 1})
                    self._picam.autofocus_cycle()
                    print("[cam] un ciclo de autofoco, despues fijo")
                else:
                    self._picam.set_controls(
                        {"AfMode": 0, "LensPosition": float(enfoque)}
                    )
                    print(f"[cam] foco manual en LensPosition={float(enfoque)}")
            except Exception as exc:
                print(f"[warn] no pude fijar el enfoque: {exc}")

        # Exposicion corta y ganancia fija: con la camara a 56 FPS el AE
        # puede estirar la exposicion hasta ~17.8 ms, y una pelota rapida
        # sale como un borron alargado. El modelo se entreno con imagenes
        # nitidas, asi que el blur le baja la confianza sin motivo aparente.
        # 2-4 ms suele alcanzar para congelar el movimiento; hay que
        # compensar con ganancia y con buena iluminacion.
        if ganancia is not None and ganancia < 1.0:
            # El minimo de AnalogueGain es 1.0. Mandar 0 hace que libcamera
            # rechace el set_controls ENTERO, y con el se cae tambien el
            # AeEnable: False. Resultado: el auto-exposicion sigue activo y
            # parece que la exposicion fija "no hace nada".
            print(f"[warn] ganancia {ganancia} invalida (minimo 1.0), se usa 1.0")
            ganancia = 1.0

        if exposicion_us is not None:
            controles = {"AeEnable": False, "ExposureTime": int(exposicion_us)}
            if ganancia is not None:
                controles["AnalogueGain"] = float(ganancia)
            try:
                self._picam.set_controls(controles)
                time.sleep(0.5)  # los controles tardan unos frames en aplicarse
                print(
                    f"[cam] pedido: exposicion {exposicion_us} us"
                    + (f", ganancia {ganancia}" if ganancia else "")
                )
                self.reportar_exposicion()
            except Exception as exc:
                print(f"[warn] no pude fijar la exposicion: {exc}")
        else:
            print("[cam] automatico (exposicion, blancos y foco los maneja "
                  "la camara, igual que la preview)")
            self.reportar_exposicion()

        self._ts_anterior = None
        self._seq = 0
        self.total_perdidos = 0
        self.ms_espera = 0.0
        self.ms_copia = 0.0

    def read(self) -> tuple[np.ndarray, dict]:
        """
        Devuelve el frame mas reciente. Los frames que se completaron
        mientras el pipeline estaba ocupado se descartan.
        """
        t0 = time.perf_counter()
        req = self._picam.capture_request()
        t1 = time.perf_counter()
        try:
            frame = req.make_array("main")
            md = req.get_metadata()
        finally:
            # Devolver el buffer YA: si no, la camara se queda sin
            # buffers libres y se frena.
            req.release()
        t2 = time.perf_counter()

        self.ms_espera = (t1 - t0) * 1000.0   # bloqueado esperando el sensor
        self.ms_copia = (t2 - t1) * 1000.0    # memcpy del buffer al array

        ts_ns = int(md.get("SensorTimestamp", time.monotonic_ns()))

        if self._ts_anterior is None:
            dt_ms = 0.0
            perdidos = 0
        else:
            dt_ms = (ts_ns - self._ts_anterior) / 1e6
            # Cuantos periodos de camara pasaron entre entregas.
            perdidos = max(0, int(round(dt_ms / self.periodo_ms)) - 1)

        self._ts_anterior = ts_ns
        self._seq += 1
        self.total_perdidos += perdidos

        return frame, {
            "ts_ns": ts_ns,
            "dt_ms": dt_ms,
            "perdidos": perdidos,
            "seq": self._seq,
            "ms_espera": self.ms_espera,
            "ms_copia": self.ms_copia,
            "exposicion_us": md.get("ExposureTime"),
            "ganancia": md.get("AnalogueGain"),
            "lux": md.get("Lux"),
        }

    def reportar_exposicion(self) -> dict:
        """
        Lee de la METADATA lo que la camara esta haciendo realmente, que no
        siempre es lo que se pidio: los controles se clampean a los rangos
        del sensor sin avisar. Si lo reportado no coincide con lo pedido,
        ahi esta el problema.
        """
        try:
            req = self._picam.capture_request()
            md = req.get_metadata()
            req.release()
        except Exception as exc:
            print(f"[warn] no pude leer metadata: {exc}")
            return {}

        datos = {
            k: md.get(k)
            for k in ("ExposureTime", "AnalogueGain", "DigitalGain", "Lux",
                      "ColourGains", "AeLocked")
            if k in md
        }
        exp = datos.get("ExposureTime")
        ag = datos.get("AnalogueGain")
        dg = datos.get("DigitalGain")
        lux = datos.get("Lux")

        partes = []
        if exp is not None:
            partes.append(f"exposicion {exp} us")
        if ag is not None:
            partes.append(f"ganancia analog {ag:.2f}")
        if dg is not None:
            partes.append(f"digital {dg:.2f}")
        if lux is not None:
            partes.append(f"lux {lux:.0f}")
        print("[cam] real:    " + ", ".join(partes))

        if lux is not None and exp is not None:
            # Referencia gruesa: f/2.2 y sensor lineal.
            if lux > 5000 and exp > 1000:
                print(f"[cam] !! {lux:.0f} lux con {exp} us: la imagen va a")
                print("[cam]    salir QUEMADA. Con sol directo usá 150-400 us.")
            elif lux < 200 and exp < 2000:
                print(f"[cam] !! {lux:.0f} lux con {exp} us: va a salir OSCURA.")
        return datos

    def modos_sensor(self) -> list:
        """
        Modos que libcamera expone para este sensor, leidos en el
        constructor (con la camara ya andando, consultarlos falla con
        'Camera must be stopped before configuring').
        """
        return self._modos

    def verificar_canales(self, muestras: int = 5) -> str:
        """
        Determina empiricamente si el array sale en RGB o en BGR, sin
        confiar en el nombre del formato.

        Apunta la camara a algo predominantemente ROJO y llama a esto.
        Devuelve "RGB" o "BGR" segun en que canal este la mayor energia.
        """
        acum = np.zeros(3, dtype=np.float64)
        for _ in range(muestras):
            f, _ = self.read()
            acum += f.reshape(-1, 3).mean(axis=0)
        acum /= muestras

        print(f"medias por canal (indice 0,1,2): {acum}")
        orden = "RGB" if acum[0] > acum[2] else "BGR"
        print(f"apuntando a algo rojo, el array parece estar en {orden}")
        print("Si el objeto era rojo y dice BGR, hay que hacer cvtColor.")
        return orden

    def close(self) -> None:
        try:
            self._picam.stop()
        finally:
            self._picam.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class ThreadedCameraSource:
    """
    Captura en un hilo de fondo que guarda SOLO el frame mas reciente.

    Por que hace falta: con captura y computo en serie, el ciclo cuesta
    (captura + computo). Copiar 2304x1296x3 son ~9 MB por frame, y ese
    memcpy se suma a la inferencia aunque los dos por separado alcancen
    el framerate objetivo. Con el hilo de fondo, la captura se solapa
    con el computo y el ciclo pasa a costar max(captura, computo).

    Semantica identica a CameraSource: read() devuelve el frame mas
    reciente. Si el consumidor es mas lento, los frames intermedios se
    pisan en memoria y se contabilizan como perdidos.

    Para la medicion con el pin de GPIO no cambia nada: el pin sigue
    envolviendo solo el computo, no la captura.
    """

    def __init__(self, *args, **kwargs):
        import threading

        self._cam = CameraSource(*args, **kwargs)
        self._lock = threading.Lock()
        self._ultimo = None
        self._nuevo = threading.Event()
        self._corriendo = True
        self._descartados_sin_leer = 0

        self._hilo = threading.Thread(target=self._bucle, daemon=True)
        self._hilo.start()

    def _bucle(self):
        while self._corriendo:
            try:
                frame, info = self._cam.read()
            except Exception:
                if not self._corriendo:
                    break
                raise
            with self._lock:
                if self._ultimo is not None and not self._nuevo.is_set():
                    pass
                elif self._ultimo is not None:
                    # habia un frame sin consumir: se pisa
                    self._descartados_sin_leer += 1
                self._ultimo = (frame, info)
                self._nuevo.set()

    def read(self, timeout: float = 2.0):
        """Bloquea hasta que haya un frame nuevo sin consumir."""
        if not self._nuevo.wait(timeout):
            raise TimeoutError("la camara no entrego frames")
        with self._lock:
            frame, info = self._ultimo
            self._nuevo.clear()
        info = dict(info)
        info["pisados"] = self._descartados_sin_leer
        return frame, info

    @property
    def total_perdidos(self) -> int:
        return self._cam.total_perdidos

    def modos_sensor(self):
        return self._cam.modos_sensor()

    def verificar_canales(self, muestras: int = 5) -> str:
        return self._cam.verificar_canales(muestras)

    def close(self) -> None:
        self._corriendo = False
        self._hilo.join(timeout=2.0)
        self._cam.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def extrapolar_centro(
    x: float,
    y: float,
    vx: float,
    vy: float,
    dt_ms: float,
    limites: tuple[int, int],
) -> tuple[float, float]:
    """
    Adelanta el centro del crop segun la velocidad estimada y el tiempo
    transcurrido. Con descarte de frames el dt es variable, asi que la
    velocidad hay que expresarla en px/ms y no en px/frame.

    Sin esto, el crop de TRACK se centra donde estaba la pelota en la
    ultima deteccion, que con frames perdidos puede quedar muy atras.
    """
    w, h = limites
    nx = min(max(x + vx * dt_ms, 0.0), float(w))
    ny = min(max(y + vy * dt_ms, 0.0), float(h))
    return nx, ny


if __name__ == "__main__":
    # Banco de la CAPTURA sola (sin modelo). Separa cuanto se va en
    # esperar al sensor y cuanto en copiar el buffer, y compara captura
    # en serie contra captura en hilo de fondo.
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=56.0)
    ap.add_argument("--exposicion-us", type=int, default=None,
                    help="exposicion fija en microsegundos (ej 3000)")
    ap.add_argument("--ganancia", type=float, default=None)
    ap.add_argument("--ancho", type=int, default=2304)
    ap.add_argument("--alto", type=int, default=1296)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--formato", type=str, default="RGB888")
    ap.add_argument("--buffers", type=int, default=4)
    ap.add_argument("--hilo", action="store_true", help="captura en hilo de fondo")
    ap.add_argument(
        "--carga-ms",
        type=float,
        default=0.0,
        help="simular un pipeline que tarda X ms por frame",
    )
    ap.add_argument("--canales", action="store_true")
    ap.add_argument("--modos", action="store_true", help="listar modos del sensor")
    args = ap.parse_args()

    Clase = ThreadedCameraSource if args.hilo else CameraSource

    with Clase(
        size=(args.ancho, args.alto),
        fps=args.fps,
        formato=args.formato,
        buffer_count=args.buffers,
        exposicion_us=args.exposicion_us,
        ganancia=args.ganancia,
    ) as cam:
        if args.modos:
            modos = cam.modos_sensor()
            print("=" * 66)
            print("  MODOS DE SENSOR QUE EXPONE LIBCAMERA")
            print("=" * 66)
            if not modos:
                print("  (vacio -- proba 'rpicam-hello --list-cameras')")
            print(f"{'size':>14} {'fps':>9} {'bits':>6}  formato")
            print("-" * 66)
            for m in modos:
                size = m.get("size")
                fps = m.get("fps", 0.0)
                marca = "  <-- alcanza lo pedido" if fps >= args.fps else ""
                print(
                    f"{str(size):>14} {fps:9.2f} {m.get('bit_depth', '?'):>6}  "
                    f"{m.get('format', '?')}{marca}"
                )
            print()
            if not any(m.get("fps", 0) >= args.fps for m in modos):
                print(f"NINGUN modo llega a {args.fps:.0f} FPS. El techo esta en el")
                print("sensor/driver, no en tu codigo.")
                print()

        if args.canales:
            cam.verificar_canales()
            print()

        dts, perd, esp, cop = [], [], [], []
        t0 = time.perf_counter()
        for _ in range(args.n):
            f, info = cam.read()
            if args.carga_ms > 0:
                fin = time.perf_counter() + args.carga_ms / 1000.0
                while time.perf_counter() < fin:
                    pass
            dts.append(info["dt_ms"])
            perd.append(info["perdidos"])
            esp.append(info.get("ms_espera", 0.0))
            cop.append(info.get("ms_copia", 0.0))
        total = time.perf_counter() - t0

    d = np.asarray(dts[1:])
    modo = "HILO DE FONDO" if args.hilo else "SERIE"
    periodo = 1000.0 / args.fps

    print("=" * 66)
    print(f"  CAPTURA EN {modo} -- {args.ancho}x{args.alto} {args.formato}")
    print("=" * 66)
    print(f"frame shape: {f.shape} dtype={f.dtype} ({f.nbytes / 1e6:.1f} MB por frame)")
    print(f"pedidos {args.fps:.1f} FPS -> periodo objetivo {periodo:.2f} ms")
    fps_pared = args.n / total
    fps_ts = 1000.0 / np.percentile(d, 50) if d.size else float("nan")
    print(f"ENTREGADOS: {fps_pared:.1f} FPS por reloj de pared")
    print(f"            {fps_ts:.1f} FPS segun timestamps del sensor")
    if args.n < 100:
        print("  !! con --n chico el primero es basura: los frames ya")
        print("     bufferizados vuelven al instante e inflan el promedio.")
        print("     Repetilo con --n 300.")
    elif abs(fps_pared - fps_ts) > 0.1 * fps_ts:
        print("  !! los dos numeros no coinciden: mira el arranque.")
    print()
    print(f"{'metrica':<28} {'p50':>8} {'p90':>8} {'max':>8}   (ms)")
    print("-" * 66)
    for nombre, arr in [
        ("dt entre entregas", d),
        ("esperando al sensor", np.asarray(esp[1:])),
        ("copia make_array", np.asarray(cop[1:])),
    ]:
        if arr.size and arr.max() > 0:
            print(
                f"{nombre:<28} {np.percentile(arr, 50):8.2f} "
                f"{np.percentile(arr, 90):8.2f} {arr.max():8.2f}"
            )
    print()
    print(f"frames perdidos: {sum(perd)} "
          f"({100 * sum(perd) / max(1, sum(perd) + args.n):.1f}%)")
    if args.carga_ms > 0:
        print(f"(simulando {args.carga_ms:.1f} ms de computo por frame)")
    print()
    print("COMO LEERLO:")
    print("  Si 'copia make_array' es alto, el cuello es el memcpy: proba")
    print("  --hilo para solaparlo con el computo, o un formato mas barato.")
    print("  Si 'esperando al sensor' domina y es mayor al periodo objetivo,")
    print("  libcamera no eligio el modo rapido: mira --modos.")
