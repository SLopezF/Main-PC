"""
main_final.py

El sistema. Ata init_sistema, las camaras, la inferencia, el tracker, la
geometria, los sectores, el motor y la GoPro.

    IDLE --iniciar()--> INIT --> BUSCANDO / SIGUIENDO --detener()--> IDLE

`iniciar()` y `detener()` son funciones Python. Que las llame una pagina web,
un boton o una tecla es un detalle posterior.

UN SOLO HILO DE DECISION
El loop principal captura, infiere y decide. Nada mas. El motor tiene su hilo
(`control_motor`) y la GoPro graba sola en su SD. El loop NUNCA espera al
motor: `apuntar()` deja el objetivo en un buzon y vuelve en microsegundos.

Medido: `apuntar()` tarda 0.026 ms en la Pi con el fierro real. Si en cambio
el loop esperara el fin del movimiento, perderia medio segundo de vision
justo cuando la jugada se esta yendo.

MISMO CODIGO DE DECISION QUE EL REPLAY
`pipeline.procesar_frame()` es el mismo que corre `replay.py`. Por eso los
umbrales ajustados mirando material grabado significan algo en la cancha, y
por eso el CSV tiene las mismas columnas: el analisis de la tesis es un solo
script para los dos.

SI LA GOPRO FALLA, EL SISTEMA SIGUE
No aborta: emite una alerta y sigue trackeando. Perder la grabacion es malo;
perder tambien el tracking (y los datos del CSV) es peor.

PANEL WEB
Por defecto se levanta `web_control.py` en el puerto 5000 y desde ahi se
arranca, se detiene y se saca una foto. `iniciar()` y `detener()` se le pasan
como hooks (`web_control.register_session_hooks`), asi que el panel y la
consola manejan EL MISMO sistema: no hay dos caminos que puedan desincronizar
el estado.

Si flask no esta instalado, o se pasa --sin-web, el panel simplemente no
levanta y todo lo demas funciona igual. El panel es una comodidad, no una
dependencia.

USO
    python3 main_final.py                    # panel en http://<ip>:5000
    python3 main_final.py --sin-web          # ENTER arranca, ENTER detiene
    python3 main_final.py --minutos 10 --sin-gopro
    python3 main_final.py --sin-motor --sin-gopro    # solo deteccion
"""

import csv
import signal
import threading
import time
from enum import Enum

import config
import config_hw as chw
import geometria
import pipeline
import preproceso
from geometria import SelectorCamara
from sectores import Sectorizador
from tracker import Mode, Tracker


class Estado(Enum):
    IDLE = 0
    INIT = 1
    GRABANDO = 2


class MainFinal:
    """
    El sistema completo. Se construye una vez y se puede arrancar y detener
    varias veces; el hardware se abre en iniciar() y se cierra en detener().
    """

    def __init__(self, con_motor: bool = True, con_gopro: bool = True,
                 con_csv: bool = True, camara_inicial: int = 0,
                 ruta_csv: str | None = None, verbose: bool = True,
                 debug_mp4: str | None = None, escala: float | None = None,
                 lineas_sectores: bool = False):
        self.con_motor = con_motor
        self.con_gopro = con_gopro
        self.con_csv = con_csv
        self.camara_inicial = int(camara_inicial)
        self.ruta_csv = ruta_csv or config.LOG_CSV_PATH
        self.verbose = verbose

        # Video anotado. Cuesta un resize y un encode por frame, asi que NO se
        # deja prendido en una corrida de partido: es para debuggear.
        self.debug_mp4 = debug_mp4
        self.escala = (escala if escala is not None
                       else getattr(config, "DEBUG_VIDEO_SCALE", 0.25))
        self.lineas_sectores = bool(lineas_sectores)
        self._video = None

        self.estado = Estado.IDLE
        self.alertas: list[str] = []

        self.web = None              # modulo web_control, si el panel levanto
        self._hilo_web = None

        self._parar = threading.Event()
        self._hilo = None

        # Se llenan en iniciar()
        self.sistema = None          # init_sistema.Sistema
        self.control = None          # control_motor.ControlMotor
        self.camaras = {}            # {indice: fuente}
        self.hailo = None
        self.tracker = None
        self.selector = None
        self.sectorizador = None
        self.omega = None

        # Estadisticas de la corrida
        self.frames = 0
        self.aceptados = 0
        self.cambios_sector = 0
        self.cambios_camara = 0
        self.t0 = None

    # ------------------------------------------------------------- alertas
    def _alerta(self, texto: str) -> None:
        """Una alerta no aborta: se registra y el sistema sigue."""
        self.alertas.append(texto)
        print(f"[ALERTA] {texto}")

    def _log(self, texto: str) -> None:
        if self.verbose:
            print(texto)

    # ------------------------------------------------------------- arranque
    def iniciar(self) -> bool:
        """
        IDLE -> INIT -> GRABANDO. Devuelve False si el arranque fallo.

        Bloquea mientras dura el init (homing, verificaciones): son segundos y
        pasa una sola vez. El loop arranca despues, en su propio hilo.
        """
        if self.estado != Estado.IDLE:
            print(f"ya esta en {self.estado.name}")
            return False

        self.estado = Estado.INIT
        self._parar.clear()
        self.alertas = []

        try:
            self._abrir_hardware()
        except Exception as exc:
            self._alerta(f"arranque fallido: {exc}")
            self.detener()
            return False

        self.estado = Estado.GRABANDO
        self.t0 = time.monotonic()
        self._hilo = threading.Thread(target=self._loop, daemon=True,
                                      name="decision")
        self._hilo.start()
        self._log("\n>>> GRABANDO\n")
        return True

    def _abrir_hardware(self) -> None:
        import init_sistema

        # --- motor y encoder, con homing verificado
        if self.con_motor:
            self._log("--- init de sistema")
            self.sistema = init_sistema.inicializar(
                con_motor=True, con_gopro=False)

            import control_motor
            self.control = control_motor.ControlMotor(
                self.sistema.motor, self.sistema.encoder,
                offset_encoder=self.sistema.offset_encoder)
        else:
            self._log("--- motor SALTEADO")

        # --- inferencia
        self._log("--- inferencia")
        from inferencia import abrir as abrir_inferencia
        self.hailo = abrir_inferencia()
        self._log(self.hailo.describe())
        self.model_hw = preproceso.get_model_hw(self.hailo)

        # --- camaras
        self._log("--- camaras")
        import fuente
        indices = (chw.CAMARAS if chw.CAM_MANTENER_ABIERTAS
                   else (self.camara_inicial,))
        for i in indices:
            self.camaras[i] = fuente.abrir_fuente(indice=i)
        self._log(f"    abiertas: {sorted(self.camaras)}")

        # --- logica
        primera = self.camaras[self.camara_inicial]
        frame, _ = primera.read()
        h, w = frame.shape[:2]
        self.frame_hw = (h, w)
        tiles = preproceso.search_tiles(self.frame_hw, self.model_hw)
        self._log(f"    frame {w}x{h}, {len(tiles)} tiles de "
                  f"{tiles[0][2]}x{tiles[0][3]}")

        self.tracker = Tracker(
            n_tiles=len(tiles), camaras=chw.CAMARAS,
            camara_inicial=self.camara_inicial,
            fn_tile=lambda x, y: preproceso.tile_para_punto(
                self.frame_hw, self.model_hw, x, y))
        self.selector = SelectorCamara(self.camara_inicial)
        self.sectorizador = Sectorizador()
        self.omega = pipeline.CalculadorOmega()

        # --- gopro: lo ultimo, para que un fallo suyo no cueste el resto
        if self.con_gopro:
            self._log("--- gopro")
            import hw_falsos
            GoPro, _ = hw_falsos.cargar("gopro_lib", "GoPro", hw_falsos.GoPro)
            self.gopro = GoPro()
            if self.gopro.init(timeout=60.0) and self.gopro.iniciar_grabacion():
                self._log(f"    grabando. {self.gopro.estado()}")
            else:
                self._alerta("la GoPro no arranco: el sistema va a trackear "
                             "pero NO va a grabar video.")
        else:
            self.gopro = None

    # ----------------------------------------------------------------- loop
    def _loop(self) -> None:
        archivo = escritor = None
        if self.con_csv:
            archivo = open(self.ruta_csv, "w", newline="", encoding="utf-8")
            escritor = csv.DictWriter(archivo, fieldnames=pipeline.COLUMNAS)
            escritor.writeheader()

        activa = self.camara_inicial
        ts_ultimo_movimiento = None

        try:
            while not self._parar.is_set():
                cam = self.camaras.get(activa)
                if cam is None:
                    # CAM_MANTENER_ABIERTAS=False: se abre la que haga falta
                    import fuente
                    for otra in list(self.camaras):
                        self.camaras.pop(otra).close()
                    cam = self.camaras[activa] = fuente.abrir_fuente(
                        indice=activa)

                try:
                    frame, info = cam.read()
                except StopIteration:
                    break
                except Exception as exc:
                    self._alerta(f"fallo la captura de la camara {activa}: {exc}")
                    time.sleep(0.1)
                    continue

                info["camara"] = activa
                fila, det, to_global, r, rs, _ = pipeline.procesar_frame(
                    frame, info, self.hailo, self.model_hw,
                    self.tracker, self.sectorizador, self.omega)

                self.frames += 1
                if r.aceptado:
                    self.aceptados += 1

                t_s = info["ts_ns"] / 1e9

                # --- que camara mirar el proximo frame
                if r.aceptado:
                    angulo = float(fila["angulo"])
                    nueva = self.selector.actualizar(angulo, r.conf, t_s)
                else:
                    nueva = self.selector.actualizar(None, None, t_s)

                if nueva != activa:
                    self._cambiar_camara(nueva, activa, t_s)
                    activa = nueva

                # --- mover el motor
                # El cambio se CUENTA y se LOGUEA aunque no haya motor: con
                # --sin-motor la decision se sigue tomando, y es justo lo que
                # se quiere ver al debuggear.
                if rs is not None and rs.cambio:
                    self.cambios_sector += 1
                    ts_ultimo_movimiento = t_s
                    destino = ("motor a" if self.control is not None
                               else "(sin motor) iria a")
                    self._log(f"  sector {rs.sector} -> {destino} "
                              f"{rs.angulo_objetivo:.0f} deg  ({rs.motivo})")
                    if self.control is not None:
                        self.control.apuntar(rs.angulo_objetivo)

                # --- sin ver nada por mucho tiempo: al centro
                sin_ver = self.tracker.segundos_sin_deteccion(info["ts_ns"])
                if (self.control is not None
                        and sin_ver >= chw.S_SEARCH_A_CENTRO
                        and (ts_ultimo_movimiento is None
                             or t_s - ts_ultimo_movimiento >= chw.S_SEARCH_A_CENTRO)):
                    self.control.apuntar(90.0)
                    self.sectorizador.forzar(90.0, t_s)
                    ts_ultimo_movimiento = t_s
                    self._log(f"  {sin_ver:.0f} s sin ver la pelota: al centro")

                # --- alertas del hilo del motor
                if self.control is not None:
                    a = self.control.alerta()
                    if a:
                        self._alerta(a)
                        self.control.limpiar_alerta()

                if escritor is not None:
                    escritor.writerow(fila)

                if self.debug_mp4:
                    self._anotar(frame, fila, det, to_global)

        except Exception as exc:
            self._alerta(f"el loop de decision murio: {exc}")
        finally:
            if archivo is not None:
                archivo.close()
            if self._video is not None:
                self._video.release()
                self._video = None
                print(f"    video de debug -> {self.debug_mp4}")

    def _anotar(self, frame, fila, det, to_global) -> None:
        """
        Escribe el frame anotado. Se usa el MISMO dibujante que replay.py para
        que los dos videos se lean igual.
        """
        import cv2

        import replay

        try:
            vis = replay.anotar(frame, fila, det, to_global, self.escala,
                                self.lineas_sectores)
            if self._video is None:
                h, w = vis.shape[:2]
                self._video = cv2.VideoWriter(
                    self.debug_mp4, cv2.VideoWriter_fourcc(*"mp4v"),
                    float(getattr(config, "CAM_FPS", 30.0)), (w, h))
                if not self._video.isOpened():
                    self._alerta(f"no pude abrir '{self.debug_mp4}' para "
                                 f"escribir; sigo sin video")
                    self.debug_mp4 = None
                    self._video = None
                    return
            self._video.write(vis)
        except Exception as exc:
            self._alerta(f"fallo el video de debug: {exc}")
            self.debug_mp4 = None

    def _cambiar_camara(self, nueva: int, vieja: int, t_s: float) -> None:
        """
        Handover. El filtro se reinicia porque un (x, y) de una camara no
        significa nada en la otra, y el ROI de la nueva se siembra con el
        angulo actual convertido a pixeles de esa camara.
        """
        x = y = None
        if self.tracker.ultima_x is not None:
            ancho = self.frame_hw[1]
            angulo = geometria.pixel_a_angulo(
                self.tracker.ultima_x, ancho, vieja)
            x = geometria.angulo_a_pixel(angulo, ancho, nueva)
            y = self.tracker.ultima_y
            if not (0 <= x < ancho):
                x = y = None          # el angulo no se ve desde la nueva

        self.tracker.cambiar_camara(nueva, x, y)
        self.omega.reiniciar()
        self.cambios_camara += 1
        self._log(f"  camara {vieja} -> {nueva}  ({self.selector.motivo})")

    # -------------------------------------------------------------- parada
    def detener(self) -> None:
        """Deja el sistema limpio: motor deshabilitado, camaras cerradas,
        GoPro detenida. Se puede llamar aunque el arranque haya fallado."""
        if self.estado == Estado.IDLE and self._hilo is None:
            return

        self._log("\n>>> deteniendo...")
        self._parar.set()
        if self._hilo is not None:
            self._hilo.join(timeout=5.0)
            self._hilo = None

        if getattr(self, "gopro", None) is not None:
            try:
                archivo = self.gopro.detener_grabacion()
                self._log(f"    gopro -> {archivo}")
                # El panel expone esto en /api/last_file, que es lo que mira
                # gopro_file_transfer.py desde la compu para bajar el archivo.
                if self.web is not None and archivo:
                    try:
                        self.web.publish_recorded_file(archivo)
                    except Exception as exc:
                        self._alerta(f"no pude publicar el archivo en el "
                                     f"panel: {exc}")
            except Exception as exc:
                self._alerta(f"no pude detener la GoPro: {exc}")
            finally:
                try:
                    self.gopro.close()
                except Exception:
                    pass
            self.gopro = None

        if self.control is not None:
            try:
                self.control.apuntar(90.0)      # dejarlo mirando al frente
                self.control.esperar_ocioso(timeout=15.0)
            except Exception:
                pass
            self.control.parar()
            self.control = None

        if self.sistema is not None:
            self.sistema.close()                # deshabilita el motor
            self.sistema = None

        for i, cam in list(self.camaras.items()):
            try:
                cam.close()
            except Exception:
                pass
        self.camaras = {}

        if self.hailo is not None:
            try:
                self.hailo.close()
            except Exception:
                pass
            self.hailo = None

        self.estado = Estado.IDLE
        self._log(self.resumen())

    # ---------------------------------------------------------------- panel
    def levantar_web(self, puerto: int = 5000) -> bool:
        """
        Levanta el panel de web_control.py en un hilo daemon y le registra
        `iniciar` y `detener` como hooks.

        Devuelve False si no se pudo (flask ausente, puerto ocupado): el panel
        es una comodidad y su ausencia NO tiene que impedir que el sistema
        corra. Por eso todo esto esta envuelto en un try y solo emite alerta.

        No se llama a `web_control.main()`: ese hace `startup_checks()`, que
        BLOQUEA esperando a la GoPro, y ademas levanta Flask en el hilo
        principal. Aca se necesita al reves: Flask de fondo, y que el arranque
        del sistema lo decida el usuario desde el panel.
        """
        try:
            import web_control
        except Exception as exc:
            self._alerta(f"no levanto el panel web ({type(exc).__name__}: "
                         f"{exc}). Si falta flask: pip install flask")
            return False

        try:
            web_control.register_session_hooks(self.iniciar, self.detener)
            self.web = web_control

            def _servir():
                # threaded=True para que /api/status (que el panel pollea cada
                # 2 s) no quede detras de un request largo.
                web_control.app.run(host="0.0.0.0", port=puerto,
                                    debug=False, use_reloader=False,
                                    threaded=True)

            self._hilo_web = threading.Thread(target=_servir, daemon=True,
                                              name="web")
            self._hilo_web.start()
            time.sleep(0.5)
            if not self._hilo_web.is_alive():
                raise RuntimeError("el hilo del servidor murio al arrancar")

            # El monitor de conexion refresca bateria y SD en el panel. Solo
            # tiene sentido si de verdad vamos a hablar con la camara.
            if self.con_gopro:
                try:
                    web_control.connection_monitor.start()
                except Exception as exc:
                    self._alerta(f"el monitor de conexion no arranco: {exc}")

            self._log(f"\n>>> panel web en http://0.0.0.0:{puerto}")
            self._log("    (usá la IP de la Pi desde el navegador; es HTTP, "
                      "no HTTPS)")
            return True
        except Exception as exc:
            self.web = None
            self._alerta(f"no pude levantar el panel web: {exc}")
            return False

    def resumen(self) -> str:
        dt = (time.monotonic() - self.t0) if self.t0 else 0.0
        pct = 100.0 * self.aceptados / self.frames if self.frames else 0.0
        lineas = [
            "", "=" * 60, "RESUMEN DE LA CORRIDA", "=" * 60,
            f"duracion            {dt:.0f} s",
            f"frames              {self.frames}"
            f"   ({self.frames / dt if dt else 0:.1f} fps)",
            f"con deteccion       {self.aceptados}  ({pct:.1f}%)",
            f"cambios de sector   {self.cambios_sector}",
            f"cambios de camara   {self.cambios_camara}",
        ]
        if self.con_csv:
            lineas.append(f"csv                 {self.ruta_csv}")
        if self.debug_mp4:
            lineas.append(f"video               {self.debug_mp4}")
        if self.alertas:
            lineas.append("")
            lineas.append(f"ALERTAS ({len(self.alertas)}):")
            for a in dict.fromkeys(self.alertas):     # sin repetir
                lineas.append(f"  - {a}")
        lineas.append("=" * 60)
        return "\n".join(lineas)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sin-motor", action="store_true")
    ap.add_argument("--sin-gopro", action="store_true")
    ap.add_argument("--sin-csv", action="store_true")
    ap.add_argument("--camara", type=int, default=0, choices=[0, 1])
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--mascara-y", type=int, default=None,
                    help="pinta de negro todo lo que este ARRIBA de esta fila, "
                         "para tapar las luces del fondo de noche. Empezá "
                         "chico y subilo. TEMPORAL: recorta donde puede estar "
                         "la pelota.")
    ap.add_argument("--debug-mp4", type=str, nargs="?",
                    const="/dev/shm/debug_main.mp4", default=None,
                    help="video anotado (cuesta un resize y un encode por "
                         "frame: para debuggear, no para el partido)")
    ap.add_argument("--escala", type=float, default=None)
    ap.add_argument("--lineas-sectores", action="store_true",
                    help="dibujar las divisiones de sector en el video")
    ap.add_argument("--sin-web", action="store_true",
                    help="no levantar el panel web; arrancar y detener por "
                         "consola con ENTER")
    ap.add_argument("--puerto-web", type=int, default=5000)
    ap.add_argument("--minutos", type=float, default=None,
                    help="detener solo despues de N minutos")
    args = ap.parse_args()

    if args.mascara_y is not None:
        config.MASCARA_Y = args.mascara_y
        print(f"MASCARA: negro por encima de la fila {args.mascara_y} "
              f"(de {config.CAM_ALTO}). Es un parche temporal.")

    sistema = MainFinal(
        con_motor=not args.sin_motor,
        con_gopro=not args.sin_gopro,
        con_csv=not args.sin_csv,
        camara_inicial=args.camara,
        ruta_csv=args.csv,
        debug_mp4=args.debug_mp4,
        escala=args.escala,
        lineas_sectores=args.lineas_sectores,
    )

    # Ctrl+C tiene que dejar el fierro limpio, no abortar a la mitad.
    def _senal(sig, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _senal)

    hay_web = False
    if not args.sin_web:
        hay_web = sistema.levantar_web(args.puerto_web)

    try:
        if hay_web and not args.minutos:
            # El arranque y la parada los decide el panel. Aca solo se espera,
            # sin input(): con el panel abierto, un input() bloqueado en la
            # consola confunde mas de lo que ayuda.
            print("\nEsperando ordenes del panel web. Ctrl+C para salir.")
            while True:
                time.sleep(0.5)
        elif args.minutos:
            if not sistema.iniciar():
                return 1
            print(f"corriendo {args.minutos:.0f} minutos (Ctrl+C para cortar)")
            fin = time.monotonic() + args.minutos * 60
            while time.monotonic() < fin and sistema.estado == Estado.GRABANDO:
                time.sleep(0.5)
        else:
            # input() vive ACA, nunca dentro del loop de decision.
            input("ENTER para arrancar... ")
            if not sistema.iniciar():
                return 1
            input("ENTER para detener... ")
    except KeyboardInterrupt:
        print("\ninterrumpido")
    finally:
        sistema.detener()

    return 1 if sistema.alertas else 0


if __name__ == "__main__":
    raise SystemExit(main())
