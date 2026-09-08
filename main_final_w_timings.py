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
Ademas de la consola, el sistema levanta el panel de web_control.py en el
puerto 5000. Los botones Iniciar/Detener del panel llaman a las MISMAS
`iniciar()` / `detener()` de aca: la pagina es otro disparador, no otro
sistema. Por eso el panel NO graba por su cuenta cuando lo levanta este
main (ver register_session_hooks en web_control.py): un solo dueno de la
camara evita mandarle comandos duplicados.

Al terminar una grabacion, el path del archivo en la SD se publica en
/api/last_file, que es lo que gopro_file_transfer.py (corriendo en la
computadora) poolea para bajarlo directo de la camara y despues pedir el
borrado. Esa parte no cambia.

USO
    python3 main_final.py                    # ENTER arranca, ENTER detiene
    python3 main_final.py --minutos 10 --sin-gopro
    python3 main_final.py --sin-motor --sin-gopro    # solo deteccion
    python3 main_final.py --sin-web           # como era antes, sin panel
    python3 main_final.py --puerto-web 8080   # panel en otro puerto
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


# --------------------------------------------------------------------------- #
# GoPro real
# --------------------------------------------------------------------------- #

class GoProReal:
    """
    Adaptador sobre gopro_controller.py con la MISMA interfaz que esperaba
    hw_falsos.GoPro (init / iniciar_grabacion / detener_grabacion / estado /
    close), asi el resto de main_final no cambia.

    La diferencia con la version falsa es que esta habla con la camara de
    verdad: se conecta a su Wi-Fi, le manda el shutter, y al detener devuelve
    el path real del archivo en la SD ("FOLDER/ARCHIVO.MP4"), que es lo que
    despues se publica para que la computadora lo baje.
    """

    def __init__(self):
        import gopro_controller as gc
        self._gc = gc
        self._cam = None

    def init(self, timeout: float = 60.0) -> bool:
        """
        Espera a que la camara este conectada y respondiendo, pero con un
        limite: wait_for_gopro() de gopro_controller bloquea para siempre, y
        aca no queremos que una camara apagada cuelgue todo el arranque del
        sistema (la regla es "si la GoPro falla, el sistema sigue").
        """
        gc = self._gc
        fin = time.monotonic() + timeout
        while time.monotonic() < fin:
            if not gc.is_connected_to_gopro_wifi():
                if gc.gopro_ssid_visible():
                    gc.connect_to_gopro_wifi()
                else:
                    time.sleep(2.0)
                    continue
            try:
                cam = gc.create_camera()
                if gc.gopro_api_reachable(cam):
                    self._cam = cam
                    gc.sync_camera_clock(cam)
                    return True
            except Exception:
                pass
            time.sleep(2.0)
        return False

    def iniciar_grabacion(self) -> bool:
        if self._cam is None:
            return False
        try:
            self._gc.start_recording(self._cam)
            return True
        except Exception:
            return False

    def detener_grabacion(self):
        """Devuelve el path en la SD ('FOLDER/ARCHIVO.MP4') o None."""
        if self._cam is None:
            return None
        return self._gc.stop_recording(self._cam)

    def estado(self) -> str:
        if self._cam is None:
            return "sin camara"
        bat = self._gc.get_battery_level(self._cam)
        sd = self._gc.get_remaining_sd_capacity(self._cam)
        return f"bateria {bat}% | SD libre {sd}"

    def close(self) -> None:
        self._cam = None


class MainFinal:
    """
    El sistema completo. Se construye una vez y se puede arrancar y detener
    varias veces; el hardware se abre en iniciar() y se cierra en detener().
    """

    def __init__(self, con_motor: bool = True, con_gopro: bool = True,
                 con_csv: bool = True, camara_inicial: int = 0,
                 ruta_csv: str | None = None, verbose: bool = True,
                 debug_mp4: str | None = None, escala: float | None = None,
                 lineas_sectores: bool = False, gopro_falsa: bool = False,
                 on_archivo_listo=None, on_alerta=None,
                 ruta_lat: str | None = None):
        self.con_motor = con_motor
        self.con_gopro = con_gopro
        self.con_csv = con_csv
        self.camara_inicial = int(camara_inicial)
        self.ruta_csv = ruta_csv or config.LOG_CSV_PATH
        self.verbose = verbose

        # GoPro simulada (hw_falsos). Solo para probar el resto del sistema
        # sin camara; con esto NO hay archivo real que transferir.
        self.gopro_falsa = bool(gopro_falsa)

        # Callbacks opcionales hacia el panel web. Si no hay panel, quedan en
        # None y el sistema se comporta exactamente como antes.
        self._on_archivo_listo = on_archivo_listo
        self._on_alerta = on_alerta

        # Video anotado. Cuesta un resize y un encode por frame, asi que NO se
        # deja prendido en una corrida de partido: es para debuggear.
        self.debug_mp4 = debug_mp4
        self.escala = (escala if escala is not None
                       else getattr(config, "DEBUG_VIDEO_SCALE", 0.25))
        self.lineas_sectores = bool(lineas_sectores)
        self._video = None

        self.estado = Estado.IDLE
        self.alertas: list[str] = []

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

        # --- latencias (ver seccion LATENCIA abajo)
        # captura del frame -> decision tomada, en TODOS los frames
        self.lat_decision_ms: list[float] = []
        # captura del frame -> apuntar() enviado, SOLO en los frames que
        # movieron el motor (en el resto no hay "hasta el motor" que medir)
        self.lat_motor_ms: list[float] = []
        self.ruta_lat = ruta_lat
        self._reloj_avisado = False

    # ------------------------------------------------------------- alertas
    def _alerta(self, texto: str) -> None:
        """Una alerta no aborta: se registra y el sistema sigue."""
        self.alertas.append(texto)
        print(f"[ALERTA] {texto}")
        if self._on_alerta is not None:
            try:
                self._on_alerta(texto)
            except Exception:
                pass          # el panel nunca puede tumbar al sistema

    def _log(self, texto: str) -> None:
        if self.verbose:
            print(texto)

    # ------------------------------------------------------------ latencia
    def _medir(self, info: dict, destino: list) -> None:
        """
        Guarda (ahora - ts_de_captura) en ms.

        `info["ts_ns"]` tiene que venir del MISMO reloj que
        time.monotonic_ns(), o la resta no significa nada. En Linux los
        drivers de camara suelen usar CLOCK_MONOTONIC, que es justo ese,
        pero no esta garantizado: si el timestamp viniera de otro reloj
        (epoch, CLOCK_BOOTTIME, o el reloj de la camara), las diferencias
        salen negativas o gigantes. Por eso se valida una sola vez en vez
        de confiar y publicar numeros sin sentido.
        """
        ts = info.get("ts_ns")
        if ts is None:
            return
        ms = (time.monotonic_ns() - ts) / 1e6

        if not self._reloj_avisado and (ms < 0 or ms > 10_000):
            self._reloj_avisado = True
            self._alerta(
                f"las latencias no son confiables: la primera dio {ms:.0f} ms. "
                f"info['ts_ns'] no parece estar en el mismo reloj que "
                f"time.monotonic_ns(); revisar de donde sale el timestamp en "
                f"fuente.py antes de usar estos numeros.")

        destino.append(ms)

    @staticmethod
    def _percentiles(valores: list) -> str:
        """p50 / p95 / max, en una linea. Sin numpy: es una lista corta."""
        if not valores:
            return "sin datos"
        v = sorted(valores)
        n = len(v)
        p50 = v[n // 2]
        p95 = v[min(n - 1, int(0.95 * n))]
        return (f"p50 {p50:6.1f} ms   p95 {p95:6.1f} ms   "
                f"max {v[-1]:6.1f} ms   (n={n})")

    def _volcar_latencias(self) -> None:
        """
        Deja las latencias en su propio CSV, aparte del de frames.

        Va en un archivo separado a proposito: el CSV de frames comparte
        columnas con replay.py (es lo que hace que el analisis de la tesis
        sea un solo script), y meterle una columna extra rompe esa simetria.
        """
        if not self.ruta_lat:
            return
        try:
            with open(self.ruta_lat, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["tipo", "latencia_ms"])
                for ms in self.lat_decision_ms:
                    w.writerow(["decision", f"{ms:.3f}"])
                for ms in self.lat_motor_ms:
                    w.writerow(["motor", f"{ms:.3f}"])
            self._log(f"    latencias -> {self.ruta_lat}")
        except Exception as exc:
            self._alerta(f"no pude escribir las latencias: {exc}")

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
            if self.gopro_falsa:
                # Camino explicito de simulacion, solo bajo pedido: sirve para
                # probar motor/vision sin camara, pero NO produce archivo real.
                import hw_falsos
                GoPro, _ = hw_falsos.cargar("gopro_lib", "GoPro", hw_falsos.GoPro)
                self.gopro = GoPro()
                self._alerta("usando GoPro SIMULADA (--gopro-falsa): no se va "
                             "a grabar ni transferir ningun archivo real.")
            else:
                self.gopro = GoProReal()

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

                # --- LATENCIA: captura -> decision tomada.
                # Mide todo el pipeline de vision (captura + preproceso +
                # Hailo + tracker + sectorizador) para ESTE frame. Se toma
                # apenas vuelve procesar_frame, antes de cualquier print o
                # escritura, para no contaminar el numero con el I/O.
                self._medir(info, self.lat_decision_ms)

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
                    # LATENCIA: captura -> orden al motor. Se toma ANTES del
                    # print y del apuntar(), asi el numero es "cuanto tardo el
                    # sistema en decidir el movimiento", sin el costo de
                    # loguearlo por consola.
                    self._medir(info, self.lat_motor_ms)
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
                # Publicar el path es lo que dispara toda la transferencia:
                # gopro_file_transfer.py (en la compu) lo ve por /api/last_file,
                # lo baja directo de la camara y despues pide el borrado.
                if archivo and self._on_archivo_listo is not None:
                    try:
                        self._on_archivo_listo(archivo)
                    except Exception as exc:
                        self._alerta(f"no pude publicar el archivo al panel: {exc}")
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
        self._volcar_latencias()
        self._log(self.resumen())

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
            "",
            "LATENCIA (desde la captura del frame)",
            f"  hasta la decision   {self._percentiles(self.lat_decision_ms)}",
            f"  hasta el motor      {self._percentiles(self.lat_motor_ms)}",
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
# Panel web
# --------------------------------------------------------------------------- #
#
# Tres cosas distintas, que corren en tres lugares distintos:
#
#   1. CHEQUEOS DE ARRANQUE (una sola vez, bloqueantes, antes de todo):
#      confirmar que la GoPro responde, sincronizar su reloj, leer bateria
#      y SD. Van en levantar_panel(), ANTES de arrancar los loops.
#
#   2. LOOPS CONTINUOS (arrancan una vez y corren solos):
#        - connection_monitor: su propio thread, vigila la conexion con la
#          camara cada CONNECTION_CHECK_INTERVAL segundos.
#        - app.run() de Flask: atiende los botones del panel y los polls de
#          gopro_file_transfer.py.
#        - MainFinal._loop: el hilo de decision, que ya existia.
#      Ninguno necesita un while nuevo aca: cada uno trae el suyo.
#
#   3. LOOP DE LA CLI (el de siempre): el input()/--minutos del main, que
#      sigue viviendo en el hilo principal y no cambia.


def levantar_panel(sistema: "MainFinal", puerto: int = 5000,
                   chequear_gopro: bool = True):
    """
    Registra a `sistema` como dueno de la camara, corre los chequeos de
    arranque y levanta el panel en un thread daemon. Devuelve el modulo
    web_control (o None si no se pudo levantar: el panel es un extra, su
    fallo no debe impedir que el sistema corra por consola).
    """
    try:
        import web_control as wc
    except Exception as exc:
        print(f"[ALERTA] no pude importar el panel web: {exc}")
        return None

    # 1) Los botones del panel llaman a ESTE sistema, no graban por su cuenta.
    wc.register_session_hooks(start=sistema.iniciar, stop=sistema.detener)

    # 2) Chequeos de arranque, una sola vez y bloqueantes. Con --sin-gopro no
    #    tiene sentido esperar a una camara que no vamos a usar.
    if chequear_gopro:
        try:
            wc.startup_checks()
        except Exception as exc:
            print(f"[ALERTA] chequeo de arranque de la GoPro fallido: {exc}")

    # 3) Loops continuos del panel.
    try:
        wc.connection_monitor.start()
    except Exception as exc:
        print(f"[ALERTA] no arranco el monitor de conexion: {exc}")

    hilo = threading.Thread(
        target=lambda: wc.app.run(host="0.0.0.0", port=puerto,
                                  debug=False, use_reloader=False),
        daemon=True, name="panel-web")
    hilo.start()
    print(f"panel web en http://0.0.0.0:{puerto}")
    return wc


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sin-motor", action="store_true")
    ap.add_argument("--sin-gopro", action="store_true")
    ap.add_argument("--sin-csv", action="store_true")
    ap.add_argument("--sin-web", action="store_true",
                    help="no levantar el panel web (comportamiento anterior)")
    ap.add_argument("--puerto-web", type=int, default=5000)
    ap.add_argument("--gopro-falsa", action="store_true",
                    help="usar la GoPro simulada de hw_falsos en vez de la real")
    ap.add_argument("--camara", type=int, default=0, choices=[0, 1])
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--lat-csv", type=str, nargs="?",
                    const="/dev/shm/latencias.csv", default=None,
                    help="volcar las latencias medidas a un CSV aparte")
    ap.add_argument("--debug-mp4", type=str, nargs="?",
                    const="/dev/shm/debug_main.mp4", default=None,
                    help="video anotado (cuesta un resize y un encode por "
                         "frame: para debuggear, no para el partido)")
    ap.add_argument("--escala", type=float, default=None)
    ap.add_argument("--lineas-sectores", action="store_true",
                    help="dibujar las divisiones de sector en el video")
    ap.add_argument("--minutos", type=float, default=None,
                    help="detener solo despues de N minutos")
    args = ap.parse_args()

    # El panel se conecta despues de construir el sistema (necesita pasarle
    # sus hooks), pero el sistema necesita las callbacks del panel para
    # publicar el archivo. Se resuelve con una caja que se llena mas abajo.
    _panel = {"wc": None}

    def _publicar_archivo(camera_file_path: str) -> None:
        if _panel["wc"] is not None:
            _panel["wc"].publish_recorded_file(camera_file_path)

    def _publicar_alerta(texto: str) -> None:
        if _panel["wc"] is not None:
            _panel["wc"].set_state(message=f"[ALERTA] {texto}")

    sistema = MainFinal(
        con_motor=not args.sin_motor,
        con_gopro=not args.sin_gopro,
        con_csv=not args.sin_csv,
        camara_inicial=args.camara,
        ruta_csv=args.csv,
        debug_mp4=args.debug_mp4,
        escala=args.escala,
        lineas_sectores=args.lineas_sectores,
        gopro_falsa=args.gopro_falsa,
        ruta_lat=args.lat_csv,
        on_archivo_listo=_publicar_archivo,
        on_alerta=_publicar_alerta,
    )

    # Ctrl+C tiene que dejar el fierro limpio, no abortar a la mitad.
    def _senal(sig, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _senal)

    # Panel web: se levanta ANTES de arrancar la sesion, para poder disparar
    # iniciar() desde el navegador sin tocar la consola.
    if not args.sin_web:
        _panel["wc"] = levantar_panel(
            sistema, puerto=args.puerto_web,
            chequear_gopro=(not args.sin_gopro and not args.gopro_falsa))

    try:
        if args.minutos:
            if not sistema.iniciar():
                return 1
            print(f"corriendo {args.minutos:.0f} minutos (Ctrl+C para cortar)")
            fin = time.monotonic() + args.minutos * 60
            while time.monotonic() < fin and sistema.estado == Estado.GRABANDO:
                time.sleep(0.5)
        elif args.sin_web:
            # Sin panel: el flujo de siempre, tal cual estaba.
            # input() vive ACA, nunca dentro del loop de decision.
            input("ENTER para arrancar... ")
            if not sistema.iniciar():
                return 1
            input("ENTER para detener... ")
        else:
            # Con panel, la sesion puede arrancar por consola O por el boton
            # de la pagina, y hay que esperar a los DOS. Dos cuidados:
            #
            #  - Un solo lector de stdin para toda la corrida. Si se abre un
            #    input() por cada espera, los threads se pelean por el stdin
            #    y el ENTER de "detener" se lo puede quedar el lector viejo.
            #
            #  - Esperar a GRABANDO, no a "distinto de IDLE". iniciar() pasa
            #    primero por INIT (homing, camaras, Hailo, GoPro), que tarda
            #    segundos; si la consola sale de la espera ahi, ve un estado
            #    que todavia no es GRABANDO, da la corrida por terminada y
            #    llama a detener() apenas apretaste el boton.
            import queue

            enters: "queue.Queue[bool]" = queue.Queue()

            def _lector_stdin():
                while True:
                    try:
                        input()
                    except (EOFError, OSError):
                        return
                    enters.put(True)

            threading.Thread(target=_lector_stdin, daemon=True,
                             name="stdin").start()

            def _hubo_enter() -> bool:
                try:
                    enters.get_nowait()
                    return True
                except queue.Empty:
                    return False

            # Sesiones sucesivas: con el panel arriba, terminar una grabacion
            # no tiene por que matar el proceso. Se vuelve a IDLE y queda
            # listo para otra (grabar, sacar fotos, grabar de nuevo) sin
            # reiniciar nada. Se sale con Ctrl+C.
            while True:
                # --- esperar el arranque (consola o panel)
                print("\nENTER para arrancar, o usa el boton del panel web...")
                arranco_por_consola = False
                while sistema.estado == Estado.IDLE:
                    if _hubo_enter():
                        arranco_por_consola = True
                        break
                    time.sleep(0.2)

                if arranco_por_consola:
                    if not sistema.iniciar():
                        print("el arranque fallo; esperando otro intento")
                        continue
                else:
                    print("arrancado desde el panel web")

                # --- esperar a que termine el INIT antes de vigilar el final.
                #     Aca estaba el bug: si se sale de la espera en INIT, el
                #     loop de abajo ve un estado que todavia no es GRABANDO,
                #     da la corrida por terminada y llama a detener() en plena
                #     apertura del hardware.
                while sistema.estado == Estado.INIT:
                    time.sleep(0.2)

                if sistema.estado != Estado.GRABANDO:
                    print("el arranque no llego a GRABANDO; esperando otro intento")
                    continue

                # --- esperar la parada (consola o panel)
                print("ENTER para detener, o usa el boton del panel web...")
                paro_por_consola = False
                while sistema.estado == Estado.GRABANDO:
                    if _hubo_enter():
                        paro_por_consola = True
                        break
                    time.sleep(0.2)

                if paro_por_consola:
                    sistema.detener()

                print("sesion terminada. Ctrl+C para salir del programa.")
    except KeyboardInterrupt:
        print("\ninterrumpido")
    finally:
        sistema.detener()

    return 1 if sistema.alertas else 0


if __name__ == "__main__":
    raise SystemExit(main())
