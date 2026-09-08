#!/usr/bin/env python3
"""
gopro_controller.py  (corre en la Raspberry Pi)

Responsabilidad única: hablar con la GoPro. No sabe nada de páginas web,
de SSH, ni de a dónde van a parar los archivos después. Eso lo resuelven
web_control.py (panel) y gopro_file_transfer.py (que corre en la compu).

Qué hace:
- Conectarse / mantenerse conectado a la Wi-Fi de la GoPro.
- Vigilar la conexión de forma CONTINUA (no solo durante una grabación),
  para poder responder en todo momento "¿estoy conectado a la cámara?".
- Iniciar/detener grabaciones (por tiempo límite, por señal o por flag).
- Sacar una foto.
- Devolver el path del último archivo grabado/capturado en la SD, en
  formato "FOLDER/FILENAME.EXT" y la URL directa para descargarlo (la
  descarga en sí la hace gopro_file_transfer.py, desde la computadora).
- Borrar un archivo de la SD (se llama solo cuando la transferencia ya
  fue confirmada del otro lado).

Requires:
    pip install goprocam
"""

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from goprocam import GoProCamera, constants

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #

GOPRO_SSID = "HERO7 Silver"
GOPRO_SSID_PASSWORD = "ZBn-6K2-BY9"
GOPRO_IP = "10.5.5.9"

RECORDING_MINUTES = 50
STOP_FLAG_FILE = Path("/tmp/gopro_stop_recording")
# Se puede pisar con la variable de entorno GOPRO_LOG_FILE (usado por
# test_pipeline.py para no tocar la ruta fija de la Raspi).
LOG_FILE = Path(os.environ.get("GOPRO_LOG_FILE", "/home/santipi/Desktop/GoPro-Tests/gopro_controller.log"))
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

CONNECTION_CHECK_INTERVAL = 5      # seg. entre chequeos del monitor continuo
CONNECTION_RETRY_INTERVAL = 10     # seg. entre reintentos de reconexión
KEEPALIVE_INTERVAL = 3             # goprocam recomienda ping cada ~3s en ops largas

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
# El archivo de log siempre guarda todo en detalle (INFO). La terminal, en
# cambio, por default solo muestra WARNING/ERROR para no inundar la
# pantalla con el detalle de cada chequeo de conexión. Se puede pisar con
# la variable de entorno GOPRO_TERMINAL_LOG_LEVEL (ej. "INFO" para volver
# al modo verboso mientras se debuguea).

TERMINAL_LOG_LEVEL = getattr(
    logging, os.environ.get("GOPRO_TERMINAL_LOG_LEVEL", "WARNING").upper(), logging.WARNING
)

_file_handler = logging.FileHandler(LOG_FILE)
_file_handler.setLevel(logging.INFO)

_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setLevel(TERMINAL_LOG_LEVEL)

logging.basicConfig(
    level=logging.INFO,  # nivel mínimo global; cada handler filtra por su cuenta
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[_file_handler, _stream_handler],
)
log = logging.getLogger("gopro_controller")


# --------------------------------------------------------------------------- #
# Fábrica de cámara — indirección deliberada para poder testear sin
# hardware real (ver test_pipeline.py, que reemplaza esta función por una
# que devuelve una GoPro simulada).
# --------------------------------------------------------------------------- #

def create_camera(ip_address: str = GOPRO_IP) -> GoProCamera.GoPro:
    return GoProCamera.GoPro(ip_address=ip_address)


# --------------------------------------------------------------------------- #
# Gestión de conexión Wi-Fi
# --------------------------------------------------------------------------- #

def is_connected_to_gopro_wifi() -> bool:
    """Chequea si la interfaz Wi-Fi activa de la Raspi es la de la GoPro."""
    try:
        result = subprocess.run(
            ["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        for line in result.stdout.strip().split("\n"):
            active, ssid = (line.split(":", 1) + [""])[:2]
            if active == "yes" and ssid == GOPRO_SSID:
                return True
        return False
    except (subprocess.SubprocessError, FileNotFoundError):
        log.exception("Chequeo nmcli falló")
        return False


def gopro_ssid_visible() -> bool:
    """Escanea si el SSID de la GoPro está visible, sin conectarse."""
    try:
        subprocess.run(["nmcli", "dev", "wifi", "rescan"], timeout=15)
        result = subprocess.run(
            ["nmcli", "-t", "-f", "ssid", "dev", "wifi"],
            capture_output=True, text=True, timeout=15,
        )
        return GOPRO_SSID in result.stdout.split("\n")
    except subprocess.SubprocessError:
        return False


def connect_to_gopro_wifi() -> bool:
    """Intenta asociarse a la red Wi-Fi de la GoPro."""
    log.info("Intentando conectar a la Wi-Fi de la GoPro (%s)...", GOPRO_SSID)
    try:
        subprocess.run(
            ["nmcli", "dev", "wifi", "connect", GOPRO_SSID, "password", GOPRO_SSID_PASSWORD],
            capture_output=True, text=True, timeout=30, check=True,
        )
        time.sleep(2)
        return is_connected_to_gopro_wifi()
    except subprocess.CalledProcessError as e:
        log.warning("Falló la conexión Wi-Fi: %s", e.stderr.strip() if e.stderr else e)
        return False


def gopro_api_reachable(gopro: Optional[GoProCamera.GoPro] = None) -> bool:
    """
    Confirma que la cámara realmente responde a su API HTTP, no solo que
    el WiFi está asociado. Esta es la fuente de verdad real de "estamos
    conectados a la GoPro".
    """
    try:
        cam = gopro or create_camera()
        return cam.getStatus(constants.Status.Status, constants.Status.STATUS.BattPercent) is not None
    except Exception:
        return False


def wait_for_gopro() -> GoProCamera.GoPro:
    """Bloquea hasta que la GoPro esté conectada y respondiendo."""
    while True:
        if not is_connected_to_gopro_wifi():
            if gopro_ssid_visible():
                if connect_to_gopro_wifi():
                    log.info("Conectado a la Wi-Fi de la GoPro.")
                else:
                    log.info("SSID de la GoPro visible pero la conexión falló, reintentando...")
                    time.sleep(CONNECTION_RETRY_INTERVAL)
                    continue
            else:
                log.info("GoPro no visible (probablemente apagada). Esperando...")
                time.sleep(CONNECTION_RETRY_INTERVAL)
                continue

        try:
            gopro = create_camera()
            if gopro_api_reachable(gopro):
                log.info("La API de la GoPro está respondiendo.")
                return gopro
        except Exception:
            log.exception("Falló la inicialización del objeto GoProCamera")

        log.info("Conectado al SSID pero la API todavía no responde, reintentando...")
        time.sleep(CONNECTION_RETRY_INTERVAL)


class ConnectionMonitor:
    """
    Vigilancia CONTINUA de la conexión con la GoPro, independiente de si
    hay una grabación en curso o no. Corre en un thread propio y avisa
    cada cambio de estado a través de `on_change`, para que web_control.py
    pueda reflejarlo en tiempo real en el panel.

    Esta es la respuesta directa a "asegurar que la Raspi está conectada
    a la GoPro": antes esta lógica (monitor_connection_loss) solo se usaba
    durante record_video y quedaba huérfana el resto del tiempo.
    """

    def __init__(self, on_change: Callable[[bool], None], interval: float = CONNECTION_CHECK_INTERVAL):
        self._on_change = on_change
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._connected: Optional[bool] = None

    def _loop(self):
        while not self._stop_event.is_set():
            connected = is_connected_to_gopro_wifi() and gopro_api_reachable()
            if connected != self._connected:
                self._connected = connected
                log.info("Estado de conexión con la GoPro: %s", "OK" if connected else "PERDIDA")
                try:
                    self._on_change(connected)
                except Exception:
                    log.exception("Error en callback on_change del monitor de conexión")
            self._stop_event.wait(self._interval)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info("Monitor de conexión iniciado (cada %ds).", self._interval)

    def stop(self):
        self._stop_event.set()

    @property
    def connected(self) -> Optional[bool]:
        """Último estado conocido (None si el monitor todavía no corrió)."""
        return self._connected


# --------------------------------------------------------------------------- #
# Control de grabación / disparo
# --------------------------------------------------------------------------- #

_stop_requested = False


def _handle_stop_signal(signum, frame):
    global _stop_requested
    log.info("Señal de stop recibida (%s).", signum)
    _stop_requested = True


signal.signal(signal.SIGUSR1, _handle_stop_signal)


def stop_requested() -> bool:
    return _stop_requested or STOP_FLAG_FILE.exists()


def clear_stop_flag():
    global _stop_requested
    _stop_requested = False
    if STOP_FLAG_FILE.exists():
        STOP_FLAG_FILE.unlink()


def start_recording(gopro: GoProCamera.GoPro):
    """Pone la cámara en modo video y arranca a grabar. No bloquea."""
    clear_stop_flag()
    log.info("Iniciando grabación...")
    gopro.mode(constants.Mode.SubMode.Video.Video)
    gopro.shutter(constants.start)


def stop_recording(gopro: GoProCamera.GoPro) -> Optional[str]:
    """Detiene la grabación y devuelve el path del archivo grabado."""
    try:
        gopro.shutter(constants.stop)
        log.info("Grabación detenida.")
    except Exception:
        log.exception("Error enviando el comando de stop (puede que la cámara ya se haya detenido sola).")
    return get_latest_media_filename(gopro)


def record_video_blocking(gopro: GoProCamera.GoPro, max_minutes: int = RECORDING_MINUTES) -> Optional[str]:
    """
    Variante bloqueante de todo el ciclo (start -> esperar límite/stop/señal
    -> stop), útil para uso por script/CLI. web_control.py usa
    start_recording()/stop_recording() por separado porque necesita
    controlarlo de forma asincrónica desde los endpoints HTTP.
    """
    start_recording(gopro)
    start_time = time.time()
    max_seconds = max_minutes * 60
    last_keepalive = time.time()

    try:
        while True:
            if stop_requested():
                log.info("Stop pedido por usuario/señal.")
                break
            if time.time() - start_time >= max_seconds:
                log.info("Se alcanzó el límite de %d minutos.", max_minutes)
                break
            if time.time() - last_keepalive >= KEEPALIVE_INTERVAL:
                if not gopro_api_reachable(gopro):
                    log.error("Se perdió la conexión con la GoPro durante la grabación.")
                    try:
                        gopro = wait_for_gopro()
                    except KeyboardInterrupt:
                        raise
                last_keepalive = time.time()
            time.sleep(1)
    finally:
        pass

    return stop_recording(gopro)


def take_photo(gopro: GoProCamera.GoPro) -> Optional[str]:
    """
    Pone la cámara en modo foto y dispara una captura. Usa el helper
    `take_photo()` que ya provee goprocam. Devuelve el path de la foto
    tomada en la SD.
    """
    log.info("Sacando foto...")
    try:
        gopro.mode(constants.Mode.SubMode.Photo.Single)
        gopro.take_photo()
        # La cámara tarda un instante en indexar el archivo.
        time.sleep(1.5)
        return get_latest_media_filename(gopro)
    except Exception:
        log.exception("Error al sacar la foto.")
        return None


def get_latest_media_filename(gopro: GoProCamera.GoPro, retries: int = 5, delay_seconds: float = 2.0) -> Optional[str]:
    """
    Devuelve el path del archivo más reciente en la SD, en la forma corta
    "FOLDER/FILENAME.EXT" (sirve tanto para el último video como para la
    última foto: getMedia() de goprocam devuelve el último archivo).
    """
    for attempt in range(1, retries + 1):
        try:
            media_url = gopro.getMedia()
            if media_url:
                path_part = media_url.split("/DCIM/", 1)[-1]
                if path_part and path_part != media_url:
                    return path_part
        except Exception:
            log.exception("No se pudo determinar el último archivo (intento %d/%d).", attempt, retries)

        if attempt < retries:
            log.info("Archivo todavía no indexado, reintentando en %.0fs...", delay_seconds)
            time.sleep(delay_seconds)

    return None


def get_gopro_download_url(camera_file_path: str, gopro_ip: str = GOPRO_IP) -> str:
    """URL HTTP directa a un archivo de la SD, para que la computadora lo descargue."""
    folder, filename = camera_file_path.split("/")
    return f"http://{gopro_ip}:8080/videos/DCIM/{folder}/{filename}"


# --------------------------------------------------------------------------- #
# Borrado — se llama solo cuando la transferencia a la compu ya fue
# confirmada (ver /api/delete_file en web_control.py).
# --------------------------------------------------------------------------- #

def delete_file_from_camera(gopro: GoProCamera.GoPro, camera_file_path: str) -> bool:
    log.info("Borrando %s de la SD...", camera_file_path)
    try:
        folder, filename = camera_file_path.split("/")
        gopro.deleteFile(folder, filename)
        log.info("Borrado de la cámara.")
        return True
    except Exception:
        log.exception("El borrado falló.")
        return False


# --------------------------------------------------------------------------- #
# Extras para un rig desatendido
# --------------------------------------------------------------------------- #

def get_battery_level(gopro: GoProCamera.GoPro) -> Optional[int]:
    try:
        return gopro.getStatus(constants.Status.Status, constants.Status.STATUS.BattPercent)
    except Exception:
        log.exception("No se pudo leer el nivel de batería.")
        return None


def get_remaining_sd_capacity(gopro: GoProCamera.GoPro):
    try:
        return gopro.getStatus(constants.Status.Status, constants.Status.STATUS.RemainingSpace)
    except Exception:
        log.exception("No se pudo leer el espacio libre de la SD.")
        return None


def sync_camera_clock(gopro: GoProCamera.GoPro):
    try:
        gopro.syncTime()
        log.info("Reloj de la cámara sincronizado.")
    except Exception:
        log.exception("Falló la sincronización de reloj.")


def set_camera_to_sleep(gopro: GoProCamera.GoPro):
    try:
        gopro.power_off()
    except Exception:
        log.exception("No se pudo apagar la cámara.")


def list_all_media(gopro: GoProCamera.GoPro):
    try:
        return gopro.listMedia(format=True)
    except Exception:
        log.exception("No se pudo listar la media.")
        return []


def disconnect_from_gopro_wifi():
    try:
        subprocess.run(["nmcli", "con", "down", GOPRO_SSID], timeout=15)
        log.info("Desconectado de la Wi-Fi de la GoPro.")
    except subprocess.SubprocessError:
        log.exception("Falló la desconexión de la Wi-Fi de la GoPro.")


if __name__ == "__main__":
    # Uso standalone simple: confirma conexión y queda vigilando.
    log.info("=== Chequeo de conexión standalone ===")
    gopro = wait_for_gopro()
    sync_camera_clock(gopro)
    log.info("Batería: %s%% | SD libre: %s", get_battery_level(gopro), get_remaining_sd_capacity(gopro))

    monitor = ConnectionMonitor(on_change=lambda ok: log.info("Cambio de estado -> %s", ok))
    monitor.start()
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        monitor.stop()
        log.info("Interrumpido por el usuario.")
