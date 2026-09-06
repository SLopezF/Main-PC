"""
gopro_lib.py

GoPro por Wi-Fi desde la Raspberry Pi, con la API que consume
main_partido.py. Portado de gopro_controller.py, con tres cambios
deliberados:

  1. SIN `signal.signal(SIGUSR1, ...)` a nivel de modulo. Registrar un
     handler de senales al importar rompe cuando el modulo se importa desde
     otro proceso (y solo funciona en el hilo principal). Aca la parada se
     pide con parar() o tocando el archivo de flag.
  2. SIN el import de gopro_drive_stream. La subida a Drive es una
     preocupacion separada; si la necesitas, importa ese modulo desde donde
     lo uses, no desde el driver de la camara.
  3. SIN logging.basicConfig al importar, que pisa la config de logging de
     todo el programa.

OJO CON LA RED
Conectarse a la GoPro pone a la Pi EN LA RED DE LA GOPRO. Mientras dure,
la Pi pierde internet y, si estas por SSH sobre esa misma interfaz, perdes
la sesion. Con Wi-Fi a la GoPro y SSH por Ethernet no hay problema.

USO DIRECTO
    python3 gopro_lib.py --estado
    python3 gopro_lib.py --foto /tmp/prueba.jpg
    python3 gopro_lib.py --grabar 2          # graba 2 minutos
"""

import logging
import subprocess
import time
import urllib.request
from pathlib import Path

from goprocam import GoProCamera, constants

import config_hw as chw

log = logging.getLogger("gopro_lib")

FLAG_PARADA = Path("/tmp/gopro_stop_recording")
INTERVALO_KEEPALIVE = 3.0     # goprocam recomienda pingear cada ~3 s
INTERVALO_REINTENTO = 10.0


class GoPro:
    def __init__(self, ssid: str = chw.GOPRO_SSID,
                 password: str = chw.GOPRO_PASSWORD,
                 ip: str = chw.GOPRO_IP):
        self.ssid = ssid
        self.password = password
        self.ip = ip
        self.cam = None
        self.listo = False
        self._parar = False

    # ------------------------------------------------------------------- wifi
    def conectado_al_wifi(self) -> bool:
        """La conexion Wi-Fi ACTIVA de la Pi es la de la GoPro."""
        try:
            r = subprocess.run(
                ["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"],
                capture_output=True, text=True, timeout=10, check=True)
            for linea in r.stdout.strip().split("\n"):
                activa, ssid = (linea.split(":", 1) + [""])[:2]
                if activa == "yes" and ssid == self.ssid:
                    return True
        except (subprocess.SubprocessError, FileNotFoundError):
            log.exception("nmcli fallo")
        return False

    def ssid_visible(self) -> bool:
        try:
            subprocess.run(["nmcli", "dev", "wifi", "rescan"], timeout=15)
            r = subprocess.run(["nmcli", "-t", "-f", "ssid", "dev", "wifi"],
                               capture_output=True, text=True, timeout=15)
            return self.ssid in r.stdout.split("\n")
        except subprocess.SubprocessError:
            return False

    def conectar_wifi(self) -> bool:
        log.info("conectando a %s...", self.ssid)
        try:
            subprocess.run(
                ["nmcli", "dev", "wifi", "connect", self.ssid,
                 "password", self.password],
                capture_output=True, text=True, timeout=30, check=True)
            time.sleep(2)
            return self.conectado_al_wifi()
        except subprocess.CalledProcessError as e:
            log.warning("no pude conectar: %s", (e.stderr or "").strip() or e)
            return False

    def desconectar_wifi(self) -> None:
        """Soltar el AP de la GoPro para que la Pi vuelva a su red normal."""
        try:
            subprocess.run(["nmcli", "con", "down", self.ssid], timeout=15)
        except subprocess.SubprocessError:
            log.exception("no pude desconectar")

    def api_responde(self) -> bool:
        """
        Que el AP este arriba no significa que la camara conteste. Esto lo
        confirma pidiendo un status real.
        """
        try:
            cam = self.cam or GoProCamera.GoPro(ip_address=self.ip)
            return cam.getStatus(constants.Status.Status,
                                 constants.Status.STATUS.BattPercent) is not None
        except Exception:
            return False

    # ------------------------------------------------------------------- init
    def init(self, timeout: float = 60.0) -> bool:
        """
        Deja la camara lista: asocia el Wi-Fi si hace falta y confirma que la
        API responde. Devuelve True/False en vez de bloquear para siempre,
        porque main_partido.py sigue sin GoPro si esto falla (no tiene sentido
        no poder debuggear la deteccion porque la camara esta sin bateria).
        """
        limite = time.time() + timeout
        while time.time() < limite:
            if not self.conectado_al_wifi():
                if not self.ssid_visible():
                    log.info("%s no visible (camara apagada?)", self.ssid)
                    time.sleep(INTERVALO_REINTENTO)
                    continue
                if not self.conectar_wifi():
                    time.sleep(INTERVALO_REINTENTO)
                    continue
            try:
                self.cam = GoProCamera.GoPro(ip_address=self.ip)
                if self.api_responde():
                    self.listo = True
                    self.sincronizar_reloj()
                    return True
            except Exception:
                log.exception("no pude crear el objeto GoProCamera")
            time.sleep(INTERVALO_REINTENTO)

        log.error("timeout de %.0f s inicializando la GoPro", timeout)
        self.listo = False
        return False

    # ----------------------------------------------------------------- estado
    def estado(self) -> dict:
        info = {"bateria": None, "espacio": None}
        if self.cam is None:
            return info
        try:
            info["bateria"] = self.cam.getStatus(
                constants.Status.Status, constants.Status.STATUS.BattPercent)
        except Exception:
            log.exception("no pude leer bateria")
        try:
            info["espacio"] = self.cam.getStatus(
                constants.Status.Status, constants.Status.STATUS.RemainingSpace)
        except Exception:
            log.exception("no pude leer espacio")
        return info

    def sincronizar_reloj(self) -> None:
        """Alinea el reloj de la camara con el de la Pi: sin esto, correlacionar
        los archivos de la GoPro con el CSV del tracker es adivinanza."""
        try:
            self.cam.syncTime()
        except Exception:
            log.exception("no pude sincronizar el reloj")

    # ------------------------------------------------------------------ fotos
    def foto(self, destino: str | None = None) -> str | None:
        """
        Una foto. Si `destino` es una ruta, la descarga a la Pi y devuelve esa
        ruta; si es None, devuelve la ruta en la SD de la camara
        ('100GOPRO/GOPR0001.JPG').
        """
        if self.cam is None:
            return None
        try:
            self.cam.mode(constants.Mode.PhotoMode)
            time.sleep(0.4)                 # el cambio de modo no es instantaneo
            self.cam.take_photo(0)
            time.sleep(1.0)                 # la camara tarda en indexar
            remoto = self.ultimo_archivo()
            if remoto is None:
                return None
            if destino is None:
                return remoto
            return self.descargar(remoto, destino)
        except Exception:
            log.exception("fallo la foto")
            return None

    # --------------------------------------------------------------- grabacion
    def iniciar_grabacion(self) -> bool:
        if self.cam is None:
            return False
        try:
            self._parar = False
            if FLAG_PARADA.exists():
                FLAG_PARADA.unlink()
            self.cam.mode(constants.Mode.SubMode.Video.Video)
            time.sleep(0.4)
            self.cam.shutter(constants.start)
            return True
        except Exception:
            log.exception("no pude arrancar la grabacion")
            return False

    def detener_grabacion(self) -> str | None:
        """Devuelve el archivo grabado, tipo '100GOPRO/GH010001.MP4'."""
        if self.cam is None:
            return None
        try:
            self.cam.shutter(constants.stop)
        except Exception:
            # La camara puede haberse detenido sola (bateria, SD llena).
            log.exception("fallo el stop; la camara pudo pararse sola")
        time.sleep(1.5)
        return self.ultimo_archivo()

    def parar(self) -> None:
        """Pide el fin de grabar_por() desde otro hilo."""
        self._parar = True

    def parada_pedida(self) -> bool:
        return self._parar or FLAG_PARADA.exists()

    def grabar_por(self, minutos: int = 50, keepalive: bool = True) -> str | None:
        """
        Graba con limite de tiempo, sosteniendo la conexion. Termina por
        tiempo, por parar()/flag, o si la camara se cae.

        Si se pierde la conexion, la GoPro SIGUE grabando en su SD: lo que se
        pierde es el control remoto, no el material. Por eso se reintenta
        conectar en vez de dar la grabacion por perdida.
        """
        if not self.iniciar_grabacion():
            return None
        t0 = time.time()
        ultimo = time.time()
        limite = minutos * 60
        try:
            while True:
                if self.parada_pedida():
                    break
                if time.time() - t0 >= limite:
                    break
                if keepalive and time.time() - ultimo >= INTERVALO_KEEPALIVE:
                    if not self.api_responde():
                        log.warning("se cayo la conexion; sigue grabando en la SD")
                        self.init(timeout=60.0)
                    ultimo = time.time()
                time.sleep(1.0)
        finally:
            archivo = self.detener_grabacion()
        return archivo

    # --------------------------------------------------------------- archivos
    def ultimo_archivo(self, reintentos: int = 5, espera: float = 2.0) -> str | None:
        """
        Ultimo archivo de la SD en forma 'CARPETA/ARCHIVO.EXT'. Se reintenta
        porque la camara tarda un momento en indexar despues del disparo.
        """
        for intento in range(1, reintentos + 1):
            try:
                url = self.cam.getMedia()
                if url:
                    # url: http://10.5.5.9/videos/DCIM/100GOPRO/GH010001.MP4
                    parte = url.split("/DCIM/", 1)[-1]
                    if parte and parte != url:
                        return parte
            except Exception:
                log.exception("no pude leer el ultimo archivo (%d/%d)",
                              intento, reintentos)
            if intento < reintentos:
                time.sleep(espera)
        return None

    def descargar(self, archivo_camara: str, destino: str) -> str | None:
        """Baja 'CARPETA/ARCHIVO.EXT' de la SD a una ruta local."""
        url = f"http://{self.ip}/videos/DCIM/{archivo_camara}"
        try:
            Path(destino).parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, destino)
            return destino
        except Exception:
            log.exception("fallo la descarga de %s", url)
            return None

    def borrar_de_camara(self, archivo_camara: str) -> bool:
        """
        Borra de la SD. Llamalo SOLO despues de confirmar que la copia quedo
        bien en otro lado: aca no hay papelera.
        """
        try:
            carpeta, nombre = archivo_camara.split("/")
            self.cam.deleteFile(carpeta, nombre)
            return True
        except Exception:
            log.exception("no pude borrar %s", archivo_camara)
            return False

    def listar_media(self):
        try:
            return self.cam.listMedia(format=True)
        except Exception:
            log.exception("no pude listar la media")
            return []

    def apagar(self) -> None:
        try:
            self.cam.power_off()
        except Exception:
            log.exception("no pude apagar la camara")

    def close(self) -> None:
        self.listo = False
        self.cam = None


# --------------------------------------------------------------------------- #
# CLI de prueba
# --------------------------------------------------------------------------- #

def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--estado", action="store_true")
    ap.add_argument("--foto", type=str, nargs="?", const="/tmp/gopro_test.jpg")
    ap.add_argument("--grabar", type=float, default=None, help="minutos")
    ap.add_argument("--listar", action="store_true")
    ap.add_argument("--desconectar", action="store_true")
    args = ap.parse_args()

    gp = GoPro()
    if not gp.init(timeout=60.0):
        print("no pude inicializar la GoPro")
        return

    if args.estado or not any((args.foto, args.grabar, args.listar)):
        print(gp.estado())
    if args.foto:
        print("foto ->", gp.foto(args.foto))
    if args.grabar:
        print("grabado ->", gp.grabar_por(minutos=args.grabar))
    if args.listar:
        for m in gp.listar_media():
            print(m)
    if args.desconectar:
        gp.desconectar_wifi()


if __name__ == "__main__":
    main()
