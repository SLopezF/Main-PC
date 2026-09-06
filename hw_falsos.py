"""
hw_falsos.py

Implementaciones FALSAS de las tres librerias de hardware. Sirven para dos
cosas:

  1. Documentar exactamente que metodos tiene que exponer cada libreria
     real. Escribí motor_lib.py, encoder_lib.py y gopro_lib.py con estas
     firmas (basandote en rpi5_motor_control_usb.py, as5047_config.py +
     test3.py, y gopro_controller.py) y el main no se toca.
  2. Poder correr main_partido.py entero sin fierros: si el import de la
     libreria real falla, se usa esta y el sistema arranca igual.

El motor falso mueve al instante y el encoder falso reporta exactamente lo
que el motor dice, o sea que NUNCA vas a ver un error de homing. Eso es
justamente lo que hace inutil al simulador para validar la mecanica: sirve
para probar el flujo del programa, nada mas.
"""

import time

import config_hw as chw


# =============================================================================
# ENCODER  ->  encoder_lib.Encoder
# =============================================================================

class Encoder:
    """
    API que tiene que cumplir encoder_lib.Encoder.

    Base real: as5047_config.py (configuracion de registros y diagnostico del
    iman) + test3.py (lectura rapida de ANGLECOM).
    """

    def __init__(self, bus: int = chw.ENC_BUS, device: int = chw.ENC_DEVICE):
        self.bus, self.device = bus, device
        self._angulo = 79.0  # el falso arranca justo en el home

    def aplicar_config(self) -> dict:
        """
        Escribe SETTINGS1/SETTINGS2 (ABI binario, DAEC off, histeresis 1 LSB).
        Es soft write: se pierde al cortar la alimentacion, hay que llamarlo
        en cada arranque. Devuelve la config leida de vuelta del chip.
        """
        return self.leer_config()

    def leer_config(self) -> dict:
        """
        Config REAL leida del chip. Tiene que traer al menos:
            abi_ppr, abi_steps, hysteresis_lsb, daec_enable, zero_position
        """
        return {
            "abi_ppr": chw.ENC_PPR_ESPERADO,
            "abi_steps": chw.ENC_PASOS_VUELTA_ESPERADO,
            "hysteresis_lsb": 1,
            "daec_enable": False,
            "zero_position": 0,
        }

    def leer_angulo(self) -> tuple[int, float, int]:
        """(raw 0..16383, grados 0..360, flag_error). Con DAEC off, ANGLEUNC."""
        return int(self._angulo / 360.0 * 16384) & 0x3FFF, self._angulo % 360.0, 0

    def diagnostico(self) -> dict:
        """Estado del iman: agc, mag_too_low, mag_too_high, magnitude."""
        return {"agc": 128, "mag_too_low": False, "mag_too_high": False,
                "magnitude": 2000}

    @staticmethod
    def estado_iman(diag: dict) -> str:
        if diag.get("mag_too_low"):
            return "IMAN MUY DEBIL/LEJOS (MAGL)"
        if diag.get("mag_too_high"):
            return "IMAN MUY FUERTE/CERCA (MAGH)"
        return "iman OK"

    # Solo del falso: para que el encoder simulado siga al motor simulado.
    def _simular(self, angulo: float) -> None:
        self._angulo = angulo % 360.0

    def close(self) -> None:
        pass


# =============================================================================
# MOTOR  ->  motor_lib.Motor
# =============================================================================

class Motor:
    """
    API que tiene que cumplir motor_lib.Motor.

    Base real: rpi5_motor_control_usb.py. Los metodos mapean uno a uno con
    los comandos de la ESP32 ('current', 'enable', 'deg', 'degto', 'zero',
    'moving'), leyendo hasta el marcador [eot].
    """

    def __init__(self, puerto=None, baud: int = chw.MOTOR_BAUD):
        self.puerto = puerto or "(simulado)"
        self._grados = 0.0
        self._habilitado = False
        self._encoder = None  # el falso lo usa para arrastrar al encoder

    # -- conexion
    def conectado(self) -> bool:
        """True si el puerto respondio al handshake ('status' con [eot])."""
        return True

    def status(self) -> list[str]:
        return [f"puerto={self.puerto}", f"pos_deg={self._grados:.2f}",
                f"enabled={int(self._habilitado)}"]

    # -- configuracion
    def set_corriente(self, mA: int) -> None:
        print(f"[motor-sim] corriente {mA} mA")

    def set_micropasos(self, n: int) -> None:
        pass

    def set_velocidad(self, pasos_por_s: float) -> None:
        pass

    def set_aceleracion(self, pasos_por_s2: float) -> None:
        pass

    def habilitar(self) -> None:
        self._habilitado = True

    def deshabilitar(self) -> None:
        self._habilitado = False

    # -- movimiento (todo en GRADOS DE MOTOR, no de eje)
    def mover_grados(self, grados: float) -> None:
        """Movimiento RELATIVO."""
        self._grados += float(grados)
        if self._encoder is not None:
            self._encoder._simular(chw.ENCODER_GRADOS_EN_MOTOR_CERO
                                   + self._grados / chw.RELACION_TRANSMISION)

    def ir_a_grados(self, grados: float) -> None:
        """Movimiento ABSOLUTO respecto del cero del motor."""
        self.mover_grados(float(grados) - self._grados)

    def posicion_grados(self) -> float:
        return self._grados

    def esperar_fin(self, timeout: float = chw.MOTOR_TIMEOUT_MOV_S) -> bool:
        """True si termino, False si se agoto el timeout."""
        time.sleep(0.05)
        return True

    def zero(self) -> None:
        """Declara la posicion actual como el cero del motor."""
        self._grados = 0.0

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


# =============================================================================
# GOPRO  ->  gopro_lib.GoPro
# =============================================================================

class GoPro:
    """
    API que tiene que cumplir gopro_lib.GoPro.

    Base real: gopro_controller.py. Al portarlo, sacale el
    `signal.signal(SIGUSR1, ...)` a nivel de modulo y el import de
    gopro_drive_stream: los dos rompen si se importa desde otro proceso.
    """

    def __init__(self, ssid=chw.GOPRO_SSID, password=chw.GOPRO_PASSWORD,
                 ip=chw.GOPRO_IP):
        self.ssid, self.ip = ssid, ip
        self.listo = False
        self._n = 0

    def init(self, timeout: float = 60.0) -> bool:
        """
        Conecta el Wi-Fi de la Pi al AP de la GoPro y confirma que la API
        responde. Devuelve True si quedo lista. Es la llamada que despues vas
        a completar; el main solo mira el bool.
        """
        self.listo = True
        print(f"[gopro-sim] init() -> conectada a {self.ssid} ({self.ip})")
        return True

    def estado(self) -> dict:
        """Al menos: bateria (0..100) y espacio libre en la SD."""
        return {"bateria": 100, "espacio": "simulado"}

    def foto(self, destino: str | None = None) -> str | None:
        """
        Saca UNA foto y devuelve la ruta (en la Pi si se descargo, o el
        path en la SD de la camara). None si fallo.
        """
        self._n += 1
        ruta = destino or f"/dev/shm/gopro_sim_{self._n:04d}.jpg"
        print(f"[gopro-sim] foto -> {ruta}")
        return ruta

    def iniciar_grabacion(self) -> bool:
        return True

    def detener_grabacion(self) -> str | None:
        """Devuelve el nombre del archivo grabado, tipo '100GOPRO/GH010001.MP4'."""
        return None

    def close(self) -> None:
        self.listo = False


# =============================================================================
# Resolucion de librerias: real si existe, falsa si no
# =============================================================================

def cargar(nombre_modulo: str, clase: str, fallback):
    """
    Importa `clase` de `nombre_modulo`; si el modulo no existe todavia (o no
    esta el SDK), devuelve la version falsa y avisa UNA vez. Asi el main
    arranca igual mientras escribis las librerias reales.
    """
    try:
        mod = __import__(nombre_modulo)
        obj = getattr(mod, clase)
        print(f"[hw] {nombre_modulo}.{clase}: real")
        return obj, True
    except Exception as exc:
        print(f"[hw] {nombre_modulo}.{clase}: SIMULADO ({type(exc).__name__}: {exc})")
        return fallback, False
