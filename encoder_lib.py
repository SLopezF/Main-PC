"""
encoder_lib.py

AS5047D por SPI. Fusion de as5047_config.py (configuracion de registros y
diagnostico del iman) con test3.py (lectura rapida aprovechando el pipeline
del chip), con la API que espera main_partido.py.

CONFIGURACION QUE SE APLICA
    ABI activo en modo BINARIO: 512 ppr = 2048 pasos/vuelta (el maximo del D;
                              el AS5047P llega a 1024/4096, este no)
    Histeresis 1 LSB          -> no hay chattering con el eje casi parado
    DAEC DESACTIVADO          -> a giro lento da menos ruido (-0.016 deg rms)
    PWM desactivado           -> el pin W queda libre
    comp_l/h_error_en = 0     -> el ABI NO se congela si el iman flaquea;
                                 el estado del iman se vigila por SPI

NADA SE GRABA EN OTP. Las escrituras van a la copia volatil ("soft write"):
surten efecto al instante y se pierden al cortar la alimentacion. Por eso
main_partido.py llama a aplicar_config() en CADA arranque. Si un dia el
sistema arranca con angulos raros, lo primero a mirar es si esa llamada
corrio.

DOS FORMAS DE LEER EL ANGULO
    leer_angulo()         2 transferencias SPI: manda el comando y despues un
                          NOP para traer el dato. Correcta siempre.
    leer_angulo_rapido()  1 transferencia: se manda el comando de ANGLE y en
                          la MISMA trama vuelve el dato del pedido ANTERIOR.
                          El dato esta atrasado un ciclo, que a 100 kHz son
                          ~160 us. Para un loop de control esta bien; para
                          verificar un homing usa la otra.

USO DIRECTO (sin el resto del sistema)
    python3 encoder_lib.py            # aplica config y lee en continuo
    python3 encoder_lib.py --show     # solo ver la config actual
    python3 encoder_lib.py --diag     # solo estado del iman
    python3 encoder_lib.py --zero-aca # poner el cero en la posicion actual
"""

import time

import spidev

import config_hw as chw

# --------------------------------------------------------------------------- #
# Registros (Figura 18 y Figura 25 del datasheet ams v2-00)
# --------------------------------------------------------------------------- #

REG_NOP = 0x0000
REG_ERRFL = 0x0001
REG_PROG = 0x0003
REG_ZPOSM = 0x0016
REG_ZPOSL = 0x0017
REG_SETTINGS1 = 0x0018
REG_SETTINGS2 = 0x0019
REG_DIAAGC = 0x3FFC
REG_MAG = 0x3FFD
REG_ANGLEUNC = 0x3FFE
REG_ANGLECOM = 0x3FFF

# Figura 30: {ppr: (codigo_ABIRES, pasos_por_vuelta)}
#
# ESTE ES UN AS5047D. En binario el tope es 512 ppr = 2048 pasos/vuelta; la
# entrada de 1024 que trae el AS5047P NO existe aca y por eso no esta en la
# tabla: pedirla levanta ValueError antes de escribir ningun registro.
#
# Los codigos ABIRES se conservan del mapeo del P (512 -> 0b001). Verificalo
# una vez con `python3 encoder_lib.py --apply --show`: si el chip devuelve
# otro ppr del que pediste, el codigo de esta tabla no es el del D y hay que
# corregirlo contra la tabla de resolucion ABI del datasheet.
ABI_DECIMAL = {
    1000: (0b000, 4000), 500: (0b001, 2000), 400: (0b010, 1600),
    300: (0b011, 1200), 200: (0b100, 800), 100: (0b101, 400),
    50: (0b110, 200), 25: (0b111, 100),
}
ABI_BINARY = {512: (0b001, 2048), 256: (0b010, 1024)}

# Figura 34
HYS_TABLE = {3: 0b00, 2: 0b01, 1: 0b10, 0: 0b11}

SPI_SPEED_HZ = 100_000  # el chip admite hasta 10 MHz; arrancar conservador

# Umbrales de aviso del AGC (0..255). 0x00 = campo demasiado fuerte,
# 0xFF = demasiado debil. Lo ideal es la zona central.
AGC_AVISO_CAMPO_BAJO = 220
AGC_AVISO_CAMPO_ALTO = 35

# Config que se escribe. Cambiala solo si sabes por que.
CONFIG = {
    "direction_reverse": True,
    "interface": "ABI",
    "abi_binary": True,
    "abi_ppr": 512,   # maximo del AS5047D en binario
    "hysteresis_lsb": 1,
    "uvw_pole_pairs": 1,
    "pwm_enable": False,
    "daec_enable": False,
    "angle_reg_cordic": True,
    "error_on_mag_low": False,
    "error_on_mag_high": False,
    "zero_position": None,  # None = no tocar el cero que ya tenga
}


class Encoder:
    """AS5047P. Cumple la API que consume main_partido.py."""

    def __init__(self, bus: int = chw.ENC_BUS, device: int = chw.ENC_DEVICE,
                 speed_hz: int = SPI_SPEED_HZ, config: dict | None = None):
        self.bus, self.device = int(bus), int(device)
        self.config = dict(config or CONFIG)
        self.spi = spidev.SpiDev()
        self.spi.open(self.bus, self.device)
        self.spi.max_speed_hz = int(speed_hz)
        self.spi.mode = 1  # CPOL=0, CPHA=1 (el datasheet dice "SPI mode=1")
        self.ultimo_error = 0
        self._xfer(self._cmd(REG_NOP, read=True))  # ceba el pipeline

    # ------------------------------------------------------------ capa fisica
    @staticmethod
    def _parity(valor: int) -> int:
        """Paridad PAR sobre los bits 0..14, resultado en el bit 15."""
        return valor | 0x8000 if bin(valor & 0x7FFF).count("1") % 2 else valor & 0x7FFF

    def _cmd(self, addr: int, read: bool = True) -> int:
        # bit14 = R/W (1 = lectura)
        return self._parity((addr & 0x3FFF) | (0x4000 if read else 0x0000))

    def _xfer(self, trama16: int) -> int:
        rx = self.spi.xfer2([(trama16 >> 8) & 0xFF, trama16 & 0xFF])
        time.sleep(0.000002)  # holgura sobre tCSn (minimo 350 ns en alto)
        return (rx[0] << 8) | rx[1]

    # -------------------------------------------------------------- registros
    def read_register(self, addr: int) -> int:
        """El dato pedido llega en la trama SIGUIENTE (Figura 15)."""
        self._xfer(self._cmd(addr, read=True))
        resp = self._xfer(self._cmd(REG_NOP, read=True))
        self.ultimo_error = (resp >> 14) & 1
        return resp & 0x3FFF

    def write_register(self, addr: int, value: int, verify: bool = True,
                       mask: int = 0x3FFF):
        self._xfer(self._cmd(addr, read=False))    # trama de comando
        self._xfer(self._parity(value & 0x3FFF))   # trama de dato
        time.sleep(0.001)
        if not verify:
            return None
        vuelta = self.read_register(addr)
        if (vuelta & mask) != (value & 0x3FFF & mask):
            raise IOError(
                f"Verificacion fallida en 0x{addr:04X}: escrito "
                f"0x{value & 0x3FFF:04X}, leido 0x{vuelta:04X}. "
                f"Revisa el cableado SPI y que el CS sea el device correcto."
            )
        return vuelta

    def limpiar_errores(self) -> dict:
        """Leer ERRFL limpia sus contenidos automaticamente."""
        err = self.read_register(REG_ERRFL)
        return {
            "framing": bool(err & 0x0001),
            "comando_invalido": bool(err & 0x0002),
            "paridad": bool(err & 0x0004),
            "raw": err,
        }

    # ------------------------------------------------------------------ angulo
    def leer_angulo(self, compensado: bool | None = None) -> tuple[int, float, int]:
        """
        (raw 0..16383, grados 0..360, flag_error).

        Con DAEC apagado se lee ANGLEUNC (0x3FFE), que es el CORDIC crudo.
        Es la lectura que hay que usar para verificar un homing.
        """
        if compensado is None:
            compensado = self.config["daec_enable"]
        raw = self.read_register(REG_ANGLECOM if compensado else REG_ANGLEUNC)
        return raw, (raw / 16384.0) * 360.0, self.ultimo_error

    def leer_angulo_rapido(self) -> tuple[int, float]:
        """
        Una sola transferencia por muestra: el dato que vuelve corresponde al
        pedido ANTERIOR, o sea que esta atrasado un ciclo de SPI. A 100 kHz
        son ~160 us de retardo. Para un loop de control da igual; para
        verificar una posicion, no.
        """
        addr = REG_ANGLECOM if self.config["daec_enable"] else REG_ANGLEUNC
        resp = self._xfer(self._cmd(addr, read=True))
        self.ultimo_error = (resp >> 14) & 1
        raw = resp & 0x3FFF
        return raw, (raw / 16384.0) * 360.0

    # ------------------------------------------------------- diagnostico iman
    def diagnostico(self) -> dict:
        d = self.read_register(REG_DIAAGC)
        mag = self.read_register(REG_MAG)
        return {
            "agc": d & 0xFF,
            "mag_too_low": bool(d & 0x800),    # MAGL, se activa con AGC = 0xFF
            "mag_too_high": bool(d & 0x400),   # MAGH, se activa con AGC = 0x00
            "cordic_overflow": bool(d & 0x200),
            "lf_ready": bool(d & 0x100),
            "magnitude": mag,
            "raw": d,
        }

    @staticmethod
    def estado_iman(d: dict) -> str:
        if d.get("mag_too_low"):
            return "IMAN MUY DEBIL/LEJOS (MAGL)"
        if d.get("mag_too_high"):
            return "IMAN MUY FUERTE/CERCA (MAGH)"
        if d.get("agc", 128) >= AGC_AVISO_CAMPO_BAJO:
            return "aviso: campo bajo, acerca el iman"
        if d.get("agc", 128) <= AGC_AVISO_CAMPO_ALTO:
            return "aviso: campo alto, aleja el iman"
        return "iman OK"

    # ----------------------------------------------------------- configuracion
    def _armar_settings(self, cfg: dict) -> tuple[int, int]:
        # SETTINGS1 (Figura 28). Bit0 FactorySetting = 1, bit1 Not Used = 0.
        s1 = 0x01
        if cfg["direction_reverse"]:
            s1 |= 1 << 2                      # DIR
        if cfg["interface"].upper() == "UVW":
            s1 |= 1 << 3                      # UVW_ABI
        if not cfg["daec_enable"]:
            s1 |= 1 << 4                      # DAECDIS (1 = compensacion OFF)
        if cfg["abi_binary"]:
            s1 |= 1 << 5                      # ABIBIN
        if cfg["angle_reg_cordic"]:
            s1 |= 1 << 6                      # Dataselect
        if cfg["pwm_enable"]:
            s1 |= 1 << 7                      # PWMon

        # SETTINGS2 (Figura 29)
        tabla = ABI_BINARY if cfg["abi_binary"] else ABI_DECIMAL
        if cfg["abi_ppr"] not in tabla:
            raise ValueError(
                f"abi_ppr={cfg['abi_ppr']} no valido con "
                f"abi_binary={cfg['abi_binary']}. Opciones: "
                f"{sorted(tabla, reverse=True)}"
            )
        if not 1 <= cfg["uvw_pole_pairs"] <= 7:
            raise ValueError("UVWPP solo admite de 1 a 7 pares de polos")

        abires, _ = tabla[cfg["abi_ppr"]]
        s2 = (cfg["uvw_pole_pairs"] - 1) & 0x07
        s2 |= (HYS_TABLE[cfg["hysteresis_lsb"]] & 0x03) << 3
        s2 |= (abires & 0x07) << 5
        return s1, s2

    def aplicar_config(self, cfg: dict | None = None,
                       cero_override: int | None = None) -> dict:
        """
        Escribe la configuracion (soft write, volatil) y devuelve lo que
        quedo REALMENTE en el chip, leido de vuelta.
        """
        cfg = cfg or self.config
        s1, s2 = self._armar_settings(cfg)

        zpos = cero_override if cero_override is not None else cfg["zero_position"]
        if zpos is None:
            # None = conservar el cero que ya tenga el chip.
            zm = self.read_register(REG_ZPOSM)
            zl = self.read_register(REG_ZPOSL)
            zpos = ((zm & 0xFF) << 6) | (zl & 0x3F)
        zpos &= 0x3FFF

        zposl = zpos & 0x3F
        if cfg["error_on_mag_low"]:
            zposl |= 1 << 6
        if cfg["error_on_mag_high"]:
            zposl |= 1 << 7

        self.write_register(REG_ZPOSM, (zpos >> 6) & 0xFF)
        self.write_register(REG_ZPOSL, zposl)
        # Bit0 de SETTINGS1 es de solo lectura: se excluye de la verificacion.
        self.write_register(REG_SETTINGS1, s1, mask=0xFFFE)
        self.write_register(REG_SETTINGS2, s2)
        self.limpiar_errores()
        return self.leer_config()

    def leer_config(self) -> dict:
        """Config REAL del chip. Es la que mira main_partido.py al arrancar."""
        s1 = self.read_register(REG_SETTINGS1)
        s2 = self.read_register(REG_SETTINGS2)
        zm = self.read_register(REG_ZPOSM)
        zl = self.read_register(REG_ZPOSL)

        binario = bool(s1 & (1 << 5))
        tabla = ABI_BINARY if binario else ABI_DECIMAL
        abires = (s2 >> 5) & 0x07
        ppr, pasos = next(
            ((k, v[1]) for k, v in tabla.items() if v[0] == abires), (None, None))
        hys = next((k for k, v in HYS_TABLE.items()
                    if v == ((s2 >> 3) & 0x03)), None)

        return {
            "SETTINGS1": s1, "SETTINGS2": s2,
            "direction_reverse": bool(s1 & (1 << 2)),
            "interface": "UVW" if s1 & (1 << 3) else "ABI",
            "daec_enable": not bool(s1 & (1 << 4)),
            "abi_binary": binario,
            "angle_reg_cordic": bool(s1 & (1 << 6)),
            "pwm_enable": bool(s1 & (1 << 7)),
            "abi_ppr": ppr,
            "abi_steps": pasos,
            "hysteresis_lsb": hys,
            "uvw_pole_pairs": (s2 & 0x07) + 1,
            "zero_position": ((zm & 0xFF) << 6) | (zl & 0x3F),
            "error_on_mag_high": bool(zl & (1 << 6)),
            "error_on_mag_low": bool(zl & (1 << 7)),
        }

    def poner_cero_aca(self, cfg: dict | None = None) -> dict:
        """
        Pone el cero del ENCODER en la posicion mecanica actual del eje.

        OJO: esto cambia el 79.0 de config_hw.ENCODER_GRADOS_EN_MOTOR_CERO.
        Si lo corres, el homing del motor queda mal hasta que vuelvas a medir
        ese numero. En el flujo normal NO hace falta: el cero del sistema lo
        define el motor, no el encoder.
        """
        cfg = cfg or self.config
        self.write_register(REG_ZPOSM, 0x00)
        self.write_register(REG_ZPOSL, 0x00)
        time.sleep(0.01)
        raw, _, _ = self.leer_angulo(compensado=False)
        return self.aplicar_config(cfg, cero_override=raw)

    def close(self) -> None:
        try:
            self.spi.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


# --------------------------------------------------------------------------- #
# CLI de prueba
# --------------------------------------------------------------------------- #

def imprimir_config(c: dict) -> None:
    print("\n--- CONFIGURACION ACTUAL ---")
    print(f"  SETTINGS1        : 0x{c['SETTINGS1']:02X}")
    print(f"  SETTINGS2        : 0x{c['SETTINGS2']:02X}")
    print(f"  Sentido invertido: {c['direction_reverse']}")
    print(f"  Interfaz         : {c['interface']}")
    print(f"  ABI              : {c['abi_ppr']} ppr = {c['abi_steps']} "
          f"pasos/vuelta ({'binaria' if c['abi_binary'] else 'decimal'})")
    if c["abi_steps"]:
        print(f"                     en x4: {360.0 / c['abi_steps']:.4f} deg/paso")
    print(f"  Histeresis       : {c['hysteresis_lsb']} LSB")
    print(f"  DAEC             : {'ON' if c['daec_enable'] else 'OFF'}")
    print(f"  Cero             : {c['zero_position']} "
          f"({c['zero_position'] / 16384.0 * 360.0:.2f} deg)")
    print("----------------------------\n")


def monitor(enc: Encoder) -> None:
    print("Lectura continua (Ctrl+C para detener)...")
    diag = enc.diagnostico()
    n = 0
    try:
        while True:
            raw, deg, err = enc.leer_angulo()
            if n % 10 == 0:                     # refresca el AGC ~5 veces/s
                diag = enc.diagnostico()
            n += 1
            if err:
                info = enc.limpiar_errores()
                print(f"\n[SPI] framing={info['framing']} "
                      f"cmd_invalido={info['comando_invalido']} "
                      f"paridad={info['paridad']} (0x{info['raw']:04X})")
                time.sleep(0.3)
                continue
            estado = Encoder.estado_iman(diag)
            alerta = "" if estado == "iman OK" else f"  <<< {estado}"
            print(f"\rRAW {raw:05d} | {deg:06.2f} deg | AGC {diag['agc']:3d} "
                  f"| MAG {diag['magnitude']:5d}{alerta}        ", end="")
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n\nfin.")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="AS5047P")
    ap.add_argument("--bus", type=int, default=chw.ENC_BUS)
    ap.add_argument("--device", type=int, default=chw.ENC_DEVICE)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--zero-aca", action="store_true")
    ap.add_argument("--monitor", action="store_true")
    args = ap.parse_args()

    sin_flags = not any((args.show, args.diag, args.apply,
                         args.zero_aca, args.monitor))

    with Encoder(args.bus, args.device) as enc:
        if args.zero_aca:
            c = enc.poner_cero_aca()
            print("cero del ENCODER movido a la posicion actual.")
            print("!! Volve a medir ENCODER_GRADOS_EN_MOTOR_CERO en config_hw.py")
            imprimir_config(c)
        elif args.apply or sin_flags:
            imprimir_config(enc.aplicar_config())

        if args.show:
            imprimir_config(enc.leer_config())
        if args.diag:
            d = enc.diagnostico()
            print(f"\nAGC {d['agc']}/255 -> {Encoder.estado_iman(d)}")
            print(f"magnitud {d['magnitude']} | MAGL {d['mag_too_low']} | "
                  f"MAGH {d['mag_too_high']} | CORDIC ovf {d['cordic_overflow']}\n")
        if args.monitor or sin_flags:
            monitor(enc)


if __name__ == "__main__":
    main()
