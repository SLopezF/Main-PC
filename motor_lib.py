"""
motor_lib.py

Motor paso a paso (ESP32 + TMC2209 + AccelStepper) por USB, con la API que
consume main_partido.py.

PROTOCOLO
Cada comando responde N lineas y termina con el marcador '[eot]'. El cliente
lee hasta ver ese marcador; leer "hasta la primera linea vacia" corta las
respuestas multilinea (como 'help' o 'status') y hace parecer que los
comandos de configuracion no andan.

Los avisos de fin de movimiento llegan asincronicamente como
'[done] pos=... deg=...' y se guardan aparte en .eventos, sin romper el
ciclo comando/respuesta.

UNIDADES
Todo lo que expone esta clase esta en GRADOS DE MOTOR respecto de su cero,
positivo en el sentido que el firmware llame positivo. La conversion
angulo-del-mundo -> grados de motor la hace geometria.py, no esta libreria:
aca no se sabe nada de camaras ni de donde apunta el eje.

POSICION: DOS FUENTES QUE PUEDEN NO COINCIDIR
posicion_grados() devuelve lo que dice el FIRMWARE (contador de pasos). Eso
es lo que el motor cree que hizo, no lo que hizo: si perdio pasos, el
contador miente y el firmware no tiene forma de saberlo. La verdad la tiene
el encoder absoluto, y por eso main_partido.py verifica cada movimiento
contra encoder_lib. Si el firmware no reporta grados en 'pos', se cae a un
acumulado local, que miente igual pero al menos no rompe.

USO DIRECTO
    python3 motor_lib.py               # terminal interactiva
    python3 motor_lib.py --test        # ida y vuelta de 90 grados
    python3 motor_lib.py --puerto /dev/ttyUSB0
"""

import glob
import time

import serial

import config_hw as chw

EOT = "[eot]"


def buscar_puerto() -> str | None:
    """Primer /dev/ttyUSB* o /dev/ttyACM* que aparezca."""
    candidatos = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    return candidatos[0] if candidatos else None


class Motor:
    def __init__(self, puerto: str | None = chw.MOTOR_PUERTO,
                 baud: int = chw.MOTOR_BAUD,
                 espera_boot: float = 2.5,
                 timeout: float = 3.0):
        if puerto is None:
            puerto = buscar_puerto()
            if puerto is None:
                raise RuntimeError(
                    "No encontre /dev/ttyUSB* ni /dev/ttyACM*. Conecta la "
                    "ESP32 por USB o pasa puerto='/dev/ttyUSB0'."
                )
        self.puerto = puerto
        self.timeout = timeout
        self.eventos: list[str] = []       # mensajes [done] asincronicos
        self._grados_local = 0.0           # respaldo si el firmware no reporta
        self._habilitado = False

        # Timeout corto en el puerto: la espera real la maneja el bucle de
        # _leer_hasta_eot, que sabe cuando la respuesta termino.
        self.ser = serial.Serial(puerto, baud, timeout=0.1)
        time.sleep(espera_boot)            # la ESP32 se resetea al abrir el USB
        self._leer_hasta_eot(timeout=2.0)  # consume el banner de arranque

    # ------------------------------------------------------------------ infra
    def _leer_hasta_eot(self, timeout: float) -> list[str]:
        lineas = []
        limite = time.time() + timeout
        while time.time() < limite:
            crudo = self.ser.readline()
            if not crudo:
                continue                   # timeout parcial: seguimos esperando
            linea = crudo.decode(errors="ignore").strip()
            if linea == EOT:
                return lineas
            if linea.startswith("[done]"):
                self.eventos.append(linea)  # asincronico, no es la respuesta
                continue
            if linea:
                lineas.append(linea)
        lineas.append("(!) timeout: no llego el marcador [eot]")
        return lineas

    def send(self, comando: str, timeout: float | None = None) -> list[str]:
        """Manda un comando crudo y devuelve la respuesta como lista de lineas."""
        self.ser.write((comando.strip() + "\n").encode())
        self.ser.flush()
        return self._leer_hasta_eot(timeout or self.timeout)

    @staticmethod
    def _tokens(lineas) -> dict:
        """Junta todos los pares clave=valor de una respuesta en un dict."""
        d = {}
        for linea in lineas:
            for tok in linea.replace(",", " ").split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    d[k.strip().lower()] = v.strip()
        return d

    # -------------------------------------------------------------- conexion
    def conectado(self) -> bool:
        """
        True si la ESP32 contesta. Se pregunta 'status' y se exige respuesta
        con [eot]: que el puerto abra no significa nada (un cable de solo
        carga abre igual y no transmite datos).
        """
        r = self.send("status", timeout=2.0)
        return bool(r) and not any(l.startswith("(!)") for l in r)

    def status(self) -> list[str]:
        return self.send("status")

    def help(self) -> list[str]:
        return self.send("help")

    # --------------------------------------------------------- configuracion
    def set_corriente(self, mA: int) -> list[str]:
        return self.send(f"current {int(mA)}")

    def set_micropasos(self, n: int) -> list[str]:
        return self.send(f"microsteps {int(n)}")

    def set_velocidad(self, pasos_por_s: float) -> list[str]:
        return self.send(f"speed {int(pasos_por_s)}")

    def set_aceleracion(self, pasos_por_s2: float) -> list[str]:
        return self.send(f"accel {int(pasos_por_s2)}")

    def set_silencioso(self, on: bool) -> list[str]:
        return self.send("silent on" if on else "silent off")

    def set_sentido_invertido(self, invertido: bool) -> list[str]:
        return self.send(f"dir {1 if invertido else 0}")

    def habilitar(self) -> None:
        self.send("enable")
        self._habilitado = True

    def deshabilitar(self) -> None:
        self.send("disable")
        self._habilitado = False

    @property
    def habilitado(self) -> bool:
        return self._habilitado

    # ------------------------------------------------------------ movimiento
    def mover_grados(self, grados: float) -> list[str]:
        """RELATIVO, en grados de motor."""
        self._grados_local += float(grados)
        return self.send(f"deg {grados:.4f}")

    def ir_a_grados(self, grados: float) -> list[str]:
        """ABSOLUTO respecto del cero del motor."""
        self._grados_local = float(grados)
        return self.send(f"degto {grados:.4f}")

    def mover_pasos(self, pasos: int) -> list[str]:
        return self.send(f"move {int(pasos)}")

    def ir_a_pasos(self, pasos: int) -> list[str]:
        return self.send(f"moveto {int(pasos)}")

    def stop(self) -> list[str]:
        return self.send("stop")

    def estop(self) -> list[str]:
        """Parada de emergencia: corta sin rampa de desaceleracion."""
        return self.send("estop")

    def zero(self) -> None:
        """Declara la posicion actual como el cero del motor."""
        self.send("zero")
        self._grados_local = 0.0

    # -------------------------------------------------------------- consultas
    def posicion_grados(self) -> float:
        """
        Grados segun el FIRMWARE. Si no reporta grados, se cae al acumulado
        local. En los dos casos es lo que el motor cree, no lo que hizo:
        contrastalo contra el encoder.
        """
        d = self._tokens(self.send("pos"))
        for clave in ("deg", "grados", "pos_deg", "angle"):
            if clave in d:
                try:
                    return float(d[clave])
                except ValueError:
                    pass
        return self._grados_local

    def info_movimiento(self) -> dict:
        """dict {moving, pos, target, restante} a partir del comando 'moving'."""
        info = {"moving": None, "pos": None, "target": None, "restante": None}
        d = self._tokens(self.send("moving"))
        for k in info:
            if k in d:
                try:
                    info[k] = int(float(d[k]))
                except ValueError:
                    pass
        return info

    def en_movimiento(self) -> bool:
        return self.info_movimiento().get("moving") == 1

    def esperar_fin(self, timeout: float = chw.MOTOR_TIMEOUT_MOV_S,
                    poll: float = 0.05) -> bool:
        """
        Espera del lado de la Pi a que el motor termine. No frena nada ni
        acelera nada: solo consulta. Devuelve False si se agoto el timeout,
        que es informacion util (el motor puede estar trabado).
        """
        t0 = time.time()
        while True:
            if not self.en_movimiento():
                return True
            if (time.time() - t0) > timeout:
                return False
            time.sleep(poll)

    def eventos_pendientes(self) -> list[str]:
        """Vacia y devuelve los avisos [done] acumulados."""
        ev, self.eventos = self.eventos, []
        return ev

    # ------------------------------------------------------------------ cierre
    def close(self) -> None:
        try:
            self.ser.close()
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

def prueba_ida_vuelta(motor: Motor, grados: float = 90.0) -> None:
    """
    Ida y vuelta. Sirve para dos cosas: ver que el sentido positivo sea el
    que esperas, y ver si vuelve exactamente al punto de partida (si no,
    hay backlash o pasos perdidos: subi la corriente o baja la aceleracion).
    """
    print(f"corriente {chw.MOTOR_CORRIENTE_MA} mA, {grados} grados y vuelta")
    motor.deshabilitar()
    motor.set_corriente(chw.MOTOR_CORRIENTE_MA)
    motor.set_micropasos(chw.MOTOR_MICROPASOS)
    motor.set_velocidad(chw.MOTOR_VELOCIDAD)
    motor.set_aceleracion(chw.MOTOR_ACELERACION)
    motor.habilitar()
    motor.zero()

    for objetivo in (grados, 0.0):
        print(f"  -> {objetivo:+.1f} deg ...", end="", flush=True)
        motor.ir_a_grados(objetivo)
        ok = motor.esperar_fin()
        print(f" {'ok' if ok else 'TIMEOUT'} | firmware dice "
              f"{motor.posicion_grados():+.2f} deg")
        for ev in motor.eventos_pendientes():
            print(f"     {ev}")
    motor.deshabilitar()


def interactiva(motor: Motor) -> None:
    print(f"Conectado a {motor.puerto}. 'help' para comandos, 'exit' para salir.\n")
    for linea in motor.status():
        print(linea)
    while True:
        try:
            cmd = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not cmd:
            continue
        if cmd.lower() in ("exit", "quit", "q"):
            break
        for linea in motor.send(cmd):
            print(linea)
        for ev in motor.eventos_pendientes():
            print(ev)
    print("Saliendo.")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--puerto", type=str, default=chw.MOTOR_PUERTO)
    ap.add_argument("--test", action="store_true", help="ida y vuelta de 90 deg")
    ap.add_argument("--grados", type=float, default=90.0)
    args = ap.parse_args()

    with Motor(args.puerto) as motor:
        if not motor.conectado():
            print("!! la ESP32 no responde. Puede estar el monitor serie del "
                  "IDE con el puerto tomado.")
            return
        if args.test:
            prueba_ida_vuelta(motor, args.grados)
        else:
            interactiva(motor)


if __name__ == "__main__":
    main()
