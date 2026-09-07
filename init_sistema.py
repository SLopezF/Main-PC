"""
init_sistema.py

Arranque completo del sistema: deja el encoder configurado, el motor
habilitado y apuntando a un angulo conocido, y la GoPro lista. Devuelve un
`Sistema` con todo abierto y un informe de los errores medidos.

POR QUE ESTE ARCHIVO EXISTE
El motor no sabe donde esta. `motor_lib.posicion_grados()` devuelve el
contador de pasos del firmware, o sea lo que el motor CREE que hizo: si
perdio pasos, el contador miente y el firmware no tiene forma de saberlo. La
verdad la tiene el encoder absoluto. Todo lo que hace este modulo es mover el
motor a lugares conocidos y preguntarle al encoder si efectivamente llego.

Si los errores se pasan de `TOLERANCIA_HOMING_DEG`, el arranque FALLA
ruidosamente y no deja grabar. Arrancar un partido con el motor descalibrado
significa una hora de video apuntando al lugar equivocado, y eso no se
recupera.

EL CERO DEL ENCODER VIVE EN SOFTWARE
`encoder_lib.poner_cero_aca()` escribe el registro ZPOS del chip. NO se usa:
reescribirlo invalida el `ENCODER_GRADOS_EN_MOTOR_CERO = 89.0` que ya esta
medido, y deja mal todos los arranques siguientes. Aca el cero es un offset
guardado en el objeto `Sistema`.

SECUENCIA
  1. Encoder: aplicar config (es soft write, se pierde al cortar la
     alimentacion), verificar el PPR, mirar el diagnostico del iman, leer el
     angulo absoluto X.
  2. Motor: verificar que la ESP32 responde, configurar corriente, micropasos,
     velocidad y aceleracion, habilitar.
  3. Homing: mover -(X - ENCODER_GRADOS_EN_MOTOR_CERO), verificar contra el
     encoder, declarar el cero del motor y guardar el offset en software.
  4. Ir a 180 grados del mundo y verificar.
  5. Ir a 0 grados del mundo y verificar.
  6. Volver a 90 (al frente) y quedarse ahi.

Los pasos 4 y 5 son los EXTREMOS del recorrido. Se verifican los dos porque un
error de escala (relacion de transmision mal, micropasos mal) no se ve en el
homing —que es un movimiento corto— y si en un movimiento de 90 grados.

USO
    python3 init_sistema.py                  # secuencia completa
    python3 init_sistema.py --sin-gopro
    python3 init_sistema.py --simular        # sin fierro, para probar el flujo
"""

import time
from dataclasses import dataclass, field

import config_hw as chw
import geometria
import hw_falsos


class ErrorArranque(RuntimeError):
    """El arranque no llego a un estado confiable. El mensaje dice que mirar."""


@dataclass
class Verificacion:
    """Un punto verificado: adonde se mando el motor y que dijo el encoder."""
    nombre: str
    grados_motor: float
    encoder_esperado: float
    encoder_medido: float
    error: float
    dentro_de_tolerancia: bool

    def __str__(self) -> str:
        marca = "ok " if self.dentro_de_tolerancia else "MAL"
        return (f"[{marca}] {self.nombre:22s} motor {self.grados_motor:+7.2f}  "
                f"encoder esperado {self.encoder_esperado:7.2f}  "
                f"medido {self.encoder_medido:7.2f}  "
                f"error {self.error:+6.2f} deg")


@dataclass
class Sistema:
    """Todo lo que queda abierto tras el arranque. Cerrar con close()."""
    encoder: object = None
    motor: object = None
    gopro: object = None

    # Angulo que marca el encoder con el motor en su cero, MEDIDO en este
    # arranque. Es el cero en software: no se escribio nada en el chip.
    offset_encoder: float = chw.ENCODER_GRADOS_EN_MOTOR_CERO

    verificaciones: list = field(default_factory=list)
    config_encoder: dict = field(default_factory=dict)
    diag_iman: dict = field(default_factory=dict)
    simulado: dict = field(default_factory=dict)

    # ------------------------------------------------------------- consultas
    def encoder_esperado(self, grados_motor: float) -> float:
        """Que deberia marcar el encoder, usando el offset MEDIDO."""
        eje = float(grados_motor) / chw.RELACION_TRANSMISION
        return (self.offset_encoder + eje) % 360.0

    def leer_encoder(self) -> float:
        return self.encoder.leer_angulo()[1]

    def error_actual(self) -> float:
        """Cuanto se desvia el eje de donde el firmware cree que esta."""
        return geometria.diferencia_angular(
            self.leer_encoder(),
            self.encoder_esperado(self.motor.posicion_grados()))

    def informe(self) -> str:
        lineas = ["", "=" * 72, "INFORME DE ARRANQUE", "=" * 72]
        if self.simulado:
            lineas.append("!! SIMULADO: " + ", ".join(
                k for k, v in self.simulado.items() if v))
            lineas.append("   el motor falso mueve exacto y el encoder falso "
                          "repite lo que dice el motor:")
            lineas.append("   NUNCA vas a ver un error de homing aca.")
            lineas.append("")
        if self.diag_iman:
            estado = type(self.encoder).estado_iman(self.diag_iman)
            lineas.append(f"iman: {estado}  (agc "
                          f"{self.diag_iman.get('agc')}, magnitud "
                          f"{self.diag_iman.get('magnitude')})")
        if self.config_encoder:
            lineas.append(f"encoder: ABI {self.config_encoder.get('abi_ppr')} ppr, "
                          f"{self.config_encoder.get('abi_steps')} pasos/vuelta, "
                          f"DAEC {'ON' if self.config_encoder.get('daec_enable') else 'off'}")
        lineas.append(f"cero del encoder (software): {self.offset_encoder:.2f} deg")
        lineas.append("")
        for v in self.verificaciones:
            lineas.append(str(v))
        peor = max((abs(v.error) for v in self.verificaciones), default=0.0)
        lineas.append("")
        lineas.append(f"peor error: {peor:.2f} deg  "
                      f"(tolerancia {chw.TOLERANCIA_HOMING_DEG:.2f})")
        lineas.append("=" * 72)
        return "\n".join(lineas)

    def close(self) -> None:
        """Deja el fierro en un estado seguro. Se llama siempre, aun si fallo."""
        for nombre, obj, accion in (
            ("motor", self.motor, "deshabilitar"),
            ("gopro", self.gopro, None),
            ("encoder", self.encoder, None),
        ):
            if obj is None:
                continue
            try:
                if accion:
                    getattr(obj, accion)()
                obj.close()
            except Exception as exc:
                print(f"[init] no pude cerrar {nombre}: {exc}")


# =============================================================================
# Pasos
# =============================================================================

def _cargar_librerias(simular: bool):
    """Las reales si estan, las falsas si no (o si se pide --simular)."""
    if simular:
        print("[init] --simular: no se toca ningun fierro")
        return hw_falsos.Encoder, hw_falsos.Motor, hw_falsos.GoPro, {
            "encoder": True, "motor": True, "gopro": True}

    Encoder, enc_real = hw_falsos.cargar("encoder_lib", "Encoder",
                                         hw_falsos.Encoder)
    Motor, mot_real = hw_falsos.cargar("motor_lib", "Motor", hw_falsos.Motor)
    GoPro, gp_real = hw_falsos.cargar("gopro_lib", "GoPro", hw_falsos.GoPro)
    return Encoder, Motor, GoPro, {
        "encoder": not enc_real, "motor": not mot_real, "gopro": not gp_real}


def _paso1_encoder(sis: Sistema, Encoder) -> float:
    print("\n--- 1. encoder")
    sis.encoder = Encoder()

    if chw.ENC_APLICAR_CONFIG_AL_ARRANCAR:
        # Soft write: la config del AS5047 se pierde al cortar la
        # alimentacion, por eso se re-aplica en cada arranque.
        sis.config_encoder = sis.encoder.aplicar_config()
    else:
        sis.config_encoder = sis.encoder.leer_config()

    ppr = sis.config_encoder.get("abi_ppr")
    pasos = sis.config_encoder.get("abi_steps")
    print(f"    ABI {ppr} ppr, {pasos} pasos/vuelta")
    if ppr != chw.ENC_PPR_ESPERADO:
        raise ErrorArranque(
            f"El encoder reporta {ppr} ppr y se esperaban "
            f"{chw.ENC_PPR_ESPERADO}. Si dice None, la config no se escribio "
            f"(revisa el cableado SPI y que /dev/spidev{chw.ENC_BUS}."
            f"{chw.ENC_DEVICE} exista). Si dice otro numero, el chip quedo con "
            f"la config de fabrica o ENC_PPR_ESPERADO esta mal: OJO que un "
            f"AS5047D llega a 512 ppr como maximo, el P llega a 1024."
        )

    sis.diag_iman = sis.encoder.diagnostico()
    estado = type(sis.encoder).estado_iman(sis.diag_iman)
    print(f"    iman: {estado}")
    if sis.diag_iman.get("mag_too_low") or sis.diag_iman.get("mag_too_high"):
        raise ErrorArranque(
            f"Diagnostico del iman: {estado}. La lectura de angulo no es "
            f"confiable y todo el homing se apoya en ella. Ajusta la distancia "
            f"entre el iman y el chip antes de seguir."
        )

    x = sis.encoder.leer_angulo()[1]
    print(f"    angulo absoluto actual: {x:.2f} deg")
    return x


def _paso2_motor(sis: Sistema, Motor) -> None:
    print("\n--- 2. motor")
    sis.motor = Motor()
    if not sis.motor.conectado():
        raise ErrorArranque(
            "La ESP32 no responde. Revisa: que el cable USB sea de datos y no "
            "de solo carga; que el monitor serie del IDE de Arduino no tenga "
            "el puerto tomado; y que MOTOR_PUERTO en config_hw apunte al "
            "puerto correcto (None = autodeteccion)."
        )
    print(f"    puerto {sis.motor.puerto}")

    # Deshabilitar ANTES de configurar: cambiar micropasos con el driver
    # energizado puede hacer que el eje salte.
    sis.motor.deshabilitar()
    sis.motor.set_corriente(chw.MOTOR_CORRIENTE_MA)
    sis.motor.set_micropasos(chw.MOTOR_MICROPASOS)
    sis.motor.set_velocidad(chw.MOTOR_VELOCIDAD)
    sis.motor.set_aceleracion(chw.MOTOR_ACELERACION)
    sis.motor.habilitar()
    print(f"    {chw.MOTOR_CORRIENTE_MA} mA, {chw.MOTOR_MICROPASOS} micropasos, "
          f"{chw.MOTOR_VELOCIDAD} pasos/s")

    # El encoder falso necesita que el motor falso lo arrastre, si no queda
    # clavado y el homing simulado no significa nada.
    if hasattr(sis.motor, "_encoder"):
        sis.motor._encoder = sis.encoder


def _paso3_homing(sis: Sistema, x: float) -> None:
    print("\n--- 3. homing")
    # El motor esta corrido (X - offset) grados de EJE respecto del cero
    # mecanico. Para volver hay que moverlo en sentido contrario, convertido a
    # grados de MOTOR por la relacion de transmision.
    desvio_eje = geometria.diferencia_angular(
        x, chw.ENCODER_GRADOS_EN_MOTOR_CERO)
    correccion = chw.HOMING_SENTIDO * desvio_eje * chw.RELACION_TRANSMISION
    print(f"    encoder {x:.2f}, cero esperado "
          f"{chw.ENCODER_GRADOS_EN_MOTOR_CERO:.2f}  ->  mover "
          f"{correccion:+.2f} deg de motor")

    sis.motor.mover_grados(correccion)
    if not sis.motor.esperar_fin():
        raise ErrorArranque(
            "Timeout esperando el fin del homing. El motor puede estar trabado, "
            "sin corriente suficiente (subi MOTOR_CORRIENTE_MA) o con la "
            "aceleracion demasiado alta para la inercia del conjunto."
        )
    time.sleep(0.2)          # que el eje termine de asentarse antes de leer

    medido = sis.encoder.leer_angulo()[1]
    error = geometria.diferencia_angular(
        medido, chw.ENCODER_GRADOS_EN_MOTOR_CERO)

    sis.verificaciones.append(Verificacion(
        "homing (cero mecanico)", 0.0, chw.ENCODER_GRADOS_EN_MOTOR_CERO,
        medido, error, abs(error) <= chw.TOLERANCIA_HOMING_DEG))
    print(f"    {sis.verificaciones[-1]}")

    if abs(error) > chw.TOLERANCIA_HOMING_DEG:
        raise ErrorArranque(
            f"El homing quedo a {error:+.2f} deg del cero (tolerancia "
            f"{chw.TOLERANCIA_HOMING_DEG}). Que mirar, en orden: "
            f"(1) si el error es cercano a -2*{desvio_eje:.1f}, el motor se "
            f"movio para el lado contrario: invertí HOMING_SENTIDO en "
            f"config_hw; (2) si es proporcional al movimiento, revisa "
            f"RELACION_TRANSMISION y MOTOR_PASOS_POR_GRADO; (3) si es chico y "
            f"aleatorio, puede ser backlash: subi la tolerancia o la corriente."
        )

    # Cero del motor declarado, y cero del encoder guardado EN SOFTWARE (no se
    # escribe el registro ZPOS del chip).
    sis.motor.zero()
    sis.offset_encoder = medido
    print(f"    cero del motor declarado; offset del encoder = "
          f"{sis.offset_encoder:.2f} deg (en software)")


def _verificar_angulo(sis: Sistema, angulo_mundo: float,
                      nombre: str) -> Verificacion:
    grados, clampeado = geometria.angulo_a_grados_motor(angulo_mundo)
    if clampeado:
        print(f"    !! {angulo_mundo:.0f} deg del mundo cae fuera de "
              f"MOTOR_GRADOS_MIN/MAX ({chw.MOTOR_GRADOS_MIN}.."
              f"{chw.MOTOR_GRADOS_MAX}): se recorto a {grados:+.2f}. "
              f"Si el eje gira libre, ampliá esos limites.")

    sis.motor.ir_a_grados(grados)
    if not sis.motor.esperar_fin():
        raise ErrorArranque(
            f"Timeout yendo a {angulo_mundo:.0f} deg del mundo "
            f"({grados:+.2f} de motor)."
        )
    time.sleep(0.2)

    esperado = sis.encoder_esperado(grados)
    medido = sis.encoder.leer_angulo()[1]
    error = geometria.diferencia_angular(medido, esperado)

    v = Verificacion(nombre, grados, esperado, medido, error,
                     abs(error) <= chw.TOLERANCIA_HOMING_DEG)
    sis.verificaciones.append(v)
    print(f"    {v}")
    return v


def _paso4y5_extremos(sis: Sistema) -> None:
    print("\n--- 4. extremo: 180 grados del mundo")
    v180 = _verificar_angulo(sis, 180.0, "mundo 180")

    print("\n--- 5. extremo: 0 grados del mundo")
    v0 = _verificar_angulo(sis, 0.0, "mundo 0")

    malos = [v for v in (v180, v0) if not v.dentro_de_tolerancia]
    if malos:
        # Un error que aparece en los extremos y NO en el homing es de escala:
        # el homing es un movimiento corto y no lo revela.
        raise ErrorArranque(
            "Los extremos no verifican: " +
            "; ".join(f"{v.nombre} error {v.error:+.2f}" for v in malos) +
            f". El homing SI dio bien, asi que no es el cero: es la ESCALA. "
            f"Un movimiento de 90 grados que se queda corto o se pasa apunta a "
            f"MOTOR_PASOS_POR_GRADO (hoy {chw.MOTOR_PASOS_POR_GRADO}) o a "
            f"RELACION_TRANSMISION (hoy {chw.RELACION_TRANSMISION}). Compará "
            f"el error de 180 con el de 0: si tienen el mismo signo es un "
            f"offset, si tienen signos opuestos es escala."
        )


def _paso6_frente(sis: Sistema) -> None:
    print("\n--- 6. al frente (90 grados) y quedarse")
    _verificar_angulo(sis, 90.0, "mundo 90 (reposo)")


def _gopro(sis: Sistema, GoPro) -> None:
    print("\n--- gopro")
    sis.gopro = GoPro()
    if sis.gopro.init(timeout=60.0):
        print(f"    lista: {sis.gopro.estado()}")
    else:
        # No se aborta: el briefing es explicito en que el sistema sigue
        # trackeando sin GoPro. No poder debuggear la deteccion porque la
        # camara esta sin bateria seria absurdo.
        print("    !! no pude inicializar la GoPro. El sistema puede trackear "
              "igual, pero NO va a grabar. Revisa bateria, que este encendida "
              "y el SSID en config_hw.")


# =============================================================================
# Entrada
# =============================================================================

def inicializar(con_motor: bool = True, con_gopro: bool = True,
                simular: bool = False) -> Sistema:
    """
    Corre la secuencia completa y devuelve el `Sistema` listo.

    Lanza ErrorArranque si algo no verifica. El que llama tiene que cerrar con
    sistema.close(), o usar el context manager de abajo.
    """
    Encoder, Motor, GoPro, simulado = _cargar_librerias(simular)
    sis = Sistema(simulado=simulado)

    try:
        x = _paso1_encoder(sis, Encoder)

        if con_motor:
            _paso2_motor(sis, Motor)
            _paso3_homing(sis, x)
            _paso4y5_extremos(sis)
            _paso6_frente(sis)
        else:
            print("\n--- motor SALTEADO (con_motor=False)")

        if con_gopro:
            _gopro(sis, GoPro)
        else:
            print("\n--- gopro SALTEADA (con_gopro=False)")

    except Exception:
        sis.close()
        raise

    print(sis.informe())
    return sis


class arranque:
    """
    Context manager: cierra el sistema pase lo que pase.

        with init_sistema.arranque() as sis:
            ...
    """

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self.sistema = None

    def __enter__(self) -> Sistema:
        self.sistema = inicializar(**self._kwargs)
        return self.sistema

    def __exit__(self, *a):
        if self.sistema is not None:
            self.sistema.close()
        return False


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sin-motor", action="store_true")
    ap.add_argument("--sin-gopro", action="store_true")
    ap.add_argument("--simular", action="store_true",
                    help="usar hw_falsos: prueba el FLUJO, no la mecanica")
    ap.add_argument("--dejar-abierto", action="store_true",
                    help="no cerrar al terminar (para inspeccionar a mano)")
    args = ap.parse_args()

    try:
        sis = inicializar(con_motor=not args.sin_motor,
                          con_gopro=not args.sin_gopro,
                          simular=args.simular)
    except ErrorArranque as exc:
        print(f"\n{'=' * 72}\nARRANQUE FALLIDO\n{'=' * 72}\n{exc}\n")
        return 1

    if not args.dejar_abierto:
        sis.close()
        print("\nsistema cerrado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
