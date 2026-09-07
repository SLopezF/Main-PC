"""
control_motor.py

Mueve el motor sin bloquear nunca al que llama.

EL PROBLEMA QUE RESUELVE
`motor_lib.esperar_fin()` hace polling por el puerto serie cada 50 ms, con
timeout de 12 s. Si eso vive en el loop principal, el sistema DEJA DE VER LA
PELOTA justo cuando el motor se mueve, que es exactamente cuando la jugada se
esta yendo a otro lado. Y no es un problema chico: a 40 fps, medio segundo de
movimiento son 20 frames perdidos.

Aca el movimiento vive en un hilo aparte y `apuntar()` solo deja el objetivo
en un buzon.

EL BUZON ES DE UN SOLO ELEMENTO Y SE PISA
`queue.Queue(maxsize=1)`: si ya hay un objetivo esperando, se descarta y se
mete el nuevo. No es una cola, es una pizarra.

Por que: una cola FIFO haria que el motor recorra uno por uno todos los
objetivos viejos antes de llegar al actual, o sea persiguiendo el pasado. El
unico objetivo que importa es el ultimo. Si mientras el motor se movia
llegaron cinco, los cuatro del medio ya no existen.

QUE HACE EL FIRMWARE Y QUE HACE ESTE MODULO
Ojo con confundir dos cosas parecidas. El firmware (AccelStepper) YA descarta
el objetivo viejo: un 'degto' nuevo durante un movimiento reemplaza el
destino, recalcula la rampa, y si el eje ya se paso, frena y vuelve. En eso
este modulo no agrega nada.

Lo que si agrega es que el LLAMADOR no bloquee. `motor_lib.ir_a_grados()`
escribe por serie y lee hasta el marcador [eot]: son milisegundos de ida y
vuelta, y `esperar_fin()` hace polling cada 50 ms hasta 12 s. Nada de eso
puede vivir en un loop de 40 fps.

Y por eso el hilo re-apunta EN CALIENTE: cuando llega un objetivo nuevo se
manda enseguida, sin esperar a que termine el anterior, porque el firmware
sabe re-apuntar. Esperar seria pagar hasta medio segundo de retraso de gusto.

QUE PASA CON EL ENCODER
Despues de cada movimiento se lee el encoder y se compara contra lo que
deberia marcar. Ese error es la unica forma de enterarse de que el motor
perdio pasos: el firmware no tiene manera de saberlo, su contador miente sin
avisar. Dos errores seguidos por encima de la tolerancia levantan una alerta.

USO
    cm = ControlMotor(motor, encoder, offset_encoder=79.0)
    cm.apuntar(110.0)          # angulo del MUNDO, no grados de motor
    cm.estado()
    cm.parar()

    python3 control_motor.py           # prueba de las 20 Hz, sin fierro
"""

import queue
import threading
import time

import config_hw as chw
import geometria


class ControlMotor:
    """
    Hilo daemon que consume objetivos de un buzon de un solo elemento.

    `apuntar()` es lo unico que llama el loop principal y no puede tardar: su
    unico trabajo es dejar un numero en el buzon.
    """

    def __init__(self, motor, encoder=None,
                 offset_encoder: float | None = None,
                 tolerancia: float | None = None,
                 verificar: bool = True,
                 timeout_mov: float | None = None):
        self.motor = motor
        self.encoder = encoder
        self.offset_encoder = (offset_encoder
                               if offset_encoder is not None
                               else chw.ENCODER_GRADOS_EN_MOTOR_CERO)
        self.tolerancia = (tolerancia if tolerancia is not None
                           else chw.TOLERANCIA_HOMING_DEG)
        self.verificar = bool(verificar) and encoder is not None
        self.timeout_mov = (timeout_mov if timeout_mov is not None
                            else chw.MOTOR_TIMEOUT_MOV_S)

        # maxsize=1 y se pisa: ver la nota de arriba.
        self._buzon: queue.Queue = queue.Queue(maxsize=1)
        self._parar = threading.Event()
        self._lock = threading.Lock()

        # --- estado compartido, siempre bajo _lock
        self._objetivo = None            # ultimo angulo del mundo pedido
        self._moviendo = False
        self._movimientos = 0
        self._descartados = 0            # objetivos pisados sin ejecutar
        self._ultimo_error = None        # contra el encoder, en grados
        self._errores_seguidos = 0
        self._alerta = None
        self._clampeados = 0
        self._pendiente = None           # objetivo que llego a mitad de camino
        self._retargets = 0              # veces que se re-apunto sin esperar

        self._hilo = threading.Thread(target=self._correr, daemon=True,
                                      name="control_motor")
        self._hilo.start()

    # ------------------------------------------------------------- interfaz
    def apuntar(self, angulo_mundo: float) -> None:
        """
        Pide apuntar a un angulo del MUNDO (0..180). No bloquea nunca: si el
        motor esta ocupado, el objetivo queda esperando y pisa al anterior.

        La conversion a grados de motor la hace el HILO, no aca: aunque sea
        barata, todo lo que se pueda sacar del camino del llamador se saca.
        """
        with self._lock:
            self._objetivo = float(angulo_mundo)

        try:
            self._buzon.put_nowait(float(angulo_mundo))
        except queue.Full:
            # Ya habia uno esperando: se tira y se pone el nuevo. Puede correr
            # con el hilo sacandolo justo ahora; en ese caso el put de abajo
            # entra y no se pierde nada.
            try:
                self._buzon.get_nowait()
                with self._lock:
                    self._descartados += 1
            except queue.Empty:
                pass
            try:
                self._buzon.put_nowait(float(angulo_mundo))
            except queue.Full:
                pass

    def estado(self) -> dict:
        with self._lock:
            return {
                "objetivo": self._objetivo,
                "ultimo_error_encoder": self._ultimo_error,
                "moviendo": self._moviendo,
                "movimientos": self._movimientos,
                "descartados": self._descartados,
                "clampeados": self._clampeados,
                "retargets": self._retargets,
                "alerta": self._alerta,
            }

    def alerta(self) -> str | None:
        with self._lock:
            return self._alerta

    def limpiar_alerta(self) -> None:
        with self._lock:
            self._alerta = None
            self._errores_seguidos = 0

    def esperar_ocioso(self, timeout: float = 15.0) -> bool:
        """Bloquea hasta que no quede nada pendiente. Solo para tests y cierre."""
        limite = time.monotonic() + timeout
        while time.monotonic() < limite:
            with self._lock:
                ocioso = not self._moviendo
            if ocioso and self._buzon.empty():
                return True
            time.sleep(0.01)
        return False

    def parar(self, timeout: float = 3.0) -> None:
        """Termina el hilo. El motor NO se deshabilita: eso lo hace quien lo abrio."""
        self._parar.set()
        try:
            self._buzon.put_nowait(None)     # despertar al hilo si esta esperando
        except queue.Full:
            pass
        self._hilo.join(timeout=timeout)

    # --------------------------------------------------------------- hilo
    def _correr(self) -> None:
        while not self._parar.is_set():
            try:
                angulo = self._buzon.get(timeout=0.2)
            except queue.Empty:
                continue
            if angulo is None or self._parar.is_set():
                break
            try:
                self._ejecutar(float(angulo))
            except Exception as exc:
                # Un fallo del puerto serie no puede matar el hilo: si el hilo
                # muere, el motor deja de responder para siempre y nada avisa.
                with self._lock:
                    self._alerta = f"error moviendo el motor: {exc}"
                    self._moviendo = False

    def _ejecutar(self, angulo_mundo: float) -> None:
        """
        Manda el objetivo y espera el fin, PERO atendiendo el buzon mientras
        tanto: si llega uno nuevo, se re-apunta en caliente sin esperar.

        POR QUE SE PUEDE RE-APUNTAR EN CALIENTE
        El firmware (AccelStepper) ya resuelve el retarget: un 'degto' nuevo
        durante un movimiento reemplaza el destino y recalcula la rampa, y si
        el eje ya se paso, frena y vuelve. No hay que esperar nada.

        Y POR QUE IMPORTA
        Sin esto, un objetivo nuevo tenia que esperar a que terminara el
        movimiento anterior: con medio segundo de recorrido, medio segundo de
        retraso pagado de gusto, justo cuando la jugada se esta yendo. El
        buzon evitaba EJECUTAR objetivos viejos, pero igual serializaba.

        La verificacion contra el encoder solo corre si el movimiento llego a
        terminar: si se re-apunto en el medio, el motor nunca estuvo donde
        decia el objetivo viejo y comparar contra el no significa nada.
        """
        grados = self._enviar(angulo_mundo)

        while True:
            termino = self._esperar_atendiendo_buzon()
            if termino is not None:          # el movimiento termino
                break
            # Llego un objetivo nuevo: se re-apunta sin esperar.
            with self._lock:
                nuevo = self._pendiente
                self._pendiente = None
                self._retargets += 1
            grados = self._enviar(nuevo)

        with self._lock:
            self._movimientos += 1
            self._moviendo = False
            if not termino:
                self._alerta = (
                    "timeout esperando el fin del movimiento: el motor puede "
                    "estar trabado o sin corriente suficiente.")

        if termino and self.verificar:
            self._verificar(grados)

    def _enviar(self, angulo_mundo: float) -> float:
        """Convierte a grados de motor, clampea y manda. Devuelve los grados."""
        grados, clampeado = geometria.angulo_a_grados_motor(angulo_mundo)

        with self._lock:
            self._moviendo = True
            if clampeado:
                self._clampeados += 1
                self._alerta = (
                    f"objetivo {angulo_mundo:.1f} deg fuera del recorrido: "
                    f"se recorto a {grados:+.1f} de motor. El motor va a mirar "
                    f"a otro lado del que pide el detector.")

        self.motor.ir_a_grados(grados)
        return grados

    def _esperar_atendiendo_buzon(self, poll: float = 0.02):
        """
        Espera el fin del movimiento sin ignorar el buzon.

        Devuelve True si termino, False si se agoto el timeout, y None si
        llego un objetivo nuevo (que queda en self._pendiente).
        """
        limite = time.monotonic() + self.timeout_mov
        while True:
            try:
                nuevo = self._buzon.get_nowait()
            except queue.Empty:
                pass
            else:
                if nuevo is None or self._parar.is_set():
                    return True          # cierre: se corta la espera
                with self._lock:
                    self._pendiente = float(nuevo)
                return None

            if not self.motor.en_movimiento():
                return True
            if time.monotonic() > limite:
                return False
            time.sleep(poll)

    def _verificar(self, grados_motor: float) -> None:
        """
        Compara el encoder contra donde el firmware cree que quedo el motor.
        Dos errores seguidos por encima de la tolerancia son pasos perdidos:
        uno solo puede ser backlash o el eje asentandose.
        """
        try:
            medido = self.encoder.leer_angulo()[1]
        except Exception as exc:
            with self._lock:
                self._alerta = f"no pude leer el encoder: {exc}"
            return

        esperado = (self.offset_encoder
                    + grados_motor / chw.RELACION_TRANSMISION) % 360.0
        error = geometria.diferencia_angular(medido, esperado)

        with self._lock:
            self._ultimo_error = error
            if abs(error) > self.tolerancia:
                self._errores_seguidos += 1
                if self._errores_seguidos >= 2:
                    self._alerta = (
                        f"PASOS PERDIDOS: dos movimientos seguidos con error "
                        f"de encoder por encima de {self.tolerancia:.2f} deg "
                        f"(ultimo {error:+.2f}). Subi MOTOR_CORRIENTE_MA o "
                        f"baja MOTOR_ACELERACION.")
            else:
                self._errores_seguidos = 0

    # ------------------------------------------------------- context manager
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.parar()
        return False


# --------------------------------------------------------------------------- #
# Prueba de las 20 Hz
# --------------------------------------------------------------------------- #

def prueba_20hz(segundos: float = 10.0, hz: float = 20.0,
                usar_falsos: bool = True) -> int:
    """
    Llama apuntar() a `hz` con angulos aleatorios y mide cuanto tarda CADA
    llamada. Criterio del briefing: ninguna por encima de 1 ms.

    Es la prueba que importa: el motor tarda cientos de milisegundos en
    moverse, asi que si apuntar() bloqueara aunque sea un poco, el loop de
    deteccion se caeria de 40 fps y las llamadas lentas apareceran aca.
    """
    import random
    import statistics

    if usar_falsos:
        import hw_falsos
        motor, encoder = hw_falsos.Motor(), hw_falsos.Encoder()
        motor._encoder = encoder
        motor.set_micropasos(chw.MOTOR_MICROPASOS)
        motor.set_velocidad(chw.MOTOR_VELOCIDAD)
        print("[prueba] hw_falsos: el motor tarda lo que tardaria el real")
    else:
        import encoder_lib
        import motor_lib
        motor, encoder = motor_lib.Motor(), encoder_lib.Encoder()
        motor.habilitar()
        print(f"[prueba] FIERRO REAL en {motor.puerto}")

    cm = ControlMotor(motor, encoder)
    tiempos = []
    n = int(segundos * hz)
    periodo = 1.0 / hz

    print(f"[prueba] {n} llamadas a {hz:.0f} Hz durante {segundos:.0f} s...")
    t_siguiente = time.perf_counter()
    for _ in range(n):
        angulo = random.uniform(0.0, 180.0)
        t0 = time.perf_counter()
        cm.apuntar(angulo)
        tiempos.append((time.perf_counter() - t0) * 1000.0)

        t_siguiente += periodo
        dormir = t_siguiente - time.perf_counter()
        if dormir > 0:
            time.sleep(dormir)

    est = cm.estado()
    cm.parar()

    tiempos.sort()
    p50 = statistics.median(tiempos)
    p95 = tiempos[int(len(tiempos) * 0.95)]
    peor = tiempos[-1]

    print()
    print(f"apuntar()   p50 {p50 * 1000:7.1f} us   "
          f"p95 {p95 * 1000:7.1f} us   peor {peor:7.3f} ms")
    print(f"movimientos ejecutados : {est['movimientos']}")
    print(f"objetivos descartados  : {est['descartados']}  "
          f"(pisados por uno mas nuevo: es lo correcto)")
    print(f"ultimo error de encoder: {est['ultimo_error_encoder']}")
    if est["alerta"]:
        print(f"ALERTA: {est['alerta']}")

    ok = peor < 1.0
    print()
    print(f"{'OK' if ok else 'FALLA'}: la peor llamada tardo {peor:.3f} ms "
          f"(criterio: < 1 ms)")
    return 0 if ok else 1


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--segundos", type=float, default=10.0)
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--real", action="store_true",
                    help="usar el motor de verdad (solo en la Pi)")
    args = ap.parse_args()
    return prueba_20hz(args.segundos, args.hz, usar_falsos=not args.real)


if __name__ == "__main__":
    raise SystemExit(main())
