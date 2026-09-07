"""
test_control_motor.py

Verifica control_motor.py con hw_falsos, sin fierro.

    python3 test_control_motor.py

Lo que importa acá es la concurrencia: que apuntar() no bloquee nunca, que el
buzon se PISE en vez de encolar, y que el hilo sobreviva a un motor que falla.
El motor falso tarda lo que tardaria el real (hw_falsos simula la duracion del
movimiento), asi que estas pruebas ejercitan el caso de verdad.
"""

import threading
import time

import config_hw as chw
import control_motor
import hw_falsos
from control_motor import ControlMotor


def _motor(velocidad=None):
    m, e = hw_falsos.Motor(), hw_falsos.Encoder()
    m._encoder = e
    m.set_micropasos(chw.MOTOR_MICROPASOS)
    m.set_velocidad(velocidad or chw.MOTOR_VELOCIDAD)
    return m, e


# =============================================================================
# Lo esencial: no bloquear
# =============================================================================

def test_apuntar_no_bloquea_ni_con_el_motor_ocupado():
    m, e = _motor(velocidad=200)          # lento a proposito
    with ControlMotor(m, e) as cm:
        cm.apuntar(0.0)                   # arranca un movimiento largo
        time.sleep(0.05)                  # el hilo ya lo tomo

        peor = 0.0
        for i in range(50):
            t0 = time.perf_counter()
            cm.apuntar(float(i * 3))
            peor = max(peor, (time.perf_counter() - t0) * 1000.0)
        assert peor < 1.0, f"apuntar() tardo {peor:.3f} ms con el motor ocupado"


def test_el_buzon_se_pisa_en_vez_de_encolar():
    """
    Una cola FIFO haria que el motor recorra uno por uno todos los objetivos
    viejos antes de llegar al actual: perseguir el pasado. Solo importa el
    ultimo.
    """
    m, e = _motor(velocidad=300)
    with ControlMotor(m, e) as cm:
        for i in range(40):
            cm.apuntar(float(20 + i * 4))
        cm.esperar_ocioso(timeout=20.0)
        est = cm.estado()
        assert est["movimientos"] < 40, est
        assert est["descartados"] > 0, est


def test_el_ultimo_objetivo_es_el_que_gana():
    m, e = _motor(velocidad=3000)
    with ControlMotor(m, e) as cm:
        for a in (20.0, 60.0, 100.0, 140.0):
            cm.apuntar(a)
        cm.apuntar(170.0)
        cm.esperar_ocioso(timeout=20.0)
        # el motor tiene que terminar apuntando al ULTIMO pedido
        import geometria
        esperado, _ = geometria.angulo_a_grados_motor(170.0)
        assert abs(m.posicion_grados() - esperado) < 1e-6, m.posicion_grados()


# =============================================================================
# Verificacion contra el encoder
# =============================================================================

def test_el_error_de_encoder_se_reporta():
    m, e = _motor()
    with ControlMotor(m, e) as cm:
        cm.apuntar(110.0)
        cm.esperar_ocioso()
        assert cm.estado()["ultimo_error_encoder"] is not None


def test_dos_errores_seguidos_levantan_alerta_de_pasos_perdidos():
    """Uno solo puede ser backlash; dos seguidos son pasos perdidos."""
    m, e = _motor()
    original = hw_falsos.Encoder.leer_angulo
    hw_falsos.Encoder.leer_angulo = lambda self: (0, (self._angulo + 30.0) % 360, 0)
    try:
        with ControlMotor(m, e) as cm:
            cm.apuntar(100.0)
            cm.esperar_ocioso()
            assert cm.alerta() is None, "una sola desviacion no deberia alertar"

            cm.apuntar(120.0)
            cm.esperar_ocioso()
            alerta = cm.alerta()
            assert alerta and "PASOS PERDIDOS" in alerta, alerta
            assert "MOTOR_CORRIENTE_MA" in alerta, "tiene que decir que hacer"
    finally:
        hw_falsos.Encoder.leer_angulo = original


def test_un_error_aislado_no_alerta_y_se_olvida():
    m, e = _motor()
    original = hw_falsos.Encoder.leer_angulo
    estado = {"desviar": True}

    def leer(self):
        d = 30.0 if estado["desviar"] else 0.0
        return (0, (self._angulo + d) % 360.0, 0)

    hw_falsos.Encoder.leer_angulo = leer
    try:
        with ControlMotor(m, e) as cm:
            cm.apuntar(100.0)
            cm.esperar_ocioso()
            estado["desviar"] = False
            cm.apuntar(120.0)
            cm.esperar_ocioso()
            cm.apuntar(140.0)
            cm.esperar_ocioso()
            assert cm.alerta() is None, cm.alerta()
    finally:
        hw_falsos.Encoder.leer_angulo = original


# =============================================================================
# Robustez
# =============================================================================

def test_el_hilo_sobrevive_a_un_motor_que_falla():
    """
    Si el hilo muere, el motor deja de responder PARA SIEMPRE y nada avisa.
    Un fallo del puerto serie tiene que quedar como alerta, no matar el hilo.
    """
    m, e = _motor()
    original = hw_falsos.Motor.ir_a_grados
    fallar = {"si": True}

    def romper(self, g):
        if fallar["si"]:
            raise OSError("puerto serie caido")
        original(self, g)

    hw_falsos.Motor.ir_a_grados = romper
    try:
        with ControlMotor(m, e) as cm:
            cm.apuntar(100.0)
            cm.esperar_ocioso()
            assert cm.alerta() is not None
            assert cm._hilo.is_alive(), "el hilo se murio"

            fallar["si"] = False
            cm.limpiar_alerta()
            cm.apuntar(120.0)
            cm.esperar_ocioso()
            assert cm.estado()["movimientos"] >= 1
    finally:
        hw_falsos.Motor.ir_a_grados = original


def test_timeout_de_movimiento_queda_como_alerta():
    """Un motor trabado nunca deja de estar en movimiento."""
    m, e = _motor()
    original = hw_falsos.Motor.en_movimiento
    hw_falsos.Motor.en_movimiento = lambda self: True
    try:
        with ControlMotor(m, e, timeout_mov=0.3) as cm:
            cm.apuntar(100.0)
            time.sleep(0.8)
            assert "timeout" in (cm.alerta() or "").lower(), cm.alerta()
    finally:
        hw_falsos.Motor.en_movimiento = original


def test_re_apunta_en_caliente_sin_esperar_el_movimiento_anterior():
    """
    El firmware (AccelStepper) sabe re-apuntar: un objetivo nuevo durante un
    movimiento reemplaza el destino. Esperar a que termine el anterior seria
    pagar hasta medio segundo de retraso de gusto.
    """
    m, e = _motor(velocidad=150)              # movimiento largo
    with ControlMotor(m, e) as cm:
        # De un extremo al otro: apuntar al angulo donde el motor YA esta
        # seria un movimiento de longitud cero, que termina al instante.
        cm.apuntar(180.0)
        time.sleep(0.1)                       # el hilo ya esta esperando
        assert m.en_movimiento(), "el movimiento tendria que seguir en curso"

        t0 = time.perf_counter()
        cm.apuntar(0.0)
        # el nuevo objetivo tiene que llegar al motor MIENTRAS el anterior
        # sigue: no se espera el fin
        for _ in range(100):
            if cm.estado()["retargets"] >= 1:
                break
            time.sleep(0.01)
        dt = time.perf_counter() - t0
        assert cm.estado()["retargets"] >= 1, "no re-apunto en caliente"
        assert dt < 0.5, f"tardo {dt:.2f} s en re-apuntar"
        cm.esperar_ocioso(timeout=30.0)


def test_solo_se_verifica_el_movimiento_que_termino():
    """
    Si se re-apunto en el medio, el motor nunca estuvo donde decia el objetivo
    viejo: comparar el encoder contra ese objetivo no significa nada.
    """
    m, e = _motor(velocidad=150)
    with ControlMotor(m, e) as cm:
        cm.apuntar(180.0)
        time.sleep(0.1)
        cm.apuntar(0.0)
        cm.esperar_ocioso(timeout=30.0)
        # un solo movimiento contado, aunque hubo dos objetivos
        assert cm.estado()["movimientos"] == 1, cm.estado()


def test_objetivo_fuera_del_recorrido_avisa():
    m, e = _motor()
    with ControlMotor(m, e) as cm:
        cm.apuntar(400.0)                 # muy fuera de MOTOR_GRADOS_MIN/MAX
        cm.esperar_ocioso()
        est = cm.estado()
        assert est["clampeados"] >= 1
        assert "fuera del recorrido" in (est["alerta"] or "")


def test_parar_termina_el_hilo():
    m, e = _motor()
    cm = ControlMotor(m, e)
    cm.apuntar(100.0)
    cm.parar()
    assert not cm._hilo.is_alive()


def test_funciona_sin_encoder():
    """El encoder es opcional: sin el no se verifica, pero el motor se mueve."""
    m, _ = _motor()
    with ControlMotor(m, encoder=None) as cm:
        cm.apuntar(110.0)
        cm.esperar_ocioso()
        assert cm.estado()["movimientos"] == 1
        assert cm.estado()["ultimo_error_encoder"] is None


def test_apuntar_desde_varios_hilos_no_rompe():
    m, e = _motor(velocidad=3000)
    with ControlMotor(m, e) as cm:
        def golpear():
            for i in range(100):
                cm.apuntar(float(i % 180))

        hilos = [threading.Thread(target=golpear) for _ in range(4)]
        for h in hilos:
            h.start()
        for h in hilos:
            h.join()
        cm.esperar_ocioso(timeout=20.0)
        assert cm._hilo.is_alive()


if __name__ == "__main__":
    fallos = 0
    for nombre, fn in sorted(globals().items()):
        if not nombre.startswith("test_"):
            continue
        try:
            fn()
            print(f"  ok    {nombre}")
        except Exception as exc:
            fallos += 1
            print(f"  FALLO {nombre}: {exc}")
    print("\nTODO OK" if not fallos else f"\n{fallos} fallo(s)")
    raise SystemExit(1 if fallos else 0)
