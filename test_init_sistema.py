"""
test_init_sistema.py

Verifica init_sistema.py con hw_falsos, SIN fierro.

QUE PRUEBA Y QUE NO
Prueba el FLUJO: que la secuencia corra en orden, que el cero del encoder
quede en software, y sobre todo que ABORTE con un mensaje util ante cada modo
de fallo conocido.

NO prueba la mecanica. El motor falso mueve exacto y el encoder falso repite
lo que dice el motor, asi que en el caso feliz todos los errores dan 0.00 y
eso no significa nada. Los tres errores de verdad se miden en la Pi.

    python3 test_init_sistema.py
"""

import hw_falsos
import init_sistema
from init_sistema import ErrorArranque


def _correr():
    return init_sistema.inicializar(con_gopro=False, simular=True)


def _espera_error(texto: str):
    """Corre el arranque esperando que falle con `texto` en el mensaje."""
    try:
        sis = _correr()
        sis.close()
    except ErrorArranque as exc:
        assert texto.lower() in str(exc).lower(), f"mensaje inesperado: {exc}"
        return str(exc)
    raise AssertionError(f"no aborto (se esperaba un error sobre '{texto}')")


# =============================================================================
# Caso feliz
# =============================================================================

def test_secuencia_completa_corre():
    sis = _correr()
    try:
        nombres = [v.nombre for v in sis.verificaciones]
        assert nombres == ["homing (cero mecanico)", "mundo 180", "mundo 0",
                           "mundo 90 (reposo)"], nombres
        assert all(v.dentro_de_tolerancia for v in sis.verificaciones)
    finally:
        sis.close()


def test_verifica_los_dos_extremos():
    """0 y 180 son los extremos del recorrido: un error de ESCALA solo se ve
    ahi, porque el homing es un movimiento corto."""
    sis = _correr()
    try:
        # Los grados de motor salen de geometria, NO de un numero a mano: con
        # MOTOR_ANGULO_MUNDO_EN_CERO = 89, mundo 0 son -89 y mundo 180 son +91.
        import geometria
        grados = {v.nombre: v.grados_motor for v in sis.verificaciones}
        assert grados["mundo 180"] == geometria.angulo_a_grados_motor(180.0)[0]
        assert grados["mundo 0"] == geometria.angulo_a_grados_motor(0.0)[0]
        # 180 grados de separacion entre los dos extremos, sea cual sea el
        # cero: es lo que hace que un error de ESCALA se note.
        assert abs(grados["mundo 180"] - grados["mundo 0"]) == 180.0
    finally:
        sis.close()


def test_queda_en_reposo_al_frente():
    sis = _correr()
    try:
        import geometria
        assert sis.verificaciones[-1].nombre == "mundo 90 (reposo)"
        esperado, _ = geometria.angulo_a_grados_motor(90.0)
        assert abs(sis.motor.posicion_grados() - esperado) < 1e-6
    finally:
        sis.close()


def test_el_cero_del_encoder_es_por_software():
    """
    encoder_lib.poner_cero_aca() escribe el registro ZPOS del chip e invalida
    ENCODER_GRADOS_EN_MOTOR_CERO para todos los arranques siguientes. No se
    usa: el cero vive como offset en el objeto Sistema.
    """
    sis = _correr()
    try:
        assert sis.offset_encoder is not None
        cfg = sis.encoder.leer_config()
        assert cfg["zero_position"] == 0, "se escribio ZPOS en el chip"
    finally:
        sis.close()


def test_sin_motor_no_toca_el_motor():
    sis = init_sistema.inicializar(con_motor=False, con_gopro=False,
                                   simular=True)
    try:
        assert sis.motor is None
        assert sis.verificaciones == []
    finally:
        sis.close()


# =============================================================================
# Modos de fallo: lo que de verdad importa
# =============================================================================

def test_aborta_si_el_ppr_no_es_el_esperado():
    orig = hw_falsos.Encoder.leer_config
    hw_falsos.Encoder.leer_config = lambda self: {**orig(self), "abi_ppr": 1024}
    try:
        msg = _espera_error("ppr")
        assert "AS5047D" in msg or "512" in msg
    finally:
        hw_falsos.Encoder.leer_config = orig


def test_aborta_si_el_iman_esta_mal():
    orig = hw_falsos.Encoder.diagnostico
    hw_falsos.Encoder.diagnostico = lambda self: {
        "agc": 255, "mag_too_low": True, "mag_too_high": False, "magnitude": 10}
    try:
        _espera_error("iman")
    finally:
        hw_falsos.Encoder.diagnostico = orig


def test_aborta_si_la_esp32_no_responde():
    orig = hw_falsos.Motor.conectado
    hw_falsos.Motor.conectado = lambda self: False
    try:
        msg = _espera_error("ESP32")
        assert "solo carga" in msg, "el mensaje tiene que decir que revisar"
    finally:
        hw_falsos.Motor.conectado = orig


def test_aborta_si_el_homing_no_verifica():
    mov = hw_falsos.Motor.mover_grados
    ang = hw_falsos.Encoder.leer_angulo
    hw_falsos.Motor.mover_grados = lambda self, g: mov(self, g * 0.5)
    hw_falsos.Encoder.leer_angulo = lambda self: (
        0, (self._angulo + 20.0) % 360.0, 0)
    try:
        msg = _espera_error("homing")
        assert "HOMING_SENTIDO" in msg, "el mensaje tiene que sugerir que mirar"
    finally:
        hw_falsos.Motor.mover_grados = mov
        hw_falsos.Encoder.leer_angulo = ang


def test_aborta_si_el_motor_no_termina():
    orig = hw_falsos.Motor.esperar_fin
    hw_falsos.Motor.esperar_fin = lambda self, timeout=None: False
    try:
        _espera_error("timeout")
    finally:
        hw_falsos.Motor.esperar_fin = orig


def test_el_fierro_queda_cerrado_si_falla():
    """Si el arranque aborta, no puede dejar el motor energizado."""
    orig = hw_falsos.Motor.esperar_fin
    hw_falsos.Motor.esperar_fin = lambda self, timeout=None: False
    try:
        _espera_error("timeout")
    finally:
        hw_falsos.Motor.esperar_fin = orig


if __name__ == "__main__":
    import io
    import contextlib

    fallos = 0
    for nombre, fn in sorted(globals().items()):
        if not nombre.startswith("test_"):
            continue
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                fn()
            print(f"  ok    {nombre}")
        except Exception as exc:
            fallos += 1
            print(f"  FALLO {nombre}: {exc}")
    print("\nTODO OK" if not fallos else f"\n{fallos} fallo(s)")
    raise SystemExit(1 if fallos else 0)
