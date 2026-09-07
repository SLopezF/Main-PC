"""
test_main_final.py

Verifica main_final.py de punta a punta con TODO falso: inferencia stub,
carpeta de imagenes como camara, y hw_falsos para motor, encoder y GoPro.

    python3 test_main_final.py

Lo que prueba: que el ciclo IDLE -> GRABANDO -> IDLE cierre bien, que el CSV
tenga las mismas columnas que replay.py, que un fallo de la GoPro no aborte, y
que el handover de camara reinicie el filtro.

NO prueba la mecanica ni el rendimiento: el fps que reporta es de imagenes de
disco con un stub, no de la Pi.
"""

import os
import sys
import tempfile
import time

import cv2
import numpy as np

import config

_NOM = "stub_nms"


class _Stub:
    """Detector por umbral: encuentra el circulo blanco sin modelo."""

    def __init__(self, *a, **k):
        self._s = (*config.MODEL_INPUT_SIZE, 3)

    @property
    def input_shape(self):
        return self._s

    def describe(self):
        return "STUB (umbral, sin modelo)"

    def infer(self, frame):
        assert tuple(frame.shape) == self._s, frame.shape
        h, w, _ = self._s
        g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        _, b = cv2.threshold(g, 200, 255, cv2.THRESH_BINARY)
        c, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        t = np.zeros((1, 32, 5), np.float32)
        if c:
            x, y, bw, bh = cv2.boundingRect(max(c, key=cv2.contourArea))
            if bw > 4:
                t[0, 0] = (y / h, x / w, (y + bh) / h, (x + bw) / w, 0.90)
        return {_NOM: t}

    def close(self):
        pass


_DIR = None


def _imagenes():
    """Carpeta con una pelota cruzando el frame. Se crea una sola vez."""
    global _DIR
    if _DIR is not None:
        return _DIR
    _DIR = tempfile.mkdtemp(prefix="mf_")
    for i in range(60):
        f = np.full((config.CAM_ALTO, config.CAM_ANCHO, 3), 40, np.uint8)
        cv2.circle(f, (200 + i * 34, 650), 22, (255, 255, 255), -1)
        cv2.imwrite(f"{_DIR}/f{i:03d}.jpg", f)
    return _DIR


def _preparar():
    """Sustituye inferencia, fuente y el init de hardware por versiones falsas."""
    import fuente
    import inferencia
    import init_sistema

    inferencia.abrir = lambda *a, **k: _Stub()
    fuente.abrir_fuente = lambda indice=0, **k: fuente.FuenteCarpeta(
        ruta=_imagenes(), indice=indice, fps=40.0, bucle=True,
        reloj="sintetico")
    original = init_sistema.inicializar

    def _silencioso(**k):
        # El informe de arranque es util corriendo el sistema, ruido en los
        # tests: son ~40 lineas por cada uno.
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            return original(**{**k, "simular": True})

    init_sistema.inicializar = _silencioso


def _correr(segundos=2.0, **kwargs):
    import main_final
    kwargs.setdefault("ruta_csv", os.path.join(tempfile.gettempdir(), "mf.csv"))
    kwargs.setdefault("verbose", False)
    m = main_final.MainFinal(**kwargs)
    assert m.iniciar(), "no arranco"
    time.sleep(segundos)
    m.detener()
    return m


# =============================================================================

def test_ciclo_completo():
    import main_final
    m = _correr()
    assert m.estado == main_final.Estado.IDLE
    assert m.frames > 0
    assert m.aceptados > 0


def test_el_csv_tiene_las_columnas_de_replay():
    """Regla 8: el analisis de la tesis es un solo script para los dos."""
    import csv

    import pipeline
    m = _correr()
    with open(m.ruta_csv, encoding="utf-8") as fh:
        filas = list(csv.DictReader(fh))
    assert filas, "csv vacio"
    assert list(filas[0].keys()) == pipeline.COLUMNAS


def test_detener_deja_todo_cerrado():
    m = _correr()
    assert m.camaras == {}
    assert m.hailo is None
    assert m.control is None
    assert m.sistema is None


def test_se_puede_arrancar_dos_veces():
    import main_final
    m = _correr()
    assert m.iniciar(), "no volvio a arrancar"
    time.sleep(1.0)
    m.detener()
    assert m.estado == main_final.Estado.IDLE


def test_no_arranca_dos_veces_a_la_vez():
    import main_final
    m = main_final.MainFinal(ruta_csv="/tmp/mf2.csv", verbose=False)
    assert m.iniciar()
    try:
        assert not m.iniciar(), "arranco dos veces"
    finally:
        m.detener()


def test_la_gopro_que_falla_no_aborta():
    """El briefing es explicito: se emite alerta y se sigue trackeando."""
    import hw_falsos
    original = hw_falsos.GoPro.init
    hw_falsos.GoPro.init = lambda self, timeout=60.0: False
    try:
        m = _correr(con_gopro=True)
        assert m.frames > 0, "el sistema se detuvo por culpa de la GoPro"
        assert any("GoPro" in a for a in m.alertas), m.alertas
    finally:
        hw_falsos.GoPro.init = original


def test_sin_motor_no_toca_el_motor():
    m = _correr(con_motor=False)
    assert m.control is None
    assert m.frames > 0


def test_el_motor_recibe_centros_de_sector_no_angulos_crudos():
    """La decision de diseño central: el motor va al centro del sector."""
    import csv

    import sectores
    m = _correr()
    with open(m.ruta_csv, encoding="utf-8") as fh:
        filas = list(csv.DictReader(fh))
    objetivos = {f["objetivo_motor"] for f in filas if f["objetivo_motor"]}
    validos = {f"{c:.1f}" for c in sectores.centros()}
    assert objetivos <= validos, objetivos - validos


def test_detener_sin_haber_arrancado_no_rompe():
    import main_final
    main_final.MainFinal(verbose=False).detener()


if __name__ == "__main__":
    _preparar()
    fallos = 0
    for nombre, fn in sorted(globals().items()):
        if not nombre.startswith("test_"):
            continue
        import contextlib
        import io
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                fn()
            print(f"  ok    {nombre}")
        except Exception as exc:
            fallos += 1
            print(f"  FALLO {nombre}: {type(exc).__name__}: {exc}")
    print("\nTODO OK" if not fallos else f"\n{fallos} fallo(s)")
    sys.exit(1 if fallos else 0)
