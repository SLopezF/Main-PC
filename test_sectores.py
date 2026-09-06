"""
test_sectores.py

Los tres casos del criterio de aceptacion del briefing, mas los bordes.
Angulos sinteticos: no hace falta video, ni motor, ni la Pi.

    python3 test_sectores.py

Cada uno de los tres ataca un modo de fallo distinto:
  - la rampa: que CAMBIE cuando tiene que cambiar (ni de mas ni de menos)
  - la senoidal: que NO oscile sobre un borde, que es el fallo que arruina
    el video
  - el salto: que reaccione RAPIDO a un pelotazo
"""

import math

import config_hw as chw
import sectores
from sectores import Sectorizador


# =============================================================================
# La particion
# =============================================================================

def test_bordes_y_centros():
    assert sectores.bordes() == [20, 40, 60, 80, 100, 120, 140, 160]
    assert sectores.centros() == [30, 50, 70, 90, 110, 130, 150]


def test_sector_de():
    assert sectores.sector_de(30) == 0
    assert sectores.sector_de(90) == 3
    assert sectores.sector_de(150) == 6
    # fuera de rango: el del extremo, el motor no llega mas alla igual
    assert sectores.sector_de(0) == 0
    assert sectores.sector_de(180) == 6


def test_arranca_mirando_al_frente():
    s = Sectorizador()
    assert s.angulo_objetivo == 90.0


def test_el_objetivo_es_el_centro_no_el_angulo():
    """La decision de diseño central: el motor va al centro del sector."""
    s = Sectorizador()
    r = s.actualizar(88.0, 0.0, 0.0)
    assert r.angulo_objetivo == 90.0, r.angulo_objetivo


# =============================================================================
# Criterio 1: rampa lenta de 20 a 160 -> exactamente 6 cambios
# =============================================================================

def test_rampa_lenta_produce_seis_cambios():
    s = Sectorizador(sector_inicial=0)
    cambios = []

    # 1 grado por segundo, muestreado a 20 Hz: bien por debajo de OMEGA_RAPIDA,
    # asi que se exige la permanencia en cada salto.
    omega = 1.0
    t = 0.0
    ang = 20.0
    while ang <= 160.0:
        r = s.actualizar(ang, omega, t)
        if r.cambio:
            cambios.append((round(ang, 1), r.sector))
        t += 0.05
        ang += omega * 0.05

    assert len(cambios) == 6, cambios
    assert [c[1] for c in cambios] == [1, 2, 3, 4, 5, 6], cambios
    assert s.angulo_objetivo == 150.0


# =============================================================================
# Criterio 2: senoidal sobre un borde durante 30 s -> CERO cambios
# =============================================================================

def test_senoidal_sobre_un_borde_no_produce_cambios():
    """
    El fallo que arruina el video: la pelota parada sobre un limite haciendo
    saltar el motor de ida y vuelta.
    """
    s = Sectorizador(sector_inicial=3)          # sector [80, 100]
    cambios = 0
    t = 0.0
    while t < 30.0:
        ang = 80.0 + 4.0 * math.sin(2 * math.pi * t / 2.0)   # +-4 sobre 80
        omega = 4.0 * (2 * math.pi / 2.0) * math.cos(2 * math.pi * t / 2.0)
        r = s.actualizar(ang, omega, t)
        cambios += int(r.cambio)
        t += 0.025
    assert cambios == 0, f"{cambios} cambios: el motor oscilaria"


def test_oscilacion_mas_grande_que_la_histeresis_si_cambia():
    """Control del test anterior: con +-12 grados si tiene que cambiar."""
    s = Sectorizador(sector_inicial=3)
    cambios = 0
    t = 0.0
    while t < 30.0:
        ang = 80.0 + 12.0 * math.sin(2 * math.pi * t / 4.0)
        r = s.actualizar(ang, 1.0, t)
        cambios += int(r.cambio)
        t += 0.025
    assert cambios > 0


# =============================================================================
# Criterio 3: salto de 60 grados con omega alta -> cambio en menos de 100 ms
# =============================================================================

def test_salto_rapido_cambia_en_menos_de_100ms():
    s = Sectorizador(sector_inicial=3)          # mirando a 90
    s.actualizar(90.0, 0.0, 0.0)

    t0 = 10.0                                    # lejos del ultimo movimiento
    omega = 600.0                                # muy por encima de OMEGA_RAPIDA
    t, cambio_en = t0, None
    ang = 150.0                                  # +60 grados de golpe
    while t < t0 + 0.5:
        r = s.actualizar(ang, omega, t)
        if r.cambio:
            cambio_en = (t - t0) * 1000.0
            break
        t += 0.025

    assert cambio_en is not None, "no cambio nunca"
    assert cambio_en < 100.0, f"tardo {cambio_en:.0f} ms"
    assert s.angulo_objetivo == 150.0


# =============================================================================
# Los tres frenos, uno por uno
# =============================================================================

def test_la_histeresis_frena_el_cruce_apenas_pasado_el_borde():
    s = Sectorizador(sector_inicial=3)           # [80, 100]
    h = chw.HISTERESIS_SECTOR_DEG
    # apenas pasado el borde, pero sin superar el margen
    for i in range(50):
        r = s.actualizar(100.0 + h - 0.5, 1.0, i * 0.1)
        assert not r.cambio
    assert s.sector == 3


def test_la_permanencia_frena_un_cruce_instantaneo():
    s = Sectorizador(sector_inicial=3)
    # supera el margen, pero un solo frame
    r = s.actualizar(120.0, 1.0, 10.0)
    assert not r.cambio and "votando" in r.motivo


def test_el_voto_se_reinicia_si_cambia_el_candidato():
    s = Sectorizador(sector_inicial=3)
    s.actualizar(120.0, 1.0, 10.0)               # vota al 5
    s.actualizar(60.0, 1.0, 10.3)                # ahora vota al 2
    # aunque ya pasaron 600 ms desde el PRIMER voto, el candidato cambio
    r = s.actualizar(60.0, 1.0, 10.7)
    assert not r.cambio, r.motivo


def test_el_piso_de_tiempo_frena_dos_movimientos_seguidos():
    s = Sectorizador(sector_inicial=3)
    r = s.actualizar(150.0, 600.0, 10.0)         # rapido: cambia ya
    assert r.cambio
    r = s.actualizar(30.0, 600.0, 10.1)          # 100 ms despues
    assert not r.cambio and "reciente" in r.motivo


def test_regimen_rapido_igual_exige_el_margen():
    """
    La velocidad saltea la permanencia, NO la histeresis: el ruido de la
    deteccion no desaparece porque la pelota vaya rapido.
    """
    s = Sectorizador(sector_inicial=3)
    h = chw.HISTERESIS_SECTOR_DEG
    r = s.actualizar(100.0 + h - 0.5, 600.0, 10.0)
    assert not r.cambio, r.motivo


def test_forzar_saltea_todo():
    s = Sectorizador(sector_inicial=0)
    r = s.forzar(90.0, 0.0)
    assert r.cambio and r.angulo_objetivo == 90.0


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
