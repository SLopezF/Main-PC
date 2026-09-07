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
    assert sectores.bordes() == [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]
    assert sectores.centros() == [10, 30, 50, 70, 90, 110, 130, 150, 170]


def test_sector_de():
    assert sectores.sector_de(10) == 0
    assert sectores.sector_de(90) == 4          # el del medio mira al frente
    assert sectores.sector_de(170) == 8
    # los extremos EXACTOS caen dentro, no fuera
    assert sectores.sector_de(0) == 0
    assert sectores.sector_de(180) == 8


def test_el_semiplano_completo_esta_cubierto():
    """
    Con 9 sectores sobre 0..180 no hay angulo del mundo sin sector. El diseño
    de 7 sobre 20..160 perdia los dos extremos, donde caen las jugadas contra
    el lateral.
    """
    for a in range(0, 181):
        i = sectores.sector_de(a)
        assert 0 <= i < chw.N_SECTORES, (a, i)
    assert sectores.centros()[sectores.sector_de(5)] == 10.0
    assert sectores.centros()[sectores.sector_de(175)] == 170.0


def test_arranca_mirando_al_frente():
    s = Sectorizador()
    assert s.angulo_objetivo == 90.0
    assert s.sector == 4


def test_el_objetivo_es_el_centro_no_el_angulo():
    """La decision de diseño central: el motor va al centro del sector."""
    s = Sectorizador()
    r = s.actualizar(88.0, 0.0, 0.0)
    assert r.angulo_objetivo == 90.0, r.angulo_objetivo


# =============================================================================
# Criterio 1: rampa lenta de 20 a 160 -> exactamente 6 cambios
# =============================================================================

def test_rampa_lenta_produce_un_cambio_por_borde():
    s = Sectorizador(sector_inicial=0)
    cambios = []

    # 1 grado por segundo, muestreado a 20 Hz: bien por debajo de OMEGA_RAPIDA,
    # asi que se exige la permanencia en cada salto.
    omega = 1.0
    t = 0.0
    ang = 0.0
    while ang <= 180.0:
        r = s.actualizar(ang, omega, t)
        if r.cambio:
            cambios.append((round(ang, 1), r.sector))
        t += 0.05
        ang += omega * 0.05

    # 9 sectores -> 8 bordes internos -> 8 cambios
    assert len(cambios) == 8, cambios
    assert [c[1] for c in cambios] == [1, 2, 3, 4, 5, 6, 7, 8], cambios
    assert s.angulo_objetivo == 170.0


# =============================================================================
# Criterio 2: senoidal sobre un borde durante 30 s -> CERO cambios
# =============================================================================

def test_senoidal_sobre_un_borde_no_produce_cambios():
    """
    El fallo que arruina el video: la pelota parada sobre un limite haciendo
    saltar el motor de ida y vuelta.
    """
    s = Sectorizador(sector_inicial=4)          # sector [80, 100]
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
    s = Sectorizador(sector_inicial=4)
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
    s = Sectorizador(sector_inicial=4)          # mirando a 90
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
    s = Sectorizador(sector_inicial=4)           # [80, 100]
    h = chw.HISTERESIS_SECTOR_DEG
    # apenas pasado el borde, pero sin superar el margen
    for i in range(50):
        r = s.actualizar(100.0 + h - 0.5, 1.0, i * 0.1)
        assert not r.cambio
    assert s.sector == 4


def test_sin_permanencia_cambia_apenas_supera_el_margen():
    """
    Configuracion ACTUAL: MS_PERMANENCIA = 0. La espera era contraproducente
    en el pelotazo, que es el caso que importa, y la excepcion por
    OMEGA_RAPIDA solo lo cubria si la estimacion de velocidad angular (que es
    ruidosa) superaba el umbral.
    """
    if chw.MS_PERMANENCIA > 0:
        return
    s = Sectorizador(sector_inicial=4)
    r = s.actualizar(120.0, 1.0, 10.0)           # un solo frame, omega baja
    assert r.cambio and r.motivo == "margen superado", r.motivo
    assert r.angulo_objetivo == 130.0


def test_la_permanencia_frena_un_cruce_instantaneo():
    """Solo aplica si se vuelve a activar la permanencia."""
    if chw.MS_PERMANENCIA <= 0:
        return
    s = Sectorizador(sector_inicial=4)
    r = s.actualizar(120.0, 1.0, 10.0)
    assert not r.cambio and "votando" in r.motivo


def test_el_voto_se_reinicia_si_cambia_el_candidato():
    if chw.MS_PERMANENCIA <= 0:
        return
    s = Sectorizador(sector_inicial=4)
    s.actualizar(120.0, 1.0, 10.0)               # vota al 6
    s.actualizar(60.0, 1.0, 10.3)                # ahora vota al 3
    # aunque ya pasaron 600 ms desde el PRIMER voto, el candidato cambio
    r = s.actualizar(60.0, 1.0, 10.7)
    assert not r.cambio, r.motivo


def test_el_piso_de_tiempo_frena_dos_movimientos_seguidos():
    s = Sectorizador(sector_inicial=4)
    r = s.actualizar(150.0, 600.0, 10.0)         # rapido: cambia ya
    assert r.cambio
    r = s.actualizar(30.0, 600.0, 10.1)          # 100 ms despues
    assert not r.cambio and "reciente" in r.motivo


def test_regimen_rapido_igual_exige_el_margen():
    """
    La velocidad saltea la permanencia, NO la histeresis: el ruido de la
    deteccion no desaparece porque la pelota vaya rapido.
    """
    s = Sectorizador(sector_inicial=4)
    h = chw.HISTERESIS_SECTOR_DEG
    r = s.actualizar(100.0 + h - 0.5, 600.0, 10.0)
    assert not r.cambio, r.motivo


def test_reaccion_al_pelotazo_sin_depender_de_omega():
    """
    El motivo del cambio: con permanencia, un pelotazo cuya omega ESTIMADA
    quedaba por debajo de OMEGA_RAPIDA tenia que esperar igual. Sin
    permanencia, reacciona en el primer frame pase lo que pase con omega.
    """
    if chw.MS_PERMANENCIA > 0:
        return
    s = Sectorizador(sector_inicial=4)
    s.actualizar(90.0, 0.0, 0.0)
    # omega chica (estimacion pobre) pero la pelota ya esta dos sectores mas
    # alla: tiene que moverse en el mismo frame
    r = s.actualizar(135.0, 2.0, 10.0)
    assert r.cambio and r.sector == 6, (r.cambio, r.sector)


# =============================================================================
# Cobertura por camara y bordes en pixeles
# =============================================================================

def test_cada_camara_ve_su_mitad_y_el_centro():
    v0, v1 = sectores.sectores_visibles(0), sectores.sectores_visibles(1)
    assert 0 in v0 and 8 not in v0
    assert 8 in v1 and 0 not in v1
    # el sector del medio (80..100, centro 90) cae en las DOS: es el solape
    assert 4 in v0 and 4 in v1


def test_bordes_en_pixeles_caen_dentro_de_la_imagen():
    ancho = 2304
    for cam in (0, 1):
        pares = sectores.bordes_en_pixeles(cam, ancho)
        assert pares, f"la camara {cam} no ve ningun borde"
        for angulo, px in pares:
            assert 0 <= px <= ancho, (cam, angulo, px)


def test_los_bordes_crecen_con_el_angulo():
    """Si esto falla, la camara esta espejada y CAM_ESPEJO no lo refleja."""
    for cam in (0, 1):
        pares = sectores.bordes_en_pixeles(cam, 2304)
        xs = [px for _, px in pares]
        assert xs == sorted(xs), (cam, pares)


def test_forzar_saltea_todo():
    s = Sectorizador(sector_inicial=0)
    r = s.forzar(90.0, 0.0)
    assert r.cambio and r.angulo_objetivo == 90.0 and r.sector == 4


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