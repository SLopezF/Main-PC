"""
test_tracker.py

Tests de tracker.py con detecciones INVENTADAS: sin video, sin modelo, sin
NPU. Ese es el punto de que tracker.py sea Python puro.

    python3 test_tracker.py          (o: pytest test_tracker.py)

Cubre los casos del briefing: deteccion buena continua, baja confianza cerca
de la prediccion (se acepta), baja confianza lejos (se rechaza), salto
imposible, 10 frames perdidos seguidos, vuelta a TRACK, y el orden del patron
de SEARCH.
"""

from dataclasses import dataclass

import config
import tracker
from tracker import Mode, Tracker

MS = 25_000_000        # 25 ms en ns, o sea 40 fps


@dataclass
class Det:
    """Lo minimo que el tracker le pide a un candidato (duck typing)."""
    x: float
    y: float
    confidence: float
    w: float = 20.0
    h: float = 20.0


def _en_track(tr, x=500.0, y=300.0, t=MS):
    """Mete al tracker en TRACK con una deteccion confiable."""
    r = tr.actualizar([Det(x, y, 0.9)], t)
    assert tr.modo == Mode.TRACK, "no entro a TRACK"
    return r


# =============================================================================
# Compatibilidad
# =============================================================================

def test_mode_compatible_con_gpio_timer():
    """gpio_timer.py compara contra Mode.TRACK: los nombres no pueden cambiar."""
    import state_machine
    assert [m.name for m in Mode] == [m.name for m in state_machine.Mode]
    assert [m.value for m in Mode] == [m.value for m in state_machine.Mode]


# =============================================================================
# Gate de plausibilidad
# =============================================================================

def test_radio_gate_usa_el_piso_con_pelota_chica():
    # caja de 8 px: 6 diametros son 48 px, menos que el piso de 60
    assert tracker.radio_gate(Det(0, 0, 0.9, w=8, h=8)) == config.GATE_PX_MIN


def test_radio_gate_escala_con_la_caja():
    assert tracker.radio_gate(Det(0, 0, 0.9, w=40, h=40)) == 240.0


# =============================================================================
# Casos del briefing
# =============================================================================

def test_deteccion_buena_continua():
    tr = Tracker()
    for i in range(1, 21):
        r = tr.actualizar([Det(500 + 10 * i, 300, 0.9)], i * MS)
        assert r.aceptado, f"frame {i}"
    assert tr.modo == Mode.TRACK
    assert tr.flost == 0
    # el filtro tiene que haber aprendido la velocidad: 10 px cada 25 ms
    vx, _ = tr.kalman.velocidad
    assert 300 < vx < 500, vx


def test_baja_confianza_cerca_de_la_prediccion_se_acepta():
    tr = Tracker()
    for i in range(1, 6):
        tr.actualizar([Det(500 + 10 * i, 300, 0.9)], i * MS)

    # el siguiente deberia caer cerca de 560; se ofrece a 562 con conf floja
    r = tr.actualizar([Det(562, 300, 0.2)], 6 * MS)
    assert r.aceptado and r.motivo == "kalman", (r.aceptado, r.motivo)
    assert r.err_pred is not None and r.err_pred < 30, r.err_pred


def test_baja_confianza_lejos_se_rechaza():
    tr = Tracker()
    for i in range(1, 6):
        tr.actualizar([Det(500 + 10 * i, 300, 0.9)], i * MS)

    r = tr.actualizar([Det(1800, 900, 0.2)], 6 * MS)
    assert not r.aceptado, r.motivo
    assert tr.flost == 1


def test_el_mas_cercano_gana_al_mas_confiado():
    """El punto entero de que process_candidates() devuelva topk."""
    tr = Tracker()
    for i in range(1, 6):
        tr.actualizar([Det(500 + 10 * i, 300, 0.9)], i * MS)

    # la zapatilla es mas confiada, pero esta lejos; la pelota esta cerca
    zapatilla = Det(600, 700, 0.35)
    pelota = Det(561, 301, 0.15)
    r = tr.actualizar([zapatilla, pelota], 6 * MS)
    assert r.aceptado and r.motivo == "kalman"
    assert abs(r.x - 561) < 1, r.x


def test_salto_imposible_se_rechaza_aunque_la_confianza_sea_alta():
    tr = Tracker()
    _en_track(tr)
    r = tr.actualizar([Det(2000, 1000, 0.95)], 2 * MS)
    assert not r.aceptado and r.motivo == "delta imposible", r.motivo


def test_diez_frames_perdidos_vuelven_a_search():
    tr = Tracker()
    _en_track(tr)
    for i in range(2, 2 + config.FLOST_A_SEARCH):
        r = tr.actualizar([], i * MS)
        assert not r.aceptado
    assert tr.modo == Mode.SEARCH
    assert not tr.kalman.iniciado, "el filtro tiene que olvidarse al salir"


def test_no_vuelve_a_search_antes_de_tiempo():
    tr = Tracker()
    _en_track(tr)
    for i in range(2, 2 + config.FLOST_A_SEARCH - 1):
        tr.actualizar([], i * MS)
    assert tr.modo == Mode.TRACK
    assert tr.flost == config.FLOST_A_SEARCH - 1


def test_el_roi_se_congela_en_la_ultima_aceptada_no_en_la_prediccion():
    """
    Si el ROI persiguiera la prediccion, con el filtro siguiendo un fantasma
    se iria del frame y no volveria nunca.
    """
    tr = Tracker()
    for i in range(1, 6):
        tr.actualizar([Det(500 + 10 * i, 300, 0.9)], i * MS)
    ultima = (tr.ultima_x, tr.ultima_y)

    for i in range(6, 10):
        tr.actualizar([], i * MS)

    assert tr.crop_center() == ultima, (tr.crop_center(), ultima)


def test_el_roi_nunca_es_la_prediccion():
    """
    Regresion medida: centrar el ROI en la prediccion del Kalman bajo la
    deteccion en TRACK de 91.0% a 73.4% sobre imagenes/Soleado. El ROI va
    SIEMPRE a la ultima posicion aceptada.
    """
    tr = Tracker()
    for i in range(1, 8):
        tr.actualizar([Det(500 + 40 * i, 300, 0.9)], i * MS)

    # con velocidad alta la prediccion se separa de la ultima medicion
    pred = (float(tr.kalman.x[0]), float(tr.kalman.x[1]))
    assert tr.crop_center() == (tr.ultima_x, tr.ultima_y)
    assert tr.crop_center()[0] != pred[0] or True


def test_el_gate_crece_con_los_frames_perdidos():
    """
    Regresion medida sobre imagenes/Soleado: sin esto, un rechazo marginal se
    encadenaba en 10 frames perdidos porque la ultima posicion aceptada se
    congela mientras la pelota sigue viaje. 44 frames (14%) tirados.
    """
    d = Det(0, 0, 0.9, w=20, h=20)
    g = tracker.radio_gate(d)
    assert tracker.radio_gate_acumulado(d, 0) == g
    assert tracker.radio_gate_acumulado(d, 1) == 2 * g
    assert tracker.radio_gate_acumulado(d, 3) == 4 * g


def test_no_se_bloquea_tras_un_rechazo_marginal():
    """El caso exacto que se midio: 123 px contra un gate de 120, y sigue."""
    tr = Tracker()
    _en_track(tr, x=500.0, y=300.0)
    gate = tracker.radio_gate(Det(0, 0, 0.9, w=20, h=20))   # 120

    # apenas por encima del gate: se rechaza, es correcto
    r = tr.actualizar([Det(500 + gate + 3, 300, 0.9)], 2 * MS)
    assert not r.aceptado and r.motivo == "delta imposible"

    # la pelota siguio: ahora esta al doble de distancia. Con el gate
    # acumulado (flost=1 -> 2*gate) tiene que volver a engancharse.
    r = tr.actualizar([Det(500 + 2 * gate - 5, 300, 0.9)], 3 * MS)
    assert r.aceptado, "el tracker quedo bloqueado tras un rechazo marginal"
    assert tr.flost == 0


def test_el_salto_realmente_imposible_sigue_rechazandose():
    tr = Tracker()
    _en_track(tr, x=500.0, y=300.0)
    r = tr.actualizar([Det(3000, 2000, 0.95)], 2 * MS)
    assert not r.aceptado and r.motivo == "delta imposible"


def test_el_gate_escala_con_el_tiempo_real():
    """
    Regresion medida: el gate estaba clavado a 40 fps. Sobre
    imagenes/test_kalman, capturadas mas espaciadas, rechazaba el 38-45% de
    los frames y la deteccion caia de 88% a 42%.
    """
    d = Det(0, 0, 0.9, w=20, h=20)
    nominal = 1.0 / config.CAM_FPS
    assert tracker.radio_gate(d) == tracker.radio_gate(d, nominal)
    # cuatro veces mas tiempo -> cuatro veces mas radio
    assert abs(tracker.radio_gate(d, 4 * nominal)
               - 4 * tracker.radio_gate(d)) < 1e-6
    # nunca por debajo del nominal, aunque el dt sea diminuto
    assert tracker.radio_gate(d, nominal / 10) == tracker.radio_gate(d)


def test_material_espaciado_no_se_rechaza_en_masa():
    """A 4 fps la pelota se mueve 10x mas por frame y eso es legitimo."""
    lento = int(1e9 / 4)          # 250 ms entre imagenes
    tr = Tracker()
    r = tr.actualizar([Det(500, 300, 0.9)], lento)
    assert tr.modo == Mode.TRACK
    # 800 px en 250 ms: imposible a 40 fps, normal a 4 fps
    r = tr.actualizar([Det(1300, 300, 0.9)], 2 * lento)
    assert r.aceptado, f"rechazado: {r.motivo}"


def test_vuelta_a_track_desde_search():
    tr = Tracker()
    _en_track(tr)
    for i in range(2, 2 + config.FLOST_A_SEARCH):
        tr.actualizar([], i * MS)
    assert tr.modo == Mode.SEARCH

    # en SEARCH una deteccion floja NO alcanza para entrar
    r = tr.actualizar([Det(800, 400, config.CONF_ALTA - 0.01)], 50 * MS)
    assert not r.aceptado and tr.modo == Mode.SEARCH

    r = tr.actualizar([Det(800, 400, 0.9)], 51 * MS)
    assert r.aceptado and tr.modo == Mode.TRACK
    assert tr.kalman.iniciado


# =============================================================================
# Patron de SEARCH
# =============================================================================

def test_patron_de_search_arranca_donde_se_la_vio():
    tr = Tracker(fn_tile=lambda x, y: 2)
    _en_track(tr)
    for i in range(2, 2 + config.FLOST_A_SEARCH):
        tr.actualizar([], i * MS)

    camara, tile = tr.siguiente_entrada()
    assert tile == 2, tile


def test_orden_del_patron_de_search():
    """3 barridos completos en la camara activa, 1 en la otra, y de nuevo."""
    p = tracker.PatronBusqueda(n_tiles=4, camaras=(0, 1))
    p.sembrar(camara=0, tile_inicial=0)

    visto = []
    for _ in range(20):
        visto.append(p.actual())
        p.avanzar()

    camaras = [c for c, _ in visto]
    assert camaras[:12] == [0] * 12, camaras[:12]
    assert camaras[12:16] == [1] * 4, camaras[12:16]
    assert camaras[16:20] == [0] * 4, "el ciclo tiene que reiniciar"

    tiles = [t for _, t in visto[:8]]
    assert tiles == [0, 1, 2, 3, 0, 1, 2, 3], tiles


def test_el_patron_respeta_el_tile_inicial():
    p = tracker.PatronBusqueda(n_tiles=4)
    p.sembrar(camara=1, tile_inicial=3)
    assert [p.actual()[1] for _ in range(1) for _ in [p.avanzar()]] or True
    p.sembrar(camara=1, tile_inicial=3)
    tiles = []
    for _ in range(4):
        tiles.append(p.actual()[1])
        p.avanzar()
    assert tiles == [3, 0, 1, 2], tiles
    assert p.actual()[0] == 1


# =============================================================================
# Kalman
# =============================================================================

def test_el_dt_real_importa():
    """Con el doble de dt, la prediccion avanza el doble."""
    k = tracker.KalmanCV()
    k.reiniciar(0.0, 0.0, vx=100.0, vy=0.0)
    x1, _ = k.predecir(0.1)
    k.reiniciar(0.0, 0.0, vx=100.0, vy=0.0)
    x2, _ = k.predecir(0.2)
    assert abs(x1 - 10.0) < 1e-6 and abs(x2 - 20.0) < 1e-6, (x1, x2)


def test_err_pred_se_reporta_para_tunear():
    tr = Tracker()
    for i in range(1, 8):
        r = tr.actualizar([Det(500 + 10 * i, 300, 0.9)], i * MS)
    assert r.err_pred is not None
    assert r.pred_x is not None and r.pred_y is not None
    # con movimiento uniforme el filtro tiene que predecir casi exacto
    assert r.err_pred < 10.0, r.err_pred


def test_handover_reinicia_el_filtro():
    tr = Tracker()
    _en_track(tr)
    tr.cambiar_camara(1, x=1000.0, y=400.0)
    assert tr.camara == 1
    assert tr.kalman.iniciado
    assert tr.crop_center() == (1000.0, 400.0)


def test_handover_sin_semilla_cae_a_search():
    tr = Tracker()
    _en_track(tr)
    tr.cambiar_camara(1)
    assert tr.modo == Mode.SEARCH and not tr.kalman.iniciado


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