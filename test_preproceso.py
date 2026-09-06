"""
test_preproceso.py

Verifica preproceso.py sin modelo, sin camara y sin imagenes: alcanza con un
frame sintetico. Corre en cualquier maquina.

    python3 test_preproceso.py

Lo que se comprueba, y por que:
  - la grilla y los tensores son IDENTICOS a los de main.py, que es de donde
    salieron: la mudanza no cambio comportamiento;
  - el orden de canales sale en RGB;
  - to_global cierra el circulo en los dos modos. Ese es el invariante que
    importa: una deteccion mal remapeada no rompe nada, simplemente apunta el
    motor a otro lado, y eso no se ve hasta mirar el video.
"""

import numpy as np

import preproceso as P

FRAME_HW = (1296, 2304)
MODEL_HW = (640, 1152)


def _frame_rojo():
    """Frame en BGR con rojo puro: B=0, G=0, R=255."""
    f = np.zeros((*FRAME_HW, 3), np.uint8)
    f[:, :, 2] = 255
    return f


def test_grilla_identica_a_main():
    import main as M
    assert P.search_tiles(FRAME_HW, MODEL_HW) == M.search_tiles(FRAME_HW, MODEL_HW)


def test_tensores_identicos_a_main():
    import main as M
    frame = _frame_rojo()
    for i in range(4):
        a, _ = P.build_search_input(frame, MODEL_HW, i)
        b, _ = M.build_search_input(frame, MODEL_HW, i)
        assert np.array_equal(a, b), f"tile {i}"


def test_canales_a_rgb():
    t, _ = P.build_search_input(_frame_rojo(), MODEL_HW, 0)
    assert tuple(int(v) for v in t[0, 0]) == (255, 0, 0)


def test_shapes():
    frame = _frame_rojo()
    for t, _ in (P.build_track_input(frame, (1200.0, 700.0), MODEL_HW),
                 P.build_search_input(frame, MODEL_HW, 0)):
        assert t.shape == (*MODEL_HW, 3) and t.dtype == np.uint8


def test_to_global_track_devuelve_el_centro():
    _, g = P.build_track_input(_frame_rojo(), (1200.0, 700.0), MODEL_HW)
    gx, gy = g(MODEL_HW[1] / 2, MODEL_HW[0] / 2)
    assert abs(gx - 1200) < 1 and abs(gy - 700) < 1


def test_clampeo_en_el_borde_no_cambia_el_shape():
    t, g = P.build_track_input(_frame_rojo(), (10.0, 10.0), MODEL_HW)
    assert t.shape == (*MODEL_HW, 3)
    assert g(0, 0) == (0, 0)


def test_to_global_search_en_los_cuatro_tiles():
    frame = _frame_rojo()
    for i, (x0, y0, rw, rh) in enumerate(P.search_tiles(FRAME_HW, MODEL_HW)):
        _, g = P.build_search_input(frame, MODEL_HW, i)
        gx, gy = g(MODEL_HW[1] / 2, MODEL_HW[0] / 2)
        assert abs(gx - (x0 + rw / 2)) < 1 and abs(gy - (y0 + rh / 2)) < 1


def test_round_trip_punto_tile_punto():
    frame = _frame_rojo()
    tiles = P.search_tiles(FRAME_HW, MODEL_HW)
    for px, py in [(300, 200), (1500, 300), (400, 900), (1900, 1000)]:
        i = P.tile_para_punto(FRAME_HW, MODEL_HW, px, py)
        x0, y0, rw, rh = tiles[i]
        _, g = P.build_search_input(frame, MODEL_HW, i)
        gx, gy = g((px - x0) * MODEL_HW[1] / rw, (py - y0) * MODEL_HW[0] / rh)
        assert abs(gx - px) < 1 and abs(gy - py) < 1, (px, py)


def test_tile_idx_es_ciclico():
    frame = _frame_rojo()
    a, _ = P.build_search_input(frame, MODEL_HW, 7)
    b, _ = P.build_search_input(frame, MODEL_HW, 3)
    assert np.array_equal(a, b)


def test_no_hay_build_search_full():
    assert not hasattr(P, "build_search_full")


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