"""
bench.py  --  P0: presupuesto de tiempo por etapa.

Todo el diseño del sistema asume que un frame de TRACK entra en 25 ms
(40 fps con una sola inferencia). Nadie lo midió todavía con el .hef
custom y sobre la Pi. Este script lo mide, y de acá sale el número que
después fija el fps objetivo del main final.

Qué mide, con p50 y p95 sobre N repeticiones:

  - preprocesado de TRACK: recorte de 1152x640 a resolución nativa,
    sin resize, más la conversión de canales.
  - preprocesado de SEARCH: recorte del cuadrante, resize a la entrada
    del modelo y conversión de canales.
  - hailo.infer() sola, sobre cada uno de los dos tensores.
  - postprocess.process_candidates() sobre la salida real.

Y reporta los dos totales que importan: un frame de TRACK (una
inferencia) y un frame de SEARCH completo (los cuatro cuadrantes).

POR QUÉ SE MIDE CADA ETAPA POR SEPARADO Y NO EL TOTAL
Si el total se pasa de 25 ms hay que saber a quién recortarle. Si el
costo está en la NPU no hay nada que hacer desde Python y lo que se
mueve es el fps objetivo; si está en el preprocesado, se puede atacar
(evitar la copia del cvtColor, pedirle a la cámara otro formato, etc.).

POR QUÉ SE MIDE CON EL VIDEO Y NO CON RUIDO
El costo de process_candidates() depende del contenido: cuántas celdas
superan CONF_CANDIDATO. Con ruido sintético no supera ninguna y el
número da optimista. Por eso --sintetico existe pero avisa.

No toca motor, encoder ni GoPro.

    python3 bench.py --video partido.mp4 --n 200
    python3 bench.py --imagenes imagenes/Soleado --n 200
"""

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import config
import postprocess

# Se importan de main.py a propósito, en vez de copiarlas: son las mismas
# funciones que corre el sistema de verdad, y medir una copia no mide nada.
# Cuando P2 las mude a preproceso.py, acá se cambia el import y listo.
from main import build_search_input, build_track_input, get_model_hw, search_tiles


# Presupuesto de un frame de TRACK. Sale de 40 fps = 25 ms por frame.
# Si config.py todavía no lo tiene, se usa este valor y se avisa.
PRESUPUESTO_TRACK_MS = getattr(config, "PRESUPUESTO_TRACK_MS", 25.0)

# Cuántos candidatos pide el tracker por frame. Es el default de
# postprocess.process_candidates(); se fija acá para medir lo mismo que
# después va a correr P3.
TOPK_CANDIDATOS = 8

# Frames de calentamiento que se descartan: la primera inferencia paga
# la carga de los buffers de HailoRT y ensucia el p95.
CALENTAMIENTO = 10

# Resolución nativa esperada de las cámaras, leída de config para no
# duplicar el número. Solo se usa para avisar si el video de prueba no la
# tiene: la grilla de SEARCH depende del tamaño del frame, así que medir
# sobre otra resolución mide otra cosa.
FRAME_NATIVO_ESPERADO = (config.CAM_ALTO, config.CAM_ANCHO)  # (alto, ancho)


class _EstadoFalso:
    """
    Lo mínimo que build_track_input() le pide al estado: dónde centrar el
    recorte. No se importa TrackerState para no arrastrar la máquina de
    estados vieja, que en P3 se reemplaza por tracker.py.
    """

    def __init__(self, cx: float, cy: float):
        self._c = (cx, cy)

    def crop_center(self) -> tuple[float, float]:
        return self._c


def percentiles(muestras: list[float]) -> tuple[float, float]:
    """p50 y p95 en ms. Se calculan sobre la lista ya en milisegundos."""
    if not muestras:
        return float("nan"), float("nan")
    ordenadas = sorted(muestras)
    p50 = statistics.median(ordenadas)
    p95 = float(np.percentile(ordenadas, 95))
    return p50, p95


def leer_frames(cap, n: int, log=print):
    """
    Generador de n frames del video. Si el archivo tiene menos, vuelve al
    principio: lo que se mide es el costo de cómputo, no el contenido, y
    repetir frames no lo altera.
    """
    import cv2

    entregados = 0
    vueltas = 0
    while entregados < n:
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            vueltas += 1
            if vueltas > 50:
                raise RuntimeError("El video no entrega frames utilizables.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        entregados += 1
        yield frame
    if vueltas:
        log(f"[info] el video se reinició {vueltas} vez/veces para llegar a {n}")


def leer_carpeta(carpeta: str, n: int, log=print):
    """
    Generador de n frames leídos de una carpeta de imágenes.

    Sirve igual que el video mientras las imágenes sean frames NATIVOS
    completos: si son recortes, la grilla de SEARCH se arma sobre otro
    tamaño y los tiempos de ese modo dejan de ser comparables.

    Las imágenes se recorren en orden alfabético y se repiten en ciclo si
    hay menos que n. Se leen todas de antemano a memoria: la lectura del
    archivo NO tiene que entrar en la medición, que es de cómputo.
    """
    import cv2

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    rutas = sorted(
        p for p in Path(carpeta).iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )
    if not rutas:
        raise SystemExit(f"No encontré imágenes en {carpeta}")

    cargadas = []
    tamanos = set()
    for p in rutas:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            log(f"[warn] no pude leer {p.name}, se saltea")
            continue
        cargadas.append(img)
        tamanos.add(img.shape[:2])

    if not cargadas:
        raise SystemExit(f"Ninguna imagen de {carpeta} se pudo leer.")

    log(f"[init] {len(cargadas)} imágenes cargadas de {carpeta}")
    if len(tamanos) > 1:
        log(f"[warn] las imágenes no tienen todas el mismo tamaño: "
            f"{sorted(tamanos)}. La grilla de SEARCH se rearma por tamaño y "
            f"el p95 mezcla poblaciones distintas.")
    if len(cargadas) < n:
        log(f"[info] hay {len(cargadas)} imágenes para {n} repeticiones: "
            f"se repiten en ciclo")

    for i in range(n):
        yield cargadas[i % len(cargadas)]


def frames_sinteticos(n: int, hw: tuple[int, int]):
    """Frames de ruido, para medir sin video. Ver el aviso del encabezado."""
    alto, ancho = hw
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, size=(alto, ancho, 3), dtype=np.uint8)
    for _ in range(n):
        yield base


def correr(hailo, frames, n: int, log=print) -> dict:
    """
    Corre el banco y devuelve las muestras crudas por etapa, en ms.

    Por cada frame se hace exactamente lo que haría el sistema en cada
    modo: un recorte de TRACK con su inferencia y su decode, y un
    cuadrante de SEARCH con lo mismo. Los cuadrantes se van rotando para
    que el costo del resize no quede atado a una sola región.
    """
    model_hw = get_model_hw(hailo)
    log(f"[init] entrada del modelo: {model_hw[0]}x{model_hw[1]} (alto x ancho)")
    # El costo de process_candidates() es proporcional a cuántas celdas
    # superan este umbral, así que el número tiene que quedar en el informe:
    # un p95 de decode medido con otro CONF_CANDIDATO no es comparable.
    log(f"[init] CONF_CANDIDATO = {getattr(config, 'CONF_CANDIDATO', 0.20)}, "
        f"topk = {TOPK_CANDIDATOS}")

    etapas = {
        "pre_track": [],
        "infer_track": [],
        "post_track": [],
        "pre_search": [],
        "infer_search": [],
        "post_search": [],
    }
    candidatos_track = []
    candidatos_search = []
    primera = True
    idx = 0
    frame_hw = FRAME_NATIVO_ESPERADO

    for frame in frames:
        alto, ancho = frame.shape[:2]

        if primera:
            frame_hw = (alto, ancho)
            log(f"[init] frame de entrada: {ancho}x{alto}")
            if (alto, ancho) != FRAME_NATIVO_ESPERADO:
                log(
                    f"[warn] el frame no es {FRAME_NATIVO_ESPERADO[1]}x"
                    f"{FRAME_NATIVO_ESPERADO[0]}: los tiempos de SEARCH no son "
                    f"comparables con los de la cámara real"
                )
            tiles = search_tiles((alto, ancho), model_hw)
            escala = model_hw[1] / tiles[0][2]
            log(
                f"[init] SEARCH: {len(tiles)} cuadrantes de {tiles[0][2]}x"
                f"{tiles[0][3]} nativos -> escala {escala:.4f}"
            )
            primera = False

        # El recorte de TRACK se centra en el medio del frame: es el caso
        # típico y evita el clampeo contra los bordes, que sería más barato.
        estado = _EstadoFalso(ancho / 2.0, alto / 2.0)

        medir = idx >= CALENTAMIENTO

        # ---- TRACK -------------------------------------------------------
        t0 = time.perf_counter()
        tensor_t, _ = build_track_input(frame, estado, model_hw)
        t1 = time.perf_counter()
        salida_t = hailo.infer(tensor_t)
        t2 = time.perf_counter()
        dets_t = postprocess.process_candidates(salida_t, model_hw, topk=TOPK_CANDIDATOS)
        t3 = time.perf_counter()

        # ---- SEARCH ------------------------------------------------------
        t4 = time.perf_counter()
        tensor_s, _ = build_search_input(frame, model_hw, idx)
        t5 = time.perf_counter()
        salida_s = hailo.infer(tensor_s)
        t6 = time.perf_counter()
        dets_s = postprocess.process_candidates(salida_s, model_hw, topk=TOPK_CANDIDATOS)
        t7 = time.perf_counter()

        if medir:
            etapas["pre_track"].append((t1 - t0) * 1e3)
            etapas["infer_track"].append((t2 - t1) * 1e3)
            etapas["post_track"].append((t3 - t2) * 1e3)
            etapas["pre_search"].append((t5 - t4) * 1e3)
            etapas["infer_search"].append((t6 - t5) * 1e3)
            etapas["post_search"].append((t7 - t6) * 1e3)
            candidatos_track.append(len(dets_t))
            candidatos_search.append(len(dets_s))

        idx += 1
        if idx % 50 == 0:
            log(f"[bench] {idx}/{n + CALENTAMIENTO}")

    return {
        "etapas": etapas,
        "candidatos_track_prom": (
            sum(candidatos_track) / len(candidatos_track) if candidatos_track else 0.0
        ),
        "candidatos_search_prom": (
            sum(candidatos_search) / len(candidatos_search) if candidatos_search else 0.0
        ),
        "model_hw": list(model_hw),
        "frame_hw": list(frame_hw),
    }


def informe(resultado: dict, n_tiles: int, log=print) -> dict:
    """Imprime la tabla y devuelve el resumen en un dict serializable."""
    etapas = resultado["etapas"]
    resumen = {}

    log("")
    log(f"{'etapa':<28}{'p50 (ms)':>12}{'p95 (ms)':>12}")
    log("-" * 52)
    for nombre, muestras in etapas.items():
        p50, p95 = percentiles(muestras)
        resumen[nombre] = {"p50_ms": round(p50, 3), "p95_ms": round(p95, 3),
                           "n": len(muestras)}
        log(f"{nombre:<28}{p50:>12.3f}{p95:>12.3f}")

    # Los totales se suman por frame y después se percentila, no se suman
    # los p95: el p95 de la suma no es la suma de los p95, y sumarlos daría
    # un techo pesimista que no ocurre en ningún frame real.
    tot_track = [
        a + b + c
        for a, b, c in zip(etapas["pre_track"], etapas["infer_track"],
                           etapas["post_track"])
    ]
    tot_search_1 = [
        a + b + c
        for a, b, c in zip(etapas["pre_search"], etapas["infer_search"],
                           etapas["post_search"])
    ]
    tot_search = [v * n_tiles for v in tot_search_1]

    p50_t, p95_t = percentiles(tot_track)
    p50_s, p95_s = percentiles(tot_search)
    resumen["total_track"] = {"p50_ms": round(p50_t, 3), "p95_ms": round(p95_t, 3)}
    resumen["total_search"] = {"p50_ms": round(p50_s, 3), "p95_ms": round(p95_s, 3),
                               "cuadrantes": n_tiles}

    log("-" * 52)
    log(f"{'TOTAL frame TRACK':<28}{p50_t:>12.3f}{p95_t:>12.3f}")
    log(f"{'TOTAL frame SEARCH x' + str(n_tiles):<28}{p50_s:>12.3f}{p95_s:>12.3f}")
    log("")
    log(f"candidatos por frame (prom): TRACK "
        f"{resultado['candidatos_track_prom']:.2f} | SEARCH "
        f"{resultado['candidatos_search_prom']:.2f}")

    fps_track = 1000.0 / p95_t if p95_t > 0 else float("inf")
    resumen["fps_track_p95"] = round(fps_track, 2)
    resumen["presupuesto_track_ms"] = PRESUPUESTO_TRACK_MS

    log("")
    if p95_t <= PRESUPUESTO_TRACK_MS:
        resumen["cumple"] = True
        log(f"[OK] TRACK p95 = {p95_t:.2f} ms <= {PRESUPUESTO_TRACK_MS:.1f} ms. "
            f"Techo sostenible: {fps_track:.1f} fps. P0 cumple.")
    else:
        resumen["cumple"] = False
        log(f"[FALLA] TRACK p95 = {p95_t:.2f} ms > {PRESUPUESTO_TRACK_MS:.1f} ms. "
            f"El fps objetivo real es {fps_track:.1f}, no 40. Ajustar acá, "
            f"no en P7.")
        peor = max(("pre_track", "infer_track", "post_track"),
                   key=lambda k: percentiles(etapas[k])[1])
        log(f"[FALLA] la etapa más cara es {peor}.")

    return resumen


def escribir_estado(path: str, resumen: dict, contexto: dict, log=print) -> None:
    """Agrega un bloque a ESTADO.md con los números medidos (regla 2)."""
    lineas = [
        "",
        f"## P0 — presupuesto de tiempo ({datetime.now():%Y-%m-%d %H:%M})",
        "",
        f"- hef: `{contexto['hef']}`",
        f"- fuente: `{contexto['fuente']}`  ({contexto['n']} frames medidos, "
        f"{CALENTAMIENTO} de calentamiento descartados)",
        f"- entrada del modelo: {resumen['pre_track']['n']} muestras por etapa",
        "",
        "| etapa | p50 (ms) | p95 (ms) |",
        "|---|---|---|",
    ]
    for k, v in resumen.items():
        if isinstance(v, dict) and "p50_ms" in v:
            lineas.append(f"| {k} | {v['p50_ms']:.2f} | {v['p95_ms']:.2f} |")
    lineas += [
        "",
        f"**Resultado:** TRACK p95 = {resumen['total_track']['p95_ms']:.2f} ms "
        f"contra un presupuesto de {resumen['presupuesto_track_ms']:.1f} ms -> "
        f"{'CUMPLE' if resumen['cumple'] else 'NO CUMPLE'}. "
        f"Techo sostenible {resumen['fps_track_p95']:.1f} fps.",
        "",
    ]
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lineas))
    log(f"[ok] {path} actualizado")


def main() -> None:
    import cv2

    ap = argparse.ArgumentParser(description="P0: presupuesto de tiempo por etapa")
    ap.add_argument("--video", default=getattr(config, "VIDEO_PATH", None),
                    help="video de entrada a resolución nativa")
    ap.add_argument("--imagenes", default=None,
                    help="carpeta con frames nativos, en vez de un video")
    ap.add_argument("--hef", default=config.HEF_PATH, help="ruta del .hef")
    ap.add_argument("--n", type=int, default=200, help="frames medidos")
    ap.add_argument("--sintetico", action="store_true",
                    help="sin video: ruido a resolución nativa (postprocess da optimista)")
    ap.add_argument("--json", default=None, help="guardar el resumen en un JSON")
    ap.add_argument("--estado", default="ESTADO.md",
                    help="archivo de estado a actualizar ('' para no tocarlo)")
    args = ap.parse_args()

    if not hasattr(config, "PRESUPUESTO_TRACK_MS"):
        print(f"[warn] config.PRESUPUESTO_TRACK_MS no existe; se usa "
              f"{PRESUPUESTO_TRACK_MS} ms. Agregalo a config.py.")

    from hailo_inference import HailoInference

    cap = None
    total = args.n + CALENTAMIENTO

    if args.sintetico:
        print("[warn] modo sintético: el costo de process_candidates() sale bajo "
              "porque no hay nada que detectar. Sirve para el preprocesado y la "
              "NPU, no para el decode.")
        fuente = f"sintético {FRAME_NATIVO_ESPERADO[1]}x{FRAME_NATIVO_ESPERADO[0]}"
        frames = frames_sinteticos(total, FRAME_NATIVO_ESPERADO)
    elif args.imagenes:
        fuente = f"carpeta {args.imagenes}"
        frames = leer_carpeta(args.imagenes, total)
    else:
        if not args.video:
            raise SystemExit("Falta --video (o config.VIDEO_PATH). "
                             "Si no tenés uno a mano, usá --sintetico.")
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"No se pudo abrir el video: {args.video}")
        fuente = args.video
        frames = leer_frames(cap, total)

    print(f"[init] hef: {args.hef}")

    try:
        with HailoInference(args.hef) as hailo:
            print(hailo.describe())
            resultado = correr(hailo, frames, args.n)
    finally:
        if cap is not None:
            cap.release()

    n_tiles = len(search_tiles(tuple(resultado["frame_hw"]),
                               tuple(resultado["model_hw"])))
    resumen = informe(resultado, n_tiles)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"fuente": fuente, "hef": args.hef, "n": args.n,
                       "resumen": resumen}, f, indent=2)
        print(f"[ok] resumen en {args.json}")

    if args.estado:
        escribir_estado(args.estado, resumen,
                        {"hef": args.hef, "fuente": fuente, "n": args.n})


if __name__ == "__main__":
    main()
