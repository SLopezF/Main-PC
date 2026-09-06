"""
replay.py

Banco de pruebas offline. Corre el pipeline de deteccion sobre material ya
grabado (un .mp4 o una carpeta de imagenes), sin motor, sin encoder y sin
GoPro, y escupe tres cosas:

    CSV por frame      una fila por frame, con las mismas columnas que va a
                       escribir main_final.py (regla 8: el analisis de la
                       tesis es un solo script para los dos)
    mp4 anotado        el frame reducido con la caja, el modo y el tile
    resumen impreso    los numeros que se comparan entre versiones

PARA QUE EXISTE
Para no volver a la cancha cada vez que se cambia un umbral. El resumen es
la metrica que se compara entre versiones del algoritmo, y es tambien la
tabla de resultados de la tesis.

EN PARTICULAR: ESTA CORRIDA ES LA LINEA DE BASE DE P3.
Aca todavia NO hay Kalman ni sectores. La logica de aceptacion es la de
state_machine.py: histeresis de confianza a secas, sin prediccion. El
criterio de aceptacion de tracker.py (P3) es que el "% de tiempo en TRACK"
suba respecto del numero que imprime esto. Correlo y anota el resultado en
ESTADO.md ANTES de escribir el tracker, o no vas a tener contra que comparar.

COLUMNAS DEL CSV
    n              indice de frame
    ts             timestamp en segundos desde el primer frame
    modo           SEARCH / TRACK, el modo con el que se proceso ESTE frame
    camara         indice de camara (0/1); con una sola fuente, siempre el mismo
    tile           cuadrante de SEARCH usado, -1 en TRACK
    conf           confianza de la deteccion aceptada
    x, y           centro en pixeles del frame NATIVO
    w, h           tamano de la caja en pixeles nativos
    angulo         angulo del mundo (0..180) segun geometria.pixel_a_angulo
    sector         VACIA en P2: la sectorizacion llega en P4
    flost          frames perdidos consecutivos en TRACK
    aceptado_por   por que se acepto: "confianza" o vacio

Las columnas que P2 no llena se dejan igual, vacias, para no tener que
cambiar el script de analisis cuando P3 y P4 empiecen a llenarlas.

EL RELOJ
Con carpeta se usa reloj SINTETICO: el tiempo avanza 1/fps por imagen, sin
importar lo que tarde el modelo. Es lo que hace la corrida reproducible: la
misma carpeta da el mismo resultado en la PC y en la Pi, con GPU o sin ella.
Sin eso, comparar dos versiones del algoritmo mide la maquina, no el
algoritmo.

USO
    python3 replay.py --carpeta imagenes/cam0
    python3 replay.py --video grabacion.mkv --salida-csv r.csv --salida-mp4 r.mp4
    python3 replay.py --carpeta imagenes --sin-mp4 --limite 300
"""

import csv
import os
import time

import numpy as np

import config
import geometria
import postprocess
import preproceso
from inferencia import abrir as abrir_inferencia
from tracker import Mode, Tracker

# Todas las salidas van a una sola carpeta, que .gitignore ignora entera. Sin
# esto los csv y los mp4 se desparraman por la raiz del repo y terminan
# commiteados por accidente.
DIR_SALIDAS = "salidas"

# pred_x / pred_y / err_pred son el residuo del Kalman: la distancia entre lo
# que el filtro predijo ANTES de ver el frame y donde aparecio la deteccion.
# Es la señal con la que se tunean KALMAN_SIGMA_ACEL y KALMAN_SIGMA_MEDICION.
class _Cand:
    """
    Candidato en pixeles del frame NATIVO. El tracker solo pide .x .y .w .h y
    .confidence; se guarda .original para poder dibujar la caja despues, que
    sigue en coordenadas del tensor.
    """
    __slots__ = ("x", "y", "w", "h", "confidence", "original")

    def __init__(self, x, y, w, h, confidence, original):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.confidence, self.original = confidence, original


COLUMNAS = ["n", "ts", "modo", "camara", "tile", "conf", "x", "y", "w", "h",
            "angulo", "sector", "flost", "aceptado_por",
            "pred_x", "pred_y", "err_pred", "vx", "vy", "motivo", "dist_ult"]


# =============================================================================
# Fuente de video
# =============================================================================

class FuenteVideo:
    """
    Un .mp4/.mkv con el mismo contrato que fuente.FuenteCarpeta: read()
    devuelve (frame, info) con ts_ns, dt_ms, perdidos y seq.

    Vive aca y no en fuente.py a proposito: fuente.py modela las dos fuentes
    del sistema REAL (camara y carpeta). Un archivo de video es una comodidad
    del banco de pruebas, no una entrada del producto.

    El reloj es sintetico, derivado del fps del contenedor: mismo motivo que
    en FuenteCarpeta, la corrida tiene que ser reproducible.
    """

    def __init__(self, ruta: str, fps: float | None = None, indice: int = 0):
        import cv2

        if not os.path.exists(ruta):
            raise FileNotFoundError(f"no encuentro el video '{ruta}'")

        self._cap = cv2.VideoCapture(ruta)
        if not self._cap.isOpened():
            raise RuntimeError(f"no pude abrir el video '{ruta}'")

        self.ruta = ruta
        self.indice = int(indice)

        if fps is None:
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0 or fps > 1000:
                fps = float(getattr(config, "NATIVE_FPS", 30.0))
                print(f"[replay] el video no declara fps valido, uso {fps:g}")
        self.fps = float(fps)
        self.periodo_ms = 1000.0 / self.fps

        self.total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.total_perdidos = 0
        self._seq = 0
        self._ts_ns = 0

        print(f"[fuente] video '{ruta}': "
              f"{self.total or '?'} frame(s) a {self.fps:g} fps")

    def read(self):
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise StopIteration(f"se acabo el video '{self.ruta}'")

        dt_ms = 0.0 if self._seq == 0 else self.periodo_ms
        self._ts_ns += int(dt_ms * 1e6)
        self._seq += 1

        return frame, {
            "ts_ns": self._ts_ns,
            "dt_ms": dt_ms,
            "perdidos": 0,
            "seq": self._seq,
            "archivo": self.ruta,
            "agotada": False,
            "camara": self.indice,
        }

    def close(self):
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


def abrir_fuente_replay(video: str | None, carpeta: str | None,
                        camara: int, fps: float | None):
    if video:
        return FuenteVideo(video, fps=fps, indice=camara)

    import fuente
    return fuente.FuenteCarpeta(ruta=carpeta, indice=camara, fps=fps,
                                bucle=False, reloj="sintetico")


# =============================================================================
# Procesamiento de un frame
# =============================================================================

def procesar_frame(frame, info, hailo, model_hw, tr):
    """
    Un frame de punta a punta: arma el tensor segun el modo que pide el
    tracker, infiere, decodifica los candidatos y deja que el tracker decida.

    Esta funcion es la que despues comparte main_final.py.

    Se llama process_candidates() y no process(): el tracker necesita VARIOS
    candidatos para poder elegir el mas cercano a la prediccion cuando la
    confianza es baja. Con uno solo, esa mitad del algoritmo no existe.
    """
    t0 = time.perf_counter()

    _, tile = tr.siguiente_entrada()

    if tr.modo == Mode.SEARCH:
        tensor, to_global = preproceso.build_search_input(
            frame, model_hw, tile)
    else:
        tensor, to_global = preproceso.build_track_input(
            frame, tr.crop_center(), model_hw)

    crudos = postprocess.process_candidates(
        hailo.infer(tensor), model_hw,
        umbral=getattr(config, "CONF_CANDIDATO", 0.05), topk=8)

    # Los candidatos llegan en coordenadas del TENSOR; el tracker trabaja en
    # pixeles del frame nativo. Se remapean antes de entregarlos.
    candidatos = []
    for d in crudos:
        gx, gy = to_global(d.x, d.y)
        candidatos.append(_Cand(gx, gy, d.w, d.h, d.confidence, d))

    r = tr.actualizar(candidatos, info["ts_ns"], info.get("camara", 0))
    ms = (time.perf_counter() - t0) * 1000.0

    ancho = frame.shape[1]
    angulo = (geometria.pixel_a_angulo(r.x, ancho, r.camara)
              if r.x is not None else None)

    det_dibujo = None
    if r.aceptado:
        for c in candidatos:
            if c.x == r.x and c.y == r.y:
                det_dibujo = c.original
                break

    fila = {
        "n": info["seq"] - 1,
        "ts": info["ts_ns"] / 1e9,
        "modo": r.modo.name,
        "camara": r.camara,
        "tile": r.tile,
        "conf": f"{r.conf:.4f}" if r.conf is not None else "",
        "x": f"{r.x:.1f}" if r.x is not None else "",
        "y": f"{r.y:.1f}" if r.y is not None else "",
        "w": f"{r.w:.1f}" if r.w is not None else "",
        "h": f"{r.h:.1f}" if r.h is not None else "",
        "angulo": f"{angulo:.2f}" if angulo is not None else "",
        "sector": "",
        "flost": r.flost,
        "aceptado_por": r.motivo if r.aceptado else "",
        "pred_x": f"{r.pred_x:.1f}" if r.pred_x is not None else "",
        "pred_y": f"{r.pred_y:.1f}" if r.pred_y is not None else "",
        "err_pred": f"{r.err_pred:.1f}" if r.err_pred is not None else "",
        "vx": f"{r.vx:.0f}" if r.vx is not None else "",
        "vy": f"{r.vy:.0f}" if r.vy is not None else "",
        # motivo se llena SIEMPRE, tambien cuando se rechaza: sin esto no se
        # puede distinguir "el modelo no vio nada" de "el gate lo rechazo".
        "motivo": r.motivo,
        "dist_ult": f"{r.dist_ult:.1f}" if r.dist_ult is not None else "",
    }
    return fila, det_dibujo, to_global, r.modo, ms


def procesar_frame_solo_search(frame, info, hailo, model_hw):
    """
    Barre los CUATRO cuadrantes de la imagen y se queda con el mejor
    candidato. No hay maquina de estados: nunca entra a TRACK.

    Para que existe: con imagenes SUELTAS (no una secuencia continua), la
    maquina de estados miente. Despues de la primera deteccion buena entra a
    TRACK y recorta 1152x640 alrededor de donde estaba la pelota en OTRA
    foto, asi que pierde detecciones por un motivo que no tiene nada que ver
    con el modelo ni con la luz. Peor: el efecto depende del orden de los
    archivos, o sea que contamina la comparacion entre carpetas.

    Con esto cada imagen se mide exactamente igual que las demas y la unica
    variable es la imagen. Es el numero honesto de recall por condicion.

    Cuesta 4 inferencias por imagen en vez de 1. Da lo mismo: esto no corre
    en tiempo real.
    """
    t0 = time.perf_counter()

    tiles = preproceso.search_tiles(frame.shape[:2], model_hw)
    mejor = None            # (deteccion, to_global, tile)

    for i in range(len(tiles)):
        tensor, to_global = preproceso.build_search_input(frame, model_hw, i)
        det = postprocess.process(hailo.infer(tensor), model_hw)
        if det is not None and (mejor is None
                                or det.confidence > mejor[0].confidence):
            mejor = (det, to_global, i)

    ms = (time.perf_counter() - t0) * 1000.0

    if mejor is None:
        deteccion, to_global, tile = None, None, -1
        gx = gy = angulo = None
    else:
        deteccion, to_global, tile = mejor
        gx, gy = to_global(deteccion.x, deteccion.y)
        angulo = geometria.pixel_a_angulo(gx, frame.shape[1],
                                          info.get("camara", 0))

    fila = {
        "n": info["seq"] - 1,
        "ts": info["ts_ns"] / 1e9,
        "modo": "SEARCH",
        "camara": info.get("camara", 0),
        "tile": tile,
        "conf": f"{deteccion.confidence:.4f}" if deteccion is not None else "",
        "x": f"{gx:.1f}" if gx is not None else "",
        "y": f"{gy:.1f}" if gy is not None else "",
        "w": f"{deteccion.w:.1f}" if deteccion is not None else "",
        "h": f"{deteccion.h:.1f}" if deteccion is not None else "",
        "angulo": f"{angulo:.2f}" if angulo is not None else "",
        "sector": "",
        "flost": 0,
        "aceptado_por": "confianza" if deteccion is not None else "",
        "pred_x": "", "pred_y": "", "err_pred": "", "vx": "", "vy": "",
        "motivo": "confianza" if deteccion is not None else "sin candidato",
        "dist_ult": "",
    }
    return fila, deteccion, to_global, Mode.SEARCH, ms


# =============================================================================
# Anotado
# =============================================================================

def anotar(frame, fila, deteccion, to_global, escala):
    """Copia reducida del frame con la caja, el modo y el tile."""
    import cv2

    vis = cv2.resize(frame, None, fx=escala, fy=escala,
                     interpolation=cv2.INTER_AREA)

    verde, rojo, blanco = (0, 255, 0), (0, 0, 255), (255, 255, 255)
    color = verde if fila["modo"] == "TRACK" else blanco

    if deteccion is not None and to_global is not None:
        x1, y1 = to_global(deteccion.x1, deteccion.y1)
        x2, y2 = to_global(deteccion.x2, deteccion.y2)
        cv2.rectangle(vis, (int(x1 * escala), int(y1 * escala)),
                      (int(x2 * escala), int(y2 * escala)), verde, 2)
        cv2.putText(vis, f"{deteccion.confidence:.2f}",
                    (int(x1 * escala), max(14, int(y1 * escala) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, verde, 1, cv2.LINE_AA)

    etiqueta = f"{fila['modo']}"
    if fila["tile"] >= 0:
        etiqueta += f" tile {fila['tile']}"
    if fila["angulo"]:
        etiqueta += f"  ang {fila['angulo']}"
    if fila["flost"]:
        etiqueta += f"  perdidos {fila['flost']}"

    cv2.putText(vis, etiqueta, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                color if fila["flost"] == 0 else rojo, 2, cv2.LINE_AA)
    cv2.putText(vis, f"#{fila['n']}", (8, vis.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, blanco, 1, cv2.LINE_AA)
    return vis


# =============================================================================
# Resumen
# =============================================================================

def _pct(a, b):
    return 100.0 * a / b if b else 0.0


def _percentil(valores, p):
    if not valores:
        return 0.0
    return float(np.percentile(np.asarray(valores, dtype=np.float64), p))


def resumir(filas, latencias, transiciones, t_corrida, solo_search=False,
            etiqueta=None, variante=None):
    n = len(filas)
    con_det = sum(1 for f in filas if f["aceptado_por"])
    en_track = sum(1 for f in filas if f["modo"] == "TRACK")
    en_search = n - en_track

    det_track = sum(1 for f in filas
                    if f["modo"] == "TRACK" and f["aceptado_por"])
    det_search = sum(1 for f in filas
                     if f["modo"] == "SEARCH" and f["aceptado_por"])

    confs = [float(f["conf"]) for f in filas if f["conf"]]
    angulos = [float(f["angulo"]) for f in filas if f["angulo"]]

    L = [
        "",
        "=" * 66,
        "RESUMEN" + (f"  --  {etiqueta}" if etiqueta else "")
        + (f"  [{variante}]" if variante else ""),
        "=" * 66,
        f"{'imagenes procesadas' if solo_search else 'frames procesados':28s} {n}",
        f"con deteccion aceptada       {con_det}  ({_pct(con_det, n):.1f}%)",
    ]

    if solo_search:
        por_tile = {}
        for f in filas:
            if f["aceptado_por"]:
                por_tile[f["tile"]] = por_tile.get(f["tile"], 0) + 1
        L += [
            "",
            "modo SOLO-SEARCH: 4 cuadrantes por imagen, sin maquina de estados.",
            "El numero de arriba es RECALL puro, comparable entre carpetas.",
            "",
            "cuadrante ganador: " + ("  ".join(
                f"t{k}:{v}" for k, v in sorted(por_tile.items())) or "-"),
        ]
    else:
        L += [
        "",
        f"tiempo en TRACK              {en_track}  ({_pct(en_track, n):.1f}%)",
        f"tiempo en SEARCH             {en_search}  ({_pct(en_search, n):.1f}%)",
        f"  deteccion en TRACK         {det_track}/{en_track}"
        f"  ({_pct(det_track, en_track):.1f}%)",
        f"  deteccion en SEARCH        {det_search}/{en_search}"
        f"  ({_pct(det_search, en_search):.1f}%)",
        "",
        f"transiciones SEARCH->TRACK   {transiciones['a_track']}",
        f"transiciones TRACK->SEARCH   {transiciones['a_search']}",
        f"cambios de sector            -   (P4)",
        ]

    if confs:
        L += [
            "",
            f"confianza  p50 {_percentil(confs, 50):.3f}"
            f"   p95 {_percentil(confs, 95):.3f}"
            f"   min {min(confs):.3f}   max {max(confs):.3f}",
        ]
    if angulos:
        L.append(f"angulo     min {min(angulos):.1f}   max {max(angulos):.1f}")

    # Residuo del Kalman: la señal para tunear el filtro. Ver config.py.
    errs = [float(f["err_pred"]) for f in filas if f.get("err_pred")]
    if errs:
        L += [
            "",
            f"residuo Kalman (px)          p50 {_percentil(errs, 50):.1f}"
            f"   p95 {_percentil(errs, 95):.1f}   max {max(errs):.1f}",
            "  grande y sistematico -> subir KALMAN_SIGMA_MEDICION o el modelo",
            "  no alcanza; chico pero la trayectoria tiembla -> bajarlo.",
        ]

    por_motivo = {}
    for f in filas:
        m = f.get("motivo") or ("confianza" if f["aceptado_por"] else "?")
        por_motivo[m] = por_motivo.get(m, 0) + 1
    if por_motivo:
        L += ["", "motivo por frame:"]
        for k, v in sorted(por_motivo.items(), key=lambda kv: -kv[1]):
            L.append(f"  {k:20s} {v:5d}  ({_pct(v, n):.1f}%)")

    # Cuanto se movio la pelota entre frames aceptados, contra el gate. Si el
    # p95 se acerca al gate tipico, el material no es continuo a la tasa que
    # dice el reloj y el gate esta rechazando detecciones legitimas.
    dist = [float(f["dist_ult"]) for f in filas if f.get("dist_ult")]
    if dist:
        L += [
            "",
            f"desplazamiento entre frames  p50 {_percentil(dist, 50):.0f} px"
            f"   p95 {_percentil(dist, 95):.0f} px"
            f"   max {max(dist):.0f} px",
        ]

    L += [
        "",
        f"latencia por frame           p50 {_percentil(latencias, 50):.1f} ms"
        f"   p95 {_percentil(latencias, 95):.1f} ms",
        f"corrida                      {t_corrida:.1f} s"
        f"   ({n / t_corrida if t_corrida else 0:.1f} frames/s)",
        "",
        "OJO: la latencia de arriba es la de ESTA maquina. Si el backend es",
        "Ultralytics en PC, no dice nada del presupuesto de 25 ms de P0, que",
        "solo se mide en la Pi con el .hef.",
        "=" * 66,
    ]
    return "\n".join(L)


# =============================================================================
# Corrida
# =============================================================================

def correr(video=None, carpeta=None, camara=0, fps=None, limite=None,
           salida_csv=DIR_SALIDAS + "/replay.csv",
           salida_mp4=DIR_SALIDAS + "/replay.mp4",
           escala=None, modelo=None, solo_search=False,
           sin_kalman=False, sin_gate=False):
    escala = escala if escala is not None else getattr(
        config, "DEBUG_VIDEO_SCALE", 0.5)

    for ruta in (salida_csv, salida_mp4):
        if ruta:
            carpeta_salida = os.path.dirname(os.path.abspath(ruta))
            os.makedirs(carpeta_salida, exist_ok=True)

    fuente = abrir_fuente_replay(video, carpeta, camara, fps)
    hailo = abrir_inferencia(modelo)
    print(hailo.describe())

    model_hw = preproceso.get_model_hw(hailo)

    tr = None            # se crea con el primer frame, que da el tamano
    filas, latencias = [], []
    transiciones = {"a_track": 0, "a_search": 0}
    escritor_mp4 = None
    archivo_csv = None
    escritor_csv = None
    primero = True

    t0 = time.perf_counter()
    try:
        if salida_csv:
            archivo_csv = open(salida_csv, "w", newline="", encoding="utf-8")
            escritor_csv = csv.DictWriter(archivo_csv, fieldnames=COLUMNAS)
            escritor_csv.writeheader()

        while True:
            if limite is not None and len(filas) >= limite:
                break
            try:
                frame, info = fuente.read()
            except StopIteration:
                break

            if primero:
                h, w = frame.shape[:2]
                tiles = preproceso.search_tiles((h, w), model_hw)
                print(f"[init] frame {w}x{h}  ->  modelo "
                      f"{model_hw[1]}x{model_hw[0]}")
                apagados = [n for n, off in
                            (("kalman", sin_kalman), ("gate", sin_gate)) if off]
                if apagados:
                    print(f"[init] APAGADO: {', '.join(apagados)}  "
                          f"(corrida de comparacion)")
                print(f"[init] SEARCH: {len(tiles)} tiles de "
                      f"{tiles[0][2]}x{tiles[0][3]}, escala "
                      f"{model_hw[1] / tiles[0][2]:.4f}")
                tr = Tracker(
                    n_tiles=len(tiles),
                    camara_inicial=camara,
                    usar_kalman=not sin_kalman,
                    usar_gate=not sin_gate,
                    fn_tile=lambda x, y: preproceso.tile_para_punto(
                        (h, w), model_hw, x, y),
                )
                primero = False

            if solo_search:
                modo_previo = Mode.SEARCH
                fila, det, to_global, _, ms = procesar_frame_solo_search(
                    frame, info, hailo, model_hw)
            else:
                modo_previo = tr.modo
                fila, det, to_global, _, ms = procesar_frame(
                    frame, info, hailo, model_hw, tr)

            # El tile rota solo si ESTE frame fue de SEARCH. En P2 se barre un
            # cuadrante por frame, igual que hacia main.py: es la linea de
            # base honesta. P3 cambia esto a los 4 cuadrantes por barrido.
            if solo_search:
                pass
            elif modo_previo == Mode.SEARCH and tr.modo == Mode.TRACK:
                transiciones["a_track"] += 1
            elif modo_previo == Mode.TRACK and tr.modo == Mode.SEARCH:
                transiciones["a_search"] += 1

            filas.append(fila)
            latencias.append(ms)
            if escritor_csv is not None:
                escritor_csv.writerow(fila)

            if salida_mp4:
                import cv2
                vis = anotar(frame, fila, det, to_global, escala)
                if escritor_mp4 is None:
                    vh, vw = vis.shape[:2]
                    escritor_mp4 = cv2.VideoWriter(
                        salida_mp4, cv2.VideoWriter_fourcc(*"mp4v"),
                        getattr(fuente, "fps", 30.0), (vw, vh))
                    if not escritor_mp4.isOpened():
                        print(f"[replay] no pude abrir '{salida_mp4}' para "
                              f"escribir; sigo sin video anotado")
                        escritor_mp4, salida_mp4 = None, None
                if escritor_mp4 is not None:
                    escritor_mp4.write(vis)

            if len(filas) % 100 == 0:
                print(f"  {len(filas)} frames...")

    finally:
        t_corrida = time.perf_counter() - t0
        if escritor_mp4 is not None:
            escritor_mp4.release()
        if archivo_csv is not None:
            archivo_csv.close()
        try:
            fuente.close()
        except Exception:
            pass
        try:
            hailo.close()
        except Exception:
            pass

    variante = "sin " + " y sin ".join(
        n for n, off in (("kalman", sin_kalman), ("gate", sin_gate)) if off
    ) if (sin_kalman or sin_gate) else "completo"
    print(resumir(filas, latencias, transiciones, t_corrida, solo_search,
                  etiqueta=carpeta or video, variante=variante))
    if salida_csv:
        print(f"csv -> {os.path.abspath(salida_csv)}")
    if salida_mp4:
        print(f"mp4 -> {os.path.abspath(salida_mp4)}")
    return filas


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="banco de pruebas offline")
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--carpeta", type=str, default=None,
                    help="carpeta de imagenes; por defecto, la de config")
    ap.add_argument("--camara", type=int, default=0, choices=[0, 1],
                    help="indice de camara, para pixel_a_angulo")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--limite", type=int, default=None,
                    help="procesar como mucho N frames")
    ap.add_argument("--modelo", type=str, default=None)
    ap.add_argument("--salida-csv", type=str,
                    default=DIR_SALIDAS + "/replay.csv")
    ap.add_argument("--salida-mp4", type=str,
                    default=DIR_SALIDAS + "/replay.mp4")
    ap.add_argument("--sin-mp4", action="store_true")
    ap.add_argument("--sin-csv", action="store_true")
    ap.add_argument("--escala", type=float, default=None)
    ap.add_argument("--sin-kalman", action="store_true",
                    help="no usar la prediccion para elegir candidatos; se "
                         "toma el mas confiado. Para comparar A/B.")
    ap.add_argument("--sin-gate", action="store_true",
                    help="no aplicar el gate de plausibilidad. Para comparar "
                         "A/B. Con --sin-kalman y --sin-gate juntos, el "
                         "tracker equivale a state_machine.py.")
    ap.add_argument("--solo-search", action="store_true",
                    help="barre los 4 cuadrantes de cada imagen y no entra "
                         "nunca a TRACK. Para imagenes SUELTAS: mide recall "
                         "puro, comparable entre carpetas.")
    args = ap.parse_args()

    correr(
        video=args.video,
        carpeta=args.carpeta,
        camara=args.camara,
        fps=args.fps,
        limite=args.limite,
        salida_csv=None if args.sin_csv else args.salida_csv,
        salida_mp4=None if args.sin_mp4 else args.salida_mp4,
        escala=args.escala,
        modelo=args.modelo,
        solo_search=args.solo_search,
        sin_kalman=args.sin_kalman,
        sin_gate=args.sin_gate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())