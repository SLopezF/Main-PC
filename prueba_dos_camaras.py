"""
prueba_dos_camaras.py  --  P1: las dos cámaras abiertas a la vez.

La pregunta que contesta es una sola: ¿el CSI de la Pi 5 aguanta los dos
IMX708 abiertos simultáneamente a 2304x1296 y 40 fps?

Importa porque el patrón de SEARCH alterna entre cámaras. Si hay que
tener abierta solo la activa, cada cambio de cámara paga un stop/start de
Picamera2 (segundos, no milisegundos) y el patrón de P3 deja de ser
viable tal como está escrito.

CÓMO SE MIDE

Tres pasadas, en este orden:

  1. control cam0 sola
  2. control cam1 sola
  3. las dos a la vez, leyendo alternado

Los controles no son decoración: si en la pasada 3 se pierden frames, hay
que saber si es por la simultaneidad o si esa cámara ya perdía sola. Sin
la línea de base, un mal resultado no dice a quién culpar. Se pueden
saltear con --sin-control cuando ya se conocen.

QUÉ SE MIRA EN CADA PASADA

  fps por timestamps  el sensor real, no el reloj de pared
  perdidos            periodos del sensor que no se entregaron
  pisados             frames que el hilo de captura sobreescribió porque
                      el consumidor no los había leído todavía

`perdidos` y `pisados` miden cosas distintas y conviene no confundirlos:
perdidos es la cámara entregando menos de lo pedido (problema de ancho de
banda o de modo del sensor); pisados es el consumidor yendo más lento que
la cámara (problema de cómputo). En este script no hay cómputo, así que
pisados alto apunta a que la lectura alternada misma es el cuello.

    python3 prueba_dos_camaras.py --n 400
    python3 prueba_dos_camaras.py --canales    # pide objeto rojo, ver 4.1
"""

import argparse
import time
from datetime import datetime

import numpy as np

import config
from camera_source import ThreadedCameraSource


# Máximo de frames perdidos aceptable, en porcentaje. Del criterio de
# aceptación de P1 en el briefing.
MAX_PERDIDOS_PCT = getattr(config, "MAX_PERDIDOS_PCT", 2.0)

# Lecturas de descarte al empezar cada pasada. Los primeros frames vienen
# de los buffers ya llenos y vuelven al instante, lo que infla el fps y
# ensucia el dt del arranque.
CALENTAMIENTO = 20


def resumen_pasada(dts: list[float], perdidos: int, pisados: int,
                   n: int, segundos: float, periodo_ms: float) -> dict:
    """Convierte las muestras crudas de una pasada en las métricas de P1."""
    d = np.asarray(dts, dtype=np.float64)
    p50 = float(np.percentile(d, 50)) if d.size else float("nan")
    p95 = float(np.percentile(d, 95)) if d.size else float("nan")

    # Los frames "posibles" son los entregados más los que se saltearon:
    # el porcentaje se calcula sobre lo que el sensor debería haber dado,
    # no sobre lo que dio.
    posibles = n + perdidos
    pct = 100.0 * perdidos / max(1, posibles)

    return {
        "n": n,
        "segundos": round(segundos, 2),
        "fps_pared": round(n / segundos, 2) if segundos > 0 else float("nan"),
        "fps_timestamps": round(1000.0 / p50, 2) if p50 > 0 else float("nan"),
        "dt_p50_ms": round(p50, 2),
        "dt_p95_ms": round(p95, 2),
        "periodo_objetivo_ms": round(periodo_ms, 2),
        "perdidos": perdidos,
        "perdidos_pct": round(pct, 2),
        "pisados": pisados,
    }


def imprimir_pasada(titulo: str, r: dict, log=print) -> None:
    log("")
    log(f"--- {titulo} " + "-" * max(0, 44 - len(titulo)))
    log(f"  fps por timestamps : {r['fps_timestamps']:.2f}   "
        f"(por reloj de pared {r['fps_pared']:.2f})")
    log(f"  dt entre entregas  : p50 {r['dt_p50_ms']:.2f} ms   "
        f"p95 {r['dt_p95_ms']:.2f} ms   objetivo {r['periodo_objetivo_ms']:.2f} ms")
    log(f"  perdidos           : {r['perdidos']} de "
        f"{r['n'] + r['perdidos']} ({r['perdidos_pct']:.2f}%)")
    log(f"  pisados sin leer   : {r['pisados']}")


def abrir(indice: int, args) -> ThreadedCameraSource:
    """Abre una cámara con los parámetros de config, sobre el conector pedido."""
    return ThreadedCameraSource(
        size=(args.ancho, args.alto),
        fps=args.fps,
        formato=args.formato,
        buffer_count=getattr(config, "CAM_BUFFERS", 4),
        exposicion_us=getattr(config, "CAM_EXPOSICION_US", None),
        ganancia=getattr(config, "CAM_GANANCIA", None),
        enfoque=getattr(config, "CAM_ENFOQUE", None),
        indice=indice,
    )


def medir_una(cam, n: int, periodo_ms: float, log=print) -> dict:
    """Pasada de control: una sola cámara, lectura directa."""
    for _ in range(CALENTAMIENTO):
        cam.read()

    perdidos_antes = cam.total_perdidos
    dts = []
    t0 = time.perf_counter()
    for _ in range(n):
        _, info = cam.read()
        dts.append(info["dt_ms"])
    segundos = time.perf_counter() - t0

    return resumen_pasada(dts, cam.total_perdidos - perdidos_antes,
                          info.get("pisados", 0), n, segundos, periodo_ms)


def medir_dos(cam0, cam1, n: int, periodo_ms: float, log=print) -> dict:
    """
    Pasada principal: las dos abiertas, leyendo alternado.

    Se lee cam0, después cam1, y así. Es el patrón más exigente que va a
    hacer el sistema y es exactamente lo que hace SEARCH cuando cambia de
    cámara. Cada read() bloquea hasta que ESA cámara tenga un frame nuevo
    sin consumir, así que si una de las dos se atrasa, arrastra a la otra.
    """
    for _ in range(CALENTAMIENTO):
        cam0.read()
        cam1.read()

    perdidos0_antes = cam0.total_perdidos
    perdidos1_antes = cam1.total_perdidos
    dts0, dts1 = [], []
    info0 = info1 = {}

    t0 = time.perf_counter()
    for _ in range(n):
        _, info0 = cam0.read()
        dts0.append(info0["dt_ms"])
        _, info1 = cam1.read()
        dts1.append(info1["dt_ms"])
    segundos = time.perf_counter() - t0

    return {
        "cam0": resumen_pasada(dts0, cam0.total_perdidos - perdidos0_antes,
                               info0.get("pisados", 0), n, segundos, periodo_ms),
        "cam1": resumen_pasada(dts1, cam1.total_perdidos - perdidos1_antes,
                               info1.get("pisados", 0), n, segundos, periodo_ms),
        "segundos": round(segundos, 2),
    }


def verificar_canales(cam0, cam1, log=print) -> dict:
    """
    Punto 2 de P1. Hay que apuntar las DOS cámaras a algo rojo.

    Es la verificación de 4.1: Picamera2 con formato "RGB888" suele
    entregar el array en orden BGR, y equivocarse cuesta un factor 8 de
    confianza en el caso peor sin que se note a ojo.
    """
    log("")
    log("=" * 60)
    log("  ORDEN DE CANALES")
    log("=" * 60)
    log("Poné algo predominantemente ROJO delante de las DOS cámaras.")
    input("ENTER cuando esté listo... ")

    log("\n[cam0]")
    orden0 = cam0.verificar_canales()
    log("\n[cam1]")
    orden1 = cam1.verificar_canales()

    if orden0 != orden1:
        log(f"\n[FALLA] cam0 dice {orden0} y cam1 dice {orden1}. Las dos usan "
            f"el mismo formato, así que no pueden diferir: o el objeto no "
            f"llenaba las dos vistas, o una está mal configurada. Repetir.")
    else:
        log(f"\n[ok] las dos entregan {orden0}.")
        esperado_bgr = getattr(config, "FUENTE_ENTREGA_BGR", None)
        if esperado_bgr is not None:
            dice_bgr = (orden0 == "BGR")
            if dice_bgr != esperado_bgr:
                log(f"[ACCION] config.FUENTE_ENTREGA_BGR = {esperado_bgr} pero la "
                    f"cámara entrega {orden0}. Cambiarlo a {dice_bgr}.")
            else:
                log(f"[ok] coincide con config.FUENTE_ENTREGA_BGR = {esperado_bgr}. "
                    f"Sacale el 'PENDIENTE DE P1' al comentario.")

    return {"cam0": orden0, "cam1": orden1}


def veredicto(dos: dict, log=print) -> bool:
    """Aplica el criterio de aceptación de P1 y explica qué hacer si falla."""
    log("")
    log("=" * 60)
    ok = True
    for nombre in ("cam0", "cam1"):
        r = dos[nombre]
        malo_perdidos = r["perdidos_pct"] > MAX_PERDIDOS_PCT
        # Un fps bastante por debajo del pedido es una falla aunque no se
        # contabilicen perdidos: querría decir que libcamera eligió un modo
        # de sensor más lento y el periodo de referencia quedó mal.
        malo_fps = r["fps_timestamps"] < 0.95 * float(config.CAM_FPS)
        if malo_perdidos or malo_fps:
            ok = False
            log(f"[FALLA] {nombre}: {r['perdidos_pct']:.2f}% perdidos, "
                f"{r['fps_timestamps']:.1f} fps efectivos.")
        else:
            log(f"[OK] {nombre}: {r['perdidos_pct']:.2f}% perdidos, "
                f"{r['fps_timestamps']:.1f} fps efectivos.")

    if ok:
        log(f"\nP1 cumple: las dos cámaras sostienen {config.CAM_FPS:.0f} fps "
            f"simultáneas con menos de {MAX_PERDIDOS_PCT:.0f}% de pérdida.")
    else:
        log(f"\nP1 NO cumple. Antes de aceptar el plan B (una sola cámara "
            f"abierta), revisar en este orden:")
        log("  1. Comparar contra las pasadas de control. Si una cámara sola "
            "también pierde, el problema no es la simultaneidad.")
        log("  2. `rpicam-hello --list-cameras`: confirmar que las dos están "
            "en el modo 2304x1296 y no en uno más lento.")
        log("  3. Subir CAM_BUFFERS: con dos cámaras hay el doble de presión "
            "sobre los buffers.")
        log("  4. Si el dt p95 es mucho mayor que el p50, el problema es "
            "jitter de la lectura alternada, no ancho de banda del CSI.")
    return ok


def escribir_estado(path: str, dos: dict, controles: dict, canales: dict | None,
                    ok: bool, log=print) -> None:
    """Deja los números en ESTADO.md, que es la memoria entre sesiones."""
    lineas = [
        "",
        f"## P1 — dos cámaras simultáneas ({datetime.now():%Y-%m-%d %H:%M})",
        "",
        f"- pedido: {config.CAM_ANCHO}x{config.CAM_ALTO} @ {config.CAM_FPS:.0f} fps",
        "",
        "| pasada | fps (timestamps) | dt p50 | dt p95 | perdidos | pisados |",
        "|---|---|---|---|---|---|",
    ]
    for nombre, r in list(controles.items()) + [
        ("cam0 + cam1 -> cam0", dos["cam0"]),
        ("cam0 + cam1 -> cam1", dos["cam1"]),
    ]:
        lineas.append(
            f"| {nombre} | {r['fps_timestamps']:.2f} | {r['dt_p50_ms']:.2f} | "
            f"{r['dt_p95_ms']:.2f} | {r['perdidos']} ({r['perdidos_pct']:.2f}%) | "
            f"{r['pisados']} |"
        )

    lineas += ["", f"**Resultado:** {'CUMPLE' if ok else 'NO CUMPLE'} el criterio "
                   f"de menos de {MAX_PERDIDOS_PCT:.0f}% de frames perdidos."]
    if canales:
        lineas += [
            "",
            f"**Orden de canales (4.1):** cam0 entrega `{canales['cam0']}`, "
            f"cam1 entrega `{canales['cam1']}`. Verificado con "
            f"`verificar_canales()` contra un objeto rojo.",
        ]
    else:
        lineas += ["", "**Orden de canales:** NO verificado en esta corrida "
                       "(correr con --canales). Sigue pendiente."]
    lineas.append("")

    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lineas))
    log(f"[ok] {path} actualizado")


def main() -> None:
    ap = argparse.ArgumentParser(description="P1: dos cámaras a la vez")
    ap.add_argument("--n", type=int, default=400, help="frames medidos por cámara")
    ap.add_argument("--fps", type=float, default=config.CAM_FPS)
    ap.add_argument("--ancho", type=int, default=config.CAM_ANCHO)
    ap.add_argument("--alto", type=int, default=config.CAM_ALTO)
    ap.add_argument("--formato", default="RGB888")
    ap.add_argument("--sin-control", action="store_true",
                    help="saltear las pasadas de una cámara sola")
    ap.add_argument("--canales", action="store_true",
                    help="verificar el orden de canales (pide un objeto rojo)")
    ap.add_argument("--estado", default="ESTADO.md",
                    help="archivo de estado a actualizar ('' para no tocarlo)")
    args = ap.parse_args()

    periodo_ms = 1000.0 / args.fps
    print(f"[init] pedido {args.ancho}x{args.alto} @ {args.fps:.0f} fps "
          f"-> periodo {periodo_ms:.2f} ms")

    controles = {}
    if not args.sin_control:
        for indice in (0, 1):
            print(f"\n[control] cámara {indice} sola...")
            with abrir(indice, args) as cam:
                r = medir_una(cam, args.n, periodo_ms)
            controles[f"cam{indice} sola"] = r
            imprimir_pasada(f"control: cam{indice} sola", r)

    print("\n[principal] abriendo las dos...")
    cam0 = abrir(0, args)
    try:
        cam1 = abrir(1, args)
    except Exception as exc:
        cam0.close()
        raise SystemExit(
            f"No pude abrir la cámara 1: {exc}\n"
            f"Si la 0 abrió bien, revisá `rpicam-hello --list-cameras`: "
            f"puede ser el conector, el cable o que solo haya una conectada."
        )

    try:
        canales = verificar_canales(cam0, cam1) if args.canales else None
        dos = medir_dos(cam0, cam1, args.n, periodo_ms)
    finally:
        cam0.close()
        cam1.close()

    imprimir_pasada("las dos abiertas: cam0", dos["cam0"])
    imprimir_pasada("las dos abiertas: cam1", dos["cam1"])
    ok = veredicto(dos)

    if args.estado:
        escribir_estado(args.estado, dos, controles, canales, ok)


if __name__ == "__main__":
    main()
