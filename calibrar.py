"""
calibrar.py

Mide el YAW de cada camara: a que angulo del mundo mira su eje optico.

QUE FALTA Y QUE NO
El ChArUco ya resolvio la LENTE: fx, fy, cx, cy y la distorsion estan en
calibracion_camaras.json, y con eso `geometria.pixel_a_angulo` desdistorsiona
el pixel y saca el angulo respecto del eje optico con un atan. Eso es exacto.

Lo que los intrinsecos NO pueden decir es hacia DONDE esta apuntada la camara:
describen la lente, no el montaje. Falta un numero por camara, el yaw, y es lo
unico que hay que medir en la cancha.

EL ENCODER ES EL TRANSPORTADOR
No hace falta transportador ni cinta metrica. El encoder absoluto tiene 2048
pasos por vuelta, o sea 0.18 grados de resolucion: es el instrumento mas
preciso que hay en el poste.

    1. Pone la pelota quieta en un punto de la cancha.
    2. Move el motor con +5 / -5 / +1 / -1 hasta que la pelota queda CENTRADA
       en la mira de la GoPro.
    3. `medir`. En ese momento el angulo del mundo de ese punto es, POR
       DEFINICION, hacia donde apunta el motor: el angulo del mundo ES donde
       mira la GoPro.
    4. Al mismo tiempo las dos camaras sacan una foto, el modelo detecta la
       pelota y se anota su x en pixeles.
    5. Repetir en al menos 3 puntos por camara, repartidos.

Para cada medicion:  yaw = angulo_del_mundo - angulo_respecto_del_eje_optico
y el yaw de la camara es el promedio.

POR QUE 3 PUNTOS Y NO 7
Con la tabla de interpolacion vieja hacian falta 7 porque la tabla tenia que
capturar la forma de la distorsion. Ahora la distorsion ya esta resuelta y
queda una sola incognita por camara, asi que 3 alcanzan. Los 3 no son para
promediar mejor: son el CONTROL DE CALIDAD. Si dan yaws que difieren en mas de
un grado, no es distorsion (ya la sacaste): es roll de la camara, o el
ChArUco calibrado a otra resolucion, o el eje del motor que no esta vertical.

USO
    python3 calibrar.py
    python3 calibrar.py --salida mi_calibracion.json
"""

import json
import time

import config
import config_hw as chw
import geometria
import preproceso

SALIDA = "calibracion_angular.json"


def _detectar(hailo, frame, model_hw):
    """
    Mejor deteccion del frame, barriendo los 4 cuadrantes. Devuelve
    (x_px_nativo, confianza) o (None, None).

    Se barre entero y no se usa el tracker a proposito: aca no hay secuencia
    ni continuidad, cada medicion es independiente.
    """
    import postprocess

    mejor = None
    tiles = preproceso.search_tiles(frame.shape[:2], model_hw)
    for i in range(len(tiles)):
        tensor, to_global = preproceso.build_search_input(frame, model_hw, i)
        det = postprocess.process(hailo.infer(tensor), model_hw)
        if det is not None and (mejor is None or det.confidence > mejor[0]):
            gx, _ = to_global(det.x, det.y)
            mejor = (det.confidence, gx)
    if mejor is None:
        return None, None
    return mejor[1], mejor[0]


def _angulo_eje_optico(px: float, ancho: int, camara: int) -> float | None:
    """
    Angulo del pixel respecto del EJE OPTICO, desdistorsionado. Es
    pixel_a_angulo sin el yaw, que es justamente lo que se esta midiendo.
    """
    lut = geometria._lut(camara, ancho)
    if lut is None:
        return None
    import numpy as np
    xs, angulos = lut
    off = float(np.interp(float(px), xs, angulos))
    if chw.CAM_ESPEJO.get(camara, False):
        off = -off
    return off


def _resumen(mediciones: dict) -> str:
    lineas = ["", "=" * 64, "YAW POR CAMARA", "=" * 64]
    yaws = {}
    for cam in sorted(mediciones):
        vals = [m["yaw"] for m in mediciones[cam]]
        if not vals:
            lineas.append(f"camara {cam}: sin mediciones")
            continue
        prom = sum(vals) / len(vals)
        disp = max(vals) - min(vals)
        yaws[cam] = prom
        lineas.append(f"camara {cam}: yaw = {prom:.2f} deg   "
                      f"({len(vals)} puntos, dispersion {disp:.2f})")
        for m in mediciones[cam]:
            lineas.append(f"    mundo {m['angulo_mundo']:7.2f}  "
                          f"x {m['x_px']:7.1f}  eje {m['off']:+7.2f}  "
                          f"yaw {m['yaw']:7.2f}  conf {m['conf']:.2f}")
        if disp > 1.0:
            lineas.append(
                f"    !! dispersion de {disp:.2f} deg. La distorsion ya esta "
                f"resuelta por el ChArUco, asi que esto NO es la lente. "
                f"Mira: roll de la camara (esta torcida), el ChArUco "
                f"calibrado a otra resolucion, o el eje del motor no vertical.")

    if yaws:
        lineas += ["", "Pegá esto en config_hw.py:", "",
                   "CAM_YAW = {" + ", ".join(
                       f"{c}: {v:.2f}" for c, v in sorted(yaws.items())) + "}"]

        # Control cruzado: en la zona de solape las dos camaras tienen que
        # coincidir. Es la unica verificacion que no depende de haber medido
        # bien: si difieren, algo esta mal en el montaje.
        if len(yaws) == 2:
            ancho = int(config.CAM_ANCHO)
            lineas += ["", "Control en la zona de solape (80..100 grados):"]
            peor = 0.0
            for a in (80.0, 90.0, 100.0):
                px0 = geometria.angulo_a_pixel(a, ancho, 0)
                px1 = geometria.angulo_a_pixel(a, ancho, 1)
                if 0 <= px0 < ancho and 0 <= px1 < ancho:
                    lineas.append(f"    {a:.0f} deg -> cam0 x={px0:.0f}  "
                                  f"cam1 x={px1:.0f}")
            lineas.append("    (con CAM_YAW ya pegado, poné la pelota ahi y "
                          "verifica que las dos camaras reporten el mismo "
                          "angulo con menos de 3 grados de diferencia)")
    lineas.append("=" * 64)
    return "\n".join(lineas)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=str, default=SALIDA)
    ap.add_argument("--sin-motor", action="store_true",
                    help="el angulo del mundo se tipea a mano")
    args = ap.parse_args()

    # --- intrinsecos: sin esto no hay nada que medir
    faltan = [c for c in chw.CAMARAS if geometria.intrinsecos(c) is None]
    if faltan:
        print(f"!! no encuentro los intrinsecos de la/s camara/s {faltan} en "
              f"'{chw.CAL_INTRINSECOS}'. Corré primero la calibracion del "
              f"ChArUco, o revisá CAM_LETRA_CALIBRACION.")
        return 1

    # --- hardware
    import fuente
    from inferencia import abrir as abrir_inferencia

    motor = encoder = None
    if not args.sin_motor:
        import encoder_lib
        import motor_lib
        motor = motor_lib.Motor()
        encoder = encoder_lib.Encoder()
        if chw.ENC_APLICAR_CONFIG_AL_ARRANCAR:
            encoder.aplicar_config()
        motor.deshabilitar()
        motor.set_corriente(chw.MOTOR_CORRIENTE_MA)
        motor.set_micropasos(chw.MOTOR_MICROPASOS)
        motor.set_velocidad(chw.MOTOR_VELOCIDAD)
        motor.set_aceleracion(chw.MOTOR_ACELERACION)
        motor.habilitar()
        print(f"motor en {motor.puerto}, encoder listo")

    hailo = abrir_inferencia()
    model_hw = preproceso.get_model_hw(hailo)
    camaras = {c: fuente.abrir_fuente(indice=c) for c in chw.CAMARAS}
    print(f"camaras abiertas: {sorted(camaras)}")

    mediciones = {c: [] for c in chw.CAMARAS}

    print(__doc__.split("USO")[0].split("EL ENCODER")[1])
    print("comandos:  +5  -5  +1  -1  |  medir  |  ver  |  fin")

    try:
        while True:
            try:
                cmd = input(">> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break
            if not cmd:
                continue

            if cmd in ("fin", "q", "exit"):
                break

            if cmd == "ver":
                print(_resumen(mediciones))
                continue

            # --- mover
            if cmd[0] in "+-" and motor is not None:
                try:
                    delta = float(cmd)
                except ValueError:
                    print("no entendi. Usá +5, -5, +1, -1")
                    continue
                motor.mover_grados(delta * chw.RELACION_TRANSMISION)
                motor.esperar_fin()
                time.sleep(0.2)
                print(f"    motor {motor.posicion_grados():+.2f} deg   "
                      f"encoder {encoder.leer_angulo()[1]:.2f} deg")
                continue

            if cmd != "medir":
                print("comandos:  +5  -5  +1  -1  |  medir  |  ver  |  fin")
                continue

            # --- medir
            if motor is not None:
                # El angulo del mundo ES hacia donde apunta el motor: la
                # GoPro esta montada sobre el, y la pelota esta centrada en su
                # mira. Se usa el motor y no el encoder crudo porque el
                # encoder tiene su propio offset, que init_sistema ya resolvio.
                angulo_mundo = geometria.grados_motor_a_angulo(
                    motor.posicion_grados())
                enc = encoder.leer_angulo()[1]
                print(f"    motor {motor.posicion_grados():+.2f}  "
                      f"encoder {enc:.2f}  ->  mundo {angulo_mundo:.2f}")
            else:
                try:
                    angulo_mundo = float(input("    angulo del mundo: "))
                except ValueError:
                    print("    numero invalido")
                    continue

            for cam, src in sorted(camaras.items()):
                frame, _ = src.read()
                ancho = frame.shape[1]
                x, conf = _detectar(hailo, frame, model_hw)
                if x is None:
                    print(f"    camara {cam}: no veo la pelota")
                    continue
                off = _angulo_eje_optico(x, ancho, cam)
                if off is None:
                    print(f"    camara {cam}: sin intrinsecos")
                    continue
                yaw = angulo_mundo - off
                mediciones[cam].append({
                    "angulo_mundo": angulo_mundo, "x_px": x,
                    "off": off, "yaw": yaw, "conf": conf})
                print(f"    camara {cam}: x={x:7.1f}  eje {off:+7.2f}  "
                      f"->  yaw {yaw:7.2f}   (conf {conf:.2f})")

    finally:
        print(_resumen(mediciones))
        try:
            with open(args.salida, "w", encoding="utf-8") as fh:
                json.dump({"mediciones": mediciones}, fh, indent=2)
            print(f"crudo -> {args.salida}")
        except Exception as exc:
            print(f"no pude guardar: {exc}")

        for src in camaras.values():
            try:
                src.close()
            except Exception:
                pass
        try:
            hailo.close()
        except Exception:
            pass
        if motor is not None:
            motor.deshabilitar()
            motor.close()
        if encoder is not None:
            encoder.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
