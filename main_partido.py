"""
main_partido.py

Sistema completo: dos camaras IMX708 (de a una a la vez) -> Hailo-8 ->
angulo -> motor, con la GoPro grabando en paralelo.

MODO DEBUG (el default, y lo unico que corre hoy)
Un ciclo por vez, avanzando con ENTER, para poder mirar cada paso:

    ENTER  -> foto con la camara activa, inferencia, imagen anotada con la
              pelota marcada, pixeles (x, y) y angulo 0..180
    ENTER  -> mueve el motor a ese angulo y verifica contra el encoder
              absoluto; despues saca una foto con la GoPro
    ENTER  -> vuelve a empezar

En el ultimo prompt se aceptan comandos: 0/1 fuerza camara, c alterna,
a acepta la sugerencia del selector con histeresis, s saltea el motor,
q sale.

RUTINA DE ARRANQUE
    1. Encoder: aplicar config (es volatil), leer PPR y el angulo absoluto X.
    2. Motor: verificar el USB de la ESP32, DESHABILITAR el driver,
       fijar corriente en 600 mA.
    3. Homing: el motor esta corrido (X - 79) grados, porque con el motor en
       su cero el encoder marca 79. Se mueve -(X - 79), se verifica contra el
       encoder y se declara zero.
    4. GoPro: init() (por ahora un handler, se completa despues).
    5. Hailo: cargar el .hef y calentar.

    Nota sobre el orden: pediste fijar la corriente antes de verificar el
    USB, pero la corriente SE FIJA POR el USB (es un comando a la ESP32),
    asi que verificar tiene que ir primero. Es el unico cambio de orden.

DECISIONES QUE VIENEN DEL CODIGO VIEJO Y SE MANTIENEN
  - Los tensores van en RGB: el modelo rinde bastante mejor (0.879 vs 0.859
    en crop nativo, 0.476 vs 0.061 en frame reducido).
  - SEARCH es un mosaico de tiles, no el frame entero reducido: a 0.25x la
    confianza medida cae a 0.0025 y la pelota deja de existir para el modelo.
    En debug se barren TODOS los tiles y se toma el mejor, porque no importa
    la latencia; en el modo partido se rota de a uno por frame.

USO
    python3 main_partido.py                 # debug, camara 0
    python3 main_partido.py --camara 1
    python3 main_partido.py --sin-motor --sin-gopro
    python3 main_partido.py --auto          # sin ENTER, ciclo continuo
"""

import argparse
import math
import os
import time

import numpy as np

import config
import config_hw as chw
import geometria
import hw_falsos
import main as pipeline
import postprocess
from geometria import SelectorCamara

# Librerias de hardware: la real si ya existe, la falsa si todavia no.
Encoder, _ENC_REAL = hw_falsos.cargar("encoder_lib", "Encoder", hw_falsos.Encoder)
Motor, _MOT_REAL = hw_falsos.cargar("motor_lib", "Motor", hw_falsos.Motor)
GoPro, _GP_REAL = hw_falsos.cargar("gopro_lib", "GoPro", hw_falsos.GoPro)


# =============================================================================
# Camaras
# =============================================================================

class GestorCamaras:
    """
    Las dos Picamera2, con acceso por indice. Se abren de forma perezosa:
    abrir cuesta ~2 s de convergencia del AE, asi que no se paga por una
    camara que nunca se usa.

    Con CAM_MANTENER_ABIERTAS=True las dos quedan abiertas y cambiar de
    camara es instantaneo (a costa de ancho de banda CSI y de RAM). Con
    False se cierra la anterior al cambiar y cada cambio cuesta esos 2 s.
    """

    def __init__(self, ancho=None, alto=None, fps=None):
        self.ancho = ancho or config.CAM_ANCHO
        self.alto = alto or config.CAM_ALTO
        self.fps = fps or config.CAM_FPS
        self._abiertas: dict[int, object] = {}

    def _abrir(self, indice: int):
        from camera_source import CameraSource

        print(f"[cam] abriendo camara {indice} "
              f"({self.ancho}x{self.alto} @ {self.fps} fps)...")
        return CameraSource(
            size=(self.ancho, self.alto),
            fps=self.fps,
            buffer_count=getattr(config, "CAM_BUFFERS", 4),
            exposicion_us=config.CAM_EXPOSICION_US,
            ganancia=config.CAM_GANANCIA,
            enfoque=config.CAM_ENFOQUE,
            indice=indice,
        )

    def leer(self, indice: int):
        """Frame mas reciente de la camara `indice`, mas su metadata."""
        if indice not in self._abiertas:
            self._abiertas[indice] = self._abrir(indice)
            if not chw.CAM_MANTENER_ABIERTAS:
                for otro in list(self._abiertas):
                    if otro != indice:
                        self._abiertas.pop(otro).close()
        return self._abiertas[indice].read()

    def precalentar(self, indices) -> None:
        for i in indices:
            self.leer(i)

    def close(self) -> None:
        for cam in self._abiertas.values():
            try:
                cam.close()
            except Exception:
                pass
        self._abiertas.clear()


# =============================================================================
# Deteccion
# =============================================================================

def detectar_pelota(frame, hailo, model_hw, completo: bool | None = None):
    """
    Devuelve la mejor deteccion del frame:
    (deteccion, x_global, y_global, tile, ms), o (None, None, None, -1, ms).

    Dos caminos, segun config_hw.SEARCH_FULL:

      completo=True   el frame ENTERO con letterbox al tamano del modelo. Una
                      sola inferencia y campo de vision completo. Sirve con la
                      pelota cerca, donde aun reducida queda de buen tamano.

      completo=False  mosaico: se barren TODOS los tiles y gana el mejor. La
                      pelota conserva mas pixeles, al precio de una inferencia
                      por tile. Necesario cuando esta lejos.

    En el modo partido el mosaico rota un tile por frame, para que el costo
    por frame sea una sola inferencia igual que en TRACK. Aca se barren todos
    porque en debug la latencia no importa y perder la pelota por estar
    mirando el cuadrante equivocado, si.
    """
    if completo is None:
        completo = getattr(chw, "SEARCH_FULL", False)

    t0 = time.perf_counter()
    mejor = None

    if completo:
        tensor, to_global = pipeline.build_search_full(frame, model_hw)
        cands = postprocess.process_candidates(
            hailo.infer(tensor), model_hw,
            umbral=config.CONF_CANDIDATO, topk=3,
        )
        for d in cands:
            if mejor is None or d.confidence > mejor[0].confidence:
                mejor = (d, to_global, -1)
    else:
        tiles = pipeline.search_tiles(frame.shape[:2], model_hw)
        for i in range(len(tiles)):
            tensor, to_global = pipeline.build_search_input(frame, model_hw, i)
            cands = postprocess.process_candidates(
                hailo.infer(tensor), model_hw,
                umbral=config.CONF_CANDIDATO, topk=3,
            )
            for d in cands:
                if mejor is None or d.confidence > mejor[0].confidence:
                    mejor = (d, to_global, i)

    ms = (time.perf_counter() - t0) * 1000.0
    if mejor is None:
        return None, None, None, -1, ms

    det, to_global, tile = mejor
    gx, gy = to_global(det.x, det.y)
    return det, gx, gy, tile, ms


def anotar(frame, camara, det, gx, gy, angulo, escala=None):
    """Copia reducida del frame con la pelota marcada y los numeros encima."""
    import cv2

    escala = escala if escala is not None else chw.DEBUG_ESCALA
    h, w = frame.shape[:2]
    vis = cv2.resize(frame, (int(w * escala), int(h * escala)),
                     interpolation=cv2.INTER_AREA)

    verde, rojo, gris = (0, 255, 0), (0, 0, 255), (180, 180, 180)

    # Regla de angulos en el borde inferior: donde cae cada 10 grados.
    y_regla = vis.shape[0] - 12
    for a in range(0, 181, 10):
        px = geometria.angulo_a_pixel(a, w, camara) * escala
        if 0 <= px < vis.shape[1]:
            largo = 10 if a % 30 == 0 else 5
            cv2.line(vis, (int(px), y_regla), (int(px), y_regla - largo), gris, 1)
            if a % 30 == 0:
                cv2.putText(vis, str(a), (int(px) - 10, y_regla + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, gris, 1, cv2.LINE_AA)

    if det is not None:
        px, py = int(gx * escala), int(gy * escala)
        r = max(6, int((det.w + det.h) / 4 * escala))
        cv2.circle(vis, (px, py), r, verde, 2)
        cv2.drawMarker(vis, (px, py), verde, cv2.MARKER_CROSS, 22, 1)
        cv2.line(vis, (px, py), (px, y_regla), verde, 1)
        texto = (f"cam{camara}  conf {det.confidence:.2f}  "
                 f"px ({int(gx)}, {int(gy)})  ang {angulo:.1f} deg  "
                 f"s{det.stride}  {det.w:.0f}x{det.h:.0f}px")
        color = verde
    else:
        texto = f"cam{camara}  SIN DETECCION"
        color = rojo

    for grosor, col in ((4, (0, 0, 0)), (1, color)):
        cv2.putText(vis, texto, (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, col, grosor, cv2.LINE_AA)
    return vis


def guardar_recorte(frame, gx, gy, ruta, margen=140):
    """
    Recorte alrededor de la deteccion, al doble de tamano. Es lo unico que
    permite responder la pregunta que importa: eso que marco, ¿es la pelota?
    """
    import cv2

    cx, cy = int(gx), int(gy)
    x1, y1 = max(0, cx - margen), max(0, cy - margen)
    x2 = min(frame.shape[1], cx + margen)
    y2 = min(frame.shape[0], cy + margen)
    rec = frame[y1:y2, x1:x2]
    if rec.size == 0:
        return
    rec = cv2.resize(rec, (rec.shape[1] * 2, rec.shape[0] * 2),
                     interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(ruta, rec)


# =============================================================================
# Arranque
# =============================================================================

class Sistema:
    """Todo el hardware ya inicializado, para pasarlo de una sola pieza."""

    def __init__(self):
        self.encoder = None
        self.motor = None
        self.gopro = None
        self.hailo = None
        self.camaras = None
        self.model_hw = None
        self.selector = None
        self.angulo_encoder_inicial = None

    def close(self):
        for nombre in ("camaras", "gopro", "hailo"):
            obj = getattr(self, nombre)
            if obj is not None:
                try:
                    obj.close()
                except Exception as exc:
                    print(f"[cierre] {nombre}: {exc}")
        if self.motor is not None:
            try:
                self.motor.deshabilitar()
                self.motor.close()
            except Exception as exc:
                print(f"[cierre] motor: {exc}")
        if self.encoder is not None:
            try:
                self.encoder.close()
            except Exception as exc:
                print(f"[cierre] encoder: {exc}")


def _titulo(t):
    print()
    print("=" * 68)
    print(f"  {t}")
    print("=" * 68)


def inicializar(args) -> Sistema:
    sis = Sistema()

    # ---------------------------------------------------------------- ENCODER
    _titulo("1/5  ENCODER")
    sis.encoder = Encoder(chw.ENC_BUS, chw.ENC_DEVICE)

    if chw.ENC_APLICAR_CONFIG_AL_ARRANCAR:
        # Soft write: se pierde al cortar la alimentacion, va en cada arranque.
        sis.encoder.aplicar_config()

    cfg = sis.encoder.leer_config()
    ppr = cfg.get("abi_ppr")
    pasos = cfg.get("abi_steps")
    print(f"  ABI: {ppr} ppr = {pasos} pasos/vuelta")
    if pasos:
        print(f"  resolucion en x4: {360.0 / pasos:.4f} deg/paso")
    if ppr != chw.ENC_PPR_ESPERADO:
        print(f"  !! esperaba {chw.ENC_PPR_ESPERADO} ppr. El chip no tomo la "
              f"config: revisa SPI antes de confiar en el homing.")

    diag = sis.encoder.diagnostico()
    print(f"  iman: AGC {diag.get('agc')}/255 -> "
          f"{sis.encoder.estado_iman(diag)}")
    if diag.get("mag_too_low") or diag.get("mag_too_high"):
        print("  !! con el iman fuera de rango la lectura de angulo no es "
              "confiable y el homing va a quedar corrido.")

    raw, X, err = sis.encoder.leer_angulo()
    sis.angulo_encoder_inicial = X
    print(f"  angulo absoluto inicial X = {X:.2f} deg (raw {raw}, err {err})")

    # ------------------------------------------------------------------ MOTOR
    _titulo("2/5  MOTOR (ESP32 por USB)")
    if args.sin_motor:
        print("  --sin-motor: se saltea")
    else:
        sis.motor = Motor(chw.MOTOR_PUERTO, chw.MOTOR_BAUD)
        if isinstance(sis.motor, hw_falsos.Motor):
            sis.motor._encoder = sis.encoder  # el simulado arrastra al encoder

        if not sis.motor.conectado():
            raise RuntimeError(
                "La ESP32 no responde por USB. Revisa el cable y que no haya "
                "otro proceso con el puerto abierto (el monitor serie del IDE)."
            )
        print(f"  conectado a {sis.motor.puerto}")
        for linea in sis.motor.status():
            print(f"    {linea}")

        # Driver APAGADO antes de tocar la corriente: cambiar la corriente con
        # las bobinas energizadas es la forma facil de calentar el TMC2209.
        sis.motor.deshabilitar()
        sis.motor.set_corriente(chw.MOTOR_CORRIENTE_MA)
        sis.motor.set_micropasos(chw.MOTOR_MICROPASOS)
        sis.motor.set_velocidad(chw.MOTOR_VELOCIDAD)
        sis.motor.set_aceleracion(chw.MOTOR_ACELERACION)
        print(f"  corriente {chw.MOTOR_CORRIENTE_MA} mA | "
              f"{chw.MOTOR_MICROPASOS} micropasos | "
              f"v={chw.MOTOR_VELOCIDAD} a={chw.MOTOR_ACELERACION}")

        # --------------------------------------------------------- HOMING
        _titulo("3/5  HOMING contra el encoder absoluto")
        desvio_eje = geometria.diferencia_angular(
            X, chw.ENCODER_GRADOS_EN_MOTOR_CERO)
        correccion = chw.HOMING_SENTIDO * desvio_eje * chw.RELACION_TRANSMISION
        print(f"  encoder marca {X:.2f}, el cero del motor es "
              f"{chw.ENCODER_GRADOS_EN_MOTOR_CERO:.2f}")
        print(f"  desvio {desvio_eje:+.2f} deg de eje -> mover "
              f"{correccion:+.2f} deg de motor")

        if abs(correccion) > 180.0:
            print("  !! correccion mayor a media vuelta: casi seguro el signo "
                  "de HOMING_SENTIDO esta al reves. NO se mueve.")
        else:
            sis.motor.habilitar()
            sis.motor.mover_grados(correccion)
            if not sis.motor.esperar_fin(chw.MOTOR_TIMEOUT_MOV_S):
                print("  !! timeout esperando el fin del movimiento")

            time.sleep(0.2)
            _, X2, _ = sis.encoder.leer_angulo()
            error = geometria.diferencia_angular(
                X2, chw.ENCODER_GRADOS_EN_MOTOR_CERO)
            print(f"  encoder despues del homing: {X2:.2f} deg "
                  f"(error {error:+.2f})")
            if abs(error) > chw.TOLERANCIA_HOMING_DEG:
                print(f"  !! error mayor a {chw.TOLERANCIA_HOMING_DEG} deg. "
                      f"Puede ser backlash, RELACION_TRANSMISION mal, o pasos "
                      f"perdidos por corriente insuficiente.")

            sis.motor.zero()
            print("  cero del motor fijado en la posicion actual")

    # ------------------------------------------------------------------ GOPRO
    _titulo("4/5  GOPRO")
    if args.sin_gopro or not chw.GOPRO_HABILITADA:
        print("  saltada")
    else:
        sis.gopro = GoPro(chw.GOPRO_SSID, chw.GOPRO_PASSWORD, chw.GOPRO_IP)
        if sis.gopro.init():
            est = sis.gopro.estado()
            print(f"  lista | bateria {est.get('bateria')}% | "
                  f"espacio {est.get('espacio')}")
            if isinstance(est.get("bateria"), int) and est["bateria"] < 20:
                print("  !! bateria baja para un partido entero")
        else:
            print("  !! no se pudo inicializar. Se sigue sin GoPro.")
            sis.gopro = None

    # --------------------------------------------------------- MODELO Y CAMARAS
    _titulo("5/5  MODELO Y CAMARAS")
    from inferencia import HailoInference     # antes: from hailo_inference import ...

    sis.hailo = HailoInference(config.HEF_PATH)
    print(sis.hailo.describe())
    sis.model_hw = pipeline.get_model_hw(sis.hailo)
    clase = getattr(chw, "CLASE_OBJETIVO", None)
    if clase is not None:
        print(f"  clase objetivo: {clase} "
              f"({getattr(chw, 'NOMBRE_CLASE', '?')}) -- modelo generico COCO")
        print(f"  esperá confianzas mas bajas y falsos positivos redondos "
              f"(cabezas, carteles)")
    print(f"  entrada del modelo: {sis.model_hw[1]}x{sis.model_hw[0]} (ancho x alto)")

    sis.camaras = GestorCamaras(args.ancho, args.alto, args.fps)
    frame, _ = sis.camaras.leer(args.camara)
    h, w = frame.shape[:2]

    completo = getattr(chw, "SEARCH_FULL", False)
    if completo:
        esc = min(sis.model_hw[1] / w, sis.model_hw[0] / h)
        print(f"  frame {w}x{h} | FRAME ENTERO con letterbox -> escala "
              f"{esc:.4f}, 1 inferencia")
    else:
        tiles = pipeline.search_tiles((h, w), sis.model_hw)
        esc = sis.model_hw[1] / tiles[0][2]
        print(f"  frame {w}x{h} | mosaico: {len(tiles)} tiles de "
              f"{tiles[0][2]}x{tiles[0][3]} -> escala {esc:.4f}")

    # Cuantos pixeles le van a quedar a la pelota. Es la unica pregunta que
    # decide si el modelo tiene alguna chance: si el objeto queda por debajo
    # de ~15 px, no hay umbral ni preprocesado que lo salve.
    diam = getattr(chw, "DIAMETRO_PELOTA_M", 0.21)
    dist = getattr(chw, "DISTANCIA_PRUEBA_M", 2.5)
    fov = chw.CAM_FOV.get(args.camara, 100.0)
    ang = 2.0 * math.degrees(math.atan((diam / 2.0) / max(dist, 0.1)))
    px_nativo = ang * (w / fov)
    px_modelo = px_nativo * esc
    print(f"  pelota de {diam * 100:.0f} cm a {dist:.1f} m: {px_nativo:.0f} px "
          f"nativos -> {px_modelo:.0f} px en la entrada del modelo")
    if px_modelo < 15:
        print("  !! menos de 15 px: el modelo no la va a ver. Acercá la "
              "pelota, o pasá a mosaico (SEARCH_FULL = False).")
    elif px_modelo > 200:
        print("  (muy grande; si te da falsos negativos, puede estar saliendo "
              "del rango de tamanos con el que se entreno COCO)")

    # Warmup: la primera inferencia paga costos de setup que ensucian la
    # primera medicion de tiempo.
    detectar_pelota(frame, sis.hailo, sis.model_hw)

    sis.selector = SelectorCamara(args.camara)
    os.makedirs(chw.DIR_DEBUG, exist_ok=True)
    print(f"  salidas de debug en {chw.DIR_DEBUG}")
    return sis


# =============================================================================
# Loop de debug
# =============================================================================

AYUDA = """
comandos en el ultimo ENTER:
    (vacio) siguiente ciclo con la misma camara
    0 / 1   forzar camara
    c       alternar camara
    a       aceptar la camara que sugiere la histeresis
    s       proximo ciclo sin mover el motor
    q       salir
"""


def loop_debug(sis: Sistema, args) -> None:
    import cv2

    camara = args.camara
    ciclo = 0
    saltear_motor = args.sin_motor
    print(AYUDA)

    while True:
        ciclo += 1
        print()
        print("-" * 68)
        print(f"CICLO {ciclo}   camara {camara}")

        if not args.auto:
            input("[ENTER] sacar foto e inferir > ")

        # ---------------------------------------------------- 1. foto + modelo
        frame, info = sis.camaras.leer(camara)
        det, gx, gy, tile, ms = detectar_pelota(frame, sis.hailo, sis.model_hw)
        t_ahora = time.perf_counter()

        if det is None:
            angulo = None
            modo_in = "frame entero" if getattr(chw, "SEARCH_FULL", False) else "mosaico"
            print(f"  sin deteccion  ({ms:.0f} ms, {modo_in})")
        else:
            angulo = geometria.pixel_a_angulo(gx, frame.shape[1], camara)
            print(f"  {getattr(chw, 'NOMBRE_CLASE', 'pelota')}: px ({gx:.0f}, {gy:.0f})  conf {det.confidence:.3f}  "
                  f"caja {det.w:.0f}x{det.h:.0f}px  {'' if tile < 0 else f'tile {tile}  '}stride {det.stride}")
            print(f"  angulo: {angulo:.1f} deg   ({ms:.0f} ms de inferencia)")
            if (det.w + det.h) / 2 < 12:
                print("  !! caja menor a 12 px: en ese rango la confianza del "
                      "modelo se cae, tomá la posicion con pinzas.")

        # imagen anotada
        vis = anotar(frame, camara, det, gx, gy, angulo if angulo else 0.0)
        ruta = os.path.join(chw.DIR_DEBUG, f"c{ciclo:04d}_cam{camara}.jpg")
        cv2.imwrite(ruta, vis)
        print(f"  imagen: {ruta}")
        if det is not None and chw.DEBUG_GUARDAR_RECORTE:
            ruta_rec = os.path.join(chw.DIR_DEBUG, f"c{ciclo:04d}_recorte.jpg")
            guardar_recorte(frame, gx, gy, ruta_rec)
            print(f"  recorte: {ruta_rec}   <- ¿es la pelota?")

        # sugerencia de camara (histeresis). En debug solo se informa.
        conf = det.confidence if det is not None else None
        sugerida = sis.selector.actualizar(angulo, conf, t_ahora)
        if sugerida != camara:
            print(f"  histeresis SUGIERE camara {sugerida} ({sis.selector.motivo}). "
                  f"Aceptala con 'a'.")

        # ------------------------------------------------------- 2. motor
        if not args.auto:
            resp = input("[ENTER] mover motor y foto GoPro > ").strip().lower()
            if resp == "q":
                break

        if angulo is None:
            print("  motor: no se mueve (no hay angulo)")
        elif saltear_motor or sis.motor is None:
            g, _ = geometria.angulo_a_grados_motor(angulo)
            print(f"  motor: SALTEADO (habria ido a {g:+.2f} deg)")
        else:
            grados, clampeado = geometria.angulo_a_grados_motor(angulo)
            if clampeado:
                print(f"  !! {angulo:.1f} deg cae fuera del recorrido: se "
                      f"clampea a {grados:+.2f} deg de motor")
            print(f"  motor -> {grados:+.2f} deg (mundo {angulo:.1f})")
            sis.motor.ir_a_grados(grados)
            if not sis.motor.esperar_fin(chw.MOTOR_TIMEOUT_MOV_S):
                print("  !! timeout del movimiento")

            time.sleep(0.15)
            _, enc, _ = sis.encoder.leer_angulo()
            esperado = geometria.encoder_esperado(grados)
            error = geometria.diferencia_angular(enc, esperado)
            print(f"  encoder: {enc:.2f} deg | esperado {esperado:.2f} | "
                  f"error {error:+.2f}")
            if abs(error) > chw.TOLERANCIA_HOMING_DEG:
                print("  !! el motor no llego a donde dice: pasos perdidos, "
                      "backlash o RELACION_TRANSMISION mal.")

        saltear_motor = args.sin_motor

        # ------------------------------------------------------- 3. GoPro
        if sis.gopro is not None:
            ruta_gp = sis.gopro.foto(
                os.path.join(chw.DIR_DEBUG, f"c{ciclo:04d}_gopro.jpg"))
            print(f"  gopro: {ruta_gp}")

        # ------------------------------------------------- 4. siguiente ciclo
        if args.auto:
            time.sleep(args.pausa)
            continue

        cmd = input("[ENTER] siguiente ciclo (0/1/c/a/s/q) > ").strip().lower()
        if cmd == "q":
            break
        elif cmd in ("0", "1"):
            camara = int(cmd)
            sis.selector.forzar(camara, time.perf_counter())
        elif cmd == "c":
            camara = 1 - camara
            sis.selector.forzar(camara, time.perf_counter())
        elif cmd == "a":
            camara = sugerida
        elif cmd == "s":
            saltear_motor = True
        elif cmd in ("h", "?"):
            print(AYUDA)


# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camara", type=int, default=0, choices=[0, 1])
    ap.add_argument("--ancho", type=int, default=config.CAM_ANCHO)
    ap.add_argument("--alto", type=int, default=config.CAM_ALTO)
    ap.add_argument("--fps", type=float, default=config.CAM_FPS)
    ap.add_argument("--full", dest="full", action="store_true", default=None,
                    help="frame entero con letterbox (pelota cerca)")
    ap.add_argument("--tiles", dest="full", action="store_false",
                    help="mosaico de tiles (pelota lejos)")
    ap.add_argument("--sin-motor", action="store_true")
    ap.add_argument("--sin-gopro", action="store_true")
    ap.add_argument("--auto", action="store_true",
                    help="sin ENTER: cicla solo")
    ap.add_argument("--pausa", type=float, default=1.0,
                    help="segundos entre ciclos con --auto")
    args = ap.parse_args()

    # --full / --tiles pisan config_hw.SEARCH_FULL para esta corrida.
    if args.full is not None:
        chw.SEARCH_FULL = args.full

    sis = None
    try:
        sis = inicializar(args)
        _titulo("LISTO")
        loop_debug(sis, args)
    except KeyboardInterrupt:
        print("\nCtrl+C")
    finally:
        if sis is not None:
            sis.close()
        print("cerrado.")


if __name__ == "__main__":
    main()
