"""Test del wrapper con un Ultralytics FALSO, para validar el contrato."""
import sys
import types

import numpy as np


# --- ultralytics falso -------------------------------------------------------
class _T:
    def __init__(self, a): self._a = np.asarray(a)
    def cpu(self): return self
    def numpy(self): return self._a


class _Boxes:
    def __init__(self, xyxyn, conf, cls):
        self.xyxyn, self.conf, self.cls = _T(xyxyn), _T(conf), _T(cls)
    def __len__(self): return len(self.conf._a)


class _Res:
    def __init__(self, boxes): self.boxes = boxes


class YOLO:
    names = {0: "ball"}

    def __init__(self, ruta):
        self.ruta = ruta
        self.names = {0: "ball"}
        self.visto = None

    def predict(self, im, **kw):
        self.visto = (im, kw)
        # caja centrada en (0.5, 0.5) del tensor, de 0.1 x 0.1
        return [_Res(_Boxes([[0.45, 0.45, 0.55, 0.55]], [0.83], [0]))]


mod = types.ModuleType("ultralytics")
mod.YOLO = YOLO
sys.modules["ultralytics"] = mod

# --- config del modelo propio ------------------------------------------------
import config
import config_hw as chw
config.MODEL_INPUT_SIZE = (640, 1152)
chw.CLASE_OBJETIVO = None            # el modelo propio tiene una sola clase

import postprocess
postprocess._chw = chw

import inferencia_ultralytics as iu

open("/tmp/fake.pt", "w").close()

hailo = iu.UltralyticsInference("/tmp/fake.pt")
print(hailo.describe())
print()

assert hailo.input_shape == (640, 1152, 3), hailo.input_shape

# shape mala -> mismo error que el real
try:
    hailo.infer(np.zeros((640, 640, 3), np.uint8))
    raise AssertionError("deberia haber tirado ValueError")
except ValueError as e:
    print("shape mala rechazada:", str(e).splitlines()[0])

# frame RGB del tamano correcto
frame = np.zeros((640, 1152, 3), np.uint8)
frame[:, :, 0] = 200          # canal R
raw = hailo.infer(frame)
print("salidas:", {k: v.shape for k, v in raw.items()})

# el wrapper le tiene que haber dado BGR a ultralytics
im, kw = hailo._modelo.visto
assert im[0, 0, 2] == 200 and im[0, 0, 0] == 0, "no invirtio RGB->BGR"
print("RGB->BGR ok | imgsz pasado:", kw["imgsz"], "| conf:", round(kw["conf"], 3))

det = postprocess.process(raw, (640, 1152))
print("\npostprocess.process ->", det)
assert det is not None
assert abs(det.x - 576) < 1 and abs(det.y - 320) < 1, (det.x, det.y)
assert abs(det.w - 115.2) < 1 and abs(det.h - 64) < 1, (det.w, det.h)
assert abs(det.confidence - 0.83) < 1e-4

cands = postprocess.process_candidates(raw, (640, 1152), umbral=0.05, topk=3)
print("process_candidates ->", len(cands), "candidato(s)")
assert len(cands) == 1

# get_model_hw de main.py tiene que leerlo bien
import main as pipeline
assert pipeline.get_model_hw(hailo) == (640, 1152)
print("main.get_model_hw ->", pipeline.get_model_hw(hailo))

# CLASE_OBJETIVO fuera de rango: tiene que avisar
chw.CLASE_OBJETIVO = 32
print()
iu.UltralyticsInference("/tmp/fake.pt")

print("\nTODO OK")
