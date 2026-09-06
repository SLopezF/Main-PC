"""
inferencia.py

Unico lugar del proyecto que decide QUE backend de inferencia se usa. El
resto del codigo (main.py, main_partido.py y despues main_final.py) pide un
objeto de inferencia y no se entera de cual le tocó: los dos backends
cumplen la misma API.

    hailo_inference.HailoInference        Hailo-8 real, en la Raspberry Pi
    inferencia_ultralytics.Ultralytics... el .pt por Ultralytics, en la PC

COMO SE ELIGE
Por orden de prioridad:

    1. la variable de entorno BACKEND ('hailo' o 'ultralytics')
    2. config.BACKEND_INFERENCIA, si existe
    3. 'auto': se intenta importar hailo_platform; si no está, Ultralytics

'auto' es el default a proposito. En la Pi el SDK esta instalado y arranca
sola con Hailo; en la PC no esta y cae a Ultralytics sin tocar nada. Y si
algun dia queres forzar uno de los dos para comparar:

    BACKEND=ultralytics python3 main_partido.py --sin-motor --sin-gopro

POR QUE NO SE TOCA hailo_inference.py
Ese archivo esta probado contra el hardware real y esta congelado. Meterle
un fallback adentro seria mezclar dos responsabilidades en un modulo que no
se puede probar en la PC. La eleccion vive aca, que es codigo nuevo y sin
hardware.
"""

import os

import config


def backend_pedido() -> str:
    """'hailo', 'ultralytics' o 'auto'."""
    v = (os.environ.get("BACKEND")
         or getattr(config, "BACKEND_INFERENCIA", None)
         or "auto")
    v = str(v).strip().lower()
    if v not in ("hailo", "ultralytics", "auto"):
        print(f"[inferencia] backend '{v}' desconocido, uso 'auto'")
        return "auto"
    return v


def _hay_hailo() -> bool:
    try:
        import hailo_platform  # noqa: F401
        return True
    except Exception:
        return False


def abrir(ruta_modelo: str | None = None, **kwargs):
    """
    Devuelve el objeto de inferencia ya construido. `ruta_modelo` es la que
    entiende el backend elegido: el .hef para Hailo, el .pt para Ultralytics.
    Si se pasa un .hef y el backend es Ultralytics, el wrapper lo ignora y
    usa el .pt configurado, avisando.
    """
    elegido = backend_pedido()

    if elegido == "auto":
        elegido = "hailo" if _hay_hailo() else "ultralytics"
        print(f"[inferencia] auto -> {elegido}")

    if elegido == "hailo":
        from hailo_inference import HailoInference as Backend
        return Backend(ruta_modelo or config.HEF_PATH, **kwargs)

    from inferencia_ultralytics import UltralyticsInference as Backend
    return Backend(ruta_modelo, **kwargs)


# Alias con el nombre viejo, para que reemplazar
#     from hailo_inference import HailoInference
# por
#     from inferencia import HailoInference
# sea el unico cambio en los mains. Es una funcion, no una clase, pero se
# construye igual y devuelve un objeto con la misma API.
def HailoInference(ruta_modelo: str | None = None, **kwargs):
    return abrir(ruta_modelo, **kwargs)


if __name__ == "__main__":
    print(f"backend pedido : {backend_pedido()}")
    print(f"hailo_platform : {'si' if _hay_hailo() else 'no'}")
    obj = abrir()
    print(obj.describe())
    obj.close()
