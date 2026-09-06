"""
hailo_inference.py

Wrapper standalone (sin GStreamer) sobre HailoRT, basado en el patrón
de hailo-apps/standalone_apps/object_detection, adaptado para:

  - Un único modelo (YOLO26n, NMS-free, una clase de interés).
  - Inferencia SÍNCRONA: se llama infer(frame) y se bloquea hasta
    tener el resultado, en vez de usar la API asíncrona con callbacks.
    Esto es intencional: para medir latencia por frame de forma limpia,
    sin colas ni pipelining de por medio, que es lo que se está midiendo
    con el pin de GPIO.

El costo de configuración (cargar el .hef, crear el device, armar los
streams) se paga una sola vez en el constructor, no en cada frame.

IMPORTANTE -- infer() devuelve TODAS las salidas.
El .hef de YOLO26n expone SEIS vstreams (tres pares caja/clase, uno por
escala). La versión anterior de este archivo devolvía solamente
`next(iter(results))`, o sea una salida arbitraria de las seis: el orden
del dict que entrega HailoRT no está garantizado, así que en una corrida
devolvía el mapa de clase de stride 8 y en la siguiente el de caja de
stride 16. Eso hacía imposible cualquier post-proceso. Ahora se devuelve
el dict completo y postprocess.py empareja las ramas por shape.

NOTA: este módulo requiere el SDK de HailoRT instalado (el paquete
`hailo_platform`) y hardware Hailo-8 real conectado. No se puede
probar fuera de la Raspberry Pi con el acelerador conectado.
"""

import numpy as np

from hailo_platform import (
    HEF,
    ConfigureParams,
    FormatType,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice,
)

import config


class HailoInference:
    """
    Encapsula la configuración de HailoRT para un único .hef y expone
    infer() como llamada síncrona, bloqueante, un frame a la vez.
    """

    def __init__(self, hef_path: str = config.HEF_PATH):
        self._hef = HEF(hef_path)

        self._target = VDevice()

        configure_params = ConfigureParams.create_from_hef(
            hef=self._hef, interface=HailoStreamInterface.PCIe
        )
        network_group = self._target.configure(self._hef, configure_params)[0]
        self._network_group = network_group
        self._network_group_params = network_group.create_params()

        input_vstream_params = InputVStreamParams.make(
            network_group, format_type=FormatType.UINT8
        )
        # FLOAT32 en la salida: HailoRT descuantiza los UINT16 del HEF
        # y entrega los logits ya en punto flotante.
        output_vstream_params = OutputVStreamParams.make(
            network_group, format_type=FormatType.FLOAT32
        )

        self._input_vstream_params = input_vstream_params
        self._output_vstream_params = output_vstream_params

        input_info = self._hef.get_input_vstream_infos()[0]
        self._input_name = input_info.name
        self._input_shape = tuple(input_info.shape)  # (H, W, C)

        self._output_infos = self._hef.get_output_vstream_infos()
        self._output_shapes = {vi.name: tuple(vi.shape) for vi in self._output_infos}

        # InferVStreams se abre una sola vez y se reutiliza en cada
        # llamada a infer(), para no pagar el costo de setup por frame.
        self._infer_pipeline = InferVStreams(
            network_group, input_vstream_params, output_vstream_params
        )
        self._infer_pipeline.__enter__()

        self._network_group_activation = network_group.activate(
            self._network_group_params
        )
        self._network_group_activation.__enter__()

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Forma esperada de entrada (H, W, C), tal como la reporta el .hef."""
        return self._input_shape

    @property
    def output_shapes(self) -> dict[str, tuple]:
        """{nombre_vstream: shape} de todas las salidas del .hef."""
        return dict(self._output_shapes)

    def describe(self) -> str:
        """Resumen legible de la estructura del modelo, para logs de arranque."""
        lineas = [f"IN  {self._input_name}: {self._input_shape}"]
        for nombre, shape in self._output_shapes.items():
            rama = "caja " if shape[-1] == 4 else "clase"
            lineas.append(f"OUT {nombre}: {shape}  ({rama})")
        return "\n".join(lineas)

    def infer(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        """
        Corre inferencia síncrona sobre un único frame ya preprocesado
        (mismo tamaño que self.input_shape, dtype uint8, canales en RGB).

        Devuelve un dict {nombre_vstream: ndarray} con TODAS las salidas
        del modelo, ya sin la dimensión de batch. El decode a detecciones
        lo hace postprocess.process().
        """
        if tuple(frame.shape) != self._input_shape:
            raise ValueError(
                f"Frame de entrada con shape {frame.shape}, "
                f"se esperaba {self._input_shape}"
            )

        input_data = {self._input_name: np.expand_dims(frame, axis=0)}
        results = self._infer_pipeline.infer(input_data)

        # Sacar la dimensión de batch de cada salida.
        return {nombre: arr[0] for nombre, arr in results.items()}

    def close(self) -> None:
        """Libera los recursos de HailoRT. Llamar al finalizar el programa."""
        self._network_group_activation.__exit__(None, None, None)
        self._infer_pipeline.__exit__(None, None, None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
