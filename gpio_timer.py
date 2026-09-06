"""
gpio_timer.py

Control de los dos pines GPIO usados para medir externamente (con
osciloscopio) el tiempo de procesamiento por frame:

- GPIO_PIN_TIMING: se enciende al tomar el frame, se apaga al tener
  la coordenada final. Mide la latencia end-to-end del frame.
- GPIO_PIN_MODE:   nivel lógico que indica en qué estado (SEARCH=LOW,
  TRACK=HIGH) se procesó el frame que se está midiendo, para poder
  separar las dos poblaciones de latencia en el análisis posterior.

Uso típico en main.py:

    with gpio_timer.FrameTimer(mode=state.mode):
        ... preprocesar, inferir, postprocesar ...

El pin de timing se levanta al entrar al `with` y se baja al salir
(incluso si hay una excepción en el medio, gracias al context manager).
El pin de modo se fija una sola vez al entrar, antes de levantar el
pin de timing, para que quede estable durante toda la medición.

Si `gpiozero` no está disponible (por ejemplo, desarrollando en una
máquina que no es la Raspberry Pi), se usa automáticamente una
implementación simulada que loguea las transiciones por consola en
vez de tocar hardware real, para poder desarrollar y testear el resto
del sistema sin necesidad de tener la Pi a mano.
"""

from state_machine import Mode

import config

try:
    from gpiozero import DigitalOutputDevice
    _GPIOZERO_AVAILABLE = True
except Exception:
    _GPIOZERO_AVAILABLE = False


class _RealPins:
    """Wrapper sobre gpiozero.DigitalOutputDevice para los dos pines."""

    def __init__(self, pin_timing: int, pin_mode: int):
        self._timing = DigitalOutputDevice(pin_timing, initial_value=False)
        self._mode = DigitalOutputDevice(pin_mode, initial_value=False)

    def timing_on(self) -> None:
        self._timing.on()

    def timing_off(self) -> None:
        self._timing.off()

    def set_mode(self, mode: Mode) -> None:
        if mode == Mode.TRACK:
            self._mode.on()
        else:
            self._mode.off()

    def close(self) -> None:
        self._timing.close()
        self._mode.close()


class _SimulatedPins:
    """
    Reemplazo sin hardware real, para desarrollar/testear fuera de la Pi.
    Loguea las transiciones en vez de tocar pines físicos.
    """

    def __init__(self, pin_timing: int, pin_mode: int):
        self._pin_timing = pin_timing
        self._pin_mode = pin_mode
        print(
            f"[gpio_timer] gpiozero no disponible: usando pines simulados "
            f"(timing=GPIO{pin_timing}, mode=GPIO{pin_mode})"
        )

    def timing_on(self) -> None:
        print(f"[gpio_timer] GPIO{self._pin_timing} (timing) -> ON")

    def timing_off(self) -> None:
        print(f"[gpio_timer] GPIO{self._pin_timing} (timing) -> OFF")

    def set_mode(self, mode: Mode) -> None:
        level = "HIGH (TRACK)" if mode == Mode.TRACK else "LOW (SEARCH)"
        print(f"[gpio_timer] GPIO{self._pin_mode} (mode) -> {level}")

    def close(self) -> None:
        pass


# Instancia única de pines, creada de forma perezosa (lazy) la primera
# vez que se usa FrameTimer, para no tocar hardware al solo importar
# este módulo (útil para tests que importan sin necesitar GPIO real).
_pins = None


def _get_pins():
    global _pins
    if _pins is None:
        if _GPIOZERO_AVAILABLE:
            _pins = _RealPins(config.GPIO_PIN_TIMING, config.GPIO_PIN_MODE)
        else:
            _pins = _SimulatedPins(config.GPIO_PIN_TIMING, config.GPIO_PIN_MODE)
    return _pins


class FrameTimer:
    """
    Context manager que rodea el bloque de procesamiento de un frame
    (desde que se toma hasta que se tiene la coordenada final).

    Ejemplo:
        with FrameTimer(mode=state.mode):
            preprocessed = ...
            raw_output = hailo.infer(preprocessed)
            detection = postprocess.process(raw_output)
    """

    def __init__(self, mode: Mode):
        self._mode = mode

    def __enter__(self):
        pins = _get_pins()
        pins.set_mode(self._mode)
        pins.timing_on()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pins = _get_pins()
        pins.timing_off()
        # No suprimimos la excepción (si la hubo): el pin ya se bajó,
        # y la excepción sigue propagándose normalmente.
        return False


def close() -> None:
    """
    Libera los pines GPIO. Llamar al finalizar el programa
    (ej. en un bloque finally de main.py), no es estrictamente
    necesario pero es buena práctica para no dejar el pin "colgado".
    """
    global _pins
    if _pins is not None:
        _pins.close()
        _pins = None
