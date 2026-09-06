# modelos/

Los `.pt` **si** se versionan en git: son la unica forma de reproducir un
resultado de la tesis meses despues. Los `.hef` no, porque se compilan a
partir del `.pt` y solo sirven en la Pi.

## Reglas

1. **Nombre versionado, nunca sobrescribir.** `..._300ep.pt`, no `modelo.pt`.
   En git el costo de espacio es el mismo (cada version queda en el historial
   igual), pero asi podes volver a una anterior por nombre.
2. **Borra los experimentos fallidos ANTES del primer commit.** Una vez
   commiteado, un `.pt` queda en el historial para siempre aunque lo borres
   despues, y el `git clone` se lo sigue bajando.
3. **Anota aca cada modelo nuevo**, con que se entreno y que dio. Sin esto, en
   dos meses hay cuatro `.pt` y ninguna forma de saber cual es cual.

## Cual esta en uso

Lo define `config.MODELO_PT`. El codigo no busca en esta carpeta por su
cuenta: o pones la ruta completa en `config.py`, o la pasas por entorno:

    MODELO_PT=modelos/run_yolo26n_..._300ep.pt python3 replay.py --carpeta ...

## Inventario

| archivo | entrada | clases | epocas | notas |
|---|---|---|---|---|
| `run_yolo26n_sesiones_1152x640px_300ep.pt` | 1152x640 | 1 (`ball`) | 300 | **EN USO.** Ver medicion abajo |

### run_yolo26n_sesiones_1152x640px_300ep.pt

Medido con `replay.py` el 2026-09-06 (detalle completo en `ESTADO.md`):

- recall con `--solo-search`: 85.1% sobre 315 imagenes de `Soleado`,
  confianza p50 0.797
- tracking sin Kalman: 84.8% de tiempo en TRACK, 79.7% de detecciones
  aceptadas

Pendiente: no se separaron falsos positivos. La confianza minima aceptada es
0.107, apenas por encima de `CONF_MIN_DETECTION = 0.1`, asi que ese 85.1% es
un techo y no el recall real.

> Nota: el nombre no coincide con los dos `.pt` que figuran en el briefing
> (`..._CROP_...200ep` y `..._DOWNSCALE_...200ep`). El de DOWNSCALE se
> descarto junto con `SEARCH_FULL`. Confirmar si este es el sucesor del CROP.
