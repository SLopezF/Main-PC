"""
sectores.py

Decide CUANDO mover el motor y ADONDE. Es la pieza que hace que el video se
parezca a una transmision y no a una camara de seguridad siguiendo un bicho.

LA IDEA
El motor NO apunta a la pelota. El rango util (20..160 grados) se parte en 7
sectores de 20 grados, y el motor va SIEMPRE al centro del sector donde esta
la jugada. Con un FOV de mas de 100 grados en la GoPro desde 3 m de altura, un
error de medio sector (10 grados) no saca la pelota del cuadro; en cambio,
una camara que persigue el angulo exacto marea y se ve amateur.

    bordes  = [20, 40, 60, 80, 100, 120, 140, 160]
    centros = [30, 50, 70, 90, 110, 130, 150]

LOS FRENOS
  1. MARGEN (histeresis). Para pasar del sector i al i+1 hay que superar el
     borde por HISTERESIS_SECTOR_DEG. Es el freno principal: sin el, una
     pelota parada sobre un limite hace saltar el motor de ida y vuelta.
  2. PERMANENCIA de FRAMES_PERMANENCIA frames (arranque: 2). Practicamente
     "reaccionar ya" —50 ms a 40 fps— pero mata el caso de un unico frame con
     una deteccion espuria que cruzo el margen.
  3. PISO DE TIEMPO entre movimientos, que limita la FRECUENCIA sin retrasar
     la primera reaccion.

QUE SE SACO, Y POR QUE
Habia una permanencia de 600 ms y una excepcion por velocidad angular
(OMEGA_RAPIDA) para no perderse un pelotazo. Los dos se sacaron.

La espera de 600 ms era contraproducente justo en el caso que mas importa, y
la excepcion que la cubria dependia de una estimacion de omega que es ruidosa:
una derivada numerica sobre una posicion ruidosa. O sea que el pelotazo, el
caso critico, quedaba a merced del ruido de la deteccion.

Lo que reemplaza a todo eso ya existia aguas arriba: el gate de plausibilidad
y el Kalman del tracker filtran las detecciones malas ANTES de que lleguen
aca. Lo unico que hace falta despues es el margen en grados y dos frames de
confirmacion.

SIN HARDWARE
Python puro: no sabe de motores, camaras ni pixeles. Entra (angulo, omega, t)
y sale una decision. Se testea con angulos sinteticos, sin video ni fierro.
"""

import bisect
from dataclasses import dataclass

import config_hw as chw


def bordes() -> list[float]:
    """[20, 40, ..., 160]: los N_SECTORES + 1 limites."""
    n = int(chw.N_SECTORES)
    desde, hasta = float(chw.SECTOR_DESDE), float(chw.SECTOR_HASTA)
    paso = (hasta - desde) / n
    return [desde + i * paso for i in range(n + 1)]


def centros() -> list[float]:
    """[30, 50, ..., 150]: adonde apunta el motor en cada sector."""
    b = bordes()
    return [(b[i] + b[i + 1]) / 2.0 for i in range(len(b) - 1)]


def sector_de(angulo: float) -> int:
    """
    Sector que contiene el angulo, SIN histeresis. Fuera de rango, el sector
    del extremo: el motor no puede apuntar mas alla de todos modos.
    """
    b = bordes()
    if angulo <= b[0]:
        return 0
    if angulo >= b[-1]:
        return len(b) - 2
    return max(0, min(bisect.bisect_right(b, angulo) - 1, len(b) - 2))


@dataclass
class ResultadoSector:
    sector: int
    cambio: bool
    angulo_objetivo: float          # el CENTRO del sector, no el angulo real
    motivo: str


class Sectorizador:
    """
    Schmitt trigger sobre los bordes de sector.

    Se llama una vez por deteccion aceptada. Devuelve el sector vigente y, si
    corresponde, la orden de mover el motor al centro de ese sector.

    No toca hardware: quien llama decide que hacer con angulo_objetivo.
    """

    def __init__(self, sector_inicial: int | None = None):
        cen = centros()
        if sector_inicial is None:
            # Arranca mirando al frente (90 grados), que es el centro del
            # sector del medio.
            sector_inicial = sector_de(90.0)
        self.sector = int(sector_inicial)
        self.angulo_objetivo = cen[self.sector]
        self.motivo = "inicial"

        self._voto_a = None          # a que sector se esta votando
        self._votos = 0              # cuantos frames seguidos lleva ese voto
        self._t_ultimo_cambio = -1e9

    # -------------------------------------------------------------- internos
    def _candidato(self, angulo: float) -> int:
        """
        Sector al que corresponde el angulo EXIGIENDO la histeresis: el sector
        actual se ensancha HISTERESIS_SECTOR_DEG hacia cada lado, asi que
        mientras el angulo caiga en esa zona ampliada no hay candidato nuevo.
        """
        b = bordes()
        h = float(chw.HISTERESIS_SECTOR_DEG)
        if (b[self.sector] - h) <= angulo <= (b[self.sector + 1] + h):
            return self.sector
        return sector_de(angulo)

    def _puede_mover(self, t_s: float) -> bool:
        return ((t_s - self._t_ultimo_cambio) * 1000.0
                >= float(chw.MS_MINIMO_ENTRE_MOVIMIENTOS))

    def _mover_a(self, sector: int, t_s: float, motivo: str) -> ResultadoSector:
        self.sector = int(sector)
        self.angulo_objetivo = centros()[self.sector]
        self._t_ultimo_cambio = t_s
        self._voto_a, self._votos = None, 0
        self.motivo = motivo
        return ResultadoSector(self.sector, True, self.angulo_objetivo, motivo)

    def _quedarse(self, motivo: str) -> ResultadoSector:
        self.motivo = motivo
        return ResultadoSector(self.sector, False, self.angulo_objetivo, motivo)

    # --------------------------------------------------------- actualizacion
    def actualizar(self, angulo: float, omega: float = 0.0,
                   t_s: float = 0.0) -> ResultadoSector:
        """
        `angulo` en grados del mundo, `t_s` en segundos.

        `omega` YA NO SE USA para decidir: queda en la firma porque el CSV la
        sigue escribiendo (es util para el analisis) y para no romper a quien
        llame con tres argumentos.
        """
        candidato = self._candidato(angulo)

        if candidato == self.sector:
            self._voto_a, self._votos = None, 0
            return self._quedarse("centrado")

        # --- permanencia en FRAMES: hay que votar al MISMO sector
        if self._voto_a != candidato:
            self._voto_a, self._votos = candidato, 1
        else:
            self._votos += 1

        faltan = int(chw.FRAMES_PERMANENCIA) - self._votos
        if faltan > 0:
            return self._quedarse(
                f"votando sector {candidato} ({self._votos}/"
                f"{chw.FRAMES_PERMANENCIA})")

        if not self._puede_mover(t_s):
            return self._quedarse("movimiento reciente, se espera")

        return self._mover_a(candidato, t_s,
                             f"margen superado ({self._votos} frames)")

    def forzar(self, angulo: float, t_s: float) -> ResultadoSector:
        """
        Manda el motor al sector de `angulo` sin pedir permiso. Lo usa el
        vuelta-al-centro tras S_SEARCH_A_CENTRO segundos sin ver nada.
        """
        return self._mover_a(sector_de(angulo), t_s, "forzado")


# =============================================================================
# Donde caen los bordes de sector en cada camara
# =============================================================================

def bordes_en_pixeles(camara: int, ancho_img: int) -> list[tuple[float, float]]:
    """
    [(angulo, pixel_x)] de cada borde de sector VISIBLE en esta camara.

    Sirve para dibujar las divisiones sobre la imagen y ver, antes de calibrar
    nada, si caen donde uno espera. Los bordes que quedan fuera del campo de
    vision de la camara se descartan: no tiene sentido dibujar una linea del
    sector 0 en la camara que mira el otro lado de la cancha.

    OJO: hasta P8 esto usa el modelo de lente NOMINAL (FOV lineal), que tiene
    error de barril en los bordes. Con CAL_PIXEL_ANGULO cargado,
    geometria.pixel_a_angulo pasa a interpolar la tabla medida, pero
    angulo_a_pixel sigue siendo nominal: es solo para dibujar.
    """
    import geometria

    salida = []
    for a in bordes():
        px = geometria.angulo_a_pixel(a, ancho_img, camara)
        if 0 <= px <= ancho_img:
            salida.append((a, px))
    return salida


def sectores_visibles(camara: int) -> list[int]:
    """Indices de los sectores que esta camara alcanza a ver, aunque sea en parte."""
    import geometria

    lo, hi = geometria.cobertura(camara)
    b = bordes()
    return [i for i in range(len(b) - 1) if b[i + 1] > lo and b[i] < hi]


# --------------------------------------------------------------------------- #
# CLI: describe la particion, sin dependencias
# --------------------------------------------------------------------------- #

def main() -> int:
    b, c = bordes(), centros()
    h = float(chw.HISTERESIS_SECTOR_DEG)

    print(f"{chw.N_SECTORES} sectores de "
          f"{(chw.SECTOR_HASTA - chw.SECTOR_DESDE) / chw.N_SECTORES:.0f} grados "
          f"sobre {chw.SECTOR_DESDE:.0f}..{chw.SECTOR_HASTA:.0f}")
    print(f"bordes   {[f'{v:.0f}' for v in b]}")
    print(f"centros  {[f'{v:.0f}' for v in c]}")
    print()
    print(f"histeresis            {h:.1f} grados")
    print(f"permanencia           {chw.FRAMES_PERMANENCIA} frames "
          f"({chw.FRAMES_PERMANENCIA / 40.0 * 1000:.0f} ms a 40 fps)")
    print(f"minimo entre movs     {chw.MS_MINIMO_ENTRE_MOVIMIENTOS:.0f} ms")
    print("regimen rapido/lento  SACADO (dependia de una omega ruidosa)")
    print()

    import geometria
    print("cobertura de cada camara (modelo nominal, sin calibrar):")
    for cam in chw.CAMARAS:
        lo, hi = geometria.cobertura(cam)
        vis = sectores_visibles(cam)
        print(f"  camara {cam}: {lo:6.1f} a {hi:6.1f} grados   "
              f"sectores {vis}")

    compartidos = (set(sectores_visibles(0)) & set(sectores_visibles(1)))
    print(f"  sectores en las DOS: {sorted(compartidos)}")
    print()

    ancho = int(getattr(__import__("config"), "CAM_ANCHO", 2304))
    for cam in chw.CAMARAS:
        print(f"bordes de sector en la camara {cam} ({ancho} px de ancho):")
        for a, px in bordes_en_pixeles(cam, ancho):
            print(f"  {a:5.0f} deg -> x = {px:7.1f} px")
        print()

    print("sector de cada angulo:")
    for a in range(0, 181, 10):
        print(f"  {a:3d} deg -> sector {sector_de(a)}  "
              f"(motor a {centros()[sector_de(a)]:.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
