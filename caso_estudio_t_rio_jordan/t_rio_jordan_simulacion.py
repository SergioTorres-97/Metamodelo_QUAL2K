"""
t_rio_jordan_simulacion.py
============================
Corre la simulación QUAL2K del Río Jordán — Tramo T1
(CABECERA -> PLAYA ARRIBA) con los valores calibrados.

Uso:
    python caso_estudio_t_rio_jordan/t_rio_jordan_simulacion.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.run_from_json import run_simulacion

JSON = str(_ROOT / "caso_estudio_t_rio_jordan" / "t_rio_jordan_simulacion.json")

if __name__ == "__main__":
    run_simulacion(JSON)
