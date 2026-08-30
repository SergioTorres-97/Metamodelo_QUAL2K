"""
chicamocha_t1_simulacion.py
============================
Corre la simulación QUAL2K del Río Chicamocha — Tramo T1
(CABECERA -> PLAYA ABAJO) con los valores calibrados.

Uso:
    python examples/chicamocha_t1_simulacion.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.run_from_json import run_simulacion

JSON = str(_ROOT / "examples" / "chicamocha_t1_simulacion.json")

if __name__ == "__main__":
    run_simulacion(JSON)
