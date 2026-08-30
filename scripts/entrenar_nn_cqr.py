"""
entrenar_nn_cqr.py
====================
Entrena el modelo de cuantiles (CQR) del metamodelo de DBO — ver
metamodelo/nn_trainer.py::entrenar_cqr. Reutiliza los hiperparametros ya
encontrados por Optuna para el modelo puntual (ver __main__ de nn_trainer.py)
como punto de partida; no repite la busqueda.

No sobrescribe ningun artefacto del modelo puntual (nn_dbo.pt, nn_conformal.json):
genera nn_dbo_cqr.pt, nn_cqr.json y las figuras en figuras_nn_cqr/, todo bajo
resultados/chicamocha_t1_metamodelo/.

Uso:
    python scripts/entrenar_nn_cqr.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from metamodelo.nn_trainer import entrenar_cqr

if __name__ == "__main__":
    entrenar_cqr(
        capas        = [32, 128],
        epochs       = 300,
        lr           = 0.0026070247583707684,
        batch_size   = 256,
        dropout      = 0.010292247147901223,
        weight_decay = 5.337032762603957e-06,
        test_size    = 0.20,
        val_size     = 0.10,
        paciencia    = 50,
        seed         = 42,
    )
