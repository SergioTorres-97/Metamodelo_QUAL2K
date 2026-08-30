"""
exportar.py
============
Exporta la BD SQLite (simulaciones_Q2K.db) a Excel con dos hojas:
  - 'simulaciones' : todos los datos fila por fila
  - 'resumen'      : estadísticas descriptivas por variable

Uso:
    python metamodelo/exportar.py
    python metamodelo/exportar.py --salida mi_archivo.xlsx
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from metamodelo.datos import BD_PATH, TABLA, FEATURES, TARGET

OUTPUT_DIR  = _ROOT / "resultados" / "chicamocha_t1_metamodelo"
EXCEL_PATH  = OUTPUT_DIR / "simulaciones_Q2K.xlsx"


# ---------------------------------------------------------------------------

def exportar(salida: str = str(EXCEL_PATH)):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[exportar] Leyendo BD: {BD_PATH}")
    conn = sqlite3.connect(BD_PATH)
    df   = pd.read_sql(f"SELECT * FROM {TABLA}", conn)
    conn.close()

    print(f"[exportar] Filas      : {len(df):,}")
    print(f"[exportar] Simulaciones: {df['sim_id'].nunique():,}")

    # ── Hoja 2: resumen estadístico ───────────────────────────────────────────
    cols_resumen = FEATURES + [TARGET]
    resumen = df[cols_resumen].describe().T.round(4)
    resumen.insert(0, "variable", resumen.index)
    resumen = resumen.reset_index(drop=True)

    # ── Escribir Excel ────────────────────────────────────────────────────────
    print(f"[exportar] Escribiendo: {salida}  (puede tardar unos segundos...)")

    with pd.ExcelWriter(salida, engine="openpyxl") as writer:

        # Hoja 1 — datos completos
        df.to_excel(writer, sheet_name="simulaciones", index=False)
        _formato_hoja(writer, "simulaciones", df)

        # Hoja 2 — resumen
        resumen.to_excel(writer, sheet_name="resumen", index=False)
        _formato_hoja(writer, "resumen", resumen)

    print(f"[exportar] ✅ Excel guardado: {salida}")


def _formato_hoja(writer: pd.ExcelWriter, nombre: str, df: pd.DataFrame):
    """Ajusta ancho de columnas y congela la primera fila."""
    ws = writer.sheets[nombre]

    # Congelar encabezado
    ws.freeze_panes = "A2"

    # Ajustar ancho de cada columna al contenido
    for col_idx, col_name in enumerate(df.columns, start=1):
        max_len = max(
            len(str(col_name)),
            df[col_name].astype(str).str.len().max() if len(df) > 0 else 0,
        )
        # Limitar a 30 caracteres para no hacer columnas gigantes
        ws.column_dimensions[
            ws.cell(row=1, column=col_idx).column_letter
        ].width = min(max_len + 2, 30)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Exporta simulaciones_Q2K.db a Excel.")
    p.add_argument(
        "--salida",
        type=str,
        default=str(EXCEL_PATH),
        help=f"Ruta del archivo Excel de salida (default: {EXCEL_PATH})",
    )
    args = p.parse_args()
    exportar(salida=args.salida)
