import subprocess
import os


def ejecutar_simulacion(exe_path: str) -> None:
    """
    Ejecuta el ejecutable FORTRAN de QUAL2K.

    Args:
        exe_path: Ruta del ejecutable q2kfortran2_12.exe
    """
    if not os.path.exists(exe_path):
        raise FileNotFoundError(f"No se encontró el ejecutable: {exe_path}")

    folder = os.path.dirname(exe_path)
    subprocess.run([exe_path], cwd=folder, check=True)
