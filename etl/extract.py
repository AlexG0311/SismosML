
from pathlib import Path
import pandas as pd


# Directorio principal del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent

# Ruta del CSV original
CSV_FILE = ROOT_DIR / "data" / "raw" / "sismos.csv"


def extraer_sismos():

    print("\n========== EXTRACCIÓN ==========")

    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {CSV_FILE}"
        )

    df = pd.read_csv(
        CSV_FILE,
        encoding="utf-8-sig",
        low_memory=False
    )

    print("Extracción completada correctamente.")
    print(f"Archivo: {CSV_FILE.name}")
    print(f"Registros: {len(df):,}")
    print(f"Columnas: {len(df.columns)}")

    return df


if __name__ == "__main__":

    datos = extraer_sismos()

    print("\nPrimeros registros:")
    print(datos.head())

    print("\nDimensiones:")
    print(datos.shape)
