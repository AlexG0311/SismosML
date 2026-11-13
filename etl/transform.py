
from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. RUTAS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

ARCHIVO_ENTRADA = (
    ROOT_DIR
    / "data"
    / "processed"
    / "sismos_colombia.csv"
)

ARCHIVO_SALIDA = (
    ROOT_DIR
    / "data"
    / "processed"
    / "sismos_limpios.csv"
)


# ============================================================
# 2. COLUMNAS
# ============================================================

COLUMNAS_ESENCIALES = [
    "id",
    "utc_time",
    "magnitude",
    "latitude",
    "longitude",
    "depth"
]

COLUMNAS_NUMERICAS = [
    "rms",
    "gap",
    "magnitude",
    "magnitude_error",
    "nst",
    "latitude",
    "latitude_error",
    "longitude",
    "longitude_error",
    "depth",
    "depth_error"
]

COLUMNAS_TEXTO = [
    "id",
    "status",
    "agency",
    "place",
    "mag_type"
]


# ============================================================
# 3. FUNCIÓN DE TRANSFORMACIÓN
# ============================================================

def transformar_sismos():

    print("\n========== TRANSFORMACIÓN Y LIMPIEZA ==========")

    # --------------------------------------------------------
    # Paso 1. Leer dataset geográficamente seleccionado
    # --------------------------------------------------------

    if not ARCHIVO_ENTRADA.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {ARCHIVO_ENTRADA}"
        )

    df = pd.read_csv(
        ARCHIVO_ENTRADA,
        encoding="utf-8-sig",
        low_memory=False
    )

    total_inicial = len(df)

    print(f"Registros iniciales: {total_inicial:,}")

    # --------------------------------------------------------
    # Paso 2. Verificar columnas requeridas
    # --------------------------------------------------------

    columnas_faltantes = (
        set(COLUMNAS_ESENCIALES)
        - set(df.columns)
    )

    if columnas_faltantes:
        raise ValueError(
            f"Faltan columnas esenciales: {columnas_faltantes}"
        )

    # --------------------------------------------------------
    # Paso 3. Reemplazar espacios vacíos por NaN
    # --------------------------------------------------------

    df.replace(
        r"^\s*$",
        np.nan,
        regex=True,
        inplace=True
    )

    # --------------------------------------------------------
    # Paso 4. Normalizar textos
    # --------------------------------------------------------

    for columna in COLUMNAS_TEXTO:

        if columna in df.columns:

            df[columna] = (
                df[columna]
                .astype("string")
                .str.strip()
            )

    # --------------------------------------------------------
    # Paso 5. Convertir variables numéricas
    # --------------------------------------------------------

    for columna in COLUMNAS_NUMERICAS:

        if columna in df.columns:

            df[columna] = pd.to_numeric(
                df[columna],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Paso 6. Convertir fechas
    # --------------------------------------------------------

    df["local_time"] = pd.to_datetime(
    df["local_time"], errors="coerce"
    )

    # Hora universal: mantenerla en UTC
    df["utc_time"] = pd.to_datetime(
    df["utc_time"], errors="coerce", utc=True
    )

    # --------------------------------------------------------
    # Paso 7. Eliminar nulos en columnas esenciales
    # --------------------------------------------------------

    antes_nulos = len(df)

    df.dropna(
        subset=COLUMNAS_ESENCIALES,
        inplace=True
    )

    eliminados_nulos = antes_nulos - len(df)

    print(
        f"Registros eliminados por nulos esenciales: "
        f"{eliminados_nulos:,}"
    )

    # --------------------------------------------------------
    # Paso 8. Validar rangos de coordenadas
    # --------------------------------------------------------

    antes_coordenadas = len(df)

    coordenadas_validas = (
        df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
    )

    df = df[coordenadas_validas].copy()

    print(
        "Registros eliminados por coordenadas inválidas:",
        f"{antes_coordenadas - len(df):,}"
    )

    # --------------------------------------------------------
    # Paso 9. Eliminar IDs duplicados
    # --------------------------------------------------------

    antes_duplicados = len(df)

    df.drop_duplicates(
        subset=["id"],
        keep="first",
        inplace=True
    )

    print(
        "IDs duplicados eliminados:",
        f"{antes_duplicados - len(df):,}"
    )

    # --------------------------------------------------------
    # Paso 10. Revisar valores faltantes
    # --------------------------------------------------------

    print("\n========== VALORES NULOS RESTANTES ==========")

    print(
        df.isnull()
        .sum()
        .to_string()
    )

    # --------------------------------------------------------
    # Paso 11. Estadísticas finales
    # --------------------------------------------------------

    print("\n========== RESULTADO DE LA LIMPIEZA ==========")

    print(f"Registros iniciales: {total_inicial:,}")
    print(f"Registros finales: {len(df):,}")
    print(
        f"Registros eliminados: "
        f"{total_inicial - len(df):,}"
    )

    # --------------------------------------------------------
    # Paso 12. Guardar dataset limpio
    # --------------------------------------------------------

    ARCHIVO_SALIDA.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        ARCHIVO_SALIDA,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nDataset limpio guardado en: {ARCHIVO_SALIDA}")

    return df


# ============================================================
# 4. EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    datos_limpios = transformar_sismos()

    print("\nProceso ETL finalizado correctamente.")
