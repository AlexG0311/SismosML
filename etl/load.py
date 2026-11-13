
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
ARCHIVO_CSV = ROOT_DIR / "data" / "processed" / "sismos_limpios.csv"

COLUMNAS = [
    "id", "status", "rms", "agency", "gap", "place",
    "local_time", "utc_time", "magnitude", "magnitude_error",
    "nst", "mag_type", "latitude", "latitude_error",
    "longitude", "longitude_error", "depth", "depth_error"
]

TAMANO_LOTE = 1000
ID_PRUEBA = "gfz2024zjyp"


def verificar_fecha_csv():
    df = pd.read_csv(
        ARCHIVO_CSV,
        encoding="utf-8-sig",
        usecols=["id", "local_time", "utc_time"],
        low_memory=False,
    )

    registro = df.loc[df["id"] == ID_PRUEBA, ["id", "local_time", "utc_time"]]

    print("\n========== VERIFICACIÓN CSV ==========")
    if registro.empty:
        print(f"No se encontró el ID {ID_PRUEBA} en {ARCHIVO_CSV}")
        return

    print(registro.to_string(index=False))

    fecha = registro["utc_time"].iloc[0]
    fecha_convertida = pd.to_datetime(
        fecha,
        errors="coerce",
        utc=True,
    ).tz_localize(None)

    print("\n========== CONVERSIÓN PYTHON ==========")
    print("Fecha original:", fecha)
    print("Fecha convertida:", fecha_convertida)


def preparar_datos():
    if not ARCHIVO_CSV.exists():
        raise FileNotFoundError(f"No se encontró: {ARCHIVO_CSV}")

    df = pd.read_csv(
        ARCHIVO_CSV,
        encoding="utf-8-sig",
        low_memory=False
    )

    if list(df.columns) != COLUMNAS:
        raise ValueError(
            "Las columnas del CSV no coinciden con las esperadas."
        )

    # Convertir fechas
    df["utc_time"] = (
        pd.to_datetime(df["utc_time"], errors="coerce", utc=True)
        .dt.tz_localize(None)
    )

    # Conserva la hora mostrada en el CSV.
    df["local_time"] = (
        pd.to_datetime(df["local_time"], errors="coerce", utc=True)
        .dt.tz_localize(None)
    )

    # Convertir NaN y NaT a None para MySQL
    df = df.astype(object).where(pd.notna(df), None)

    # Mantener dtype=object para que pandas no reconstruya sus Timestamp.
    for columna in ["local_time", "utc_time"]:
        fechas = [
            valor.to_pydatetime()
            if isinstance(valor, pd.Timestamp)
            else valor
            for valor in df[columna]
        ]
        df[columna] = pd.Series(
            fechas,
            index=df.index,
            dtype=object,
        )

    return df


def generar_lotes(df):
    for inicio in range(0, len(df), TAMANO_LOTE):
        lote = df.iloc[inicio:inicio + TAMANO_LOTE]
        registros = list(lote.itertuples(index=False, name=None))
        yield inicio, registros


def verificar_antes_de_insertar():
    df = preparar_datos()
    indice_utc = COLUMNAS.index("utc_time")
    registro_prueba = df.loc[df["id"] == ID_PRUEBA, ["id", "local_time", "utc_time"]]

    print("\n========== ANTES DE INSERTAR ==========")
    if registro_prueba.empty:
        print(f"No se encontró el ID {ID_PRUEBA} en {ARCHIVO_CSV}")
        return

    print(registro_prueba.to_string(index=False))

    for inicio, registros in generar_lotes(df):
        for fila in registros:
            if fila[0] == ID_PRUEBA:
                print("\n========== PARÁMETROS DEL INSERT ==========")
                print("Lote:", inicio // TAMANO_LOTE + 1)
                print("ID:", fila[0])
                print("UTC:", fila[indice_utc])
                print("Tipo:", type(fila[indice_utc]))
                return


def cargar_sismos():
    from database import obtener_conexion

    print("\n========== CARGA DE DATOS A MYSQL ==========")
    df = preparar_datos()
    print(f"Registros encontrados: {len(df):,}")

    columnas_sql = ", ".join(
        f"`{columna}`" for columna in COLUMNAS
    )
    sql = f"""
    INSERT INTO `sismos` ({columnas_sql})
    VALUES ({", ".join(["%s"] * len(COLUMNAS))})
    """
    conexion = obtener_conexion()
    total_cargado = 0

    try:
        with conexion.cursor() as cursor:
            for _, registros in generar_lotes(df):
                cursor.executemany(sql, registros)
                conexion.commit()

                total_cargado += len(registros)

                print(
                    f"Insertados: {total_cargado:,} / {len(df):,}"
                )

        print("\n========== CARGA COMPLETADA ==========")
        print(f"Total insertado: {total_cargado:,}")


    

    except Exception:
        conexion.rollback()
        raise

    finally:
        conexion.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="Verifica el CSV o carga los sismos a MySQL.")
    modos = parser.add_mutually_exclusive_group()
    modos.add_argument("--cargar", action="store_true", help="Inserta los sismos en MySQL.")
    modos.add_argument(
        "--verificar-insert",
        action="store_true",
        help="Muestra los parámetros del INSERT para el ID de prueba.",
    )
    args = parser.parse_args()

    if args.cargar:
        cargar_sismos()
    elif args.verificar_insert:
        verificar_antes_de_insertar()
    else:
        verificar_fecha_csv()
