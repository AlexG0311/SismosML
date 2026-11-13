
from pathlib import Path
import re
import unicodedata

import geopandas as gpd
import pandas as pd


# ============================================================
# 1. RUTAS DEL PROYECTO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

ARCHIVO_GEOGRAFICO = (
    ROOT_DIR
    / "data"
    / "boundaries"
    / "land_eez"
    / "EEZ_land_union_v4_202410.shp"
)

ARCHIVO_PAISES = (
    ROOT_DIR
    / "data"
    / "boundaries"
    / "countries"
    / "ne_10m_admin_0_countries.shp"
)

CARPETA_PROCESADOS = ROOT_DIR / "data" / "processed"


# ============================================================
# 2. LOCALIDADES Y DEPARTAMENTOS COLOMBIANOS
# ============================================================

LUGARES_COLOMBIANOS = [
    # Departamentos
    "choco",
    "nariño",
    "la guajira",
    "antioquia",
    "sucre",
    "cordoba",
    "bolivar",
    "magdalena",
    "cesar",
    "atlantico",
    "santander",
    "norte de santander",
    "boyaca",
    "cundinamarca",
    "meta",
    "casanare",
    "arauca",
    "guaviare",
    "guainia",
    "vichada",
    "caqueta",
    "putumayo",
    "amazonas",
    "valle del cauca",
    "cauca",
    "tolima",
    "huila",
    "caldas",
    "risaralda",
    "quindio",
    "bogota",

    # Localidades identificadas en la auditoría
    "jurado",
    "acandi",
    "quibdo",
    "uribia",
    "cumbal",
    "puerto santander",
    "becerril",
    "cubara",
    "manaure balcon del cesar",
    "la jagua de ibirico",
    "agustin codazzi",
    "saravena",
    "unguia",
    "villa del rosario",
    "herran",
    "ragonvalia",
    "albania",
    "el molino",
    "maicao",
    "la dorada",
    "riosucio",
    "carlosama",
    "potosi",
    "ricaurte",
    "san andres de tumaco",
    "barrancas",
    "toledo",
    "arauquita",
    "archipielago de san andres",
]


# ============================================================
# 3. DESCRIPCIONES QUE DEBEN EXCLUIRSE
# ============================================================

PATRONES_EXCLUIDOS = [
    # Países vecinos
    r"venezuela",
    r"panama",
    r"ecuador",

    # Fronteras internacionales
    r"frontera",
    r"region fronteriza",
    r"border region",

    # Descripciones geográficas generales
    r"cerca de la costa",
    r"near coast",
    r"near north coast",
    r"norte colombia",
    r"northern colombia",

    # Océanos y mares
    r"oceano pacifico",
    r"mar caribe",

    # Otras referencias generales
    r"sur de panama",
    r"norte de panama",
]


# ============================================================
# 4. NORMALIZACIÓN DE TEXTO
# ============================================================

def normalizar_texto(texto):
    """
    Convierte el texto a minúsculas y elimina tildes
    para facilitar las comparaciones.
    """

    texto = str(texto).lower().strip()

    texto = unicodedata.normalize("NFD", texto)

    texto = "".join(
        caracter
        for caracter in texto
        if unicodedata.category(caracter) != "Mn"
    )

    return texto


# ============================================================
# 5. CONSTRUIR GEOMETRÍA DE ESTUDIO
# ============================================================

def seleccionar_geografia():

    print("\n========== SELECCIÓN GEOGRÁFICA ==========")

    union = gpd.read_file(ARCHIVO_GEOGRAFICO)
    paises = gpd.read_file(ARCHIVO_PAISES)

    # Territorio principal de Colombia en Marine Regions.
    colombia = union[
        (union["ISO_TER1"] == "COL")
        & (union["POL_TYPE"] == "Union EEZ and country")
        & (union["UNION"] == "Colombia")
    ]

    # Zonas de régimen conjunto relacionadas con Colombia.
    regimen_conjunto = union[
        (union["POL_TYPE"] == "Joint regime (EEZ)")
        & (
            (union["ISO_TER1"] == "COL")
            | (union["ISO_TER2"] == "COL")
            | (union["ISO_TER3"] == "COL")
        )
    ]

    # Geometría terrestre de Colombia de Natural Earth.
    colombia_terrestre = paises[
        paises["ADM0_A3"] == "COL"
    ]

    geometrias = (
        list(colombia.geometry)
        + list(regimen_conjunto.geometry)
        + list(colombia_terrestre.geometry)
    )

    area_estudio = gpd.GeoSeries(
        geometrias,
        crs="EPSG:4326"
    ).union_all()

    print("Geometría de estudio construida correctamente.")
    print(f"Tipo de geometría: {area_estudio.geom_type}")

    return area_estudio


# ============================================================
# 6. CLASIFICACIÓN DE LOS SISMOS
# ============================================================

def probar_filtro_sismico(df, area_estudio):

    print("\n========== CLASIFICACIÓN DE SISMOS ==========")

    df = df.copy()

    # Convertir coordenadas a valores numéricos.
    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    # Validar coordenadas geográficas.
    coordenadas_validas = (
        df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
    )

    df_validos = df[coordenadas_validas].copy()

    df_invalidos = df[~coordenadas_validas].copy()

    # Construir puntos geográficos.
    puntos = gpd.GeoDataFrame(
        df_validos,
        geometry=gpd.points_from_xy(
            df_validos["longitude"],
            df_validos["latitude"]
        ),
        crs="EPSG:4326"
    )

    # Comprobar qué sismos están dentro de la geometría.
    mascara_dentro = puntos.geometry.intersects(
        area_estudio
    )

    sismos_dentro = puntos[mascara_dentro].copy()

    sismos_fuera = puntos[~mascara_dentro].copy()

    print(f"Registros originales: {len(df):,}")
    print(f"Dentro de la geometría: {len(sismos_dentro):,}")
    print(f"Fuera de la geometría: {len(sismos_fuera):,}")
    print(f"Coordenadas inválidas: {len(df_invalidos):,}")

    # ========================================================
    # 7. FILTRO POR NOMBRES
    # ========================================================

    lugares = (
        sismos_fuera["place"]
        .fillna("")
        .apply(normalizar_texto)
    )

    # Primero identificamos las descripciones excluidas.
    mascara_exclusion = pd.Series(
        False,
        index=sismos_fuera.index
    )

    for patron in PATRONES_EXCLUIDOS:

        mascara_exclusion |= lugares.str.contains(
            patron,
            regex=True,
            na=False
        )

    # Después buscamos localidades colombianas.
    mascara_colombia = pd.Series(
        False,
        index=sismos_fuera.index
    )

    for lugar in LUGARES_COLOMBIANOS:

        lugar_normalizado = normalizar_texto(lugar)

        patron = (
            rf"(?<![a-z])"
            rf"{re.escape(lugar_normalizado)}"
            rf"(?![a-z])"
        )

        mascara_colombia |= lugares.str.contains(
            patron,
            regex=True,
            na=False
        )

    # Recuperar solamente coincidencias colombianas
    # que no hayan sido excluidas.
    mascara_recuperados = (
        mascara_colombia
        & ~mascara_exclusion
    )

    sismos_recuperados = sismos_fuera[
        mascara_recuperados
    ].copy()

    sismos_excluidos = sismos_fuera[
        mascara_exclusion
    ].copy()

    # Los registros exteriores que no coinciden con
    # localidades ni con exclusiones quedan sin clasificar.
    mascara_sin_clasificar = (
        ~mascara_colombia
        & ~mascara_exclusion
    )

    sismos_sin_clasificar = sismos_fuera[
        mascara_sin_clasificar
    ].copy()

    # ========================================================
    # 8. UNIR LOS REGISTROS SELECCIONADOS
    # ========================================================

    dentro_sin_geometria = sismos_dentro.drop(
        columns="geometry"
    )

    recuperados_sin_geometria = sismos_recuperados.drop(
        columns="geometry"
    )

    resultado = pd.concat(
        [
            dentro_sin_geometria,
            recuperados_sin_geometria
        ],
        ignore_index=True
    )

    # Evitar duplicados mediante el identificador.
    resultado = resultado.drop_duplicates(
        subset=["id"],
        keep="first"
    )

    # ========================================================
    # 9. MOSTRAR RESULTADOS
    # ========================================================

    print("\n========== RESULTADOS DEL FILTRO ==========")

    print(
        f"Sismos dentro de la geometría: "
        f"{len(sismos_dentro):,}"
    )

    print(
        f"Sismos recuperados por nombre: "
        f"{len(sismos_recuperados):,}"
    )

    print(
        f"Sismos excluidos por descripción: "
        f"{len(sismos_excluidos):,}"
    )

    print(
        f"Sismos sin clasificar: "
        f"{len(sismos_sin_clasificar):,}"
    )

    print(
        f"Sismos con coordenadas inválidas: "
        f"{len(df_invalidos):,}"
    )

    print(f"Sismos finales: {len(resultado):,}")

    print("\n========== LOCALIDADES RECUPERADAS ==========")

    if not sismos_recuperados.empty:

        print(
            sismos_recuperados["place"]
            .value_counts()
            .head(30)
            .to_string()
        )

    else:
        print("No se recuperaron registros por nombre.")

    print("\n========== REGISTROS SIN CLASIFICAR ==========")

    if not sismos_sin_clasificar.empty:

        print(
            sismos_sin_clasificar["place"]
            .value_counts()
            .head(30)
            .to_string()
        )

    else:
        print("No hay registros sin clasificar.")

    # ========================================================
    # 10. GUARDAR ARCHIVOS
    # ========================================================

    CARPETA_PROCESADOS.mkdir(
        parents=True,
        exist_ok=True
    )

    resultado.to_csv(
        CARPETA_PROCESADOS / "sismos_colombia.csv",
        index=False,
        encoding="utf-8-sig"
    )

    sismos_dentro.drop(
        columns="geometry"
    ).to_csv(
        CARPETA_PROCESADOS / "sismos_dentro.csv",
        index=False,
        encoding="utf-8-sig"
    )

    sismos_fuera.drop(
        columns="geometry"
    ).to_csv(
        CARPETA_PROCESADOS / "sismos_fuera.csv",
        index=False,
        encoding="utf-8-sig"
    )

    sismos_recuperados.drop(
        columns="geometry"
    ).to_csv(
        CARPETA_PROCESADOS / "sismos_recuperados.csv",
        index=False,
        encoding="utf-8-sig"
    )

    sismos_excluidos.drop(
        columns="geometry"
    ).to_csv(
        CARPETA_PROCESADOS / "sismos_excluidos.csv",
        index=False,
        encoding="utf-8-sig"
    )

    sismos_sin_clasificar.drop(
        columns="geometry"
    ).to_csv(
        CARPETA_PROCESADOS / "sismos_sin_clasificar.csv",
        index=False,
        encoding="utf-8-sig"
    )

    df_invalidos.to_csv(
        CARPETA_PROCESADOS / "sismos_coordenadas_invalidas.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n========== ARCHIVOS GUARDADOS ==========")

    print("sismos_colombia.csv")
    print("sismos_dentro.csv")
    print("sismos_fuera.csv")
    print("sismos_recuperados.csv")
    print("sismos_excluidos.csv")
    print("sismos_sin_clasificar.csv")
    print("sismos_coordenadas_invalidas.csv")

    print("\nProceso geográfico finalizado.")

    return resultado


# ============================================================
# 11. EJECUCIÓN PRINCIPAL
# ============================================================

if __name__ == "__main__":

    from etl.extract import extraer_sismos

    datos = extraer_sismos()

    geometria = seleccionar_geografia()

    resultado = probar_filtro_sismico(
        datos,
        geometria
    )
