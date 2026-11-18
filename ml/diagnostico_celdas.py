import numpy as np
import pandas as pd

from database_cli import consultar_datos_sin_cache

from ml.preparacion_celdas import (
    REGION_NOMBRE,
    LATITUD_MINIMA,
    LATITUD_MAXIMA,
    LONGITUD_MINIMA,
    LONGITUD_MAXIMA,
    TAMANO_CELDA,
    MAGNITUD_SIGNIFICATIVA
)


print("Cargando datos desde MySQL...")

registros = consultar_datos_sin_cache()
df = pd.DataFrame(registros)

if df.empty:
    raise ValueError(
        "La base de datos no contiene registros."
    )


for columna in [
    "latitude",
    "longitude",
    "magnitude",
    "depth"
]:
    df[columna] = pd.to_numeric(
        df[columna],
        errors="coerce"
    )


print(
    f"Registros cargados: {len(df):,}"
)

print(
    "Latitud mínima y máxima:",
    df["latitude"].min(),
    df["latitude"].max()
)

print(
    "Longitud mínima y máxima:",
    df["longitude"].min(),
    df["longitude"].max()
)


# Filtrar primero por ubicación
region = df[
    df["latitude"].between(
        LATITUD_MINIMA,
        LATITUD_MAXIMA
    )
    & df["longitude"].between(
        LONGITUD_MINIMA,
        LONGITUD_MAXIMA
    )
].copy()


if region.empty:
    raise ValueError(
        "No existen registros dentro de Santander. "
        "Revisa las coordenadas de latitude y longitude."
    )


# Procesar fechas después del filtro espacial
fecha_utc = pd.to_datetime(
    region["utc_time"],
    errors="coerce",
    utc=True,
    format="mixed"
)

fecha_local = pd.to_datetime(
    region["local_time"],
    errors="coerce",
    utc=True,
    format="mixed"
)

region["fecha"] = fecha_utc.where(
    fecha_utc.notna(),
    fecha_local
)

region["fecha"] = pd.to_datetime(
    region["fecha"],
    errors="coerce",
    utc=True,
    format="mixed"
)

region = region.dropna(
    subset=[
        "fecha",
        "magnitude"
    ]
)

if region.empty:
    raise ValueError(
        "Santander tiene registros, pero no hay fechas "
        "válidas en utc_time ni local_time."
    )


region["lat_celda"] = (
    np.floor(
        region["latitude"] / TAMANO_CELDA
    ) * TAMANO_CELDA
).round(2)

region["lon_celda"] = (
    np.floor(
        region["longitude"] / TAMANO_CELDA
    ) * TAMANO_CELDA
).round(2)

region["celda"] = (
    region["lat_celda"].astype(str)
    + "_"
    + region["lon_celda"].astype(str)
)


fecha_sin_zona = (
    region["fecha"]
    .dt.tz_localize(None)
)

region["semana"] = (
    fecha_sin_zona
    .dt.to_period("W-SUN")
    .dt.start_time
)


resumen_celdas = (
    region.groupby(
        [
            "celda",
            "lat_celda",
            "lon_celda"
        ]
    )
    .agg(
        registros=("id", "count"),
        magnitud_maxima=("magnitude", "max"),
        magnitud_promedio=("magnitude", "mean"),
        eventos_45=(
            "magnitude",
            lambda valores: (
                valores >= MAGNITUD_SIGNIFICATIVA
            ).sum()
        ),
        semanas_activas=("semana", "nunique")
    )
    .reset_index()
    .sort_values(
        "registros",
        ascending=False
    )
)


semanal = (
    region.groupby(
        [
            "semana",
            "celda"
        ]
    )
    .agg(
        cantidad_sismos=("id", "count"),
        magnitud_maxima=("magnitude", "max"),
        eventos_45=(
            "magnitude",
            lambda valores: (
                valores >= MAGNITUD_SIGNIFICATIVA
            ).sum()
        )
    )
    .reset_index()
)


semanas = pd.date_range(
    start=region["semana"].min(),
    end=region["semana"].max(),
    freq="7D"
)

celdas = resumen_celdas[
    [
        "celda",
        "lat_celda",
        "lon_celda"
    ]
].drop_duplicates()

panel = (
    celdas.assign(llave=1)
    .merge(
        pd.DataFrame(
            {
                "semana": semanas,
                "llave": 1
            }
        ),
        on="llave"
    )
    .drop(columns="llave")
)

panel = panel.merge(
    semanal,
    on=[
        "semana",
        "celda"
    ],
    how="left"
)

panel["cantidad_sismos"] = (
    panel["cantidad_sismos"].fillna(0)
)

panel["magnitud_maxima"] = (
    panel["magnitud_maxima"].fillna(0)
)

panel["eventos_45"] = (
    panel["eventos_45"].fillna(0)
)

panel["evento_significativo"] = (
    panel["eventos_45"] > 0
).astype(int)


print("\nDIAGNÓSTICO DE CELDAS")
print("---------------------")

print(
    f"Región: {REGION_NOMBRE}"
)

print(
    f"Registros regionales: "
    f"{len(region):,}"
)

print(
    f"Celdas detectadas: "
    f"{region['celda'].nunique():,}"
)

print(
    f"Semanas disponibles: "
    f"{len(semanas):,}"
)

print(
    f"Filas del panel espacial: "
    f"{len(panel):,}"
)

print(
    f"Fecha mínima: "
    f"{region['fecha'].min()}"
)

print(
    f"Fecha máxima: "
    f"{region['fecha'].max()}"
)

print(
    f"Magnitud máxima: "
    f"{region['magnitude'].max()}"
)

print(
    f"Eventos >= {MAGNITUD_SIGNIFICATIVA}: "
    f"{(region['magnitude'] >= MAGNITUD_SIGNIFICATIVA).sum():,}"
)

print(
    "Semanas-celda positivas: "
    f"{panel['evento_significativo'].sum():,}"
)

print(
    "Porcentaje de semanas-celda positivas: "
    f"{panel['evento_significativo'].mean():.2%}"
)

print(
    f"Duplicados por id: "
    f"{region['id'].duplicated().sum():,}"
)

print(
    f"Duplicados completos: "
    f"{region.duplicated().sum():,}"
)


print("\nTOP 15 CELDAS POR REGISTROS")
print("---------------------------")

print(
    resumen_celdas
    .head(15)
    .to_string(index=False)
)


print("\nTOP 15 CELDAS POR EVENTOS >= 4.5")
print("----------------------------------")

print(
    resumen_celdas
    .sort_values(
        "eventos_45",
        ascending=False
    )
    .head(15)
    .to_string(index=False)
)