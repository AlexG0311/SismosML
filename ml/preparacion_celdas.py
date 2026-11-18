import numpy as np
import pandas as pd


REGION_NOMBRE = "Santander - Bucaramanga"

LATITUD_MINIMA = 6.7
LATITUD_MAXIMA = 7.5

LONGITUD_MINIMA = -74.2
LONGITUD_MAXIMA = -73.0

TAMANO_CELDA = 0.25
MAGNITUD_SIGNIFICATIVA = 3.5


def obtener_nombre_zona(valores):
    valores = (
        valores
        .dropna()
        .astype(str)
        .str.strip()
    )

    valores = valores[
        valores != ""
    ]

    if valores.empty:
        return "Zona de Santander"

    return valores.value_counts().index[0]


def construir_datos_celdas(
    df,
    incluir_objetivo=True
):
    datos = df.copy()

    for columna in [
        "latitude",
        "longitude",
        "magnitude",
        "depth"
    ]:
        datos[columna] = pd.to_numeric(
            datos[columna],
            errors="coerce"
        )

    if "place" not in datos.columns:
        datos["place"] = "Zona de Santander"

    fecha_utc = pd.to_datetime(
        datos["utc_time"],
        errors="coerce",
        utc=True,
        format="mixed"
    )

    fecha_local = pd.to_datetime(
        datos["local_time"],
        errors="coerce",
        utc=True,
        format="mixed"
    )

    datos["fecha"] = fecha_utc.where(
        fecha_utc.notna(),
        fecha_local
    )

    datos["fecha"] = pd.to_datetime(
        datos["fecha"],
        errors="coerce",
        utc=True,
        format="mixed"
    )

    datos = datos.dropna(
        subset=[
            "fecha",
            "latitude",
            "longitude",
            "magnitude"
        ]
    )

    datos = datos[
        datos["latitude"].between(
            LATITUD_MINIMA,
            LATITUD_MAXIMA
        )
        & datos["longitude"].between(
            LONGITUD_MINIMA,
            LONGITUD_MAXIMA
        )
    ].copy()

    if datos.empty:
        raise ValueError(
            "No existen registros dentro de Santander."
        )

    fecha_sin_zona = (
        datos["fecha"]
        .dt.tz_localize(None)
    )

    datos["semana"] = (
        fecha_sin_zona
        .dt.to_period("W-SUN")
        .dt.start_time
    )

    datos["lat_celda"] = (
        np.floor(
            datos["latitude"] / TAMANO_CELDA
        ) * TAMANO_CELDA
    ).round(2)

    datos["lon_celda"] = (
        np.floor(
            datos["longitude"] / TAMANO_CELDA
        ) * TAMANO_CELDA
    ).round(2)

    datos["celda"] = (
        datos["lat_celda"].astype(str)
        + "_"
        + datos["lon_celda"].astype(str)
    )

    nombres_celdas = (
        datos.groupby("celda")["place"]
        .agg(obtener_nombre_zona)
        .reset_index(name="zona_nombre")
    )

    agregados = (
        datos.groupby(
            [
                "semana",
                "celda",
                "lat_celda",
                "lon_celda"
            ]
        )
        .agg(
            cantidad_sismos=("id", "count"),
            magnitud_promedio=("magnitude", "mean"),
            magnitud_maxima=("magnitude", "max"),
            profundidad_promedio=("depth", "mean"),

            eventos_magnitud_3=(
                "magnitude",
                lambda valores: (
                    valores >= 3.0
                ).sum()
            ),

            eventos_magnitud_35=(
                "magnitude",
                lambda valores: (
                    valores >= MAGNITUD_SIGNIFICATIVA
                ).sum()
            )
        )
        .reset_index()
    )

    semanas = pd.date_range(
        start=agregados["semana"].min(),
        end=agregados["semana"].max(),
        freq="7D"
    )

    celdas = agregados[
        [
            "celda",
            "lat_celda",
            "lon_celda"
        ]
    ].drop_duplicates()

    celdas["_llave"] = 1

    semanas_df = pd.DataFrame(
        {
            "semana": semanas,
            "_llave": 1
        }
    )

    panel = (
        celdas
        .merge(
            semanas_df,
            on="_llave"
        )
        .drop(columns="_llave")
    )

    panel = panel.merge(
        agregados,
        on=[
            "semana",
            "celda",
            "lat_celda",
            "lon_celda"
        ],
        how="left"
    )

    panel = panel.merge(
        nombres_celdas,
        on="celda",
        how="left"
    )

    columnas_cantidad = [
        "cantidad_sismos",
        "eventos_magnitud_3",
        "eventos_magnitud_35"
    ]

    for columna in columnas_cantidad:
        panel[columna] = (
            panel[columna].fillna(0)
        )

    columnas_numericas = [
        "magnitud_promedio",
        "magnitud_maxima",
        "profundidad_promedio"
    ]

    for columna in columnas_numericas:
        panel[columna] = (
            panel[columna].fillna(0)
        )

    panel["zona_nombre"] = (
        panel["zona_nombre"]
        .fillna("Zona de Santander")
    )

    panel = panel.sort_values(
        [
            "celda",
            "semana"
        ]
    ).reset_index(drop=True)

    for retraso in [1, 2, 3, 4, 8, 12]:
        panel[
            f"sismos_lag_{retraso}"
        ] = (
            panel.groupby("celda")[
                "cantidad_sismos"
            ].shift(retraso)
        )

        panel[
            f"eventos_35_lag_{retraso}"
        ] = (
            panel.groupby("celda")[
                "eventos_magnitud_35"
            ].shift(retraso)
        )

    for ventana in [2, 4, 8, 12]:
        panel[
            f"sismos_ultimas_{ventana}_semanas"
        ] = (
            panel.groupby("celda")[
                "cantidad_sismos"
            ]
            .transform(
                lambda serie: (
                    serie
                    .shift(1)
                    .rolling(ventana)
                    .sum()
                )
            )
        )

        panel[
            f"eventos_35_ultimas_{ventana}_semanas"
        ] = (
            panel.groupby("celda")[
                "eventos_magnitud_35"
            ]
            .transform(
                lambda serie: (
                    serie
                    .shift(1)
                    .rolling(ventana)
                    .sum()
                )
            )
        )

        panel[
            f"magnitud_maxima_ultimas_{ventana}_semanas"
        ] = (
            panel.groupby("celda")[
                "magnitud_maxima"
            ]
            .transform(
                lambda serie: (
                    serie
                    .shift(1)
                    .rolling(ventana)
                    .max()
                )
            )
        )

        panel[
            f"magnitud_promedio_ultimas_{ventana}_semanas"
        ] = (
            panel.groupby("celda")[
                "magnitud_promedio"
            ]
            .transform(
                lambda serie: (
                    serie
                    .shift(1)
                    .rolling(ventana)
                    .mean()
                )
            )
        )

    columnas_modelo = [
        "lat_celda",
        "lon_celda",
        "cantidad_sismos",
        "magnitud_promedio",
        "magnitud_maxima",
        "profundidad_promedio",
        "eventos_magnitud_3",
        "eventos_magnitud_35",

        "sismos_lag_1",
        "sismos_lag_2",
        "sismos_lag_3",
        "sismos_lag_4",
        "sismos_lag_8",
        "sismos_lag_12",

        "eventos_35_lag_1",
        "eventos_35_lag_2",
        "eventos_35_lag_3",
        "eventos_35_lag_4",
        "eventos_35_lag_8",
        "eventos_35_lag_12",

        "sismos_ultimas_2_semanas",
        "sismos_ultimas_4_semanas",
        "sismos_ultimas_8_semanas",
        "sismos_ultimas_12_semanas",

        "eventos_35_ultimas_2_semanas",
        "eventos_35_ultimas_4_semanas",
        "eventos_35_ultimas_8_semanas",
        "eventos_35_ultimas_12_semanas",

        "magnitud_maxima_ultimas_2_semanas",
        "magnitud_maxima_ultimas_4_semanas",
        "magnitud_maxima_ultimas_8_semanas",
        "magnitud_maxima_ultimas_12_semanas",

        "magnitud_promedio_ultimas_2_semanas",
        "magnitud_promedio_ultimas_4_semanas",
        "magnitud_promedio_ultimas_8_semanas",
        "magnitud_promedio_ultimas_12_semanas"
    ]

    if incluir_objetivo:
        siguiente_semana = (
            panel.groupby("celda")[
                "eventos_magnitud_35"
            ].shift(-1)
        )

        panel[
            "evento_significativo_siguiente_semana"
        ] = (
            siguiente_semana.gt(0).astype(int)
        )

        panel = panel.loc[
            siguiente_semana.notna()
        ].copy()

        panel = panel.dropna(
            subset=columnas_modelo
        )

    else:
        panel = panel.dropna(
            subset=columnas_modelo
        )

    if panel.empty:
        raise ValueError(
            "No hay suficientes datos para construir "
            "el panel espacial."
        )

    return panel, columnas_modelo