import numpy as np
import pandas as pd


LAT_MIN = -4.5
LAT_MAX = 13.5
LON_MIN = -81.5
LON_MAX = -66.5

TAMANO_CELDA = 0.5
MAGNITUD_SIGNIFICATIVA = 3.5


COLUMNAS_MODELO = [
    "cantidad_sismos",
    "eventos_significativos",
    "magnitud_maxima",
    "magnitud_promedio",
    "profundidad_promedio",
    "sismos_lag_1",
    "sismos_lag_2",
    "sismos_lag_3",
    "sismos_lag_4",
    "sismos_ultimas_4_semanas",
    "sismos_ultimas_8_semanas",
    "eventos_ultimas_4_semanas",
    "eventos_ultimas_8_semanas",
    "mes",
    "semana_anio",
]


def convertir_fecha(df):
    """
    Convierte utc_time y local_time a UTC.

    Se utiliza texto intermedio para evitar conflictos
    entre datetime64[us], datetime64[ns], fechas con zona
    horaria y fechas sin zona horaria.
    """

    datos = df.copy()

    utc_texto = (
        datos["utc_time"]
        .astype("string")
    )

    local_texto = (
        datos["local_time"]
        .astype("string")
    )

    utc_validada = pd.to_datetime(
        utc_texto,
        errors="coerce",
        utc=True,
    )

    usar_local = utc_validada.isna()

    fecha_texto = utc_texto.where(
        ~usar_local,
        local_texto,
    )

    datos["fecha"] = pd.to_datetime(
        fecha_texto,
        errors="coerce",
        utc=True,
    )

    return datos


def obtener_region(latitud, longitud):
    """
    Clasificación aproximada para análisis regional.
    """

    if (
        6.0 <= latitud <= 8.5
        and -74.5 <= longitud <= -72.5
    ):
        return "Santander"

    if (
        latitud >= 7.0
        and longitud <= -75.0
    ):
        return "Caribe"

    if (
        1.0 <= latitud <= 8.0
        and -78.5 <= longitud <= -75.0
    ):
        return "Pacifico"

    if (
        3.0 <= latitud <= 8.5
        and -75.0 <= longitud <= -71.0
    ):
        return "Andina"

    if (
        2.0 <= latitud <= 7.0
        and -71.0 <= longitud <= -67.0
    ):
        return "Orinoquia"

    if (
        latitud <= 2.0
        and longitud >= -75.0
    ):
        return "Amazonia"

    return "Otros"


def construir_panel_nacional(
    df,
    incluir_objetivo=True,
):
    """
    Construye el panel celda-semana nacional.

    Las variables disponibles hasta la semana actual
    se utilizan para estimar un evento durante la
    semana siguiente.
    """

    datos = convertir_fecha(df)

    for columna in [
        "latitude",
        "longitude",
        "magnitude",
        "depth",
    ]:
        datos[columna] = pd.to_numeric(
            datos[columna],
            errors="coerce",
        )

    datos = datos.dropna(
        subset=[
            "fecha",
            "latitude",
            "longitude",
            "magnitude",
        ]
    ).copy()

    if datos.empty:
        raise ValueError(
            "No existen fechas o coordenadas válidas."
        )

    datos = datos[
        datos["latitude"].between(
            LAT_MIN,
            LAT_MAX,
        )
        & datos["longitude"].between(
            LON_MIN,
            LON_MAX,
        )
    ].copy()

    if datos.empty:
        raise ValueError(
            "No existen registros dentro de "
            "la cuadrícula nacional."
        )

    # ---------------------------------------------------------------
    # Celdas geográficas fijas
    # ---------------------------------------------------------------

    datos["lat_celda"] = (
        np.floor(
            (
                datos["latitude"]
                - LAT_MIN
            )
            / TAMANO_CELDA
        )
        * TAMANO_CELDA
        + LAT_MIN
    ).round(2)

    datos["lon_celda"] = (
        np.floor(
            (
                datos["longitude"]
                - LON_MIN
            )
            / TAMANO_CELDA
        )
        * TAMANO_CELDA
        + LON_MIN
    ).round(2)

    datos["celda"] = (
        datos["lat_celda"].map(
            lambda valor: f"{valor:.2f}"
        )
        + "_"
        + datos["lon_celda"].map(
            lambda valor: f"{valor:.2f}"
        )
    )

    datos["region"] = datos.apply(
        lambda fila: obtener_region(
            fila["latitude"],
            fila["longitude"],
        ),
        axis=1,
    )

    # ---------------------------------------------------------------
    # Conversión segura a semanas
    # ---------------------------------------------------------------

    fechas_sin_zona = (
        datos["fecha"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )

    datos["semana"] = (
        fechas_sin_zona
        .dt.to_period("W-SUN")
        .dt.start_time
    )

    datos["semana"] = pd.to_datetime(
        datos["semana"],
        errors="coerce",
        utc=True,
    )

    datos = datos.dropna(
        subset=["semana"]
    ).copy()

    if datos.empty:
        raise ValueError(
            "No se pudieron crear semanas válidas."
        )

    # ---------------------------------------------------------------
    # Filtrar semanas completas
    # ---------------------------------------------------------------

    fecha_maxima_catalogo = datos["semana"].max()

    ahora_utc = pd.Timestamp.now(
        tz="UTC"
    )

    ahora_sin_zona = ahora_utc.tz_localize(
        None
    )

    semana_actual_sin_zona = (
        ahora_sin_zona
        .to_period("W-SUN")
        .start_time
    )

    semana_actual_utc = pd.Timestamp(
        semana_actual_sin_zona,
        tz="UTC",
    )

    ultima_semana_completa = min(
        fecha_maxima_catalogo,
        semana_actual_utc
        - pd.Timedelta(days=7),
    )

    datos = datos[
        datos["semana"]
        <= ultima_semana_completa
    ].copy()

    if datos.empty:
        raise ValueError(
            "No hay semanas completas disponibles."
        )

    primera_semana = datos["semana"].min()
    ultima_semana = datos["semana"].max()

    semanas = pd.date_range(
        start=primera_semana,
        end=ultima_semana,
        freq="7D",
        tz="UTC",
    )

    # ---------------------------------------------------------------
    # Crear cuadrícula fija
    # ---------------------------------------------------------------

    latitudes = np.arange(
        LAT_MIN,
        LAT_MAX,
        TAMANO_CELDA,
    ).round(2)

    longitudes = np.arange(
        LON_MIN,
        LON_MAX,
        TAMANO_CELDA,
    ).round(2)

    celdas = []

    for latitud in latitudes:
        for longitud in longitudes:
            celdas.append(
                {
                    "lat_celda": latitud,
                    "lon_celda": longitud,
                    "celda": (
                        f"{latitud:.2f}_"
                        f"{longitud:.2f}"
                    ),
                    "region": obtener_region(
                        latitud,
                        longitud,
                    ),
                }
            )

    celdas = pd.DataFrame(celdas)

    calendario = pd.DataFrame(
        [
            {
                "celda": fila["celda"],
                "lat_celda": fila["lat_celda"],
                "lon_celda": fila["lon_celda"],
                "region": fila["region"],
                "semana": semana,
            }
            for _, fila in celdas.iterrows()
            for semana in semanas
        ]
    )

    # ---------------------------------------------------------------
    # Resumen sísmico semanal
    # ---------------------------------------------------------------

    datos["evento_significativo"] = (
        datos["magnitude"]
        >= MAGNITUD_SIGNIFICATIVA
    ).astype(int)

    resumen = (
        datos.groupby(
            [
                "celda",
                "lat_celda",
                "lon_celda",
                "region",
                "semana",
            ],
            as_index=False,
        )
        .agg(
            cantidad_sismos=(
                "id",
                "count",
            ),
            eventos_significativos=(
                "evento_significativo",
                "sum",
            ),
            magnitud_maxima=(
                "magnitude",
                "max",
            ),
            magnitud_promedio=(
                "magnitude",
                "mean",
            ),
            profundidad_promedio=(
                "depth",
                "mean",
            ),
        )
    )

    panel = calendario.merge(
        resumen,
        on=[
            "celda",
            "lat_celda",
            "lon_celda",
            "region",
            "semana",
        ],
        how="left",
    )

    columnas_actividad = [
        "cantidad_sismos",
        "eventos_significativos",
        "magnitud_maxima",
        "magnitud_promedio",
        "profundidad_promedio",
    ]

    panel[columnas_actividad] = (
        panel[columnas_actividad]
        .fillna(0)
        .astype(float)
    )

    # ---------------------------------------------------------------
    # Nombre de zona
    # ---------------------------------------------------------------

    if "place" in datos.columns:
        nombres_zona = (
            datos.groupby("celda")["place"]
            .agg(
                lambda serie: (
                    serie.dropna()
                    .astype(str)
                    .value_counts()
                    .index[0]
                    if not serie.dropna().empty
                    else "Zona sin nombre"
                )
            )
        )

        panel["zona_nombre"] = (
            panel["celda"]
            .map(nombres_zona)
            .fillna("Zona sin registros")
        )
    else:
        panel["zona_nombre"] = (
            "Zona sin nombre"
        )

    panel = panel.sort_values(
        [
            "celda",
            "semana",
        ]
    ).reset_index(drop=True)

    grupo = panel.groupby(
        "celda",
        group_keys=False,
    )

    # ---------------------------------------------------------------
    # Variables históricas
    # ---------------------------------------------------------------

    panel["sismos_lag_1"] = (
        grupo["cantidad_sismos"]
        .shift(1)
    )

    panel["sismos_lag_2"] = (
        grupo["cantidad_sismos"]
        .shift(2)
    )

    panel["sismos_lag_3"] = (
        grupo["cantidad_sismos"]
        .shift(3)
    )

    panel["sismos_lag_4"] = (
        grupo["cantidad_sismos"]
        .shift(4)
    )

    panel["sismos_ultimas_4_semanas"] = (
        grupo["cantidad_sismos"]
        .transform(
            lambda serie: serie.rolling(
                window=4,
                min_periods=1,
            ).sum()
        )
    )

    panel["sismos_ultimas_8_semanas"] = (
        grupo["cantidad_sismos"]
        .transform(
            lambda serie: serie.rolling(
                window=8,
                min_periods=1,
            ).sum()
        )
    )

    panel["eventos_ultimas_4_semanas"] = (
        grupo["eventos_significativos"]
        .transform(
            lambda serie: serie.rolling(
                window=4,
                min_periods=1,
            ).sum()
        )
    )

    panel["eventos_ultimas_8_semanas"] = (
        grupo["eventos_significativos"]
        .transform(
            lambda serie: serie.rolling(
                window=8,
                min_periods=1,
            ).sum()
        )
    )

    panel["mes"] = panel["semana"].dt.month

    panel["semana_anio"] = (
        panel["semana"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    # ---------------------------------------------------------------
    # Objetivo: evento en la siguiente semana
    # ---------------------------------------------------------------

    panel["evento_semana_siguiente"] = (
        grupo["eventos_significativos"]
        .shift(-1)
    )

    panel["objetivo"] = (
        panel["evento_semana_siguiente"]
        .gt(0)
        .astype(float)
    )

    # La última semana no tiene objetivo conocido.
    panel.loc[
        panel["evento_semana_siguiente"].isna(),
        "objetivo",
    ] = np.nan

    panel[COLUMNAS_MODELO] = (
        panel[COLUMNAS_MODELO]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    if incluir_objetivo:
        panel = panel.dropna(
            subset=["objetivo"]
        ).copy()

        panel["objetivo"] = (
            panel["objetivo"]
            .astype(int)
        )

    return panel, COLUMNAS_MODELO