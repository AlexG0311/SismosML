from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from database import consultar_datos
from ml.preparacion_nacional import (
    construir_panel_nacional,
)


BASE_DIR = Path(__file__).resolve().parent
MODELOS_DIR = BASE_DIR / "modelos"

MAGNITUD_OBJETIVO = 3.5


st.set_page_config(
    page_title="SeismicAI",
    page_icon="🌎",
    layout="wide",
)


@st.cache_resource
def cargar_modelos():
    modelo = joblib.load(
        MODELOS_DIR / "modelo_nacional.pkl"
    )

    columnas_modelo = joblib.load(
        MODELOS_DIR / "columnas_nacional.pkl"
    )

    metricas = joblib.load(
        MODELOS_DIR / "metricas_nacional.pkl"
    )

    tasas_historicas = joblib.load(
        MODELOS_DIR
        / "tasas_historicas_nacionales.pkl"
    )

    return (
        modelo,
        columnas_modelo,
        metricas,
        tasas_historicas,
    )


def convertir_datos(df):
    datos = df.copy()

    for columna in [
        "magnitude",
        "latitude",
        "longitude",
        "depth",
    ]:
        if columna in datos.columns:
            datos[columna] = pd.to_numeric(
                datos[columna],
                errors="coerce",
            )

    utc_texto = datos[
        "utc_time"
    ].astype("string")

    local_texto = datos[
        "local_time"
    ].astype("string")

    fecha_utc = pd.to_datetime(
        utc_texto,
        errors="coerce",
        utc=True,
    )

    fecha_texto = utc_texto.where(
        ~fecha_utc.isna(),
        local_texto,
    )

    datos["fecha_modelo"] = pd.to_datetime(
        fecha_texto,
        errors="coerce",
        utc=True,
    )

    return datos.dropna(
        subset=["fecha_modelo"]
    ).copy()


def porcentaje(valor):
    if valor is None or pd.isna(valor):
        return "N/D"

    return f"{float(valor):.2%}"


def numero(valor):
    if valor is None or pd.isna(valor):
        return "N/D"

    return f"{float(valor):.6f}"


def construir_ranking(
    modelo,
    columnas_modelo,
    tasas_historicas,
    panel,
):
    ultima_semana = panel["semana"].max()

    actual = panel[
        panel["semana"] == ultima_semana
    ].copy()

    if actual.empty:
        return pd.DataFrame()

    X_actual = actual.reindex(
        columns=columnas_modelo,
        fill_value=0,
    )

    actual["score_calibrado"] = (
        modelo.predict_proba(X_actual)[:, 1]
    )

    actual["tasa_historica"] = (
        actual["celda"]
        .map(tasas_historicas)
        .fillna(0)
    )

    actual["diferencia_historica"] = (
        actual["score_calibrado"]
        - actual["tasa_historica"]
    )

    actual["lift"] = (
        actual["score_calibrado"]
        / actual[
            "tasa_historica"
        ].replace(0, pd.NA)
    )

    actual["lift"] = actual[
        "lift"
    ].fillna(0)

    return actual.sort_values(
        "score_calibrado",
        ascending=False,
    ).reset_index(drop=True)


def crear_tabla_ranking(ranking):
    columnas = [
        "zona_nombre",
        "region",
        "score_calibrado",
        "tasa_historica",
        "diferencia_historica",
        "lift",
        "cantidad_sismos",
        "eventos_significativos",
        "magnitud_maxima",
    ]

    columnas = [
        columna
        for columna in columnas
        if columna in ranking.columns
    ]

    tabla = ranking[
        columnas
    ].copy()

    tabla.insert(
        0,
        "posicion",
        range(1, len(tabla) + 1),
    )

    tabla = tabla.rename(
        columns={
            "posicion": "Posición",
            "zona_nombre": "Zona",
            "region": "Región",
            "score_calibrado": (
                "Score calibrado"
            ),
            "tasa_historica": (
                "Tasa histórica"
            ),
            "diferencia_historica": (
                "Diferencia histórica"
            ),
            "lift": "Lift",
            "cantidad_sismos": (
                "Sismos de referencia"
            ),
            "eventos_significativos": (
                "Eventos ≥ 3.5"
            ),
            "magnitud_maxima": (
                "Magnitud máxima"
            ),
        }
    )

    for columna in [
        "Score calibrado",
        "Tasa histórica",
        "Diferencia histórica",
    ]:
        if columna in tabla.columns:
            tabla[columna] = tabla[
                columna
            ].map(porcentaje)

    if "Lift" in tabla.columns:
        tabla["Lift"] = tabla[
            "Lift"
        ].map(
            lambda valor: (
                f"{float(valor):.2f}x"
                if pd.notna(valor)
                and float(valor) > 0
                else "N/D"
            )
        )

    return tabla


# -------------------------------------------------------------------
# Encabezado
# -------------------------------------------------------------------

st.title("🌎 SeismicAI")

st.caption(
    "Plataforma de análisis de actividad sísmica "
    "en Colombia mediante ingeniería de datos "
    "y aprendizaje automático."
)

st.sidebar.title("Navegación")

opcion = st.sidebar.radio(
    "Selecciona una sección",
    [
        "Inicio",
        "Análisis sísmico",
        "Mapa de sismos",
        "Análisis temporal",
        "Predicción ML",
    ],
)

if st.sidebar.button("Actualizar datos"):
    st.session_state.pop(
        "datos_sismos",
        None,
    )

    consultar_datos.clear()
    cargar_modelos.clear()


st.sidebar.caption(
    "Los datos se conservan al cambiar de sección."
)


# -------------------------------------------------------------------
# Carga de datos
# -------------------------------------------------------------------

try:
    if "datos_sismos" not in st.session_state:
        with st.spinner(
            "Cargando registros desde MySQL..."
        ):
            registros = consultar_datos()
            datos = pd.DataFrame(registros)

            if datos.empty:
                st.warning(
                    "No hay registros disponibles."
                )
                st.stop()

            st.session_state[
                "datos_sismos"
            ] = convertir_datos(datos)

    df = st.session_state[
        "datos_sismos"
    ]

except Exception as error:
    st.error(
        f"No fue posible cargar los datos: {error}"
    )
    st.stop()


# -------------------------------------------------------------------
# Inicio
# -------------------------------------------------------------------

if opcion == "Inicio":
    st.header(
        "🌎 Bienvenido a SeismicAI"
    )

    st.markdown(
        """
        **SeismicAI** es una plataforma de ingeniería de datos
        y aprendizaje automático para analizar la actividad
        sísmica registrada en Colombia.

        El sistema integra información histórica, procesos de
        limpieza, almacenamiento en MySQL, visualizaciones
        interactivas y un modelo XGBoost para ordenar zonas
        según su señal estadística de actividad sísmica.
        """
    )

    st.info(
        "Este proyecto analiza patrones históricos. "
        "No predice con certeza la fecha, ubicación o magnitud "
        "exacta de futuros terremotos."
    )

    fecha_minima = df[
        "fecha_modelo"
    ].min()

    fecha_maxima = df[
        "fecha_modelo"
    ].max()

    eventos_significativos = int(
        (
            df["magnitude"]
            >= MAGNITUD_OBJETIVO
        ).sum()
    )

    st.subheader(
        "Indicadores generales"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Registros sísmicos",
        f"{len(df):,}",
    )

    col2.metric(
        "Magnitud máxima",
        f"{df['magnitude'].max():.2f}",
    )

    col3.metric(
        "Eventos ≥ 3.5",
        f"{eventos_significativos:,}",
    )

    col4.metric(
        "Agencias",
        (
            df["agency"].nunique()
            if "agency" in df.columns
            else 0
        ),
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Fecha inicial",
        (
            fecha_minima.strftime("%Y-%m-%d")
            if pd.notna(fecha_minima)
            else "N/D"
        ),
    )

    col2.metric(
        "Última fecha disponible",
        (
            fecha_maxima.strftime("%Y-%m-%d")
            if pd.notna(fecha_maxima)
            else "N/D"
        ),
    )

    col3.metric(
        "Latitud mínima",
        numero(df["latitude"].min()),
    )

    col4.metric(
        "Longitud máxima",
        numero(df["longitude"].max()),
    )

    st.subheader(
        "Actividad sísmica por año"
    )

    datos_anuales = df.copy()

    datos_anuales["fecha_modelo"] = pd.to_datetime(
        datos_anuales["fecha_modelo"],
        errors="coerce",
        utc=True,
    )

    datos_anuales = datos_anuales.dropna(
        subset=["fecha_modelo"]
    ).copy()

    datos_anuales["año"] = (
        datos_anuales[
            "fecha_modelo"
        ].dt.year
    )

    resumen_anual = (
        datos_anuales.groupby("año")
        .size()
        .reset_index(
            name="cantidad"
        )
    )

    figura = px.bar(
        resumen_anual,
        x="año",
        y="cantidad",
        title="Cantidad de sismos por año",
        labels={
            "año": "Año",
            "cantidad": "Sismos",
        },
        color="cantidad",
        color_continuous_scale="Viridis",
    )

    st.plotly_chart(
        figura,
        use_container_width=True,
    )


# -------------------------------------------------------------------
# Análisis sísmico
# -------------------------------------------------------------------

elif opcion == "Análisis sísmico":
    st.header(
        "📊 Análisis avanzado de actividad sísmica"
    )

    datos = df.copy()

    datos["fecha_modelo"] = pd.to_datetime(
        datos["fecha_modelo"],
        errors="coerce",
        utc=True,
    )

    datos["magnitude"] = pd.to_numeric(
        datos["magnitude"],
        errors="coerce",
    )

    datos["depth"] = pd.to_numeric(
        datos["depth"],
        errors="coerce",
    )

    datos = datos.dropna(
        subset=[
            "fecha_modelo",
            "magnitude",
        ]
    ).copy()

    st.write(
        "Explora la evolución temporal, las magnitudes, "
        "la profundidad y los principales periodos de "
        "actividad sísmica."
    )

    fecha_minima = datos[
        "fecha_modelo"
    ].min().date()

    fecha_maxima = datos[
        "fecha_modelo"
    ].max().date()

    col1, col2, col3 = st.columns(3)

    with col1:
        fecha_inicio = st.date_input(
            "Fecha inicial",
            value=fecha_minima,
            min_value=fecha_minima,
            max_value=fecha_maxima,
        )

    with col2:
        fecha_fin = st.date_input(
            "Fecha final",
            value=fecha_maxima,
            min_value=fecha_minima,
            max_value=fecha_maxima,
        )

    with col3:
        magnitud_minima = st.slider(
            "Magnitud mínima",
            min_value=0.0,
            max_value=float(
                datos["magnitude"].max()
            ),
            value=0.0,
            step=0.1,
        )

    if fecha_inicio > fecha_fin:
        st.error(
            "La fecha inicial no puede ser posterior "
            "a la fecha final."
        )
        st.stop()

    datos["fecha_dia"] = (
        datos["fecha_modelo"].dt.date
    )

    datos_filtrados = datos[
        (datos["fecha_dia"] >= fecha_inicio)
        & (datos["fecha_dia"] <= fecha_fin)
        & (
            datos["magnitude"]
            >= magnitud_minima
        )
    ].copy()

    if datos_filtrados.empty:
        st.warning(
            "No hay registros con los filtros seleccionados."
        )
        st.stop()

    datos_filtrados["año"] = (
        datos_filtrados[
            "fecha_modelo"
        ].dt.year
    )

    datos_filtrados["mes"] = (
        datos_filtrados[
            "fecha_modelo"
        ].dt.month
    )

    datos_filtrados["hora"] = (
        datos_filtrados[
            "fecha_modelo"
        ].dt.hour
    )

    datos_filtrados["dia_semana"] = (
        datos_filtrados[
            "fecha_modelo"
        ].dt.dayofweek
    )

    eventos_significativos = int(
        (
            datos_filtrados["magnitude"]
            >= MAGNITUD_OBJETIVO
        ).sum()
    )

    st.subheader(
        "Indicadores del periodo seleccionado"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Sismos analizados",
        f"{len(datos_filtrados):,}",
    )

    col2.metric(
        "Magnitud máxima",
        f"{datos_filtrados['magnitude'].max():.2f}",
    )

    col3.metric(
        "Magnitud promedio",
        f"{datos_filtrados['magnitude'].mean():.2f}",
    )

    col4.metric(
        "Eventos ≥ 3.5",
        f"{eventos_significativos:,}",
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "Evolución",
            "Calendario",
            "Patrones horarios",
        ]
    )

    with tab1:
        actividad_mensual = (
            datos_filtrados
            .set_index("fecha_modelo")
            .resample("MS")
            .size()
            .reset_index(
                name="cantidad"
            )
        )

        actividad_mensual[
            "promedio_movil"
        ] = (
            actividad_mensual["cantidad"]
            .rolling(
                3,
                min_periods=1,
            )
            .mean()
        )

        figura = px.line(
            actividad_mensual,
            x="fecha_modelo",
            y=[
                "cantidad",
                "promedio_movil",
            ],
            markers=True,
            title=(
                "Actividad mensual y promedio móvil"
            ),
            labels={
                "fecha_modelo": "Fecha",
                "value": "Cantidad",
                "variable": "Indicador",
            },
        )

        st.plotly_chart(
            figura,
            use_container_width=True,
        )

        actividad_mensual[
            "acumulado"
        ] = actividad_mensual[
            "cantidad"
        ].cumsum()

        figura_acumulada = px.area(
            actividad_mensual,
            x="fecha_modelo",
            y="acumulado",
            title="Actividad acumulada",
            labels={
                "fecha_modelo": "Fecha",
                "acumulado": "Sismos acumulados",
            },
        )

        st.plotly_chart(
            figura_acumulada,
            use_container_width=True,
        )

        significativos = datos_filtrados[
            datos_filtrados["magnitude"]
            >= MAGNITUD_OBJETIVO
        ]

        if not significativos.empty:
            resumen_significativos = (
                significativos
                .set_index("fecha_modelo")
                .resample("MS")
                .size()
                .reset_index(
                    name="eventos"
                )
            )

            figura_significativos = px.bar(
                resumen_significativos,
                x="fecha_modelo",
                y="eventos",
                title=(
                    "Eventos significativos "
                    "por mes"
                ),
                labels={
                    "fecha_modelo": "Fecha",
                    "eventos": "Eventos ≥ 3.5",
                },
                color="eventos",
                color_continuous_scale="Reds",
            )

            st.plotly_chart(
                figura_significativos,
                use_container_width=True,
            )

    with tab2:
        calendario = (
            datos_filtrados
            .groupby(
                [
                    "año",
                    "mes",
                ]
            )
            .size()
            .reset_index(
                name="cantidad"
            )
        )

        tabla_calendario = (
            calendario
            .pivot(
                index="año",
                columns="mes",
                values="cantidad",
            )
            .fillna(0)
        )

        figura_calendario = px.imshow(
            tabla_calendario,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="YlOrRd",
            title=(
                "Mapa de calor de actividad "
                "por año y mes"
            ),
            labels={
                "x": "Mes",
                "y": "Año",
                "color": "Sismos",
            },
        )

        st.plotly_chart(
            figura_calendario,
            use_container_width=True,
        )

        dias = {
            0: "Lunes",
            1: "Martes",
            2: "Miércoles",
            3: "Jueves",
            4: "Viernes",
            5: "Sábado",
            6: "Domingo",
        }

        actividad_dias = (
            datos_filtrados
            .groupby("dia_semana")
            .size()
            .reset_index(
                name="cantidad"
            )
        )

        actividad_dias["día"] = (
            actividad_dias[
                "dia_semana"
            ].map(dias)
        )

        figura_dias = px.bar(
            actividad_dias,
            x="día",
            y="cantidad",
            title="Actividad por día de la semana",
            labels={
                "día": "Día",
                "cantidad": "Sismos",
            },
            color="cantidad",
            color_continuous_scale="Teal",
        )

        st.plotly_chart(
            figura_dias,
            use_container_width=True,
        )

    with tab3:
        actividad_horaria = (
            datos_filtrados
            .groupby("hora")
            .size()
            .reset_index(
                name="cantidad"
            )
        )

        figura_horaria = px.bar(
            actividad_horaria,
            x="hora",
            y="cantidad",
            title="Actividad por hora UTC",
            labels={
                "hora": "Hora UTC",
                "cantidad": "Sismos",
            },
            color="cantidad",
            color_continuous_scale="Electric",
        )

        st.plotly_chart(
            figura_horaria,
            use_container_width=True,
        )

        intervalo = (
            datos_filtrados[
                "fecha_modelo"
            ]
            .sort_values()
            .diff()
            .dt.total_seconds()
            / 3600
        )

        intervalo = intervalo.dropna()

        if not intervalo.empty:
            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Intervalo mediano",
                f"{intervalo.median():.2f} horas",
            )

            col2.metric(
                "Intervalo promedio",
                f"{intervalo.mean():.2f} horas",
            )

            col3.metric(
                "Intervalo máximo",
                f"{intervalo.max():.2f} horas",
            )

            figura_intervalo = px.histogram(
                intervalo,
                nbins=40,
                title=(
                    "Tiempo entre registros sísmicos"
                ),
                labels={
                    "value": "Horas",
                    "count": "Intervalos",
                },
            )

            st.plotly_chart(
                figura_intervalo,
                use_container_width=True,
            )

    st.info(
        "Los patrones temporales describen el catálogo histórico. "
        "No constituyen una predicción determinista de futuros "
        "terremotos."
    )


# -------------------------------------------------------------------
# Mapa histórico
# -------------------------------------------------------------------

elif opcion == "Mapa de sismos":
    st.subheader(
        "🌎 Mapa histórico de actividad sísmica"
    )

    col1, col2 = st.columns(2)

    with col1:
        magnitud_minima = st.slider(
            "Magnitud mínima",
            min_value=0.0,
            max_value=8.0,
            value=0.0,
            step=0.1,
        )

    with col2:
        profundidad_maxima = st.slider(
            "Profundidad máxima",
            min_value=0.0,
            max_value=700.0,
            value=700.0,
            step=10.0,
        )

    mapa = df.dropna(
        subset=[
            "latitude",
            "longitude",
            "magnitude",
            "depth",
        ]
    ).copy()

    mapa = mapa[
        mapa["latitude"].between(-90, 90)
        & mapa["longitude"].between(-180, 180)
    ]

    mapa = mapa[
        (mapa["magnitude"] >= magnitud_minima)
        & (mapa["depth"] <= profundidad_maxima)
    ]

    st.metric(
        "Epicentros encontrados",
        f"{len(mapa):,}",
    )

    if mapa.empty:
        st.warning(
            "No hay datos con los filtros seleccionados."
        )
    else:
        figura = px.scatter_map(
            mapa,
            lat="latitude",
            lon="longitude",
            color="magnitude",
            size="magnitude",
            hover_name="place",
            hover_data={
                "magnitude": True,
                "depth": True,
                "local_time": True,
                "latitude": False,
                "longitude": False,
            },
            color_continuous_scale="YlOrRd",
            size_max=12,
            zoom=4,
            height=650,
        )

        figura.update_layout(
            map_style="open-street-map",
            margin={
                "r": 0,
                "t": 20,
                "l": 0,
                "b": 0,
            },
        )

        st.plotly_chart(
            figura,
            use_container_width=True,
        )


# -------------------------------------------------------------------
# Análisis temporal
# -------------------------------------------------------------------

elif opcion == "Análisis temporal":
    st.subheader(
        "📈 Análisis temporal de actividad sísmica"
    )

    datos = df.copy()

    datos["fecha_modelo"] = pd.to_datetime(
        datos["fecha_modelo"],
        errors="coerce",
        utc=True,
    )

    datos = datos.dropna(
        subset=["fecha_modelo"]
    ).copy()

    datos["año"] = (
        datos[
            "fecha_modelo"
        ].dt.year
    )

    resumen = (
        datos.groupby("año")
        .size()
        .reset_index(
            name="cantidad"
        )
    )

    figura = px.bar(
        resumen,
        x="año",
        y="cantidad",
        title="Actividad sísmica anual",
    )

    st.plotly_chart(
        figura,
        use_container_width=True,
    )


# -------------------------------------------------------------------
# Predicción ML
# -------------------------------------------------------------------

elif opcion == "Predicción ML":
    st.subheader(
        "🤖 Pronóstico experimental por zonas"
    )

    st.warning(
        "Este resultado es un score estadístico "
        "para ordenar zonas. No es una alerta "
        "sísmica oficial ni garantiza que ocurra "
        "un terremoto."
    )

    try:
        (
            modelo,
            columnas_modelo,
            metricas,
            tasas_historicas,
        ) = cargar_modelos()

        panel, _ = construir_panel_nacional(
            df,
            incluir_objetivo=False,
        )

        ranking = construir_ranking(
            modelo,
            columnas_modelo,
            tasas_historicas,
            panel,
        )

        if ranking.empty:
            st.error(
                "No fue posible construir el ranking."
            )
            st.stop()

        regiones = sorted(
            ranking["region"]
            .dropna()
            .unique()
            .tolist()
        )

        region = st.selectbox(
            "Selecciona una región",
            ["Todas"] + regiones,
        )

        ranking_filtrado = ranking.copy()

        if region != "Todas":
            ranking_filtrado = ranking[
                ranking["region"] == region
            ].copy()

        if ranking_filtrado.empty:
            st.warning(
                "No hay zonas para esta región."
            )
            st.stop()

        ultima_semana = ranking_filtrado[
            "semana"
        ].iloc[0]

        fecha_referencia = pd.Timestamp(
            ultima_semana
        ).date()

        fecha_inicio = (
            pd.Timestamp(ultima_semana)
            + pd.Timedelta(days=7)
        ).date()

        fecha_fin = (
            pd.Timestamp(ultima_semana)
            + pd.Timedelta(days=13)
        ).date()

        zona_principal = ranking_filtrado.iloc[0]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Zona priorizada",
            zona_principal[
                "zona_nombre"
            ],
        )

        col2.metric(
            "Score relativo",
            porcentaje(
                zona_principal[
                    "score_calibrado"
                ]
            ),
        )

        col3.metric(
            "Tasa histórica",
            porcentaje(
                zona_principal[
                    "tasa_historica"
                ]
            ),
        )

        col4.metric(
            "Lift histórico",
            (
                f"{zona_principal['lift']:.2f}x"
                if zona_principal["lift"] > 0
                else "N/D"
            ),
        )

        st.info(
            f"Semana de referencia: "
            f"{fecha_referencia}. "
            f"Periodo estimado: "
            f"{fecha_inicio} hasta {fecha_fin}. "
            f"Magnitud objetivo: ≥ "
            f"{MAGNITUD_OBJETIVO}."
        )

        st.subheader(
            "Ranking de zonas"
        )

        tabla = crear_tabla_ranking(
            ranking_filtrado.head(15)
        )

        st.dataframe(
            tabla,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader(
            "Mapa de zonas priorizadas"
        )

        mapa = ranking_filtrado.dropna(
            subset=[
                "lat_celda",
                "lon_celda",
                "score_calibrado",
            ]
        ).copy()

        if not mapa.empty:
            mapa["tamano"] = (
                mapa["score_calibrado"]
                * 100
            ).clip(lower=8)

            figura = px.scatter_map(
                mapa,
                lat="lat_celda",
                lon="lon_celda",
                color="score_calibrado",
                size="tamano",
                hover_name="zona_nombre",
                hover_data={
                    "region": True,
                    "score_calibrado": ":.2%",
                    "tasa_historica": ":.2%",
                    "diferencia_historica": ":.2%",
                    "lift": ":.2f",
                    "lat_celda": False,
                    "lon_celda": False,
                    "tamano": False,
                },
                color_continuous_scale="Turbo",
                size_max=28,
                zoom=5,
                height=650,
            )

            figura.update_layout(
                map_style="open-street-map",
                margin={
                    "r": 0,
                    "t": 20,
                    "l": 0,
                    "b": 0,
                },
            )

            st.plotly_chart(
                figura,
                use_container_width=True,
            )

        st.subheader(
            "Evaluación del modelo"
        )

        st.caption(
            "Estas métricas se calculan sobre el conjunto "
            "de prueba. Precision y recall usan el umbral "
            "seleccionado durante la validación."
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Precision",
            porcentaje(
                metricas.get(
                    "precision"
                )
            ),
        )

        col2.metric(
            "Recall",
            porcentaje(
                metricas.get(
                    "recall"
                )
            ),
        )

        col3.metric(
            "F1-score",
            porcentaje(
                metricas.get(
                    "f1"
                )
            ),
        )

        col4.metric(
            "Accuracy",
            porcentaje(
                metricas.get(
                    "accuracy"
                )
            ),
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Precision@5",
            porcentaje(
                metricas.get(
                    "precision_at_5"
                )
            ),
        )

        col2.metric(
            "Recall@5",
            porcentaje(
                metricas.get(
                    "recall_at_5"
                )
            ),
        )

        col3.metric(
            "Precision@10",
            porcentaje(
                metricas.get(
                    "precision_at_10"
                )
            ),
        )

        col4.metric(
            "Recall@10",
            porcentaje(
                metricas.get(
                    "recall_at_10"
                )
            ),
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Brier modelo",
            numero(
                metricas.get(
                    "brier_modelo"
                )
            ),
        )

        col2.metric(
            "Brier línea base",
            numero(
                metricas.get(
                    "brier_baseline"
                )
            ),
        )

        col3.metric(
            "ROC-AUC",
            numero(
                metricas.get(
                    "roc_auc"
                )
            ),
        )

        col4.metric(
            "Average Precision",
            numero(
                metricas.get(
                    "average_precision"
                )
            ),
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Umbral utilizado",
                numero(
                    metricas.get(
                        "umbral"
                    )
                ),
            )

        with col2:
            st.metric(
                "Mejora Brier",
                (
                    f"{float(metricas['mejora_brier']):+.6f}"
                    if metricas.get(
                        "mejora_brier"
                    ) is not None
                    else "N/D"
                ),
            )

        if metricas.get(
            "modelo_tiene_mejora",
            False,
        ):
            st.success(
                "El modelo mejora la línea base "
                "histórica."
            )
        else:
            st.warning(
                "El modelo presenta utilidad para "
                "ordenar zonas, pero todavía no "
                "mejora la calibración de la línea "
                "base histórica."
            )

        st.info(
            "Interpretación: el modelo obtuvo una "
            f"precision de {porcentaje(metricas.get('precision'))} "
            "sobre el conjunto de prueba. El score "
            "se utiliza principalmente para priorizar "
            "zonas y no representa una probabilidad "
            "exacta de terremoto."
        )

        with st.expander(
            "Ver información técnica"
        ):
            st.json(
                {
                    "Modelo": metricas.get(
                        "modelo"
                    ),
                    "Región": metricas.get(
                        "region"
                    ),
                    "Magnitud objetivo": metricas.get(
                        "magnitud_objetivo"
                    ),
                    "Nivel de confianza": metricas.get(
                        "nivel_confianza"
                    ),
                    "Umbral": metricas.get(
                        "umbral"
                    ),
                    "Método del umbral": metricas.get(
                        "metodo_umbral"
                    ),
                    "Registros totales": metricas.get(
                        "registros_totales"
                    ),
                    "Registros de entrenamiento": (
                        metricas.get(
                            "registros_entrenamiento"
                        )
                    ),
                    "Registros de validación": (
                        metricas.get(
                            "registros_validacion"
                        )
                    ),
                    "Registros de prueba": metricas.get(
                        "registros_prueba"
                    ),
                    "Eventos totales": metricas.get(
                        "eventos_totales"
                    ),
                    "Celdas": metricas.get(
                        "celdas"
                    ),
                    "Semanas": metricas.get(
                        "semanas"
                    ),
                }
            )

        st.caption(
            "El score permite ordenar zonas según "
            "su señal estadística relativa. No predice "
            "la fecha, ubicación o magnitud exacta de "
            "un terremoto."
        )

    except FileNotFoundError:
        st.error(
            "No se encontraron los archivos del modelo."
        )

        st.code(
            "python -m ml.entrenar_nacional",
            language="powershell",
        )

    except Exception as error:
        st.error(
            f"No fue posible ejecutar el modelo: {error}"
        )