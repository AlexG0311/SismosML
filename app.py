from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from database import consultar_datos

from ml.preparacion_celdas import (
    construir_datos_celdas
)


BASE_DIR = Path(__file__).resolve().parent


st.set_page_config(
    page_title="SeismicAI",
    page_icon="🌎",
    layout="wide"
)


@st.cache_resource
def cargar_modelos_celdas():
    carpeta = BASE_DIR / "modelos"

    modelo = joblib.load(
        carpeta / "modelo_celdas_santander.pkl"
    )

    calibrador = joblib.load(
        carpeta / "calibrador_celdas_santander.pkl"
    )

    columnas = joblib.load(
        carpeta / "columnas_celdas_santander.pkl"
    )

    metricas = joblib.load(
        carpeta / "metricas_celdas_santander.pkl"
    )

    importancia = joblib.load(
        carpeta / "importancia_celdas_santander.pkl"
    )

    tasas_historicas = joblib.load(
        carpeta / "tasas_historicas_celdas.pkl"
    )

    return (
        modelo,
        calibrador,
        columnas,
        metricas,
        importancia,
        tasas_historicas
    )


st.title("🌎 SeismicAI")

st.caption(
    "Plataforma de análisis de actividad sísmica en Colombia"
)


st.sidebar.title("Navegación")

opcion = st.sidebar.radio(
    "Selecciona una sección",
    [
        "Inicio",
        "Análisis sísmico",
        "Mapa de sismos",
        "Análisis temporal",
        "Predicción ML"
    ],
    key="seccion"
)


if st.sidebar.button("Actualizar datos"):
    st.session_state.pop("datos_sismos", None)
    consultar_datos.clear()
    cargar_modelos_celdas.clear()


st.sidebar.caption(
    "Los datos se conservan al cambiar de sección."
)


try:
    if "datos_sismos" not in st.session_state:
        with st.spinner(
            "Cargando registros sísmicos..."
        ):
            registros = consultar_datos()
            df = pd.DataFrame(registros)

            if df.empty:
                st.warning(
                    "No hay registros disponibles."
                )
                st.stop()

            df["utc_time"] = pd.to_datetime(
                df["utc_time"],
                errors="coerce",
                utc=True,
                format="mixed"
            )

            df["local_time"] = pd.to_datetime(
                df["local_time"],
                errors="coerce",
                utc=True,
                format="mixed"
            )

            for columna in [
                "magnitude",
                "latitude",
                "longitude",
                "depth"
            ]:
                df[columna] = pd.to_numeric(
                    df[columna],
                    errors="coerce"
                )

            st.session_state["datos_sismos"] = df

    df = st.session_state["datos_sismos"]

except Exception as error:
    st.error(
        f"No se pudieron cargar los datos: {error}"
    )
    st.stop()


if opcion == "Inicio":
    st.subheader("Indicadores generales")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Sismos registrados",
        f"{len(df):,}"
    )

    col2.metric(
        "Magnitud máxima",
        f"{df['magnitude'].max():.2f}"
    )

    col3.metric(
        "Profundidad promedio",
        f"{df['depth'].mean():.2f} km"
    )

    col4.metric(
        "Agencias",
        df["agency"].nunique()
    )

    datos = df.dropna(
        subset=["magnitude"]
    )

    fig = px.histogram(
        datos,
        x="magnitude",
        nbins=30,
        title="Distribución de magnitudes"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


elif opcion == "Análisis sísmico":
    st.subheader(
        "Análisis de actividad sísmica"
    )

    datos = df.dropna(
        subset=["utc_time"]
    ).copy()

    datos["año"] = datos["utc_time"].dt.year

    anual = (
        datos.groupby("año")
        .size()
        .reset_index(name="cantidad")
    )

    fig = px.bar(
        anual,
        x="año",
        y="cantidad",
        title="Cantidad de sismos por año",
        text_auto=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


elif opcion == "Mapa de sismos":
    st.subheader(
        "🌎 Mapa de actividad sísmica"
    )

    magnitud_min = st.slider(
        "Magnitud mínima",
        0.0,
        8.0,
        0.0,
        0.1
    )

    profundidad_max = st.slider(
        "Profundidad máxima",
        0.0,
        700.0,
        700.0,
        10.0
    )

    mapa = df.dropna(
        subset=[
            "latitude",
            "longitude",
            "magnitude",
            "depth"
        ]
    ).copy()

    mapa = mapa[
        mapa["latitude"].between(-90, 90)
        & mapa["longitude"].between(-180, 180)
    ]

    mapa = mapa[
        (mapa["magnitude"] >= magnitud_min)
        & (mapa["depth"] <= profundidad_max)
    ]

    st.metric(
        "Epicentros encontrados",
        f"{len(mapa):,}"
    )

    if not mapa.empty:
        fig = px.scatter_map(
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
                "longitude": False
            },
            color_continuous_scale="YlOrRd",
            size_max=12,
            zoom=2,
            height=650
        )

        fig.update_layout(
            map_style="open-street-map",
            margin={
                "r": 0,
                "t": 30,
                "l": 0,
                "b": 0
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


elif opcion == "Análisis temporal":
    st.subheader(
        "📈 Análisis temporal"
    )

    datos = df.dropna(
        subset=[
            "local_time",
            "magnitude",
            "depth"
        ]
    ).copy()

    datos["año"] = datos["local_time"].dt.year

    anual = (
        datos.groupby("año")
        .size()
        .reset_index(name="cantidad")
    )

    fig1 = px.bar(
        anual,
        x="año",
        y="cantidad",
        title="Actividad sísmica anual",
        text_auto=True
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    fig2 = px.histogram(
        datos,
        x="depth",
        nbins=40,
        title="Distribución de profundidades"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )


elif opcion == "Predicción ML":
    st.subheader(
        "🤖 Zonas con mayor probabilidad sísmica"
    )

    st.markdown(
        """
        El modelo analiza Santander-Bucaramanga dividido
        en celdas geográficas.

        La predicción estima qué zonas tienen mayor probabilidad
        relativa de registrar al menos un sismo de magnitud
        igual o superior a 3.5 durante los próximos siete días.
        """
    )

    carpeta = BASE_DIR / "modelos"

    archivos = [
        carpeta / "modelo_celdas_santander.pkl",
        carpeta / "calibrador_celdas_santander.pkl",
        carpeta / "columnas_celdas_santander.pkl",
        carpeta / "metricas_celdas_santander.pkl",
        carpeta / "importancia_celdas_santander.pkl",
        carpeta / "tasas_historicas_celdas.pkl"
    ]

    faltantes = [
        archivo.name
        for archivo in archivos
        if not archivo.exists()
    ]

    if faltantes:
        st.warning(
            "Faltan archivos del modelo:"
        )

        st.code(
            "\n".join(faltantes)
        )

        st.info(
            "Ejecuta primero:\n\n"
            "python -m ml.entrenar_xgboost_celdas"
        )

    else:
        try:
            (
                modelo,
                calibrador,
                columnas_modelo,
                metricas,
                importancia,
                tasas_historicas
            ) = cargar_modelos_celdas()

            datos_celdas, _ = (
                construir_datos_celdas(
                    df,
                    incluir_objetivo=False
                )
            )

            ultima_semana = datos_celdas[
                "semana"
            ].max()

            actuales = datos_celdas[
                datos_celdas["semana"]
                == ultima_semana
            ].copy()

            X_actual = actuales[
                columnas_modelo
            ]

            probabilidades_raw = (
                modelo.predict_proba(
                    X_actual
                )[:, 1]
            )

            if calibrador is not None:
                probabilidades = (
                    calibrador.predict_proba(
                        probabilidades_raw.reshape(-1, 1)
                    )[:, 1]
                )
            else:
                probabilidades = (
                    probabilidades_raw
                )

            actuales["probabilidad"] = (
                probabilidades
            )

            actuales["probabilidad_historica"] = (
                actuales["celda"]
                .map(tasas_historicas)
                .fillna(
                    metricas["probabilidad_base"]
                )
            )

            actuales["diferencia_historica"] = (
                actuales["probabilidad"]
                - actuales["probabilidad_historica"]
            )

            actuales["lift"] = (
                actuales["probabilidad"]
                / actuales[
                    "probabilidad_historica"
                ].replace(0, 0.0001)
            )

            actuales = actuales.sort_values(
                [
                    "probabilidad",
                    "lift"
                ],
                ascending=False
            ).reset_index(drop=True)

            mejor_zona = actuales.iloc[0]

            fecha_inicio = (
                ultima_semana
                + pd.Timedelta(days=7)
            )

            fecha_fin = (
                fecha_inicio
                + pd.Timedelta(days=6)
            )

            st.markdown(
                "### Resultado principal"
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Zona principal",
                str(mejor_zona["zona_nombre"])
            )

            col2.metric(
                "Probabilidad",
                f"{mejor_zona['probabilidad']:.2%}"
            )

            col3.metric(
                "Histórico de la zona",
                f"{mejor_zona['probabilidad_historica']:.2%}"
            )

            col4.metric(
                "Lift",
                f"{mejor_zona['lift']:.2f}x"
            )

            st.markdown(
                "### Ventana de predicción"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Última semana disponible",
                ultima_semana.strftime("%Y-%m-%d")
            )

            col2.metric(
                "Inicio",
                fecha_inicio.strftime("%Y-%m-%d")
            )

            col3.metric(
                "Fin",
                fecha_fin.strftime("%Y-%m-%d")
            )

            if not metricas["modelo_tiene_mejora"]:
                st.error(
                    "El modelo no demuestra una mejora suficiente "
                    "frente a la línea base. El ranking es "
                    "exploratorio."
                )
            else:
                st.success(
                    "El modelo presenta mejora frente a la línea "
                    "base y ordena las zonas por riesgo relativo."
                )

            st.markdown(
                "### Ranking de zonas"
            )

            ranking = actuales[
                [
                    "zona_nombre",
                    "probabilidad",
                    "probabilidad_historica",
                    "diferencia_historica",
                    "lift",
                    "cantidad_sismos",
                    "magnitud_maxima",
                    "sismos_ultimas_4_semanas",
                    "eventos_35_ultimas_4_semanas"
                ]
            ].head(10).copy()

            ranking["probabilidad"] = (
                ranking["probabilidad"]
                .map(lambda x: f"{x:.2%}")
            )

            ranking["probabilidad_historica"] = (
                ranking["probabilidad_historica"]
                .map(lambda x: f"{x:.2%}")
            )

            ranking["diferencia_historica"] = (
                ranking["diferencia_historica"]
                .map(lambda x: f"{x:+.2%}")
            )

            ranking["lift"] = (
                ranking["lift"]
                .map(lambda x: f"{x:.2f}x")
            )

            ranking = ranking.rename(
                columns={
                    "zona_nombre": "Zona",
                    "probabilidad": "Probabilidad",
                    "probabilidad_historica": "Histórico",
                    "diferencia_historica": "Diferencia",
                    "lift": "Lift",
                    "cantidad_sismos": "Sismos registrados",
                    "magnitud_maxima": "Magnitud máxima",
                    "sismos_ultimas_4_semanas": (
                        "Sismos últimas 4 semanas"
                    ),
                    "eventos_35_ultimas_4_semanas": (
                        "Eventos >= 3.5"
                    )
                }
            )

            st.dataframe(
                ranking,
                use_container_width=True,
                hide_index=True
            )

            st.markdown(
                "### Mapa de probabilidad por zona"
            )

            mapa = actuales.copy()

            mapa["tamano"] = (
                mapa["probabilidad"] * 20
            ) + 3

            fig_mapa = px.scatter_map(
                mapa,
                lat="lat_celda",
                lon="lon_celda",
                color="probabilidad",
                size="tamano",
                hover_name="zona_nombre",
                hover_data={
                    "probabilidad": ":.2%",
                    "probabilidad_historica": ":.2%",
                    "diferencia_historica": ":.2%",
                    "lift": ":.2f",
                    "celda": True,
                    "lat_celda": False,
                    "lon_celda": False,
                    "tamano": False
                },
                color_continuous_scale="YlOrRd",
                size_max=24,
                zoom=7,
                height=650,
                title=(
                    "Zonas con mayor probabilidad sísmica"
                )
            )

            fig_mapa.update_layout(
                map_style="open-street-map",
                margin={
                    "r": 0,
                    "t": 50,
                    "l": 0,
                    "b": 0
                }
            )

            st.plotly_chart(
                fig_mapa,
                use_container_width=True
            )

            st.markdown(
                "### Métricas del modelo"
            )

            col1, col2, col3, col4 = st.columns(4)

            roc_auc = metricas.get(
                "roc_auc"
            )

            average_precision = metricas.get(
                "average_precision"
            )

            col1.metric(
                "ROC-AUC",
                (
                    f"{roc_auc:.3f}"
                    if roc_auc is not None
                    else "N/D"
                )
            )

            col2.metric(
                "Average Precision",
                (
                    f"{average_precision:.3f}"
                    if average_precision is not None
                    else "N/D"
                )
            )

            col3.metric(
                "Brier Score",
                f"{metricas['brier_score']:.3f}"
            )

            col4.metric(
                "Mejora Brier",
                f"{metricas['mejora_brier']:+.3f}"
            )

            st.markdown(
                "### Importancia de variables"
            )

            importancia_df = pd.DataFrame(
                {
                    "variable": columnas_modelo,
                    "importancia": importancia
                }
            ).sort_values(
                "importancia",
                ascending=True
            ).tail(15)

            fig_importancia = px.bar(
                importancia_df,
                x="importancia",
                y="variable",
                orientation="h",
                title="Variables más importantes"
            )

            st.plotly_chart(
                fig_importancia,
                use_container_width=True
            )

            with st.expander(
                "Información técnica"
            ):
                st.write(
                    {
                        "Región": metricas["region"],
                        "Celdas": metricas["celdas"],
                        "Semanas": metricas["semanas"],
                        "Magnitud objetivo": metricas[
                            "magnitud_objetivo"
                        ],
                        "Modelo mejora": metricas[
                            "modelo_tiene_mejora"
                        ],
                        "Última semana": str(
                            ultima_semana.date()
                        )
                    }
                )

            st.warning(
                "Esta es una clasificación relativa basada "
                "en registros históricos. No representa una "
                "alerta sísmica ni predice con certeza la "
                "ocurrencia, ubicación o magnitud de un terremoto."
            )

        except Exception as error:
            st.error(
                f"No fue posible ejecutar la predicción: {error}"
            )