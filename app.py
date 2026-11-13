
import streamlit as st
import pandas as pd
import plotly.express as px

from database import consultar_datos

st.set_page_config(
    page_title="SeismicAI",
    page_icon="🌎",
    layout="wide"
)

st.title("🌎 SeismicAI")
st.caption("Plataforma de análisis de actividad sísmica")

st.sidebar.title("Navegación")
opcion = st.sidebar.radio(
    "Selecciona una sección",
    ["Inicio", "Análisis sísmico", "Mapa de sismos", "Análisis temporal"],
    key="seccion"
)
st.caption(f"Sección seleccionada: {opcion}")

if st.sidebar.button("Actualizar datos"):
    st.session_state.pop("datos_sismos", None)
    consultar_datos.clear()
st.sidebar.caption("Los datos se conservan al cambiar de sección.")

try:
    if "datos_sismos" not in st.session_state:
        with st.spinner("Cargando y preparando los registros de la base de datos…"):
            registros = consultar_datos()
            df = pd.DataFrame(registros)

            if df.empty:
                st.warning("No hay registros disponibles.")
                st.stop()

            df["utc_time"] = pd.to_datetime(
                df["utc_time"], errors="coerce", utc=True
            )
            df["local_time"] = pd.to_datetime(
                df["local_time"], errors="coerce"
            )
            for columna in ["magnitude", "latitude", "longitude", "depth"]:
                df[columna] = pd.to_numeric(df[columna], errors="coerce")

            st.session_state["datos_sismos"] = df
    df = st.session_state["datos_sismos"]
except Exception as e:
    st.error(f"No se pudieron cargar los datos de la sección «{opcion}»: {e}")
    st.info("Comprueba la conexión y los datos de MySQL y pulsa «Actualizar datos».")
    st.stop()

try:
    if opcion == "Inicio":
        st.subheader("Indicadores generales")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Sismos registrados", f"{len(df):,}")
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

        st.subheader("Distribución de magnitudes")

        datos = df.dropna(subset=["magnitude"])

        fig = px.histogram(
            datos,
            x="magnitude",
            nbins=30,
            title="Frecuencia de magnitudes sísmicas"
        )

        st.plotly_chart(fig, use_container_width=True)

    elif opcion == "Análisis sísmico":
        st.subheader("Análisis de actividad sísmica")

        datos = df.dropna(subset=["utc_time"]).copy()
        datos["año"] = datos["utc_time"].dt.year

        sismos_por_anio = (
            datos.groupby("año")
            .size()
            .reset_index(name="cantidad")
        )

        fig = px.bar(
            sismos_por_anio,
            x="año",
            y="cantidad",
            title="Cantidad de sismos por año"
        )

        st.plotly_chart(fig, use_container_width=True)


    elif opcion == "Mapa de sismos":
        st.subheader("🌎 Mapa de actividad sísmica")

        st.write(
            "Visualización geográfica de los epicentros "
            "registrados en nuestra base de datos."
        )

        col1, col2 = st.columns(2)

        with col1:
            magnitud_min = st.slider(
                "Magnitud mínima", min_value=0.0, max_value=8.0,
                value=0.0, step=0.1
            )

        with col2:
            profundidad_max = st.slider(
                "Profundidad máxima (km)", min_value=0.0, max_value=700.0,
                value=700.0, step=10.0
            )

        mapa = df.dropna(
            subset=["latitude", "longitude", "magnitude", "depth"]
        ).copy()
        mapa = mapa[
            mapa["latitude"].between(-90, 90)
            & mapa["longitude"].between(-180, 180)
        ]
        mapa = mapa[
            (mapa["magnitude"] >= magnitud_min)
            & (mapa["depth"] <= profundidad_max)
        ]

        st.metric("Epicentros encontrados", f"{len(mapa):,}")

        if mapa.empty:
            st.warning("No hay sismos que coincidan con los filtros.")
        else:
            fig = px.scatter_map(
                mapa, lat="latitude", lon="longitude", color="magnitude",
                size="magnitude", hover_name="place",
                hover_data={
                    "magnitude": True, "depth": True, "local_time": True,
                    "latitude": False, "longitude": False
                },
                color_continuous_scale="YlOrRd", size_max=12, zoom=2,
                height=650, title="Distribución geográfica de epicentros"
            )
            fig.update_layout(
                map_style="open-street-map",
                margin={"r": 0, "t": 50, "l": 0, "b": 0}
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Cada punto representa un evento sísmico. "
                "El color y tamaño representan su magnitud."
            )

    elif opcion == "Análisis temporal":
        st.subheader("📈 Análisis temporal de actividad sísmica")

        datos = df.dropna(
            subset=["local_time", "magnitude", "depth"]
        ).copy()

        if datos.empty:
            st.warning("No hay datos temporales completos para analizar.")
        else:
            datos["año"] = datos["local_time"].dt.year

            st.markdown("### 1. Cantidad de sismos por año")
            sismos_anuales = (
                datos.groupby("año").size().reset_index(name="cantidad")
            )
            fig1 = px.bar(
                sismos_anuales, x="año", y="cantidad",
                title="Actividad sísmica anual",
                labels={"año": "Año", "cantidad": "Cantidad de sismos"},
                text_auto=True
            )
            st.plotly_chart(fig1, use_container_width=True)

            st.markdown("### 2. Distribución de profundidades")
            fig2 = px.histogram(
                datos, x="depth", nbins=40,
                title="Distribución de profundidades sísmicas",
                labels={"depth": "Profundidad (km)"}
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("### 3. Magnitud promedio por año")
            magnitud_anual = (
                datos.groupby("año")["magnitude"].mean().reset_index()
            )
            fig3 = px.line(
                magnitud_anual, x="año", y="magnitude", markers=True,
                title="Magnitud promedio anual",
                labels={"año": "Año", "magnitude": "Magnitud promedio"}
            )
            st.plotly_chart(fig3, use_container_width=True)

    


except Exception as e:
    st.error(f"Error al mostrar la sección «{opcion}»: {e}")
