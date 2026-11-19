from pathlib import Path

import joblib
import pandas as pd

from database_cli import consultar_datos_sin_cache
from ml.preparacion_nacional import (
    MAGNITUD_SIGNIFICATIVA,
    construir_panel_nacional,
)


BASE_DIR = Path(__file__).resolve().parent.parent
MODELOS_DIR = BASE_DIR / "modelos"


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


print("Cargando datos desde MySQL...")

registros = consultar_datos_sin_cache()
df = pd.DataFrame(registros)

if df.empty:
    raise ValueError(
        "La base de datos no contiene registros."
    )


print("Construyendo panel de predicción...")

panel, _ = construir_panel_nacional(
    df,
    incluir_objetivo=False,
)

if panel.empty:
    raise ValueError(
        "No fue posible construir el panel nacional."
    )


# Última semana completa disponible.
# Esta semana se utiliza como referencia.
ultima_semana = panel["semana"].max()

datos_actuales = panel[
    panel["semana"] == ultima_semana
].copy()

if datos_actuales.empty:
    raise ValueError(
        "No existen datos para la última semana."
    )


X_actual = datos_actuales.reindex(
    columns=columnas_modelo,
    fill_value=0,
)


# El modelo ya está calibrado.
datos_actuales["score_calibrado"] = (
    modelo.predict_proba(X_actual)[:, 1]
)


datos_actuales["tasa_historica"] = (
    datos_actuales["celda"]
    .map(tasas_historicas)
    .fillna(0)
)


datos_actuales["diferencia_historica"] = (
    datos_actuales["score_calibrado"]
    - datos_actuales["tasa_historica"]
)


datos_actuales["lift"] = (
    datos_actuales["score_calibrado"]
    / datos_actuales[
        "tasa_historica"
    ].replace(0, pd.NA)
)

datos_actuales["lift"] = (
    datos_actuales["lift"].fillna(0)
)


datos_actuales = datos_actuales.sort_values(
    "score_calibrado",
    ascending=False,
).reset_index(drop=True)


datos_actuales.insert(
    0,
    "posicion",
    range(
        1,
        len(datos_actuales) + 1,
    ),
)


# La última semana es la referencia.
fecha_referencia = ultima_semana.date()

# El pronóstico corresponde a la semana siguiente.
fecha_inicio = (
    ultima_semana
    + pd.Timedelta(days=7)
).date()

fecha_fin = (
    ultima_semana
    + pd.Timedelta(days=13)
).date()


print("\nRANKING NACIONAL")
print("----------------")

print(
    "Última semana utilizada como referencia: "
    f"{fecha_referencia}"
)

print(
    "Periodo estimado: "
    f"{fecha_inicio} hasta {fecha_fin}"
)

print(
    "Magnitud objetivo: >= "
    f"{MAGNITUD_SIGNIFICATIVA}"
)

print(
    "Nivel de confianza: "
    f"{metricas.get('nivel_confianza', 'Baja')}"
)

print(
    "Brier del modelo: "
    f"{metricas.get('brier_modelo', 'N/D')}"
)

print(
    "Brier de la línea base: "
    f"{metricas.get('brier_baseline', 'N/D')}"
)


print("\nTOP 10 ZONAS")

columnas_mostrar = [
    "posicion",
    "zona_nombre",
    "region",
    "celda",
    "score_calibrado",
    "tasa_historica",
    "diferencia_historica",
    "lift",
]

columnas_mostrar = [
    columna
    for columna in columnas_mostrar
    if columna in datos_actuales.columns
]

top_10 = datos_actuales[
    columnas_mostrar
].head(10).copy()

print(
    top_10.to_string(
        index=False
    )
)


ruta_salida = (
    MODELOS_DIR
    / "predicciones_nacionales.csv"
)

datos_actuales.to_csv(
    ruta_salida,
    index=False,
    encoding="utf-8-sig",
)

print(
    "\nPredicciones guardadas correctamente en:"
)

print(ruta_salida)