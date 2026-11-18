from pathlib import Path

import joblib
import pandas as pd

from database_cli import consultar_datos_sin_cache

from ml.preparacion_celdas import (
    construir_datos_celdas
)


BASE_DIR = Path(__file__).resolve().parent.parent
MODELOS_DIR = BASE_DIR / "modelos"


modelo = joblib.load(
    MODELOS_DIR / "modelo_celdas_santander.pkl"
)

calibrador = joblib.load(
    MODELOS_DIR / "calibrador_celdas_santander.pkl"
)

columnas_modelo = joblib.load(
    MODELOS_DIR / "columnas_celdas_santander.pkl"
)

metricas = joblib.load(
    MODELOS_DIR / "metricas_celdas_santander.pkl"
)

tasas_historicas = joblib.load(
    MODELOS_DIR / "tasas_historicas_celdas.pkl"
)


print("Cargando registros desde MySQL...")

registros = consultar_datos_sin_cache()
df = pd.DataFrame(registros)

if df.empty:
    raise ValueError(
        "La base de datos no contiene registros."
    )


datos, _ = construir_datos_celdas(
    df,
    incluir_objetivo=False
)

ultima_semana = datos["semana"].max()

actuales = datos[
    datos["semana"] == ultima_semana
].copy()

X_actual = actuales[
    columnas_modelo
]

probabilidades_raw = modelo.predict_proba(
    X_actual
)[:, 1]


if calibrador is not None:
    probabilidades = (
        calibrador.predict_proba(
            probabilidades_raw.reshape(-1, 1)
        )[:, 1]
    )
else:
    probabilidades = probabilidades_raw


predicciones = actuales[
    [
        "semana",
        "celda",
        "zona_nombre",
        "lat_celda",
        "lon_celda",
        "cantidad_sismos",
        "magnitud_maxima",
        "sismos_ultimas_4_semanas",
        "eventos_35_ultimas_4_semanas"
    ]
].copy()

predicciones["probabilidad"] = (
    probabilidades
)

predicciones["probabilidad_historica"] = (
    predicciones["celda"]
    .map(tasas_historicas)
    .fillna(
        metricas["probabilidad_base"]
    )
)

predicciones["diferencia_historica"] = (
    predicciones["probabilidad"]
    - predicciones["probabilidad_historica"]
)

predicciones["lift"] = (
    predicciones["probabilidad"]
    / predicciones[
        "probabilidad_historica"
    ].replace(0, 0.0001)
)

predicciones = predicciones.sort_values(
    [
        "probabilidad",
        "lift"
    ],
    ascending=False
).reset_index(drop=True)

predicciones.to_csv(
    MODELOS_DIR
    / "predicciones_celdas_santander.csv",
    index=False
)


print("\nTOP 10 ZONAS")
print("------------")

print(
    predicciones[
        [
            "zona_nombre",
            "probabilidad",
            "probabilidad_historica",
            "diferencia_historica",
            "lift"
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print(
    "\nArchivo guardado en:"
)

print(
    MODELOS_DIR
    / "predicciones_celdas_santander.csv"
)