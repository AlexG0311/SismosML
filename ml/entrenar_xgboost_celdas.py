from pathlib import Path

import joblib
import pandas as pd

from database_cli import consultar_datos_sin_cache

from ml.preparacion_celdas import (
    construir_datos_celdas,
    REGION_NOMBRE,
    MAGNITUD_SIGNIFICATIVA
)

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)


BASE_DIR = Path(__file__).resolve().parent.parent
MODELOS_DIR = BASE_DIR / "modelos"

MODELOS_DIR.mkdir(exist_ok=True)


print("Cargando registros desde MySQL...")

registros = consultar_datos_sin_cache()
df = pd.DataFrame(registros)

if df.empty:
    raise ValueError(
        "La base de datos no contiene registros."
    )


print("Construyendo panel espacial...")

datos, columnas_modelo = construir_datos_celdas(
    df,
    incluir_objetivo=True
)

X = datos[columnas_modelo]

y = datos[
    "evento_significativo_siguiente_semana"
]


if y.nunique() < 2:
    raise ValueError(
        "El objetivo solo contiene una clase."
    )


semanas = sorted(
    datos["semana"].unique()
)

total_semanas = len(semanas)

corte_entrenamiento = int(
    total_semanas * 0.60
)

corte_validacion = int(
    total_semanas * 0.80
)

semanas_entrenamiento = semanas[
    :corte_entrenamiento
]

semanas_validacion = semanas[
    corte_entrenamiento:corte_validacion
]

semanas_prueba = semanas[
    corte_validacion:
]

train_mask = datos["semana"].isin(
    semanas_entrenamiento
)

val_mask = datos["semana"].isin(
    semanas_validacion
)

test_mask = datos["semana"].isin(
    semanas_prueba
)

X_train = X.loc[train_mask]
y_train = y.loc[train_mask]

X_val = X.loc[val_mask]
y_val = y.loc[val_mask]

X_test = X.loc[test_mask]
y_test = y.loc[test_mask]


if y_train.nunique() < 2:
    raise ValueError(
        "El entrenamiento solo contiene una clase."
    )

positivos = int(y_train.sum())
negativos = int(
    len(y_train) - positivos
)

peso_clase = (
    negativos / positivos
    if positivos > 0
    else 1.0
)


modelo = XGBClassifier(
    n_estimators=350,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=2.0,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=peso_clase,
    random_state=42
)


print("Entrenando modelo espacial...")

modelo.fit(
    X_train,
    y_train,
    eval_set=[
        (X_val, y_val)
    ],
    verbose=False
)


probabilidades_val_raw = modelo.predict_proba(
    X_val
)[:, 1]

probabilidades_test_raw = modelo.predict_proba(
    X_test
)[:, 1]


calibrador = None

if y_val.nunique() == 2:
    calibrador = LogisticRegression(
        random_state=42
    )

    calibrador.fit(
        probabilidades_val_raw.reshape(-1, 1),
        y_val
    )

    probabilidades_test = (
        calibrador.predict_proba(
            probabilidades_test_raw.reshape(-1, 1)
        )[:, 1]
    )
else:
    probabilidades_test = (
        probabilidades_test_raw
    )


umbral = 0.50

predicciones_test = (
    probabilidades_test >= umbral
).astype(int)


brier_modelo = brier_score_loss(
    y_test,
    probabilidades_test
)

probabilidad_base = float(
    y_train.mean()
)

probabilidades_baseline = [
    probabilidad_base
] * len(y_test)

predicciones_baseline = (
    pd.Series(probabilidades_baseline) >= 0.50
).astype(int)

brier_baseline = brier_score_loss(
    y_test,
    probabilidades_baseline
)

accuracy_baseline = accuracy_score(
    y_test,
    predicciones_baseline
)

mejora_brier = (
    brier_baseline - brier_modelo
)


if y_test.nunique() == 2:
    roc_auc_modelo = roc_auc_score(
        y_test,
        probabilidades_test
    )

    average_precision_modelo = (
        average_precision_score(
            y_test,
            probabilidades_test
        )
    )
else:
    roc_auc_modelo = None
    average_precision_modelo = None


modelo_tiene_mejora = (
    roc_auc_modelo is not None
    and roc_auc_modelo >= 0.50
    and mejora_brier > 0
)


tasas_historicas_celdas = (
    datos.loc[train_mask]
    .groupby("celda")[
        "evento_significativo_siguiente_semana"
    ]
    .mean()
)


metricas = {
    "region": REGION_NOMBRE,
    "accuracy": float(
        accuracy_score(
            y_test,
            predicciones_test
        )
    ),
    "precision": float(
        precision_score(
            y_test,
            predicciones_test,
            zero_division=0
        )
    ),
    "recall": float(
        recall_score(
            y_test,
            predicciones_test,
            zero_division=0
        )
    ),
    "f1": float(
        f1_score(
            y_test,
            predicciones_test,
            zero_division=0
        )
    ),
    "brier_score": float(
        brier_modelo
    ),
    "brier_baseline": float(
        brier_baseline
    ),
    "mejora_brier": float(
        mejora_brier
    ),
    "accuracy_baseline": float(
        accuracy_baseline
    ),
    "roc_auc": (
        float(roc_auc_modelo)
        if roc_auc_modelo is not None
        else None
    ),
    "average_precision": (
        float(average_precision_modelo)
        if average_precision_modelo is not None
        else None
    ),
    "modelo_tiene_mejora": bool(
        modelo_tiene_mejora
    ),
    "umbral": umbral,
    "probabilidad_base": probabilidad_base,
    "registros_totales": int(
        len(datos)
    ),
    "registros_entrenamiento": int(
        len(X_train)
    ),
    "registros_validacion": int(
        len(X_val)
    ),
    "registros_prueba": int(
        len(X_test)
    ),
    "eventos_totales": int(
        y.sum()
    ),
    "magnitud_objetivo": MAGNITUD_SIGNIFICATIVA,
    "celdas": int(
        datos["celda"].nunique()
    ),
    "semanas": int(
        datos["semana"].nunique()
    )
}


print("\nRESULTADOS ESPACIALES")
print("---------------------")

for nombre, valor in metricas.items():
    print(f"{nombre}: {valor}")


print("\nMatriz de confusión")

print(
    confusion_matrix(
        y_test,
        predicciones_test
    )
)


print("\nReporte de clasificación")

print(
    classification_report(
        y_test,
        predicciones_test,
        zero_division=0
    )
)


joblib.dump(
    modelo,
    MODELOS_DIR / "modelo_celdas_santander.pkl"
)

joblib.dump(
    calibrador,
    MODELOS_DIR / "calibrador_celdas_santander.pkl"
)

joblib.dump(
    columnas_modelo,
    MODELOS_DIR / "columnas_celdas_santander.pkl"
)

joblib.dump(
    metricas,
    MODELOS_DIR / "metricas_celdas_santander.pkl"
)

joblib.dump(
    modelo.feature_importances_,
    MODELOS_DIR / "importancia_celdas_santander.pkl"
)

joblib.dump(
    tasas_historicas_celdas,
    MODELOS_DIR / "tasas_historicas_celdas.pkl"
)


print("\nArchivos generados en:")
print(MODELOS_DIR)