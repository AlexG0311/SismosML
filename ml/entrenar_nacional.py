from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from database_cli import consultar_datos_sin_cache
from ml.preparacion_nacional import (
    MAGNITUD_SIGNIFICATIVA,
    construir_panel_nacional,
)

from xgboost import XGBClassifier

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit


BASE_DIR = Path(__file__).resolve().parent.parent
MODELOS_DIR = BASE_DIR / "modelos"

MODELOS_DIR.mkdir(
    exist_ok=True
)

PRECISION_OBJETIVO = 0.70


def crear_modelo(peso_clase):
    modelo_base = XGBClassifier(
        n_estimators=350,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.80,
        colsample_bytree=0.80,
        min_child_weight=8,
        reg_alpha=0.5,
        reg_lambda=5.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=peso_clase,
        random_state=42,
        n_jobs=-1,
    )

    return CalibratedClassifierCV(
        estimator=modelo_base,
        method="sigmoid",
        cv=TimeSeriesSplit(
            n_splits=5
        ),
    )


def seleccionar_umbral(
    y_validacion,
    probabilidades_validacion,
    precision_objetivo=0.70,
):
    precision, recall, umbrales = (
        precision_recall_curve(
            y_validacion,
            probabilidades_validacion,
        )
    )

    candidatos = []

    for precision_actual, recall_actual, umbral in zip(
        precision[:-1],
        recall[:-1],
        umbrales,
    ):
        if precision_actual >= precision_objetivo:
            candidatos.append(
                {
                    "precision": float(
                        precision_actual
                    ),
                    "recall": float(
                        recall_actual
                    ),
                    "umbral": float(umbral),
                }
            )

    if candidatos:
        # Elegimos el umbral que mantenga
        # la precisión objetivo y maximice recall.
        candidatos.sort(
            key=lambda fila: (
                fila["recall"],
                fila["precision"],
            ),
            reverse=True,
        )

        elegido = candidatos[0]

        return (
            elegido["umbral"],
            elegido["precision"],
            elegido["recall"],
            "precision_objetivo",
        )

    # Si no existe un umbral con 70% de precision,
    # usamos el umbral que maximiza F1.
    mejor_f1 = -1.0
    mejor_umbral = 0.50
    mejor_precision = 0.0
    mejor_recall = 0.0

    for precision_actual, recall_actual, umbral in zip(
        precision[:-1],
        recall[:-1],
        umbrales,
    ):
        f1_actual = (
            2
            * precision_actual
            * recall_actual
            / (
                precision_actual
                + recall_actual
            )
            if (
                precision_actual
                + recall_actual
            )
            > 0
            else 0
        )

        if f1_actual > mejor_f1:
            mejor_f1 = f1_actual
            mejor_umbral = float(umbral)
            mejor_precision = float(
                precision_actual
            )
            mejor_recall = float(
                recall_actual
            )

    return (
        mejor_umbral,
        mejor_precision,
        mejor_recall,
        "maximo_f1",
    )


print("Cargando registros desde MySQL...")

registros = consultar_datos_sin_cache()
df = pd.DataFrame(registros)

if df.empty:
    raise ValueError(
        "La base de datos no contiene registros."
    )


print("Construyendo panel nacional...")

panel, columnas_modelo = (
    construir_panel_nacional(
        df,
        incluir_objetivo=True,
    )
)

panel = panel.sort_values(
    "semana"
).reset_index(
    drop=True
)

X = panel[
    columnas_modelo
].copy()

y = panel[
    "objetivo"
].astype(int)


if y.nunique() < 2:
    raise ValueError(
        "El objetivo solo contiene una clase."
    )


# ---------------------------------------------------------------
# División cronológica por semanas
# ---------------------------------------------------------------

semanas = sorted(
    panel["semana"].unique()
)

total_semanas = len(semanas)

corte_entrenamiento = int(
    total_semanas * 0.60
)

corte_validacion = int(
    total_semanas * 0.80
)

semana_corte_entrenamiento = (
    semanas[corte_entrenamiento]
)

semana_corte_validacion = (
    semanas[corte_validacion]
)

mask_entrenamiento = (
    panel["semana"]
    < semana_corte_entrenamiento
)

mask_validacion = (
    (panel["semana"]
     >= semana_corte_entrenamiento)
    & (
        panel["semana"]
        < semana_corte_validacion
    )
)

mask_prueba = (
    panel["semana"]
    >= semana_corte_validacion
)

X_train = X.loc[
    mask_entrenamiento
]

y_train = y.loc[
    mask_entrenamiento
]

X_val = X.loc[
    mask_validacion
]

y_val = y.loc[
    mask_validacion
]

X_test = X.loc[
    mask_prueba
]

y_test = y.loc[
    mask_prueba
]


if y_train.nunique() < 2:
    raise ValueError(
        "El conjunto de entrenamiento "
        "solo contiene una clase."
    )

if y_val.nunique() < 2:
    raise ValueError(
        "El conjunto de validación "
        "solo contiene una clase."
    )

if y_test.nunique() < 2:
    raise ValueError(
        "El conjunto de prueba "
        "solo contiene una clase."
    )


# ---------------------------------------------------------------
# Peso de clases
# ---------------------------------------------------------------

positivos_train = int(
    y_train.sum()
)

negativos_train = int(
    len(y_train)
    - positivos_train
)

peso_clase = (
    negativos_train
    / positivos_train
    if positivos_train > 0
    else 1.0
)


# ---------------------------------------------------------------
# Entrenamiento con datos históricos
# ---------------------------------------------------------------

print(
    "Entrenando modelo para validación..."
)

modelo_validacion = crear_modelo(
    peso_clase
)

modelo_validacion.fit(
    X_train,
    y_train,
)

probabilidades_val = (
    modelo_validacion
    .predict_proba(X_val)[:, 1]
)


# ---------------------------------------------------------------
# Selección del umbral
# ---------------------------------------------------------------

(
    umbral,
    precision_validacion,
    recall_validacion,
    metodo_umbral,
) = seleccionar_umbral(
    y_val,
    probabilidades_val,
    precision_objetivo=PRECISION_OBJETIVO,
)


print(
    "\nUmbral seleccionado:"
)

print(
    f"Valor: {umbral:.6f}"
)

print(
    f"Método: {metodo_umbral}"
)

print(
    f"Precision validación: "
    f"{precision_validacion:.4f}"
)

print(
    f"Recall validación: "
    f"{recall_validacion:.4f}"
)


# ---------------------------------------------------------------
# Evaluación sobre prueba
# ---------------------------------------------------------------

probabilidades_test = (
    modelo_validacion
    .predict_proba(X_test)[:, 1]
)

predicciones_test = (
    probabilidades_test >= umbral
).astype(int)


precision_test = precision_score(
    y_test,
    predicciones_test,
    zero_division=0,
)

recall_test = recall_score(
    y_test,
    predicciones_test,
    zero_division=0,
)

f1_test = f1_score(
    y_test,
    predicciones_test,
    zero_division=0,
)

accuracy_test = accuracy_score(
    y_test,
    predicciones_test,
)

brier_modelo = brier_score_loss(
    y_test,
    probabilidades_test,
)

probabilidad_base = float(
    y_train.mean()
)

brier_baseline = brier_score_loss(
    y_test,
    np.full(
        len(y_test),
        probabilidad_base,
    ),
)


roc_auc = roc_auc_score(
    y_test,
    probabilidades_test,
)

average_precision = (
    average_precision_score(
        y_test,
        probabilidades_test,
    )
)


# ---------------------------------------------------------------
# Métricas Top-K
# ---------------------------------------------------------------

def calcular_top_k(k):
    resultados = pd.DataFrame(
        {
            "probabilidad": probabilidades_test,
            "real": y_test.to_numpy(),
        }
    )

    resultados = resultados.sort_values(
        "probabilidad",
        ascending=False,
    )

    seleccion = resultados.head(k)

    precision_k = (
        seleccion["real"].mean()
        if len(seleccion) > 0
        else 0
    )

    total_positivos = (
        resultados["real"].sum()
    )

    recall_k = (
        seleccion["real"].sum()
        / total_positivos
        if total_positivos > 0
        else 0
    )

    return (
        float(precision_k),
        float(recall_k),
    )


precision_at_5, recall_at_5 = (
    calcular_top_k(5)
)

precision_at_10, recall_at_10 = (
    calcular_top_k(10)
)


# ---------------------------------------------------------------
# Nivel de confianza
# ---------------------------------------------------------------

mejora_brier = (
    brier_baseline
    - brier_modelo
)

modelo_mejora = (
    mejora_brier >= 0
    and average_precision
    > probabilidad_base
)

if modelo_mejora:
    nivel_confianza = "Media"
else:
    nivel_confianza = "Baja"


metricas = {
    "modelo": "XGBoost calibrado",
    "region": "Nacional",
    "magnitud_objetivo": (
        MAGNITUD_SIGNIFICATIVA
    ),
    "precision_objetivo": (
        PRECISION_OBJETIVO
    ),
    "umbral": float(umbral),
    "metodo_umbral": metodo_umbral,
    "accuracy": float(accuracy_test),
    "precision": float(precision_test),
    "recall": float(recall_test),
    "f1": float(f1_test),
    "brier_modelo": float(brier_modelo),
    "brier_baseline": float(brier_baseline),
    "mejora_brier": float(mejora_brier),
    "probabilidad_base": (
        probabilidad_base
    ),
    "roc_auc": float(roc_auc),
    "average_precision": float(
        average_precision
    ),
    "precision_at_5": (
        precision_at_5
    ),
    "recall_at_5": recall_at_5,
    "precision_at_10": (
        precision_at_10
    ),
    "recall_at_10": recall_at_10,
    "precision_validacion": (
        float(precision_validacion)
    ),
    "recall_validacion": (
        float(recall_validacion)
    ),
    "modelo_tiene_mejora": (
        bool(modelo_mejora)
    ),
    "nivel_confianza": (
        nivel_confianza
    ),
    "registros_totales": int(
        len(panel)
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
    "celdas": int(
        panel["celda"].nunique()
    ),
    "semanas": int(
        panel["semana"].nunique()
    ),
}


print("\nRESULTADOS DEL MODELO")
print("----------------------")

for nombre, valor in metricas.items():
    print(
        f"{nombre}: {valor}"
    )


print("\nMATRIZ DE CONFUSIÓN")

print(
    confusion_matrix(
        y_test,
        predicciones_test,
    )
)


print("\nREPORTE DE CLASIFICACIÓN")

print(
    classification_report(
        y_test,
        predicciones_test,
        zero_division=0,
    )
)


print("\nMÉTRICAS TOP-K")

print(
    f"Precision@5: "
    f"{precision_at_5:.4f}"
)

print(
    f"Recall@5: "
    f"{recall_at_5:.4f}"
)

print(
    f"Precision@10: "
    f"{precision_at_10:.4f}"
)

print(
    f"Recall@10: "
    f"{recall_at_10:.4f}"
)


# ---------------------------------------------------------------
# Entrenamiento definitivo con todos los datos
# ---------------------------------------------------------------

print(
    "\nEntrenando modelo definitivo..."
)

positivos_totales = int(
    y.sum()
)

negativos_totales = int(
    len(y)
    - positivos_totales
)

peso_clase_total = (
    negativos_totales
    / positivos_totales
    if positivos_totales > 0
    else 1.0
)

modelo_final = crear_modelo(
    peso_clase_total
)

modelo_final.fit(
    X,
    y,
)


# Tasas históricas calculadas únicamente
# sobre el periodo de entrenamiento.
tasas_historicas = (
    panel.loc[
        mask_entrenamiento
    ]
    .groupby("celda")["objetivo"]
    .mean()
    .to_dict()
)


# ---------------------------------------------------------------
# Guardar archivos
# ---------------------------------------------------------------

joblib.dump(
    modelo_final,
    MODELOS_DIR
    / "modelo_nacional.pkl",
)

joblib.dump(
    None,
    MODELOS_DIR
    / "calibrador_nacional.pkl",
)

joblib.dump(
    list(columnas_modelo),
    MODELOS_DIR
    / "columnas_nacional.pkl",
)

joblib.dump(
    metricas,
    MODELOS_DIR
    / "metricas_nacional.pkl",
)

joblib.dump(
    tasas_historicas,
    MODELOS_DIR
    / "tasas_historicas_nacionales.pkl",
)


print(
    "\nArchivos generados correctamente en:"
)

print(MODELOS_DIR)