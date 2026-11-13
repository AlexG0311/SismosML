# 🌎 SeismicAI — Análisis de Datos Sísmicos en Colombia

**SeismicAI** es un proyecto de ingeniería de datos y aprendizaje automático orientado al procesamiento, almacenamiento, análisis y visualización de información sísmica de Colombia.

El proyecto integra un proceso **ETL (Extract, Transform, Load)**, una base de datos **MySQL alojada en Railway** y una aplicación interactiva desarrollada con **Streamlit**.

Como siguiente etapa, se contempla la integración de modelos de Machine Learning con **XGBoost** para analizar patrones históricos y desarrollar modelos predictivos sobre características de los registros sísmicos.

## 🎯 Objetivo

Construir una solución que permita:

- Extraer y procesar registros sísmicos.
- Seleccionar eventos correspondientes al territorio colombiano y áreas marítimas de interés.
- Limpiar, transformar y validar los datos.
- Almacenar la información estructurada en MySQL.
- Consultar y visualizar registros mediante Streamlit.
- Incorporar modelos de aprendizaje automático para el análisis de patrones sísmicos.

> **Nota:** El aprendizaje automático no permite predecir de manera confiable la fecha, ubicación y magnitud exactas de futuros terremotos. Los modelos de este proyecto se orientan al análisis estadístico de datos históricos.

## 🛠️ Tecnologías utilizadas

| Tecnología | Función |
|---|---|
| Python | Lenguaje principal |
| Pandas | Procesamiento y limpieza de datos |
| MySQL | Almacenamiento de registros |
| Railway | Alojamiento de la base de datos |
| PyMySQL | Conexión entre Python y MySQL |
| Streamlit | Aplicación web interactiva |
| XGBoost | Modelado de Machine Learning (etapa de desarrollo) |
| Git y GitHub | Control de versiones |

## 📂 Estructura del proyecto

```text
SismosML/
│
├── data/
│   ├── raw/                 # Datos originales
│   └── processed/           # Datos transformados
│
├── etl/
│   ├── extract.py
│   ├── seleccionar_geografia.py
│   ├── transform.py
│   └── load.py
│
├── ml/                      # Desarrollo de modelos
├── modelos/                 # Modelos entrenados
│
├── app.py                   # Aplicación Streamlit
├── database.py              # Conexión a MySQL
├── requirements.txt         # Dependencias
├── requirements-lock.txt    # Versiones de referencia
├── .env.example             # Ejemplo de configuración
├── .gitignore
└── README.md
```

Algunas carpetas pueden estar vacías o pendientes de implementación según la etapa de desarrollo.

## 🚀 Instalación y ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/USUARIO/REPOSITORIO.git
cd REPOSITORIO
```

Reemplaza `USUARIO/REPOSITORIO` por la dirección real de tu repositorio en GitHub.

### 2. Crear el entorno virtual

Desde la terminal de Windows (CMD):

```bat
python -m venv .venv
```

### 3. Activar el entorno virtual

**Windows — CMD:**

```bat
.venv\Scripts\activate.bat
```

**Windows — PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 4. Instalar las dependencias

```bash
python -m pip install -r requirements.txt
```

### 5. Configurar las variables de entorno

Crea un archivo llamado `.env` en la raíz del proyecto:

```dotenv
MYSQLHOST=tu_host
MYSQLPORT=3306
MYSQLUSER=tu_usuario
MYSQLPASSWORD=tu_contraseña
MYSQL_DATABASE=tu_base_de_datos
```

Estas credenciales deben corresponder a la instancia de MySQL configurada en Railway.

**Importante:** El archivo `.env` contiene información sensible y no debe subirse a GitHub.

### 6. Ejecutar la aplicación

```bash
python -m streamlit run app.py
```

Streamlit mostrará en la terminal la dirección local de acceso a la aplicación, normalmente:

`http://localhost:8501`

## 🔄 Proceso ETL

El procesamiento de datos se organiza en cuatro etapas.

### Extracción

El módulo `etl/extract.py` se utiliza para la extracción y preparación inicial de los datos de origen.

### Selección geográfica

El módulo `etl/seleccionar_geografia.py` permite identificar registros asociados a Colombia y a las áreas geográficas contempladas en el proyecto.

### Transformación

El módulo `etl/transform.py` realiza las operaciones de limpieza, conversión de tipos de datos y preparación de los registros para su almacenamiento.

### Carga

El módulo `etl/load.py` inserta los registros procesados en MySQL por lotes.

El conjunto procesado utilizado durante el desarrollo contiene **399.472 registros sísmicos**, distribuidos en 18 campos.

Para verificar los parámetros de carga de un registro de prueba:

```bash
python -m etl.load --verificar-insert
```

Para realizar la carga de datos:

```bash
python -m etl.load --cargar
```

**Precaución:** La carga requiere disponer previamente del CSV procesado y de una base de datos configurada. Ejecutarla repetidamente puede producir conflictos por claves primarias duplicadas.

## 🗄️ Base de datos

Los registros se almacenan en la tabla `sismos`, cuya estructura contempla información como:

- Identificador del evento.
- Fecha y hora local y UTC.
- Magnitud y tipo de magnitud.
- Latitud y longitud.
- Profundidad.
- Lugar del evento.
- Agencia de origen.
- Indicadores adicionales de calidad y precisión del registro.

**Consideración técnica:** En las consultas SQL se recomienda escribir el nombre de la columna `` `utc_time` `` entre comillas invertidas, para evitar conflictos con la función `UTC_TIME()` de MySQL.

## 🤖 Machine Learning

La etapa de Machine Learning contempla:

1. Definición del objetivo de predicción o clasificación.
2. Selección y preparación de variables.
3. División cronológica de los datos para entrenamiento y evaluación.
4. Entrenamiento de modelos con XGBoost.
5. Evaluación mediante métricas apropiadas.
6. Almacenamiento del modelo entrenado.
7. Integración de resultados en Streamlit.

Esta funcionalidad se encuentra en desarrollo.

## 🔐 Seguridad

El repositorio debe excluir archivos y directorios locales como:

```gitignore
.env
.venv/
__pycache__/
*.pyc
data/raw/
data/processed/
```

Las credenciales de Railway deben configurarse mediante variables de entorno y nunca escribirse directamente en el código fuente.

## 📌 Estado del proyecto

| Componente | Estado |
|---|---|
| Extracción de datos | Implementado |
| Selección geográfica | Implementado |
| Limpieza y transformación | Implementado |
| Integración con MySQL | Implementado |
| Validación de fechas | Verificada |
| Aplicación Streamlit | En desarrollo |
| Modelo XGBoost | Pendiente de integración |

## 📈 Próximas mejoras

- Implementar y evaluar modelos XGBoost.
- Ampliar las visualizaciones interactivas.
- Documentar la metodología de selección geográfica.
- Incorporar métricas de evaluación y análisis de resultados.
- Preparar el despliegue de la aplicación.

---

**SeismicAI** — Proyecto de análisis de datos sísmicos mediante ingeniería de datos y aprendizaje automático.