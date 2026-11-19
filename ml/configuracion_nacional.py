REGION_NOMBRE_NACIONAL = "Nacional"

MAGNITUD_SIGNIFICATIVA = 3.5

TAMANO_CELDA = 0.5

INCLUIR_ZONAS_MARITIMAS = True


# Cuadrícula amplia para Colombia y zonas marítimas
if INCLUIR_ZONAS_MARITIMAS:
    LATITUD_MINIMA = -4.5
    LATITUD_MAXIMA = 13.5
    LONGITUD_MINIMA = -81.5
    LONGITUD_MAXIMA = -66.5
else:
    LATITUD_MINIMA = -4.5
    LATITUD_MAXIMA = 13.5
    LONGITUD_MINIMA = -79.5
    LONGITUD_MAXIMA = -66.5


# Regiones aproximadas para análisis comparativo.
# La predicción principal sigue utilizando
# el modelo nacional compartido.
REGIONES = {
    "Nacional": {
        "lat_min": LATITUD_MINIMA,
        "lat_max": LATITUD_MAXIMA,
        "lon_min": LONGITUD_MINIMA,
        "lon_max": LONGITUD_MAXIMA
    },

    "Santander": {
        "lat_min": 6.7,
        "lat_max": 7.5,
        "lon_min": -74.2,
        "lon_max": -73.0
    },

    "Caribe": {
        "lat_min": 7.5,
        "lat_max": 13.5,
        "lon_min": -75.5,
        "lon_max": -71.0
    },

    "Pacifico": {
        "lat_min": 1.0,
        "lat_max": 8.5,
        "lon_min": -79.5,
        "lon_max": -76.0
    },

    "Andina": {
        "lat_min": 3.0,
        "lat_max": 7.5,
        "lon_min": -76.0,
        "lon_max": -72.0
    },

    "Orinoquia": {
        "lat_min": 2.5,
        "lat_max": 7.5,
        "lon_min": -72.0,
        "lon_max": -66.5
    },

    "Amazonia": {
        "lat_min": -4.5,
        "lat_max": 3.0,
        "lon_min": -75.0,
        "lon_max": -66.5
    }
}