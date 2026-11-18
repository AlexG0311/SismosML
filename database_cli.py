import os
import time

import pymysql

from dotenv import load_dotenv


load_dotenv(override=True)


def obtener_conexion():
    host = os.getenv("MYSQLHOST")
    puerto = int(os.getenv("MYSQLPORT", 3306))
    usuario = os.getenv("MYSQLUSER")
    password = os.getenv("MYSQLPASSWORD")
    base_datos = os.getenv("MYSQL_DATABASE")

    if not host or not usuario or not password or not base_datos:
        raise ValueError(
            "Faltan variables de conexión en el archivo .env."
        )

    ultimo_error = None

    for intento in range(3):
        try:
            print(
                f"Intentando conectar con MySQL "
                f"{intento + 1}/3..."
            )

            return pymysql.connect(
                host=host,
                port=puerto,
                user=usuario,
                password=password,
                database=base_datos,
                cursorclass=pymysql.cursors.DictCursor,
                connect_timeout=30,
                read_timeout=300,
                write_timeout=300,
                autocommit=True
            )

        except pymysql.MySQLError as error:
            ultimo_error = error

            if intento < 2:
                time.sleep(3)

    raise ConnectionError(
        "No se pudo conectar con MySQL después de "
        "tres intentos."
    ) from ultimo_error


def consultar_datos_sin_cache():
    conexion = obtener_conexion()

    try:
        with conexion.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    magnitude,
                    latitude,
                    longitude,
                    depth,
                    place,
                    local_time,
                    utc_time,
                    agency
                FROM sismos
                ORDER BY id
                """
            )

            return cursor.fetchall()

    finally:
        conexion.close()