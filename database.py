
import os
import pymysql
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def obtener_conexion():
    return pymysql.connect(
        host=os.getenv("MYSQLHOST"),
        port=int(os.getenv("MYSQLPORT", 3306)),
        user=os.getenv("MYSQLUSER"),
        password=os.getenv("MYSQLPASSWORD"),
        database=os.getenv("MYSQL_DATABASE"),
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=30,
        read_timeout=60,
        write_timeout=60
    )

@st.cache_data(ttl=300)
def consultar_datos():
    conexion = obtener_conexion()

    try:
        with conexion.cursor() as cursor:
            cursor.execute("""
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
            """)

            resultados = cursor.fetchall()

        return resultados

    finally:
        conexion.close()
