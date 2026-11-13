from etl.extract import extraer_sismos
from etl.transform import transformar_sismos


# EXTRACCIÓN
df = extraer_sismos()

# TRANSFORMACIÓN
df_limpio = transformar_sismos(df)

print("\n========== RESULTADO ==========")
print("Dimensiones originales:", df.shape)
print("Dimensiones transformadas:", df_limpio.shape)

print("\nProceso finalizado.")