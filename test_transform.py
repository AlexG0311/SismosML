from etl.extract import extraer_sismos
from etl.transform import transformar_sismos

df = extraer_sismos()

df_limpio = transformar_sismos(df)

print("\nTransformación finalizada correctamente.")