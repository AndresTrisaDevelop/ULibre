import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# --- Configuración ---
NUM_ARTICULOS = 500
FECHA_INICIO = datetime(2014, 1, 1)
FECHA_FIN = datetime(2023, 12, 31)

# --- Listas de palabras clave para generar resúmenes ---
temas_positivos = ['innovación', 'flexibilidad', 'accesibilidad', 'mejora', 'éxito', 'efectividad', 'inclusión', 'oportunidades']
temas_negativos = ['deserción', 'brecha digital', 'aislamiento', 'dificultades', 'costos', 'fraude', 'limitaciones']
temas_neutrales = ['plataformas', 'MOOCs', 'e-learning', 'metodología', 'evaluación', 'tecnología', 'modelo híbrido', 'pedagogía']
conectores = ['además', 'sin embargo', 'por lo tanto', 'en conclusión', 'asimismo', 'en contraste']

def generar_resumen_aleatorio():
    """Genera un resumen simulado con un sentimiento predominante."""
    sentimiento = random.choice(['positivo', 'negativo', 'neutral'])
    palabras = []
    
    if sentimiento == 'positivo':
        palabras.extend(random.sample(temas_positivos, k=min(len(temas_positivos), 3)))
        palabras.extend(random.sample(temas_neutrales, k=min(len(temas_neutrales), 2)))
    elif sentimiento == 'negativo':
        palabras.extend(random.sample(temas_negativos, k=min(len(temas_negativos), 3)))
        palabras.extend(random.sample(temas_neutrales, k=min(len(temas_neutrales), 2)))
    else: # Neutral
        palabras.extend(random.sample(temas_neutrales, k=min(len(temas_neutrales), 4)))

    random.shuffle(palabras)
    
    # Añadir conectores para mayor realismo
    if len(palabras) > 2:
        pos_conector = random.randint(1, len(palabras)-2)
        palabras.insert(pos_conector, random.choice(conectores))

    return f"Este estudio analiza {palabras[0]} y {palabras[1]} en el contexto de la educación online. Se investiga la {palabras[2]} y su impacto en {palabras[3]}. {' '.join(palabras[4:])}."

def generar_datos(n):
    """Genera el DataFrame completo."""
    data = []
    for i in range(n):
        # Generar fecha aleatoria con tendencia a ser más reciente
        dias_rango = (FECHA_FIN - FECHA_INICIO).days
        offset_dias = int(np.sqrt(random.uniform(0, dias_rango**2)))
        fecha_publicacion = FECHA_INICIO + timedelta(days=offset_dias)
        
        data.append({
            'id_articulo': 1000 + i,
            'titulo': f"Estudio sobre {random.choice(temas_neutrales)} en la era digital",
            'resumen': generar_resumen_aleatorio(),
            'fecha_publicacion': fecha_publicacion.strftime('%Y-%m-%d'),
            'pais_autor': random.choice(['España', 'México', 'Colombia', 'Argentina', 'Chile', 'Perú']),
            'citas': random.randint(0, 200)
        })
    return pd.DataFrame(data)

# --- Generar y guardar el archivo CSV ---
if __name__ == "__main__":
    df_articulos = generar_datos(NUM_ARTICULOS)
    df_articulos.to_csv('articulos_educacion_online.csv', index=False, encoding='utf-8')
    print("Archivo 'articulos_educacion_online.csv' generado con éxito.")
    print(df_articulos.head())