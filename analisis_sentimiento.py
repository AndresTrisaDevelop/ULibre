import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

def analizar_sentimiento(df):
    """
    Aplica análisis de sentimiento y devuelve un NUEVO DataFrame con los resultados.
    """
    # Aunque VADER está optimizado para inglés, funciona aceptablemente para detectar
    # palabras con carga positiva/negativa en textos con cognados o anglicismos.
    # Para un análisis en español más preciso, se usarían librerías como 'pysentimiento' o modelos de Hugging Face.
    analyzer = SentimentIntensityAnalyzer()
    
    # Es una buena práctica no modificar el DataFrame de entrada directamente.
    df_resultado = df.copy()

    # Aplicamos el analizador y expandimos el diccionario resultante en nuevas columnas.
    # Esto es más eficiente que crear un DataFrame intermedio.
    df_resultado[['neg', 'neu', 'pos', 'compound']] = df_resultado['resumen'].apply(
        lambda res: pd.Series(analyzer.polarity_scores(res))
    )
    
    # Clasificamos el sentimiento general basado en el score 'compound'
    df_resultado['sentimiento_general'] = df_resultado['compound'].apply(
        lambda c: 'positivo' if c > 0.05 else ('negativo' if c < -0.05 else 'neutral')
    )
    
    return df_resultado

def visualizar_sentimiento_temporal(df):
    """
    Visualiza la evolución del sentimiento promedio a lo largo de los años.
    """
    # Trabajar con una copia para evitar modificar el DataFrame original (buena práctica)
    df_plot = df.copy()
    df_plot['fecha_publicacion'] = pd.to_datetime(df_plot['fecha_publicacion'])
    df_plot['año'] = df_plot['fecha_publicacion'].dt.year
    
    sentimiento_por_año = df_plot.groupby('año')['compound'].mean().reset_index()
    
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=sentimiento_por_año, x='año', y='compound', marker='o', color='royalblue')
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.title('Evolución del Sentimiento Promedio en Publicaciones sobre Educación Online (2014-2023)', fontsize=16)
    plt.xlabel('Año de Publicación', fontsize=12)
    plt.ylabel('Sentimiento Promedio (Compound Score)', fontsize=12)
    plt.xticks(sentimiento_por_año['año'])
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('evolucion_sentimiento.png')
    plt.show()

def main():
    """Main function to run the sentiment analysis workflow."""
    # Cargar los datos
    try:
        df = pd.read_csv('articulos_educacion_online.csv')
    except FileNotFoundError:
        print("Error: El archivo 'articulos_educacion_online.csv' no fue encontrado.")
        print("Por favor, ejecuta primero 'generar_dataset.py' para crearlo.")
        return # Exit the function

    # 1. Realizar el análisis de sentimiento
    df_con_sentimiento = analizar_sentimiento(df)

    print("--- Muestra de datos con análisis de sentimiento ---")
    print(df_con_sentimiento[['resumen', 'compound', 'sentimiento_general']].head())

    # 2. Contar la distribución de sentimientos
    distribucion = df_con_sentimiento['sentimiento_general'].value_counts()
    print("\n--- Distribución de Sentimientos ---")
    print(distribucion)

    # 3. Visualizar la evolución del sentimiento
    print("\nGenerando gráfico de evolución del sentimiento...")
    visualizar_sentimiento_temporal(df_con_sentimiento)
    print("Gráfico guardado como 'evolucion_sentimiento.png'")

if __name__ == "__main__":
    main()