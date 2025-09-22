import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings

warnings.filterwarnings("ignore")

def preparar_serie_temporal(df):
    """
    Agrupa las publicaciones por mes para crear una serie temporal.
    """
    df['fecha_publicacion'] = pd.to_datetime(df['fecha_publicacion'])
    # Creamos un índice mensual (Mes-Año)
    df.set_index('fecha_publicacion', inplace=True)
    # Contamos las publicaciones por mes, rellenando meses sin datos con 0
    serie_temporal = df['id_articulo'].resample('MS').count()
    
    # Asegurarnos de que no haya huecos en el índice de tiempo
    idx = pd.date_range(serie_temporal.index.min(), serie_temporal.index.max(), freq='MS')
    serie_temporal = serie_temporal.reindex(idx, fill_value=0)
    
    return serie_temporal

def analizar_y_predecir(serie):
    """
    Analiza la serie temporal con un modelo ARIMA y predice valores futuros.
    """
    print("--- Análisis de la Serie Temporal ---")
    
    # Visualización de la serie
    plt.figure(figsize=(14, 7))
    plt.plot(serie, label='Publicaciones por Mes')
    plt.title('Número de Publicaciones Mensuales sobre Educación Online')
    plt.xlabel('Fecha')
    plt.ylabel('Cantidad de Publicaciones')
    plt.legend()
    plt.savefig('serie_temporal_publicaciones.png')
    plt.show()

    # Gráficos de autocorrelación para ayudar a elegir los parámetros p, d, q de ARIMA
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(serie, ax=ax1, lags=24)
    plot_pacf(serie, ax=ax2, lags=24)
    plt.tight_layout()
    plt.savefig('acf_pacf_plots.png')
    plt.show()

    print("\n--- Ajustando Modelo ARIMA y Realizando Predicción ---")
    # Parámetros (p,d,q) para el modelo ARIMA.
    # p: orden auto-regresivo (lags de ACF)
    # d: orden de diferenciación (para hacer la serie estacionaria)
    # q: orden de media móvil (lags de PACF)
    # Estos valores son un punto de partida y requerirían un análisis más profundo.
    # Usamos (5,1,0) como un ejemplo común.
    modelo = ARIMA(serie, order=(5, 1, 0))
    modelo_ajustado = modelo.fit()
    
    print(modelo_ajustado.summary())
    
    # Realizar predicción para los próximos 24 meses (2 años)
    prediccion = modelo_ajustado.get_forecast(steps=24)
    pred_ci = prediccion.conf_int() # Intervalos de confianza

    # Visualizar la predicción
    plt.figure(figsize=(14, 7))
    ax = serie.plot(label='Observado', color='royalblue')
    prediccion.predicted_mean.plot(ax=ax, label='Predicción', color='darkorange', linestyle='--')
    
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='sandybrown', alpha=.3, label='Intervalo de Confianza')

    ax.set_xlabel('Fecha')
    ax.set_ylabel('Número de Publicaciones')
    plt.title('Predicción de Publicaciones Futuras con ARIMA')
    plt.legend()
    plt.grid(True)
    plt.savefig('prediccion_arima.png')
    plt.show()
    
    print("\nPredicción para los próximos 12 meses:")
    print(prediccion.predicted_mean.head(12))


if __name__ == "__main__":
    # Cargar los datos
    try:
        df = pd.read_csv('articulos_educacion_online.csv')
    except FileNotFoundError:
        print("Error: El archivo 'articulos_educacion_online.csv' no fue encontrado.")
        print("Por favor, ejecuta primero 'generar_dataset.py' para crearlo.")
        exit()

    # 1. Preparar la serie temporal
    serie_publicaciones = preparar_serie_temporal(df)
    
    # 2. Analizar y predecir
    analizar_y_predecir(serie_publicaciones)
    print("\nGráficos guardados como 'serie_temporal_publicaciones.png', 'acf_pacf_plots.png' y 'prediccion_arima.png'")