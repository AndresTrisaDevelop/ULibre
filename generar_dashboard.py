import pandas as pd
import json
from jinja2 import Template
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def procesar_datos_para_dashboard(df):
    """
    Procesa el DataFrame para extraer todas las métricas necesarias para el dashboard.
    """
    # --- Métricas Generales ---
    num_articulos = len(df)
    num_paises = df['pais_autor'].nunique()
    
    # --- Análisis Temporal ---
    df['año'] = pd.to_datetime(df['fecha_publicacion']).dt.year
    publicaciones_por_año = df.groupby('año').size().reset_index(name='cantidad')
    
    # --- Análisis Geográfico ---
    publicaciones_por_pais = df['pais_autor'].value_counts().reset_index()
    publicaciones_por_pais.columns = ['pais', 'cantidad']
    
    # --- Análisis de Sentimiento ---
    analyzer = SentimentIntensityAnalyzer()
    df['compound'] = df['resumen'].apply(lambda res: analyzer.get_polarity_scores(res)['compound'])
    df['sentimiento_general'] = df['compound'].apply(
        lambda c: 'Positivo' if c > 0.05 else ('Negativo' if c < -0.05 else 'Neutral')
    )
    distribucion_sentimiento = df['sentimiento_general'].value_counts().reset_index()
    distribucion_sentimiento.columns = ['sentimiento', 'cantidad']
    
    # --- Evolución del Sentimiento ---
    sentimiento_por_año = df.groupby('año')['compound'].mean().reset_index()
    
    # --- Preparar datos para JSON ---
    # Convertimos los dataframes a formatos de diccionario que Chart.js pueda leer fácilmente.
    chart_data = {
        "num_articulos": num_articulos,
        "num_paises": num_paises,
        "fecha_min": df['año'].min(),
        "fecha_max": df['año'].max(),
        "publicaciones_por_año": {
            "labels": publicaciones_por_año['año'].tolist(),
            "data": publicaciones_por_año['cantidad'].tolist()
        },
        "publicaciones_por_pais": {
            "labels": publicaciones_por_pais['pais'].tolist(),
            "data": publicaciones_por_pais['cantidad'].tolist()
        },
        "distribucion_sentimiento": {
            "labels": distribucion_sentimiento['sentimiento'].tolist(),
            "data": distribucion_sentimiento['cantidad'].tolist()
        },
        "evolucion_sentimiento": {
            "labels": sentimiento_por_año['año'].tolist(),
            "data": sentimiento_por_año['compound'].tolist()
        }
    }
    
    return chart_data

def generar_html(data):
    """
    Toma los datos procesados y los inserta en una plantilla HTML.
    """
    # Usamos una plantilla HTML dentro del script para simplicidad.
    # Para proyectos más grandes, esto estaría en un archivo .html separado.
    html_template = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dashboard de Publicaciones sobre Educación Online</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f9; color: #333; margin: 0; padding: 20px; }
            h1, h2 { color: #0056b3; text-align: center; }
            .dashboard-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
            .card { background-color: #fff; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); padding: 20px; }
            .kpi-container { display: flex; justify-content: space-around; text-align: center; margin-bottom: 20px; }
            .kpi { background-color: #e7f1ff; padding: 15px; border-radius: 8px; width: 30%; }
            .kpi h3 { margin: 0; color: #0056b3; }
            .kpi p { font-size: 2em; margin: 5px 0 0 0; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Dashboard: Análisis de Publicaciones sobre Educación Online</h1>
        
        <div class="kpi-container">
            <div class="kpi"><h3>Total Artículos</h3><p>{{ num_articulos }}</p></div>
            <div class="kpi"><h3>Países Analizados</h3><p>{{ num_paises }}</p></div>
            <div class="kpi"><h3>Periodo</h3><p>{{ fecha_min }} - {{ fecha_max }}</p></div>
        </div>

        <div class="dashboard-container">
            <div class="card"><h2>Publicaciones por Año</h2><canvas id="pubPorAnoChart"></canvas></div>
            <div class="card"><h2>Publicaciones por País</h2><canvas id="pubPorPaisChart"></canvas></div>
            <div class="card"><h2>Distribución de Sentimiento</h2><canvas id="sentimientoChart"></canvas></div>
            <div class="card"><h2>Evolución del Sentimiento Promedio</h2><canvas id="evolucionSentimientoChart"></canvas></div>
        </div>

        <script>
            const chartData = {{ chart_data|tojson }};

            new Chart(document.getElementById('pubPorAnoChart'), {
                type: 'line',
                data: { labels: chartData.publicaciones_por_año.labels, datasets: [{ label: 'Nº de Artículos', data: chartData.publicaciones_por_año.data, borderColor: 'rgba(75, 192, 192, 1)', backgroundColor: 'rgba(75, 192, 192, 0.2)', fill: true, tension: 0.1 }] }
            });

            new Chart(document.getElementById('pubPorPaisChart'), {
                type: 'bar',
                data: { labels: chartData.publicaciones_por_pais.labels, datasets: [{ label: 'Nº de Artículos', data: chartData.publicaciones_por_pais.data, backgroundColor: 'rgba(153, 102, 255, 0.6)' }] },
                options: { indexAxis: 'y' }
            });

            new Chart(document.getElementById('sentimientoChart'), {
                type: 'doughnut',
                data: { labels: chartData.distribucion_sentimiento.labels, datasets: [{ data: chartData.distribucion_sentimiento.data, backgroundColor: ['rgba(75, 192, 192, 0.6)', 'rgba(255, 99, 132, 0.6)', 'rgba(201, 203, 207, 0.6)'] }] }
            });

            new Chart(document.getElementById('evolucionSentimientoChart'), {
                type: 'line',
                data: { labels: chartData.evolucion_sentimiento.labels, datasets: [{ label: 'Sentimiento Promedio (Compound)', data: chartData.evolucion_sentimiento.data, borderColor: 'rgba(255, 159, 64, 1)', tension: 0.1 }] }
            });
        </script>
    </body>
    </html>
    """
    template = Template(html_template)
    html_content = template.render(chart_data=data, **data)
    
    with open('dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

def main():
    """Función principal para generar el dashboard."""
    try:
        df = pd.read_csv('articulos_educacion_online.csv')
    except FileNotFoundError:
        print("Error: El archivo 'articulos_educacion_online.csv' no fue encontrado.")
        print("Por favor, ejecuta primero 'generar_dataset.py' para crearlo.")
        return

    print("Procesando datos para el dashboard...")
    datos_dashboard = procesar_datos_para_dashboard(df)
    
    print("Generando archivo 'dashboard.html'...")
    generar_html(datos_dashboard)
    print("\n¡Dashboard generado con éxito! Abre el archivo 'dashboard.html' en tu navegador.")

if __name__ == "__main__":
    # Para ejecutar este script, necesitas instalar jinja2: pip install Jinja2
    main()