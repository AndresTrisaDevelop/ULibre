import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def analisis_exploratorio(df):
    """Realiza y visualiza un análisis exploratorio básico."""
    print("--- Análisis Exploratorio Básico ---")
    
    # Publicaciones por país
    plt.figure(figsize=(12, 6))
    sns.countplot(y=df['pais_autor'], order=df['pais_autor'].value_counts().index, palette='viridis')
    plt.title('Número de Publicaciones por País')
    plt.xlabel('Cantidad de Artículos')
    plt.ylabel('País')
    plt.tight_layout()
    plt.savefig('publicaciones_por_pais.png')
    plt.show()
    
    # Publicaciones por año
    df['año'] = pd.to_datetime(df['fecha_publicacion']).dt.year
    plt.figure(figsize=(12, 6))
    sns.countplot(x=df['año'], palette='plasma')
    plt.title('Número de Publicaciones por Año')
    plt.xlabel('Año')
    plt.ylabel('Cantidad de Artículos')
    plt.tight_layout()
    plt.savefig('publicaciones_por_año.png')
    plt.show()

def clustering_tematico(df, num_clusters=4):
    """
    Aplica clustering K-Means para identificar temas en los resúmenes.
    """
    print(f"\n--- Realizando Clustering Temático con {num_clusters} clústeres ---")
    
    # Vectorización TF-IDF de los resúmenes
    # Se ignoran "stop words" comunes en español y palabras que aparecen en menos de 5 documentos.
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=5, ngram_range=(1,2))
    X = vectorizer.fit_transform(df['resumen'])
    
    # Aplicar K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)
    
    # Analizar los términos más importantes por clúster
    print("\n--- Términos más relevantes por clúster ---")
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    for i in range(num_clusters):
        top_terms = [terms[ind] for ind in order_centroids[i, :10]]
        print(f"Cluster {i}: {', '.join(top_terms)}")
        
    return df, X

def visualizar_clusters(df, X):
    """
    Visualiza los clústeres de documentos usando PCA para reducir la dimensionalidad.
    """
    print("\nGenerando visualización de clústeres...")
    # Reducción de dimensionalidad a 2D para poder graficar
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X.toarray())
    
    df_coords = pd.DataFrame(coords, columns=['pca1', 'pca2'])
    df_plot = pd.concat([df_coords, df['cluster']], axis=1)
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df_plot,
        x='pca1',
        y='pca2',
        hue='cluster',
        palette=sns.color_palette('hsv', n_colors=len(df['cluster'].unique())),
        legend='full',
        alpha=0.7
    )
    plt.title('Visualización de Clústeres Temáticos de Artículos (PCA)')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend(title='Clúster')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('visualizacion_clusters.png')
    plt.show()


if __name__ == "__main__":
    # Cargar los datos
    try:
        df = pd.read_csv('articulos_educacion_online.csv')
    except FileNotFoundError:
        print("Error: El archivo 'articulos_educacion_online.csv' no fue encontrado.")
        print("Por favor, ejecuta primero 'generar_dataset.py' para crearlo.")
        exit()

    # 1. Análisis exploratorio
    analisis_exploratorio(df.copy()) # Usamos copia para no modificar el df original con el año
    
    # 2. Clustering
    df_clustered, X_tfidf = clustering_tematico(df, num_clusters=4)
    print("\n--- Muestra de datos con clúster asignado ---")
    print(df_clustered[['resumen', 'cluster']].head())
    
    # 3. Visualización de clústeres
    visualizar_clusters(df_clustered, X_tfidf)
    print("Gráficos guardados como 'publicaciones_por_pais.png', 'publicaciones_por_año.png' y 'visualizacion_clusters.png'")