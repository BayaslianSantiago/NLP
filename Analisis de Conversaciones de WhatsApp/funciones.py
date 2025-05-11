import string
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud


#nltk.download('stopwords')
stopwords_es = set(stopwords.words('spanish'))

def cargar_conversacion(path):
    # Regex adaptado al formato "3/3/25 9:25 a. m. - Nombre: Mensaje"
    patron_mensaje = re.compile(
        r'^(\d{1,2}/\d{1,2}/\d{2,4}) (\d{1,2}:\d{2})\s?(a\.?\s?m\.?|p\.?\s?m\.?) - ([^:]+): (.+)', re.IGNORECASE)
    
    datos = []

    with open(path, 'r', encoding='utf-8') as archivo:
        for linea in archivo:
            linea = linea.strip()
            coincidencia = patron_mensaje.match(linea)
            if coincidencia:
                fecha, hora, periodo, autor, mensaje = coincidencia.groups()
                hora_completa = f"{hora} {periodo}"
                datos.append({
                    'fecha': fecha,
                    'hora': hora_completa,
                    'autor': autor,
                    'mensaje': mensaje
                })
            elif datos:
                # Línea continua del mensaje anterior
                datos[-1]['mensaje'] += ' ' + linea

    return pd.DataFrame(datos)

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'http\S+', '', texto)  # eliminar links
    texto = re.sub(r'[^\w\s]', '', texto)  # eliminar signos de puntuación
    texto = re.sub(r'\d+', '', texto)      # eliminar números
    texto = re.sub(r'[^\x00-\x7F]+', '', texto)  # eliminar emojis y caracteres raros
    palabras = texto.split()
    palabras = [p for p in palabras if p not in stopwords_es]
    palabras_excluir = stopwords_es.union({'multimedia', 'omitido','edit','mensaje'})
    palabras = [p for p in palabras if p not in palabras_excluir]

    return ' '.join(palabras)


def mensajes_mas_frecuentes(df):
    # Unir todos los mensajes limpios en un solo string
    texto_total = ' '.join(df['mensaje_limpio'])

    # Dividir en palabras y contarlas
    palabras = texto_total.split()
    conteo = Counter(palabras)

    # Obtener las 10 más comunes
    palabras_comunes = conteo.most_common(10)

    # Mostrar en gráfico de barras
    palabras, frecuencias = zip(*palabras_comunes)

    plt.figure(figsize=(10, 6))
    plt.bar(palabras, frecuencias, color='skyblue')
    plt.title('10 Palabras Más Frecuentes')
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def nube_de_palabras(df):
    # Unir todos los mensajes limpios
    texto_total = ' '.join(df['mensaje_limpio'])

    # Crear la nube de palabras
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=100
    ).generate(texto_total)

    # Mostrarla
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Nube de Palabras de la Conversación')
    plt.show()
    
def nube_de_palabras_por_autor(df):
    autores = df['autor'].unique()

    for autor in autores:
        mensajes_autor = df[df['autor'] == autor]['mensaje_limpio']
        texto = ' '.join(mensajes_autor)

        if texto.strip():  # asegurarse de que haya contenido
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='plasma',
                max_words=100
            ).generate(texto)

            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Nube de Palabras - {autor}')
            plt.show()
            
