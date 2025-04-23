import nltk
import string
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, FreqDist
from collections import Counter

#Descargas por unica vez
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

corpus_original = [
    "Python is an interpreted and high-level language, while CPlus is a compiled and low-level language.",
    "JavaScript runs in web browsers, while Python is used in various applications, including data science and artificial intelligence.",
    "JavaScript is dynamically and weakly typed, while Rust is statically typed and ensures greater data security.",
    "Python and JavaScript are interpreted languages, while Java, CPlus, and Rust require compilation before execution.",
    "JavaScript is widely used in web development, while Go is ideal for servers and cloud applications.",
    "Python is slower than CPlus and Rust due to its interpreted nature.",
    "JavaScript has a strong ecosystem with Node.js for backend development, while Python is widely used in data science.",
    "JavaScript does not require compilation, while CPlus and Rust require code compilation before execution.",
    "Python and JavaScript have large communities and an extensive number of available libraries.",
    "Python is ideal for beginners, while Rust and CPlus are more suitable for experienced programmers."
]

def procesamiento(corpus_bruto):
    tokens = word_tokenize(corpus_bruto.lower())  # tokenizamos y pasamos a minusculas
    tokens = [word for word in tokens if word.isalpha()] #sacamos la puntuación
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # sacamos las stopwords
    lematizador = WordNetLemmatizer() #creamos el lematizador
    tokens = [lematizador.lemmatize(word) for word in tokens]  # lematizamos
    return ' '.join(tokens) #aca convertimos la lista de tokens en un string
#esto lo hacemos porque luego TfidfVectorizer va a buscar recibir texto plano y no una lista

corpus_preparado = [procesamiento(w) for w in corpus_original]

print("Corpus preparado:")
for i, frase in enumerate(corpus_preparado, 1):#le damos de parametro el corpus para enumera empezando por el 1
    print(f"{i}: {frase}")

#aca empezamos el TF-IDF
vectorizador = TfidfVectorizer()
tfidf_matrix = vectorizador.fit_transform(corpus_preparado)
vocabulario = vectorizador.get_feature_names_out()

print("\n Matriz TF-IDF (como array):")
print(tfidf_matrix.toarray())
"""recordemos que cada fila representa una oración del corpus 
y cada columna una palabra única del vocalulario
a la hora de leer la matriz lo haremos viendo el vocabulario y la oración"""

print("\n Vocabulario generado:")
print(vocabulario)

todas_las_palabras = ' '.join(corpus_preparado).split() #junta todas las frases, separadas por un espacio
#luego con el split dividimos las palabras en una lista
contador = Counter(todas_las_palabras)
#esta funcion cuenta cuantas veces aparece cada palabra en la lista (desde la libreria 'collections')

palabras_mas_usadas = contador.most_common(6)
print("\n 6 palabras más usadas:")
print(palabras_mas_usadas)

palabra_menos_usada = contador.most_common()[-1]
print("\n Palabra menos usada:")
print(palabra_menos_usada)

print("\n Palabras más repetidas en una misma oración:")
for i, frase in enumerate(corpus_preparado, 1):
    palabras = frase.split()
    repeticiones = Counter(palabras).most_common(1)
    print(f"Oración {i}: {repeticiones}")

# Gráfico de distribución de frecuencia absoluta
frecuencia_absoluta = FreqDist(todas_las_palabras)
plt.figure(figsize=(10,6))
frecuencia_absoluta.plot(30, title="Distribución de Frecuencia de Palabras")
plt.show()