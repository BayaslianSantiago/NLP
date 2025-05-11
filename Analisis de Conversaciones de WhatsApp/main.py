import string
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from funciones import *
import wordcloud


def main():
    df = cargar_conversacion("chat.txt")
    #print(df.head())

    #Aplicar limpieza al DataFrame
    df['mensaje_limpio'] = df['mensaje'].apply(limpiar_texto)

    print(df[['autor', 'mensaje', 'mensaje_limpio']].head())
    #mensajes_mas_frecuentes(df)
    nube_de_palabras(df)
if __name__ == "__main__":
    main()