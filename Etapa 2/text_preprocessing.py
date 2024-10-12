from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer
import joblib
import ftfy
import re
import unicodedata
import inflect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer, WordNetLemmatizer

stopwords_espaniol = set(stopwords.words('spanish'))
stemmer = LancasterStemmer()
lemmatizer = WordNetLemmatizer()

def corregir_texto(texto):
    return ftfy.fix_text(texto)

def corregir_manual(texto):
    # Reemplazos de caracteres especiales y tildes
    if isinstance(texto, str):
        texto = texto.replace('Ã¡', 'á').replace('Ã©', 'é').replace('Ã­', 'í')
        texto = texto.replace('Ã³', 'ó').replace('Ãº', 'ú').replace('Ã', 'Á')
        texto = texto.replace('Ã‰', 'É').replace('Ã', 'Í').replace('Ã“', 'Ó')
        texto = texto.replace('Ãš', 'Ú').replace('Ã±', 'ñ').replace('Ã‘', 'Ñ')
        texto = texto.replace('â€œ', '“').replace('â€', '”').replace('â€˜', '‘')
        texto = texto.replace('â€™', '’').replace('â€“', '–').replace('â€¦', '…')
        texto = texto.replace('Â¿', '¿').replace('Â¡', '¡').replace('Â', '')
        texto = texto.replace('â€œ', '"').replace('â€', '"').replace('â€™', "'")
        texto = texto.replace('â€“', '-').replace('â€¦', '...')
    return texto

def eliminar_no_ascii(palabras):
    return [unicodedata.normalize('NFKD', palabra).encode('ascii', 'ignore').decode('utf-8', 'ignore') for palabra in palabras]

def a_minusculas(palabras):
    return [palabra.lower() for palabra in palabras]

def remover_puntuacion(palabras):
    return [re.sub(r'[^\w\s]', '', palabra) for palabra in palabras if palabra != '']

def reemplazar_numeros(palabras):
    p = inflect.engine()
    return [p.number_to_words(palabra) if palabra.isdigit() else palabra for palabra in palabras]

def eliminar_stopwords(palabras):
    return [palabra for palabra in palabras if palabra not in stopwords_espaniol]

def obtener_raices(palabras):
    return [stemmer.stem(palabra) for palabra in palabras]

def lematizar_verbos(palabras):
    return [lemmatizer.lemmatize(palabra) for palabra in palabras]

def procesamiento_completo(texto):
    # Corregir y limpiar el texto
    texto = corregir_texto(texto)
    texto = corregir_manual(texto)
    # Tokenizar y aplicar las funciones de procesamiento
    palabras = word_tokenize(texto)
    palabras = eliminar_no_ascii(palabras)
    palabras = a_minusculas(palabras)
    palabras = remover_puntuacion(palabras)
    palabras = reemplazar_numeros(palabras)
    palabras = eliminar_stopwords(palabras)
    palabras = obtener_raices(palabras)
    palabras = lematizar_verbos(palabras)
    return ' '.join(palabras)  # Devuelve el texto procesado como una cadena

def aplicar_procesamiento(x):
    return [procesamiento_completo(texto) for texto in x]