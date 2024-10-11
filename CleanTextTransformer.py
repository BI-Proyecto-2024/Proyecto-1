from sklearn.base import BaseEstimator, TransformerMixin
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer


nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

class CleanTextTransformer(BaseEstimator, TransformerMixin):


    stemmer = LancasterStemmer()
    lemmatizer = WordNetLemmatizer()
    stopwords_espaniol = set(stopwords.words('spanish'))

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Aquí puedes aplicar tu proceso de limpieza de texto
        return [' '.join(self.clean_text(text)) for text in X]

    def clean_text(self, text):
        # Ejemplo de limpieza simple: remover caracteres no deseados y convertir a minúsculas
        text= self.corregir_texto(text)
        text= self.corregir_manual(text)
        text= contractions.fix(text)
        text= word_tokenize(text)
        text= self.aplicar_procesamiento(text)
        text= self.raices_y_lemas(text)
        return text

    def aplicar_procesamiento(self,palabras):
        palabras = self.eliminar_no_ascii(palabras)
        palabras = self.a_minusculas(palabras)
        palabras = self.remover_puntuacion(palabras)
        palabras = self.reemplazar_numeros(palabras)
        palabras = self.eliminar_stopwords(palabras)
        return palabras

    def corregir_texto(self,texto):
        texto = ftfy.fix_text(texto)
        return texto

    def corregir_manual(self,texto):
        if isinstance(texto, str):
            # Reemplazos para letras con tildes
            texto = texto.replace('Ã¡', 'á')  # á
            texto = texto.replace('Ã©', 'é')  # é
            texto = texto.replace('Ã­', 'í')  # í
            texto = texto.replace('Ã³', 'ó')  # ó
            texto = texto.replace('Ãº', 'ú')  # ú
            texto = texto.replace('Ã', 'Á')  # Á
            texto = texto.replace('Ã‰', 'É')  # É
            texto = texto.replace('Ã', 'Í')  # Í
            texto = texto.replace('Ã“', 'Ó')  # Ó
            texto = texto.replace('Ãš', 'Ú')  # Ú

            # Reemplazos para la ñ y Ñ
            texto = texto.replace('Ã±', 'ñ')  # ñ
            texto = texto.replace('Ã‘', 'Ñ')  # Ñ

            # Reemplazos para símbolos de comillas y otros
            texto = texto.replace('â€œ', '“')  # Comillas de apertura
            texto = texto.replace('â€', '”')  # Comillas de cierre
            texto = texto.replace('â€˜', '‘')  # Comilla simple apertura
            texto = texto.replace('â€™', '’')  # Comilla simple cierre
            texto = texto.replace('â€“', '–')  # Guion largo
            texto = texto.replace('â€¦', '…')  # Puntos suspensivos

            # Reemplazos para caracteres especiales
            texto = texto.replace('Â¿', '¿')  # ¿
            texto = texto.replace('Â¡', '¡')  # ¡
            texto = texto.replace('Â', '')    # Espacio no deseado (precede algunos caracteres)

            # Otros reemplazos posibles
            texto = texto.replace('â€œ', '"')  # Comillas dobles
            texto = texto.replace('â€', '"')   # Comillas dobles
            texto = texto.replace('â€™', "'")  # Comilla simple
            texto = texto.replace('â€“', '-')  # Guion largo
            texto = texto.replace('â€¦', '...')  # Puntos suspensivos

        return texto

    def eliminar_no_ascii(self, palabras):
        nuevas_palabras = []
        for palabra in palabras:
            nueva_palabra = unicodedata.normalize('NFKD', palabra).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            nuevas_palabras.append(nueva_palabra)
        return nuevas_palabras

    def a_minusculas(self,palabras):
        nuevas_palabras = [palabra.lower() for palabra in palabras]
        return nuevas_palabras

    def remover_puntuacion(self, palabras):
        nuevas_palabras = []
        for palabra in palabras:
            nueva_palabra = re.sub(r'[^\w\s]', '', palabra)
            if nueva_palabra != '':
                nuevas_palabras.append(nueva_palabra)
        return nuevas_palabras

    def reemplazar_numeros(self,palabras):
        p = inflect.engine()
        nuevas_palabras = []
        for palabra in palabras:
            if palabra.isdigit():
             nuevas_palabras.append(p.number_to_words(palabra))
            else:
             nuevas_palabras.append(palabra)
        return nuevas_palabras

    def eliminar_stopwords(self, palabras):
        nuevas_palabras = []
        for palabra in palabras:
            if palabra not in self.stopwords_espaniol:
              nuevas_palabras.append(palabra)
        return nuevas_palabras

    def obtener_raices(self,palabras):
        raices = [stemmer.stem(palabra) for palabra in palabras]
        return raices

    def lematizar_verbos(self,palabras):
        # Lematizar los verbos
        lemas = []
        for palabra in palabras:
            lema = lemmatizer.lemmatize(palabra)
            lemas.append(lema)
        return lemas

    def raices_y_lemas(self,palabras):
        raices = self.obtener_raices(palabras)
        lemas = self.lematizar_verbos(raices)
        return lemas
