import joblib
from CleanTextTransformer import CleanTextTransformer

pipeline= joblib.load("Etapa 2\modelo_tfidf_randomforest.joblib")

print(pipeline.predict(["Salud"]))
print("sadasdas9")