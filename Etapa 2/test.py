import joblib
from CleanTextTransformer import CleanTextTransformer

pipeline= joblib.load("modelo_tfidf_randomforest.joblib")

print(pipeline.predict(["Mujeres"]))
print("sadasdas9")