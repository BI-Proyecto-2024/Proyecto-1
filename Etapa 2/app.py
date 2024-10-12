from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import os
from CleanTextTransformer import CleanTextTransformer
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

CleanTextTransformer = CleanTextTransformer()
pipeline= joblib.load("pipeline_2.pkl") # Lo estoy probando con un modelo de otros ODS que me pasaron
#pipeline = joblib.load("best_model.joblib") # Este es el modelo que se generó en la etapa 2
#pipeline = joblib.load("pipeline.joblib")
    

templates = Jinja2Templates(directory="templates")

# Configura CORS para permitir todas las solicitudes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Esto permite todas las solicitudes, pero puedes configurarlo de manera más restrictiva para producción
    allow_credentials=True,
    allow_methods=["*"],  # Puedes especificar los métodos permitidos (GET, POST, etc.)
    allow_headers=["*"],  # Puedes especificar los encabezados permitidos
)

# Modelo de la petición de predicción
class PredictionRequest(BaseModel):
    texts: List[str]
    
# Modelo de la petición de reentrenamiento
class RetrainRequest(BaseModel):
    texts: List[str]
    labels: List[int]
    
@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        
        # Realizamos predicciones con el modelo
        predictions = pipeline.predict(request.texts)
        probabilities = pipeline.predict_proba(request.texts).max(axis=1)

        # Devolvemos las predicciones y sus probabilidades
        return {"predictions": predictions.tolist(), "probabilities": probabilities.tolist()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error haciendo predicción: {e}')
    
@app.post("/retrain")
async def retrain(request: RetrainRequest):
    try:
        # Preparamos los datos para el reetnrenamiento
        new_data = pd.DataFrame({"text": request.texts, "label": request.labels})
        
        # Guardamos los nueos datos para historial
        if not os.path.exists('data'):
            os.makedirs('data')
        new_data.to_csv('data/new_training_data.csv', index=False)
        
        # Reentrena el modelo usando el pipeline con los nuevos datos
        pipeline.fit(new_data["text"], new_data["label"])
        
        # Persistimos el modelo actualizado
        joblib.dump(pipeline, "modelo_tfidf_randomforest.joblib")
        
        # Realizamos predicciones en los datos nueos para obtener las métricas
        predictions = pipeline.predict(new_data["text"])
        
        # Calculamos las métricas
        precision = precision_score(new_data["label"], predictions, average='weighted')
        recall = recall_score(new_data["label"], predictions, average='weighted')
        f1 = f1_score(new_data["label"], predictions, average='weighted')
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error reentrenando modelo: {e}')

