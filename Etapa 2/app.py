from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import os
from text_preprocessing import aplicar_procesamiento
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum

class RetrainMethod(str, Enum):
    FULL_RETRAINING = "full_retraining"
    INCREMENTAL_RETRAINING = "incremental_retraining"
    TRANSFER_LEARNING = "transfer_learning"


app = FastAPI()

pipeline= joblib.load("pipeline_2.pkl") # Pipeline con el modelo y el preprocesamiento

    

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
    method: RetrainMethod
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
        # Preparamos los datos para el reentrenamiento
        new_data = pd.DataFrame({"text": request.texts, "label": request.labels})

        # Guardamos los nuevos datos para historial
        if not os.path.exists('data'):
            os.makedirs('data')
        new_data.to_csv('data/new_training_data.csv', index=False)

        # Determinamos el método de reentrenamiento
        if request.method == RetrainMethod.FULL_RETRAINING:
            # Reentrenamiento completo
            pipeline.fit(new_data["text"], new_data["label"])

        elif request.method == RetrainMethod.INCREMENTAL_RETRAINING:
            # Reentrenamiento incremental
            # Aquí asumimos que el modelo ya está entrenado y solo se añaden los nuevos datos
            existing_data = pd.read_csv('data/new_training_data.csv')
            all_data = pd.concat([existing_data, new_data], ignore_index=True)
            pipeline.fit(all_data["text"], all_data["label"])

        elif request.method == RetrainMethod.TRANSFER_LEARNING:
            # Transfer learning
            # En este caso, estamos asumiendo que el modelo se ajusta con los nuevos datos
            # La lógica es similar a un reentrenamiento completo, pero se puede ajustar más si se desea
            pipeline.fit(new_data["text"], new_data["label"])

        # Persistimos el modelo actualizado
        joblib.dump(pipeline, "pipeline_2.pkl")

        # Realizamos predicciones en los datos nuevos para obtener las métricas
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