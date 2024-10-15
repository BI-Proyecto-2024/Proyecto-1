from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import os
from text_preprocessing import aplicar_procesamiento
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO

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
    
# Endpoint para predicción
@app.post("/predict")
async def predict(
    text_input: str = Form(None),
    file: UploadFile = File(None)
):
    try:
        if file:
            # Verificamos si el archivo es un Excel
            if not file.filename.endswith('.xlsx'):
                return JSONResponse(content={"error": "El archivo debe ser un Excel (.xlsx)"}, status_code=400)
            
            # Leemos el archivo Excel
            data = pd.read_excel(BytesIO(file.file.read()), engine='openpyxl')
        
        elif text_input:
            # Creamos un DataFrame con el texto proporcionado
            data = pd.DataFrame({"Textos_espanol": [text_input]})
        
        else:
            return JSONResponse(content={"error": "No se proporcionó texto o archivo"}, status_code=400)
        
        # Realiza predicciones
        predictions = pipeline.predict(data['Textos_espanol'])
        probabilities = pipeline.predict_proba(data['Textos_espanol']).max(axis=1)

        # Convertimos las predicciones y probabilidades a una lista
        predictions_list = predictions.tolist()
        probabilities_list = probabilities.tolist()

        return JSONResponse(content={"predictions": predictions_list, "probabilities": probabilities_list}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": f"Error durante la predicción: {e}"}, status_code=500)

# Endpoint para reentrenamiento
@app.post("/retrain")
async def retrain(file: UploadFile = File(...)):
    try:
        if file:
            if file.filename.endswith('.csv'):
                data = pd.read_csv(BytesIO(await file.read()))
            elif file.filename.endswith('.xlsx'):
                data = pd.read_excel(BytesIO(await file.read()))
            else:
                raise HTTPException(status_code=400, detail="El archivo debe ser un .csv o .xlsx.")

            if 'Textos_espanol' not in data.columns or 'sdg' not in data.columns:
                raise HTTPException(status_code=400, detail="El archivo debe contener las columnas 'Textos_espanol' y 'sdg'.")
            
            texts = data['Textos_espanol'].tolist()
            labels = data['sdg'].tolist()

            # Asegurarse de que hay textos y etiquetas para reentrenar
            if not texts or not labels:
                raise HTTPException(status_code=400, detail="Se debe proporcionar texto y etiquetas para reentrenar.")
            
            # Reentrena el modelo usando el pipeline con los nuevos datos
            new_data = pd.DataFrame({"Textos_espanol": texts, "sdg": labels})

            # Guardamos los nuevos datos para historial (opcional)
            if not os.path.exists('content'):
                os.makedirs('content')
            new_data.to_excel("content/new_training_data.xlsx", index=False)

            # Reentrenar el modelo
            pipeline.fit(new_data["Textos_espanol"], new_data["sdg"])

            # Persistimos el modelo actualizado
            joblib.dump(pipeline, 'pipeline_2.pkl')

            # Realiza predicciones en los datos nuevos para obtener las métricas
            predictions = pipeline.predict(new_data["Textos_espanol"])

            # Calcula métricas
            precision = precision_score(new_data["sdg"], predictions, average='weighted')
            recall = recall_score(new_data["sdg"], predictions, average='weighted')
            f1 = f1_score(new_data["sdg"], predictions, average='weighted')

            return {
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante el reentrenamiento: {e}")