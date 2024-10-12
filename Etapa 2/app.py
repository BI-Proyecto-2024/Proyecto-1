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
import traceback
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
        probabilities = pipeline.predict_proba(data['Textos_espanol'])
        data["words"]=aplicar_procesamiento(data["Textos_espanol"])
        data["sdg"]  =predictions

        # Convertimos las predicciones y probabilidades a una lista
        predictions_list = predictions.tolist()
        probabilities_list = probabilities.tolist()
        print(probabilities)
        # Obtenemos las clases (grupos) del pipeline
        class_labels = pipeline.classes_.tolist()

        # Creamos una lista para almacenar las probabilidades junto con las keywords
        results = []

        for i in range(len(predictions_list)):
            # Creamos un diccionario para cada texto
            palabras= [palabra for lista_palabras in data['words'] for palabra in lista_palabras]
            result = {
                'texto': palabras,  # Guarda el texto original
                'prediccion': predictions_list[i],
                'probabilidades': {class_labels[j]: probabilities[i][j] for j in range(len(class_labels))}
            }
            results.append(result)
        return JSONResponse(content={"results": results}, status_code=200)

    except Exception as e:
        traceback.print_exc()
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
            new_data = pd.DataFrame({"text": texts, "label": labels})

            # Guardamos los nuevos datos para historial (opcional)
            if not os.path.exists('content'):
                os.makedirs('content')
            new_data.to_excel("content/new_training_data.xlsx", index=False)

            # Reentrenar el modelo
            pipeline.fit(new_data["text"], new_data["label"])

            # Persistimos el modelo actualizado
            joblib.dump(pipeline, 'pipeline_2.pkl')

            # Realiza predicciones en los datos nuevos para obtener las métricas
            predictions = pipeline.predict(new_data["text"])

            # Calcula métricas
            precision = precision_score(new_data["label"], predictions, average='weighted')
            recall = recall_score(new_data["label"], predictions, average='weighted')
            f1 = f1_score(new_data["label"], predictions, average='weighted')

            return JSONResponse(content={"metrics": {
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante el reentrenamiento: {e}")