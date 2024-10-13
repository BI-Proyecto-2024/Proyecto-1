from flask import Flask, request, render_template, jsonify
import requests
import os
from werkzeug.utils import secure_filename
from io import BytesIO
import traceback
app = Flask(__name__)

API_URL = "http://127.0.0.1:8000"  # Cambia esto a la URL de tu API


# Función para verificar extensiones de archivos permitidos
def allowed_file(filename):
    allowed_extensions = {'xlsx','.csv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


@app.route('/')
def index():
    return render_template('indexapp.html')

from flask import jsonify
import traceback

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text')
    opinion_file = request.files.get('opinion_file')
    files = {}
    data = {}
    
    try:
        if opinion_file and allowed_file(opinion_file.filename):
            # Guardar archivo de forma temporal
            file_path = os.path.join('temp', secure_filename(opinion_file.filename))
            opinion_file.save(file_path)
            files = {'file': open(file_path, 'rb')}
        else: 
            data = {'text_input': text}    

        response = requests.post(f"{API_URL}/predict", files=files, data=data)
        response.raise_for_status()  # Lanza un error si la respuesta es 4xx o 5xx
        prediction = response.json()
        return jsonify(prediction)  # Asegúrate de devolver el JSON aquí
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500  # Devolver error en formato JSON


@app.route('/retrain', methods=['POST'])
def retrain():
    labels_file = request.files.get('labels_file')
    files={}
    if labels_file and allowed_file(labels_file.filename):
        # Guardar archivo de forma temporal
        file_path = os.path.join('temp', secure_filename(labels_file.filename))
        labels_file.save(file_path)
        files = {'file': open(file_path, 'rb')}
     
    response = requests.post(f"{API_URL}/retrain", files=files)
    metrics = response.json()

    return metrics

if __name__ == '__main__':
    app.run(debug=True)