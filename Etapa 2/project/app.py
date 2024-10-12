from flask import Flask, request, render_template
import requests
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

API_URL = "http://127.0.0.1:8000"  # Cambia esto a la URL de tu API


# Función para verificar extensiones de archivos permitidos
def allowed_file(filename):
    allowed_extensions = {'xlsx','.csv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


@app.route('/')
def index():
    return render_template('indexapp.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text')
    opinion_file = request.files.get('opinion_file')
    files={}
    data={}
    if opinion_file and allowed_file(opinion_file.filename):
        # Guardar archivo de forma temporal
        file_path = os.path.join('temp', secure_filename(opinion_file.filename))
        opinion_file.save(file_path)
        files = {'file': open(file_path, 'rb')}
    else: 
         data = {'text_input': text}    

    response = requests.post(f"{API_URL}/predict", files=files, data=data)
    prediction = response.json()

    return prediction

@app.route('/retrain', methods=['POST'])
def retrain():
    opinion_file = request.files.get('opinion_file')
    files={}
    if opinion_file and allowed_file(opinion_file.filename):
        # Guardar archivo de forma temporal
        file_path = os.path.join('temp', secure_filename(opinion_file.filename))
        opinion_file.save(file_path)
        files = {'file': open(file_path, 'rb')}
     
    response = requests.post(f"{API_URL}/retrain", files=files)
    metrics = response.json()

    return metrics

if __name__ == '__main__':
    app.run(debug=True)