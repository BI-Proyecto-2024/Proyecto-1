from flask import Flask, request, render_template
import requests

app = Flask(__name__)

API_URL = "http://127.0.0.1:8000"  # Cambia esto a la URL de tu API

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text')
    response = requests.post(f"{API_URL}/predict", json={"texts": [text]})
    prediction = response.json()
    return render_template('index.html', prediction=prediction)

@app.route('/retrain', methods=['POST'])
def retrain():
    texts = request.form.getlist('texts')  # Recoge múltiples textos
    labels = request.form.getlist('labels')  # Recoge las etiquetas correspondientes
    response = requests.post(f"{API_URL}/retrain", json={"texts": texts, "labels": labels})
    metrics = response.json()
    return render_template('index.html', metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True)