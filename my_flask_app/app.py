import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image
import io
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas

# Cargar el modelo previamente entrenado
model = load_model("model/escanner_entrenado.h5")

# Función para preprocesar la imagen capturada
def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (150, 150))
    normalized_image = resized_image / 255.0
    reshaped_image = normalized_image.reshape(-1, 150, 150, 1)
    return reshaped_image

# Función para mostrar la etiqueta predicha
def show_prediction(prediction):
    predicted_label = np.argmax(prediction)
    labels = ["No demencia", "Demencia muy temprana", "Demencia temprana", "Demencia moderada"]
    return labels[predicted_label]

# Ruta para manejar las solicitudes POST de la aplicación web React
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener la imagen capturada desde la solicitud POST
        data = request.json
        image_data = data['image']
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))

        # Convertir la imagen a un formato utilizable por OpenCV
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Procesar la imagen y realizar la predicción
        preprocessed_image = preprocess_image(image)
        prediction = model.predict(preprocessed_image)
        predicted_label = show_prediction(prediction)

        # Devolver la predicción como JSON
        return jsonify({'prediction': predicted_label})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
