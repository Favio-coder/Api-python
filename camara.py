import os
from PIL import Image, ImageTk
import numpy as np
import cv2
import tkinter as tk
from tensorflow.keras.models import load_model  # type: ignore

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
    if predicted_label == 0:
        etiqueta = "No demencia"
    elif predicted_label == 1:
        etiqueta = "Demencia muy temprana"
    elif predicted_label == 2:
        etiqueta = "Demencia temprana"
    elif predicted_label == 3:
        etiqueta = "Demencia moderada"
    return etiqueta

# Función para capturar una imagen
def capture_image():
    global frame, captured_frame, camera_active
    camera_active = False
    captured_frame = frame.copy()
    img = Image.fromarray(cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGBA))
    imgtk = ImageTk.PhotoImage(image=img)
    camera_label.imgtk = imgtk
    camera_label.configure(image=imgtk)
    capture_button.pack_forget()
    analyze_button.pack(side=tk.LEFT, padx=10)
    retake_button.pack(side=tk.RIGHT, padx=10)

# Función para analizar la imagen capturada
def analyze_image():
    global captured_frame, model
    preprocessed_frame = preprocess_image(captured_frame)
    prediction = model.predict(preprocessed_frame)
    predicted_label = show_prediction(prediction)
    label_text.set(predicted_label)

# Función para volver a la cámara
def retake_image():
    global camera_active
    camera_active = True
    label_text.set("")
    analyze_button.pack_forget()
    retake_button.pack_forget()
    capture_button.pack()

# Cargar el modelo previamente entrenado
model = load_model("escanner_entrenado.h5")

# Configurar la ventana de Tkinter
root = tk.Tk()
root.title("Alzheimer Prediction")

# Configurar la cámara
cap = cv2.VideoCapture(0)
camera_active = True

label_text = tk.StringVar()
label = tk.Label(root, textvariable=label_text, font=('Helvetica', 16))
label.pack()

# Configurar los botones
capture_button = tk.Button(root, text="Capturar Imagen", command=capture_image)
capture_button.pack()

analyze_button = tk.Button(root, text="Analizar Imagen", command=analyze_image)
retake_button = tk.Button(root, text="Volver a la Cámara", command=retake_image)

# Función para actualizar el frame de la cámara
def update_frame():
    global frame, camera_active
    if camera_active:
        ret, frame = cap.read()
        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            camera_label.imgtk = imgtk
            camera_label.configure(image=imgtk)
    camera_label.after(10, update_frame)

# Configurar el label de la cámara
camera_label = tk.Label(root)
camera_label.pack()

update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
