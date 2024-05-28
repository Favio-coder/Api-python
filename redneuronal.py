import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Necesarias para manipular archivos
import io
import os
import re
import shutil
import string
import urllib.request
import zipfile
import io

# Importar las capas y modelos de Keras directamente de TensorFlow
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model # type: ignore

# Configurar los paths para los datos
train_dir_no_demencia = './Alzheimer_s Dataset/train/NonDemented' # Etiquetar como no demencia
train_dir_muy_temprano_demencia = './Alzheimer_s Dataset/train/VeryMildDemented' # Etiquetar como demencia muy temprano
train_dir_demencia_temprana = './Alzheimer_s Dataset/train/MildDemented' # Etiquetar como demencia temprana
train_dir_demencia_moderada= './Alzheimer_s Dataset/train/ModerateDemented' # Etiquetar con demencia moderada 

def load_images_from_folder(folder, label, num_images=3000):
    images = []
    labels = []
    for i, filename in enumerate(os.listdir(folder)):
        if i >= num_images:
            break
        if filename.endswith(('jpg', 'jpeg', 'png')):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).resize((150, 150))  # Redimensionar las imágenes
            img = np.array(img)
            images.append(img)
            labels.append(label)
    return images, labels

# Cargar las imagenes que quieras
images_no_demencia, labels_no_demencia = load_images_from_folder(train_dir_no_demencia, 0)
images_muy_temprano_demencia, labels_muy_temprano_demencia = load_images_from_folder(train_dir_muy_temprano_demencia, 1)
images_demencia_temprana, labels_demencia_temprana = load_images_from_folder(train_dir_demencia_temprana, 2)
images_demencia_moderada, labels_demencia_moderada = load_images_from_folder(train_dir_demencia_moderada, 3)

def show_images(images, labels, num_images=5):
    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i], cmap='gray')  # Si tus imágenes son en escala de grises
        # plt.imshow(images[i])  # Si tus imágenes son en color (RGB)
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    plt.show()

# Mostrar algunas imágenes de cada clase
show_images(images_no_demencia, labels_no_demencia) #0
show_images(images_muy_temprano_demencia, labels_muy_temprano_demencia) #1
show_images(images_demencia_temprana, labels_demencia_temprana) #2
show_images(images_demencia_moderada, labels_demencia_moderada) #3



# Combinar las imágenes y etiquetas en listas
all_images = images_no_demencia + images_muy_temprano_demencia + images_demencia_temprana + images_demencia_moderada
all_labels = labels_no_demencia + labels_muy_temprano_demencia + labels_demencia_temprana + labels_demencia_moderada

all_images = np.array(all_images)
all_labels = np.array(all_labels)

# Normalizar las imágenes
all_images = all_images / 255.0

# Convertir las etiquetas a formato categórico
all_labels = to_categorical(all_labels, num_classes=4)

# Dividir los datos en conjuntos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

# Modificar la forma de los datos de entrada
# Si tus imágenes son en escala de grises
X_train = X_train.reshape(-1, 150, 150, 1)
X_val = X_val.reshape(-1, 150, 150, 1)

# Si tus imágenes son en color (RGB)
# X_train = X_train.reshape(-1, 150, 150, 3)
# X_val = X_val.reshape(-1, 150, 150, 3)

# Definir el modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)),  # Ajustar la entrada al número de canales
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(4, activation='softmax')  # 4 clases
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=200, validation_data=(X_val, y_val))

# Graficar la precisión y la pérdida durante el entrenamiento
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Graficar la precisión y la pérdida durante el entrenamiento
epochs_range = range(200)  # Ajustar el rango de épocas aquí

plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# Cargar una imagen de prueba
test_image_path = './Alzheimer_s Dataset/test/NonDemented/26 (65).jpg'
test_image = Image.open(test_image_path).resize((150, 150))  # Redimensionar la imagen
test_image = np.array(test_image) / 255.0  # Normalizar la imagen

# Modificar la forma de la imagen de prueba
# Si la imagen es en escala de grises
test_image = test_image.reshape(-1, 150, 150, 1)
# Si la imagen es en color (RGB)
# test_image = test_image.reshape(-1, 150, 150, 3)

# Realizar la predicción
prediction = model.predict(test_image)

# Obtener la etiqueta predicha (clase con la probabilidad más alta)
predicted_label = np.argmax(prediction)

if predicted_label == 0:
    etiqueta = "No demencia"
elif predicted_label == 1:
    etiqueta = "Demencia muy temprana"
elif predicted_label == 2:
    etiqueta = "Demencia temprana"
elif predicted_label == 3:
    etiqueta = "Demencia moderada"

# Mostrar la predicción
print("La imagen de prueba pertenece a la clase:", etiqueta)

model.save("escanner_entrenado.h5")

