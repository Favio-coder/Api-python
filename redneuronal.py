import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from tensorflow.keras.applications import VGG16 # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau # type: ignore

# Configurar los paths para los datos
base_dir = './Alzheimer_s Dataset'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Creación de un ImageDataGenerator para la carga y augmentación de las imágenes
datagen = ImageDataGenerator(
    rescale=1./255,             # Normaliza los valores de las imágenes al rango [0, 1]
    validation_split=0.2,       # Define el 20% de los datos como datos de validación
    rotation_range=40,          # Rango de grados para rotar las imágenes aleatoriamente
    width_shift_range=0.2,      # Rango de desplazamiento horizontal aleatorio
    height_shift_range=0.2,     # Rango de desplazamiento vertical aleatorio
    shear_range=0.2,            # Rango para aplicar transformaciones de corte
    zoom_range=0.2,             # Rango para aplicar zoom aleatorio
    horizontal_flip=True,       # Volteo horizontal aleatorio
    fill_mode='nearest'         # Método de llenado para las nuevas píxeles creadas tras la transformación
)

# Generadores de datos para el conjunto de entrenamiento y validación
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Definir el modelo con Transfer Learning usando VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
for layer in base_model.layers:
    layer.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

# Compilar el modelo
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Ajuste de la tasa de aprendizaje
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)

# Entrenar el modelo
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=[lr_scheduler]
)

# Graficar resultados
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(20)

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

# Cargar y predecir en una imagen de prueba
test_image_path = './Alzheimer_s Dataset/test/NonDemented/26 (65).jpg'
test_image = Image.open(test_image_path).resize((150, 150))
test_image = np.array(test_image) / 255.0

# Si es escala de grises
if len(test_image.shape) == 2:
    test_image = np.stack((test_image,)*3, axis=-1)
# Si es color
else:
    test_image = test_image.reshape(-1, 150, 150, 3)

# Realizar predicción
prediction = model.predict(np.expand_dims(test_image, axis=0))
predicted_label = np.argmax(prediction)

# Etiqueta predicha
if predicted_label == 0:
    etiqueta = "No demencia"
elif predicted_label == 1:
    etiqueta = "Demencia muy temprana"
elif predicted_label == 2:
    etiqueta = "Demencia temprana"
elif predicted_label == 3:
    etiqueta = "Demencia moderada"

print("La imagen de prueba pertenece a la clase:", etiqueta)

# Guardar modelo entrenado
model.save("escanner_entrenado_mejorado.h5")
