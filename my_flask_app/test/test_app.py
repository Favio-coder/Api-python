import unittest
import numpy as np
import sys
sys.path.append('../')  # Agregar el directorio 'src' al sys.path

from app import preprocess_image, show_prediction


class TestPreprocessImage(unittest.TestCase):
    def test_preprocess_image(self):
        # Crear una imagen de ejemplo
        image = np.zeros((150, 150, 3), dtype=np.uint8)

        # Probar la función preprocess_image
        processed_image = preprocess_image(image)

        # Verificar que el resultado tenga la forma correcta
        self.assertEqual(processed_image.shape, (1, 150, 150, 1))

class TestShowPrediction(unittest.TestCase):
    def test_show_prediction(self):
        # Crear una predicción de ejemplo
        prediction = np.array([0.1, 0.8, 0.05, 0.05])

        # Probar la función show_prediction
        label = show_prediction(prediction)

        # Verificar que la etiqueta sea la esperada
        self.assertEqual(label, "Demencia muy temprana")

if __name__ == '__main__':
    unittest.main()
