import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Para mejorar el entrenamiento, generando imagenes algo rotadas (mejor generalización)
# Cargar Fashion-MNIST (dataset integrado de Keras)
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalizar
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape (28,28) → (28,28,1) para compatibilidad con CNN
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Clases
class_names = [
    "T-shirt/top","Trouser","Pullover","Dress","Coat",
    "Sandal","Shirt","Sneaker","Bag","Ankle boot"
]

# Construir modelo CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

# Entrenar
# muy simple, para mejorar lo cambio // model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

model.fit(datagen.flow(x_train, y_train, batch_size=32),
          epochs=15,
          validation_data=(x_test, y_test))

# Guardar modelo entrenado
model.save("fashion_mnist_cnn.keras")
print("Modelo guardado como fashion_mnist_cnn.keras")
