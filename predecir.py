import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

class_names = [
    "T-shirt/top","Trouser","Pullover","Dress","Coat",
    "Sandal","Shirt","Sneaker","Bag","Ankle boot"
]

# Cargar modelo guardado
model = tf.keras.models.load_model("fashion_mnist_cnn.keras")

# Cargar imagen local y preprocesar
img_path = "/home/rusok/Descargas/ropita.png"
img = image.load_img(img_path, color_mode="grayscale", target_size=(28,28))

plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()

# Preprocesar para predicción
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, 0) / 255.0

predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions)]

print(f"Predicción: {predicted_class}")