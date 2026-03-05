from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import io
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Carga modelo
model = tf.keras.models.load_model("fashion_mnist_cnn.keras")

class_names = [
    "T-shirt/top","Trouser","Pullover","Dress","Coat",
    "Sandal","Shirt","Sneaker","Bag","Ankle boot"
]

@app.route("/", methods=["GET", "POST"]) # Cualdo lega petición GET, solo muestro página
def home():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            img_bytes = file.read()
            img = image.load_img(
                io.BytesIO(img_bytes),
                color_mode="grayscale",
                target_size=(28,28)
            )

            # Invierto color porque tenía problemas por el fondo claro ya que el dataset tiene fondo negro
            img_array = image.img_to_array(img)
            img_array = 255 - img_array  # invirtiendo
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, 0)

            pred = model.predict(img_array)
            label = class_names[np.argmax(pred)]

            return render_template("resultado.html", prediction=label)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
