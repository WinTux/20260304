from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
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
        img = image.load_img(file, color_mode="grayscale", target_size=(28,28))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, 0)

        pred = model.predict(img_array)
        label = class_names[np.argmax(pred)]
        return render_template("resultao.html", prediction=label)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)