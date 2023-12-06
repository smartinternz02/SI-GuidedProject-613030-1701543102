from flask import Flask, render_template, request, flash, redirect, url_for
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

# Loading the pre-trained model "cnn.h5" here
from keras.models import load_model
model_path='C:/Users/DIVYA VARSHINI/OneDrive/Desktop/BOOKS/Smart_Bridge/cnn.h5'
model = tf.keras.models.load_model(model_path)

@app.route('/')
def index():
    return render_template("wce.html")

@app.route('/predict')
def predict():
    return render_template("started.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")


@app.route('/output', methods=['POST'])
def output():
    if request.method == 'POST':
        # Get the file from the POST request
        f = request.files['file']

        if 'file' in request.files:
           f = request.files['file']

           basepath = os.path.dirname(__file__)
           filepath = os.path.join(basepath, 'uploads', f.filename)
           f.save(filepath)

        # Load and preprocess the image
           img = load_img(filepath, target_size=(224, 224))
        # Convert the image to an array and normalize it
           image_array = np.array(img)
        # Add a batch dimension
           image_array = np.expand_dims(image_array, axis=0)

        # Use the pre-trained model to make a prediction
           pred = np.argmax(model.predict(image_array), axis=1)
           index = ['Normal', 'Ulcerative_colitis', 'Polyps', 'Esophagitis']
           prediction = index[int(pred)]

        # Print the prediction (optional)
           print("Prediction:", prediction)

        # Render the template with the prediction
           return render_template("result.html", result=prediction)
    else:
        # Redirect back to the form or show an error message
        return redirect(url_for('predict'))

    # If the request method is not POST, just render the template
    return render_template("result.html", result=None)

if __name__ == '__main__':
    app.run(debug=True)