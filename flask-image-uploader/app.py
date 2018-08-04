import os
import time
import numpy as np
from flask import Flask, render_template, request, Response
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('images')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

classifier = load_model('apple_model.h5')
classifier._make_predict_function()


def get_label(result):
    switcher = {
        0: "Disease",
        1: "Disease",
        2: "Healthy"
    }
    return switcher.get(result, "Invalid Class")


def classify(picture):
    test_image = image.load_img(picture, target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)

    return get_label(result.argmax())


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

    # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)
    file.save(f)
    # return render_template('index.html')

    while not os.path.exists("images/" + file.filename):
        print("Waiting...")
        time.sleep(1)

    if os.path.isfile("images/" + file.filename):
        print("images/" + file.filename)
        return classify("images/" + file.filename)


app.run()
