import os
import numpy as np
from flask import Flask, render_template, request, Response
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('images')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def classify(picture):
    classifier = load_model('tomato_model.h5')
    test_image = image.load_img(picture, target_size=(256, 256))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)

    return result


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

    # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)
    file.save(f)
    jpgfile = Image.open("./images/" + file.filename)
    return classify(jpgfile)


app.run()
