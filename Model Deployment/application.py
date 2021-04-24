from __future__ import division, print_function

# coding=utf-8
import os
import cv2
import numpy as np
# Flask utils
from flask import Flask, request, render_template
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
from tensorflow.python.keras.models import load_model
from werkzeug.utils import secure_filename

# Define a flask app
application = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/VGG16_2.h5'
#MODEL_PATH = 'model/resnet50.h5'
ASSETS_FOLDER = os.path.join('static','pics')

# Load your trained model
application.config['UPLOAD_FOLDER'] = ASSETS_FOLDER
model = load_model(MODEL_PATH)
model.make_predict_function()         # Necessary
print('Model loaded. Start serving...')

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img = np.array(img)
    img = cv2.resize(img,(224,224))
    img = img.reshape(1,224,224,3)
    preds = model.predict(img)
    return preds

@application.route('/', methods=['GET'])
def index():
    # Main page
    full_filename = os.path.join(application.config['UPLOAD_FOLDER'], 'china.jpg')
    return render_template('index.html',user_image = full_filename)

@application.route('/team')
def team():
    return render_template('team.html')
@application.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print(file_path)
        # Make prediction
        preds = model_predict(file_path, model)
        print(preds)
        # Process your result for human
        pred_class = preds.argmax(axis=1)            # Simple argmax
        print(pred_class)
        #pred_class = decode_predictions(preds, top=1)  # ImageNet Decode
        #result = str(pred_class[0][0][1])  # Convert to string
        #pred_class = decode_predictions(preds, top=1)  # ImageNet Decode
        #pred_class = decode_predictions(preds, top=1)
        result = str(pred_class[0])  # Convert to string
        if result == "0":
            result="China"
        elif result == "1":
            result="USA"
        return result
    return None


#if __name__ == '__main__':
#   application.run(debug=True)
if __name__ == '__main__':
    application.run(host='0.0.0.0',port=80)
