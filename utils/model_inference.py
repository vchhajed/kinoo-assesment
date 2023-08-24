# Import necessary libraries
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image
import io
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Preprocess input image
def preprocess_image(image):
    '''Resize image to match model input size'''
    image = image.resize((224, 224)) 
    image_array = np.array(image)
    preprocessed_image = preprocess_input(image_array)
    return preprocessed_image

# Decode model predictions
def decode_predictions(prediction):
    '''Post-processing step'''
    decoded_preds = decode_predictions(prediction, top=3)[0]
    results = [{'label': label, 'description': description, 'probability': probability} for (_, label, probability) in decoded_preds]
    return results

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_data = request.files['image'].read()
        image = Image.open(io.BytesIO(image_data))

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Make predictions
        prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))

        # Decode predictions
        decoded_predictions = decode_predictions(prediction)

        return jsonify(decoded_predictions)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
