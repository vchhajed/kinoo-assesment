from flask import Flask, request, jsonify
import torch
from src.models.model import CustomModel  # Import your model class
from src.data.preprocessing import preprocess_image
app = Flask(__name__)

model = CustomModel(num_classes=10)  # Instantiate your model

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    img = preprocess_image(data)
    inputs = torch.tensor(img)  # Convert input data to tensor
    outputs = model(inputs)
    predicted_class = torch.argmax(outputs).item()

    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
