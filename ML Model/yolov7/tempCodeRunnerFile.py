from flask import Flask, request, jsonify
import torch
import numpy as np
import cv2
import os
import sys

# Add yolov7 directory to the Python path
sys.path.append(os.path.join(os.getcwd(), 'yolov7'))

# Import attempt_load from models.experimental
from models.experimental import attempt_load

app = Flask(__name__)

# Load YOLOv7 model
weights_path = r'C:\College\Graduation_project\Object detection\yolov7-server\best.pt'
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Weights file not found at: {weights_path}")

try:
    # Load the model using attempt_load function
    model = attempt_load(weights_path, map_location='cpu')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        img_bytes = file.read()
        img = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        # Resize the input image to match the expected size of the model (640x640)
        img = cv2.resize(img, (640, 640))

        # Convert the image to PyTorch tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()

        # Perform inference
        results = model(img_tensor)

        # Print the structure of the results object
        print(results)

        return jsonify({'message': 'Inference successful'})
    except Exception as e:
        print(f"Error during inference: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
