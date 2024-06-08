import requests
import cv2
import numpy as np
import json

# URL of the Flask server
url = 'http://127.0.0.1:5000/predict'

# Path to the image file you want to send  "C:\Users\ofady\Downloads\picture.jpg"
file_path = r'"C:\Users\ofady\Downloads\picture.jpg"'

# Read the image
image = cv2.imread(file_path)

# Resize the image to match the expected input size of the model (640x640)
resized_image = cv2.resize(image, (640, 640))

# Convert the resized image to RGB format
resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

# Convert the resized image to bytes
_, img_encoded = cv2.imencode('.jpg', resized_image_rgb)
img_bytes = img_encoded.tobytes()

# Send the resized image as a POST request to the server
try:
    response = requests.post(url, files={'image': img_bytes})
    if response.status_code == 200:
        result = response.json()
        print(result)
    else:
        print(f"Error: {response.status_code} - {response.text}")
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
