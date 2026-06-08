import requests
import cv2
import numpy as np
import json

IMG_SIZE = (224, 224)
image_path = "data/Testing/glioma/Te-gl_1.jpg"

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, IMG_SIZE)
image = image / 255.0
image = np.expand_dims(image, axis=0)

data = {"instances": image.tolist()}

response = requests.post(
    "http://127.0.0.1:5001/invocations",
    headers={"Content-Type": "application/json"},
    data=json.dumps(data)
)

print(response.json())