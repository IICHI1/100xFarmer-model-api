from fastapi import FastAPI, UploadFile, File, Form
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uvicorn
import pickle
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the CNN model (.h5 format)
models = {
    "disease": tf.keras.models.load_model("models/disease.keras"),
    "pest": tf.keras.models.load_model("models/pest.keras"),
}

with open("models/disease.pkl", "rb") as f:
    categories1 = pickle.load(f)
with open("models/pest.pkl", "rb") as f:
    categories2 = pickle.load(f)

def preprocess_image1(image):
    # Preprocess the image for MobileNetV2
    IMG_HEIGHT, IMG_WIDTH = 224, 224
    image = image.convert("RGB")
    image = image.resize((IMG_HEIGHT, IMG_WIDTH))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image

def preprocess_image2(image):
    IMG_HEIGHT, IMG_WIDTH = 384, 384
    image = image.convert("RGB")
    image = image.resize((IMG_HEIGHT, IMG_WIDTH))
    image = np.array(image)  # Convert PIL Image to NumPy array
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image


@app.get("/")
def home():
    return {"message": "Welcome to 100xFarmer Model Backend!"}


@app.post("/predict/disease")
async def predict(model_name: str = Form(...), file: UploadFile = File(...)):
    if model_name not in models:
        return {"error": "Model not found"}
    try:
        image = Image.open(io.BytesIO(await file.read()))
        image = preprocess_image1(image)
        model = models[model_name]
        prediction = model.predict(image)
        predicted_class_index = int(np.argmax(prediction))
        predicted_class_name = categories1[predicted_class_index]
        confidence = float(np.max(prediction))
        return {
            "predicted_class_index": predicted_class_index,
            "predicted_class_name": predicted_class_name,
            "confidence": confidence
        }
    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}


@app.post("/predict/pest")
async def predict(model_name: str = Form(...), file: UploadFile = File(...)):
    if model_name not in models:
        return {"error": "Model not found"}
    try:
        image = Image.open(io.BytesIO(await file.read()))
        image = preprocess_image2(image)
        model = models[model_name]
        prediction = model.predict(image)
        # print(f"Prediction: {prediction}")
        predicted_class_index = int(np.argmax(prediction))
        # print(f"Predicted class index: {predicted_class_index}")
        predicted_class_name = categories2[predicted_class_index]
        confidence = float(np.max(prediction))
        return {
            "predicted_class_index": predicted_class_index,
            "predicted_class_name": predicted_class_name,
            "confidence": confidence
        }
    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
