from fastapi import FastAPI, UploadFile, File, Form
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uvicorn
import pickle
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define image constants
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Load the CNN model (.h5 format)
models = {
    "disease": tf.keras.models.load_model("models/disease.keras"),
}

# Load categories (labels) from a pickle file
with open("disease.pkl", "rb") as f:
    categories = pickle.load(f)

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_HEIGHT, IMG_WIDTH))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict/disease")
async def predict(model_name: str = Form(...), file: UploadFile = File(...)):
    if model_name not in models:
        return {"error": "Model not found"}
    try:
        image = Image.open(io.BytesIO(await file.read()))
        image = preprocess_image(image)
        model = models[model_name]
        prediction = model.predict(image)
        predicted_class_index = int(np.argmax(prediction))
        predicted_class_name = categories[predicted_class_index]
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
