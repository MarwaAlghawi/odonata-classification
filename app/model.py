import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load model once when API starts
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'odonata_v2.keras')
model = load_model(MODEL_PATH)

def predict_image(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    pred = model.predict(img_array, verbose=0)[0][0]
    
    label = 'Dragonfly' if pred > 0.5 else 'Damselfly'
    confidence = float(pred if pred > 0.5 else 1 - pred)
    
    return {
        "prediction": label,
        "confidence": round(confidence * 100, 2),
        "model_version": "v2"
    }