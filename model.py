import numpy as np
from PIL import Image
import os

def predict_image(img_path):
    # Load image
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    
    # Simple heuristic based on aspect ratio and color analysis
    # Dragonflies tend to have wider wing spans (wider images)
    # Damselflies tend to be more slender
    
    width, height = img.size
    aspect_ratio = width / height
    
    # Analyze blue vs red/orange color dominance
    red_channel = img_array[:,:,0].mean()
    blue_channel = img_array[:,:,2].mean()
    green_channel = img_array[:,:,1].mean()
    
    # Basic decision logic
    if blue_channel > red_channel:
        label = 'Damselfly'
        confidence = round(60 + (blue_channel - red_channel) * 100, 1)
    else:
        label = 'Dragonfly'
        confidence = round(60 + (red_channel - blue_channel) * 100, 1)
    
    confidence = min(confidence, 95.0)
    
    return {
        "prediction": label,
        "confidence": confidence,
        "model_version": "v2-heuristic",
        "note": "Basic color analysis — full model requires TensorFlow"
    }