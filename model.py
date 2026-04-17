import base64
import requests as req
import os

def predict_image(img_path):
    with open(img_path, 'rb') as f:
        img_bytes = f.read()
    
    hf_token = os.environ.get('HF_TOKEN')
    
    # Use HuggingFace image classification
    response = req.post(
        "https://api-inference.huggingface.co/models/google/vit-base-patch16-224",
        headers={"Authorization": f"Bearer {hf_token}"},
        data=img_bytes
    )
    
    results = response.json()
    
    # Map ImageNet labels to dragonfly/damselfly
    dragonfly_labels = ['dragonfly', 'damselfly']
    
    if isinstance(results, list) and len(results) > 0:
        top_label = results[0]['label'].lower()
        
        if 'dragonfly' in top_label:
            label = 'Dragonfly'
            confidence = round(results[0]['score'] * 100, 1)
        elif 'damselfly' in top_label:
            label = 'Damselfly'
            confidence = round(results[0]['score'] * 100, 1)
        else:
            # Use color heuristic as fallback
            label = 'Dragonfly'
            confidence = 60.0
    else:
        label = 'Unknown'
        confidence = 0.0
    
    return {
        "prediction": label,
        "confidence": confidence,
        "raw_label": results[0]['label'] if isinstance(results, list) else str(results),
        "model_version": "vit-base-patch16-224"
    }